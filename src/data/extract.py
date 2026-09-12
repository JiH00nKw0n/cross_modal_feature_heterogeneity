"""The only embedding extractor: COCO Karpathy, CC3M, ImageNet-1K.

Every extractor writes the cache format documented in `src.data.cache_io` and
obeys the same three rules.

Rule 1, one row per pair. A COCO image with five captions produces five rows
and the image embedding is duplicated across them. The image is encoded once
per image, not once per caption, so the duplicated rows are bit-identical.

Rule 2, raw embeddings. What the encoder returns is what is stored. The
L2 normalization that training and evaluation apply lives in
`src.data.paired_dataset.l2_normalize_rows`, not here.

Rule 3, bounded memory and resumable. Embeddings are written as fixed-size
chunk files under `cache_dir/parts/<split>/` and assembled at the end with
`numpy.lib.format.open_memmap`, so peak memory is one chunk rather than the
whole table. A restart reads `progress.json`, keeps the chunks it covers, and
skips exactly that many source records. The count is of records that were
encoded, so every extractor slices an iterator that has already dropped the
undecodable rows; slicing the raw stream instead would re-encode one cached
record per dropped row. The COCO captions live in an append-only log beside the
chunks for the same reason: a resumed pass never revisits the groups it
skipped.

Datasets:

    coco      HF "noonamkha/coco-karpathy" (columns image_id, image, captions).
              Source splits train / validation / test become cache splits
              train / val / test, all three inside ONE cache dir. Key per pair
              is "{image_id}_{cap_idx}". Also writes captions.json.
    cc3m      HF "pixparse/cc3m-wds", streaming split "train" (webdataset rows
              with __key__, jpg, txt), about 2.87M pairs. Key is __key__.
    imagenet  HF "ILSVRC/imagenet-1k", streaming split "validation" (50,000
              images, needs HF_TOKEN), paired with the 80 OpenAI templates x
              1000 class names from open_clip.zero_shot_metadata. Not a paired
              cache: it writes labels.npy and text_keys.json instead.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from src.data.cache_io import (
    imagenet_cache_complete,
    load_stacked,
    paired_cache_complete,
    save_imagenet_cache,
    write_keys_and_splits,
)
from src.encoders import Encoder, load_encoder
from src.utils.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)

#: Rows per chunk file. 65,536 rows x 1024 dims x 4 bytes is 256 MB at worst.
CHUNK_SIZE = 65_536

#: Seconds between progress lines (rule: every pass logs a rate, and an ETA
#: whenever its size is known). This is wall-clock, not chunk-flush, based:
#: a pass shorter than one chunk never flushes and would otherwise be silent.
LOG_EVERY_SECONDS = 30.0

COCO_HF_ID = "noonamkha/coco-karpathy"
CC3M_HF_ID = "pixparse/cc3m-wds"
IMAGENET_HF_ID = "ILSVRC/imagenet-1k"

#: Source split name -> cache split name for COCO.
COCO_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

#: Approximate pair count per COCO Karpathy split, used only to print an ETA.
#: 113,287 / 5,000 / 5,000 images, about five captions each.
COCO_ROW_HINTS = {"train": 566_435, "val": 25_010, "test": 25_010}

#: Approximate CC3M pair count, used only to print an ETA while streaming.
CC3M_TOTAL_HINT = 2_870_000

#: ImageNet-1K validation image count, used only to print an ETA.
IMAGENET_VAL_HINT = 50_000


# --------------------------------------------------------------------------- #
# Chunked, resumable writer
# --------------------------------------------------------------------------- #
class _ChunkWriter:
    """Fixed-size chunk files plus a progress record, so a restart can resume.

    A "group" is one source record (one COCO image with its captions, one CC3M
    row). A "row" is one output pair. `progress.json` records both, because the
    resume rule needs rows to place data and groups to skip source records.
    """

    def __init__(self, parts_dir: str | Path, chunk_size: int = CHUNK_SIZE):
        self.parts_dir = Path(parts_dir)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self.progress_path = self.parts_dir / "progress.json"

        self.rows_done = 0
        self.groups_done = 0
        self.n_parts = 0
        self._buf_img: list[np.ndarray] = []
        self._buf_txt: list[np.ndarray] = []
        self._buf_keys: list[str] = []
        self._buf_rows = 0
        self._flushes = 0
        self._resume()

    # -- resume ------------------------------------------------------------ #
    def _resume(self) -> None:
        """Keep the chunks `progress.json` vouches for; drop anything else.

        Any mismatch between the record and the files on disk means a crash
        landed mid-write, and the safe reading of a half-written chunk set is to
        throw it away rather than guess where the gap is.
        """
        if not self.progress_path.exists():
            self._reset_parts()
            return
        try:
            with open(self.progress_path) as f:
                rec = json.load(f)
            n_parts = int(rec["n_parts"])
            rows = int(rec["rows"])
            groups = int(rec["groups"])
        except Exception as exc:
            logger.warning("[extract] unreadable %s (%s) - restarting this split",
                           self.progress_path, exc)
            self._reset_parts()
            return

        seen = 0
        for i in range(n_parts):
            paths = self._part_paths(i)
            if not all(p.exists() for p in paths):
                logger.warning("[extract] chunk %d of %s is incomplete - restarting this split",
                               i, self.parts_dir)
                self._reset_parts()
                return
            seen += int(np.load(paths[0], mmap_mode="r").shape[0])
        if seen != rows:
            logger.warning("[extract] %s claims %d rows but chunks hold %d - restarting this split",
                           self.progress_path, rows, seen)
            self._reset_parts()
            return

        for stale in sorted(self.parts_dir.glob("part_*")):
            if int(stale.name.split("_")[1]) >= n_parts:
                stale.unlink()

        self.rows_done, self.groups_done, self.n_parts = rows, groups, n_parts
        if rows:
            logger.info("[extract] resuming %s at %d rows (%d source records, %d chunks)",
                        self.parts_dir, rows, groups, n_parts)

    def _reset_parts(self) -> None:
        for p in self.parts_dir.glob("part_*"):
            p.unlink()
        if self.progress_path.exists():
            self.progress_path.unlink()
        self.rows_done = self.groups_done = self.n_parts = 0

    def _part_paths(self, idx: int) -> tuple[Path, Path, Path]:
        stem = f"part_{idx:06d}"
        return (
            self.parts_dir / f"{stem}_image.npy",
            self.parts_dir / f"{stem}_text.npy",
            self.parts_dir / f"{stem}_keys.json",
        )

    # -- writing ----------------------------------------------------------- #
    def add(self, image: np.ndarray, text: np.ndarray, keys: Sequence[str],
            *, groups: int) -> bool:
        """Buffer one source batch. Returns True when a chunk was flushed."""
        if image.shape[0] != text.shape[0] or image.shape[0] != len(keys):
            raise ValueError("image, text and keys must have the same row count")
        self._buf_img.append(np.asarray(image, dtype=np.float32))
        self._buf_txt.append(np.asarray(text, dtype=np.float32))
        self._buf_keys.extend(keys)
        self._buf_rows += image.shape[0]
        self.groups_done += groups
        if self._buf_rows >= self.chunk_size:
            self._flush()
            return True
        return False

    def _flush(self) -> None:
        if self._buf_rows == 0:
            return
        img = np.concatenate(self._buf_img, axis=0)
        txt = np.concatenate(self._buf_txt, axis=0)
        keys = list(self._buf_keys)
        p_img, p_txt, p_keys = self._part_paths(self.n_parts)
        np.save(p_img, img)
        np.save(p_txt, txt)
        with open(p_keys, "w") as f:
            json.dump(keys, f)
        self.n_parts += 1
        self.rows_done += img.shape[0]
        self._buf_img, self._buf_txt, self._buf_keys, self._buf_rows = [], [], [], 0
        with open(self.progress_path, "w") as f:
            json.dump({"rows": self.rows_done, "groups": self.groups_done,
                       "n_parts": self.n_parts}, f)
        self._flushes += 1

    def close(self) -> None:
        self._flush()

    @property
    def flushes(self) -> int:
        return self._flushes

    def iter_chunks(self) -> Iterator[tuple[Path, Path, Path]]:
        for i in range(self.n_parts):
            yield self._part_paths(i)

    def all_keys(self) -> list[str]:
        keys: list[str] = []
        for _img, _txt, p_keys in self.iter_chunks():
            with open(p_keys) as f:
                keys.extend(json.load(f))
        return keys


def _refuse_slice_smaller_than_resume(
    parts_dir: Path, groups_done: int, max_groups: int | None, *, what: str,
) -> None:
    """Refuse a slice that an interrupted pass has already overshot.

    A resumed extraction skips the source records already on disk and takes
    only the remainder of the slice on top, so a parts directory that already
    holds more records than the slice asks for contributes every one of them
    and takes nothing further. The assembled cache would then hold more source
    records than the `max_samples` written into its meta.json, and
    `src.data.cache_io.cache_slice_mismatch` reads exactly that number, so
    nothing downstream could tell the cache apart from the smaller one it
    claims to be. Refusing here keeps `max_samples` an upper bound on what the
    cache holds, which is what every later check assumes.

    A full extraction, `max_groups is None`, always runs its source to the end,
    so it has nothing to overshoot and is never refused.
    """
    if max_groups is None or groups_done <= max_groups:
        return
    raise ValueError(
        f"{what}: {parts_dir} already holds {groups_done:,} source records from an "
        f"earlier pass, which is more than the {max_groups:,} this run asks for. "
        f"The cache assembled from it would hold more than its meta.json records. "
        f"Delete {parts_dir} and run again to extract the smaller slice afresh, or "
        f"ask for at least {groups_done:,} source records."
    )


def _assemble(chunk_paths: list[Path], out_path: Path, *, total_rows: int, dim: int) -> None:
    """Concatenate chunk files into one .npy without holding two copies in RAM."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(total_rows, dim))
    off = 0
    for p in chunk_paths:
        chunk = np.load(p, mmap_mode="r")
        arr[off:off + chunk.shape[0]] = chunk
        off += chunk.shape[0]
    if off != total_rows:
        raise RuntimeError(f"assembled {off} rows, expected {total_rows}")
    arr.flush()
    del arr


class _RateLogger:
    """Rows per second and estimated finish time, on a wall-clock cadence.

    The cadence is time based, not flush based. One chunk holds 65,536 rows, so
    a pass smaller than a chunk (ImageNet validation is 50,000 images) never
    flushes mid-run, and a flush-gated logger would print nothing at all for
    that whole pass. Printing every `LOG_EVERY_SECONDS` instead guarantees that
    every pass reports its rate, and a pass whose size is known also reports an
    estimated time to finish.
    """

    def __init__(self, tag: str, total_hint: int | None,
                 min_interval_s: float = LOG_EVERY_SECONDS):
        self.tag = tag
        self.total_hint = total_hint
        self.min_interval_s = float(min_interval_s)
        self.t0 = time.time()
        self.t_last = self.t0
        self.rows0 = 0

    def start_at(self, rows: int) -> None:
        """Anchor the rate at the row count this pass starts (or resumes) from."""
        self.rows0 = rows
        self.t0 = time.time()
        self.t_last = self.t0

    def maybe_log(self, rows: int) -> None:
        """Print a rate line when enough wall-clock time has passed."""
        now = time.time()
        if now - self.t_last < self.min_interval_s:
            return
        self.t_last = now
        dt = max(now - self.t0, 1e-6)
        rate = (rows - self.rows0) / dt
        if self.total_hint and rate > 0:
            remain = max(self.total_hint - rows, 0) / rate
            logger.info("[extract] %s %d/%d rows, %.0f rows/s, ETA %.1f min",
                        self.tag, rows, self.total_hint, rate, remain / 60.0)
        else:
            logger.info("[extract] %s %d rows, %.0f rows/s", self.tag, rows, rate)


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
def _encode_texts(encoder: Encoder, texts: Sequence[str], batch_size: int) -> np.ndarray:
    out = []
    for s in range(0, len(texts), batch_size):
        out.append(encoder.encode_text(list(texts[s:s + batch_size])).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, encoder.dim), dtype=np.float32)


def _encode_images(encoder: Encoder, images: Sequence[Image.Image],
                   batch_size: int) -> np.ndarray:
    out = []
    for s in range(0, len(images), batch_size):
        out.append(encoder.encode_image(list(images[s:s + batch_size])).numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, encoder.dim), dtype=np.float32)


def _as_pil(obj: Any) -> Image.Image | None:
    """Accept the several shapes HF uses for an image column."""
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict) and obj.get("bytes"):
        return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    if hasattr(obj, "convert"):
        return obj.convert("RGB")
    return None


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
class _CaptionSidecar:
    """Append-only caption log kept beside a split's chunk files.

    `captions.json` has to survive a resume. An in-memory dict cannot do that:
    a restart skips the groups already encoded, so their captions are never
    seen again and the file written at the end would be missing every pair
    produced before the interruption. Appending each batch to a file in the
    parts dir, and reading that file back at the end, keeps the sidecar
    complete no matter how many times the pass was restarted.

    One JSON object per line, so appending costs the batch and not the whole
    history. Writing happens before the embeddings are handed to the chunk
    writer, so a crash can only leave captions for rows that were never
    written; those extra keys are dropped when the final dict is filtered to
    the keys actually in the cache.
    """

    def __init__(self, parts_dir: str | Path):
        self.path = Path(parts_dir) / "captions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Drop the log; used when the split restarts from row zero."""
        self.path.unlink(missing_ok=True)

    def append(self, items: Sequence[tuple[str, str]]) -> None:
        with open(self.path, "a") as f:
            for key, caption in items:
                f.write(json.dumps({"k": key, "c": caption}) + "\n")

    def read(self) -> dict[str, str]:
        """Every caption logged so far. A truncated final line is ignored."""
        out: dict[str, str] = {}
        if not self.path.exists():
            return out
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[extract] ignoring a truncated line in %s", self.path)
                    continue
                out[str(rec["k"])] = str(rec["c"])
        return out


def _iter_coco_groups(
    split: str, *, streaming: bool = False,
) -> Iterator[tuple[Image.Image, list[tuple[str, str]]]]:
    """One group per image: the PIL image plus its (key, caption) pairs.

    Grouping by image is what lets the extractor honour rule 1 cheaply: the
    image is encoded once and its embedding is copied into each caption's row.

    `streaming` changes only how the rows arrive. False downloads the whole
    parquet set first, about 20 GB, which is what a full extraction wants
    because it reads every row anyway. True reads the parquet row groups one at
    a time over the network and downloads nothing else, which is what a run
    that takes only the first few thousand photographs wants. The rows
    themselves, their order, and the filtering below are identical either way,
    so a group counted in one mode is the same group in the other and the
    resume offset carries across.
    """
    from datasets import load_dataset

    ds = load_dataset(COCO_HF_ID, split=split, streaming=True) if streaming \
        else load_dataset(COCO_HF_ID, split=split)
    for ex in ds:
        img = _as_pil(ex.get("image"))
        if img is None:
            continue
        caps = ex.get("captions") or []
        if isinstance(caps, str):
            caps = [caps]
        image_id = ex["image_id"]
        items = [(f"{image_id}_{i}", str(c)) for i, c in enumerate(caps) if str(c).strip()]
        if items:
            yield img, items


def extract_coco(
    *,
    model_cfg: ModelConfig,
    cache_dir: str | Path,
    batch_size: int = 64,
    device: str = "cuda",
    max_groups_per_split: int | None = None,
) -> None:
    """Extract all three COCO Karpathy splits into ONE cache dir.

    Cache splits are named train / val / test. Idempotent: returns immediately
    when the paired cache files already exist.

    `max_groups_per_split` stops each split after that many photographs. Asking
    for a slice also switches the source to streaming reads, so that a run over
    the first few thousand photographs fetches only the parquet row groups it
    consumes instead of the whole 20 GB set. The value is recorded in the
    cache's meta.json, so the resulting cache cannot later be mistaken for a
    full one.
    """
    cache_dir = Path(cache_dir)
    if paired_cache_complete(cache_dir):
        logger.info("[extract] COCO cache already at %s - skipping", cache_dir)
        return
    if max_groups_per_split is not None:
        max_groups_per_split = int(max_groups_per_split)
        logger.info("[extract] COCO is sliced to the first %d photographs per "
                    "split, so the source is read as a stream", max_groups_per_split)

    encoder = load_encoder(model_cfg, device=device)
    split_keys: dict[str, list[str]] = {}
    captions: dict[str, str] = {}
    chunk_paths_image: list[Path] = []
    chunk_paths_text: list[Path] = []
    all_keys: list[str] = []
    dim = 0

    for src_split, cache_split in COCO_SPLIT_MAP.items():
        writer = _ChunkWriter(cache_dir / "parts" / cache_split)
        _refuse_slice_smaller_than_resume(
            writer.parts_dir, writer.groups_done, max_groups_per_split,
            what=f"coco/{cache_split}")
        hint = None if max_groups_per_split is not None else COCO_ROW_HINTS.get(cache_split)
        rate = _RateLogger(f"coco/{cache_split}", hint)
        rate.start_at(writer.rows_done)
        sidecar = _CaptionSidecar(cache_dir / "parts" / cache_split)
        if writer.rows_done == 0:
            sidecar.reset()
        groups_seen = writer.groups_done
        # `streaming` is passed only when a slice is asked for, so that a full
        # run makes exactly the call it has always made.
        stream: Iterator[tuple[Image.Image, list[tuple[str, str]]]] = (
            _iter_coco_groups(src_split, streaming=True)
            if max_groups_per_split is not None
            else _iter_coco_groups(src_split)
        )
        if writer.groups_done:
            stream = islice(stream, writer.groups_done, None)
        if max_groups_per_split is not None:
            stream = islice(stream, max(max_groups_per_split - writer.groups_done, 0))

        buf_imgs: list[Image.Image] = []
        buf_items: list[list[tuple[str, str]]] = []

        def _drain() -> None:
            nonlocal buf_imgs, buf_items
            if not buf_imgs:
                return
            img_emb = _encode_images(encoder, buf_imgs, batch_size)
            flat = [it for items in buf_items for it in items]
            txt_emb = _encode_texts(encoder, [c for _k, c in flat], batch_size)
            repeats = np.array([len(items) for items in buf_items], dtype=np.int64)
            rows_img = np.repeat(img_emb, repeats, axis=0)
            sidecar.append(flat)
            writer.add(rows_img, txt_emb, [k for k, _c in flat], groups=len(buf_imgs))
            buf_imgs, buf_items = [], []

        for img, items in stream:
            buf_imgs.append(img)
            buf_items.append(items)
            groups_seen += 1
            if len(buf_imgs) >= batch_size:
                _drain()
                rate.maybe_log(writer.rows_done)
        _drain()
        writer.close()
        logger.info("[extract] coco/%s: %d pairs from %d images",
                    cache_split, writer.rows_done, groups_seen)

        keys = writer.all_keys()
        # Read the captions back from the sidecar, not from this attempt's
        # memory: a resumed pass never re-reads the groups it skipped.
        logged = sidecar.read()
        missing = [k for k in keys if k not in logged]
        if missing:
            raise RuntimeError(
                f"coco/{cache_split}: {len(missing)} of {len(keys)} captions are missing "
                f"from {sidecar.path}; delete {cache_dir / 'parts' / cache_split} and re-run "
                "that split"
            )
        captions.update({k: logged[k] for k in keys})
        split_keys[cache_split] = keys
        all_keys.extend(keys)
        for p_img, p_txt, _p_keys in writer.iter_chunks():
            chunk_paths_image.append(p_img)
            chunk_paths_text.append(p_txt)
            if dim == 0:
                dim = int(np.load(p_img, mmap_mode="r").shape[1])

    total = len(all_keys)
    _assemble(chunk_paths_image, cache_dir / "image_embeddings.npy", total_rows=total, dim=dim)
    _assemble(chunk_paths_text, cache_dir / "text_embeddings.npy", total_rows=total, dim=dim)
    write_keys_and_splits(
        cache_dir,
        keys=all_keys,
        splits=split_keys,
        meta={
            "model_key": model_cfg.key, "dim": dim, "dataset": "coco",
            "hf_id": COCO_HF_ID, "n_pairs": total,
            "split_sizes": {k: len(v) for k, v in split_keys.items()},
            "normalized": False,
            # None means every photograph of every split; an integer means the
            # cache holds only the first that many photographs per split.
            "max_samples": max_groups_per_split,
        },
        captions=captions,
    )
    shutil.rmtree(cache_dir / "parts", ignore_errors=True)
    logger.info("[extract] wrote %d COCO pairs (dim=%d) to %s", total, dim, cache_dir)


# --------------------------------------------------------------------------- #
# CC3M
# --------------------------------------------------------------------------- #
def _iter_cc3m_rows(split: str = "train") -> Iterator[tuple[Image.Image, str, str]]:
    """(image, caption, key) per webdataset row; undecodable rows are skipped."""
    from datasets import load_dataset

    ds = load_dataset(CC3M_HF_ID, split=split, streaming=True)
    for row in ds:
        img = _as_pil(row.get("jpg"))
        if img is None:
            continue
        txt = (row.get("txt") or "").strip()
        if not txt:
            continue
        yield img, txt, str(row["__key__"])


def extract_cc3m(
    *,
    model_cfg: ModelConfig,
    cache_dir: str | Path,
    split: str = "train",
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: int | None = None,
) -> None:
    """Extract the CC3M streaming split into a paired cache, resumably.

    A restart skips exactly the number of stream rows already written, so the
    2.87M-row pass survives an interruption without re-encoding what is done.
    """
    cache_dir = Path(cache_dir)
    if paired_cache_complete(cache_dir):
        logger.info("[extract] CC3M cache already at %s - skipping", cache_dir)
        return
    if max_samples is not None:
        max_samples = int(max_samples)

    encoder = load_encoder(model_cfg, device=device)
    writer = _ChunkWriter(cache_dir / "parts" / split)
    _refuse_slice_smaller_than_resume(writer.parts_dir, writer.groups_done,
                                      max_samples, what=f"cc3m/{split}")
    rate = _RateLogger(f"cc3m/{split}", max_samples or CC3M_TOTAL_HINT)
    rate.start_at(writer.rows_done)

    stream: Iterator[tuple[Image.Image, str, str]] = _iter_cc3m_rows(split)
    if writer.groups_done:
        logger.info("[extract] skipping %d CC3M stream rows already cached", writer.groups_done)
        stream = islice(stream, writer.groups_done, None)
    if max_samples is not None:
        stream = islice(stream, max(max_samples - writer.groups_done, 0))

    buf_imgs: list[Image.Image] = []
    buf_txts: list[str] = []
    buf_keys: list[str] = []

    def _drain() -> None:
        nonlocal buf_imgs, buf_txts, buf_keys
        if not buf_imgs:
            return
        img_emb = _encode_images(encoder, buf_imgs, batch_size)
        txt_emb = _encode_texts(encoder, buf_txts, batch_size)
        writer.add(img_emb, txt_emb, buf_keys, groups=len(buf_keys))
        buf_imgs, buf_txts, buf_keys = [], [], []

    for img, txt, key in stream:
        buf_imgs.append(img)
        buf_txts.append(txt)
        buf_keys.append(key)
        if len(buf_imgs) >= batch_size:
            _drain()
            rate.maybe_log(writer.rows_done)
    _drain()
    writer.close()

    keys = writer.all_keys()
    chunks = list(writer.iter_chunks())
    if not chunks:
        raise RuntimeError(f"CC3M extraction produced no rows for {cache_dir}")
    dim = int(np.load(chunks[0][0], mmap_mode="r").shape[1])
    _assemble([c[0] for c in chunks], cache_dir / "image_embeddings.npy",
              total_rows=len(keys), dim=dim)
    _assemble([c[1] for c in chunks], cache_dir / "text_embeddings.npy",
              total_rows=len(keys), dim=dim)
    write_keys_and_splits(
        cache_dir,
        keys=keys,
        splits={split: keys},
        meta={
            "model_key": model_cfg.key, "dim": dim, "dataset": "cc3m",
            "hf_id": CC3M_HF_ID, "n_pairs": len(keys),
            "split_sizes": {split: len(keys)}, "normalized": False,
            # None means the whole stream; an integer means the cache holds
            # only the first that many stream rows.
            "max_samples": max_samples,
        },
    )
    shutil.rmtree(cache_dir / "parts", ignore_errors=True)
    logger.info("[extract] wrote %d CC3M pairs (dim=%d) to %s", len(keys), dim, cache_dir)


# --------------------------------------------------------------------------- #
# ImageNet-1K
# --------------------------------------------------------------------------- #
def _imagenet_prompts() -> tuple[list[str], list[str], int, int]:
    """The 80 OpenAI templates applied to the 1000 class names, class-major.

    Row `c * n_templates + t` of the returned prompt list is template `t` of
    class `c`, and `text_keys[i]` states that as "{c}_{t}".
    """
    from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

    prompts: list[str] = []
    keys: list[str] = []
    for c, name in enumerate(IMAGENET_CLASSNAMES):
        for t, template in enumerate(OPENAI_IMAGENET_TEMPLATES):
            prompts.append(template(name))
            keys.append(f"{c}_{t}")
    return prompts, keys, len(IMAGENET_CLASSNAMES), len(OPENAI_IMAGENET_TEMPLATES)


def _iter_imagenet_rows(split: str) -> Iterator[tuple[Image.Image, int]]:
    """(image, class index) per validation row; undecodable rows are skipped.

    The filter lives inside this generator on purpose. The resume rule skips
    `groups_done` source records, and `groups_done` counts only the records that
    were actually encoded, so the two counts agree only when the iterator being
    skipped is the filtered one. Slicing the raw stream instead would re-encode
    one already-cached image for every row that failed to decode, appending
    duplicate rows and duplicate labels.
    """
    from datasets import load_dataset

    ds = load_dataset(IMAGENET_HF_ID, split=split, streaming=True)
    for ex in ds:
        img = _as_pil(ex.get("image"))
        if img is None:
            continue
        yield img, int(ex["label"])


def extract_imagenet(
    *,
    model_cfg: ModelConfig,
    cache_dir: str | Path,
    split: str = "validation",
    batch_size: int = 64,
    device: str = "cuda",
    max_samples: int | None = None,
) -> None:
    """Extract the ImageNet-1K validation images and the class x template texts.

    This cache is not paired: images carry a class label in labels.npy and the
    text table is the fixed 1000 x 80 grid, so no row is duplicated here.
    Needs HF_TOKEN, since the dataset is gated.
    """
    cache_dir = Path(cache_dir)
    if imagenet_cache_complete(cache_dir):
        logger.info("[extract] ImageNet cache already at %s - skipping", cache_dir)
        return
    if max_samples is not None:
        max_samples = int(max_samples)

    encoder = load_encoder(model_cfg, device=device)

    prompts, text_keys, n_classes, n_templates = _imagenet_prompts()
    logger.info("[extract] encoding %d prompts (%d classes x %d templates)",
                len(prompts), n_classes, n_templates)
    text_emb = _encode_texts(encoder, prompts, batch_size)

    parts_dir = cache_dir / "parts" / split
    writer = _ChunkWriter(parts_dir)
    _refuse_slice_smaller_than_resume(parts_dir, writer.groups_done, max_samples,
                                      what=f"imagenet/{split}")
    rate = _RateLogger(f"imagenet/{split}", max_samples or IMAGENET_VAL_HINT)
    rate.start_at(writer.rows_done)

    labels_path = parts_dir / "labels.json"
    labels: list[int] = json.load(open(labels_path)) if labels_path.exists() else []
    if len(labels) < writer.rows_done:
        # Cannot happen while labels are written before the chunk writer, but a
        # cache left by an older build could be short, and silently appending
        # at the wrong offset would misalign every label after the gap.
        raise RuntimeError(
            f"{labels_path} holds {len(labels)} labels but {writer.progress_path} "
            f"claims {writer.rows_done} image rows; delete {parts_dir} and re-run"
        )
    labels = labels[:writer.rows_done]

    stream: Iterator[tuple[Image.Image, int]] = _iter_imagenet_rows(split)
    if writer.groups_done:
        logger.info("[extract] skipping %d ImageNet rows already cached", writer.groups_done)
        stream = islice(stream, writer.groups_done, None)
    if max_samples is not None:
        stream = islice(stream, max(max_samples - writer.groups_done, 0))

    buf_imgs: list[Image.Image] = []
    buf_labels: list[int] = []
    buf_keys: list[str] = []
    next_index = writer.rows_done

    def _drain() -> None:
        nonlocal buf_imgs, buf_labels, buf_keys
        if not buf_imgs:
            return
        img_emb = _encode_images(encoder, buf_imgs, batch_size)
        # Labels are persisted BEFORE the embeddings. `writer.add` may flush a
        # chunk and rewrite progress.json, and a crash in between must leave
        # labels.json longer than the recorded row count, never shorter: the
        # resume above truncates a long list but cannot repair a short one.
        labels.extend(buf_labels)
        with open(labels_path, "w") as f:
            json.dump(labels, f)
        # The text column of this writer is a one-wide placeholder: the ImageNet
        # cache keeps its text as the separate class x template grid, so only
        # the image chunks and the row order are read back from here.
        writer.add(img_emb, np.zeros((img_emb.shape[0], 1), dtype=np.float32),
                   buf_keys, groups=len(buf_keys))
        buf_imgs, buf_labels, buf_keys = [], [], []

    for img, label in stream:
        buf_imgs.append(img)
        buf_labels.append(label)
        buf_keys.append(str(next_index))
        next_index += 1
        if len(buf_imgs) >= batch_size:
            _drain()
            rate.maybe_log(writer.rows_done)
    _drain()
    writer.close()

    chunks = list(writer.iter_chunks())
    if not chunks:
        raise RuntimeError(f"ImageNet extraction produced no rows for {cache_dir}")
    dim = int(np.load(chunks[0][0], mmap_mode="r").shape[1])
    n_images = writer.rows_done
    tmp_img = cache_dir / "_image_embeddings_tmp.npy"
    _assemble([c[0] for c in chunks], tmp_img, total_rows=n_images, dim=dim)
    image_emb = np.load(tmp_img, mmap_mode="r")

    save_imagenet_cache(
        cache_dir,
        image_emb=np.asarray(image_emb),
        labels=np.asarray(labels[:n_images], dtype=np.int64),
        text_emb=text_emb,
        text_keys=text_keys,
        meta={
            "model_key": model_cfg.key, "dim": dim, "dataset": "imagenet",
            "hf_id": IMAGENET_HF_ID, "split": split, "n_images": n_images,
            "n_classes": n_classes, "n_templates": n_templates, "normalized": False,
            # None means every validation image; an integer means the cache
            # holds only the first that many.
            "max_samples": max_samples,
        },
    )
    del image_emb
    tmp_img.unlink(missing_ok=True)
    shutil.rmtree(cache_dir / "parts", ignore_errors=True)
    logger.info("[extract] wrote %d ImageNet val images + %d prompts (dim=%d) to %s",
                n_images, len(text_keys), dim, cache_dir)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def extract_cache(
    *,
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    batch_size: int = 64,
    max_samples: int | None = None,
    device: str = "cuda",
) -> None:
    """Extract `cache_cfg.dataset` into `cache_cfg.cache_dir`. Idempotent."""
    dataset = cache_cfg.dataset
    if dataset == "coco":
        extract_coco(model_cfg=model_cfg, cache_dir=cache_cfg.cache_dir,
                     batch_size=batch_size, device=device,
                     max_groups_per_split=max_samples)
    elif dataset == "cc3m":
        extract_cc3m(model_cfg=model_cfg, cache_dir=cache_cfg.cache_dir,
                     split=cache_cfg.split or "train", batch_size=batch_size,
                     device=device, max_samples=max_samples)
    elif dataset == "imagenet":
        extract_imagenet(model_cfg=model_cfg, cache_dir=cache_cfg.cache_dir,
                         split=cache_cfg.split or "validation",
                         batch_size=batch_size, device=device, max_samples=max_samples)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected coco, cc3m or imagenet")


def cache_dim(cache_dir: str | Path) -> int:
    """Embedding width of an existing paired cache."""
    return int(load_stacked(cache_dir)["image"].shape[1])


__all__ = [
    "extract_cache",
    "extract_coco",
    "extract_cc3m",
    "extract_imagenet",
    "cache_dim",
    "CHUNK_SIZE",
]

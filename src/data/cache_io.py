"""The one cache format used by every stage of this repository.

Two kinds of cache exist. Both store RAW encoder outputs: nothing here is
L2-normalized. Normalization happens at load time, in
`src.data.paired_dataset.l2_normalize_rows`, so that the stored bytes are the
encoder's own output and every consumer applies the identical rule.

Paired cache (COCO, CC3M) under `cache_dir/`:

    image_embeddings.npy   float32 (N, dim)   one row per PAIR; the image row is
                                              duplicated when an image has
                                              several captions
    text_embeddings.npy    float32 (N, dim)   row i is the caption of image row i
    keys.json              list[str], length N, key of each row
    splits.json            {"train": [keys], "val": [keys], "test": [keys]}
    meta.json              {"model_key", "dim", "dataset", "n_pairs", ...}
    captions.json          {key: caption text}   (COCO only)

ImageNet cache (not paired: images carry a class label, text is a fixed
class x template grid) under `cache_dir/`:

    image_embeddings.npy   float32 (50000, dim)
    labels.npy             int64   (50000,)        class index of each image
    text_embeddings.npy    float32 (80000, dim)    class-major: row c*80 + t
    text_keys.json         ["{class_idx}_{tmpl_idx}", ...], length 80000
    meta.json              {"model_key", "dim", "n_classes", "n_templates", ...}

COCO key convention is `"{image_id}_{cap_idx}"`. CC3M key convention is the
webdataset `__key__`. Several analyses recover the image id from a COCO key by
splitting on the last underscore, so that shape is part of the format rather
than an implementation detail.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PAIRED_FILES = ("image_embeddings.npy", "text_embeddings.npy", "keys.json", "splits.json")
IMAGENET_FILES = ("image_embeddings.npy", "text_embeddings.npy", "labels.npy", "text_keys.json")


# --------------------------------------------------------------------------- #
# Paired caches
# --------------------------------------------------------------------------- #
def save_stacked(
    cache_dir: str | Path,
    *,
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    keys: list[str],
    splits: dict[str, list[str]],
    meta: dict[str, Any],
    captions: dict[str, str] | None = None,
) -> None:
    """Write a paired cache in the format documented at the top of this module.

    `image_emb` and `text_emb` must already be one row per pair and must be the
    encoder's RAW output; this function does not normalize.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if image_emb.shape[0] != len(keys) or text_emb.shape[0] != len(keys):
        raise ValueError(
            f"row count mismatch: image={image_emb.shape[0]} "
            f"text={text_emb.shape[0]} keys={len(keys)}"
        )
    np.save(cache_dir / "image_embeddings.npy", np.asarray(image_emb, dtype=np.float32))
    np.save(cache_dir / "text_embeddings.npy", np.asarray(text_emb, dtype=np.float32))
    write_keys_and_splits(cache_dir, keys=keys, splits=splits, meta=meta, captions=captions)


def write_keys_and_splits(
    cache_dir: str | Path,
    *,
    keys: list[str],
    splits: dict[str, list[str]],
    meta: dict[str, Any],
    captions: dict[str, str] | None = None,
) -> None:
    """Write everything except the two embedding arrays.

    Split out so the chunked extractors can assemble the arrays with
    `numpy.lib.format.open_memmap` and then add the sidecar files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "keys.json", "w") as f:
        json.dump(keys, f)
    with open(cache_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if captions is not None:
        with open(cache_dir / "captions.json", "w") as f:
            json.dump(captions, f)


def paired_cache_complete(cache_dir: str | Path) -> bool:
    """True when every required file of a paired cache is present."""
    cache_dir = Path(cache_dir)
    return all((cache_dir / name).exists() for name in PAIRED_FILES)


def load_stacked(cache_dir: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load a paired cache. Embeddings come back RAW, as memmaps when `mmap`.

    Returns `{"image", "text", "keys", "splits", "meta"}`. Nothing is
    normalized here; see `src.data.paired_dataset`.
    """
    cache_dir = Path(cache_dir)
    if not paired_cache_complete(cache_dir):
        missing = [n for n in PAIRED_FILES if not (cache_dir / n).exists()]
        raise FileNotFoundError(f"incomplete paired cache at {cache_dir}; missing {missing}")
    mode = "r" if mmap else None
    with open(cache_dir / "keys.json") as f:
        keys = json.load(f)
    with open(cache_dir / "splits.json") as f:
        splits = json.load(f)
    meta_path = cache_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return {
        "image": np.load(cache_dir / "image_embeddings.npy", mmap_mode=mode),
        "text": np.load(cache_dir / "text_embeddings.npy", mmap_mode=mode),
        "keys": keys,
        "splits": splits,
        "meta": meta,
    }


def cache_slice_mismatch(cache_dir: str | Path, max_samples: int | None) -> str | None:
    """Why the cache on disk cannot stand in for the one being asked for.

    Returns None when it can, and one sentence naming what to do otherwise.

    Every extractor records how much of its corpus it took, as `max_samples` in
    meta.json: None for the whole corpus, an integer for the first that many
    source records. A sliced cache has exactly the shape of a full one, only
    fewer rows, so nothing downstream can tell the difference on its own, and a
    full run that reused a slice left behind by a quick check would train on
    that slice and report its numbers as the full result. A cache written
    before this field existed carries no `max_samples` key and is read as a
    full one, which is what it is.

    The fix is always the same, and the returned sentence says it: delete the
    cache directory and run again, or ask for the slice the cache holds.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable %s (%s); the slice it holds cannot be checked",
                       meta_path, exc)
        return None
    recorded = meta.get("max_samples")
    recorded = None if recorded is None else int(recorded)
    wanted = None if max_samples is None else int(max_samples)
    if recorded == wanted:
        return None
    held = "the whole corpus" if recorded is None \
        else f"only the first {recorded:,} source records"
    asked = "the whole corpus" if wanted is None \
        else f"the first {wanted:,} source records"
    return (
        f"the cache at {cache_dir} holds {held}, but this run asks for {asked}. "
        f"Delete {cache_dir} and run again to extract it afresh, or set "
        f"max_samples to {recorded!r} in the config to use the cache as it is."
    )


def require_cache_slice(cache_dir: str | Path, max_samples: int | None) -> None:
    """Raise when the cache on disk was built from a different slice.

    The thin wrapper around `cache_slice_mismatch` that every pipeline calls,
    so that the refusal reads the same wherever it comes from.
    """
    why = cache_slice_mismatch(cache_dir, max_samples)
    if why is not None:
        raise ValueError(why)


def load_captions(cache_dir: str | Path) -> dict[str, str]:
    """Caption text per key; empty when the cache carries no captions.json."""
    path = Path(cache_dir) / "captions.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def split_rows(cache: dict[str, Any], split: str) -> np.ndarray:
    """Row indices of `split` into the stacked tables, in the split's own order.

    A split names its rows by key, so this resolves keys to rows once instead of
    letting every caller build its own mapping.
    """
    keys = cache["keys"]
    splits = cache["splits"]
    if split not in splits:
        raise KeyError(f"split {split!r} not in cache; have {sorted(splits)}")
    key_to_row = {k: i for i, k in enumerate(keys)}
    wanted = splits[split]
    rows = np.array([key_to_row[k] for k in wanted if k in key_to_row], dtype=np.int64)
    if rows.shape[0] != len(wanted):
        logger.warning(
            "split %s: %d/%d keys are missing from keys.json and were dropped",
            split, len(wanted) - rows.shape[0], len(wanted),
        )
    return rows


# --------------------------------------------------------------------------- #
# ImageNet cache
# --------------------------------------------------------------------------- #
def save_imagenet_cache(
    cache_dir: str | Path,
    *,
    image_emb: np.ndarray,
    labels: np.ndarray,
    text_emb: np.ndarray,
    text_keys: list[str],
    meta: dict[str, Any],
) -> None:
    """Write the ImageNet cache documented at the top of this module."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if image_emb.shape[0] != labels.shape[0]:
        raise ValueError(f"image rows {image_emb.shape[0]} != label rows {labels.shape[0]}")
    if text_emb.shape[0] != len(text_keys):
        raise ValueError(f"text rows {text_emb.shape[0]} != text keys {len(text_keys)}")
    np.save(cache_dir / "image_embeddings.npy", np.asarray(image_emb, dtype=np.float32))
    np.save(cache_dir / "text_embeddings.npy", np.asarray(text_emb, dtype=np.float32))
    np.save(cache_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    with open(cache_dir / "text_keys.json", "w") as f:
        json.dump(text_keys, f)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def imagenet_cache_complete(cache_dir: str | Path) -> bool:
    """True when every required file of an ImageNet cache is present."""
    cache_dir = Path(cache_dir)
    return all((cache_dir / name).exists() for name in IMAGENET_FILES)


def load_imagenet_cache(cache_dir: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load the ImageNet cache. Embeddings come back RAW, as memmaps when `mmap`.

    Returns `{"image", "labels", "text", "text_keys", "meta"}`. Text row
    `c * n_templates + t` is template `t` of class `c`, which is what
    `text_keys[i] == f"{c}_{t}"` states row by row.
    """
    cache_dir = Path(cache_dir)
    if not imagenet_cache_complete(cache_dir):
        missing = [n for n in IMAGENET_FILES if not (cache_dir / n).exists()]
        raise FileNotFoundError(f"incomplete ImageNet cache at {cache_dir}; missing {missing}")
    mode = "r" if mmap else None
    with open(cache_dir / "text_keys.json") as f:
        text_keys = json.load(f)
    meta_path = cache_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    return {
        "image": np.load(cache_dir / "image_embeddings.npy", mmap_mode=mode),
        "labels": np.load(cache_dir / "labels.npy"),
        "text": np.load(cache_dir / "text_embeddings.npy", mmap_mode=mode),
        "text_keys": text_keys,
        "meta": meta,
    }


__all__ = [
    "save_stacked",
    "write_keys_and_splits",
    "load_stacked",
    "load_captions",
    "cache_slice_mismatch",
    "require_cache_slice",
    "split_rows",
    "paired_cache_complete",
    "save_imagenet_cache",
    "load_imagenet_cache",
    "imagenet_cache_complete",
]

"""Shared ground for every rebuttal analysis, so no two of them can drift apart.

An analysis module under `src/rebuttal/` never decides for itself which
checkpoint is model A, which panel carries the correlations, what counts as a
matched pair, or how a confidence interval is drawn. All of that is decided
here, once.

What this module does NOT decide, because `src.alignment.panel` already did:

  Alive.      A latent is alive when it fired at least once over the rows the
              panel was built on. The masks are read from panel.npz as
              `alive_image` and `alive_text`. There is no firing-rate threshold
              in this repository.
  Correlation. `C` in panel.npz, signed Pearson, never passed through abs().
  Matching.   `perm` in panel.npz, one signed alive-restricted Hungarian
              assignment. `usable` marks the rows alive on both sides, and
              every statistic over matched pairs restricts to those rows.

A `Setting` names one trained configuration of the paper, together with every
path an analysis needs to reach it. Two settings exist, both built by
`settings_from_config` from the single post-rebuttal config:

  coco_k8    the paper's Figure 2 point. COCO training split, 8 active latents
             per input, a total latent budget of 8192 (4096 per modality),
             30 epochs.
  cc3m_k32   the paper's Table 1 point. CC3M training split, 32 active latents
             per input, the same latent budget, 10 epochs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.request import Request, urlopen

import numpy as np

from src.alignment.panel import load_panel

logger = logging.getLogger(__name__)

#: The two settings analysed, in the order they are reported.
SETTING_TAGS = ("coco_k8", "cc3m_k32")

#: One-line description per setting, for report headings.
SETTING_TITLES = {
    "coco_k8": ("COCO training split, 8 active latents per input, "
                "total latent budget 8192 (4096 per modality), 30 epochs "
                "(the paper's Figure 2 point)"),
    "cc3m_k32": ("CC3M training split, 32 active latents per input, "
                 "total latent budget 8192 (4096 per modality), 10 epochs "
                 "(the paper's Table 1 point)"),
}

#: URL of the COCO 2014 instance annotations, the external ground truth the
#: COCO-80 tests need. 241 MB compressed.
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)

#: The two files pulled out of that zip.
COCO_INSTANCE_FILES = ("instances_val2014.json", "instances_train2014.json")


# --------------------------------------------------------------------------- #
# Setting
# --------------------------------------------------------------------------- #
@dataclass
class Setting:
    """One trained configuration, with every path an analysis reads or writes.

    tag             "coco_k8" or "cc3m_k32".
    dataset         "coco" or "cc3m", the corpus the SAEs were trained on.
    cache_dir       Embedding cache of that corpus, one row per image-caption
                    pair.
    split           Split the panels were built on. Always "train".
    ckpt_a          Model A, the checkpoint every analysis measures. Seed 0.
    ckpt_b          Model B, an independent training run of the same
                    configuration with a different seed. Only the same-modality
                    comparisons read it.
    panel_img_txt   The image-to-text panel Figure 2 or Table 1 already wrote.
                    Not rebuilt here.
    panels_dir      Directory holding the extra panels the rebuttal stage
                    builds: img_img.npz, txt_txt.npz, txt_txt_diffcap.npz
                    (coco_k8 only, CC3M has one caption per image) and
                    img_txt_null.npz.
    out_dir         Where this setting's analyses write their json, md and pdf.
    baselines       Alignment arms the COCO-80 agreement test compares against,
                    as method name to checkpoint directory. "noalign" maps to
                    model A itself and is scored with the identity permutation
                    rather than the learned one. Empty for coco_k8, whose
                    pipeline trains only the modality-specific method.
    coco_cache      COCO embedding cache, for analyses that need the COCO test
                    split or COCO captions. Equal to `cache_dir` for coco_k8.
    k               Active latents per input.
    latent_size     TOTAL latent budget; a modality-specific model holds half
                    of it per side.
    num_epochs      Training epochs.
    """

    tag: str
    dataset: str
    cache_dir: Path
    split: str
    ckpt_a: Path
    ckpt_b: Path
    panel_img_txt: Path
    panels_dir: Path
    out_dir: Path
    baselines: dict[str, Path] = field(default_factory=dict)
    coco_cache: Path = Path("cache/clip_b32_coco")
    k: int = 8
    latent_size: int = 8192
    num_epochs: int = 30

    @property
    def latents_per_side(self) -> int:
        """Latents on one modality: half the total budget."""
        return int(self.latent_size) // 2

    def panel_path(self, pairing: str) -> Path:
        """Path of one extra panel, by pairing name.

        "img_txt" returns the panel Figure 2 or Table 1 wrote; every other name
        returns `panels_dir/<pairing>.npz`, which the rebuttal stage builds.
        """
        if pairing == "img_txt":
            return self.panel_img_txt
        return self.panels_dir / f"{pairing}.npz"

    def title(self) -> str:
        return SETTING_TITLES.get(self.tag, self.tag)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict of the paths, so an analysis can record its provenance."""
        return {
            "tag": self.tag,
            "dataset": self.dataset,
            "cache_dir": str(self.cache_dir),
            "split": self.split,
            "ckpt_a": str(self.ckpt_a),
            "ckpt_b": str(self.ckpt_b),
            "panel_img_txt": str(self.panel_img_txt),
            "panels_dir": str(self.panels_dir),
            "out_dir": str(self.out_dir),
            "baselines": {k: str(v) for k, v in self.baselines.items()},
            "coco_cache": str(self.coco_cache),
            "k": int(self.k),
            "latent_size": int(self.latent_size),
            "num_epochs": int(self.num_epochs),
        }


def settings_from_config(cfg) -> dict[str, Setting]:
    """Build both settings from a loaded post-rebuttal config.

    The paths come from the two pipeline configs the post-rebuttal config
    carries, `cfg.figure2` and `cfg.table1`, so that a checkpoint the rebuttal
    reads is always the one those pipelines wrote. Returns a dict keyed by tag,
    holding only the tags listed in `cfg.rebuttal.settings`, in the order they
    appear in `SETTING_TAGS`.
    """
    if cfg.figure2 is None or cfg.table1 is None:
        raise ValueError(
            "a post_rebuttal config must carry both figure2 and table1; "
            "load them with !ref from configs/post_rebuttal/"
        )
    root = Path(cfg.output.root)
    rebuttal_root = root / "rebuttal"
    wanted = set(getattr(cfg.rebuttal, "settings", SETTING_TAGS) or SETTING_TAGS)

    fig = cfg.figure2
    tab = cfg.table1
    model_key = fig.models[0].key if fig.models else tab.model.key
    coco_cache = Path(fig.cache.cache_dir.replace("{key}", model_key))

    built: dict[str, Setting] = {}

    if "coco_k8" in wanted:
        fig_root = Path(fig.output.root)
        seed_b = int(getattr(cfg.rebuttal, "coco_seed_b", 1))
        built["coco_k8"] = Setting(
            tag="coco_k8",
            dataset=fig.cache.dataset,
            cache_dir=coco_cache,
            split=fig.cache.split or "train",
            ckpt_a=fig_root / model_key / "final",
            ckpt_b=rebuttal_root / "coco_k8" / f"seed{seed_b}" / "final",
            panel_img_txt=fig_root / model_key / "panel.npz",
            panels_dir=rebuttal_root / "coco_k8" / "panels",
            out_dir=rebuttal_root / "coco_k8",
            baselines={},
            coco_cache=coco_cache,
            k=int(fig.training.k),
            latent_size=int(fig.training.latent_size),
            num_epochs=int(fig.training.num_epochs),
        )

    if "cc3m_k32" in wanted:
        tab_root = Path(tab.output.root)
        seeds = tab.training.resolved_seeds()
        seed_a = int(seeds[0])
        seed_b = int(seeds[1]) if len(seeds) > 1 else seed_a + 1
        ckpt_a = tab_root / f"seed{seed_a}" / "separated" / "final"
        trained = {m.name for m in tab.methods}
        baselines: dict[str, Path] = {}
        for name in ("shared", "iso_align", "group_sparse"):
            if name in trained:
                baselines[name] = tab_root / f"seed{seed_a}" / name / "final"
        # "noalign" is model A scored with the identity permutation, so it needs
        # no checkpoint of its own; the analysis reads the flag from the name.
        baselines["noalign"] = ckpt_a
        built["cc3m_k32"] = Setting(
            tag="cc3m_k32",
            dataset=tab.cache.dataset,
            cache_dir=Path(tab.cache.cache_dir),
            split=tab.cache.split or "train",
            ckpt_a=ckpt_a,
            ckpt_b=tab_root / f"seed{seed_b}" / "separated" / "final",
            panel_img_txt=tab_root / f"seed{seed_a}" / "ours" / "panel.npz",
            panels_dir=rebuttal_root / "cc3m_k32" / "panels",
            out_dir=rebuttal_root / "cc3m_k32",
            baselines=baselines,
            coco_cache=coco_cache,
            k=int(tab.training.k),
            latent_size=int(tab.training.latent_size),
            num_epochs=int(tab.training.num_epochs),
        )

    return {tag: built[tag] for tag in SETTING_TAGS if tag in built}


# --------------------------------------------------------------------------- #
# Reading what the panels and the checkpoints hold
# --------------------------------------------------------------------------- #
def load_panel_or_raise(path: str | Path) -> dict[str, Any]:
    """Read a panel, failing with the command that would build it.

    An analysis that silently skips a missing panel reports fewer rows than it
    claims to, so a missing panel is an error here rather than a warning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build it with the rebuttal stage: "
            "python run.py configs/post_rebuttal/clip_b32.yaml --stage rebuttal"
        )
    return load_panel(path)


def unit_decoder(ckpt: str | Path, side: str) -> np.ndarray:
    """Decoder directions of one side as unit-norm rows, shape (L_side, dim).

    `side` is "image" or "text". The rows of `W_dec` are the directions the
    paper calls feature directions; normalizing them is what makes a dot
    product between two of them a cosine.
    """
    if side not in ("image", "text"):
        raise ValueError(f"side must be 'image' or 'text', got {side!r}")
    ckpt = Path(ckpt)
    key = f"{side}_sae.W_dec"
    safet = ckpt / "model.safetensors"
    if safet.exists():
        from safetensors.torch import load_file

        state = load_file(str(safet))
    else:
        import torch

        binary = ckpt / "pytorch_model.bin"
        if not binary.exists():
            raise FileNotFoundError(
                f"{ckpt} holds neither model.safetensors nor pytorch_model.bin"
            )
        state = torch.load(binary, map_location="cpu")
    if key not in state:
        raise KeyError(f"{ckpt} has no tensor {key!r}; it holds {sorted(state)[:8]}")
    w = state[key].float().numpy()
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def matched_distance(
    Wa: np.ndarray, Wb: np.ndarray, perm: np.ndarray, usable: np.ndarray,
) -> np.ndarray:
    """Cosine distance between each usable latent and its assigned partner.

    One value per row of `usable` that is True, in row order. Rows outside
    `usable` are dropped, because the assignment gives every row a partner
    including rows that never fired, and their partner is arbitrary.
    """
    rows = np.where(usable)[0]
    cos = (Wa[rows] * Wb[perm[rows]]).sum(axis=1)
    return 1.0 - cos


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def describe(values: np.ndarray | Sequence[float]) -> dict[str, Any]:
    """Percentile summary of a set of values.

    Returns n, mean, p05, p25, median, p75 and p95. An empty input returns
    `{"n": 0}` and nothing else, so a caller cannot read a mean that was never
    computed.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


def bootstrap_ci(
    values: np.ndarray | Sequence[float],
    stat: Callable[..., Any] = np.median,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and a 95 percent percentile bootstrap interval.

    Returns (point, lo, hi). `values` is resampled with replacement `n_boot`
    times, `stat` is applied along axis 1 of the resampled matrix, and lo and hi
    are the 2.5th and 97.5th percentiles of those replicates. `stat` therefore
    has to accept an `axis` argument, which `np.median` and `np.mean` do.

    Fewer than two values gives the point estimate with a pair of NaN bounds,
    because a one-element resample has no spread to report.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(v))
    if v.size < 2 or n_boot < 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    reps = np.asarray(stat(v[idx], axis=1), dtype=np.float64)
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def fmt(x: Any, nd: int = 3) -> str:
    """One cell of a table: a number at fixed precision, or "not available".

    A None or a non-finite value prints as "n/a" rather than as "nan", so a
    reader is never left deciding whether a NaN is a real measurement.
    """
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{int(x):,}"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def pct(x: Any, nd: int = 1) -> str:
    """A fraction in [0, 1] as a percentage, with enough digits to stay nonzero.

    A chance rate near 0.002 prints as 0.200 percent rather than as 0.0 percent,
    which is the difference between a small baseline and no baseline.
    """
    if x is None:
        return "n/a"
    try:
        v = 100.0 * float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    if v == 0:
        return "0%"
    return f"{v:.{nd}f}%" if abs(v) >= 1 else f"{v:.3f}%"


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table as one string, no trailing newline.

    Cells are written with `str()`, so a caller formats its own numbers, which
    is deliberate: the number of digits a column deserves depends on what the
    column measures, and this function cannot know that. Every cell is expected
    to carry its unit or to sit under a header that names it.
    """
    headers = [str(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    width = len(headers)
    for i, row in enumerate(body):
        if len(row) != width:
            raise ValueError(
                f"row {i} has {len(row)} cells but the table has {width} columns"
            )
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def write_md(
    path: str | Path,
    title: str,
    paragraphs: Sequence[str],
    tables: Sequence[tuple[str, str]] = (),
) -> Path:
    """Write one analysis report.

    `title` becomes the single level-1 heading. `paragraphs` are written in
    order, one blank line apart. `tables` is a sequence of (caption, table)
    pairs, where `table` is the string `md_table` returned; a caption becomes a
    level-2 heading above its table, and an empty caption places the table with
    no heading.

    The file is self-contained by construction: it never refers to another
    output file for a number a reader needs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", ""]
    for para in paragraphs:
        parts.append(para.strip())
        parts.append("")
    for caption, table in tables:
        if caption:
            parts.append(f"## {caption}")
            parts.append("")
        parts.append(table)
        parts.append("")
    path.write_text("\n".join(parts).rstrip() + "\n")
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write an analysis payload, converting numpy scalars on the way out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"cannot serialize {type(o).__name__}")


# --------------------------------------------------------------------------- #
# COCO instance annotations
# --------------------------------------------------------------------------- #
def ensure_coco_annotations(cache_dir: str | Path = "cache/coco_annotations") -> dict[str, Path]:
    """Make sure the COCO 2014 instance annotations are on disk, and say where.

    The COCO-80 tests need a concept label that the model had no part in
    producing, and COCO's hand-drawn object annotations are that label. They
    ship in one 241 MB zip which is downloaded once into `cache_dir` and unpacked
    into `instances_val2014.json` and `instances_train2014.json` directly under
    it.

    Idempotent: returns immediately when both json files are already there.
    Resumable: a partial download is kept as `<name>.part` and continued with an
    HTTP range request on the next call. Progress is logged every 16 MB.

    Returns a dict mapping each file name to its path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: cache_dir / name for name in COCO_INSTANCE_FILES}
    if all(p.exists() and p.stat().st_size > 0 for p in targets.values()):
        logger.info("[coco-annotations][skip] %s already holds %s",
                    cache_dir, ", ".join(COCO_INSTANCE_FILES))
        return targets

    zip_path = cache_dir / "annotations_trainval2014.zip"
    if not zip_path.exists():
        _download_resumable(COCO_ANNOTATIONS_URL, zip_path)

    logger.info("[coco-annotations] extracting %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for name, out_path in targets.items():
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            member = names.get(name)
            if member is None:
                raise KeyError(f"{zip_path} does not contain {name}; it holds {sorted(names)}")
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            logger.info("[coco-annotations] wrote %s (%.1f MB)",
                        out_path, out_path.stat().st_size / 1e6)
    return targets


def _download_resumable(url: str, out_path: Path, chunk: int = 1 << 20) -> None:
    """Download `url` to `out_path`, continuing a `.part` file if one exists."""
    part = out_path.with_suffix(out_path.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "cross-modal-feature-heterogeneity/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
        logger.info("[coco-annotations] resuming %s at %.1f MB", url, have / 1e6)
    else:
        logger.info("[coco-annotations] downloading %s (241 MB)", url)

    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        # A server that ignores the range header answers 200 and restarts the
        # body at byte zero; appending it to the part file would corrupt the
        # archive, so the part file is dropped instead.
        if have and resp.status != 206:
            logger.warning("[coco-annotations] server ignored the range request, "
                           "restarting the download")
            part.unlink()
            have = 0
        total = have + int(resp.headers.get("Content-Length") or 0)
        mode = "ab" if have else "wb"
        done = have
        next_log = done + (16 << 20)
        with open(part, mode) as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if done >= next_log:
                    pctage = f"{100 * done / total:.0f}%" if total else "unknown"
                    logger.info("[coco-annotations] %.1f MB of %.1f MB (%s)",
                                done / 1e6, total / 1e6, pctage)
                    next_log = done + (16 << 20)
    os.replace(part, out_path)
    logger.info("[coco-annotations] downloaded %s (%.1f MB)",
                out_path, out_path.stat().st_size / 1e6)


__all__ = [
    "Setting",
    "settings_from_config",
    "SETTING_TAGS",
    "SETTING_TITLES",
    "load_panel_or_raise",
    "unit_decoder",
    "matched_distance",
    "describe",
    "bootstrap_ci",
    "fmt",
    "pct",
    "md_table",
    "write_md",
    "write_json",
    "ensure_coco_annotations",
    "COCO_ANNOTATIONS_URL",
    "COCO_INSTANCE_FILES",
]

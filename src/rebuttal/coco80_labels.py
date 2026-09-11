"""Per-image labels over COCO's 80 object categories, from hand-drawn annotations.

The two COCO-80 analyses that follow this one both need a statement about each
photograph that the model had no part in producing. COCO supplies it: every
photograph in the 2014 annotation release carries hand-drawn outlines of the
objects it contains, over a fixed vocabulary of 80 categories. This module turns
those outlines into a matrix of one row per photograph and one column per
category, and caches it as json so the analyses downstream read the same labels.

Three decisions are made here, and each one changes the answer.

Which photographs. Only the COCO test split of the embedding cache. Holding the
test split out matters for the setting whose sparse autoencoders were trained on
COCO: a label test run on the photographs the model was fitted to would be
measuring memorization rather than concept structure. The setting trained on
CC3M never saw any COCO photograph, so for it the restriction costs nothing and
keeps both settings on the identical population.

Which annotations count. A CLIP image embedding summarizes a whole photograph in
one vector, so a twenty-pixel object in a corner leaves almost no trace in it.
A photograph counts as positive for a category only when the objects of that
category cover at least `area_frac` of the frame, 5 percent by default. The
variant with no area condition is built as well and carried in the same cache
file, so that every number downstream can be reported both ways.

Which half of the photographs. The analyses need the image side and the text
side to look at different photographs, or an agreement between them could come
out of the image-caption pairing rather than out of the concept. Each photograph
is assigned to half 0 or half 1 by `md5(image_id) % 2`, which is stable across
machines and across restarts in a way Python's own `hash` is not.

Output written under the setting's out_dir:

    coco80_labels.json   the label matrix, the halves and the per-category counts
    coco80_labels.md     what was built and how many categories are usable
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.cache_io import load_captions, load_stacked
from src.rebuttal.coco80_synonyms import COCO_80, matches
from src.rebuttal.common import (
    Setting,
    ensure_coco_annotations,
    fmt,
    md_table,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Name of this analysis, and the stem of every file it writes.
NAME = "coco80_labels"

#: Smallest share of the frame an object has to cover for its photograph to
#: count as positive for that object's category.
DEFAULT_AREA_FRAC = 0.05

#: Fewest positive photographs a category needs, on each half separately, before
#: any statistic is computed for it.
DEFAULT_MIN_COUNT = 50

#: Split of the COCO embedding cache the labels are built over.
DEFAULT_COCO_SPLIT = "test"

#: Annotation file the labels are read from.
DEFAULT_INSTANCES = "instances_val2014.json"


# --------------------------------------------------------------------------- #
# The label matrix
# --------------------------------------------------------------------------- #
@dataclass
class Coco80Labels:
    """Labels for one population of photographs, both area variants.

    image_ids     COCO image id of each row, ascending. Length is the number of
                  photographs that are both in the chosen cache split and in the
                  annotation file.
    area          Boolean matrix, one row per photograph and one column per
                  category in `COCO_80` order. True when that category's objects
                  cover at least `area_frac_threshold` of the frame.
    no_area       The same matrix with no area condition: True whenever the
                  category is annotated at all.
    half          0 or 1 per photograph, from `md5(image_id) % 2`. The image side
                  of an analysis reads half 0 and the text side reads half 1.
    image_rows    Row of the embedding cache holding this photograph's image
                  vector. One row per photograph.
    caption_rows  Rows of the embedding cache holding the captions, over all
                  photographs concatenated.
    caption_owner For each entry of `caption_rows`, the index into `image_ids`
                  of the photograph that caption belongs to.
    caption_text  The caption strings, aligned with `caption_rows`. Empty strings
                  when the cache carries no captions.json.
    n_images_all_splits
                  How many annotated photographs the cache could serve if every
                  split counted, not only `split`. Recorded for reference
                  because the paper's own script built its labels over that
                  larger population; nothing here reads it as data.
    """

    image_ids: np.ndarray
    area: np.ndarray
    no_area: np.ndarray
    half: np.ndarray
    image_rows: np.ndarray
    caption_rows: np.ndarray
    caption_owner: np.ndarray
    caption_text: list[str]
    area_frac_threshold: float
    split: str
    instances_path: str
    n_images_all_splits: int = 0

    def matrix(self, variant: str) -> np.ndarray:
        """The label matrix of one variant: "area_filtered" or "no_area"."""
        if variant == "area_filtered":
            return self.area
        if variant == "no_area":
            return self.no_area
        raise ValueError(
            f"variant must be 'area_filtered' or 'no_area', got {variant!r}")

    @property
    def n_images(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def n_captions(self) -> int:
        return int(self.caption_rows.shape[0])


#: The two label variants, in the order they are reported.
VARIANTS = ("area_filtered", "no_area")

#: Human-readable name of each variant, for table cells.
VARIANT_TITLES = {
    "area_filtered": "object covers at least 5 percent of the frame",
    "no_area": "no area condition",
}


def image_half(image_id: int) -> int:
    """Which half a photograph belongs to, 0 or 1.

    Uses md5 of the decimal image id rather than Python's `hash`, whose salt
    changes between processes and would reshuffle the halves on every run.
    """
    return int(hashlib.md5(str(int(image_id)).encode()).hexdigest(), 16) % 2


def _image_id_of(key: str) -> int | None:
    """COCO image id from a cache key "{image_id}_{cap_idx}", or None."""
    head = key.rsplit("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _split_image_rows(cache: dict[str, Any], split: str) -> tuple[dict[int, int],
                                                                 dict[int, list[int]]]:
    """Rows of one split, grouped by photograph.

    Returns the row holding each photograph's image vector, which is the first
    row that photograph appears in, and the list of rows holding its captions.
    The image vector is duplicated across a photograph's captions by the cache
    format, so any of its rows would serve; taking the first keeps the choice
    deterministic.
    """
    keys = cache["keys"]
    key_to_row = {k: i for i, k in enumerate(keys)}
    if split not in cache["splits"]:
        raise KeyError(
            f"split {split!r} is not in the COCO cache; it has "
            f"{sorted(cache['splits'])}"
        )
    image_row: dict[int, int] = {}
    caption_rows: dict[int, list[int]] = {}
    for key in cache["splits"][split]:
        row = key_to_row.get(key)
        if row is None:
            continue
        image_id = _image_id_of(key)
        if image_id is None:
            continue
        image_row.setdefault(image_id, row)
        caption_rows.setdefault(image_id, []).append(row)
    return image_row, caption_rows


def _annotated_in_every_split(cache: dict[str, Any],
                              frame_area: dict[int, float]) -> int:
    """How many annotated photographs the cache holds across all of its splits.

    Reported alongside the labels so that the cost of restricting them to one
    split is visible in the output rather than only in this module's docstring.
    The paper's own script built its labels over exactly this larger population.
    """
    seen: set[int] = set()
    for keys in cache["splits"].values():
        for key in keys:
            image_id = _image_id_of(key)
            if image_id is not None:
                seen.add(image_id)
    return len(seen & set(frame_area))


def build_labels(
    *,
    instances_path: str | Path,
    coco_cache: str | Path,
    split: str = DEFAULT_COCO_SPLIT,
    area_frac: float = DEFAULT_AREA_FRAC,
) -> Coco80Labels:
    """Read the annotations and the cache, and build both label variants.

    `instances_path` is one of COCO's instance annotation files. `coco_cache` is
    the paired embedding cache whose keys are "{image_id}_{cap_idx}". Only
    photographs present in both are kept.
    """
    instances_path = Path(instances_path)
    logger.info("[%s] reading %s", NAME, instances_path)
    with open(instances_path) as f:
        inst = json.load(f)

    # COCO's category ids run from 1 to 90 with gaps, so they are mapped onto
    # the 0..79 positions of COCO_80 by name rather than by arithmetic.
    name_to_col = {name: i for i, name in enumerate(COCO_80)}
    catid_to_col = {
        int(c["id"]): name_to_col[c["name"]]
        for c in inst["categories"] if c["name"] in name_to_col
    }
    absent = sorted(set(COCO_80) - {c["name"] for c in inst["categories"]})
    if absent:
        logger.warning("[%s] categories absent from the annotation file: %s", NAME, absent)

    frame_area = {int(im["id"]): float(im["width"]) * float(im["height"])
                  for im in inst["images"]}

    cache = load_stacked(coco_cache, mmap=True)
    image_row, caption_rows = _split_image_rows(cache, split)
    captions = load_captions(coco_cache)
    cache_keys = cache["keys"]

    usable = sorted(set(frame_area) & set(image_row))
    every_split = _annotated_in_every_split(cache, frame_area)
    logger.info("[%s] annotated photographs %d, split %r photographs %d, both %d "
                "(all splits together would give %d)",
                NAME, len(frame_area), split, len(image_row), len(usable),
                every_split)
    if not usable:
        raise ValueError(
            f"no photograph of split {split!r} in {coco_cache} appears in "
            f"{instances_path}; the annotation file and the cache split do not overlap"
        )

    row_of = {image_id: r for r, image_id in enumerate(usable)}
    n_images = len(usable)
    area_px = np.zeros((n_images, 80), dtype=np.float64)
    present = np.zeros((n_images, 80), dtype=bool)
    for a in inst["annotations"]:
        col = catid_to_col.get(int(a["category_id"]))
        row = row_of.get(int(a["image_id"]))
        if col is None or row is None:
            continue
        present[row, col] = True
        # A crowd region covers frame area like any other annotation, so its
        # area is added; nothing here counts instances separately.
        area_px[row, col] += float(a.get("area", 0.0))

    denom = np.array([frame_area[i] for i in usable], dtype=np.float64)[:, None]
    frac = np.clip(area_px / denom, 0.0, 1.0)
    area = present & (frac >= float(area_frac))

    image_ids = np.array(usable, dtype=np.int64)
    half = np.array([image_half(int(i)) for i in image_ids], dtype=np.int8)
    image_rows = np.array([image_row[int(i)] for i in image_ids], dtype=np.int64)

    cap_rows: list[int] = []
    cap_owner: list[int] = []
    cap_text: list[str] = []
    for pos, image_id in enumerate(image_ids):
        for r in caption_rows.get(int(image_id), ()):
            cap_rows.append(int(r))
            cap_owner.append(pos)
            cap_text.append(captions.get(cache_keys[int(r)], ""))

    logger.info("[%s] photographs %d, captions %d", NAME, n_images, len(cap_rows))
    return Coco80Labels(
        image_ids=image_ids,
        area=area,
        no_area=present,
        half=half,
        image_rows=image_rows,
        caption_rows=np.array(cap_rows, dtype=np.int64),
        caption_owner=np.array(cap_owner, dtype=np.int64),
        caption_text=cap_text,
        area_frac_threshold=float(area_frac),
        split=str(split),
        instances_path=str(instances_path),
        n_images_all_splits=every_split,
    )


# --------------------------------------------------------------------------- #
# Counting what survives
# --------------------------------------------------------------------------- #
def category_counts(labels: Coco80Labels, variant: str,
                    with_mentions: bool = False) -> dict[str, np.ndarray]:
    """Positive counts per category, on each side of the population split.

    `image_positives` counts photographs of half 0, the population the image side
    of an analysis reads. `text_positives` counts captions of half 1, the
    population the text side reads; a caption inherits its photograph's labels.
    `caption_mentions` counts the subset of those captions that also name the
    object in words, which is reported as a feasibility figure only and is never
    used as a filter. Counting the mentions costs one regular expression match
    per caption and per category, so it is only done when `with_mentions` asks
    for it and the array is all zeros otherwise.
    """
    Y = labels.matrix(variant)
    Y_cap = Y[labels.caption_owner] if labels.n_captions else np.zeros((0, 80), dtype=bool)
    img_half = labels.half == 0
    cap_in_text_half = (labels.half == 1)[labels.caption_owner] \
        if labels.n_captions else np.zeros(0, dtype=bool)

    image_positives = Y[img_half].sum(axis=0).astype(np.int64)
    text_positives = Y_cap[cap_in_text_half].sum(axis=0).astype(np.int64)

    mentions = np.zeros(80, dtype=np.int64)
    if with_mentions and labels.caption_text and any(labels.caption_text):
        rows = np.where(cap_in_text_half)[0]
        for col, category in enumerate(COCO_80):
            hit = 0
            for r in rows:
                if Y_cap[r, col] and matches(labels.caption_text[int(r)], category):
                    hit += 1
            mentions[col] = hit
    return {
        "image_positives": image_positives,
        "text_positives": text_positives,
        "caption_mentions": mentions,
    }


def usable_categories(labels: Coco80Labels, variant: str,
                      min_count: int = DEFAULT_MIN_COUNT) -> np.ndarray:
    """Indices of the categories with enough positives on both sides."""
    counts = category_counts(labels, variant)
    ok = (counts["image_positives"] >= int(min_count)) & \
         (counts["text_positives"] >= int(min_count))
    return np.where(ok)[0]


# --------------------------------------------------------------------------- #
# Cache on disk
# --------------------------------------------------------------------------- #
def _to_payload(labels: Coco80Labels, min_count: int) -> dict[str, Any]:
    """The json form: positives as per-photograph lists of category indices."""
    def positives(Y: np.ndarray) -> list[list[int]]:
        return [np.where(row)[0].astype(int).tolist() for row in Y]

    payload: dict[str, Any] = {
        "split": labels.split,
        "instances_path": labels.instances_path,
        "area_frac_threshold": labels.area_frac_threshold,
        "min_count": int(min_count),
        "categories": list(COCO_80),
        "n_images": labels.n_images,
        "n_images_all_splits": int(labels.n_images_all_splits),
        "n_captions": labels.n_captions,
        "image_ids": labels.image_ids.astype(int).tolist(),
        "half": labels.half.astype(int).tolist(),
        "image_rows": labels.image_rows.astype(int).tolist(),
        "caption_rows": labels.caption_rows.astype(int).tolist(),
        "caption_owner": labels.caption_owner.astype(int).tolist(),
        "positives_area_filtered": positives(labels.area),
        "positives_no_area": positives(labels.no_area),
        "counts": {},
    }
    for variant in VARIANTS:
        counts = category_counts(labels, variant, with_mentions=True)
        keep = usable_categories(labels, variant, min_count)
        payload["counts"][variant] = {
            "image_positives": counts["image_positives"].astype(int).tolist(),
            "text_positives": counts["text_positives"].astype(int).tolist(),
            "caption_mentions": counts["caption_mentions"].astype(int).tolist(),
            "n_categories_usable": int(keep.size),
            "categories_usable": [COCO_80[int(c)] for c in keep],
            "categories_dropped": [COCO_80[c] for c in range(80) if c not in set(keep.tolist())],
        }
    return payload


def _from_payload(payload: dict[str, Any], caption_text: list[str]) -> Coco80Labels:
    """Rebuild the arrays from the json form."""
    n = int(payload["n_images"])

    def matrix(key: str) -> np.ndarray:
        Y = np.zeros((n, 80), dtype=bool)
        for r, cols in enumerate(payload[key]):
            if cols:
                Y[r, np.asarray(cols, dtype=np.int64)] = True
        return Y

    return Coco80Labels(
        image_ids=np.asarray(payload["image_ids"], dtype=np.int64),
        area=matrix("positives_area_filtered"),
        no_area=matrix("positives_no_area"),
        half=np.asarray(payload["half"], dtype=np.int8),
        image_rows=np.asarray(payload["image_rows"], dtype=np.int64),
        caption_rows=np.asarray(payload["caption_rows"], dtype=np.int64),
        caption_owner=np.asarray(payload["caption_owner"], dtype=np.int64),
        caption_text=caption_text,
        area_frac_threshold=float(payload["area_frac_threshold"]),
        split=str(payload["split"]),
        instances_path=str(payload["instances_path"]),
        n_images_all_splits=int(payload.get("n_images_all_splits", 0)),
    )


def load_or_build(
    setting: Setting,
    out_dir: str | Path,
    *,
    area_frac: float = DEFAULT_AREA_FRAC,
    min_count: int = DEFAULT_MIN_COUNT,
    coco_split: str = DEFAULT_COCO_SPLIT,
    annotations_dir: str | Path = "cache/coco_annotations",
    instances_file: str = DEFAULT_INSTANCES,
) -> Coco80Labels:
    """The labels for this setting, built once and cached under `out_dir`.

    Downloads the COCO annotations on first use, through
    `common.ensure_coco_annotations`. The caption strings are always read from
    the cache rather than from the json, because they are only needed for the
    feasibility count and there is no reason to duplicate them on disk.
    """
    out_path = Path(out_dir) / f"{NAME}.json"
    captions = load_captions(setting.coco_cache)
    cache_keys = load_stacked(setting.coco_cache, mmap=True)["keys"]

    if out_path.exists():
        payload = json.loads(out_path.read_text())
        text = [captions.get(cache_keys[int(r)], "") for r in payload["caption_rows"]]
        logger.info("[%s][skip] %s already holds %d photographs",
                    NAME, out_path, payload["n_images"])
        return _from_payload(payload, text)

    paths = ensure_coco_annotations(annotations_dir)
    instances_path = paths.get(instances_file)
    if instances_path is None:
        raise KeyError(
            f"{instances_file} is not one of the annotation files; "
            f"available are {sorted(paths)}"
        )
    labels = build_labels(
        instances_path=instances_path,
        coco_cache=setting.coco_cache,
        split=coco_split,
        area_frac=area_frac,
    )
    write_json(out_path, _to_payload(labels, min_count))
    logger.info("[%s] wrote %s", NAME, out_path)
    return labels


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def support_phrase(min_support: float) -> str:
    """One clause stating the minimum firing requirement, or that there is none.

    Shared with the two analyses downstream so that all three reports describe
    the same rule in the same words.
    """
    if float(min_support) <= 0:
        return "no minimum firing requirement is applied"
    return (f"a coordinate is dropped from a category unless it fires on at "
            f"least {100 * float(min_support):.0f} percent of that category's "
            f"positives")


def _write_report(setting: Setting, out_dir: Path, labels: Coco80Labels,
                  min_count: int) -> Path:
    counts = {v: category_counts(labels, v, with_mentions=True) for v in VARIANTS}
    keep = {v: usable_categories(labels, v, min_count) for v in VARIANTS}

    intro = (
        f"COCO's hand-drawn object annotations, turned into one label per "
        f"photograph and per object category, for the {labels.split} split of the "
        f"COCO embedding cache at {setting.coco_cache}. "
        f"{labels.n_images:,} photographs of that split appear in "
        f"{Path(labels.instances_path).name} and carry {labels.n_captions:,} "
        f"captions between them. Each photograph is assigned to half 0 or half 1 "
        f"by the md5 of its image id, so that an analysis can read photographs on "
        f"one side and captions on the other without the two sides ever seeing the "
        f"same photograph. Two label variants are built. In the first, a "
        f"photograph counts as positive for a category only when that category's "
        f"objects cover at least "
        f"{100 * labels.area_frac_threshold:.0f} percent of the frame. In the "
        f"second there is no area condition and an annotation of any size counts. "
        f"A category enters an analysis only when it has {min_count} or more "
        f"positives on each side separately."
    )
    population = (
        f"Photographs in half 0, the side an analysis reads as images: "
        f"{int((labels.half == 0).sum()):,}. Photographs in half 1: "
        f"{int((labels.half == 1).sum()):,}, carrying "
        f"{int((labels.half[labels.caption_owner] == 1).sum()):,} of the captions."
    )
    held_out = (
        f"Counting every split of the cache rather than the {labels.split} split "
        f"alone would give {labels.n_images_all_splits:,} annotated photographs, "
        f"against the {labels.n_images:,} used here. The larger population is the "
        f"one the paper's own script labelled. It is not used, because the "
        f"sparse autoencoders of the COCO setting were fitted on the training "
        f"split of this same cache, and scoring them on photographs they were "
        f"fitted to would measure memorization rather than concept structure. "
        f"The consequence is that fewer object categories reach "
        f"{min_count} positives per half here than in the paper, so the counts "
        f"below are not comparable to the published ones."
    )

    rows = []
    for variant in VARIANTS:
        c = counts[variant]
        rows.append([
            VARIANT_TITLES[variant],
            f"{int(keep[variant].size)} of 80",
            fmt(int(np.median(c['image_positives']))),
            fmt(int(c['image_positives'].min())),
            fmt(int(c['image_positives'].max())),
            fmt(int(np.median(c['text_positives']))),
        ])
    summary = md_table(
        ["Label variant",
         f"Categories with {min_count} or more positives on both sides",
         "Positive photographs per category, median",
         "Positive photographs per category, fewest",
         "Positive photographs per category, most",
         "Positive captions per category, median"],
        rows,
    )

    per_cat_rows = []
    ca = counts["area_filtered"]
    cn = counts["no_area"]
    keep_area = set(keep["area_filtered"].tolist())
    for col, category in enumerate(COCO_80):
        per_cat_rows.append([
            category,
            fmt(int(ca["image_positives"][col])),
            fmt(int(ca["text_positives"][col])),
            fmt(int(ca["caption_mentions"][col])),
            fmt(int(cn["image_positives"][col])),
            "yes" if col in keep_area else "no",
        ])
    per_cat = md_table(
        ["Object category",
         "Positive photographs, half 0, area condition applied",
         "Positive captions, half 1, area condition applied",
         "Of those captions, the number that also name the object",
         "Positive photographs, half 0, no area condition",
         f"Enters the analyses ({min_count} or more positives on both sides)"],
        per_cat_rows,
    )

    return write_md(
        Path(out_dir) / f"{NAME}.md",
        f"COCO-80 object labels, {setting.tag}",
        [intro, population, held_out],
        [("How many categories survive", summary),
         ("Positives per object category", per_cat)],
    )


def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        **knobs: Any) -> dict[str, Any]:
    """Build the COCO-80 label matrix for one setting and report what it holds.

    Writes `<out_dir>/coco80_labels.json` and `<out_dir>/coco80_labels.md`, and
    skips both when the json is already there. `device` is accepted for the
    common analysis signature and is unused: nothing here runs a model.

    Knobs, all optional: `area_frac` (default 0.05), `min_count` (default 50),
    `coco_split` (default "test"), `annotations_dir` (default
    "cache/coco_annotations") and `instances_file` (default
    "instances_val2014.json"). Any other keyword is ignored.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{NAME}.json"
    min_count = int(knobs.get("min_count", DEFAULT_MIN_COUNT))

    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        payload = json.loads(out_json.read_text())
        if not (out_dir / f"{NAME}.md").exists():
            labels = load_or_build(
                setting, out_dir,
                area_frac=float(knobs.get("area_frac", DEFAULT_AREA_FRAC)),
                min_count=min_count,
                coco_split=str(knobs.get("coco_split", DEFAULT_COCO_SPLIT)),
                annotations_dir=knobs.get("annotations_dir", "cache/coco_annotations"),
                instances_file=str(knobs.get("instances_file", DEFAULT_INSTANCES)),
            )
            _write_report(setting, out_dir, labels, min_count)
        return payload

    labels = load_or_build(
        setting, out_dir,
        area_frac=float(knobs.get("area_frac", DEFAULT_AREA_FRAC)),
        min_count=min_count,
        coco_split=str(knobs.get("coco_split", DEFAULT_COCO_SPLIT)),
        annotations_dir=knobs.get("annotations_dir", "cache/coco_annotations"),
        instances_file=str(knobs.get("instances_file", DEFAULT_INSTANCES)),
    )
    _write_report(setting, out_dir, labels, min_count)
    return json.loads(out_json.read_text())


__all__ = [
    "DEFAULT_AREA_FRAC",
    "DEFAULT_COCO_SPLIT",
    "DEFAULT_INSTANCES",
    "DEFAULT_MIN_COUNT",
    "NAME",
    "VARIANTS",
    "VARIANT_TITLES",
    "Coco80Labels",
    "build_labels",
    "category_counts",
    "image_half",
    "load_or_build",
    "run",
    "support_phrase",
    "usable_categories",
]

"""How far apart the two modalities put one concept, measured without co-activation.

The paper's headline number pairs an image latent with a text latent by their
co-activation correlation and then measures the angle between the two decoder
directions. The correlation is read out of the very latent space whose
heterogeneity is being claimed, so in principle the noise of that space could be
manufacturing the gap. Settling that requires pairing the two sides from outside
the model.

The COCO-80 agreement test already does the pairing from outside: for each object
category the image side picks its best separating coordinate from photographs and
labels alone, and the text side picks its own from captions and labels alone.
This analysis takes those two picks and measures the cosine between their decoder
directions directly. The correlation matrix is never consulted, not even to
decide which coordinates may be picked: a candidate here is any coordinate that
fires at least once on the labelled data of its own modality.

One angle on its own says nothing, so the identical procedure is run a second
time without crossing modalities. The image side picks a coordinate on one half
of the photographs and picks again on the other half. That pair carries the same
label noise, the same estimation noise and the same finite-sample noise as the
cross-modal pair, and differs from it in exactly one respect, which is that no
modality boundary is crossed. The difference between the two numbers is the part
only heterogeneity explains. The same is done inside the text side.

Two further references fix the scale. Pairing one category's image pick with a
different category's text pick gives the angle between unrelated concepts.
Random unit vectors in the embedding dimension give the floor.

A final row bridges back to the correlation. For the same image coordinate the
labels chose, it reports the angle to the partner the co-activation permutation
assigns instead of to the partner the labels chose. If the two routes land at a
similar angle, the correlation was not the source of the gap.

Output written under the setting's out_dir:

    coco80_heterogeneity.json   every number, both label variants
    coco80_heterogeneity.md     the tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.data.cache_io import load_stacked
from src.eval.eval_utils import load_sae
from src.rebuttal.coco80_correspondence import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_SUPPORT,
    side_scores,
    sparse_latents,
)
from src.rebuttal.coco80_labels import (
    DEFAULT_MIN_COUNT,
    VARIANT_TITLES,
    VARIANTS,
    Coco80Labels,
    load_or_build,
    support_phrase,
    usable_categories,
)
from src.rebuttal.coco80_synonyms import COCO_80
from src.rebuttal.common import (
    Setting,
    bootstrap_ci,
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Name of this analysis, and the stem of every file it writes.
NAME = "coco80_heterogeneity"

#: Random unit vector pairs drawn for the floor.
DEFAULT_N_RANDOM_PAIRS = 2000

#: The comparisons reported, in table order, with what each one pairs.
COMPARISONS = (
    ("within_image_two_halves",
     ("The image side's own pick on one half of the photographs, "
      "against its pick on the other half")),
    ("within_text_two_halves",
     ("The text side's own pick on one half of the captions, "
      "against its pick on the other half")),
    ("cross_modal_same_category",
     "The image side's pick against the text side's pick, same object category"),
    ("cross_modal_different_category",
     "The image side's pick against the text side's pick, different object categories"),
    ("random_unit_vectors",
     "Two random unit vectors in the embedding dimension"),
    ("coactivation_partner_of_the_same_image_coordinate",
     ("The image side's pick against the partner the co-activation permutation "
      "assigns it, instead of the one the labels chose")),
)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _summarize(values: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Median, mean, both with a percentile bootstrap, plus the tails."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    median, med_lo, med_hi = bootstrap_ci(v, np.median, n_boot=n_boot, seed=seed)
    mean, mean_lo, mean_hi = bootstrap_ci(v, np.mean, n_boot=n_boot, seed=seed + 1)
    return {
        "n": int(v.size),
        "cosine_median": median,
        "cosine_median_ci95": [med_lo, med_hi],
        "cosine_mean": mean,
        "cosine_mean_ci95": [mean_lo, mean_hi],
        "cosine_distance_median": 1.0 - median,
        "cosine_p05": float(np.percentile(v, 5)),
        "cosine_p95": float(np.percentile(v, 95)),
        "share_above_0p9": float(np.mean(v > 0.9)),
        "share_below_0p3": float(np.mean(v < 0.3)),
    }


def _pick(auc: np.ndarray, candidates: np.ndarray, category: int) -> int | None:
    """The candidate coordinate with the best separation, or None if there is none."""
    if candidates.size == 0:
        return None
    column = auc[candidates, category]
    if not np.isfinite(column).any():
        return None
    return int(candidates[int(np.argmax(column))])


# --------------------------------------------------------------------------- #
# One label variant
# --------------------------------------------------------------------------- #
def _measure_variant(
    *,
    variant: str,
    labels: Coco80Labels,
    image_triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    text_triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    W_image: np.ndarray,
    W_text: np.ndarray,
    n_latents: int,
    min_count: int,
    min_support: float,
    n_boot: int,
    seed: int,
    panel: dict[str, Any] | None,
) -> dict[str, Any]:
    """Every comparison for one label variant."""
    rng = np.random.default_rng(seed)
    Y = labels.matrix(variant)
    Y_cap = Y[labels.caption_owner]
    image_half_mask = labels.half == 0
    caption_in_text_half = (labels.half == 1)[labels.caption_owner]

    i_samp, i_lat, i_val = image_triples
    t_samp, t_lat, t_val = text_triples

    # Candidates are decided by firing alone. Restricting them by the
    # permutation would put co-activation back into a measurement whose whole
    # point is to avoid it.
    candidates_image = np.unique(i_lat) if i_lat.size else np.zeros(0, dtype=np.int64)
    candidates_text = np.unique(t_lat) if t_lat.size else np.zeros(0, dtype=np.int64)

    auc_image_a, _ = side_scores(i_samp, i_lat, i_val, Y, image_half_mask,
                                 n_latents, min_support)
    auc_image_b, _ = side_scores(i_samp, i_lat, i_val, Y, ~image_half_mask,
                                 n_latents, min_support)
    auc_text_a, _ = side_scores(t_samp, t_lat, t_val, Y_cap, caption_in_text_half,
                                n_latents, min_support)
    auc_text_b, _ = side_scores(t_samp, t_lat, t_val, Y_cap, ~caption_in_text_half,
                                n_latents, min_support)

    scored = usable_categories(labels, variant, min_count)
    counts_image = Y[image_half_mask].sum(axis=0)
    counts_text = Y_cap[caption_in_text_half].sum(axis=0)

    rows: list[dict[str, Any]] = []
    for c in scored.tolist():
        image_a = _pick(auc_image_a, candidates_image, c)
        image_b = _pick(auc_image_b, candidates_image, c)
        text_a = _pick(auc_text_a, candidates_text, c)
        text_b = _pick(auc_text_b, candidates_text, c)
        if image_a is None or text_a is None:
            continue
        rows.append({
            "category": COCO_80[c],
            "image_coordinate": image_a,
            "text_coordinate": text_a,
            "image_coordinate_other_half": image_b,
            "text_coordinate_other_half": text_b,
            "cross_modal_cosine": float(W_image[image_a] @ W_text[text_a]),
            "within_image_cosine": (float(W_image[image_a] @ W_image[image_b])
                                    if image_b is not None else None),
            "within_text_cosine": (float(W_text[text_a] @ W_text[text_b])
                                   if text_b is not None else None),
            "positive_photographs": int(counts_image[c]),
            "positive_captions": int(counts_text[c]),
        })

    result: dict[str, Any] = {
        "variant": variant,
        "n_categories": len(rows),
        "n_candidate_coordinates_image": int(candidates_image.size),
        "n_candidate_coordinates_text": int(candidates_text.size),
        "embedding_dim": int(W_image.shape[1]),
        "comparisons": {},
        "per_category": rows,
    }
    if not rows:
        logger.warning("[%s] %s: no category survived; nothing to summarize",
                       NAME, variant)
        return result

    cross = np.array([r["cross_modal_cosine"] for r in rows], dtype=np.float64)
    within_image = np.array([r["within_image_cosine"] for r in rows
                             if r["within_image_cosine"] is not None], dtype=np.float64)
    within_text = np.array([r["within_text_cosine"] for r in rows
                            if r["within_text_cosine"] is not None], dtype=np.float64)

    # Unrelated concepts: this category's image pick against every other
    # category's text pick. The same coordinates, the wrong correspondence.
    picks_image = np.array([r["image_coordinate"] for r in rows], dtype=np.int64)
    picks_text = np.array([r["text_coordinate"] for r in rows], dtype=np.int64)
    grid = W_image[picks_image] @ W_text[picks_text].T
    mismatched = grid[~np.eye(len(rows), dtype=bool)]

    n_random = 2 * int(DEFAULT_N_RANDOM_PAIRS)
    g = rng.standard_normal((n_random, W_image.shape[1]))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    random_cos = (g[::2] * g[1::2]).sum(axis=1)

    result["comparisons"] = {
        "within_image_two_halves": _summarize(within_image, n_boot=n_boot, seed=seed),
        "within_text_two_halves": _summarize(within_text, n_boot=n_boot, seed=seed + 10),
        "cross_modal_same_category": _summarize(cross, n_boot=n_boot, seed=seed + 20),
        "cross_modal_different_category": _summarize(mismatched, n_boot=n_boot,
                                                     seed=seed + 30),
        "random_unit_vectors": _summarize(random_cos, n_boot=n_boot, seed=seed + 40),
    }

    # Paired, because both numbers exist for the same category.
    paired = [(r["cross_modal_cosine"], r["within_image_cosine"]) for r in rows
              if r["within_image_cosine"] is not None]
    if paired:
        a = np.array([x for x, _ in paired], dtype=np.float64)
        b = np.array([y for _, y in paired], dtype=np.float64)
        diff = b - a
        mean, lo, hi = bootstrap_ci(diff, np.mean, n_boot=n_boot, seed=seed + 50)
        try:
            from scipy.stats import wilcoxon

            p_value = float(wilcoxon(b, a).pvalue)
        except ValueError:
            # Wilcoxon refuses an input whose differences are all zero.
            p_value = float("nan")
        result["within_image_minus_cross_modal"] = {
            "n_categories": int(diff.size),
            "median": float(np.median(diff)),
            "mean": mean,
            "mean_ci95": [lo, hi],
            "wilcoxon_p": p_value,
            "share_of_categories_where_within_image_is_larger":
                float(np.mean(diff > 0)),
        }
        result["categories_where_both_halves_picked_the_same_coordinate"] = int(sum(
            r["image_coordinate"] == r["image_coordinate_other_half"] for r in rows
        ))

    # The bridge back to the correlation: same image coordinate, partner chosen
    # by co-activation rather than by labels.
    if panel is not None:
        perm = np.asarray(panel["perm"], dtype=np.int64)
        usable = np.asarray(panel["usable"], dtype=bool)
        C = np.asarray(panel["C"], dtype=np.float64)
        bridged: list[float] = []
        correlations: list[float] = []
        for r in rows:
            i = int(r["image_coordinate"])
            if i >= usable.shape[0] or not usable[i]:
                continue
            bridged.append(float(W_image[i] @ W_text[perm[i]]))
            correlations.append(float(C[i, perm[i]]))
        if bridged:
            summary = _summarize(np.array(bridged), n_boot=n_boot, seed=seed + 60)
            summary["matched_correlation_median"] = float(np.median(correlations))
            summary["n_categories_covered"] = len(bridged)
            result["comparisons"]["coactivation_partner_of_the_same_image_coordinate"] = \
                summary

    # The two ends of the ranking, never overlapping: with fewer than four
    # categories there is no end to speak of and both lists stay empty.
    ranked = sorted(rows, key=lambda r: -r["cross_modal_cosine"])
    take = min(5, len(ranked) // 2) if len(ranked) >= 4 else 0
    result["most_aligned_categories"] = [
        {"category": r["category"], "cosine": r["cross_modal_cosine"]}
        for r in ranked[:take]
    ]
    result["least_aligned_categories"] = [
        {"category": r["category"], "cosine": r["cross_modal_cosine"]}
        for r in ranked[len(ranked) - take:][::-1]
    ]
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _comparison_table(result: dict[str, Any]) -> str:
    rows = []
    for key, label in COMPARISONS:
        entry = result["comparisons"].get(key)
        if not entry or not entry.get("n"):
            continue
        ci = entry["cosine_mean_ci95"]
        rows.append([
            label,
            fmt(entry["cosine_median"]),
            fmt(entry["cosine_mean"]),
            f"[{fmt(ci[0])}, {fmt(ci[1])}]",
            f"{entry['n']:,}",
        ])
    return md_table(
        ["What was paired",
         "Median cosine between the two decoder directions",
         "Mean cosine between the two decoder directions",
         "95 percent bootstrap interval of that mean",
         "Number of pairs"],
        rows,
    )


def _extremes_sentence(result: dict[str, Any]) -> str:
    """The two ends of the cross-modal ranking, or nothing when there are too few."""
    high = result.get("most_aligned_categories") or []
    low = result.get("least_aligned_categories") or []
    if not high or not low:
        return ""
    def listing(entries: list[dict[str, Any]]) -> str:
        return ", ".join(f"{e['category']} at {fmt(e['cosine'], 2)}" for e in entries)
    return (f"The {len(high)} categories with the largest cross-modal cosine are "
            f"{listing(high)}, and the {len(low)} with the smallest are "
            f"{listing(low)}.")


def _sensitivity_table(variants: dict[str, dict[str, Any]]) -> str:
    rows = []
    for key, label in COMPARISONS:
        cells = [label]
        for variant in VARIANTS:
            entry = variants[variant]["comparisons"].get(key)
            cells.append(fmt(entry["cosine_median"]) if entry and entry.get("n")
                         else "n/a")
        rows.append(cells)
    return md_table(
        ["What was paired",
         f"Median cosine, {VARIANT_TITLES['area_filtered']}",
         f"Median cosine, {VARIANT_TITLES['no_area']}"],
        rows,
    )


def _write_report(setting: Setting, out_dir: Path, labels: Coco80Labels,
                  variants: dict[str, dict[str, Any]], min_count: int,
                  min_support: float) -> Path:
    head = variants["area_filtered"]
    comparisons = head["comparisons"]

    def median_of(key: str) -> float | None:
        entry = comparisons.get(key)
        return entry["cosine_median"] if entry and entry.get("n") else None

    intro = (
        f"The angle between the image direction and the text direction of one "
        f"object category, with the two directions chosen from COCO's object "
        f"annotations and never from the co-activation correlation. Measured on "
        f"the {labels.split} split of the COCO embedding cache at "
        f"{setting.coco_cache}: {labels.n_images:,} photographs and "
        f"{labels.n_captions:,} captions, split into two halves by the md5 of "
        f"the image id. {head['n_categories']} of the 80 object categories have "
        f"{min_count} or more positives on each half and are measured. The "
        f"model is {setting.ckpt_a}. Its setting is {setting.title()}. Candidate "
        f"coordinates are every coordinate that fires at least once on the "
        f"labelled data of its own modality, which is "
        f"{head['n_candidate_coordinates_image']:,} on the image side and "
        f"{head['n_candidate_coordinates_text']:,} on the text side."
    )
    method = (
        f"For each category, each side ranks its candidate coordinates by how "
        f"well the coordinate's activation separates that category's positives "
        f"from its negatives, measured as the area under the ROC curve with a "
        f"tie counting as one half, and keeps the best; "
        f"{support_phrase(min_support)}. The "
        f"image side reads half 0 of the photographs and the text side reads the "
        f"captions of half 1, so the two never see the same photograph. The "
        f"cosine between the two chosen decoder directions is then read off "
        f"directly. The within-modality rows repeat the identical procedure on "
        f"the two halves of one modality, which carries the same label noise and "
        f"the same sampling noise while crossing no modality boundary."
    )

    paired = head.get("within_image_minus_cross_modal")
    if paired:
        same = head.get("categories_where_both_halves_picked_the_same_coordinate", 0)
        result = (
            f"Inside the image side, the coordinate picked on one half and the "
            f"coordinate picked on the other half have a median cosine of "
            f"{fmt(median_of('within_image_two_halves'))}, and in {same} of the "
            f"{head['n_categories']} categories the two halves picked the "
            f"identical coordinate. Across the modalities the same procedure "
            f"gives a median cosine of "
            f"{fmt(median_of('cross_modal_same_category'))}. Taken category by "
            f"category, the within-image cosine minus the cross-modal cosine "
            f"averages {fmt(paired['mean'])} with a 95 percent interval of "
            f"[{fmt(paired['mean_ci95'][0])}, {fmt(paired['mean_ci95'][1])}] and "
            f"a Wilcoxon signed-rank p-value of {fmt(paired['wilcoxon_p'], 6)}; "
            f"the within-image cosine is the larger of the two in "
            f"{pct(paired['share_of_categories_where_within_image_is_larger'])} "
            f"of the categories."
        )
    else:
        result = (
            f"The cross-modal pairs have a median cosine of "
            f"{fmt(median_of('cross_modal_same_category'))}. The paired "
            f"within-image comparison could not be formed, because no category "
            f"produced a pick on both halves of the photographs."
        )

    share_high = comparisons["cross_modal_same_category"]["share_above_0p9"]
    share_low = comparisons["cross_modal_same_category"]["share_below_0p3"]
    scale = (
        f"For reference, pairs built from two different categories have a median "
        f"cosine of {fmt(median_of('cross_modal_different_category'))} and random "
        f"unit vectors {fmt(median_of('random_unit_vectors'))}. Of the "
        f"{head['n_categories']} categories, "
        f"{round(share_high * head['n_categories'])} have a cross-modal "
        f"cosine above 0.9 and {round(share_low * head['n_categories'])} "
        f"have one below 0.3."
    )
    extremes = _extremes_sentence(head)
    if extremes:
        scale = f"{scale} {extremes}"

    bridge_entry = comparisons.get("coactivation_partner_of_the_same_image_coordinate")
    if bridge_entry:
        bridge = (
            f"Holding the image coordinate fixed and choosing its partner by "
            f"co-activation instead of by labels gives a median cosine of "
            f"{fmt(bridge_entry['cosine_median'])}, over the "
            f"{bridge_entry['n_categories_covered']} categories whose image "
            f"coordinate the panel marks usable, at a median matched correlation "
            f"of {fmt(bridge_entry['matched_correlation_median'])}. Choosing the "
            f"partner by labels gives "
            f"{fmt(median_of('cross_modal_same_category'))}."
        )
    else:
        bridge = ("The co-activation partner row could not be formed, because no "
                  "category's image coordinate is marked usable in the panel.")

    per_category = md_table(
        ["Object category",
         "Cosine between the image pick and the text pick",
         "Cosine between the image side's two halves",
         "Cosine between the text side's two halves",
         "Positive photographs",
         "Positive captions"],
        [[r["category"], fmt(r["cross_modal_cosine"]),
          fmt(r["within_image_cosine"]), fmt(r["within_text_cosine"]),
          fmt(r["positive_photographs"]), fmt(r["positive_captions"])]
         for r in sorted(head["per_category"], key=lambda r: -r["cross_modal_cosine"])],
    )

    return write_md(
        Path(out_dir) / f"{NAME}.md",
        f"COCO-80 heterogeneity without co-activation, {setting.tag}",
        [intro, method, result, scale, bridge],
        [(f"Every comparison ({VARIANT_TITLES['area_filtered']})",
          _comparison_table(head)),
         ("Sensitivity to the area condition", _sensitivity_table(variants)),
         ("One row per object category", per_category)],
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        **knobs: Any) -> dict[str, Any]:
    """Measure cross-modal heterogeneity from labels alone, for one setting.

    Writes `<out_dir>/coco80_heterogeneity.json` and
    `<out_dir>/coco80_heterogeneity.md`, and skips both when the json exists.
    Only the modality-specific checkpoint `setting.ckpt_a` is measured, because
    a single shared dictionary has one direction per concept and so has no
    cross-modal angle to report.

    Knobs, all optional: `n_boot` (default 1000, and the pipeline always passes
    it), `min_support` (default 0.05), `min_count` (default 50), `seed`
    (default 0), `batch_size` (default 4096), plus everything
    `coco80_labels.load_or_build` takes. Any other keyword is ignored.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return json.loads(out_json.read_text())

    min_count = int(knobs.get("min_count", DEFAULT_MIN_COUNT))
    min_support = float(knobs.get("min_support", DEFAULT_MIN_SUPPORT))
    n_boot = int(knobs.get("n_boot", 1000))
    seed = int(knobs.get("seed", 0))
    batch_size = int(knobs.get("batch_size", DEFAULT_BATCH_SIZE))

    labels = load_or_build(
        setting, out_dir,
        area_frac=float(knobs.get("area_frac", 0.05)),
        min_count=min_count,
        coco_split=str(knobs.get("coco_split", "test")),
        annotations_dir=knobs.get("annotations_dir", "cache/coco_annotations"),
        instances_file=str(knobs.get("instances_file", "instances_val2014.json")),
    )

    model = load_sae(setting.ckpt_a, "separated")
    n_latents = int(model.image_sae.latent_size)
    cache = load_stacked(setting.coco_cache, mmap=True)
    logger.info("[%s] encoding %d photographs and %d captions",
                NAME, labels.n_images, labels.n_captions)
    image_triples = sparse_latents(model.image_sae, cache["image"], labels.image_rows,
                                   batch_size=batch_size, device=device)
    text_triples = sparse_latents(model.text_sae, cache["text"], labels.caption_rows,
                                  batch_size=batch_size, device=device)

    W_image = unit_decoder(setting.ckpt_a, "image")
    W_text = unit_decoder(setting.ckpt_a, "text")
    panel = load_panel_or_raise(setting.panel_img_txt)

    variants: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variants[variant] = _measure_variant(
            variant=variant, labels=labels,
            image_triples=image_triples, text_triples=text_triples,
            W_image=W_image, W_text=W_text, n_latents=n_latents,
            min_count=min_count, min_support=min_support,
            n_boot=n_boot, seed=seed, panel=panel,
        )
        entry = variants[variant]["comparisons"].get("cross_modal_same_category")
        logger.info("[%s] %s: %d categories, cross-modal median cosine %s",
                    NAME, variant, variants[variant]["n_categories"],
                    fmt(entry["cosine_median"]) if entry else "n/a")

    payload = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "checkpoint": str(setting.ckpt_a),
        "labels": {
            "split": labels.split,
            "n_photographs": labels.n_images,
            "n_captions": labels.n_captions,
            "area_frac_threshold": labels.area_frac_threshold,
            "instances_path": labels.instances_path,
        },
        "min_count": min_count,
        "min_support": min_support,
        "n_boot": n_boot,
        "seed": seed,
        "selection": ("the coordinate with the best separation on each side, "
                      "chosen on disjoint halves of the photographs; the "
                      "co-activation correlation is used only for the bridge row"),
        "variants": variants,
    }
    write_json(out_json, payload)
    _write_report(setting, out_dir, labels, variants, min_count, min_support)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


__all__ = ["COMPARISONS", "DEFAULT_N_RANDOM_PAIRS", "NAME", "run"]

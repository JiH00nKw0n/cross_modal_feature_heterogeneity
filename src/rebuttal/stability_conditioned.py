"""Is the cross-modal distance still there for concepts two runs both recover?

A dictionary direction that only one training run produces says nothing about
the data, so a distance measured against it says nothing either. This module
restricts attention to directions that two independently trained image SAEs
both find, and asks whether the cross-modal distance survives on those.

Reproducibility is scored the way Papadimitriou et al. (arXiv 2504.11695) score
it, following Fel et al. (2025) and Spielman et al. (2012): the rows of the two
dictionaries are aligned with the assignment that maximizes total cosine
similarity, and each concept's stability is the cosine it achieves with its
counterpart in the other run. Concepts are then binned by that score, from the
top 1 percent down to the whole range.

Three things are reported that a single headline would hide.

  The whole decile curve, because the answer moves with the quantile. Reporting
  one cut would be choosing the answer.

  The same measurement with both sides matched the same way. The same-modality
  side is paired by decoder cosine, which is the quantity being reported, while
  the cross-modal side is paired by co-activation correlation. Part of any gap
  between the two columns is therefore the choice of pairing rule rather than
  modality, and pairing the cross-modal side by decoder cosine too separates
  the two contributions.

  The measurement restricted to pairs that genuinely co-activate. A stable
  direction whose cross-modal partner never fires alongside it is not a
  semantic correspondence and should not be read as one. The matched
  correlation is printed next to every cell so a reader can see where that
  stops being true.

Every pair that appears here is a pair the method's own Hungarian assignment
produced, read from the panel and then filtered. Nothing is rematched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.rebuttal.common import (
    Setting,
    bootstrap_ci,
    describe,
    fmt,
    load_panel_or_raise,
    md_table,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

NAME = "stability_conditioned"

#: Stability cuts reported in the main table, most reproducible first.
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

#: Stability cuts reported inside the co-activating subset.
CO_QUANTILES = (0.10, 0.25, 0.50, 1.00)

#: Stability cuts reported over the method's own pairs.
PAIR_QUANTILES = (0.01, 0.05, 0.10)

#: Stability cuts reported over the method's own pairs after the co-activation
#: filter. The last entry keeps the whole filtered pool.
PAIR_CORR_QUANTILES = (0.01, 0.05, 0.10, 1.00)

#: The two-way grid of thresholds: minimum stability of the weaker endpoint,
#: and minimum co-activation correlation of the pair.
GRID_STABILITY = (0.0, 0.8, 0.9, 0.95)
GRID_CORRELATION = (0.0, 0.4, 0.6)

#: Below this many co-activating concepts the conditional block is not reported.
MIN_CO_ACTIVATING = 5


# --------------------------------------------------------------------------- #
# stability
# --------------------------------------------------------------------------- #
def geometry_stability(Wa: np.ndarray, Wb: np.ndarray,
                       alive_a: np.ndarray, alive_b: np.ndarray) -> dict[str, Any]:
    """Pair two dictionaries by direction similarity and score each concept.

    This is the cited definition. The assignment maximizes the total cosine
    between paired concept vectors over the latents alive on each side, so a
    concept's stability is the cosine it reaches with its counterpart.

    Returns the alive row indices, the partner each one was given, the cosine
    of each pair, and the mean of those cosines.
    """
    ra, rb = np.where(alive_a)[0], np.where(alive_b)[0]
    if ra.size == 0 or rb.size == 0:
        return {"rows": ra.astype(np.int64), "partner": np.zeros(0, dtype=np.int64),
                "score": np.zeros(0), "mean_stability": float("nan")}
    S = Wa[ra] @ Wb[rb].T
    row, col = linear_sum_assignment(-S)
    return {
        "rows": ra[row],
        "partner": rb[col],
        "score": S[row, col],
        "mean_stability": float(np.mean(S[row, col])),
    }


def _matched_correlation(C: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Correlation of each row's assigned partner, one value per row."""
    rows = np.arange(C.shape[0])
    return C[rows, perm[rows]].astype(np.float64)


def _median_or_none(values: np.ndarray) -> float | None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _measure(setting: Setting, *, co_activation_min: float, n_boot: int,
             seed: int) -> dict[str, Any]:
    p_ii = load_panel_or_raise(setting.panel_path("img_img"))
    p_it = load_panel_or_raise(setting.panel_path("img_txt"))

    # In a same-modality panel the two alive masks are side A and side B, which
    # here are the image sides of the two runs.
    alive_a = np.asarray(p_ii["alive_image"], dtype=bool)
    alive_b = np.asarray(p_ii["alive_text"], dtype=bool)
    alive_i = np.asarray(p_it["alive_image"], dtype=bool)
    alive_t = np.asarray(p_it["alive_text"], dtype=bool)

    Wa = unit_decoder(setting.ckpt_a, "image")
    Wb = unit_decoder(setting.ckpt_b, "image")
    Wt = unit_decoder(setting.ckpt_a, "text")

    stab = geometry_stability(Wa, Wb, alive_a, alive_b)
    logger.info("[%s] stability over %d paired concepts: mean %.4f",
                NAME, len(stab["rows"]), stab["mean_stability"])

    C_it = np.asarray(p_it["C"], dtype=np.float64)
    perm = np.asarray(p_it["perm"], dtype=np.int64)
    usable = np.asarray(p_it["usable"], dtype=bool)
    matched_c = _matched_correlation(C_it, perm)

    cross_geom = geometry_stability(Wa, Wt, alive_i, alive_t)
    geom_partner = np.full(Wa.shape[0], -1, dtype=np.int64)
    geom_partner[cross_geom["rows"]] = cross_geom["partner"]

    # Keep the concepts alive on both sides of both comparisons.
    keep = usable[stab["rows"]]
    rows = stab["rows"][keep]
    s = stab["score"][keep]
    d_same = 1.0 - s
    d_cross = 1.0 - (Wa[rows] * Wt[perm[rows]]).sum(axis=1)
    has_geom = geom_partner[rows] >= 0
    d_cross_geom = np.full(len(rows), np.nan)
    d_cross_geom[has_geom] = 1.0 - (
        Wa[rows[has_geom]] * Wt[geom_partner[rows[has_geom]]]
    ).sum(axis=1)
    c_match = matched_c[rows]
    logger.info("[%s] concepts usable in both comparisons: %d", NAME, len(rows))

    order = np.argsort(-s)  # most reproducible first
    same_med, same_lo, same_hi = bootstrap_ci(d_same, np.median, n_boot=n_boot, seed=seed)
    cross_med, cross_lo, cross_hi = bootstrap_ci(d_cross, np.median, n_boot=n_boot,
                                                 seed=seed + 1)

    report: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "ckpt_a": str(setting.ckpt_a),
        "ckpt_b": str(setting.ckpt_b),
        "alive_rule": p_it.get("_meta", {}).get("alive_rule", "fire_count >= 1"),
        "n_samples": int(p_it["n_samples"]),
        "co_activation_min": float(co_activation_min),
        "mean_stability": stab["mean_stability"],
        "n_concepts": int(len(rows)),
        "same_modality_distance_all": describe(d_same),
        "cross_modal_distance_all": describe(d_cross),
        "cross_modal_distance_all_matched_by_geometry": describe(d_cross_geom[has_geom]),
        "bootstrap": {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "note": ("percentile bootstrap over the concepts; the paper's script "
                     "reported these medians without an interval"),
            "same_modality_distance_median_ci95": [same_lo, same_hi],
            "cross_modal_distance_median_ci95": [cross_lo, cross_hi],
        },
        "operator_vs_modality": {
            "note": ("same-modality pairing maximizes decoder cosine, cross-modal "
                     "pairing maximizes co-activation correlation, so the two "
                     "columns are not produced the same way; pairing the "
                     "cross-modal side by decoder cosine too isolates the "
                     "modality component"),
            "same_modality": same_med,
            "cross_modal_matched_by_geometry": _median_or_none(d_cross_geom),
            "cross_modal_matched_by_coactivation": cross_med,
            "attributable_to_operator": None,
            "attributable_to_modality": None,
        },
        "matched_correlation_all": describe(c_match),
        "by_stability_quantile": {},
        "by_decile": {},
        "co_activating_only": {},
    }
    geom_med = report["operator_vs_modality"]["cross_modal_matched_by_geometry"]
    if geom_med is not None and np.isfinite(cross_med) and np.isfinite(same_med):
        report["operator_vs_modality"]["attributable_to_operator"] = float(cross_med - geom_med)
        report["operator_vs_modality"]["attributable_to_modality"] = float(geom_med - same_med)

    for q in QUANTILES:
        if len(rows) == 0:
            break
        k = max(1, int(round(q * len(rows))))
        sel = order[:k]
        report["by_stability_quantile"][f"top_{int(q * 100)}pct"] = {
            "n": int(k),
            "stability_median": float(np.median(s[sel])),
            "same_modality_distance_median": float(np.median(d_same[sel])),
            "cross_modal_distance_median": float(np.median(d_cross[sel])),
            "cross_modal_distance_median_matched_by_geometry":
                _median_or_none(d_cross_geom[sel]),
            "matched_correlation_median": float(np.median(c_match[sel])),
        }

    for dcl in range(10):
        lo, hi = int(dcl * len(rows) / 10), int((dcl + 1) * len(rows) / 10)
        sel = order[lo:hi]
        if len(sel) == 0:
            continue
        report["by_decile"][f"d{dcl + 1}"] = {
            "n": int(len(sel)),
            "stability_median": float(np.median(s[sel])),
            "cross_modal_distance_median": float(np.median(d_cross[sel])),
        }

    # The cut that matters for the paper's claim: concepts whose cross-modal
    # partner genuinely co-activates with them.
    co = c_match >= co_activation_min
    if co.sum() >= MIN_CO_ACTIVATING:
        s_co, d_co = s[co], d_cross[co]
        o_co = np.argsort(-s_co)
        entry: dict[str, Any] = {
            "threshold": float(co_activation_min),
            "n": int(co.sum()),
            "cross_modal_distance": describe(d_co),
            "same_modality_distance": describe(d_same[co]),
        }
        for q in CO_QUANTILES:
            k = max(1, int(round(q * len(s_co))))
            sel = o_co[:k]
            entry[f"top_{int(q * 100)}pct_by_stability"] = {
                "n": int(k),
                "stability_median": float(np.median(s_co[sel])),
                "cross_modal_distance_median": float(np.median(d_co[sel])),
            }
        report["co_activating_only"] = entry
    else:
        report["co_activating_only"] = {
            "threshold": float(co_activation_min),
            "n": int(co.sum()),
            "note": "too few co-activating pairs to condition on",
        }

    # Reproducible on both sides AND actually corresponding.
    #
    # Conditioning on stability alone breaks in both directions. Restricting to
    # stable image atoms and following the assignment leaves the text endpoint
    # unconstrained, while forcing a bijection inside the top few percent of
    # each side pairs concepts that have no reason to correspond. Either way the
    # matched correlation collapses and the pair stops being a correspondence.
    # The defensible cut requires both endpoints to be reproducible and the pair
    # to genuinely co-activate.
    txt_txt = setting.panel_path("txt_txt")
    if txt_txt.exists():
        report["stable_and_corresponding"] = _both_endpoints(
            setting, txt_txt, Wa=Wa, Wt=Wt, perm=perm, usable=usable,
            matched_c=matched_c, img_stab=stab, co_activation_min=co_activation_min,
        )
    else:
        logger.warning("[%s] %s: no text-to-text panel at %s, the both-endpoint "
                       "block is not reported", NAME, setting.tag, txt_txt)
        report["stable_and_corresponding"] = {}
    return report


def _both_endpoints(setting: Setting, txt_txt: Path, *, Wa: np.ndarray, Wt: np.ndarray,
                    perm: np.ndarray, usable: np.ndarray, matched_c: np.ndarray,
                    img_stab: dict[str, Any],
                    co_activation_min: float) -> dict[str, Any]:
    """Filter the method's own pairs by both endpoints' stability and by correlation."""
    p_tt = load_panel_or_raise(txt_txt)
    alive_ta = np.asarray(p_tt["alive_image"], dtype=bool)
    alive_tb = np.asarray(p_tt["alive_text"], dtype=bool)
    Wtb = unit_decoder(setting.ckpt_b, "text")
    stab_t = geometry_stability(Wt, Wtb, alive_ta, alive_tb)

    img_score = np.full(Wa.shape[0], -np.inf)
    img_score[img_stab["rows"]] = img_stab["score"]
    txt_score = np.full(Wt.shape[0], -np.inf)
    txt_score[stab_t["rows"]] = stab_t["score"]

    pair_rows = np.where(usable)[0]
    pair_cos = (Wa[pair_rows] * Wt[perm[pair_rows]]).sum(axis=1)
    pair_c = matched_c[pair_rows]
    pair_stab = np.minimum(img_score[pair_rows], txt_score[perm[pair_rows]])
    finite = np.isfinite(pair_stab)

    grid: dict[str, Any] = {}
    for s_min in GRID_STABILITY:
        for c_min in GRID_CORRELATION:
            sel = finite & (pair_stab >= s_min) & (pair_c >= c_min)
            n = int(sel.sum())
            grid[f"stability>={s_min}, c>={c_min}"] = {
                "n": n,
                "cosine_median": float(np.median(pair_cos[sel])) if n else None,
                "distance_median": float(np.median(1.0 - pair_cos[sel])) if n else None,
                "matched_correlation_median": float(np.median(pair_c[sel])) if n else None,
                "pair_stability_median": float(np.median(pair_stab[sel])) if n else None,
            }

    quant: dict[str, Any] = {}
    sp_ok, cos_ok, c_ok = pair_stab[finite], pair_cos[finite], pair_c[finite]
    order_s = np.argsort(-sp_ok)
    for q in PAIR_QUANTILES:
        if sp_ok.size == 0:
            break
        k = max(1, int(round(q * len(sp_ok))))
        sel = order_s[:k]
        quant[f"top_{int(q * 100)}pct_by_stability"] = {
            "n": int(k),
            "pair_stability_median": float(np.median(sp_ok[sel])),
            "cosine_median": float(np.median(cos_ok[sel])),
            "distance_median": float(np.median(1.0 - cos_ok[sel])),
            "matched_correlation_median": float(np.median(c_ok[sel])),
        }

    # Keep only pairs that genuinely correspond, then take the most reproducible
    # 1, 5 and 10 percent of those, and the whole filtered pool.
    quant_corr: dict[str, Any] = {}
    corr = finite & (pair_c >= co_activation_min)
    sp_c, cos_c, c_c = pair_stab[corr], pair_cos[corr], pair_c[corr]
    order_c = np.argsort(-sp_c)
    for q in PAIR_CORR_QUANTILES:
        if len(sp_c) == 0:
            break
        k = max(1, int(round(q * len(sp_c))))
        sel = order_c[:k]
        quant_corr[f"top_{int(q * 100)}pct"] = {
            "n": int(k),
            "pool_size": int(len(sp_c)),
            "pair_stability_median": float(np.median(sp_c[sel])),
            "cosine_median": float(np.median(cos_c[sel])),
            "distance_median": float(np.median(1.0 - cos_c[sel])),
            "matched_correlation_median": float(np.median(c_c[sel])),
        }

    return {
        "co_activation_min": float(co_activation_min),
        "by_stability_quantile_corresponding": quant_corr,
        "by_stability_quantile": quant,
        "text_mean_stability": stab_t["mean_stability"],
        "n_pairs_scored": int(finite.sum()),
        "note": ("pair stability is the weaker of the two endpoints; every cell "
                 "is one of the method's own Hungarian pairs, filtered, never "
                 "rematched"),
        "grid": grid,
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _quantile_label(key: str) -> str:
    """Turn a key such as top_5pct into the phrase most reproducible 5 percent."""
    digits = "".join(ch for ch in key if ch.isdigit())
    if digits == "100":
        return "all concepts"
    return f"most reproducible {digits} percent"


def _report(setting: Setting, r: dict[str, Any], path: Path) -> None:
    cmin = r["co_activation_min"]
    intro = (
        f"Setting: {setting.title()}. Two independently trained image "
        f"dictionaries are paired by the assignment that maximizes their total "
        f"decoder cosine, and each concept's stability is the cosine it reaches "
        f"with its counterpart in the other run. The mean stability over the "
        f"paired concepts is {fmt(r['mean_stability'])}. "
        f"{r['n_concepts']:,} concepts are alive on both sides of that "
        f"comparison and also matched and alive on both sides of the "
        f"image-to-text panel, which is the set every table below is computed "
        f"on. The panels were built on the full {setting.split} split, "
        f"{r['n_samples']:,} image-caption pairs, and a latent counts as alive "
        f"when it fired at least once over them."
    )
    caution = (
        "The same-modality column and the cross-modal column are not produced "
        "the same way. The same-modality side is paired by decoder cosine and "
        "then reports that cosine, so it optimizes exactly the quantity it "
        "reports, while the cross-modal side reports the decoder cosine of a "
        "pair chosen by co-activation correlation. Part of the gap is therefore "
        "the pairing rule rather than modality, so a column pairing the "
        "cross-modal side by decoder cosine too is carried alongside."
    )

    q_rows = [
        [_quantile_label(name), f"{e['n']:,}", fmt(e["stability_median"]),
         fmt(e["same_modality_distance_median"]),
         fmt(e["cross_modal_distance_median"]),
         fmt(e["cross_modal_distance_median_matched_by_geometry"]),
         fmt(e["matched_correlation_median"])]
        for name, e in r["by_stability_quantile"].items()
    ]
    q_table = md_table(
        ["concepts kept, most reproducible first", "concepts",
         "stability (cosine between the two runs), median",
         "cosine distance to the other run, median",
         "cosine distance to the text partner chosen by co-activation, median",
         "cosine distance to the text partner chosen by decoder cosine, median",
         "co-activation correlation of the cross-modal pair, median"],
        q_rows,
    ) if q_rows else ""

    ov = r["operator_vs_modality"]
    ov_table = md_table(
        ["quantity", "cosine distance"],
        [
            ["same modality, partner chosen by decoder cosine", fmt(ov["same_modality"])],
            ["cross modality, partner chosen by decoder cosine",
             fmt(ov["cross_modal_matched_by_geometry"])],
            ["cross modality, partner chosen by co-activation correlation",
             fmt(ov["cross_modal_matched_by_coactivation"])],
            ["difference attributable to the pairing rule",
             fmt(ov["attributable_to_operator"])],
            ["difference attributable to modality", fmt(ov["attributable_to_modality"])],
        ],
    )

    boot = r["bootstrap"]
    boot_table = md_table(
        ["quantity", "median", "95 percent interval"],
        [
            ["same-modality cosine distance over all concepts",
             fmt(ov["same_modality"]),
             (f"[{fmt(boot['same_modality_distance_median_ci95'][0])}, "
              f"{fmt(boot['same_modality_distance_median_ci95'][1])}]")],
            ["cross-modal cosine distance over all concepts",
             fmt(ov["cross_modal_matched_by_coactivation"]),
             (f"[{fmt(boot['cross_modal_distance_median_ci95'][0])}, "
              f"{fmt(boot['cross_modal_distance_median_ci95'][1])}]")],
        ],
    )

    d_rows = [
        [f"decile {i + 1} of 10", f"{e['n']:,}", fmt(e["stability_median"]),
         fmt(e["cross_modal_distance_median"])]
        for i, e in enumerate(r["by_decile"].values())
    ]
    d_table = md_table(
        ["group, most reproducible first", "concepts",
         "stability (cosine between the two runs), median",
         "cosine distance to the text partner, median"],
        d_rows,
    ) if d_rows else ""

    tables = [
        ("Cosine distance by how reproducible the concept is", q_table),
        ("The same measurement with both sides paired the same way", ov_table),
        ("Bootstrap interval on the two headline medians", boot_table),
        ("Cosine distance to the text partner, by decile of stability", d_table),
    ]
    paragraphs = [intro, caution]

    co = r["co_activating_only"]
    if co.get("n", 0) >= MIN_CO_ACTIVATING:
        co_rows = [[
            "all of them", f"{co['n']:,}", "n/a",
            fmt(co["cross_modal_distance"].get("median")),
        ]]
        for q in CO_QUANTILES:
            e = co.get(f"top_{int(q * 100)}pct_by_stability")
            if e is None:
                continue
            co_rows.append([
                _quantile_label(f"top_{int(q * 100)}pct"), f"{e['n']:,}",
                fmt(e["stability_median"]), fmt(e["cross_modal_distance_median"]),
            ])
        tables.append((
            (f"Concepts whose cross-modal partner co-activates with "
             f"correlation at least {cmin:g}"),
            md_table(
                ["concepts kept, most reproducible first", "concepts",
                 "stability (cosine between the two runs), median",
                 "cosine distance to the text partner, median"],
                co_rows,
            ),
        ))
    else:
        paragraphs.append(
            f"Only {co.get('n', 0)} concepts have a cross-modal partner whose "
            f"co-activation correlation reaches {cmin:g}, which is fewer than "
            f"{MIN_CO_ACTIVATING}, so the conditional table is not reported."
        )

    sc = r.get("stable_and_corresponding") or {}
    if sc:
        paragraphs.append(
            "Conditioning on stability alone selects pairs that are no longer "
            "correspondences, which the matched correlation column makes "
            "visible: a reproducible direction whose partner fires on different "
            "inputs is not the same concept. The table below therefore applies "
            "both conditions at once. Every row is a subset of the pairs the "
            "method's own assignment produced, never a rematching, and a pair's "
            "stability is the weaker of its two endpoints. The text dictionaries "
            f"of the two runs reach mean stability {fmt(sc['text_mean_stability'])}, "
            f"and {sc['n_pairs_scored']:,} pairs could be scored on both endpoints."
        )
        rows = []
        for key, e in sc.get("by_stability_quantile_corresponding", {}).items():
            digits = "".join(ch for ch in key if ch.isdigit())
            label = (f"correlation at least {cmin:g}, all of them" if digits == "100"
                     else f"correlation at least {cmin:g}, most reproducible {digits} percent")
            rows.append([label, f"{e['n']:,}", fmt(e["cosine_median"]),
                         fmt(e["distance_median"]),
                         fmt(e["matched_correlation_median"])])
        for key, e in sc.get("by_stability_quantile", {}).items():
            digits = "".join(ch for ch in key if ch.isdigit())
            rows.append([f"no correlation condition, most reproducible {digits} percent",
                         f"{e['n']:,}", fmt(e["cosine_median"]),
                         fmt(e["distance_median"]),
                         fmt(e["matched_correlation_median"])])
        for key, e in sc.get("grid", {}).items():
            s_part, c_part = key.split(", ")
            label = (f"pair stability at least {s_part.split('>=')[1]}, "
                     f"correlation at least {c_part.split('>=')[1]}")
            rows.append([label, f"{e['n']:,}", fmt(e["cosine_median"]),
                         fmt(e["distance_median"]),
                         fmt(e["matched_correlation_median"])])
        if rows:
            tables.append((
                ("The method's own pairs, filtered by both endpoints' "
                 "stability and by co-activation"),
                md_table(
                    ["condition", "pairs",
                     "cosine between the two decoder directions, median",
                     "cosine distance, median",
                     "co-activation correlation, median"],
                    rows,
                ),
            ))
    else:
        paragraphs.append(
            "The text-to-text panel is not available for this setting, so the "
            "block that requires both endpoints of a pair to be reproducible is "
            "not reported."
        )

    write_md(path, "Cross-modal distance on concepts that two runs both recover",
             paragraphs, [(c, t) for c, t in tables if t])


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        tau: float = 0.4, n_boot: int = 1000, co_activation_min: float = 0.6,
        seed: int = 0, **knobs: Any) -> dict[str, Any]:
    """Condition the cross-modal distance on how reproducible a concept is.

    setting            the configuration to measure. Reads the image-to-image
                       panel, the image-to-text panel, the text-to-text panel
                       when it exists, and both checkpoints.
    out_dir            directory for stability_conditioned.json and .md.
    device             accepted for a uniform interface; every step runs on the
                       CPU over arrays the panels already hold.
    tau                accepted for a uniform interface and recorded in the
                       json. The correlation threshold this analysis conditions
                       on is `co_activation_min`, which the paper's script fixed
                       at 0.6 and which the two-way grid also evaluates at 0.0
                       and 0.4.
    n_boot             bootstrap resamples behind the interval on the two
                       headline medians. The paper's script reported those
                       medians with no interval.
    co_activation_min  correlation above which a cross-modal pair counts as a
                       genuine correspondence.
    seed               seed of the bootstrap resampling.

    Returns the payload it wrote. Skips the work and returns the payload
    already on disk when stability_conditioned.json exists.
    """
    out_dir = Path(out_dir)
    json_path = out_dir / f"{NAME}.json"
    if json_path.exists():
        logger.info("[%s][skip] %s exists", NAME, json_path)
        return json.loads(json_path.read_text())

    report = _measure(setting, co_activation_min=co_activation_min,
                      n_boot=n_boot, seed=seed)
    report["tau"] = float(tau)
    write_json(json_path, report)
    _report(setting, report, out_dir / f"{NAME}.md")
    logger.info("[%s] wrote %s", NAME, json_path)
    return report


__all__ = ["run", "NAME", "geometry_stability", "QUANTILES"]

"""How strong, and how unambiguous, are the matches the method actually makes?

A reviewer asked for the correlation distribution of the Hungarian matches and
the share of low-confidence ones. "Low confidence" has no standard definition,
so this reports three readings of it.

Strength is the first reading: the co-activation correlation of each matched
pair, reported as a histogram in bands rather than as a handful of percentiles,
so that the length of the weak tail is visible. Statistical significance is
deliberately not reported: with millions of paired samples almost any non-zero
correlation clears a significance bar, so it would answer a question nobody
asked.

Ambiguity is the second reading: how far the assigned partner beats the runner
up. A row whose top two candidates are nearly tied could have been matched
elsewhere without changing the assignment's total by much.

Reciprocity is the third: whether the two latents are each other's first choice,
rather than one settling for the other after the assignment resolved a conflict.

A noise floor comes from the shuffled panel `img_txt_null.npz`, which the
rebuttal stage builds by destroying the image-to-caption pairing with a row
shuffle and recomputing the correlations and the assignment from scratch. The
cheaper trick of permuting the columns of a finished correlation matrix does not
work, because each row's maximum survives it and the assignment simply finds the
same value again.

Every statistic over matched pairs is restricted to the panel's `usable` rows,
the rows alive on both sides. The alive rule, the correlation and the assignment
are all read from the panel; none of them is recomputed here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plotting.multi_density import BIN_EDGES, bin_mask, bin_name  # noqa: E402
from src.rebuttal.common import (  # noqa: E402
    Setting,
    describe,
    fmt,
    load_panel_or_raise,
    matched_distance,
    md_table,
    pct,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Stem of the three files this analysis writes.
NAME = "match_confidence"

#: Pairing name of the shuffled panel that supplies the noise floor. It resolves
#: to `<panels_dir>/img_txt_null.npz`, which is a separate file from the paper's
#: image-to-text panel.
NULL_PAIRING = "img_txt_null"

#: Width of the fine bands, the ones the paper's table used. The coarse bands
#: are the Figure 2 edges, imported above so the two tables cannot drift apart.
FINE_BAND_WIDTH = 0.1

#: Percentile of the shuffled panel's matched correlations taken as the floor.
FLOOR_PERCENTILE = 99.0

#: Cutoffs whose cumulative share is marked on the figure's right panel.
FIGURE_CUTOFFS = (0.1, 0.2, 0.4)


# --------------------------------------------------------------------------- #
# Reading a panel
# --------------------------------------------------------------------------- #
def matched_correlation(panel: dict[str, Any]) -> np.ndarray:
    """Correlation of every row's assigned partner, one value per row.

    `perm[i]` is the partner of row `i`, so this is `C[i, perm[i]]`. The value
    is meaningless for a row outside `usable`, whose partner is arbitrary, so a
    caller masks with `usable` before summarizing.
    """
    C = np.asarray(panel["C"], dtype=np.float64)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    return C[np.arange(C.shape[0]), perm]


def usable_matched_correlation(panel: dict[str, Any]) -> np.ndarray:
    """Matched correlation of the usable rows only, in row order."""
    usable = np.asarray(panel["usable"], dtype=bool)
    return matched_correlation(panel)[usable]


# --------------------------------------------------------------------------- #
# The three readings
# --------------------------------------------------------------------------- #
def coarse_bands(c_matched: np.ndarray, d_matched: np.ndarray) -> list[dict[str, Any]]:
    """Matched pairs per Figure 2 correlation band, with their cosine distance.

    Bands are the Figure 2 edges [0, 0.2, 0.4, 0.6, 0.8, 1.0], top band closed
    on the right. A matched pair whose correlation is negative sits below every
    band and is returned as a final row named "below 0.0", so that no pair is
    silently dropped.
    """
    total = int(c_matched.size)
    rows: list[dict[str, Any]] = []
    for b in range(len(BIN_EDGES) - 1):
        mask = bin_mask(c_matched, b)
        n = int(mask.sum())
        row: dict[str, Any] = {
            "band": bin_name(b),
            "lo": BIN_EDGES[b],
            "hi": BIN_EDGES[b + 1],
            "n_pairs": n,
            "share_of_matched_pairs": n / total if total else 0.0,
        }
        if n:
            row["cosine_distance_median"] = float(np.median(d_matched[mask]))
            row["cosine_distance_mean"] = float(np.mean(d_matched[mask]))
        rows.append(row)
    neg = c_matched < BIN_EDGES[0]
    n_neg = int(neg.sum())
    row = {
        "band": "below 0.0",
        "lo": None,
        "hi": BIN_EDGES[0],
        "n_pairs": n_neg,
        "share_of_matched_pairs": n_neg / total if total else 0.0,
    }
    if n_neg:
        row["cosine_distance_median"] = float(np.median(d_matched[neg]))
        row["cosine_distance_mean"] = float(np.mean(d_matched[neg]))
    rows.append(row)
    return rows


def fine_bands(c_matched: np.ndarray) -> list[dict[str, Any]]:
    """Matched pairs in bands of 0.1, counted from the strongest band down.

    This is the table the paper reported. The bands run [0.9, 1.0] down to
    [0.0, 0.1), followed by one row for the negative correlations, and each row
    carries the share of matched pairs at or above its lower edge, so a reader
    can see at a glance how much mass sits above any one cutoff.
    """
    total = int(c_matched.size)
    rows: list[dict[str, Any]] = []
    n_steps = int(round(1.0 / FINE_BAND_WIDTH))
    for step in range(n_steps - 1, -1, -1):
        lo = step * FINE_BAND_WIDTH
        hi = lo + FINE_BAND_WIDTH
        if step == n_steps - 1:
            mask = c_matched >= lo
            label = f"[{lo:.1f}, 1.0]"
        else:
            mask = (c_matched >= lo) & (c_matched < hi)
            label = f"[{lo:.1f}, {hi:.1f})"
        n = int(mask.sum())
        rows.append({
            "band": label,
            "n_pairs": n,
            "share_of_matched_pairs": n / total if total else 0.0,
        })
    neg = c_matched < 0.0
    n_neg = int(neg.sum())
    rows.append({
        "band": "below 0.0",
        "n_pairs": n_neg,
        "share_of_matched_pairs": n_neg / total if total else 0.0,
    })
    running = 0.0
    for row in rows:
        running += row["share_of_matched_pairs"]
        row["cumulative_share_from_the_top"] = running
    return rows


def ambiguity(C: np.ndarray, rows: np.ndarray, cols_alive: np.ndarray) -> dict[str, Any]:
    """How far each usable row's best candidate beats its second best.

    Candidates are the alive columns only, since a dead column can never be
    assigned. Returns the margin (best minus second best), the ratio (second
    best over best) and the share of rows whose runner up sits within 10 and
    within 50 percent of the winner. A panel with fewer than two alive columns
    has no runner up and returns `{"n": 0}`.
    """
    if cols_alive.size < 2 or rows.size == 0:
        return {"n": 0}
    sub = C[np.ix_(rows, cols_alive)]
    part = np.partition(sub, -2, axis=1)
    best, second = part[:, -1], part[:, -2]
    margin = best - second
    ratio = second / np.maximum(best, 1e-9)
    return {
        "n": int(rows.size),
        "best_minus_second_best": describe(margin),
        "second_best_over_best": describe(ratio),
        "share_runner_up_within_10_percent": float(np.mean(ratio > 0.9)),
        "share_runner_up_within_50_percent": float(np.mean(ratio > 0.5)),
    }


def reciprocity(C: np.ndarray, rows: np.ndarray, rows_alive: np.ndarray,
                cols_alive: np.ndarray, perm: np.ndarray) -> dict[str, Any]:
    """Share of usable rows that are their partner's first choice and vice versa.

    A row and its assigned column are mutual first choices when the row's
    highest correlation over the alive columns falls on that column, and the
    column's highest correlation over the alive rows falls back on that row.
    """
    if rows.size == 0 or cols_alive.size == 0 or rows_alive.size == 0:
        return {"n": 0}
    row_best_col = cols_alive[C[np.ix_(rows, cols_alive)].argmax(axis=1)]
    partner = perm[rows]
    col_best_row = rows_alive[C[np.ix_(rows_alive, partner)].argmax(axis=0)]
    mutual = (row_best_col == partner) & (col_best_row == rows)
    return {
        "n": int(rows.size),
        "n_mutual_first_choice": int(mutual.sum()),
        "share_mutual_first_choice": float(np.mean(mutual)),
    }


def noise_floor(null_panel: dict[str, Any], c_matched: np.ndarray) -> dict[str, Any]:
    """The correlation the same procedure finds once the pairing is destroyed.

    The floor is the 99th percentile of the shuffled panel's own matched
    correlations, computed on that panel and on nothing else, together with the
    share of the real matched pairs that fall below it.
    """
    c_null = usable_matched_correlation(null_panel)
    if c_null.size == 0:
        return {"n": 0}
    floor = float(np.percentile(c_null, FLOOR_PERCENTILE))
    meta = null_panel.get("_meta", {})
    return {
        "n": int(c_null.size),
        "shuffle_seed": int(meta.get("shuffle_seed", 0)),
        "n_samples": int(meta.get("n_samples", int(null_panel["n_samples"]))),
        "matched_correlation": describe(c_null),
        "percentile": FLOOR_PERCENTILE,
        "floor": floor,
        "share_of_real_matches_below_floor": float(np.mean(c_matched < floor)),
        "n_real_matches_below_floor": int(np.sum(c_matched < floor)),
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot(out_path: Path, c_matched: np.ndarray, c_null: np.ndarray | None) -> None:
    """Histogram of the matched correlations, and their cumulative share."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.9))

    ax = axes[0]
    bins = np.linspace(-0.1, 1.0, 90)
    ax.hist(c_matched, bins=bins, color="#206987", alpha=0.75, label="matched pairs")
    if c_null is not None and c_null.size:
        ax.hist(c_null, bins=bins, color="#df3a3d", alpha=0.55, label="pairing destroyed")
    ax.set_yscale("log")
    ax.set_xlabel("co-activation correlation of the matched pair", fontsize=7.5, labelpad=1)
    ax.set_ylabel("matched pairs", fontsize=7.5, labelpad=2)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    order = np.sort(c_matched)
    ax.plot(order, np.arange(1, order.size + 1) / max(order.size, 1),
            color="#206987", lw=1.0)
    for cut in FIGURE_CUTOFFS:
        ax.axvline(cut, color="0.6", ls=":", lw=0.7)
        ax.text(cut, 0.04, f"{100 * float(np.mean(c_matched < cut)):.0f}%", fontsize=6,
                ha="right", va="bottom", color="0.35", rotation=90)
    ax.set_xlabel("co-activation correlation", fontsize=7.5, labelpad=1)
    ax.set_ylabel("share of matches below", fontsize=7.5, labelpad=2)
    ax.set_ylim(0, 1)

    for a in axes:
        a.tick_params(labelsize=6.5, pad=1)
        a.grid(axis="y", alpha=0.15, linewidth=0.4)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(out_path.with_suffix(ext), dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _coarse_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append([
            row["band"],
            fmt(row["n_pairs"]),
            pct(row["share_of_matched_pairs"]),
            fmt(row.get("cosine_distance_median"), 4),
            fmt(row.get("cosine_distance_mean"), 4),
        ])
    return md_table(
        ["co-activation correlation of the matched pair",
         "matched pairs (count)",
         "share of all matched pairs (percent)",
         "decoder cosine distance, median",
         "decoder cosine distance, mean"],
        body,
    )


def _fine_table(rows: list[dict[str, Any]]) -> str:
    body = [[
        row["band"],
        fmt(row["n_pairs"]),
        pct(row["share_of_matched_pairs"]),
        pct(row["cumulative_share_from_the_top"]),
    ] for row in rows]
    return md_table(
        ["co-activation correlation of the matched pair",
         "matched pairs (count)",
         "share of all matched pairs (percent)",
         "share of matched pairs in this band or a stronger one (percent)"],
        body,
    )


def _summary_table(payload: dict[str, Any]) -> str:
    quantities = [
        ("co-activation correlation of the matched pair (unitless, -1 to 1)",
         payload["matched_correlation"]),
        ("decoder cosine distance of the matched pair (unitless, 0 to 2)",
         payload["matched_cosine_distance"]),
    ]
    amb = payload.get("ambiguity", {})
    if amb.get("n"):
        quantities.append(
            ("best correlation minus second best, over alive text latents (unitless)",
             amb["best_minus_second_best"]))
        quantities.append(
            ("second best correlation divided by best, over alive text latents (unitless)",
             amb["second_best_over_best"]))
    body = []
    for label, d in quantities:
        body.append([
            label, fmt(d.get("n")), fmt(d.get("mean"), 4), fmt(d.get("p05"), 4),
            fmt(d.get("p25"), 4), fmt(d.get("median"), 4), fmt(d.get("p75"), 4),
            fmt(d.get("p95"), 4),
        ])
    return md_table(
        ["quantity", "matched pairs (count)", "mean", "5th percentile",
         "25th percentile", "median", "75th percentile", "95th percentile"],
        body,
    )


def _confidence_table(payload: dict[str, Any]) -> str:
    amb = payload.get("ambiguity", {})
    rec = payload.get("reciprocity", {})
    body = [
        ["matched pairs whose runner-up correlation is within 10 percent of the "
         "assigned partner's",
         pct(amb.get("share_runner_up_within_10_percent")) if amb.get("n") else "n/a"],
        ["matched pairs whose runner-up correlation is within 50 percent of the "
         "assigned partner's",
         pct(amb.get("share_runner_up_within_50_percent")) if amb.get("n") else "n/a"],
        ["matched pairs that are each other's first choice",
         pct(rec.get("share_mutual_first_choice")) if rec.get("n") else "n/a"],
    ]
    return md_table(["quantity", "share of all matched pairs (percent)"], body)


def _floor_table(floor: dict[str, Any]) -> str:
    d = floor["matched_correlation"]
    body = [
        ["matched pairs on the shuffled panel (count)", fmt(floor["n"])],
        ["median correlation of those pairs (unitless)", fmt(d["median"], 4)],
        ["95th percentile of those pairs (unitless)", fmt(d["p95"], 4)],
        [f"{floor['percentile']:.0f}th percentile of those pairs, the noise floor "
         "(unitless)", fmt(floor["floor"], 4)],
        ["real matched pairs falling below that floor (count)",
         fmt(floor["n_real_matches_below_floor"])],
        ["real matched pairs falling below that floor (percent)",
         pct(floor["share_of_real_matches_below_floor"])],
    ]
    return md_table(["quantity", "value"], body)


def _paragraphs(setting: Setting, payload: dict[str, Any]) -> list[str]:
    floor = payload.get("noise_floor", {})
    lead = (
        f"Setting {setting.tag}: {setting.title()}. The co-activation correlation "
        f"matrix was accumulated over {payload['n_samples']:,} image-caption pairs of "
        f"the {setting.split} split, and one signed Hungarian assignment was run on it, "
        f"restricted to the latents that fire at least once. "
        f"{payload['n_alive_image']:,} of the {payload['n_latents_image']:,} image "
        f"latents and {payload['n_alive_text']:,} of the "
        f"{payload['n_latents_text']:,} text latents are alive, which leaves "
        f"{payload['n_matched_usable']:,} matched pairs whose two sides are both alive. "
        "Every number below is computed on those matched pairs. The decoder cosine "
        "distance of a matched pair is 1 minus the cosine between the image latent's "
        "decoder direction and its partner's, both normalized to unit length."
    )
    second = (
        "Three readings of match confidence are reported. The strength of a match is "
        "its co-activation correlation, tabulated in bands. Its ambiguity is how far "
        "the assigned partner beats the next best candidate among the alive text "
        "latents, measured both as a difference and as a ratio. Its reciprocity is "
        "whether the two latents are each other's highest-correlation candidate."
    )
    paras = [lead, second]
    if floor.get("n"):
        paras.append(
            "The noise floor comes from a second panel built on the same checkpoint "
            f"with the image-to-caption pairing destroyed by a row shuffle with seed "
            f"{floor['shuffle_seed']}, over {floor['n_samples']:,} rows, after which "
            "the correlations and the assignment were recomputed from scratch. "
            f"The floor is the {floor['percentile']:.0f}th percentile of that "
            "shuffled panel's own matched correlations."
        )
    return paras


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        **knobs: Any) -> dict[str, Any]:
    """Measure how strong and how unambiguous the Hungarian matches are.

    Reads the setting's image-to-text panel and the shuffled `img_txt_null`
    panel, plus model A's two decoders. Writes `match_confidence.json`,
    `match_confidence.md`, `match_confidence.pdf` and `match_confidence.png`
    into `out_dir` and returns the payload. Skips the work when the json is
    already there.

    `device` and the pipeline's knobs (`tau`, `n_boot`, `null_seed`) are
    accepted and unused: this analysis reads finished panels, so it does no
    tensor work and draws no bootstrap.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        import json

        return json.loads(out_json.read_text())

    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    C = np.asarray(panel["C"], dtype=np.float64)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    usable = np.asarray(panel["usable"], dtype=bool)
    alive_image = np.asarray(panel["alive_image"], dtype=bool)
    alive_text = np.asarray(panel["alive_text"], dtype=bool)
    rows = np.where(usable)[0]
    rows_alive = np.where(alive_image)[0]
    cols_alive = np.where(alive_text)[0]
    c_matched = matched_correlation(panel)[usable]
    logger.info("[%s] %d usable matched pairs over %d rows",
                NAME, rows.size, int(panel["n_samples"]))

    W_img = unit_decoder(setting.ckpt_a, "image")
    W_txt = unit_decoder(setting.ckpt_a, "text")
    d_matched = matched_distance(W_img, W_txt, perm, usable)

    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "null_panel": str(setting.panel_path(NULL_PAIRING)),
        "n_samples": int(panel["n_samples"]),
        "n_latents_image": int(alive_image.size),
        "n_latents_text": int(alive_text.size),
        "n_alive_image": int(alive_image.sum()),
        "n_alive_text": int(alive_text.sum()),
        "n_matched_usable": int(rows.size),
        "matched_correlation": describe(c_matched),
        "matched_cosine_distance": describe(d_matched),
        "correlation_bands_0p2": coarse_bands(c_matched, d_matched),
        "correlation_bands_0p1": fine_bands(c_matched),
        "ambiguity": ambiguity(C, rows, cols_alive),
        "reciprocity": reciprocity(C, rows, rows_alive, cols_alive, perm),
    }

    null_panel = load_panel_or_raise(setting.panel_path(NULL_PAIRING))
    payload["noise_floor"] = noise_floor(null_panel, c_matched)
    c_null = usable_matched_correlation(null_panel)

    write_json(out_json, payload)
    if c_matched.size:
        plot(out_dir / NAME, c_matched, c_null)

    tables = [
        ("Matched pairs by correlation band, with the decoder cosine distance "
         "of the pairs in each band", _coarse_table(payload["correlation_bands_0p2"])),
        ("Matched pairs by correlation band, in steps of 0.1",
         _fine_table(payload["correlation_bands_0p1"])),
        ("Distribution of the matched pairs", _summary_table(payload)),
        ("Ambiguity and reciprocity of the assignment", _confidence_table(payload)),
    ]
    if payload["noise_floor"].get("n"):
        tables.append(
            ("Noise floor, from the panel rebuilt with the pairing destroyed",
             _floor_table(payload["noise_floor"])))
    write_md(out_dir / f"{NAME}.md",
             f"Match confidence, {setting.tag}",
             _paragraphs(setting, payload),
             tables)
    logger.info("[%s] wrote %s", NAME, out_dir / f"{NAME}.md")
    return payload


__all__ = [
    "run",
    "matched_correlation",
    "usable_matched_correlation",
    "coarse_bands",
    "fine_bands",
    "ambiguity",
    "reciprocity",
    "noise_floor",
    "NAME",
    "NULL_PAIRING",
    "FLOOR_PERCENTILE",
]

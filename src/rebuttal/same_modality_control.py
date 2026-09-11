"""How far apart two dictionaries land when only the training run differs.

The reviewer's question is whether the distance the paper reports between an
image feature direction and its text partner is larger than the distance two
SAEs land apart for no reason other than having been trained twice. This
module answers it by decomposition, reading up to four panels of one setting
and reporting the same statistic on each.

  image side, two runs, same photograph
      Both SAEs read the identical embedding, so whatever distance appears is
      pure training variability.
  text side, two runs, same caption
      The same comparison on the other modality.
  text side, two runs, two different captions of one photograph
      Same modality, but the two SAEs now read different sentences about the
      same scene. The extra distance over the previous row is what describing a
      scene two ways costs, with modality held fixed. COCO has five captions per
      photograph so this panel exists there; CC3M has one, so it does not.
  image side against text side of one run
      The measurement the paper reports. It adds modality on top of the two
      effects above.

Every number here is computed over matched pairs only. The panel carries one
signed alive-restricted Hungarian assignment in `perm` and marks in `usable`
the rows that are alive on both sides; a latent enters the statistics exactly
once, as the distance to the partner that assignment gave it. That is a
deliberate difference from the paper's script, which summarized every cell of
the correlation matrix above a threshold and then collapsed each row to its own
median so that one frequently firing latent could not dominate. With one pair
per latent that collapsing step has nothing left to do, so the json reports a
single set of numbers per restriction, under names that say what they measure.
They are `median_cosine_distance` and `mean_cosine_distance` over
`n_matched_pairs` pairs. The paper's script called its own quantities
`median_over_cells` and `median_over_latents`, and those names are deliberately
not reused here, because a value computed over one partner per latent is not the
value they held and sharing a name would invite the two to be compared as though
they were.

Two training runs are compared and no more. Model A is the setting's seed 0 and
model B is the one further run the pipeline trains, so every same-modality row
rests on exactly one pair of independent runs. The paper's script trained three
runs and tabulated all three pairings of them. That spread across run pairs is
not measured here, and the report says so rather than leaving a reader to assume
it was.

The reference value drawn in the figure is 1.0, the cosine distance between two
directions drawn independently at random in a space of this many dimensions.
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
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde, wilcoxon  # noqa: E402

from src.rebuttal.common import (  # noqa: E402
    Setting,
    bootstrap_ci,
    describe,
    fmt,
    load_panel_or_raise,
    matched_distance,
    md_table,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

NAME = "same_modality_control"

#: Correlation bands, the same edges the paper's figures use. The top band is
#: closed on the right so a pair whose correlation is exactly 1 is counted.
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BIN_COLORS = ("#df3a3d", "#d96627", "#dfb246", "#389076", "#206987")
FILL_ALPHA = 0.2
KDE_GRID = np.linspace(0.0, 1.4, 400)
XTICKS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

#: Two directions drawn independently at random sit at cosine distance 1.
RANDOM_NULL_DISTANCE = 1.0

#: Panels in the order they are reported, and which decoder each side reads.
#: The second entry of each pair says whether side B comes from the second
#: training run ("b") or from the same run as side A ("a").
PANELS: dict[str, tuple[str, str, str]] = {
    "img_img": ("image", "image", "b"),
    "txt_txt": ("text", "text", "b"),
    "txt_txt_diffcap": ("text", "text", "b"),
    "img_txt": ("image", "text", "a"),
}

#: Short title above each column of the figure.
FIGURE_TITLES = {
    "img_img": "image SAE, two runs",
    "txt_txt": "text SAE, two runs",
    "txt_txt_diffcap": "text SAE, two runs\ndifferent captions",
    "img_txt": "image vs text SAE\n(paper's measurement)",
}

#: One sentence naming what each panel compares, for the report tables.
PANEL_DESCRIPTIONS = {
    "img_img": "image side of two runs, both reading the same photograph",
    "txt_txt": "text side of two runs, both reading the same caption",
    "txt_txt_diffcap": ("text side of two runs, reading two different captions "
                        "of one photograph"),
    "img_txt": "image side against text side of one run (the paper's measurement)",
}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _band_key(lo: float, hi: float, last: bool) -> str:
    """Label of one correlation band, closed on the right for the top band."""
    return f"[{lo},{hi}]" if last else f"[{lo},{hi})"


def _bands(corr: np.ndarray, dist: np.ndarray) -> dict[str, dict[str, Any]]:
    """Count and median distance inside each correlation band."""
    out: dict[str, dict[str, Any]] = {}
    n_bands = len(BIN_EDGES) - 1
    for b in range(n_bands):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        last = b == n_bands - 1
        sel = (corr >= lo) & (corr <= hi) if last else (corr >= lo) & (corr < hi)
        n = int(sel.sum())
        out[_band_key(lo, hi, last)] = {
            "n_pairs": n,
            "median": float(np.median(dist[sel])) if n else None,
            "mean": float(np.mean(dist[sel])) if n else None,
        }
    return out


def _summarize(corr: np.ndarray, dist: np.ndarray, thr: float,
               n_boot: int, seed: int) -> dict[str, Any]:
    """Distance over the matched pairs whose correlation clears `thr`.

    One value per latent, because a latent has exactly one matched partner, so
    `n_matched_pairs` is both the number of pairs and the number of latents
    behind every figure in the entry. The `quantity` field spells that out
    inside the json, so the file states what it holds without its reader
    consulting this module.
    """
    sel = corr >= thr
    vals = dist[sel]
    med, lo, hi = bootstrap_ci(vals, np.median, n_boot=n_boot, seed=seed)
    mean, mlo, mhi = bootstrap_ci(vals, np.mean, n_boot=n_boot, seed=seed + 1)
    n = int(vals.size)
    return {
        "threshold": float(thr),
        "quantity": ("cosine distance between a latent and the single partner "
                     "the panel's Hungarian assignment gave it, over the "
                     "latents alive on both sides whose matched correlation is "
                     "at least the threshold"),
        "n_matched_pairs": n,
        "median_cosine_distance": med if n else float("nan"),
        "mean_cosine_distance": mean if n else float("nan"),
        "sd_cosine_distance": float(np.std(vals, ddof=1)) if n > 1 else float("nan"),
        "iqr_cosine_distance": (
            [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]
            if n else [float("nan"), float("nan")]
        ),
        "ci95_median": [lo, hi],
        "ci95_mean": [mlo, mhi],
    }


def _measure_panel(setting: Setting, pairing: str, panel: dict[str, Any], *,
                   headline_c: float, fallback_c: float, n_boot: int,
                   seed: int) -> dict[str, Any]:
    """Matched distance, matched correlation and the bands, for one panel."""
    side_a, side_b, which_b = PANELS[pairing]
    ckpt_b = setting.ckpt_a if which_b == "a" else setting.ckpt_b
    Wa = unit_decoder(setting.ckpt_a, side_a)
    Wb = unit_decoder(ckpt_b, side_b)

    C = np.asarray(panel["C"], dtype=np.float64)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    usable = np.asarray(panel["usable"], dtype=bool)
    if Wa.shape[0] != C.shape[0] or Wb.shape[0] != C.shape[1]:
        raise ValueError(
            f"panel {pairing} is {C.shape[0]}x{C.shape[1]} but the decoders hold "
            f"{Wa.shape[0]} and {Wb.shape[0]} latents"
        )

    rows = np.where(usable)[0]
    dist = matched_distance(Wa, Wb, perm, usable)
    corr = C[rows, perm[rows]]

    entry = {
        "pairing": pairing,
        "description": PANEL_DESCRIPTIONS[pairing],
        "panel_path": str(setting.panel_path(pairing)),
        "ckpt_a": str(setting.ckpt_a),
        "ckpt_b": str(ckpt_b),
        "side_a": side_a,
        "side_b": side_b,
        "n_alive_a": int(np.asarray(panel["alive_image"], dtype=bool).sum()),
        "n_alive_b": int(np.asarray(panel["alive_text"], dtype=bool).sum()),
        "n_usable": int(rows.size),
        "n_samples": int(panel["n_samples"]),
        "distance": describe(dist),
        "matched_correlation": describe(corr),
        "headline": _summarize(corr, dist, headline_c, n_boot, seed),
        "fallback": _summarize(corr, dist, fallback_c, n_boot, seed + 2),
        "bins": _bands(corr, dist),
        "n_below_bands": int((corr < BIN_EDGES[0]).sum()),
    }
    entry["_dist"] = dist
    entry["_corr"] = corr
    return entry


def _paired_difference(panels: dict[str, dict[str, Any]], thr: float,
                       n_boot: int, seed: int) -> dict[str, Any]:
    """Cross-modal minus same-modality distance, latent by latent.

    Both panels index the image latents of model A in the same order, so the
    comparison can be made on the latents that are usable in both and clear the
    correlation threshold in both, rather than between two summaries computed on
    different sets of latents.
    """
    a, b = panels.get("img_img"), panels.get("img_txt")
    if a is None or b is None:
        return {"threshold": float(thr), "n_latents": 0,
                "note": "the image-to-image panel is missing, so no latent is shared"}

    keep_a = np.zeros(a["_usable"].shape, dtype=bool)
    keep_a[np.where(a["_usable"])[0]] = a["_corr"] >= thr
    keep_b = np.zeros(b["_usable"].shape, dtype=bool)
    keep_b[np.where(b["_usable"])[0]] = b["_corr"] >= thr
    shared = keep_a & keep_b
    if not shared.any():
        return {"threshold": float(thr), "n_latents": 0,
                "note": "no image latent clears the threshold in both panels"}

    da = np.zeros(a["_usable"].shape)
    da[np.where(a["_usable"])[0]] = a["_dist"]
    db = np.zeros(b["_usable"].shape)
    db[np.where(b["_usable"])[0]] = b["_dist"]
    diff = db[shared] - da[shared]

    med, lo, hi = bootstrap_ci(diff, np.median, n_boot=n_boot, seed=seed)
    mean, mlo, mhi = bootstrap_ci(diff, np.mean, n_boot=n_boot, seed=seed + 1)
    try:
        pval = float(wilcoxon(diff)[1])
    except ValueError:  # every difference is exactly zero
        pval = float("nan")
    return {
        "threshold": float(thr),
        "n_latents": int(shared.sum()),
        "median_difference": med,
        "ci95": [lo, hi],
        "mean_difference": mean,
        "ci95_mean": [mlo, mhi],
        "wilcoxon_p": pval,
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def _draw(ax, entry: dict[str, Any], min_bin: int) -> int:
    """One column of the figure. Returns how many bands were drawn."""
    corr, dist = entry["_corr"], entry["_dist"]
    drawn = 0
    n_bands = len(BIN_EDGES) - 1
    for b in range(n_bands):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        last = b == n_bands - 1
        sel = (corr >= lo) & (corr <= hi) if last else (corr >= lo) & (corr < hi)
        vals = dist[sel]
        if vals.size < min_bin or float(np.std(vals)) < 1e-8:
            continue
        try:
            density = gaussian_kde(vals)(KDE_GRID)
        except np.linalg.LinAlgError:
            continue
        ax.fill_between(KDE_GRID, density, color=BIN_COLORS[b],
                        alpha=FILL_ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=BIN_COLORS[b], alpha=0.95, lw=0.7)
        drawn += 1

    ax.axvline(RANDOM_NULL_DISTANCE, color="0.45", ls=":", lw=0.8)
    ax.set_title(FIGURE_TITLES[entry["pairing"]], fontsize=7.5,
                 fontweight="bold", pad=3)
    ax.set_xlim(0.0, 1.4)
    ax.set_xticks(list(XTICKS))
    ax.set_xticklabels([f"{t:g}" for t in XTICKS])
    ax.set_xlabel(r"cosine distance $1-\cos$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=6.5, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)
    return drawn


def _figure(entries: list[dict[str, Any]], out_stem: Path, min_bin: int) -> list[str]:
    """Draw one column per panel and write the pdf and the png."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(1.85 * n, 1.75), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, entry in zip(axes, entries):
        _draw(ax, entry, min_bin)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel("density", fontsize=7.5, labelpad=2)

    n_bands = len(BIN_EDGES) - 1
    handles = [
        Patch(facecolor=to_rgba(BIN_COLORS[b], FILL_ALPHA),
              edgecolor=to_rgba(BIN_COLORS[b], 1.0), linewidth=1.0,
              label=(rf"$c \in [{BIN_EDGES[b]:.1f},{BIN_EDGES[b + 1]:.1f}]$"
                     if b == n_bands - 1 else
                     rf"$c \in [{BIN_EDGES[b]:.1f},{BIN_EDGES[b + 1]:.1f})$"))
        for b in range(n_bands)
    ] + [Line2D([], [], color="0.45", ls=":", lw=0.8, label="random directions")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6.5,
               frameon=False, handlelength=1.1, handletextpad=0.3, columnspacing=0.9,
               bbox_to_anchor=(0.5, -0.13))
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.32, top=0.84, wspace=0.18)

    written = []
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = out_stem.with_suffix(ext)
        fig.savefig(str(path), dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
        written.append(str(path))
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _report(setting: Setting, payload: dict[str, Any], path: Path) -> None:
    headline_c = payload["headline_c"]
    fallback_c = payload["fallback_c"]
    panels = payload["panels"]

    n_samples = max((e["n_samples"] for e in panels.values()), default=0)
    intro = (
        f"Setting: {setting.title()}. Each row of the table below is one "
        f"comparison between two dictionaries. For every latent that the panel's "
        f"Hungarian assignment matched to a partner and that is alive on both "
        f"sides, the cosine distance to that partner's decoder direction is "
        f"measured, together with the co-activation correlation of the matched "
        f"pair. Latents that are not alive on both sides are excluded, because "
        f"the assignment hands them an arbitrary partner. The panels were built "
        f"on the full {setting.split} split of {setting.dataset.upper()}, "
        f"{n_samples:,} image-caption pairs. Two directions drawn independently "
        f"at random in this {payload['dim']}-dimensional space sit at cosine "
        f"distance {RANDOM_NULL_DISTANCE:.2f}, which is the reference line in "
        f"the figure."
    )
    counting = (
        "Because each latent has exactly one matched partner, a pair is a "
        "latent and the count of pairs is the count of latents. Confidence "
        "intervals are 95 percent percentile bootstrap intervals over "
        f"{payload['n_boot']:,} resamples of those latents."
    )
    runs = (
        f"The two training runs compared are {setting.ckpt_a} and "
        f"{setting.ckpt_b}. They differ in their random seed and in nothing "
        f"else. Exactly one pair of runs is measured, so every row below that "
        f"names two runs reports the distance between that one pair; how much "
        f"the distance itself varies from one pair of runs to another is not "
        f"measured here and no number below should be read as bounding it."
    )

    summary_rows = []
    for e in panels.values():
        d, c = e["distance"], e["matched_correlation"]
        summary_rows.append([
            e["description"],
            f"{e['n_alive_a']:,} / {e['n_alive_b']:,}",
            f"{e['n_usable']:,}",
            fmt(d.get("median")), fmt(d.get("mean")),
            f"{fmt(d.get('p25'))} to {fmt(d.get('p75'))}",
            fmt(c.get("median")),
        ])
    summary_rows.append([
        "two directions drawn independently at random", "n/a", "n/a",
        f"{RANDOM_NULL_DISTANCE:.2f}", f"{RANDOM_NULL_DISTANCE:.2f}", "n/a", "n/a",
    ])
    summary = md_table(
        ["comparison", "alive latents, side A / side B", "matched pairs",
         "cosine distance, median", "cosine distance, mean",
         "cosine distance, 25th to 75th percentile",
         "co-activation correlation of the matched pair, median"],
        summary_rows,
    )

    head_rows = []
    for e in panels.values():
        for label, key in ((f"correlation at least {headline_c:g}", "headline"),
                           (f"correlation at least {fallback_c:g}", "fallback")):
            h = e[key]
            head_rows.append([
                e["description"], label, f"{h['n_matched_pairs']:,}",
                fmt(h["median_cosine_distance"]),
                f"[{fmt(h['ci95_median'][0])}, {fmt(h['ci95_median'][1])}]",
                fmt(h["mean_cosine_distance"]),
                f"[{fmt(h['ci95_mean'][0])}, {fmt(h['ci95_mean'][1])}]",
            ])
    head = md_table(
        ["comparison", "restriction", "matched pairs kept",
         "cosine distance, median", "median, 95 percent interval",
         "cosine distance, mean", "mean, 95 percent interval"],
        head_rows,
    )

    band_keys = list(next(iter(panels.values()))["bins"]) if panels else []
    band_rows = []
    for e in panels.values():
        row = [e["description"]]
        for key in band_keys:
            cell = e["bins"][key]
            row.append(f"{fmt(cell['median'])} ({cell['n_pairs']:,})")
        row.append(f"{e['n_below_bands']:,}")
        band_rows.append(row)
    bands = md_table(
        ["comparison"]
        + [f"correlation in {k}" for k in band_keys]
        + ["matched pairs below every band"],
        band_rows,
    ) if band_keys else ""

    paragraphs = [intro, counting, runs]
    tables = [
        ("Cosine distance between a latent and its matched partner", summary),
        (("The same distance restricted to pairs whose correlation clears a "
          "threshold"), head),
        (("Median cosine distance per correlation band, with the number of "
          "matched pairs in that band in brackets. A matched pair whose "
          "correlation is negative falls below every band and is counted in "
          "the last column"), bands),
    ]

    paired = payload.get("paired_img_txt_minus_img_img", {})
    if paired.get("n_latents"):
        paired_table = md_table(
            ["quantity", "value"],
            [
                ["image latents usable and above the threshold in both panels",
                 f"{paired['n_latents']:,}"],
                ["extra cosine distance when crossing modality, median",
                 fmt(paired["median_difference"])],
                ["median, 95 percent interval",
                 f"[{fmt(paired['ci95'][0])}, {fmt(paired['ci95'][1])}]"],
                ["extra cosine distance when crossing modality, mean",
                 fmt(paired["mean_difference"])],
                ["mean, 95 percent interval",
                 f"[{fmt(paired['ci95_mean'][0])}, {fmt(paired['ci95_mean'][1])}]"],
                ["Wilcoxon signed-rank p value",
                 "n/a" if not np.isfinite(paired["wilcoxon_p"])
                 else f"{paired['wilcoxon_p']:.2e}"],
            ],
        )
        paragraphs.append(
            "The image-to-image panel and the image-to-text panel index the same "
            "image latents, so the difference between them can be taken one "
            "latent at a time instead of between two summaries. The table below "
            "is that paired difference over the image latents that are usable in "
            f"both panels and whose matched correlation is at least {headline_c:g} "
            "in both."
        )
        tables.append(
            ("Cross-modal distance minus same-modality distance, latent by latent",
             paired_table)
        )
    elif paired:
        paragraphs.append(
            "The paired comparison between the image-to-image panel and the "
            f"image-to-text panel was not computed: {paired.get('note', 'no shared latents')}."
        )

    missing = payload.get("panels_not_available", [])
    if missing:
        paragraphs.append(
            "Panels not available for this setting, and therefore absent from "
            "every table above: " + ", ".join(missing) + "."
        )

    write_md(path, "Same-modality control: two training runs against two modalities",
             paragraphs, [(c, t) for c, t in tables if t])


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        tau: float = 0.4, n_boot: int = 1000, headline_c: float = 0.6,
        min_bin: int = 30, seed: int = 0, **knobs: Any) -> dict[str, Any]:
    """Measure the matched cosine distance on every panel of one setting.

    setting     the configuration to measure, holding the checkpoints and the
                paths of the panels.
    out_dir     directory for same_modality_control.json, .md, .pdf and .png.
    device      accepted for a uniform interface; every step here runs on the
                CPU over arrays the panels already hold.
    tau         correlation threshold of the second restriction reported in the
                table, called the fallback threshold in the paper's script.
    n_boot      bootstrap resamples behind every 95 percent interval.
    headline_c  correlation threshold of the first restriction, and the
                threshold the paired latent-by-latent comparison uses.
    min_bin     a correlation band with fewer matched pairs than this is
                reported in the tables but not drawn in the figure, because a
                density estimate over a handful of points is not informative.
    seed        seed of the bootstrap resampling.

    Returns the payload it wrote. Skips the work and returns the payload
    already on disk when same_modality_control.json exists.
    """
    out_dir = Path(out_dir)
    json_path = out_dir / f"{NAME}.json"
    if json_path.exists():
        import json

        logger.info("[%s][skip] %s exists", NAME, json_path)
        return json.loads(json_path.read_text())

    entries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for pairing in PANELS:
        path = setting.panel_path(pairing)
        if pairing != "img_txt" and not path.exists():
            logger.warning("[%s] %s: no %s panel at %s, skipping that comparison",
                           NAME, setting.tag, pairing, path)
            missing.append(f"{pairing} ({path})")
            continue
        panel = load_panel_or_raise(path)
        entries[pairing] = _measure_panel(
            setting, pairing, panel, headline_c=headline_c, fallback_c=tau,
            n_boot=n_boot, seed=seed,
        )
        entries[pairing]["_usable"] = np.asarray(panel["usable"], dtype=bool)
        logger.info("[%s] %s: %d matched pairs, median distance %.4f",
                    NAME, pairing, entries[pairing]["n_usable"],
                    entries[pairing]["distance"].get("median", float("nan")))

    dim = int(unit_decoder(setting.ckpt_a, "image").shape[1])
    paired = _paired_difference(entries, headline_c, n_boot, seed + 3)
    figures = _figure(list(entries.values()), out_dir / NAME, min_bin)

    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "dim": dim,
        "random_null_distance": RANDOM_NULL_DISTANCE,
        "headline_c": float(headline_c),
        "fallback_c": float(tau),
        "n_boot": int(n_boot),
        "min_bin": int(min_bin),
        "seed": int(seed),
        "bin_edges": list(BIN_EDGES),
        "panels": {
            name: {k: v for k, v in e.items() if not k.startswith("_")}
            for name, e in entries.items()
        },
        "panels_not_available": missing,
        "paired_img_txt_minus_img_img": paired,
        "figures": figures,
    }
    write_json(json_path, payload)
    _report(setting, payload, out_dir / f"{NAME}.md")
    logger.info("[%s] wrote %s", NAME, json_path)
    return payload


__all__ = ["run", "NAME", "BIN_EDGES", "PANELS", "RANDOM_NULL_DISTANCE"]

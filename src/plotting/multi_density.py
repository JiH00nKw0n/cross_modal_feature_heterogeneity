"""Figure 2: decoder cosine distance per co-activation bin, one panel per model.

What the figure shows. For one model, take EVERY ordered pair of latents (i, j)
where i is an image latent and j is a text latent, compute the cosine distance
`1 - cos(W_img[i], W_txt[j])` between their decoder directions, split the pairs
into five bins by their co-activation correlation `C[i, j]`, and draw one
kernel density estimate per bin.

Three rules govern the figure and they differ from the rules used elsewhere in
this repository, deliberately.

No filter. All (i, j) pairs take part: no alive mask, no Hungarian matching, no
correlation threshold. A dead latent has correlation 0 against every partner
and therefore lands in the [0, 0.2) bin; that is intended and is what makes the
lowest bin the reference distribution the other bins are read against.

Bins. Edges are [0, 0.2, 0.4, 0.6, 0.8, 1.0]. The top bin is closed on the
right, so a perfectly correlated pair is not dropped. Pairs with a negative
correlation fall below every bin and are excluded from the drawn curves; the
bin-statistics file reports how many those are.

Subsampling. With 4096 latents per side there are about 16.8 million pairs per
model, which no kernel density estimate needs. Each bin is subsampled uniformly
at random, with a fixed seed, to at most `MAX_KDE_SAMPLES` pairs before the
estimate. The statistics file is computed on the FULL bin, not the subsample,
so the reported means and medians are exact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

logger = logging.getLogger(__name__)

BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = [
    rf"$c_{{i,j}} \in [{BIN_EDGES[i]:.1f},{BIN_EDGES[i + 1]:.1f})$"
    for i in range(len(BIN_EDGES) - 1)
]
BIN_COLORS = ["#df3a3d", "#d96627", "#dfb246", "#389076", "#206987"]
ALPHA = 0.2

#: Cosine distance d = 1 - cos. cos in [-1, 1] gives d in [0, 2]; decoder pairs
#: concentrate in [0, 1.3] in practice, which is the drawn range.
KDE_GRID = np.linspace(0.0, 1.4, 400)
XTICKS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
XTICK_LABELS = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25"]

#: Pairs kept per bin before the kernel density estimate (see the subsampling rule).
MAX_KDE_SAMPLES = 2_000_000
SUBSAMPLE_SEED = 0


def unit_rows(W: np.ndarray) -> np.ndarray:
    """Decoder directions as unit-norm rows."""
    W = np.asarray(W, dtype=np.float32)
    return W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)


def bin_mask(c: np.ndarray, b: int) -> np.ndarray:
    """Membership of bin `b`; the top bin is closed so cos == 1 is kept."""
    lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
    if b < len(BIN_EDGES) - 2:
        return (c >= lo) & (c < hi)
    return (c >= lo) & (c <= hi)


def bin_name(b: int) -> str:
    lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
    closing = ")" if b < len(BIN_EDGES) - 2 else "]"
    return f"[{lo:.1f}, {hi:.1f}{closing}"


def all_pairs(C: np.ndarray, W_img_unit: np.ndarray, W_txt_unit: np.ndarray,
              ) -> tuple[np.ndarray, np.ndarray]:
    """Flattened correlation and cosine distance over EVERY (i, j) latent pair.

    Implements the no-filter rule: nothing is masked out before flattening.
    """
    C = np.asarray(C, dtype=np.float32)
    cos = W_img_unit @ W_txt_unit.T
    if cos.shape != C.shape:
        raise ValueError(f"C is {C.shape} but the decoder product is {cos.shape}")
    return C.reshape(-1), (1.0 - cos).reshape(-1)


def _describe(v: np.ndarray) -> dict[str, float]:
    """The summary the paper's bin-statistics table reports."""
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
    }


def bin_statistics(c_all: np.ndarray, dist_all: np.ndarray) -> dict:
    """Per-bin counts and cosine-distance statistics, on the FULL bin."""
    total = int(c_all.size)
    bins = []
    for b in range(len(BIN_EDGES) - 1):
        mask = bin_mask(c_all, b)
        n = int(mask.sum())
        row: dict = {"bin": bin_name(b), "lo": BIN_EDGES[b], "hi": BIN_EDGES[b + 1],
                     "n": n, "share_of_pairs": n / total if total else 0.0}
        if n:
            row["cosine_distance"] = _describe(dist_all[mask])
            row["cosine"] = _describe(1.0 - dist_all[mask])
            row["correlation"] = _describe(c_all[mask])
        bins.append(row)
    return {
        "n_pairs": total,
        "n_negative_correlation_excluded": int((c_all < BIN_EDGES[0]).sum()),
        "quantity": "cosine distance 1 - cos(W_img[i], W_txt[j]) over all (i, j) pairs",
        "bins": bins,
        "all_pairs": {
            "cosine_distance": _describe(dist_all),
            "cosine": _describe(1.0 - dist_all),
            "correlation": _describe(c_all),
        },
    }


def _subsample(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random subsample to at most MAX_KDE_SAMPLES, seeded."""
    if values.size <= MAX_KDE_SAMPLES:
        return values
    idx = rng.choice(values.size, size=MAX_KDE_SAMPLES, replace=False)
    return values[idx]


def _plot_panel(ax, c_all: np.ndarray, dist_all: np.ndarray, title: str) -> None:
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    for b in range(len(BIN_EDGES) - 1):
        mask = bin_mask(c_all, b)
        if int(mask.sum()) < 5:
            continue
        vals = _subsample(dist_all[mask], rng)
        if np.std(vals) < 1e-8:
            continue
        density = gaussian_kde(vals)(KDE_GRID)
        ax.fill_between(KDE_GRID, density, color=BIN_COLORS[b], alpha=ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=BIN_COLORS[b],
                alpha=min(ALPHA + 0.3, 1.0), lw=0.6)
    _style_panel(ax, title)


def _style_panel(ax, title: str) -> None:
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)
    ax.set_xlim(0.0, 1.3)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Cosine Distance", fontsize=8, labelpad=1)
    ax.set_ylim(0, 10.5)
    ax.set_yticks([0, 4, 8])
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=7, pad=1)
    ax.grid(axis="y", alpha=0.15, linewidth=0.4)


def plot_density(
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: str | Path,
) -> Path:
    """Draw Figure 2 and write the PDF, the PNG, the caption and the bin statistics.

    `panels` maps a display name to `(C, W_img_unit, W_txt_unit)`, where C is
    the co-activation correlation matrix from that model's panel.npz and the
    two decoder matrices already have unit-norm rows.

    Writes `<out_path>.pdf`, `<out_path>.png`, `figure2_caption.md` and
    `figure2_bin_stats.md` (plus the same statistics as JSON) next to the PDF.
    """
    if not panels:
        raise ValueError("plot_density needs at least one model")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    flattened: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (C, W_img_unit, W_txt_unit) in panels.items():
        c_all, dist_all = all_pairs(C, W_img_unit, W_txt_unit)
        flattened[name] = (c_all, dist_all)
        stats[name] = bin_statistics(c_all, dist_all)
        logger.info("[figure2] %s: %d pairs", name, c_all.size)

    ncols = len(panels)
    fig_w = 1.35 * ncols + 0.4
    fig_h = 0.85 + 0.7
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    for col, (name, (c_all, dist_all)) in enumerate(flattened.items()):
        _plot_panel(axes[0][col], c_all, dist_all, name)
        axes[0][col].set_ylabel("Density" if col == 0 else "", fontsize=8, labelpad=2)

    handles = [
        Patch(facecolor=to_rgba(BIN_COLORS[b], alpha=ALPHA),
              edgecolor=to_rgba(BIN_COLORS[b], alpha=1.0),
              linewidth=1.0, label=BIN_LABELS[b])
        for b in range(len(BIN_EDGES) - 1)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(BIN_LABELS),
               fontsize=8, frameon=False, handlelength=1.0, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.35, top=0.88,
                        wspace=0.28, hspace=0.7)

    pdf_path = out_path.with_suffix(".pdf")
    for ext in (".pdf", ".png"):
        fig.savefig(out_path.with_suffix(ext), dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
        logger.info("[figure2] saved %s", out_path.with_suffix(ext))
    plt.close(fig)

    _write_caption(pdf_path.parent / "figure2_caption.md", stats)
    _write_bin_stats(pdf_path.parent / "figure2_bin_stats.md", stats)
    with open(pdf_path.parent / "figure2_bin_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return pdf_path


def _write_caption(path: Path, stats: dict[str, dict]) -> None:
    names = ", ".join(stats)
    per_model = next(iter(stats.values()))["n_pairs"]
    lines = [
        "# Figure 2 caption",
        "",
        f"Distribution of decoder cosine distance by co-activation correlation bin, "
        f"for {names}.",
        "",
        "For each model, the two modality-specific dictionaries are compared over "
        f"every ordered pair of latents (i, j), which is {per_model:,} pairs per model. "
        "A pair contributes the cosine distance 1 - cos between the image latent i's "
        "decoder direction and the text latent j's decoder direction, and is assigned "
        "to a bin by its co-activation correlation over the training split. No filter "
        "is applied: there is no alive mask, no Hungarian matching and no correlation "
        "threshold, so latents that never fire have correlation 0 against every partner "
        "and fall in the lowest bin.",
        "",
        f"Each bin is subsampled uniformly at random, with a fixed seed "
        f"({SUBSAMPLE_SEED}), to at most {MAX_KDE_SAMPLES:,} pairs before the kernel "
        "density estimate is fitted. The per-bin counts, means and medians reported in "
        "figure2_bin_stats.md are computed on the full bin, not on the subsample.",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info("[figure2] saved %s", path)


def _write_bin_stats(path: Path, stats: dict[str, dict]) -> None:
    lines = ["# Figure 2 - per-bin statistics", "",
             "Cosine distance `1 - cos(W_img[i], W_txt[j])` of every latent pair, by "
             "co-activation correlation bin. This is the quantity the density figure "
             "draws on its x axis. Computed on the full bin, not on the subsample the "
             "kernel density estimate uses.", ""]
    for name, s in stats.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Latent pairs: {s['n_pairs']:,}. "
                     f"Pairs with a negative correlation, which fall below the lowest "
                     f"bin and are excluded from the rows: "
                     f"{s['n_negative_correlation_excluded']:,}.")
        lines.append("")
        lines.append("| correlation bin | pairs | share | mean | median | std | min | max |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for b in s["bins"]:
            if not b["n"]:
                lines.append(f"| {b['bin']} | 0 | 0.0% | - | - | - | - | - |")
                continue
            d = b["cosine_distance"]
            lines.append(
                f"| {b['bin']} | {b['n']:,} | {100 * b['share_of_pairs']:.1f}% | "
                f"{d['mean']:.4f} | {d['median']:.4f} | {d['std']:.4f} | "
                f"{d['min']:.4f} | {d['max']:.4f} |"
            )
        a = s["all_pairs"]["cosine_distance"]
        lines.append(f"| **all pairs** | {s['n_pairs']:,} | 100.0% | {a['mean']:.4f} | "
                     f"{a['median']:.4f} | {a['std']:.4f} | {a['min']:.4f} | {a['max']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines))
    logger.info("[figure2] saved %s", path)


__all__ = [
    "plot_density", "all_pairs", "bin_statistics", "unit_rows",
    "BIN_EDGES", "MAX_KDE_SAMPLES",
]

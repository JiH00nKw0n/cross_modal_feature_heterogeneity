#!/usr/bin/env python3
"""Per-model density of decoder cosine distance, laid out as a Base row
(top) and Large row (bottom).

Large runs are not available yet, so ``LARGE_MODELS`` is empty and only
the Base row is drawn. Populate ``LARGE_MODELS`` to grow the figure to a
2xN grid without code changes.

Data pipeline (no Hungarian matching): for each model's two-SAE, drop
dead latents (firing rate <= ALIVE_THR), then for every (i, j) over the
alive_i × alive_t cartesian product collect (C[i, j], cos(W_i^dec[i],
W_t^dec[j])) and overlay KDEs of cosine distance d = 1 - cos split into
5 bins of the train-time cross-correlation C.

Usage:
    python scripts/real_alpha/plot_multi_model_density_base_large.py \\
        --out outputs/multi_model_density_base_large.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from safetensors.torch import load_file
from scipy.stats import gaussian_kde


BASE_MODELS = [
    {"name": "CLIP ViT-B/32", "run_dir": "outputs/clip_b32/two_sae/final"},
    {"name": "MetaCLIP B/32", "run_dir": "outputs/metaclip_b32/two_sae/final"},
    {"name": "OpenCLIP B/32", "run_dir": "outputs/datacomp_b32/two_sae/final"},
    {"name": "SigLIP2 Base",  "run_dir": "outputs/siglip2_base/two_sae/final"},
]

# Large-variant names pulled from scripts/real_alpha/run_multi_model_pipeline.sh.
# Panels render with title only until run_dir materializes.
LARGE_MODELS = [
    {"name": "CLIP ViT-L/14", "run_dir": "outputs/clip_l14/two_sae/final"},
    {"name": "MetaCLIP L/14", "run_dir": "outputs/metaclip_l14/two_sae/final"},
    {"name": "OpenCLIP L/14", "run_dir": "outputs/datacomp_l14/two_sae/final"},
    {"name": "SigLIP2 Large", "run_dir": "outputs/siglip2_large/two_sae/final"},
]


BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BIN_LABELS = [
    rf"$c_{{i,j}} \in [{BIN_EDGES[i]:.1f},{BIN_EDGES[i+1]:.1f})$"
    for i in range(len(BIN_EDGES) - 1)
]
BIN_COLORS = ["#df3a3d", "#d96627", "#dfb246", "#389076", "#206987"]
ALIVE_THR = 0.001
ALPHA = 0.2
# Cosine distance d = 1 - cos. cos ∈ [-1, 1] → d ∈ [0, 2]; matched
# decoder pairs concentrate in [0, 1.3] in practice.
KDE_GRID = np.linspace(0.0, 1.4, 400)
XTICKS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
XTICK_LABELS = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25"]


def load_model_data(run_dir: str):
    rd = Path(run_dir)
    C = np.load(rd / "diagnostic_B_C_train.npy")
    sd = load_file(str(rd / "model.safetensors"))
    W_i = sd["image_sae.W_dec"].float().numpy()
    W_t = sd["text_sae.W_dec"].float().numpy()
    W_i_n = W_i / (np.linalg.norm(W_i, axis=1, keepdims=True) + 1e-12)
    W_t_n = W_t / (np.linalg.norm(W_t, axis=1, keepdims=True) + 1e-12)

    rates_path = rd / "diagnostic_B_firing_rates.npz"
    if rates_path.exists():
        rates = np.load(rates_path)
        alive_i = np.where(rates["rate_i"] > ALIVE_THR)[0]
        alive_t = np.where(rates["rate_t"] > ALIVE_THR)[0]
    else:
        var_i = np.var(C, axis=1)
        var_t = np.var(C, axis=0)
        alive_i = np.where(var_i > 1e-8)[0]
        alive_t = np.where(var_t > 1e-8)[0]

    C_sub = C[np.ix_(alive_i, alive_t)]
    cos_full = W_i_n[alive_i] @ W_t_n[alive_t].T
    C_all = C_sub.reshape(-1)
    dist_all = (1.0 - cos_full).reshape(-1)
    return C_all, dist_all, len(alive_i), len(alive_t)


def plot_density_subplot(ax, C_matched, dist_matched, title: str):
    for b in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        mask = (C_matched >= lo) & (C_matched < hi)
        if int(mask.sum()) < 5:
            continue
        vals = dist_matched[mask]
        if np.std(vals) < 1e-8:
            continue
        kde = gaussian_kde(vals)
        density = kde(KDE_GRID)
        ax.fill_between(KDE_GRID, density, color=BIN_COLORS[b], alpha=ALPHA, linewidth=0)
        ax.plot(KDE_GRID, density, color=BIN_COLORS[b],
                alpha=min(ALPHA + 0.3, 1.0), lw=0.6)

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


def draw_placeholder_subplot(ax, title: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="outputs/multi_model_density_base_large.pdf")
    parser.add_argument("--base", type=str, default="",
                        help="JSON with base-model list (overrides BASE_MODELS).")
    parser.add_argument("--large", type=str, default="",
                        help="JSON with large-model list (overrides LARGE_MODELS).")
    args = parser.parse_args()

    base = json.load(open(args.base)) if args.base else BASE_MODELS
    large = json.load(open(args.large)) if args.large else LARGE_MODELS

    rows = []
    if base:
        rows.append(base)
    if large:
        rows.append(large)
    assert rows, "no models configured"
    nrows = len(rows)
    ncols = max(len(r) for r in rows)

    fig_w = 1.35 * ncols + 0.4
    # Halved per-row panel height vs. the earlier square layout — each
    # panel is now ~2:1 landscape. Legend lives below the bottom row.
    fig_h = 0.85 * nrows + 0.7
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              sharex=True, sharey=True, squeeze=False)

    for r, models in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            if c < len(models):
                model = models[c]
                if Path(model["run_dir"]).exists():
                    C_m, dist_m, n_ai, n_at = load_model_data(model["run_dir"])
                    print(f"{model['name']}: alive_i={n_ai}, alive_t={n_at}, "
                          f"pairs={len(C_m)}")
                    plot_density_subplot(ax, C_m, dist_m, model["name"])
                else:
                    print(f"{model['name']}: run_dir missing — placeholder panel")
                    draw_placeholder_subplot(ax, model["name"])
            else:
                ax.axis("off")

    # xlabel only on the bottom row, but keep x tick labels on every row.
    for r in range(nrows - 1):
        for c in range(ncols):
            axes[r][c].set_xlabel("")
            axes[r][c].tick_params(axis="x", labelbottom=True)
    # ylabel only on the leftmost column.
    for r in range(nrows):
        for c in range(ncols):
            if c == 0:
                axes[r][c].set_ylabel("Density", fontsize=8, labelpad=2)
            else:
                axes[r][c].set_ylabel("")

    legend_handles = [
        Patch(
            facecolor=to_rgba(BIN_COLORS[b], alpha=ALPHA),
            edgecolor=to_rgba(BIN_COLORS[b], alpha=1.0),
            linewidth=1.0, label=BIN_LABELS[b],
        )
        for b in range(len(BIN_EDGES) - 1)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(BIN_LABELS),
        fontsize=8, frameon=False,
        handlelength=1.0, handletextpad=0.3, columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.0),
    )

    bottom = 0.35 if nrows == 1 else 0.22
    top = 0.88 if nrows == 1 else 0.92
    hspace = 0.7
    plt.subplots_adjust(left=0.07, right=0.99, bottom=bottom, top=top,
                        wspace=0.28, hspace=hspace)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.02)
    print(f"saved {out_path}")
    suffix = out_path.suffix.lower()
    for ext in (".pdf", ".png", ".svg"):
        if ext == suffix:
            continue
        alt = out_path.with_suffix(ext)
        fig.savefig(alt, dpi=200, bbox_inches="tight",
                    facecolor="white", pad_inches=0.02)
        print(f"saved {alt}")
    plt.close()


if __name__ == "__main__":
    main()

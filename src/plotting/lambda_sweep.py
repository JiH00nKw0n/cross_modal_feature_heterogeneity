#!/usr/bin/env python3
"""Lambda sweep plot: 3-panel (GRE / ESim / FSim) at fixed α=0.5.

Panels:
  (a) GT Recovery Error   (GRE)  -- untied compute_gre_top1 (lower is better;
                                    uses trained encoder, not V^T)
  (b) Embedding SIM              -- pair_cos_mean  (higher is better)
  (c) Feature SIM                -- probe_vec_cos (raw for Shared SAE;
                                    probe_vec_cos_posthoc for Post-hoc Matching)

Reconstruction Error panel removed (2026-04-27): redundant with GRE for the
phase-transition story, and the figure now focuses on GT recovery + alignment.

Methods (5 lines):
  - Shared SAE             (baseline hline, single_recon)
  - Separated SAE          (baseline hline, two_recon, raw)
  - Post-hoc Matching      (baseline hline, two_recon, posthoc on FSim)
  - Iso-Energy Align       (curve across lambda)
  - Group-Sparse           (curve across lambda)

Usage:
    python scripts/plot_lambda_sweep.py \\
        --result outputs/theorem2_v2_lambda_sweep/runs/<run>/result.json \\
        --params-dir outputs/theorem2_v2_lambda_sweep/runs/<run>/params \\
        --out outputs/theorem2_v2_lambda_sweep/lambda_sweep.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from palette import (
    STRAWBERRY_RED, SEAWEED, BLUE_SLATE, WILLOW_GREEN, ATOMIC_TANGERINE,
)
from src.configs.experiment import ExperimentConfig
from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.metrics.synthetic_eval import compute_gre_top1
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE


MULTIPLIERS = [1/64, 1/4, 1, 4, 64]
MULT_LABELS = ["1/64", "1/4", "1", "4", "64"]

IA_BASE = 0.03
GS_BASE = 0.05

METHODS = {
    "iso_align": {
        "base": IA_BASE, "color": SEAWEED,
        "label": "Iso-Energy Alignment", "marker": "^",
    },
    "group_sparse": {
        "base": GS_BASE, "color": BLUE_SLATE,
        "label": "Group-Sparse", "marker": "D",
    },
}

BASELINES = {
    "single_recon": {"color": WILLOW_GREEN,     "label": "Shared SAE",             "ls": "--"},
    "two_recon":    {"color": ATOMIC_TANGERINE, "label": "Modality-Specific SAEs", "ls": ":"},
}

# Post-hoc Alignment (Ours): solid line, no marker (highlighted as the proposed method).
POSTHOC_STYLE = {
    "color": STRAWBERRY_RED, "label": "Post-hoc Alignment (Ours)", "marker": "",
}

PANELS = ["GRE", "ESim", "FSim"]

# Metrics that should be plotted as (1 - value), so higher original ⇒ lower
# plotted value. Lets us draw "alignment distance" panels where lower = more
# aligned, matching the recon-error panels' "lower = better" convention.
INVERT_METRICS = {"ESim", "FSim"}


def _maybe_invert(metric: str, ms: tuple[float, float]) -> tuple[float, float]:
    """Apply ``1 - mean`` for metrics in ``INVERT_METRICS``; std unchanged."""
    if metric not in INVERT_METRICS:
        return ms
    mean, std = ms
    if np.isnan(mean):
        return ms
    return 1.0 - mean, std

NPZ_RE = re.compile(r"^alpha([\d.]+)_seed(\d+)_(.+)\.npz$")


def _parse(name: str):
    m = NPZ_RE.match(name)
    if m is None:
        return None
    return round(float(m.group(1)), 2), int(m.group(2)), m.group(3)


def compute_gre_for_method(
    params_dirs: list[Path], method_id: str,
) -> tuple[float, float]:
    """Average GRE across seeds for a given method_id, searching multiple dirs.

    First dir to contain matching files wins; later dirs are fallback.
    """
    paths = []
    for d in params_dirs:
        hits = sorted(d.glob(f"alpha*_seed*_{method_id}.npz"))
        if hits:
            paths = hits
            break
    vals = []
    for p in paths:
        npz = np.load(p, allow_pickle=True)
        gre_i = compute_gre_top1(
            npz["w_enc_img"], npz["b_enc_img"],
            npz["w_dec_img"], npz["b_dec_img"],
            npz["phi_S"], k=1,
        )
        gre_t = compute_gre_top1(
            npz["w_enc_txt"], npz["b_enc_txt"],
            npz["w_dec_txt"], npz["b_dec_txt"],
            npz["psi_S"], k=1,
        )
        vals.append(0.5 * (gre_i + gre_t))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _load_topk(w_enc, b_enc, w_dec, b_dec, d: int, L: int, k: int,
               device: torch.device) -> TopKSAE:
    cfg = TopKSAEConfig(hidden_size=d, latent_size=L, k=k, normalize_decoder=True,
                        weight_tie=False)
    m = TopKSAE(cfg)
    with torch.no_grad():
        m.encoder.weight.copy_(torch.from_numpy(w_enc))
        m.encoder.bias.copy_(torch.from_numpy(b_enc))
        m.W_dec.copy_(torch.from_numpy(w_dec))
        m.b_dec.copy_(torch.from_numpy(b_dec))
    return m.to(device).eval()


def _dense_latents(sae: TopKSAE, x: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    out_chunks = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            hs = x[i:i + batch].to(device=device, dtype=torch.float32).unsqueeze(1)
            out = sae(hidden_states=hs, return_dense_latents=True)
            out_chunks.append(out.dense_latents.squeeze(1).cpu())
    return torch.cat(out_chunks, dim=0)


def compute_esim_posthoc(params_dirs: list[Path], cfg: ExperimentConfig,
                        device: torch.device) -> tuple[float, float]:
    """Offline Post-hoc Matching ESim for two_recon: mean cos(z_I, pi(z_T)) over eval pairs.

    Hungarian permutation pi derived from train-set latent correlation matrix.
    Searches the primary dir first, falls back to secondary. Cached at the
    primary dir to avoid recomputation when tweaking plot styling.
    """
    params_dir = params_dirs[0]
    cache_path = params_dir / ".esim_posthoc_cache.json"
    if cache_path.exists():
        try:
            c = json.load(open(cache_path))
            return float(c["mean"]), float(c["std"])
        except Exception:
            pass

    d = cfg.data
    t = cfg.training
    vals = []
    # two_recon baseline always lives in the primary dir
    for p in sorted(params_dir.glob("alpha*_seed*_two_recon.npz")):
        npz = np.load(p, allow_pickle=True)
        alpha = float(npz["alpha_target"])
        seed = int(npz["seed"])
        L = int(npz["latent_size_img"])
        sae_i = _load_topk(npz["w_enc_img"], npz["b_enc_img"],
                           npz["w_dec_img"], npz["b_dec_img"],
                           d.representation_dim, L, t.k, device)
        sae_t = _load_topk(npz["w_enc_txt"], npz["b_enc_txt"],
                           npz["w_dec_txt"], npz["b_dec_txt"],
                           d.representation_dim, L, t.k, device)

        builder = SyntheticPairedBuilder(
            n_shared=d.n_shared, n_image=d.n_image, n_text=d.n_text,
            representation_dim=d.representation_dim, sparsity=d.sparsity,
            beta=d.beta, obs_noise_std=d.obs_noise_std,
            max_interference=d.max_interference,
            alpha_target=alpha, num_train=d.num_train, num_eval=d.num_eval,
            seed=seed,
        )
        ds = builder.build()
        train_img = torch.from_numpy(ds["train"]["image_representation"])
        train_txt = torch.from_numpy(ds["train"]["text_representation"])
        eval_img = torch.from_numpy(ds["eval"]["image_representation"])
        eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

        # Train-set correlation -> alive-restricted Hungarian permutation.
        # Restricting to alive latents makes Hungarian O((alive)^3) instead of O(L^3).
        zi_tr = _dense_latents(sae_i, train_img, t.batch_size, device).numpy()
        zt_tr = _dense_latents(sae_t, train_txt, t.batch_size, device).numpy()
        fire_i = (zi_tr > 0).any(axis=0)
        fire_t = (zt_tr > 0).any(axis=0)
        alive_i = np.where(fire_i)[0]
        alive_t = np.where(fire_t)[0]
        mu_i = zi_tr[:, alive_i].mean(0); sd_i = zi_tr[:, alive_i].std(0) + 1e-8
        mu_t = zt_tr[:, alive_t].mean(0); sd_t = zt_tr[:, alive_t].std(0) + 1e-8
        Zi = (zi_tr[:, alive_i] - mu_i) / sd_i
        Zt = (zt_tr[:, alive_t] - mu_t) / sd_t
        C = (Zi.T @ Zt) / Zi.shape[0]
        row_ind_sub, col_ind_sub = linear_sum_assignment(-np.abs(C))

        # Build full-latent permutation with identity on unmatched slots.
        L_t = zi_tr.shape[1]
        perm = np.arange(L_t)
        perm[alive_i[row_ind_sub]] = alive_t[col_ind_sub]

        # Eval posthoc cosine (L-dim vectors, permuted z_T)
        zi_ev = _dense_latents(sae_i, eval_img, t.batch_size, device)
        zt_ev = _dense_latents(sae_t, eval_txt, t.batch_size, device)
        zt_ev_perm = zt_ev[:, torch.as_tensor(perm, dtype=torch.long)]
        cos = torch.nn.functional.cosine_similarity(zi_ev, zt_ev_perm, dim=1, eps=1e-12)
        vals.append(float(cos.mean().item()))
        print(f"  [alpha={alpha:.2f} seed={seed}] alive_i={len(alive_i)} alive_t={len(alive_t)} "
              f"cos_mean={vals[-1]:.4f}")
        del sae_i, sae_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def extract_from_json(jlist, metric: str, method_key: str, posthoc: bool = False):
    """Return (mean, std) across seeds for given (method, metric).

    Accepts a list of parsed result.json objects; uses the first one whose
    seeds contain ``method_key``. Lets us merge a primary sweep with an
    extension (e.g., adding multipliers) without re-running existing entries.
    """
    if metric == "RE":
        field = "avg_eval_loss"
    elif metric == "ESim":
        field = "pair_cos_mean"  # no posthoc version in result.json
    elif metric == "FSim":
        field = "probe_vec_cos_posthoc" if posthoc else "probe_vec_cos"
    else:
        raise ValueError(metric)
    if not isinstance(jlist, (list, tuple)):
        jlist = [jlist]
    for j in jlist:
        sr = j["sweep_results"][0]
        seeds = sr["per_seed"]
        vals = [s[method_key][field] for s in seeds if method_key in s]
        if vals:
            return float(np.mean(vals)), float(np.std(vals))
    return float("nan"), float("nan")


def _resolve_result(rp: str) -> str:
    if "*" in rp:
        matches = sorted(glob.glob(rp))
        if not matches:
            raise FileNotFoundError(f"no match: {rp}")
        return matches[-1]
    return rp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str,
                        default="outputs/theorem2_v2_lambda_sweep/runs/*/result.json",
                        help="Primary result.json (baselines + 1/4, 1, 4 multipliers).")
    parser.add_argument("--params-dir", type=str,
                        default="outputs/theorem2_v2_lambda_sweep/runs/run_20260418_112601/params",
                        help="Primary params dir.")
    parser.add_argument("--result-extra", type=str, default=None,
                        help="Extension result.json (e.g., 1/64 and 64 multipliers).")
    parser.add_argument("--params-dir-extra", type=str, default=None,
                        help="Extension params dir.")
    parser.add_argument("--out", type=str,
                        default="outputs/theorem2_v2_lambda_sweep/lambda_sweep.pdf")
    args = parser.parse_args()

    rp = _resolve_result(args.result)
    j_main = json.load(open(rp))
    j_list = [j_main]
    print(f"result: {rp}")
    if args.result_extra:
        rp_extra = _resolve_result(args.result_extra)
        j_list.append(json.load(open(rp_extra)))
        print(f"result-extra: {rp_extra}")

    params_dirs = [Path(args.params_dir)]
    if args.params_dir_extra:
        params_dirs.append(Path(args.params_dir_extra))
    print(f"params: {params_dirs}")

    # Load cfg (for regenerating data when we need posthoc ESim offline)
    cfg = ExperimentConfig.from_yaml("configs/synthetic/lambda_sweep.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Collect metric values ----
    # Shared SAE / Separated SAE: horizontal reference lines
    bases: dict[str, dict[str, tuple[float, float]]] = {"single_recon": {}, "two_recon": {}}
    for metric in PANELS:
        for method_key in ("single_recon", "two_recon"):
            if metric == "GRE":
                ms = compute_gre_for_method(params_dirs, method_key)
            else:
                ms = extract_from_json(j_list, metric, method_key, posthoc=False)
            bases[method_key][metric] = _maybe_invert(metric, ms)

    # Post-hoc Matching: solid-line curve with circle marker
    posthoc: dict[str, tuple[float, float]] = {}
    for metric in PANELS:
        if metric == "GRE":
            # decoder columns unchanged -> reuse
            posthoc[metric] = bases["two_recon"]["GRE"]
        elif metric == "FSim":
            ms = extract_from_json(j_list, "FSim", "two_recon", posthoc=True)
            posthoc[metric] = _maybe_invert("FSim", ms)
        elif metric == "ESim":
            print("[posthoc] computing ESim offline (Hungarian-permuted) ...")
            ms = compute_esim_posthoc(params_dirs, cfg, device)
            print(f"  ESim posthoc raw = {ms[0]:.4f} +- {ms[1]:.4f}")
            posthoc[metric] = _maybe_invert("ESim", ms)

    # Sweep curves (IA, GS) per lambda:
    sweep_data: dict[str, dict[str, list[tuple[float, float]]]] = {m: {} for m in METHODS}
    for mk, mstyle in METHODS.items():
        for metric in PANELS:
            row = []
            for mult in MULTIPLIERS:
                w = round(mstyle["base"] * mult, 9)
                method_id = f"{mk}_w{w}"
                if metric == "GRE":
                    ms = compute_gre_for_method(params_dirs, method_id)
                else:
                    ms = extract_from_json(j_list, metric, method_id, posthoc=False)
                row.append(_maybe_invert(metric, ms))
            sweep_data[mk][metric] = row

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.015))
    x = np.arange(len(MULTIPLIERS))
    all_handles, all_labels = [], []

    for i, (ax, metric) in enumerate(zip(axes, PANELS)):
        # Baselines as hlines (Shared SAE, Modality-Specific SAEs).
        # Modality-Specific gets the highest zorder so its dotted line is not
        # hidden by the sweep curves / Post-hoc line that overlap at the same y.
        for bk, bstyle in BASELINES.items():
            m, s = bases[bk][metric]
            if np.isnan(m):
                continue
            line_z = 6 if bk == "two_recon" else 1
            band_z = 5 if bk == "two_recon" else 0
            h = ax.axhline(m, color=bstyle["color"], ls=bstyle["ls"],
                           lw=1.2 if bk == "two_recon" else 1.0,
                           label=bstyle["label"], zorder=line_z)
            if not np.isnan(s):
                ax.axhspan(m - s, m + s, color=bstyle["color"], alpha=0.08, zorder=band_z)
            if i == 0 and bstyle["label"] not in all_labels:
                all_handles.append(h)
                all_labels.append(bstyle["label"])

        # Sweep curves (IA, GS)
        for mk, mstyle in METHODS.items():
            row = sweep_data[mk][metric]
            means = np.array([r[0] for r in row])
            stds  = np.array([r[1] for r in row])
            h, = ax.plot(x, means, f"-{mstyle['marker']}", color=mstyle["color"],
                         lw=1.2, ms=3, label=mstyle["label"], zorder=3)
            ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=mstyle["color"],
                        capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
            if i == 0 and mstyle["label"] not in all_labels:
                all_handles.append(h)
                all_labels.append(mstyle["label"])

        # Post-hoc Alignment (Ours): solid line, no marker (flat — lambda-independent)
        m_ph, s_ph = posthoc[metric]
        if not np.isnan(m_ph):
            means_ph = np.full(len(x), m_ph)
            stds_ph  = np.full(len(x), s_ph if not np.isnan(s_ph) else 0.0)
            h, = ax.plot(x, means_ph, "-",
                         color=POSTHOC_STYLE["color"], lw=1.4,
                         label=POSTHOC_STYLE["label"], zorder=3)
            ax.errorbar(x, means_ph, yerr=stds_ph, fmt="none",
                        ecolor=POSTHOC_STYLE["color"],
                        capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4)
            if i == 0 and POSTHOC_STYLE["label"] not in all_labels:
                all_handles.append(h)
                all_labels.append(POSTHOC_STYLE["label"])

        ax.set_xticks(x)
        ax.set_xticklabels(MULT_LABELS)
        ax.set_xlabel(r"$\lambda$ multiplier", fontsize=8, labelpad=1)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        if metric == "GRE":
            ax.set_yscale("log")
            ax.set_ylim(0.07, 1.1)
            ax.set_yticks([0.1, 0.2, 0.5, 1.0])
            plain = FuncFormatter(lambda y, _: f"{y:g}")
            ax.yaxis.set_major_formatter(plain)
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        elif metric == "ESim":
            # plotting 1 - ESim → "ESim distance"; lower = more aligned.
            ax.set_ylim(0.78, 1.02)
            ax.set_yticks([0.8, 0.9, 1.0])
        elif metric == "FSim":
            # plotting 1 - FSim → "FSim distance"; lower = more aligned.
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(alpha=0.15, linewidth=0.4, which="both")

    fig.legend(
        handles=all_handles, labels=all_labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(all_handles), frameon=False, fontsize=6,
        handlelength=1.5, columnspacing=1.0, handletextpad=0.3,
    )
    plt.subplots_adjust(top=0.72, bottom=0.22, left=0.07, right=0.99, wspace=0.40)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            sib = str(out_path).replace(".pdf", ext)
            fig.savefig(sib, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
            print(f"saved {sib}")
    plt.close()


if __name__ == "__main__":
    main()

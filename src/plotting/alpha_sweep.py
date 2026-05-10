#!/usr/bin/env python3
"""Alpha sweep plot: 3-panel (CR, RE, GRE) computed from saved npz params.

Recomputes CR/RE/GRE offline from decoder + encoder weights and regenerated
eval data, so the figure can be built from a ``runs/run_<ts>/params/`` dir
without relying on ``result.json``. GRE is the untied variant (encoder =
trained ``w_enc``, not ``V^T``).

Usage:
    python scripts/plot_alpha_sweep.py \\
        --params-dir outputs/theorem2_v2_1R2R_5seeds_coarse/runs/<run>/params \\
        --config     configs/synthetic/alpha_1R_2R_L8192_5seeds_coarse.yaml \\
        --out        outputs/theorem2_v2_1R2R_5seeds_coarse/alpha_sweep.pdf \\
        --alphas 0.0,0.2,0.4,0.6,0.8,1.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.configs.experiment import ExperimentConfig
from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.metrics.synthetic_eval import compute_gre_top1, compute_merged_fraction
from src.models.configuration_sae import TopKSAEConfig
from src.models.modeling_sae import TopKSAE


C_NO  = "#90be6d"   # Willow Green
C_YES = "#f9844a"   # Atomic Tangerine
LBL_NO  = "Shared SAE"
LBL_YES = "Modality-Specific SAEs"

METHODS = [("single_recon", C_NO, "-o", LBL_NO),
           ("two_recon",    C_YES, "-s", LBL_YES)]

PANELS = [
    ("CR",  (-0.05, 1.05)),
    ("RE",  (0.145, 0.20)),
    ("GRE", (0.05, 0.22)),
]

NPZ_RE = re.compile(r"^alpha([\d.]+)_seed(\d+)_(.+)\.npz$")


def _parse(name: str):
    m = NPZ_RE.match(name)
    if m is None:
        return None
    return round(float(m.group(1)), 2), int(m.group(2)), m.group(3)


def _load_topk(w_enc, b_enc, w_dec, b_dec, d: int, L: int,
               weight_tie: bool, k: int, device: torch.device) -> TopKSAE:
    cfg = TopKSAEConfig(
        hidden_size=d, latent_size=L,
        k=k, normalize_decoder=True, weight_tie=weight_tie,
    )
    model = TopKSAE(cfg)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.from_numpy(w_enc))
        model.encoder.bias.copy_(torch.from_numpy(b_enc))
        if not weight_tie:
            model.W_dec.copy_(torch.from_numpy(w_dec))
        model.b_dec.copy_(torch.from_numpy(b_dec))
    return model.to(device).eval()


def _eval_loss(sae: TopKSAE, data: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """Mean recon loss across all samples, single modality."""
    sae.eval()
    n = data.shape[0]
    sum_loss, steps = 0.0, 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = data[i:i + batch_size].to(device=device, dtype=torch.float32).unsqueeze(1)
            out = sae(hidden_states=x)
            sum_loss += float(out.recon_loss.item())
            steps += 1
    return sum_loss / max(steps, 1)


def compute_metrics_for_run(
    params_dir: Path,
    cfg: ExperimentConfig,
    alphas_target: list[float],
    device: torch.device,
):
    """Returns {metric: {method: (alphas, means, stds)}}."""
    d = cfg.data
    t = cfg.training

    # group by (alpha, seed, method)
    files: dict[tuple[float, int, str], Path] = {}
    for p in sorted(params_dir.glob("alpha*_seed*_*.npz")):
        parsed = _parse(p.name)
        if parsed is None:
            continue
        a, seed, method = parsed
        if a not in alphas_target:
            continue
        files[(a, seed, method)] = p
    print(f"found {len(files)} npz files")

    # metrics[(alpha, seed, method)] = dict(CR, RE, GRE)
    metrics: dict[tuple[float, int, str], dict[str, float]] = {}

    # Build eval data once per (alpha, seed) — shared across methods
    seeds_for_alpha: dict[float, set[int]] = {}
    for (a, s, _m) in files.keys():
        seeds_for_alpha.setdefault(a, set()).add(s)

    for alpha in sorted(seeds_for_alpha.keys()):
        for seed in sorted(seeds_for_alpha[alpha]):
            builder = SyntheticPairedBuilder(
                n_shared=d.n_shared, n_image=d.n_image, n_text=d.n_text,
                representation_dim=d.representation_dim, sparsity=d.sparsity,
                beta=d.beta, obs_noise_std=d.obs_noise_std,
                max_interference=d.max_interference,
                alpha_target=alpha, num_train=d.num_train, num_eval=d.num_eval,
                seed=seed,
            )
            ds = builder.build()
            eval_img = torch.from_numpy(ds["eval"]["image_representation"])
            eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

            for (a2, s2, method), npz_path in files.items():
                if a2 != alpha or s2 != seed:
                    continue
                npz = np.load(npz_path, allow_pickle=True)
                same_model = bool(int(npz["same_model_flag"]))
                L_i = int(npz["latent_size_img"])
                L_t = int(npz["latent_size_txt"])

                sae_i = _load_topk(
                    npz["w_enc_img"], npz["b_enc_img"],
                    npz["w_dec_img"], npz["b_dec_img"],
                    d.representation_dim, L_i, t.weight_tie, t.k, device,
                )
                if same_model:
                    sae_t = sae_i
                else:
                    sae_t = _load_topk(
                        npz["w_enc_txt"], npz["b_enc_txt"],
                        npz["w_dec_txt"], npz["b_dec_txt"],
                        d.representation_dim, L_t, t.weight_tie, t.k, device,
                    )

                re_img = _eval_loss(sae_i, eval_img, t.batch_size, device)
                re_txt = _eval_loss(sae_t, eval_txt, t.batch_size, device)
                re_avg = (re_img + re_txt) / 2

                w_dec_img = sae_i.W_dec.detach().cpu().numpy()
                w_dec_txt = sae_t.W_dec.detach().cpu().numpy()
                cr = compute_merged_fraction(
                    w_dec_img, w_dec_txt, npz["phi_S"], npz["psi_S"],
                )
                gre_i = compute_gre_top1(
                    npz["w_enc_img"], npz["b_enc_img"],
                    w_dec_img, sae_i.b_dec.detach().cpu().numpy(),
                    npz["phi_S"], k=1,
                )
                gre_t = compute_gre_top1(
                    npz["w_enc_txt"], npz["b_enc_txt"],
                    w_dec_txt, sae_t.b_dec.detach().cpu().numpy(),
                    npz["psi_S"], k=1,
                )

                metrics[(alpha, seed, method)] = {
                    "CR": float(cr),
                    "RE": float(re_avg),
                    "GRE": float(0.5 * (gre_i + gre_t)),
                }
                print(f"  α={alpha:.2f} seed={seed} {method}: "
                      f"CR={cr:.3f} RE={re_avg:.4f} GRE={(gre_i+gre_t)/2:.4f}")

                if sae_i is not sae_t:
                    del sae_i
                del sae_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Reshape to {metric: {method: (alphas, means, stds)}}
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for metric in ("CR", "RE", "GRE"):
        out[metric] = {}
        for method, *_ in METHODS:
            alphas_arr, means, stds = [], [], []
            for alpha in alphas_target:
                vals = [v[metric]
                        for (a, s, m), v in metrics.items()
                        if a == alpha and m == method]
                if not vals:
                    continue
                alphas_arr.append(alpha)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            out[metric][method] = (np.array(alphas_arr), np.array(means), np.array(stds))
    return out


def make_fig(series, alphas_target, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.015))
    handles = []

    for i, (ax, (metric, ylim)) in enumerate(zip(axes, PANELS)):
        for method, color, style, label in METHODS:
            alphas_arr, means, stds = series[metric][method]
            if means.size == 0:
                continue
            h, = ax.plot(alphas_arr, means, style, color=color, lw=1.2, ms=3, label=label, zorder=3)
            ax.errorbar(alphas_arr, means, yerr=stds, fmt="none", ecolor=color,
                        capsize=3, capthick=1.0, elinewidth=1.0, zorder=4)
            if i == 0:
                handles.append(h)

        ax.set_xlabel(r"$\cos(\phi_i, \psi_i)$", fontsize=8, labelpad=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(sorted(alphas_target))
        ax.tick_params(axis="both", labelsize=6.5, pad=1)

        if ylim is not None:
            ax.set_ylim(*ylim)
        elif metric in ("RE", "GRE"):
            all_vals = np.concatenate([series[metric][m][1] for m, *_ in METHODS
                                       if series[metric][m][1].size > 0])
            if all_vals.size:
                lo, hi = all_vals.min(), all_vals.max()
                margin = (hi - lo) * 0.12 + 1e-6
                ax.set_ylim(lo - margin, hi + margin)
        if metric == "RE":
            ax.set_yticks([0.15, 0.17, 0.19])
        elif metric == "GRE":
            ax.set_yticks([0.05, 0.10, 0.15, 0.20])
        ax.grid(alpha=0.15, linewidth=0.4, which="both")

    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2, frameon=False, fontsize=7,
        handlelength=1.5, columnspacing=1.5, handletextpad=0.4,
    )
    plt.subplots_adjust(top=0.72, bottom=0.22, left=0.05, right=0.99, wspace=0.30)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    print(f"saved {out_path}")
    if str(out_path).endswith(".pdf"):
        for ext in (".png", ".svg"):
            sib = str(out_path).replace(".pdf", ext)
            plt.savefig(sib, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
            print(f"saved {sib}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-dir", required=True,
                    help="Dir containing alpha*_seed*_*.npz files (merged across runs).")
    ap.add_argument("--config", required=True,
                    help="YAML used for the sweep (for data/training hyperparams).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    alphas_target = [round(float(x), 2) for x in args.alphas.split(",")]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    series = compute_metrics_for_run(Path(args.params_dir), cfg, alphas_target, device)
    make_fig(series, alphas_target, args.out)


if __name__ == "__main__":
    main()

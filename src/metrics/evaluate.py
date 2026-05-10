"""Top-level evaluation orchestrator and cross-seed aggregation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.alignment import compute_latent_correlation
from src.metrics.normalize import normalize_rows
from src.metrics.synthetic_eval import (
    compute_gre_top1,
    compute_joint_mgt,
    compute_mcc_and_uniqueness,
    compute_merged_fraction,
    compute_posthoc_joint_mgt,
    compute_recovery_metrics_multi_tau,
    cross_corr_mean_parts,
    eval_pair_latent_cosine,
    probe_gt_pair_activation,
)


# ------------------------------------------------------------------
# Helpers inlined from legacy files to avoid cross-dependency
# ------------------------------------------------------------------


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _eval_loss_single_modality(
    model, data: torch.Tensor, batch_size: int, device: torch.device,
) -> float:
    """Eval reconstruction loss for one modality."""
    model.eval()
    dev = _model_device(model)
    loader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    loss_sum, steps = 0.0, 0
    with torch.no_grad():
        for (batch,) in loader:
            hs = batch.to(device=dev, dtype=torch.float32).unsqueeze(1)
            out = model(hidden_states=hs)
            loss_sum += float(out.recon_loss.item())
            steps += 1
    return loss_sum / max(steps, 1)


def _gt_based_shared_alignment(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
) -> dict[str, float]:
    """Cross-SAE alignment via GT-matched decoder cosine."""
    n_S = phi_S.shape[1]
    img_norm = normalize_rows(w_dec_img.astype(np.float64))
    txt_norm = normalize_rows(w_dec_txt.astype(np.float64))
    phi_norm = normalize_rows(phi_S.T.astype(np.float64))
    psi_norm = normalize_rows(psi_S.T.astype(np.float64))

    sim_img = np.abs(img_norm @ phi_norm.T)
    sim_txt = np.abs(txt_norm @ psi_norm.T)
    best_img = sim_img.argmax(axis=0)
    best_txt = sim_txt.argmax(axis=0)

    cross_cos = np.array([
        np.abs(img_norm[best_img[i]] @ txt_norm[best_txt[i]])
        for i in range(n_S)
    ])
    return {
        "cross_cos_mean": float(cross_cos.mean()),
        "alpha_proxy_mean": float(cross_cos.mean()),
    }


# ------------------------------------------------------------------
# Main evaluate_method
# ------------------------------------------------------------------


_JOINT_TAUS: tuple[float, ...] = (0.9, 0.95, 0.99)


def evaluate_method(
    *,
    method: str,
    sae_i,
    sae_t,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    phi_I: Optional[np.ndarray],
    psi_T: Optional[np.ndarray],
    n_shared: int,
    m_S: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute the full metrics dict for a trained SAE pair.

    For shared-decoder methods pass the same model for *sae_i* and *sae_t*.
    """
    same_model = sae_i is sae_t

    img_eval = _eval_loss_single_modality(sae_i, eval_img, batch_size, device)
    txt_eval = _eval_loss_single_modality(sae_t, eval_txt, batch_size, device)

    w_dec_img = sae_i.W_dec.detach().cpu().numpy()
    w_dec_txt = sae_t.W_dec.detach().cpu().numpy()
    w_enc_img = sae_i.encoder.weight.detach().cpu().numpy()
    w_enc_txt = sae_t.encoder.weight.detach().cpu().numpy()
    b_enc_img = sae_i.encoder.bias.detach().cpu().numpy()
    b_enc_txt = sae_t.encoder.bias.detach().cpu().numpy()
    b_dec_img = sae_i.b_dec.detach().cpu().numpy()
    b_dec_txt = sae_t.b_dec.detach().cpu().numpy()

    img_mgt = compute_recovery_metrics_multi_tau(w_dec_img, phi_S, _JOINT_TAUS)
    txt_mgt = compute_recovery_metrics_multi_tau(w_dec_txt, psi_S, _JOINT_TAUS)

    img_gre = compute_gre_top1(w_enc_img, b_enc_img, w_dec_img, b_dec_img, phi_S, k=1)
    txt_gre = compute_gre_top1(w_enc_txt, b_enc_txt, w_dec_txt, b_dec_txt, psi_S, k=1)

    phi_full = np.concatenate([phi_I, phi_S], axis=1) if phi_I is not None and phi_I.size > 0 else phi_S
    psi_full = np.concatenate([psi_S, psi_T], axis=1) if psi_T is not None and psi_T.size > 0 else psi_S

    img_mgt_full = compute_recovery_metrics_multi_tau(w_dec_img, phi_full, _JOINT_TAUS)
    txt_mgt_full = compute_recovery_metrics_multi_tau(w_dec_txt, psi_full, _JOINT_TAUS)

    img_mcc = compute_mcc_and_uniqueness(w_dec_img, phi_S)
    txt_mcc = compute_mcc_and_uniqueness(w_dec_txt, psi_S)
    img_mcc_full = compute_mcc_and_uniqueness(w_dec_img, phi_full)
    txt_mcc_full = compute_mcc_and_uniqueness(w_dec_txt, psi_full)

    cross = _gt_based_shared_alignment(w_dec_img, w_dec_txt, phi_S, psi_S)

    m_S_for_diag = m_S if method == "ours" else n_shared
    top_diag, rest_diag = cross_corr_mean_parts(
        sae_i, sae_t, eval_img, eval_txt, batch_size, device, m_S_for_diag,
    )

    pair_cos = eval_pair_latent_cosine(sae_i, sae_t, eval_img, eval_txt, batch_size, device)

    joint_raw = compute_joint_mgt(w_dec_img, w_dec_txt, phi_S, psi_S, _JOINT_TAUS)
    joint_ph = compute_posthoc_joint_mgt(
        sae_i, sae_t, train_img, train_txt, phi_S, psi_S,
        same_model=same_model, batch_size=batch_size, device=device, taus=_JOINT_TAUS,
    )

    merged_frac = compute_merged_fraction(w_dec_img, w_dec_txt, phi_S, psi_S)

    probe = probe_gt_pair_activation(sae_i, sae_t, phi_S, psi_S, device)
    if same_model:
        probe_ph = probe
    else:
        # Canonical alive-restricted Hungarian — same procedure as ESim plot
        # path (`scripts/plot_lambda_sweep.py`). See src/metrics/canonical_perm.py
        # for why both call sites must share this single helper.
        from src.metrics.canonical_perm import compute_canonical_perm
        canon = compute_canonical_perm(
            sae_i, sae_t, train_img, train_txt, batch_size, device,
        )
        probe_ph = probe_gt_pair_activation(
            sae_i, sae_t, phi_S, psi_S, device,
            txt_permutation=canon["perm"],
        )

    return {
        "img_eval_loss": img_eval,
        "txt_eval_loss": txt_eval,
        "avg_eval_loss": (img_eval + txt_eval) / 2,
        "img_mip_shared": img_mgt["mip"],
        "txt_mip_shared": txt_mgt["mip"],
        "img_mgt_shared_tau0.9": img_mgt["mgt_tau0.9"],
        "img_mgt_shared_tau0.95": img_mgt["mgt_tau0.95"],
        "img_mgt_shared_tau0.99": img_mgt["mgt_tau0.99"],
        "txt_mgt_shared_tau0.9": txt_mgt["mgt_tau0.9"],
        "txt_mgt_shared_tau0.95": txt_mgt["mgt_tau0.95"],
        "txt_mgt_shared_tau0.99": txt_mgt["mgt_tau0.99"],
        "img_gre_shared": img_gre,
        "txt_gre_shared": txt_gre,
        "gre_avg_shared": (img_gre + txt_gre) / 2,
        "img_mip_full": img_mgt_full["mip"],
        "txt_mip_full": txt_mgt_full["mip"],
        "img_mgt_full_tau0.9": img_mgt_full["mgt_tau0.9"],
        "img_mgt_full_tau0.95": img_mgt_full["mgt_tau0.95"],
        "img_mgt_full_tau0.99": img_mgt_full["mgt_tau0.99"],
        "txt_mgt_full_tau0.9": txt_mgt_full["mgt_tau0.9"],
        "txt_mgt_full_tau0.95": txt_mgt_full["mgt_tau0.95"],
        "txt_mgt_full_tau0.99": txt_mgt_full["mgt_tau0.99"],
        "img_mcc_shared": img_mcc["mcc"],
        "txt_mcc_shared": txt_mcc["mcc"],
        "img_uniqueness_shared_raw": img_mcc["uniqueness_raw"],
        "txt_uniqueness_shared_raw": txt_mcc["uniqueness_raw"],
        "img_uniqueness_shared_norm": img_mcc["uniqueness_norm"],
        "txt_uniqueness_shared_norm": txt_mcc["uniqueness_norm"],
        "img_mcc_full": img_mcc_full["mcc"],
        "txt_mcc_full": txt_mcc_full["mcc"],
        "img_uniqueness_full_raw": img_mcc_full["uniqueness_raw"],
        "txt_uniqueness_full_raw": txt_mcc_full["uniqueness_raw"],
        "img_uniqueness_full_norm": img_mcc_full["uniqueness_norm"],
        "txt_uniqueness_full_norm": txt_mcc_full["uniqueness_norm"],
        "merged_fraction": merged_frac,
        "pair_cos_mean": pair_cos,
        "cross_cos_top_mS_mean": top_diag,
        "cross_cos_rest_mean": rest_diag,
        "cross_cos_gt_mean": cross["cross_cos_mean"],
        "alpha_proxy_mean": cross["alpha_proxy_mean"],
        "joint_match_mean_raw": joint_raw["joint_match_mean"],
        "joint_mgt_raw_tau0.9": joint_raw["joint_mgt_tau0.9"],
        "joint_mgt_raw_tau0.95": joint_raw["joint_mgt_tau0.95"],
        "joint_mgt_raw_tau0.99": joint_raw["joint_mgt_tau0.99"],
        "joint_match_mean_posthoc": joint_ph["joint_match_mean"],
        "joint_mgt_posthoc_tau0.9": joint_ph["joint_mgt_tau0.9"],
        "joint_mgt_posthoc_tau0.95": joint_ph["joint_mgt_tau0.95"],
        "joint_mgt_posthoc_tau0.99": joint_ph["joint_mgt_tau0.99"],
        "probe_top1_agree": probe["probe_top1_agree"],
        "probe_vec_cos": probe["probe_vec_cos"],
        "probe_topk_jaccard": probe["probe_topk_jaccard"],
        "probe_top1_both_valid": probe["probe_top1_both_valid"],
        "probe_top1_agree_posthoc": probe_ph["probe_top1_agree"],
        "probe_vec_cos_posthoc": probe_ph["probe_vec_cos"],
        "probe_topk_jaccard_posthoc": probe_ph["probe_topk_jaccard"],
        "probe_top1_both_valid_posthoc": probe_ph["probe_top1_both_valid"],
    }


# ------------------------------------------------------------------
# METRIC_SUFFIXES (keys that appear in evaluate_method output)
# ------------------------------------------------------------------

METRIC_SUFFIXES = [
    "img_eval_loss", "txt_eval_loss", "avg_eval_loss",
    "img_mip_shared", "txt_mip_shared",
    "img_mgt_shared_tau0.9", "img_mgt_shared_tau0.95", "img_mgt_shared_tau0.99",
    "txt_mgt_shared_tau0.9", "txt_mgt_shared_tau0.95", "txt_mgt_shared_tau0.99",
    "img_gre_shared", "txt_gre_shared", "gre_avg_shared",
    "img_mip_full", "txt_mip_full",
    "img_mgt_full_tau0.9", "img_mgt_full_tau0.95", "img_mgt_full_tau0.99",
    "txt_mgt_full_tau0.9", "txt_mgt_full_tau0.95", "txt_mgt_full_tau0.99",
    "img_mcc_shared", "txt_mcc_shared",
    "img_uniqueness_shared_raw", "txt_uniqueness_shared_raw",
    "img_uniqueness_shared_norm", "txt_uniqueness_shared_norm",
    "img_mcc_full", "txt_mcc_full",
    "img_uniqueness_full_raw", "txt_uniqueness_full_raw",
    "img_uniqueness_full_norm", "txt_uniqueness_full_norm",
    "merged_fraction",
    "pair_cos_mean",
    "cross_cos_top_mS_mean", "cross_cos_rest_mean",
    "cross_cos_gt_mean", "alpha_proxy_mean",
    "joint_match_mean_raw",
    "joint_mgt_raw_tau0.9", "joint_mgt_raw_tau0.95", "joint_mgt_raw_tau0.99",
    "joint_match_mean_posthoc",
    "joint_mgt_posthoc_tau0.9", "joint_mgt_posthoc_tau0.95", "joint_mgt_posthoc_tau0.99",
    "probe_top1_agree", "probe_vec_cos", "probe_topk_jaccard", "probe_top1_both_valid",
    "probe_top1_agree_posthoc", "probe_vec_cos_posthoc",
    "probe_topk_jaccard_posthoc", "probe_top1_both_valid_posthoc",
]


# ------------------------------------------------------------------
# Cross-seed aggregation
# ------------------------------------------------------------------


def aggregate(
    seed_results: list[dict[str, Any]],
    method_ids: list[str],
) -> dict[str, Any]:
    """Aggregate per-seed results into mean/std per method per metric."""
    agg: dict[str, Any] = {}
    for method_id in method_ids:
        for suf in METRIC_SUFFIXES:
            vals: list[float] = []
            for sr in seed_results:
                m_dict = sr.get(method_id)
                if m_dict is None:
                    continue
                v = m_dict.get(suf)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            key = f"{method_id}/{suf}"
            if vals:
                agg[f"{key}/mean"] = float(np.mean(vals))
                agg[f"{key}/std"] = float(np.std(vals))
            else:
                agg[f"{key}/mean"] = float("nan")
                agg[f"{key}/std"] = float("nan")
    return agg

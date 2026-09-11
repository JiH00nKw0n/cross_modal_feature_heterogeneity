"""Evaluation metrics for synthetic Theorem 2 experiments.

All functions are stateless; they take model weights (numpy) or SAE model
objects and return dicts of scalar metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.metrics.alignment import _paired_loader, compute_latent_correlation
from src.metrics.normalize import normalize_rows


# ------------------------------------------------------------------
# Decoder-side metrics
# ------------------------------------------------------------------


def compute_mcc_and_uniqueness(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
) -> dict[str, float]:
    """MCC (Hungarian 1-to-1) + Feature Uniqueness.

    Args:
        learned_vectors: ``(L, d)`` SAE decoder rows.
        gt_matrix: ``(d, N)`` GT atoms as columns.
    """
    if gt_matrix.shape[1] == 0 or learned_vectors.shape[0] == 0:
        return {"mcc": float("nan"), "uniqueness_raw": float("nan"), "uniqueness_norm": float("nan")}

    L = int(learned_vectors.shape[0])
    N = int(gt_matrix.shape[1])

    learned_norm = normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)

    row_ind, col_ind = linear_sum_assignment(-sim)
    matched = sim[row_ind, col_ind]
    mcc = float(matched.sum() / max(min(L, N), 1))

    best_gt_per_row = sim.argmax(axis=1)
    num_distinct = int(np.unique(best_gt_per_row).size)

    return {
        "mcc": mcc,
        "uniqueness_raw": float(num_distinct / max(L, 1)),
        "uniqueness_norm": float(num_distinct / max(min(L, N), 1)),
    }


def compute_merged_fraction(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    tau: float = 0.95,  # accepted for backward-compat; unused
) -> float:
    """Cross-Modal Feature Collapse Index (paper §, eq. CCI).

    Fraction of shared GT atoms for which the *same learned column index*
    is selected as the best match in both modalities — i.e. the SAE
    merged the two modality-specific representations into a single
    latent component.

    For each shared GT concept i ∈ [n_S]:
        best_i = argmax_j cos(V[:,j], phi_i)
        best_t = argmax_j cos(W[:,j], psi_i)
        merged_i = (best_i == best_t)
    Return mean(merged_i) ∈ [0, 1].

    For a single-decoder SAE (V == W byte-for-byte), this trips whenever
    the same column j best explains both phi_i and psi_i — which is the
    Theorem-2 collapse signature when phi_i ≠ psi_i and the dictionary is
    capacity-constrained. For two independently trained decoders the
    column indices live in disjoint orderings, so CCI ≈ 1/m (noise floor).
    """
    if phi_S.size == 0 or psi_S.size == 0:
        return float("nan")
    v = normalize_rows(w_dec_img.astype(np.float64))
    w = normalize_rows(w_dec_txt.astype(np.float64))
    phi = normalize_rows(phi_S.T.astype(np.float64))
    psi = normalize_rows(psi_S.T.astype(np.float64))
    best_i = (v @ phi.T).argmax(axis=0)        # (n_S,)  best slot for phi_i in V
    best_t = (w @ psi.T).argmax(axis=0)        # (n_S,)  best slot for psi_i in W
    return float((best_i == best_t).mean())


def compute_gre_top1(
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    w_dec: np.ndarray,
    b_dec: np.ndarray,
    gt_matrix: np.ndarray,
    k: int = 1,
) -> float:
    """Ground-truth Recovery Error through the trained (untied) SAE pipeline.

    For each GT atom ``g_i`` (column of ``gt_matrix``), run the full forward
    pass of the trained SAE (encoder → ReLU → top-k → decoder) and measure
    the reconstruction error against ``g_i``:

        z_i  = top_k( ReLU(w_enc @ g_i + b_enc) )
        ĝ_i  = w_dec.T @ z_i + b_dec
        GRE  = mean_i ||g_i - ĝ_i||_2^2

    Unlike the earlier tied formulation (which used ``w_dec`` both as encoder
    and decoder via ``V^T``), this variant uses the actually-trained encoder.
    This exposes the shrinkage induced by aux losses such as group-sparse,
    which bias ``w_enc`` outputs toward smaller magnitudes while leaving the
    decoder rows unit-norm.

    Args:
        w_enc: ``(L, d)`` encoder weights.
        b_enc: ``(L,)`` encoder bias.
        w_dec: ``(L, d)`` decoder rows.
        b_dec: ``(d,)`` decoder bias.
        gt_matrix: ``(d, n_gt)`` GT atoms as columns.
        k: Top-k sparsity at inference. Defaults to 1 (matches the paper's
            ``σ = top-1`` definition).

    Returns:
        Mean per-atom squared reconstruction error, or ``nan`` when empty.
    """
    if gt_matrix.shape[1] == 0 or w_enc.shape[0] == 0:
        return float("nan")
    W_e = w_enc.astype(np.float64)
    b_e = b_enc.astype(np.float64)
    W_d = w_dec.astype(np.float64)
    b_d = b_dec.astype(np.float64)
    G = gt_matrix.astype(np.float64)

    pre = W_e @ G + b_e[:, None]  # (L, n_gt)
    pre = np.maximum(pre, 0.0)     # ReLU (matches TopKSAE)
    # Top-k sparsify along slot axis
    kk = min(k, pre.shape[0])
    # argsort ascending; take last kk indices
    idx = np.argsort(pre, axis=0)[-kk:, :]  # (kk, n_gt)
    z = np.zeros_like(pre)
    vals = np.take_along_axis(pre, idx, axis=0)
    np.put_along_axis(z, idx, vals, axis=0)
    recon = W_d.T @ z + b_d[:, None]  # (d, n_gt)
    err = np.sum((G - recon) ** 2, axis=0)
    return float(err.mean())


def compute_recovery_metrics_multi_tau(
    learned_vectors: np.ndarray,
    gt_matrix: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Multi-threshold MGT + MIP in one pass.

    Args:
        learned_vectors: ``(L, d)`` decoder rows.
        gt_matrix: ``(d, n_gt)`` GT atoms as columns.
    """
    if gt_matrix.shape[1] == 0:
        out: dict[str, float] = {"mip": float("nan")}
        for tau in taus:
            out[f"mgt_tau{tau}"] = float("nan")
        return out
    learned_norm = normalize_rows(learned_vectors.astype(np.float64))
    gt_norm = normalize_rows(gt_matrix.T.astype(np.float64))
    sim = np.abs(learned_norm @ gt_norm.T)
    best = sim.max(axis=0)
    out = {"mip": float(best.mean())}
    for tau in taus:
        out[f"mgt_tau{tau}"] = float((best > tau).mean())
    return out


def compute_joint_mgt(
    w_dec_img: np.ndarray,
    w_dec_txt: np.ndarray,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint cross-modal recovery: ``max_i 0.5*(|cos(W_img[i], phi_g)| + |cos(W_txt[i], psi_g)|)``."""
    if (
        phi_S.shape[1] == 0 or psi_S.shape[1] == 0
        or w_dec_img.shape[0] == 0 or w_dec_txt.shape[0] == 0
    ):
        out: dict[str, float] = {"joint_match_mean": float("nan")}
        for tau in taus:
            out[f"joint_mgt_tau{tau}"] = float("nan")
        return out
    Wi = normalize_rows(w_dec_img.astype(np.float64))
    Wt = normalize_rows(w_dec_txt.astype(np.float64))
    Pi = normalize_rows(phi_S.T.astype(np.float64))
    Pt = normalize_rows(psi_S.T.astype(np.float64))
    joint = 0.5 * (np.abs(Wi @ Pi.T) + np.abs(Wt @ Pt.T))
    best = joint.max(axis=0)
    out = {"joint_match_mean": float(best.mean())}
    for tau in taus:
        out[f"joint_mgt_tau{tau}"] = float((best > tau).mean())
    return out


# ------------------------------------------------------------------
# Encoder-side metrics (require live SAE forward pass)
# ------------------------------------------------------------------


def _dense(out: object) -> torch.Tensor:
    """``(B, latent_size)`` dense latents from a TopKSAE output."""
    return out.dense_latents.squeeze(1)  # type: ignore[union-attr]


def probe_gt_pair_activation(
    sae_i,
    sae_t,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    device: torch.device,
    *,
    txt_permutation: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Encoder-side GT probe (Cross-Modal Alignment).

    Feeds pure GT atoms through each SAE and checks agreement.
    """
    n_S = phi_S.shape[1]
    if n_S == 0 or psi_S.shape[1] == 0:
        return {
            "probe_top1_agree": float("nan"),
            "probe_vec_cos": float("nan"),
            "probe_topk_jaccard": float("nan"),
            "probe_top1_both_valid": float("nan"),
        }
    dev = next(sae_i.parameters()).device
    x_img = torch.from_numpy(phi_S.T.astype(np.float32)).unsqueeze(1).to(dev)
    x_txt = torch.from_numpy(psi_S.T.astype(np.float32)).unsqueeze(1).to(dev)
    sae_i.eval()
    sae_t.eval()
    with torch.no_grad():
        out_i = sae_i(hidden_states=x_img, return_dense_latents=True)
        out_t = sae_t(hidden_states=x_txt, return_dense_latents=True)
        z_i = _dense(out_i)
        z_t = _dense(out_t)
    if txt_permutation is not None:
        perm = torch.as_tensor(txt_permutation, dtype=torch.long, device=z_t.device)
        z_t = z_t.index_select(dim=1, index=perm)

    top1_i = z_i.argmax(dim=1)
    top1_t = z_t.argmax(dim=1)
    agree = (top1_i == top1_t).float()
    val_i = z_i.gather(1, top1_i.unsqueeze(1)).squeeze(1)
    val_t = z_t.gather(1, top1_t.unsqueeze(1)).squeeze(1)
    both_valid = ((val_i > 0) & (val_t > 0)).float()
    valid_agree = (agree * both_valid).mean().item()

    vec_cos = F.cosine_similarity(z_i, z_t, dim=1, eps=1e-12).mean().item()

    k = int(sae_i.cfg.k)
    k_eff = min(k, z_i.shape[1], z_t.shape[1])
    topk_i = z_i.topk(k_eff, dim=1).indices.cpu().numpy()
    topk_t = z_t.topk(k_eff, dim=1).indices.cpu().numpy()
    jaccards = []
    for g in range(topk_i.shape[0]):
        a = set(topk_i[g].tolist())
        b = set(topk_t[g].tolist())
        union = len(a | b)
        jaccards.append(len(a & b) / union if union > 0 else 0.0)

    return {
        "probe_top1_agree": float(valid_agree),
        "probe_vec_cos": float(vec_cos),
        "probe_topk_jaccard": float(np.mean(jaccards)),
        "probe_top1_both_valid": float(both_valid.mean().item()),
    }


def compute_posthoc_joint_mgt(
    sae_i,
    sae_t,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    phi_S: np.ndarray,
    psi_S: np.ndarray,
    same_model: bool,
    batch_size: int,
    device: torch.device,
    taus: tuple[float, ...],
) -> dict[str, float]:
    """Joint MGT after Hungarian-optimal post-hoc reindexing."""
    w_img = sae_i.W_dec.detach().cpu().numpy()
    w_txt = sae_t.W_dec.detach().cpu().numpy()
    if same_model:
        return compute_joint_mgt(w_img, w_txt, phi_S, psi_S, taus)
    C = compute_latent_correlation(sae_i, sae_t, train_img, train_txt, batch_size, device)
    _, col_ind = linear_sum_assignment(-np.abs(C))
    w_txt_aligned = w_txt[col_ind]
    return compute_joint_mgt(w_img, w_txt_aligned, phi_S, psi_S, taus)


def eval_pair_latent_cosine(
    sae_i,
    sae_t,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Mean per-sample latent cosine between paired image/text codes."""
    sae_i.eval()
    sae_t.eval()
    device = next(sae_i.parameters()).device
    loader = _paired_loader(eval_img, eval_txt, batch_size, device)
    total = 0
    sum_cos = 0.0
    with torch.no_grad():
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = _dense(out_i)
            zt = _dense(out_t)
            cos = F.cosine_similarity(zi, zt, dim=1, eps=1e-12)
            sum_cos += float(cos.sum().item())
            total += cos.shape[0]
    return sum_cos / max(total, 1)


def cross_corr_mean_parts(
    sae_i,
    sae_t,
    eval_img: torch.Tensor,
    eval_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    m_S: int,
) -> tuple[float, float]:
    """Eval-set Pearson diagonal split into top-``m_S`` and rest."""
    C = compute_latent_correlation(sae_i, sae_t, eval_img, eval_txt, batch_size, device)
    diag = np.diag(C)
    n = diag.shape[0]
    m_S_eff = max(0, min(int(m_S), n))
    top = float(diag[:m_S_eff].mean()) if m_S_eff > 0 else float("nan")
    rest = float(diag[m_S_eff:].mean()) if m_S_eff < n else float("nan")
    return top, rest

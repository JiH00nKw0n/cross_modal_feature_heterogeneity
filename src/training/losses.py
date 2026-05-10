"""Auxiliary loss functions for SAE paired training.

Includes group-sparse and trace losses (previously in
``synthetic_theory_simplified.py``) and iso/alignment losses
(previously in ``synthetic_theorem2_method.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_sparse_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """L_{2,1} norm over paired sparse codes."""
    return torch.sqrt(z_img.pow(2) + z_txt.pow(2) + 1e-12).sum(dim=-1).mean()


def trace_alignment_loss(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """Trace-based alignment: ``-Tr(Z_img_norm @ Z_txt_norm^T) / B``."""
    b = z_img.shape[0]
    z_img_norm = F.normalize(z_img, p=2, dim=-1)
    z_txt_norm = F.normalize(z_txt, p=2, dim=-1)
    return -(z_img_norm * z_txt_norm).sum() / b


def iso_alignment_penalty(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """IsoEnergy alignment penalty (top-1-masked cosine).

    Masks each modality's top-1 latent index in *both* halves before
    computing cosine similarity.  Returns a scalar in ``[-1, 1]``.
    """
    n = z_img.shape[0]
    top_1_img = torch.topk(z_img, k=1, dim=1).indices.squeeze(1)
    top_1_txt = torch.topk(z_txt, k=1, dim=1).indices.squeeze(1)
    arange = torch.arange(n, device=z_img.device)

    mask_i = torch.ones_like(z_img)
    mask_t = torch.ones_like(z_txt)
    mask_i[arange, top_1_img] = 0.0
    mask_i[arange, top_1_txt] = 0.0
    mask_t[arange, top_1_img] = 0.0
    mask_t[arange, top_1_txt] = 0.0

    masked_i = z_img * mask_i
    masked_t = z_txt * mask_t
    cos = F.cosine_similarity(masked_i, masked_t, dim=1).mean()
    return -cos


def _batch_cross_corr(z_i: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Per-batch standardized cross-correlation matrix C[i, j] = corr(z_i[:,i], z_t[:,j])."""
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    n_batch = zi.shape[0]
    return (zi / si).t() @ (zt / st) / max(n_batch, 1)


def naive_diag_aux_loss_masked(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    frozen_mask: torch.Tensor,
) -> torch.Tensor:
    """Naive diagonal target gated by ``frozen_mask``.

    For frozen slot i: push C[i, i] -> 1.
    For unsettled slot i: push C[i, i] -> 0.
    Each group averaged over its own size, giving balanced 1:1 mixing.
    """
    C = _batch_cross_corr(z_i, z_t)
    diag = C.diagonal()
    mask_f = frozen_mask.to(diag.dtype)
    mask_u = 1.0 - mask_f
    n_f = mask_f.sum().clamp(min=1.0)
    n_u = mask_u.sum().clamp(min=1.0)
    shared = ((diag - 1.0) ** 2 * mask_f).sum() / n_f
    private = (diag ** 2 * mask_u).sum() / n_u
    return shared + private


def barlow_twins_aux_loss_masked(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    frozen_mask: torch.Tensor,
    lambda_off: float = 0.005,
) -> torch.Tensor:
    """Barlow-Twins variant of slot alignment, gated by ``frozen_mask``.

    For frozen slots in F:
      - on-diagonal of C pushed toward 1
      - off-diagonal entries in the F x F block pushed toward 0
    For unsettled slots:
      - on-diagonal pushed toward 0 (private term)
    """
    C = _batch_cross_corr(z_i, z_t)
    diag = C.diagonal()
    mask_f = frozen_mask.to(diag.dtype)
    mask_u = 1.0 - mask_f
    n_f = mask_f.sum().clamp(min=1.0)
    n_u = mask_u.sum().clamp(min=1.0)

    on_diag = ((diag - 1.0) ** 2 * mask_f).sum() / n_f
    private_diag = (diag ** 2 * mask_u).sum() / n_u

    # Off-diagonal in F x F block: use outer mask, subtract diagonal contribution.
    if frozen_mask.sum().item() > 1:
        outer_f = torch.outer(mask_f, mask_f)
        outer_f = outer_f - torch.diag(mask_f)  # zero out diagonal so we don't double count
        off_count = outer_f.sum().clamp(min=1.0)
        off_diag = ((C ** 2) * outer_f).sum() / off_count
    else:
        off_diag = C.new_tensor(0.0)

    return on_diag + lambda_off * off_diag + private_diag


def slot_infonce_loss(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    frozen_mask: torch.Tensor,
    log_tau: torch.Tensor,
    alive_mask_i: torch.Tensor | None = None,
    alive_mask_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric slot-level InfoNCE on the per-batch cross-correlation matrix.

    For each frozen slot i (``frozen_mask[i] = True``), positive = (i, i)
    diagonal. Negatives are restricted to slots in the ``alive_mask_*`` (dead
    slots are excluded from the softmax denominator to avoid a large constant
    inflation of the partition function by slots that can never be non-zero).

    Average over both directions (row / column) and normalize by the number of
    frozen slots so the loss scale is invariant to ``|F|``.

    Args:
        z_i, z_t: dense latent codes of shape ``[batch, n]``.
        frozen_mask: bool tensor of shape ``[n]``. Slots with True receive the
            InfoNCE positive pressure.
        log_tau: scalar parameter; effective temperature is ``exp(-log_tau)``.
        alive_mask_i: bool tensor ``[n]`` for image-side alive slots. Used as
            the valid negative pool in the column-wise InfoNCE. If ``None``,
            all slots are treated as alive.
        alive_mask_t: bool tensor ``[n]`` for text-side alive slots. Used as
            the valid negative pool in the row-wise InfoNCE. If ``None``, all
            slots are treated as alive.

    Returns:
        Scalar loss. If no slots are frozen, returns 0.
    """
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    zi_n = zi / si
    zt_n = zt / st
    n_batch = zi.shape[0]
    C = (zi_n.t() @ zt_n) / max(n_batch, 1)  # [n, n]

    inv_tau = torch.exp(log_tau).clamp(max=100.0)
    logits = C * inv_tau
    diag = logits.diagonal()

    # Restrict softmax denominator to alive slots (dead = -inf → contributes 0).
    neg_inf = torch.finfo(logits.dtype).min
    if alive_mask_t is not None:
        logits_row = torch.where(alive_mask_t.to(torch.bool)[None, :], logits, logits.new_full((), neg_inf))
    else:
        logits_row = logits
    if alive_mask_i is not None:
        logits_col = torch.where(alive_mask_i.to(torch.bool)[:, None], logits, logits.new_full((), neg_inf))
    else:
        logits_col = logits

    nll_row = -diag + torch.logsumexp(logits_row, dim=1)
    nll_col = -diag + torch.logsumexp(logits_col, dim=0)
    per_slot = 0.5 * (nll_row + nll_col)  # [n]

    mask_f = frozen_mask.to(per_slot.dtype)
    n_frozen = mask_f.sum().clamp(min=1.0)
    return (per_slot * mask_f).sum() / n_frozen


def auxiliary_alignment_loss(
    z_i: torch.Tensor,
    z_t: torch.Tensor,
    m_S: int,
    norm: str = "group",
) -> torch.Tensor:
    """Diag-only auxiliary alignment loss (Algorithm 1).

    For the first ``m_S`` latent dims push diagonal correlation to 1;
    for the rest push to 0.

    ``norm="group"``: each term averaged over its own group size.
    ``norm="global"``: both sums divided by the total latent count.
    """
    zi = z_i - z_i.mean(dim=0, keepdim=True)
    zt = z_t - z_t.mean(dim=0, keepdim=True)
    si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
    diag = (zi * zt).mean(dim=0) / (si * st)

    zero = diag.new_tensor(0.0)
    n = diag.shape[0]

    if norm == "group":
        shared_term = ((diag[:m_S] - 1.0) ** 2).mean() if m_S > 0 else zero
        private_term = (diag[m_S:] ** 2).mean() if m_S < n else zero
    elif norm == "global":
        shared_sum = ((diag[:m_S] - 1.0) ** 2).sum() if m_S > 0 else zero
        private_sum = (diag[m_S:] ** 2).sum() if m_S < n else zero
        shared_term = shared_sum / max(n, 1)
        private_term = private_sum / max(n, 1)
    else:
        raise ValueError(f"Unknown norm '{norm}'; expected 'group' or 'global'.")
    return shared_term + private_term

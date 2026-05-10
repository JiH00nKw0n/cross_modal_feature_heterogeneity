"""Canonical alive-restricted Hungarian permutation for synthetic SAE pairs.

Single source of truth for the post-hoc text→image slot map used in
`evaluate.py` (FSim) and `plot_lambda_sweep.py` (ESim). Both call sites
previously computed Hungarian inline with *different* procedures — full
L×L raw correlation in evaluate.py vs alive-restricted standardized
correlation in plot_lambda_sweep.py — producing two distinct perms from
the same checkpoint and metric values that did not correspond to the
same matching.

This module exposes one helper, `compute_canonical_perm`, that:
  1. Streams train embeddings through (sae_i, sae_t) → dense latents.
  2. Builds Pearson C on alive×alive subset (fire>0 across train batch).
  3. Hungarian on -|C|, then lifts back to L-dim with identity on unmatched.

Returns the perm, alive masks, and full-L C for downstream metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def _dense_latents(sae, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Encode (N, H) → dense (N, L) sparse latents on CPU."""
    sae = sae.to(device).eval()
    out_chunks = []
    for s in range(0, x.shape[0], batch_size):
        xb = x[s:s + batch_size].to(device)
        out = sae(hidden_states=xb.unsqueeze(1), return_dense_latents=True)
        out_chunks.append(out.dense_latents.squeeze(1).cpu())
    return torch.cat(out_chunks, dim=0)


def compute_canonical_perm(
    sae_i,
    sae_t,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    abs_cost: bool = True,
) -> dict:
    """Canonical alive-restricted Hungarian text→image perm.

    Args:
        sae_i, sae_t: trained TopK SAEs (image / text side).
        train_img, train_txt: (N, H) paired training embeddings.
        batch_size: encoding batch size.
        device: torch device.
        abs_cost: if True, use -|C| as cost (sign-invariant matching);
                  if False, use -C (sign-respecting). Legacy default = True.

    Returns:
        {
          "perm":    (L,) int64 — z_T_aligned[:, i] = z_T[:, perm[i]]
          "alive_i": (L,) bool  — image-side fire>0 mask on training batch
          "alive_t": (L,) bool  — text-side fire>0 mask on training batch
          "C":       (L, L) float64 — full Pearson correlation (alive entries
                     only are meaningful; dead rows/cols are 0/NaN).
        }
    """
    zi = _dense_latents(sae_i, train_img, batch_size, device).numpy()
    zt = _dense_latents(sae_t, train_txt, batch_size, device).numpy()
    L = zi.shape[1]

    fire_i = (zi > 0).any(axis=0)
    fire_t = (zt > 0).any(axis=0)
    alive_i_idx = np.where(fire_i)[0]
    alive_t_idx = np.where(fire_t)[0]

    perm = np.arange(L, dtype=np.int64)
    C_full = np.zeros((L, L), dtype=np.float64)

    if alive_i_idx.size > 0 and alive_t_idx.size > 0:
        zi_a = zi[:, alive_i_idx]
        zt_a = zt[:, alive_t_idx]
        mu_i = zi_a.mean(0); sd_i = zi_a.std(0) + 1e-8
        mu_t = zt_a.mean(0); sd_t = zt_a.std(0) + 1e-8
        Zi = (zi_a - mu_i) / sd_i
        Zt = (zt_a - mu_t) / sd_t
        C_alive = (Zi.T @ Zt) / Zi.shape[0]

        cost = -np.abs(C_alive) if abs_cost else -C_alive
        row_sub, col_sub = linear_sum_assignment(cost)
        perm[alive_i_idx[row_sub]] = alive_t_idx[col_sub]

        # Lift sub-correlation back to full L×L (only alive×alive entries).
        for i_sub, i_full in enumerate(alive_i_idx):
            C_full[i_full, alive_t_idx] = C_alive[i_sub]

    return {
        "perm": perm,
        "alive_i": fire_i,
        "alive_t": fire_t,
        "C": C_full,
    }


def save_perm(out_path: str, payload: dict) -> None:
    """Save canonical perm payload to .npz so all downstream metrics share it."""
    np.savez(
        out_path,
        perm=payload["perm"],
        alive_i=payload["alive_i"],
        alive_t=payload["alive_t"],
        C=payload["C"],
    )


def load_perm(path: str) -> dict:
    data = np.load(path)
    return {
        "perm": data["perm"],
        "alive_i": data["alive_i"] if "alive_i" in data.files else None,
        "alive_t": data["alive_t"] if "alive_t" in data.files else None,
        "C": data["C"] if "C" in data.files else None,
    }

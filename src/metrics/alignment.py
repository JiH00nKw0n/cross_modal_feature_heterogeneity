"""Latent correlation, firing rates, and alive-mask utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _paired_loader(
    img: torch.Tensor,
    txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
    *,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        TensorDataset(img, txt),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def compute_latent_correlation(
    sae_i,
    sae_t,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Signed Pearson correlation between paired dense latents.

    Computed over the full dataset in float64.  Returns ``(n, n)`` numpy array
    where *n* is the per-SAE latent size (both SAEs must match).
    """
    assert sae_i.latent_size == sae_t.latent_size
    sae_i.eval()
    sae_t.eval()
    # Use model device (Trainer may have moved it)
    device = next(sae_i.parameters()).device
    loader = _paired_loader(train_img, train_txt, batch_size, device)

    n = int(sae_i.latent_size)
    sum_i = np.zeros(n, dtype=np.float64)
    sum_t = np.zeros(n, dtype=np.float64)
    sum_ii = np.zeros(n, dtype=np.float64)
    sum_tt = np.zeros(n, dtype=np.float64)
    sum_it = np.zeros((n, n), dtype=np.float64)
    N = 0

    with torch.no_grad():
        for img_b, txt_b in loader:
            hs_i = img_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            hs_t = txt_b.to(device=device, dtype=torch.float32).unsqueeze(1)
            out_i = sae_i(hidden_states=hs_i, return_dense_latents=True)
            out_t = sae_t(hidden_states=hs_t, return_dense_latents=True)
            zi = out_i.dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            zt = out_t.dense_latents.squeeze(1).cpu().to(torch.float64).numpy()
            B = zi.shape[0]
            sum_i += zi.sum(axis=0)
            sum_t += zt.sum(axis=0)
            sum_ii += (zi * zi).sum(axis=0)
            sum_tt += (zt * zt).sum(axis=0)
            sum_it += zi.T @ zt
            N += B

    if N == 0:
        return np.zeros((n, n), dtype=np.float64)

    mean_i = sum_i / N
    mean_t = sum_t / N
    var_i = (sum_ii / N) - mean_i * mean_i
    var_t = (sum_tt / N) - mean_t * mean_t
    cov = (sum_it / N) - np.outer(mean_i, mean_t)
    denom = np.sqrt(np.clip(var_i[:, None] * var_t[None, :], 1e-16, None))
    corr = cov / denom
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_firing_rates(
    model,
    data: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Per-latent firing rate (fraction of samples where z_k > 0)."""
    model.eval()
    n = int(model.latent_size)
    counts = np.zeros(n, dtype=np.float64)
    N = 0
    loader = DataLoader(
        TensorDataset(data), batch_size=batch_size, shuffle=False, num_workers=0,
    )
    with torch.no_grad():
        for (batch,) in loader:
            hs = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
            out = model(hidden_states=hs, return_dense_latents=True)
            z = out.dense_latents.squeeze(1).cpu().numpy()
            counts += (z > 0).sum(axis=0)
            N += z.shape[0]
    return counts / max(N, 1)


def alive_mask(rates: np.ndarray, threshold: float = 0.001) -> np.ndarray:
    """Return indices where firing rate exceeds *threshold*."""
    return np.where(rates > threshold)[0]

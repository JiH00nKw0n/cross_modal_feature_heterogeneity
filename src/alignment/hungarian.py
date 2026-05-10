"""Post-hoc Hungarian alignment for `Ours` (paper §4.3).

Build ONE perm from the SAE's training distribution and reuse for every
downstream eval (cross-modal steering, MS, retrieval, zero-shot).

The previous codebase mistakenly recomputed perm per-eval-dataset, which
inflated downstream metrics. This module enforces single-perm.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.data.paired_dataset import PairedEmbeddingDataset
from src.models import TwoSidedTopKSAE

logger = logging.getLogger(__name__)


def _dense_latents(sae, x: torch.Tensor) -> torch.Tensor:
    out = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
    return out.dense_latents.squeeze(1)


def _correlation(z_i: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of (image latent, text latent)."""
    zi = z_i - z_i.mean(axis=0, keepdims=True)
    zt = z_t - z_t.mean(axis=0, keepdims=True)
    si = np.linalg.norm(zi, axis=0) + 1e-8
    st = np.linalg.norm(zt, axis=0) + 1e-8
    C = (zi / si).T @ (zt / st)
    return C


@torch.no_grad()
def build_perm(
    *,
    model: TwoSidedTopKSAE,
    cache_dir: str | Path,
    split: str = "train",
    max_samples: int = 50_000,
    batch_size: int = 1024,
    device: str = "cuda",
    alive_min_fires: int = 1,
) -> dict:
    """Compute alive-restricted Hungarian text→image slot perm from training distribution.

    Dead slots (fire count < alive_min_fires) on either side are penalized with
    BIG_NEG so they cannot steal alive-alive matches. Returns:
        perm:        (L_per_side,) int64 — text slot permutation
        C:           (L, L)        float32 — raw Pearson correlation
        alive_image: (L,)          bool
        alive_text:  (L,)          bool
        fire_count_image / _text:  (L,) int64
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    dataset = PairedEmbeddingDataset(cache_dir, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    z_imgs, z_txts = [], []
    n = 0
    for batch in loader:
        if n >= max_samples:
            break
        img = batch["image"].to(dev)
        txt = batch["text"].to(dev)
        z_i = _dense_latents(model.image_sae, img).cpu().numpy()
        z_t = _dense_latents(model.text_sae, txt).cpu().numpy()
        z_imgs.append(z_i)
        z_txts.append(z_t)
        n += img.shape[0]

    Z_img = np.concatenate(z_imgs, axis=0)[:max_samples]
    Z_txt = np.concatenate(z_txts, axis=0)[:max_samples]
    L = Z_img.shape[1]

    fire_i = (Z_img != 0).sum(axis=0).astype(np.int64)
    fire_t = (Z_txt != 0).sum(axis=0).astype(np.int64)
    alive_i = fire_i >= alive_min_fires
    alive_t = fire_t >= alive_min_fires
    logger.info(
        "[perm] alive image=%d/%d (%.1f%%), text=%d/%d (%.1f%%)",
        int(alive_i.sum()), L, 100 * alive_i.mean(),
        int(alive_t.sum()), L, 100 * alive_t.mean(),
    )

    C = _correlation(Z_img, Z_txt).astype(np.float32)
    BIG_NEG = -1e9
    C_masked = C.astype(np.float64).copy()
    C_masked[~alive_i, :] = BIG_NEG
    C_masked[:, ~alive_t] = BIG_NEG

    row, col = linear_sum_assignment(-C_masked)
    perm = np.zeros(L, dtype=np.int64)
    perm[row] = col
    logger.info("[perm] mean diag C (alive only) = %.4f",
                float(C[alive_i][:, alive_t].diagonal().mean() if alive_i.any() else 0.0))
    return {
        "perm": perm, "C": C,
        "alive_image": alive_i, "alive_text": alive_t,
        "fire_count_image": fire_i, "fire_count_text": fire_t,
    }


def save_perm(out_path: str | Path, payload: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)


def load_perm(path: str | Path) -> dict:
    data = np.load(path)
    return {"perm": data["perm"], "C": data["C"] if "C" in data.files else None}

"""Shared helpers for downstream evaluation (retrieval / zeroshot / steering / MS).

Supports four method labels — ``shared``, ``separated``, ``aux``, and ``ours``.

  * ``load_sae``         dispatches the right SAE class for the method.
  * ``load_pair_dataset`` returns a torch dataset wrapping a paired cache dir.
  * ``encode_image`` / ``encode_text`` extract dense latents per method;
    ``ours`` permutes the text side via the saved Hungarian perm so matched
    slots share the same column index as the image side.
  * ``normalize_rows`` is L2-normalize across the last axis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.datasets.cached_imagenet_pairs import CachedImageNetPairsDataset
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE

logger = logging.getLogger(__name__)

Method = Literal["shared", "separated", "aux", "ours"]
Dataset = Literal["coco", "imagenet", "cc3m"]


def load_sae(ckpt_dir: str | Path, method: Method):
    """Load a saved SAE checkpoint.

    For ``method ∈ {shared, aux}`` returns a ``TopKSAE``; for
    ``method ∈ {separated, ours}`` returns a ``TwoSidedTopKSAE`` (``ours``
    re-uses the ``separated`` checkpoint with a post-hoc permutation).
    """
    ckpt_dir = str(ckpt_dir)
    if method in ("shared", "aux"):
        return TopKSAE.from_pretrained(ckpt_dir)
    if method in ("separated", "ours"):
        return TwoSidedTopKSAE.from_pretrained(ckpt_dir)
    raise ValueError(f"unknown method {method!r}")


def load_pair_dataset(
    cache_dir: str | Path,
    dataset: Dataset,
    split: str,
    max_per_class: int | None = None,
):
    if dataset == "coco":
        return CachedClipPairsDataset(cache_dir, split=split, l2_normalize=True)
    if dataset == "imagenet":
        return CachedImageNetPairsDataset(
            cache_dir, split=split,
            max_per_class=max_per_class, l2_normalize=True,
        )
    if dataset == "cc3m":
        return CachedClipPairsDataset(cache_dir, split=split, l2_normalize=True)
    raise ValueError(f"unknown dataset {dataset!r}")


@torch.no_grad()
def _stream_dense_latents(
    sae: TopKSAE,
    embeds: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode a tensor of embeddings ``(N, H)`` to dense latents ``(N, L)``."""
    sae.eval()
    sae.to(device)
    out = torch.empty(embeds.shape[0], int(sae.latent_size), dtype=torch.float32)
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        z = sae(hidden_states=chunk, return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


@torch.no_grad()
def encode_image(
    model,
    x: torch.Tensor,
    method: Method,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode image embeddings ``(N, H)`` to dense latents ``(N, L)``.

    For ``shared`` / ``aux`` the single SAE is applied directly. For
    ``separated`` / ``ours`` the image side of the two-sided SAE is used.
    """
    if method in ("shared", "aux"):
        sae = model
    else:
        sae = model.image_sae
    return _stream_dense_latents(sae, x, batch_size, device)


@torch.no_grad()
def encode_text(
    model,
    y: torch.Tensor,
    method: Method,
    device: torch.device,
    perm: np.ndarray | None = None,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode text embeddings ``(N, H)`` to dense latents ``(N, L)``.

    For ``ours`` the text-side latent columns are reindexed by ``perm`` so
    matched slots share the same column index as the image side.
    """
    if method in ("shared", "aux"):
        sae = model
    else:
        sae = model.text_sae
    z_t = _stream_dense_latents(sae, y, batch_size, device)
    if method == "ours":
        if perm is None:
            raise ValueError("perm required for method='ours'")
        perm_t = torch.as_tensor(perm, dtype=torch.long)
        z_t = z_t[:, perm_t]
    return z_t


def normalize_rows(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


__all__ = [
    "load_sae",
    "load_pair_dataset",
    "encode_image",
    "encode_text",
    "normalize_rows",
]

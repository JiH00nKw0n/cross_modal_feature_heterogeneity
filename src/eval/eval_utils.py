"""Shared helpers for downstream evaluation (reconstruction, retrieval, zero-shot).

There is exactly one SAE implementation in this repository, `src.models`
(`topk_sae.py`), and this module loads checkpoints from it. Four method labels
are supported:

    shared      one TopKSAE applied to both modalities
    separated   TwoSidedTopKSAE, two disjoint dictionaries
    aux         one TopKSAE trained with an auxiliary alignment loss; the
                config names iso_align and group_sparse map here too
    ours        the `separated` checkpoint plus the saved Hungarian permutation

Embeddings arrive from the unified cache raw and are L2-normalized here with
`src.data.paired_dataset.l2_normalize_rows`, the one normalization rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.data.cache_io import load_imagenet_cache, load_stacked, split_rows
from src.data.paired_dataset import l2_normalize_rows
from src.models import TopKSAE, TwoSidedTopKSAE

logger = logging.getLogger(__name__)

Method = Literal["shared", "separated", "aux", "ours", "iso_align", "group_sparse"]

#: Method names that train a single dictionary shared by both modalities.
SINGLE_SAE_METHODS = ("shared", "aux", "iso_align", "group_sparse")


def load_sae(ckpt_dir: str | Path, method: str):
    """Load a saved checkpoint as the class the method implies.

    `shared`, `aux`, `iso_align` and `group_sparse` return a `TopKSAE`;
    `separated` and `ours` return a `TwoSidedTopKSAE`, since `ours` re-uses the
    `separated` checkpoint and only adds a post-hoc permutation.
    """
    ckpt_dir = str(ckpt_dir)
    if method in SINGLE_SAE_METHODS:
        return TopKSAE.from_pretrained(ckpt_dir)
    if method in ("separated", "ours"):
        return TwoSidedTopKSAE.from_pretrained(ckpt_dir)
    raise ValueError(f"unknown method {method!r}")


@dataclass
class PairedSplit:
    """One split of a paired cache, already L2-normalized.

    `image[i]` and `text[i]` are the two sides of pair `i`, and `keys[i]` is
    that pair's cache key ("{image_id}_{cap_idx}" for COCO).
    """

    image: torch.Tensor
    text: torch.Tensor
    keys: list[str]

    def __len__(self) -> int:
        return len(self.keys)


def load_paired_split(cache_dir: str | Path, split: str) -> PairedSplit:
    """Materialize one split of a paired cache as normalized tensors.

    Used by the evaluators, which run on the COCO test split (25,010 pairs) and
    similar sizes. The CC3M training split is never loaded this way; the panel
    builder streams it instead.
    """
    cache = load_stacked(cache_dir, mmap=True)
    rows = split_rows(cache, split)
    # np.array, not np.asarray: the tables are memory-mapped read-only, and a
    # tensor that shares that buffer is not writable.
    image = torch.from_numpy(np.array(cache["image"][rows], dtype=np.float32))
    text = torch.from_numpy(np.array(cache["text"][rows], dtype=np.float32))
    keys = [cache["keys"][int(r)] for r in rows]
    return PairedSplit(image=l2_normalize_rows(image), text=l2_normalize_rows(text), keys=keys)


@dataclass
class ImageNetSplit:
    """The ImageNet cache, already L2-normalized.

    `image` is the validation images, `labels` their class indices, and `text`
    the class x template grid where row `c * n_templates + t` is template `t`
    of class `c`.
    """

    image: torch.Tensor
    labels: np.ndarray
    text: torch.Tensor
    n_classes: int
    n_templates: int


def load_imagenet_split(cache_dir: str | Path) -> ImageNetSplit:
    """Materialize the ImageNet cache as normalized tensors."""
    cache = load_imagenet_cache(cache_dir, mmap=True)
    image = torch.from_numpy(np.array(cache["image"], dtype=np.float32))
    text = torch.from_numpy(np.array(cache["text"], dtype=np.float32))
    meta = cache["meta"]
    n_classes = int(meta.get("n_classes", 1000))
    n_templates = int(meta.get("n_templates", 80))
    if text.shape[0] != n_classes * n_templates:
        raise ValueError(
            f"text table has {text.shape[0]} rows, expected "
            f"{n_classes} classes x {n_templates} templates"
        )
    return ImageNetSplit(
        image=l2_normalize_rows(image),
        labels=np.asarray(cache["labels"], dtype=np.int64),
        text=l2_normalize_rows(text),
        n_classes=n_classes,
        n_templates=n_templates,
    )


@torch.no_grad()
def _stream_dense_latents(sae, embeds: torch.Tensor, batch_size: int,
                          device: torch.device) -> torch.Tensor:
    """Encode embeddings (N, dim) to dense latents (N, L)."""
    sae.eval()
    sae.to(device)
    out = torch.empty(embeds.shape[0], int(sae.latent_size), dtype=torch.float32)
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        z = sae(hidden_states=chunk, return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def image_sae_of(model, method: str):
    """The module that encodes the image side for this method."""
    return model if method in SINGLE_SAE_METHODS else model.image_sae


def text_sae_of(model, method: str):
    """The module that encodes the text side for this method."""
    return model if method in SINGLE_SAE_METHODS else model.text_sae


@torch.no_grad()
def encode_image(model, x: torch.Tensor, method: str, device: torch.device,
                 batch_size: int = 2048) -> torch.Tensor:
    """Encode image embeddings (N, dim) to dense latents (N, L)."""
    return _stream_dense_latents(image_sae_of(model, method), x, batch_size, device)


@torch.no_grad()
def encode_text(model, y: torch.Tensor, method: str, device: torch.device,
                perm: np.ndarray | None = None,
                batch_size: int = 2048) -> torch.Tensor:
    """Encode text embeddings (N, dim) to dense latents (N, L).

    For `ours` the text columns are reindexed by the saved permutation, so a
    matched text latent lands on the same column index as its image partner.
    For `separated` the columns are left raw, which is the point of that
    comparison: without the permutation the two dictionaries share no index.
    """
    z_t = _stream_dense_latents(text_sae_of(model, method), y, batch_size, device)
    if method == "ours":
        if perm is None:
            raise ValueError("perm required for method='ours'")
        z_t = z_t[:, torch.as_tensor(np.asarray(perm), dtype=torch.long)]
    return z_t


def normalize_rows(z: torch.Tensor) -> torch.Tensor:
    """L2-normalize latent rows before a cosine comparison.

    This one clamps the norm, unlike the embedding rule: a latent row can be
    all zero when every top-k slot lands on a dead column, and such a row must
    stay finite rather than become NaN.
    """
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def load_perm(perm_path: str | Path) -> np.ndarray:
    """The permutation stored in panel.npz (or the legacy perm.npz)."""
    data = np.load(str(perm_path))
    return np.asarray(data["perm"], dtype=np.int64)


def coco_image_id(key: str) -> str:
    """Image id of a COCO cache key "{image_id}_{cap_idx}"."""
    return key.rsplit("_", 1)[0]


__all__ = [
    "load_sae",
    "load_paired_split",
    "load_imagenet_split",
    "PairedSplit",
    "ImageNetSplit",
    "encode_image",
    "encode_text",
    "image_sae_of",
    "text_sae_of",
    "normalize_rows",
    "load_perm",
    "coco_image_id",
    "SINGLE_SAE_METHODS",
]

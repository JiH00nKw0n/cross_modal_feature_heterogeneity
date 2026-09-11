"""Paired (image, text) embeddings for SAE training, read from the unified cache.

The cache stores raw encoder output. This module owns the one normalization
rule the whole repository uses: every embedding is divided by its own L2 norm,
computed the way `transformers.models.clip.modeling_clip._get_vector_norm`
computes it, with NO epsilon added. Training, panel building and evaluation all
call `l2_normalize_rows`, so no two stages can disagree about what a row means.

Rows are addressed through the split's row indices into the stacked tables, and
the tables stay memory-mapped, so a 2.87M-row CC3M cache costs page cache
rather than resident memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cache_io import load_stacked, split_rows


def l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """Divide each row by its L2 norm, with no epsilon.

    This is the CLIP vector norm, `sqrt(sum(x**2))` along the last axis. The
    absence of an epsilon is deliberate and matches `CLIPModel.forward`; an
    all-zero row would produce NaN, which the encoders never emit.
    """
    norm = torch.pow(torch.sum(torch.pow(x, 2), dim=-1, keepdim=True), 0.5)
    return x / norm


def normalize_np(x: np.ndarray) -> np.ndarray:
    """`l2_normalize_rows` for a numpy block, used by the streaming readers."""
    arr = np.asarray(x, dtype=np.float32)
    norm = np.sqrt(np.sum(arr * arr, axis=-1, keepdims=True))
    return arr / norm


class PairedEmbeddingDataset(Dataset):
    """One split of a paired cache, L2-normalized at access time.

    Item `i` is `{"image": Tensor(dim), "text": Tensor(dim)}` for the split's
    i-th pair. The underlying tables are memory-mapped, so only the requested
    rows are materialized.
    """

    def __init__(self, cache_dir: str | Path, split: str = "train", mmap: bool = True):
        cache = load_stacked(cache_dir, mmap=mmap)
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.rows = split_rows(cache, split)
        self._image = cache["image"]
        self._text = cache["text"]
        self.keys = [cache["keys"][int(r)] for r in self.rows]
        self.dim = int(self._image.shape[1])

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[idx])
        img = torch.from_numpy(np.array(self._image[row], dtype=np.float32))
        txt = torch.from_numpy(np.array(self._text[row], dtype=np.float32))
        return {"image": l2_normalize_rows(img), "text": l2_normalize_rows(txt)}


__all__ = ["PairedEmbeddingDataset", "l2_normalize_rows", "normalize_np"]

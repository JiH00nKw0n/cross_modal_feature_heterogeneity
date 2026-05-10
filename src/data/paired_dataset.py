"""mmap-backed paired (image, text) dataset for SAE training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cache_io import load_stacked


class PairedEmbeddingDataset(Dataset):
    """In-memory (or mmap) paired embeddings keyed by `split`."""

    def __init__(self, cache_dir: str | Path, split: str = "train", mmap: bool = True):
        cache = load_stacked(cache_dir, mmap=mmap)
        keys_in_split = cache["splits"].get(split, cache["keys"])
        # build index: key -> row
        key_to_row = {k: i for i, k in enumerate(cache["keys"])}
        rows = np.array([key_to_row[k] for k in keys_in_split if k in key_to_row], dtype=np.int64)
        self.image = np.asarray(cache["image"])[rows]
        self.text = np.asarray(cache["text"])[rows]
        self.keys = [keys_in_split[i] for i in range(len(rows))]
        self.dim = int(self.image.shape[1])

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(np.asarray(self.image[idx], dtype=np.float32)),
            "text": torch.from_numpy(np.asarray(self.text[idx], dtype=np.float32)),
        }

"""Torch Dataset wrapper for synthetic paired numpy arrays."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticPairedDataset(Dataset):
    """Wraps numpy image/text arrays as ``{"image_embeds", "text_embeds"}``."""

    def __init__(self, image_repr: np.ndarray, text_repr: np.ndarray) -> None:
        self.image = torch.from_numpy(image_repr).float()
        self.text = torch.from_numpy(text_repr).float()

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image_embeds": self.image[idx], "text_embeds": self.text[idx]}

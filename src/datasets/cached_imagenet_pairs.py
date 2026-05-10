"""Dataset for pre-cached ImageNet-1K CLIP embedding pairs.

Unlike CachedClipPairsDataset (COCO), pairs are (image_idx, class_idx)
and text embeddings are looked up by class × template. Each __getitem__
randomly selects a template, providing stochastic augmentation.

Loads embeddings saved by `scripts/real_alpha/extract_imagenet_cache.py`.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

Split = Literal["train", "val"]


def _clip_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True), 0.5)


class CachedImageNetPairsDataset(Dataset):
    """ImageNet-1K paired embedding dataset with per-class text templates.

    Args:
        cache_dir: Path to cache directory with image_embeddings.pt,
            text_embeddings.pt, splits.json.
        split: "train" or "val".
        max_per_class: If set, subsample to at most this many images per class.
            All classes are capped at min(max_per_class, smallest_class_count)
            to guarantee equal representation.
        n_templates: Number of text templates per class (default 80).
        l2_normalize: Apply L2 normalization matching CLIP inference.
        dtype: Tensor dtype for embeddings.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: Split,
        max_per_class: int | None = None,
        n_templates: int = 80,
        l2_normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        cache_dir = Path(cache_dir)

        with open(cache_dir / "splits.json") as f:
            splits = json.load(f)
        if split not in splits:
            raise KeyError(f"split '{split}' missing in {cache_dir}/splits.json")

        pairs = [tuple(p) for p in splits[split]]

        if max_per_class is not None:
            by_class: dict[int, list] = defaultdict(list)
            for pair in pairs:
                by_class[pair[1]].append(pair)
            min_count = min(len(v) for v in by_class.values())
            cap = min(min_count, max_per_class)
            if cap < max_per_class:
                logger.warning(
                    "max_per_class=%d but smallest class has %d images; "
                    "capping all classes at %d for balance",
                    max_per_class, min_count, cap,
                )
            pairs = []
            for class_idx in sorted(by_class):
                pairs.extend(by_class[class_idx][:cap])

        self.pairs = pairs
        self.n_templates = n_templates

        image_dict = torch.load(cache_dir / "image_embeddings.pt", map_location="cpu")
        text_dict = torch.load(cache_dir / "text_embeddings.pt", map_location="cpu")
        self._image_dict = {int(k): v.to(dtype) for k, v in image_dict.items()}
        self._text_dict = {str(k): v.to(dtype) for k, v in text_dict.items()}

        if l2_normalize:
            for k, v in self._image_dict.items():
                self._image_dict[k] = v / _clip_vector_norm(v)
            for k, v in self._text_dict.items():
                self._text_dict[k] = v / _clip_vector_norm(v)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_id, class_idx = self.pairs[idx]
        img = self._image_dict[int(image_id)]
        template_idx = random.randint(0, self.n_templates - 1)
        txt = self._text_dict[f"{int(class_idx)}_{template_idx}"]
        return {"image_embeds": img, "text_embeds": txt}


__all__ = ["CachedImageNetPairsDataset"]

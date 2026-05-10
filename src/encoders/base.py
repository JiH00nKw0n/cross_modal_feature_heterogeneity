"""Encoder protocol for VLM image/text embedders.

Backends differ (transformers, open_clip) but expose the same minimal API:
    encode_image(pil_images) -> (B, dim) tensor on `device`
    encode_text(strings)     -> (B, dim) tensor on `device`
    dim                      -> embedding dim (int)

Embeddings are L2-normalized in CLIP-style; use raw if `.normalize=False`.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
from PIL import Image


class Encoder(Protocol):
    dim: int
    device: torch.device

    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor: ...
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor: ...

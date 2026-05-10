"""open_clip backend (OpenCLIP, MetaCLIP, MobileCLIP, …).

Model identified by `(arch, pretrained)` pair, e.g. ("ViT-B-32", "metaclip_400m").
"""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image

from src.utils.config import ModelConfig


class OpenCLIPEncoder:
    def __init__(self, cfg: ModelConfig, device: torch.device | str = "cuda"):
        import open_clip  # local — heavy
        self.cfg = cfg
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.arch, pretrained=cfg.pretrained or None, device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.arch)
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=self.device)
            self.dim = int(self.model.encode_image(dummy).shape[-1])

    @torch.no_grad()
    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(p) for p in images]).to(self.device)
        return self.model.encode_image(batch).float().cpu()

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(texts)).to(self.device)
        return self.model.encode_text(tokens).float().cpu()

"""HuggingFace transformers CLIP / SigLIP encoder."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.utils.config import ModelConfig


class TransformersEncoder:
    def __init__(self, cfg: ModelConfig, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(cfg.hf_id).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(cfg.hf_id)
        mt = getattr(self.model.config, "model_type", "")
        self.is_siglip = mt == "siglip" or cfg.is_siglip
        if self.is_siglip:
            self.dim = self.model.config.vision_config.hidden_size
        else:
            self.dim = self.model.config.projection_dim
        self.text_max_length = 64 if self.is_siglip else cfg.text_max_length

    @torch.no_grad()
    def encode_image(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        v_out = self.model.vision_model(pixel_values=inputs["pixel_values"].to(self.device))
        if self.is_siglip:
            return v_out.pooler_output.float().cpu()
        return self.model.visual_projection(v_out.pooler_output).float().cpu()

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.processor(
            text=list(texts), return_tensors="pt",
            padding="max_length" if self.is_siglip else True,
            truncation=True, max_length=self.text_max_length,
        )
        input_ids = tok["input_ids"].to(self.device)
        attn = tok.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        t_out = self.model.text_model(input_ids=input_ids, attention_mask=attn)
        if self.is_siglip:
            return t_out.pooler_output.float().cpu()
        return self.model.text_projection(t_out.pooler_output).float().cpu()

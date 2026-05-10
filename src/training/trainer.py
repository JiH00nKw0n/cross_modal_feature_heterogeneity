"""SAE trainer covering the 5 paper methods.

method.name dispatches loss assembly:

  shared        single TopKSAE; both modalities concat into a single batch.
  separated     TwoSidedTopKSAE; per-modality recon only.
  iso_align     single TopKSAE; recon + β·iso_alignment_penalty(z_img, z_txt).
  group_sparse  single TopKSAE; recon + λ·group_sparse_loss(z_img, z_txt).
  ours          identical training to `separated`; correspondence emerges
                from a post-hoc Hungarian perm built at eval time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.paired_dataset import PairedEmbeddingDataset
from src.models import (
    TopKSAE,
    TopKSAEConfig,
    TwoSidedTopKSAE,
    TwoSidedTopKSAEConfig,
)
from src.training.losses import group_sparse_loss, iso_alignment_penalty
from src.utils.config import MethodConfig, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    model: nn.Module
    method: str
    history: list[dict[str, float]]
    save_dir: Path


def _build_model(method: str, hidden_size: int, training: TrainingConfig) -> nn.Module:
    if method in ("shared", "iso_align", "group_sparse"):
        cfg = TopKSAEConfig(
            hidden_size=hidden_size,
            latent_size=training.latent_size,
            k=training.k,
            normalize_decoder=True,
        )
        return TopKSAE(cfg)
    if method in ("separated", "ours"):
        cfg = TwoSidedTopKSAEConfig(
            hidden_size=hidden_size,
            latent_size=training.latent_size,
            k=training.k,
            normalize_decoder=True,
        )
        return TwoSidedTopKSAE(cfg)
    raise ValueError(f"Unknown method {method!r}")


def _step_loss(model: nn.Module, batch: dict, method: MethodConfig) -> dict:
    img = batch["image"]    # (B, dim)
    txt = batch["text"]     # (B, dim)

    if method.name == "shared":
        out_i = model(hidden_states=img.unsqueeze(1))
        out_t = model(hidden_states=txt.unsqueeze(1))
        recon = (out_i.recon_loss + out_t.recon_loss) / 2
        return {"loss": recon, "recon": recon}

    if method.name == "iso_align":
        out_i = model(hidden_states=img.unsqueeze(1), return_dense_latents=True)
        out_t = model(hidden_states=txt.unsqueeze(1), return_dense_latents=True)
        z_i = out_i.dense_latents.squeeze(1)
        z_t = out_t.dense_latents.squeeze(1)
        recon = (out_i.recon_loss + out_t.recon_loss) / 2
        aux = iso_alignment_penalty(z_i, z_t)
        return {"loss": recon + method.aux_weight * aux, "recon": recon, "aux": aux}

    if method.name == "group_sparse":
        out_i = model(hidden_states=img.unsqueeze(1), return_dense_latents=True)
        out_t = model(hidden_states=txt.unsqueeze(1), return_dense_latents=True)
        z_i = out_i.dense_latents.squeeze(1)
        z_t = out_t.dense_latents.squeeze(1)
        recon = (out_i.recon_loss + out_t.recon_loss) / 2
        aux = group_sparse_loss(z_i, z_t)
        return {"loss": recon + method.aux_weight * aux, "recon": recon, "aux": aux}

    if method.name in ("separated", "ours"):
        out = model(image_embeds=img, text_embeds=txt)
        return {"loss": out.loss, "recon": out.recon_loss}

    raise ValueError(f"Unknown method {method.name!r}")


def train_method(
    *,
    method: MethodConfig,
    training: TrainingConfig,
    cache_dir: str | Path,
    hidden_size: int,
    save_dir: str | Path,
) -> TrainingArtifacts:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_dir = save_dir / "final"

    if (final_dir / "config.json").exists():
        logger.info("[train] skip %s — ckpt exists at %s", method.name, final_dir)
        model = _build_model(method.name, hidden_size, training)
        state_path = final_dir / "model.safetensors"
        if not state_path.exists():
            state_path = final_dir / "pytorch_model.bin"
        state = torch.load(state_path, map_location="cpu") if state_path.suffix == ".bin" else None
        if state is not None:
            model.load_state_dict(state)
        else:
            from safetensors.torch import load_model
            load_model(model, str(state_path))
        return TrainingArtifacts(model=model, method=method.name, history=[], save_dir=final_dir)

    torch.manual_seed(training.seed)
    device = torch.device(training.device if torch.cuda.is_available() else "cpu")

    dataset = PairedEmbeddingDataset(cache_dir, split="train")
    loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)

    model = _build_model(method.name, hidden_size, training).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=training.lr, weight_decay=training.weight_decay, betas=(0.9, 0.999),
    )
    total_steps = training.num_epochs * max(1, len(loader))
    warmup_steps = max(1, int(training.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history = []
    step = 0
    for epoch in range(training.num_epochs):
        ep_losses: list[float] = []
        pbar = tqdm(loader, desc=f"{method.name} ep{epoch}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            losses = _step_loss(model, batch, method)
            loss = losses["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
            opt.step()
            sched.step()
            with torch.no_grad():
                if hasattr(model, "set_decoder_norm_to_unit_norm"):
                    model.set_decoder_norm_to_unit_norm()
            ep_losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = sum(ep_losses) / max(1, len(ep_losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        logger.info("[train] %s ep%d loss=%.4f", method.name, epoch, mean_loss)

    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    return TrainingArtifacts(model=model, method=method.name, history=history, save_dir=final_dir)

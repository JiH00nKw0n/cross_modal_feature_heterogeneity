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
import multiprocessing
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


#: Weight file names `save_pretrained` may leave behind, newest format first.
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

#: Worker processes to use when the start method makes them free.
_FORK_DATALOADER_WORKERS = 2

#: Rows below which worker processes cost more to start than they save.
_WORKER_MIN_ROWS = 100_000


def _find_weights(final_dir: Path) -> Path | None:
    """The weight file of a saved checkpoint, or None when none was written.

    `save_pretrained` writes config.json before the weights, so a directory
    holding only config.json is an interrupted save, not a finished checkpoint,
    and the caller has to retrain rather than skip.
    """
    for name in _WEIGHT_FILES:
        path = final_dir / name
        if path.exists():
            return path
    return None


def _dataloader_workers(n_rows: int) -> int:
    """Worker processes for the embedding loader, decided by the start method.

    Under fork, which is what Linux uses, a worker inherits the parent's address
    space: the memory-mapped embedding tables stay mapped and nothing is copied,
    so workers are free parallelism and reading rows stops competing with the
    training step. Under spawn, which is what macOS and Windows use, the dataset
    is pickled into each worker, and pickling a numpy memmap materializes the
    whole table, about 6 GB per worker for CC3M, so the loader stays
    single-process there.

    A split smaller than `_WORKER_MIN_ROWS` also stays single-process, because
    starting workers then costs more than the reading they take over.
    """
    if n_rows < _WORKER_MIN_ROWS:
        return 0
    try:
        method = multiprocessing.get_start_method(allow_none=False)
    except Exception:  # pragma: no cover - only on exotic platforms
        return 0
    return _FORK_DATALOADER_WORKERS if method == "fork" else 0


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


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight-decay grouping identical to HF Trainer, which the paper runs used.

    HF Trainer's ``get_decay_parameter_names`` excludes every parameter whose
    name contains ``bias`` (and norm layers, which the SAE has none of) from
    weight decay. So ``encoder.bias`` gets no decay while ``encoder.weight``,
    ``W_dec`` and ``b_dec`` do; ``b_dec`` is not matched by the ``bias``
    pattern and therefore is decayed, exactly as in the paper runs.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if "bias" in name else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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
    seed: int | None = None,
    split: str = "train",
) -> TrainingArtifacts:
    """Train one method and save the checkpoint under `save_dir/final`.

    `seed` overrides `training.seed`, which is how the pipeline trains the same
    method once per entry of `training.seeds`. Idempotent: an existing
    `save_dir/final` is loaded and returned rather than retrained.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_dir = save_dir / "final"

    # A checkpoint counts as finished only when its weights are on disk.
    # `save_pretrained` writes config.json first, so a directory holding just
    # that one file is a save that was interrupted; skipping on it would send
    # the reload path looking for a weight file that was never written.
    state_path = _find_weights(final_dir) if (final_dir / "config.json").exists() else None
    if state_path is not None:
        logger.info("[train] skip %s - checkpoint exists at %s", method.name, final_dir)
        model = _build_model(method.name, hidden_size, training)
        if state_path.suffix == ".bin":
            model.load_state_dict(torch.load(state_path, map_location="cpu"))
        else:
            from safetensors.torch import load_model
            load_model(model, str(state_path))
        return TrainingArtifacts(model=model, method=method.name, history=[], save_dir=final_dir)
    if (final_dir / "config.json").exists():
        logger.warning(
            "[train] %s holds config.json but none of %s - the previous save was "
            "interrupted, retraining", final_dir, ", ".join(_WEIGHT_FILES),
        )

    effective_seed = int(training.seed if seed is None else seed)
    torch.manual_seed(effective_seed)
    device = torch.device(training.device
                          if (training.device == "cpu" or torch.cuda.is_available())
                          else "cpu")

    dataset = PairedEmbeddingDataset(cache_dir, split=split)
    # drop_last=False matches the paper runs, which went through HF Trainer
    # with its default dataloader_drop_last=False (the final partial batch of
    # every epoch is trained on, not discarded).
    # The worker count depends on the process start method; see
    # `_dataloader_workers` for why fork gets workers and spawn does not.
    workers = _dataloader_workers(len(dataset))
    loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True,
                        num_workers=workers, drop_last=False,
                        persistent_workers=workers > 0)

    model = _build_model(method.name, hidden_size, training).to(device)
    opt = torch.optim.AdamW(
        _param_groups(model, training.weight_decay),
        lr=training.lr, betas=(0.9, 0.999),
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

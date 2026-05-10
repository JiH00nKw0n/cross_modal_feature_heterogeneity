"""Multi-VLM density pipeline (paper §3.3, Fig multi_density).

For each model in `cfg.models`:
  1. Extract paired COCO embeddings.
  2. Train modality-specific TwoSidedTopKSAE (no method sweep — `separated` only).
  3. Compute Hungarian-aligned decoder cosine density.
  4. Aggregate into a single density figure across all models.

Output: <output_root>/<model_key>/{ckpt, perm.npz} per model;
        <output_root>/multi_density.{svg,pdf,png} for the joint figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.alignment import build_perm, save_perm
from src.data.extract import extract_cache
from src.training.trainer import train_method
from src.utils.config import CacheConfig, Config, MethodConfig

logger = logging.getLogger(__name__)


def _density_decoder_cosine(image_W: np.ndarray, text_W: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """cos(W_img[i], W_txt[perm[i]]) for each i, alive-mask filtered."""
    img = image_W / (np.linalg.norm(image_W, axis=1, keepdims=True) + 1e-8)
    txt = text_W / (np.linalg.norm(text_W, axis=1, keepdims=True) + 1e-8)
    return (img * txt[perm]).sum(axis=1)


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "multi_density", f"Wrong kind: {cfg.kind}"
    assert cfg.models, "cfg.models is empty"
    assert cfg.cache is not None, "cfg.cache required (defines dataset / split)"
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    cosines: dict[str, np.ndarray] = {}
    method = MethodConfig(name="separated")

    for model_cfg in cfg.models:
        m_root = out_root / model_cfg.key
        cache = CacheConfig(
            cache_dir=cfg.cache.cache_dir.replace("{key}", model_cfg.key),
            dataset=cfg.cache.dataset, split=cfg.cache.split,
        )
        if stage in ("all", "extract"):
            extract_cache(model_cfg=model_cfg, cache_cfg=cache,
                          batch_size=64, num_workers=2, device=cfg.training.device)
        if stage == "extract":
            continue

        if stage in ("all", "train"):
            train_method(
                method=method, training=cfg.training,
                cache_dir=cache.cache_dir, hidden_size=model_cfg.hidden_size,
                save_dir=m_root,
            )

        if stage in ("all", "perm", "density"):
            from src.models import TwoSidedTopKSAE
            ckpt = m_root / "final"
            model = TwoSidedTopKSAE.from_pretrained(ckpt)
            perm_path = m_root / "perm.npz"
            if not perm_path.exists():
                payload = build_perm(
                    model=model, cache_dir=cache.cache_dir, split="train",
                    max_samples=50_000, batch_size=cfg.training.batch_size,
                    device=cfg.training.device,
                )
                save_perm(perm_path, payload)
            else:
                payload = dict(np.load(perm_path))

            W_img = model.image_sae.W_dec.detach().cpu().numpy()
            W_txt = model.text_sae.W_dec.detach().cpu().numpy()
            cosines[model_cfg.key] = _density_decoder_cosine(W_img, W_txt, payload["perm"])

    # Joint figure
    if stage in ("all", "plot"):
        from src.plotting.multi_density import plot_density
        plot_density(cosines, out_root / "multi_density")
        logger.info("[done] multi_density → %s", out_root)

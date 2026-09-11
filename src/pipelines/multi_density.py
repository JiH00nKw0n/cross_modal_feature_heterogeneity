"""Figure 2 pipeline: one density panel per vision-language model.

For each model listed in `cfg.models`:

  extract  Paired COCO embeddings into the model's own cache dir (the
           `{key}` placeholder in `cfg.cache.cache_dir` is substituted).
  train    A modality-specific `TwoSidedTopKSAE` with the hyper-parameters in
           `cfg.training`, saved to `<root>/<model_key>/final`.
  panel    The co-activation panel on the FULL COCO training split, written to
           `<root>/<model_key>/panel.npz`.
  plot     One figure across all models, at `<root>/multi_density.pdf`, plus
           `figure2_caption.md` and `figure2_bin_stats.md`.

The figure reads ALL (i, j) latent pairs with no filter, which is why the panel
it consumes is only used for its correlation matrix: the alive masks and the
Hungarian assignment inside panel.npz belong to Table 1, not to this figure.

Every stage is idempotent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.alignment import build_panel, load_panel, panel_mismatch, save_panel
from src.data.extract import extract_cache
from src.training.trainer import train_method
from src.utils.config import CacheConfig, Config, MethodConfig, ModelConfig

logger = logging.getLogger(__name__)

#: Stages this pipeline understands. `run.py` offers the union of every
#: pipeline's stages, so a name meant for another kind has to be refused here
#: rather than silently doing nothing and exiting successfully.
STAGES = ("all", "extract", "train", "perm", "density", "plot")

#: Display name per model key, used as the panel title in the figure.
MODEL_DISPLAY_NAMES = {
    "clip_b32": "CLIP ViT-B/32",
    "clip_l14": "CLIP ViT-L/14",
    "metaclip_b32": "MetaCLIP B/32",
    "metaclip_l14": "MetaCLIP L/14",
    "openclip_b32": "OpenCLIP B/32",
    "openclip_l14": "OpenCLIP L/14",
    "mobileclip2_b": "MobileCLIP2-B",
    "mobileclip2_l14": "MobileCLIP2-L/14",
    "siglip2_base": "SigLIP2 Base",
    "siglip2_large": "SigLIP2 Large",
}


def _cache_for(cfg: Config, model_cfg: ModelConfig) -> CacheConfig:
    return CacheConfig(
        cache_dir=cfg.cache.cache_dir.replace("{key}", model_cfg.key),
        dataset=cfg.cache.dataset,
        split=cfg.cache.split,
    )


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "multi_density", f"Wrong kind: {cfg.kind}"
    if stage not in STAGES:
        raise ValueError(
            f"stage {stage!r} does not exist for kind 'multi_density'; "
            f"it understands {', '.join(STAGES)}"
        )
    assert cfg.models, "cfg.models is empty"
    assert cfg.cache is not None, "cfg.cache is required (it names the dataset and split)"
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    method = MethodConfig(name="separated")
    train_split = cfg.cache.split or "train"
    panels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for model_cfg in cfg.models:
        m_root = out_root / model_cfg.key
        cache = _cache_for(cfg, model_cfg)

        if stage in ("all", "extract"):
            extract_cache(model_cfg=model_cfg, cache_cfg=cache,
                          batch_size=64, device=cfg.training.device)
        if stage == "extract":
            continue

        if stage in ("all", "train"):
            train_method(method=method, training=cfg.training,
                         cache_dir=cache.cache_dir, hidden_size=model_cfg.hidden_size,
                         save_dir=m_root, seed=cfg.training.resolved_seeds()[0],
                         split=train_split)
        if stage == "train":
            continue

        if stage in ("all", "perm", "density", "plot"):
            panels[model_cfg.key] = _panel_for(cfg, model_cfg, cache, m_root,
                                               train_split, stage)

    if stage in ("all", "density", "plot"):
        from src.plotting.multi_density import plot_density

        named = {MODEL_DISPLAY_NAMES.get(k, k): v for k, v in panels.items() if v is not None}
        if not named:
            logger.warning("[multi_density] no panels available - nothing to plot")
            return
        pdf = plot_density(named, out_root / "multi_density")
        logger.info("[done] multi_density -> %s", pdf)


def _panel_for(cfg: Config, model_cfg: ModelConfig, cache: CacheConfig,
               m_root: Path, train_split: str, stage: str):
    """Load or build one model's panel and return (C, W_img_unit, W_txt_unit)."""
    from src.models import TwoSidedTopKSAE
    from src.plotting.multi_density import unit_rows

    ckpt = m_root / "final"
    if not ckpt.exists():
        logger.warning("[multi_density] missing checkpoint %s - skipping %s",
                       ckpt, model_cfg.key)
        return None
    model = TwoSidedTopKSAE.from_pretrained(ckpt)

    panel_path = m_root / "panel.npz"
    payload = None
    if panel_path.exists():
        candidate = load_panel(panel_path)
        # An existing panel is reused only when it was built under the rules
        # this figure needs: the full split, the image-to-text pairing and the
        # true pairing. A quick-check or noise-floor panel left on disk has the
        # same shape and would otherwise be plotted as if it were the real one.
        why = panel_mismatch(candidate, split=train_split, pairing="img_txt",
                             max_samples=0, shuffle_seed=0)
        if why is None:
            logger.info("[multi_density][skip] %s exists", panel_path)
            payload = candidate
        else:
            logger.warning("[multi_density] rebuilding %s: %s", panel_path, why)
    if payload is None:
        payload = build_panel(
            model=model, cache_dir=cache.cache_dir, split=train_split,
            batch_size=cfg.training.batch_size, device=cfg.training.device,
            max_samples=0, pairing="img_txt", ckpt_a=ckpt,
        )
        save_panel(panel_path, payload)
        logger.info("[multi_density] saved %s", panel_path)

    W_img = unit_rows(model.image_sae.W_dec.detach().cpu().numpy())
    W_txt = unit_rows(model.text_sae.W_dec.detach().cpu().numpy())
    return np.asarray(payload["C"]), W_img, W_txt

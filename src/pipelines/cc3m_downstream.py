"""CC3M-trained downstream pipeline.

Stages (selected via ``--stage``):
  1. ``extract`` — paired (image, text) CC3M embeddings into ``cfg.cache.cache_dir``.
  2. ``train``   — one SAE per ``cfg.methods`` entry (shared / separated /
                  iso_align / group_sparse / ours; ``ours`` re-uses the
                  ``separated`` checkpoint, no separate training).
  3. ``perm``    — single Hungarian text→image slot permutation from CC3M
                  train embeddings, saved to ``<root>/ours/perm.npz``.
  4. ``eval``    — extract auxiliary caches if missing (COCO test for
                  retrieval/steering, ImageNet val for zero-shot, an
                  external-encoder CC3M cache for MS), then for each
                  method run any of {retrieval, zeroshot, steering, ms}
                  enabled in ``cfg.eval``. Each evaluator writes its own
                  JSON / CSV under ``<root>/eval/<method>/``.
  5. A ``config.json`` snapshot is always written for reproducibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.alignment import build_perm, save_perm
from src.data.extract import extract_cache
from src.training.trainer import train_method
from src.utils.config import Config

logger = logging.getLogger(__name__)


def _step(name: str, fn, *, output_marker: Path | None = None):
    if output_marker and output_marker.exists():
        logger.info("[skip] %s — exists: %s", name, output_marker)
        return
    logger.info("[run]  %s", name)
    fn()


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "cc3m_downstream", f"Wrong kind: {cfg.kind}"
    assert cfg.model is not None and cfg.cache is not None
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Extract cache (idempotent inside).
    if stage in ("all", "extract"):
        extract_cache(model_cfg=cfg.model, cache_cfg=cfg.cache,
                      batch_size=64, num_workers=2, device=cfg.training.device)

    if stage == "extract":
        return

    hidden_size = cfg.model.hidden_size
    method_artifacts: dict[str, Path] = {}

    # 2) Train every method declared in cfg.methods.
    if stage in ("all", "train"):
        for method in cfg.methods:
            save_dir = out_root / method.name
            arts = train_method(
                method=method, training=cfg.training,
                cache_dir=cfg.cache.cache_dir, hidden_size=hidden_size,
                save_dir=save_dir,
            )
            method_artifacts[method.name] = arts.save_dir

    # 3) Build single Hungarian perm from CC3M train (only for `ours`).
    has_ours = any(m.name == "ours" for m in cfg.methods)
    perm_path = out_root / "ours" / "perm.npz"
    if has_ours and stage in ("all", "perm"):
        if not perm_path.exists():
            from src.models import TwoSidedTopKSAE
            sep_dir = out_root / "separated" / "final"
            if not sep_dir.exists():
                logger.warning("[perm] separated ckpt missing at %s — skip", sep_dir)
            else:
                model = TwoSidedTopKSAE.from_pretrained(sep_dir)
                payload = build_perm(
                    model=model, cache_dir=cfg.cache.cache_dir,
                    split="train", max_samples=50_000,
                    batch_size=cfg.training.batch_size, device=cfg.training.device,
                )
                save_perm(perm_path, payload)
                logger.info("[perm] saved %s", perm_path)

    # 4) Downstream evals: COCO retrieval, ImageNet zero-shot, cross-modal
    #    steering, MonoSemanticity. Each writes its own JSON/CSV under
    #    <root>/eval/<method>/. Auxiliary caches are extracted on demand.
    if stage in ("all", "eval") and cfg.eval is not None:
        _run_eval_stage(cfg, out_root)

    # 5) Drop a config snapshot so the run is reproducible.
    with open(out_root / "config.json", "w") as f:
        json.dump({
            "kind": cfg.kind,
            "model": asdict(cfg.model),
            "cache": asdict(cfg.cache),
            "training": asdict(cfg.training),
            "methods": [asdict(m) for m in cfg.methods],
        }, f, indent=2)
    logger.info("[done] cc3m_downstream → %s", out_root)


def _run_eval_stage(cfg: Config, out_root: Path) -> None:
    """Build any missing auxiliary caches and dispatch each requested eval."""
    from src.data.extract_coco import extract as extract_coco
    from src.data.extract_imagenet import extract as extract_imagenet
    from src.data.extract_cc3m import extract as extract_cc3m
    from src.eval.retrieval import run as run_retrieval
    from src.eval.zeroshot import run as run_zeroshot
    from src.eval.steering import run as run_steering
    from src.eval.ms import run as run_ms
    from src.eval.coco_concepts import COCO_CONCEPTS
    from src.utils.config import load_config

    eval_cfg = cfg.eval
    eval_root = out_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    coco_cache = Path(f"cache/{cfg.model.key}_coco")
    inet_cache = Path(f"cache/{cfg.model.key}_imagenet")
    ext_key = getattr(eval_cfg, "external_encoder", "metaclip_b32")
    ext_cache = Path(f"cache/{ext_key}_cc3m")

    if eval_cfg.retrieval or eval_cfg.steering:
        if not (coco_cache / "image_embeddings.pt").exists():
            extract_coco(model_cfg=cfg.model, cache_dir=str(coco_cache),
                         splits=["train", "validation", "test"])
    if eval_cfg.zeroshot:
        if not (inet_cache / "text_embeddings.pt").exists():
            extract_imagenet(model_cfg=cfg.model, cache_dir=str(inet_cache),
                             splits=["validation"])
    if eval_cfg.monosemanticity:
        if not (ext_cache / "image_embeddings.pt").exists():
            ext_model_cfg = load_config(f"configs/models/{ext_key}.yaml")
            extract_cc3m(model_cfg=ext_model_cfg.model, cache_dir=str(ext_cache),
                         splits=[cfg.cache.split])

    perm_path = out_root / "ours" / "perm.npz"
    for method in cfg.methods:
        ckpt = (out_root / "separated" / "final") if method.name == "ours" \
               else (out_root / method.name / "final")
        if not ckpt.exists():
            logger.warning("[eval] missing ckpt %s — skipping %s", ckpt, method.name)
            continue
        m_perm = perm_path if method.name == "ours" else None

        if eval_cfg.retrieval:
            out_path = eval_root / method.name / "retrieval.json"
            if not out_path.exists():
                run_retrieval(ckpt=ckpt, method=method.name,
                              cache_dir=coco_cache, output=out_path,
                              split="test", perm_path=m_perm,
                              device=cfg.training.device)
            else:
                logger.info("[eval][skip] retrieval — %s exists", out_path)

        if eval_cfg.zeroshot:
            out_path = eval_root / method.name / "zeroshot.json"
            if not out_path.exists():
                run_zeroshot(ckpt=ckpt, method=method.name,
                             cache_dir=inet_cache, output=out_path,
                             perm_path=m_perm,
                             max_fire_rate=getattr(eval_cfg, "max_fire_rate", 0.5),
                             device=cfg.training.device)
            else:
                logger.info("[eval][skip] zeroshot — %s exists", out_path)

        if eval_cfg.steering:
            out_dir = eval_root / method.name / "steering"
            summary = out_dir / "summary.json"
            if not summary.exists():
                run_steering(ckpt=ckpt, method=method.name,
                             cache_dir=coco_cache,
                             captions_json=coco_cache / "captions.json",
                             output_dir=out_dir,
                             perm_path=m_perm,
                             alphas=getattr(eval_cfg, "steering_alphas",
                                            (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)),
                             concepts=COCO_CONCEPTS,
                             n_base_images=getattr(eval_cfg, "steering_n_base", 100),
                             variant=method.name,
                             device=cfg.training.device)
            else:
                logger.info("[eval][skip] steering — %s exists", summary)

        if eval_cfg.monosemanticity:
            out_dir = eval_root / method.name / "ms"
            summary = out_dir / "ms_summary.json"
            if not summary.exists():
                run_ms(ckpt=ckpt, method=method.name,
                       train_cache=cfg.cache.cache_dir,
                       ext_cache=ext_cache,
                       output_dir=out_dir,
                       dataset=cfg.cache.dataset, split=cfg.cache.split,
                       perm_path=m_perm, variant=method.name,
                       device=cfg.training.device)
            else:
                logger.info("[eval][skip] ms — %s exists", summary)

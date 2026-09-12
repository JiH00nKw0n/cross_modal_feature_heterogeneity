"""CC3M-trained downstream pipeline: trains every method, then writes Table 1.

Stages (selected with `--stage`):

  extract  Paired CC3M embeddings into `cfg.cache.cache_dir`.
  train    One SAE per entry of `cfg.methods`, per entry of
           `cfg.training.seeds`, saved to `<root>/seed{S}/<method>/final`.
           `ours` re-uses the `separated` checkpoint and trains nothing.
  perm     The co-activation panel for `ours`, built from that seed's
           `separated` checkpoint on the FULL CC3M training split, written to
           `<root>/seed{S}/ours/panel.npz` (and `perm.npz`, the same content
           under the older name).
  eval     Extracts the COCO and ImageNet caches on demand, then runs the
           reconstruction, retrieval and zero-shot evaluations per method into
           `<root>/seed{S}/eval/<method>/`. The ImageNet cache is extracted
           only when an evaluation reads it, which is zero-shot classification
           or the ImageNet half of reconstruction.
  table    `<root>/table1.md` and `<root>/table1.tex` across all seeds.

Every stage is idempotent: an artifact that already exists is not rebuilt.

The alive rule and the sample count are fixed, not configurable: a latent is
alive when it fires at least once, and the panel is built on every pair of the
training split.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.alignment import build_panel, load_panel, panel_mismatch, save_panel
from src.data.cache_io import require_cache_slice
from src.data.extract import extract_cache
from src.reporting.table1 import zeroshot_filename
from src.training.trainer import train_method
from src.utils.config import Config, EvalConfig, ModelConfig

logger = logging.getLogger(__name__)

#: Stages this pipeline understands. `run.py` offers the union of every
#: pipeline's stages, so a name meant for another kind is refused here rather
#: than silently doing nothing and exiting successfully.
STAGES = ("all", "extract", "train", "perm", "eval", "table")


def _coco_cache_dir(eval_cfg: EvalConfig, model_key: str) -> Path:
    """Where the COCO evaluation cache lives, with `{key}` filled in."""
    return Path(eval_cfg.coco_cache_dir.replace("{key}", model_key))


def _imagenet_cache_dir(eval_cfg: EvalConfig, model_key: str) -> Path:
    """Where the ImageNet evaluation cache lives, with `{key}` filled in."""
    return Path(eval_cfg.imagenet_cache_dir.replace("{key}", model_key))


def _wants_imagenet(eval_cfg: EvalConfig) -> bool:
    """Whether any configured evaluation reads the ImageNet cache.

    Two do: zero-shot classification, and the ImageNet half of the
    reconstruction evaluation. With both off, the cache is never extracted, and
    the run needs no Hugging Face token.
    """
    return bool(eval_cfg.zeroshot or (eval_cfg.recon and eval_cfg.recon_imagenet))


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "cc3m_downstream", f"Wrong kind: {cfg.kind}"
    if stage not in STAGES:
        raise ValueError(
            f"stage {stage!r} does not exist for kind 'cc3m_downstream'; "
            f"it understands {', '.join(STAGES)}"
        )
    assert cfg.model is not None and cfg.cache is not None
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = cfg.training.resolved_seeds()
    logger.info("[cc3m] seeds=%s methods=%s", seeds, [m.name for m in cfg.methods])

    # Refused before anything reads the cache, so that a slice left behind by a
    # smoke run is never trained on as if it were the whole corpus.
    require_cache_slice(cfg.cache.cache_dir, cfg.cache.max_samples)

    if stage in ("all", "extract"):
        extract_cache(model_cfg=cfg.model, cache_cfg=cfg.cache,
                      batch_size=64, device=cfg.training.device,
                      max_samples=cfg.cache.max_samples)
    if stage == "extract":
        return

    for seed in seeds:
        seed_root = out_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        if stage in ("all", "train"):
            _train_stage(cfg, seed_root, seed)
        if stage in ("all", "perm"):
            _panel_stage(cfg, seed_root)
        if stage in ("all", "eval") and cfg.eval is not None:
            _eval_stage(cfg, seed_root)

    if stage in ("all", "eval", "table"):
        _table_stage(cfg, out_root, seeds)

    with open(out_root / "config.json", "w") as f:
        json.dump({
            "kind": cfg.kind,
            "model": asdict(cfg.model),
            "cache": asdict(cfg.cache),
            "training": asdict(cfg.training),
            "seeds": seeds,
            "methods": [asdict(m) for m in cfg.methods],
            "eval": asdict(cfg.eval),
            "alive_rule": "fire_count >= 1 on the full train split",
        }, f, indent=2)
    logger.info("[done] cc3m_downstream -> %s", out_root)


def _train_stage(cfg: Config, seed_root: Path, seed: int) -> None:
    """Train every method for one seed. `ours` shares the separated checkpoint."""
    for method in cfg.methods:
        if method.name == "ours":
            continue
        train_method(
            method=method, training=cfg.training, cache_dir=cfg.cache.cache_dir,
            hidden_size=cfg.model.hidden_size, save_dir=seed_root / method.name,
            seed=seed, split=cfg.cache.split or "train",
        )


def _panel_stage(cfg: Config, seed_root: Path) -> None:
    """Build the panel for `ours` on the FULL training split of this seed's model."""
    if not any(m.name == "ours" for m in cfg.methods):
        return
    panel_path = seed_root / "ours" / "panel.npz"
    perm_path = seed_root / "ours" / "perm.npz"
    split = cfg.cache.split or "train"
    if panel_path.exists() and perm_path.exists():
        # Reuse only a panel built under the rules Table 1 needs: the full
        # split, the image-to-text pairing, the true pairing. A quick-check
        # panel (max_samples > 0) or a noise-floor panel (shuffle_seed != 0)
        # left on disk has the same shape and would otherwise be picked up as
        # the permutation behind every `ours` number.
        why = panel_mismatch(load_panel(panel_path), split=split,
                             pairing="img_txt", max_samples=0, shuffle_seed=0)
        if why is None:
            logger.info("[panel][skip] %s exists", panel_path)
            return
        logger.warning("[panel] rebuilding %s: %s", panel_path, why)
    sep_dir = seed_root / "separated" / "final"
    if not sep_dir.exists():
        logger.warning("[panel] separated checkpoint missing at %s - skipping", sep_dir)
        return

    from src.models import TwoSidedTopKSAE

    model = TwoSidedTopKSAE.from_pretrained(sep_dir)
    payload = build_panel(
        model=model, cache_dir=cfg.cache.cache_dir,
        split=split,
        batch_size=cfg.training.batch_size, device=cfg.training.device,
        max_samples=0, pairing="img_txt", ckpt_a=sep_dir,
    )
    save_panel(panel_path, payload)
    # perm.npz carries the same arrays, for call sites that still ask for it.
    save_panel(perm_path, payload)
    logger.info("[panel] saved %s and %s", panel_path, perm_path)


def _eval_stage(cfg: Config, seed_root: Path) -> None:
    """Run the configured evaluations for every method of one seed."""
    from src.data.extract import extract_coco, extract_imagenet
    from src.data.cache_io import imagenet_cache_complete, paired_cache_complete
    from src.eval.recon import run as run_recon
    from src.eval.retrieval import run as run_retrieval
    from src.eval.zeroshot import run as run_zeroshot

    eval_cfg = cfg.eval
    model_cfg: ModelConfig = cfg.model
    eval_root = seed_root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    coco_cache = _coco_cache_dir(eval_cfg, model_cfg.key)
    inet_cache = _imagenet_cache_dir(eval_cfg, model_cfg.key)
    # Both evaluation caches are extracted on demand below, so both are checked
    # here: one built from a different slice would otherwise be evaluated on
    # silently, and its recall and accuracy are not comparable with the full
    # run's.
    require_cache_slice(coco_cache, eval_cfg.max_samples)
    require_cache_slice(inet_cache, eval_cfg.max_samples)

    # `ours` is the separated checkpoint plus the permutation, so it cannot be
    # evaluated without the panel. Check before anything runs: the retrieval
    # call would otherwise fail on a bare missing-file error, after the other
    # four methods had already been evaluated, and nothing in that error would
    # say which stage was skipped.
    panel_path = seed_root / "ours" / "panel.npz"
    wants_ours = any(m.name == "ours" for m in cfg.methods)
    sep_ckpt = seed_root / "separated" / "final"
    if wants_ours and sep_ckpt.exists() and not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} is missing, so method 'ours' cannot be evaluated. "
            "Run the perm stage first (python run.py <config> --stage perm), or "
            "use --stage all, which runs it in order."
        )

    if (eval_cfg.retrieval or eval_cfg.recon) and not paired_cache_complete(coco_cache):
        extract_coco(model_cfg=model_cfg, cache_dir=coco_cache,
                     device=cfg.training.device,
                     max_groups_per_split=eval_cfg.max_samples)
    if _wants_imagenet(eval_cfg) and not imagenet_cache_complete(inet_cache):
        extract_imagenet(model_cfg=model_cfg, cache_dir=inet_cache,
                         device=cfg.training.device,
                         max_samples=eval_cfg.max_samples)

    zs_name = zeroshot_filename(eval_cfg.zeroshot_variant)

    for method in cfg.methods:
        ckpt = (seed_root / "separated" / "final") if method.name == "ours" \
            else (seed_root / method.name / "final")
        if not ckpt.exists():
            logger.warning("[eval] missing checkpoint %s - skipping %s", ckpt, method.name)
            continue
        m_perm = panel_path if method.name == "ours" else None
        out_dir = eval_root / method.name
        out_dir.mkdir(parents=True, exist_ok=True)

        if eval_cfg.recon:
            _once(out_dir / "recon_coco.json", lambda p: run_recon(
                ckpt=ckpt, method=method.name, cache_dir=coco_cache, output=p,
                dataset="coco", split="test", device=cfg.training.device))
            if eval_cfg.recon_imagenet:
                _once(out_dir / "recon_imagenet.json", lambda p: run_recon(
                    ckpt=ckpt, method=method.name, cache_dir=inet_cache, output=p,
                    dataset="imagenet", device=cfg.training.device,
                    template_seed=eval_cfg.recon_template_seed))

        if eval_cfg.retrieval:
            _once(out_dir / "retrieval.json", lambda p: run_retrieval(
                ckpt=ckpt, method=method.name, cache_dir=coco_cache, output=p,
                split="test", perm_path=m_perm, device=cfg.training.device))

        if eval_cfg.zeroshot:
            _once(out_dir / zs_name, lambda p: run_zeroshot(
                ckpt=ckpt, method=method.name, cache_dir=inet_cache, output=p,
                perm_path=m_perm, variant=eval_cfg.zeroshot_variant,
                max_fire_rate=eval_cfg.max_fire_rate, device=cfg.training.device))


def _once(out_path: Path, fn) -> None:
    """Run `fn(out_path)` unless the output already exists."""
    if out_path.exists():
        logger.info("[eval][skip] %s exists", out_path)
        return
    fn(out_path)


def _table_stage(cfg: Config, out_root: Path, seeds: list[int]) -> None:
    from src.reporting.table1 import write_table1

    methods = [m.name for m in cfg.methods]
    title = (f"Table 1 - trained on {cfg.cache.dataset} with {cfg.model.key}, "
             f"{cfg.training.num_epochs} epochs, L={cfg.training.latent_size}, "
             f"k={cfg.training.k}")
    write_table1(out_root, methods, seeds, title=title)

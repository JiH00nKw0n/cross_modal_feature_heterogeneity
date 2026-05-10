"""Synthetic α / λ sweep pipeline (paper §5, Fig 1 + Fig 2).

For each (alpha, latent_size, seed, method) combo:
  1. Generate synthetic paired data via SyntheticPairedBuilder.
  2. Train SAE for `method`.
  3. Compute (CR, RE, GRE, ESim, FSim) — the synthetic metric suite.
  4. Save W_enc / W_dec / b_enc / b_dec for offline replay.

The plot scripts under src.plotting.{alpha,lambda}_sweep consume the
saved `params/run_*.npz` shards to draw the final figures.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import yaml

from src.utils.config import Config

logger = logging.getLogger(__name__)


def run(cfg: Config, stage: str = "all") -> None:
    """Synthetic sweep delegates to the HF-Trainer-based driver in
    ``run_synthetic_v2.py`` at the repo root, which already supports the
    YAML schema used here.
    """
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    _allowed_training = {"lr", "num_epochs", "batch_size", "weight_decay",
                         "max_grad_norm", "k", "device", "weight_tie"}
    training_d = {k: v for k, v in cfg.training.__dict__.items() if k in _allowed_training}

    _name_remap = {"shared": "single_recon", "separated": "two_recon"}
    methods_d = []
    for m in cfg.methods:
        d = dict(m.__dict__)
        d["name"] = _name_remap.get(d["name"], d["name"])
        methods_d.append(d)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump({
        "data": cfg.data.__dict__,
        "training": training_d,
        "methods": methods_d,
        "sweep": cfg.sweep.__dict__,
        "output": cfg.output.__dict__,
    }, tmp)
    tmp.close()

    logger.info("[synthetic_sweep] dispatching to run_synthetic_v2 with %s", tmp.name)
    import run_synthetic_v2  # imported lazily so plot-only flows don't pay the cost
    run_synthetic_v2.main(["--config", tmp.name])
    logger.info("[done] synthetic_sweep → %s", out_root)

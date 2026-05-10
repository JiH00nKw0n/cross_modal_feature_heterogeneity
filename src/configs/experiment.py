"""Experiment configuration dataclasses for synthetic Theorem 2 experiments.

Load from YAML via ``ExperimentConfig.from_yaml(path)``.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _coerce_floats(d: dict[str, Any]) -> dict[str, Any]:
    """YAML parses ``5e-4`` as str; coerce numeric-looking strings to float."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


@dataclass
class DataConfig:
    n_shared: int = 1024
    n_image: int = 512
    n_text: int = 512
    representation_dim: int = 256
    sparsity: float = 0.99
    beta: float = 1.0
    obs_noise_std: float = 0.05
    max_interference: float = 0.1
    num_train: int = 50_000
    num_eval: int = 10_000
    l2_normalize: bool = False


@dataclass
class TrainingConfig:
    lr: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 256
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    k: int = 16
    device: str = "cuda"
    weight_tie: bool = False


@dataclass
class MethodConfig:
    name: str
    aux_weight: float = 0.0
    lambda_aux: float = 1.0
    m_S: int = 512
    k_align: int = 6
    aux_norm: str = "group"
    # --- aux-alignment variant fields (used by `paired_aux_alignment` method) ---
    variant_name: str = ""               # informative tag, e.g. "naive_once"
    aux_loss: str = "naive_diag"          # "none"|"naive_diag"|"barlow"|"infonce"
    hungarian_schedule: str = "once"      # "none"|"once"|"per_epoch_partitioned"
    revive_dead: bool = False
    rho0: float = 0.2
    barlow_lambda_off: float = 0.005


@dataclass
class SweepConfig:
    alpha: list[float] = field(default_factory=lambda: [0.5])
    latent_size: list[int] = field(default_factory=lambda: [8192])
    num_seeds: int = 3
    seed_base: int = 1


@dataclass
class OutputConfig:
    root: str = "outputs/synthetic_theorem2"
    run_tag: str = ""
    save_decoders: bool = True


@dataclass
class DiagnosticConfig:
    rho: float = 0.3


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    methods: list[MethodConfig] = field(default_factory=list)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diagnostic: DiagnosticConfig = field(default_factory=DiagnosticConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        data = DataConfig(**_coerce_floats(raw.get("data", {})))
        training = TrainingConfig(**_coerce_floats(raw.get("training", {})))

        methods = []
        for m in raw.get("methods", []):
            methods.append(MethodConfig(**_coerce_floats(m)))

        sweep_raw = raw.get("sweep", {})
        sweep = SweepConfig(**sweep_raw)

        output = OutputConfig(**raw.get("output", {}))
        diagnostic = DiagnosticConfig(**raw.get("diagnostic", {}))

        return cls(
            data=data,
            training=training,
            methods=methods,
            sweep=sweep,
            output=output,
            diagnostic=diagnostic,
        )

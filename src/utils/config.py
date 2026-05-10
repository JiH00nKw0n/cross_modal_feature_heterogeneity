"""YAML config loader with `!ref` cross-file references.

Single schema covers all 3 pipeline kinds (synthetic_sweep, multi_density,
cc3m_downstream). The `kind` field at top level dispatches to the right pipeline.

Usage:
    cfg = load_config("configs/cc3m/overrides/clip_l14.yaml")
    cfg.kind            # "cc3m_downstream"
    cfg.model.hf_id     # resolved from configs/models/clip_l14.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# YAML loader with !ref support
# --------------------------------------------------------------------------- #
class _RefLoader(yaml.SafeLoader):
    pass


def _construct_ref(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Any:
    """`!ref path/to/file.yaml#sub.key` → loaded value at that key."""
    raw = loader.construct_scalar(node)
    if "#" in raw:
        path, key = raw.split("#", 1)
    else:
        path, key = raw, ""
    base_dir = Path(getattr(loader, "_base_dir", "."))
    target = (base_dir / path).resolve() if not Path(path).is_absolute() else Path(path)
    with open(target) as f:
        sub_loader = _RefLoader(f)
        sub_loader._base_dir = target.parent  # type: ignore[attr-defined]
        try:
            data = sub_loader.get_single_data()
        finally:
            sub_loader.dispose()
    if not key:
        return data
    cur = data
    for part in key.split("."):
        cur = cur[part]
    return cur


_RefLoader.add_constructor("!ref", _construct_ref)


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        loader = _RefLoader(f)
        loader._base_dir = path.parent  # type: ignore[attr-defined]
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Defines a vision-language encoder."""
    key: str
    backend: str                    # "transformers" | "openclip"
    hf_id: str = ""                 # transformers model id
    pretrained: str = ""            # openclip pretrained tag
    arch: str = ""                  # openclip arch (e.g. ViT-B-32)
    hidden_size: int = 0
    text_max_length: int = 77
    is_siglip: bool = False
    image_size: int = 224

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheConfig:
    cache_dir: str
    dataset: str = "coco"           # coco | cc3m | imagenet
    split: str = "train"
    captions_json: str = ""


@dataclass
class TrainingConfig:
    lr: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 1024
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    k: int = 32
    latent_size: int = 8192
    warmup_ratio: float = 0.05
    device: str = "cuda"
    seed: int = 0


@dataclass
class MethodConfig:
    name: str                       # shared | separated | iso_align | group_sparse | ours
    aux_weight: float = 0.0


@dataclass
class EvalConfig:
    """Per-eval config; what gets run depends on pipeline kind."""
    retrieval: bool = True
    zeroshot: bool = True
    steering: bool = True
    monosemanticity: bool = True
    steering_alphas: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    )
    steering_n_base: int = 100
    ms_dataset: str = "cc3m"
    ms_split: str = "validation"
    external_encoder: str = "metaclip_b32"


@dataclass
class SyntheticDataConfig:
    n_shared: int = 1024
    n_image: int = 512
    n_text: int = 512
    representation_dim: int = 256
    sparsity: float = 0.99
    beta: float = 1.0
    obs_noise_std: float = 0.05
    max_interference: float = 0.10
    num_train: int = 50_000
    num_eval: int = 10_000
    l2_normalize: bool = False


@dataclass
class SweepConfig:
    alpha: list[float] = field(default_factory=lambda: [0.5])
    latent_size: list[int] = field(default_factory=lambda: [8192])
    num_seeds: int = 5
    seed_base: int = 1


@dataclass
class OutputConfig:
    root: str = "outputs/run"
    save_decoders: bool = True


@dataclass
class Config:
    """Unified config — all pipelines.  Some fields unused per kind."""
    kind: str = ""                  # synthetic_sweep | multi_density | cc3m_downstream
    model: ModelConfig | None = None
    models: list[ModelConfig] = field(default_factory=list)  # multi_density only
    cache: CacheConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    methods: list[MethodConfig] = field(default_factory=list)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # synthetic-only:
    data: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    # multi-density-only:
    extraction: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | os.PathLike) -> Config:
    raw = _load_yaml(Path(path))
    return _from_dict(raw)


def _coerce(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce numeric strings (5e-4) to float. YAML quirks."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and v.replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1).replace("e", "", 1).lstrip("-").isdigit():
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def _from_dict(raw: dict[str, Any]) -> Config:
    cfg = Config()
    cfg.kind = raw.get("kind", "")

    if "model" in raw and raw["model"]:
        cfg.model = ModelConfig.from_dict(raw["model"])
    if "models" in raw and raw["models"]:
        cfg.models = [ModelConfig.from_dict(m) for m in raw["models"]]
    if "cache" in raw and raw["cache"]:
        cfg.cache = CacheConfig(**_coerce(raw["cache"]))

    if "training" in raw:
        cfg.training = TrainingConfig(**_coerce(raw["training"]))
    if "methods" in raw:
        cfg.methods = [MethodConfig(**_coerce(m)) for m in raw["methods"]]
    if "eval" in raw:
        cfg.eval = EvalConfig(**raw["eval"])
    if "output" in raw:
        cfg.output = OutputConfig(**raw["output"])
    if "data" in raw:
        cfg.data = SyntheticDataConfig(**_coerce(raw["data"]))
    if "sweep" in raw:
        cfg.sweep = SweepConfig(**raw["sweep"])
    if "extraction" in raw:
        cfg.extraction = raw["extraction"]
    return cfg

"""YAML config loader with `!ref` cross-file references.

Single schema covers all 3 pipeline kinds (synthetic_sweep, multi_density,
cc3m_downstream). The `kind` field at top level dispatches to the right pipeline.

Usage:
    cfg = load_config("configs/cc3m/overrides/clip_l14.yaml")
    cfg.kind            # "cc3m_downstream"
    cfg.model.hf_id     # resolved from configs/models/clip_l14.yaml
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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
    """Training hyper-parameters.

    `latent_size` is the TOTAL latent budget. A two-sided model splits it in
    half, so `latent_size: 8192` means 4096 latents per modality.

    `seeds` is the list of training seeds a pipeline loops over. `seed` is the
    older single-seed field and is kept so existing configs still load. Read
    the value through `resolved_seeds()` rather than from either field.
    """

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
    seeds: list[int] = field(default_factory=list)

    def resolved_seeds(self) -> list[int]:
        """The seeds to train: `seeds` when it is set, otherwise `[seed]`."""
        if self.seeds:
            return [int(s) for s in self.seeds]
        return [int(self.seed)]


@dataclass
class MethodConfig:
    name: str                       # shared | separated | iso_align | group_sparse | ours
    aux_weight: float = 0.0


@dataclass
class EvalConfig:
    """Which downstream evaluations the cc3m_downstream pipeline runs.

    `zeroshot_variant` picks between the two zero-shot protocols implemented in
    `src.eval.zeroshot`: "raw" uses every latent column and is the one Table 1
    reports, "filtered" drops columns whose image-side firing rate exceeds
    `max_fire_rate`.
    """

    recon: bool = True
    retrieval: bool = True
    zeroshot: bool = True
    zeroshot_variant: str = "raw"
    max_fire_rate: float = 0.5
    recon_template_seed: int = 0


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
class RebuttalConfig:
    """Knobs of the post-rebuttal analyses.

    `tau` is the co-activation correlation a latent pair has to reach before
    its two feature directions are compared. 0.4 is the working threshold; the
    reports also show 0.6.

    `null_seed` is the row shuffle that destroys the image-to-caption pairing
    and so turns a panel into a noise floor. It has to be non-zero, because
    zero means "do not shuffle".

    `n_boot` is the number of bootstrap resamples behind every confidence
    interval.

    `settings` names which trained configurations to analyse: "coco_k8" is the
    paper's Figure 2 point, "cc3m_k32" its Table 1 point.

    `analyses` names which analysis modules to run, or holds the single entry
    "all".

    `coco_seed_b` is the training seed of the second COCO model, the
    independent run that the same-modality comparisons need. The Figure 2
    pipeline trains seed 0; the rebuttal stage trains this second one itself.
    """

    tau: float = 0.4
    null_seed: int = 7
    n_boot: int = 1000
    settings: list[str] = field(default_factory=lambda: ["coco_k8", "cc3m_k32"])
    analyses: list[str] = field(default_factory=lambda: ["all"])
    coco_seed_b: int = 1


@dataclass
class Config:
    """Unified config — all pipelines.  Some fields unused per kind."""
    # synthetic_sweep | multi_density | cc3m_downstream | post_rebuttal
    kind: str = ""
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
    # post_rebuttal-only: the knobs, plus the two pipeline configs it drives.
    # `figure2` and `table1` are whole Configs of kind multi_density and
    # cc3m_downstream, pulled in with !ref, so the rebuttal reads exactly the
    # checkpoints and panels those two pipelines wrote.
    rebuttal: RebuttalConfig = field(default_factory=RebuttalConfig)
    figure2: "Config | None" = None
    table1: "Config | None" = None


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


def _known_only(cls: type, d: dict[str, Any], where: str) -> dict[str, Any]:
    """Drop keys the dataclass does not define, naming them in a warning.

    Keeps an older config loadable after a field is retired, instead of failing
    with a TypeError that says nothing about which key is stale.
    """
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    unknown = [k for k in d if k not in fields]
    if unknown:
        logger.warning("ignoring unknown %s keys: %s", where, ", ".join(sorted(unknown)))
    return {k: v for k, v in d.items() if k in fields}


def _from_dict(raw: dict[str, Any]) -> Config:
    cfg = Config()
    cfg.kind = raw.get("kind", "")

    if "model" in raw and raw["model"]:
        cfg.model = ModelConfig.from_dict(raw["model"])
    if "models" in raw and raw["models"]:
        cfg.models = [ModelConfig.from_dict(m) for m in raw["models"]]
    if "cache" in raw and raw["cache"]:
        cfg.cache = CacheConfig(**_known_only(CacheConfig, _coerce(raw["cache"]), "cache"))

    if "training" in raw:
        cfg.training = TrainingConfig(
            **_known_only(TrainingConfig, _coerce(raw["training"]), "training"))
    if "methods" in raw:
        cfg.methods = [MethodConfig(**_known_only(MethodConfig, _coerce(m), "method"))
                       for m in raw["methods"]]
    if "eval" in raw:
        cfg.eval = EvalConfig(**_known_only(EvalConfig, raw["eval"], "eval"))
    if "output" in raw:
        cfg.output = OutputConfig(**raw["output"])
    if "data" in raw:
        cfg.data = SyntheticDataConfig(**_coerce(raw["data"]))
    if "sweep" in raw:
        cfg.sweep = SweepConfig(**raw["sweep"])
    if "extraction" in raw:
        cfg.extraction = raw["extraction"]
    if "rebuttal" in raw and raw["rebuttal"]:
        cfg.rebuttal = RebuttalConfig(
            **_known_only(RebuttalConfig, _coerce(raw["rebuttal"]), "rebuttal"))
    # A post_rebuttal config carries two whole pipeline configs, so the same
    # parser runs on them: whatever `load_config` would have made of
    # clip_b32_coco.yaml on its own is what lands in `cfg.figure2`.
    for nested in ("figure2", "table1"):
        if raw.get(nested):
            setattr(cfg, nested, _from_dict(raw[nested]))
    return cfg

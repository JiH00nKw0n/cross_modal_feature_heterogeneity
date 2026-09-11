"""The YAML loader: !ref resolution, the seed list, and retired keys."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import TrainingConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_shipped_config_loads() -> None:
    """A runnable config names a kind; fragments pulled in with !ref do not."""
    for path in sorted(CONFIGS.rglob("*.yaml")):
        if path.parent.name == "models" or path.name.startswith("_"):
            continue
        cfg = load_config(path)
        assert cfg.kind, f"{path} has no kind"


def test_ref_pulls_a_subkey_from_another_file() -> None:
    cfg = load_config(CONFIGS / "cc3m" / "overrides" / "clip_b32.yaml")
    assert cfg.model is not None and cfg.model.key == "clip_b32"
    assert cfg.model.hidden_size == 512
    assert [m.name for m in cfg.methods] == [
        "shared", "separated", "iso_align", "group_sparse", "ours"]


def test_seeds_default_to_the_single_seed_field() -> None:
    assert TrainingConfig(seed=7).resolved_seeds() == [7]
    assert TrainingConfig(seed=7, seeds=[1, 2]).resolved_seeds() == [1, 2]


def test_post_rebuttal_configs_carry_the_stated_settings() -> None:
    coco = load_config(CONFIGS / "post_rebuttal" / "clip_b32_coco.yaml")
    assert coco.kind == "multi_density"
    assert [m.key for m in coco.models] == ["clip_b32"]
    assert coco.training.k == 8
    assert coco.training.latent_size == 8192
    assert coco.training.num_epochs == 30
    assert coco.output.root == "outputs/post_rebuttal/coco_clip_b32"

    cc3m = load_config(CONFIGS / "post_rebuttal" / "clip_b32_cc3m.yaml")
    assert cc3m.kind == "cc3m_downstream"
    assert cc3m.training.resolved_seeds() == [0, 1, 2]
    assert cc3m.eval.recon and cc3m.eval.retrieval and cc3m.eval.zeroshot
    assert cc3m.eval.zeroshot_variant == "raw"
    assert cc3m.output.root == "outputs/post_rebuttal/cc3m_clip_b32"


def test_a_retired_eval_key_is_dropped_with_a_warning(tmp_path: Path, caplog) -> None:
    """An older config that still names steering must load, minus that key."""
    path = tmp_path / "old.yaml"
    path.write_text(
        "kind: cc3m_downstream\n"
        "eval:\n  retrieval: true\n  steering: true\n  monosemanticity: false\n"
    )
    with caplog.at_level("WARNING"):
        cfg = load_config(path)
    assert cfg.eval.retrieval is True
    assert not hasattr(cfg.eval, "steering")
    assert "steering" in caplog.text

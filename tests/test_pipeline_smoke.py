"""End to end on tiny fake caches: cc3m_downstream must reach table1.md.

The pipeline is pointed at a COCO-shaped cache and an ImageNet-shaped cache
that the test writes itself, so no encoder is loaded and nothing is
downloaded. Both caches are already complete, which is what makes the
extraction stage skip: that is the same idempotence check the real run relies
on, exercised here rather than mocked away.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.cc3m_downstream import run as run_cc3m
from src.utils.config import (
    CacheConfig,
    Config,
    EvalConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)
from tests.conftest import make_coco_cache, make_imagenet_cache

DIM = 16
METHODS = ["shared", "separated", "iso_align", "group_sparse", "ours"]


def _build_config(root: Path) -> Config:
    cfg = Config()
    cfg.kind = "cc3m_downstream"
    cfg.model = ModelConfig(key="fake", backend="transformers", hidden_size=DIM)
    cfg.cache = CacheConfig(cache_dir="cache/fake_coco", dataset="coco", split="train")
    cfg.training = TrainingConfig(
        lr=1e-3, num_epochs=1, batch_size=8, k=2, latent_size=8,
        warmup_ratio=0.5, device="cpu", seeds=[0],
    )
    cfg.methods = [MethodConfig(name=m) for m in METHODS]
    cfg.eval = EvalConfig(recon=True, retrieval=True, zeroshot=True, zeroshot_variant="raw")
    cfg.output = OutputConfig(root=str(root))
    return cfg


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding cache/fake_coco and cache/fake_imagenet.

    The pipeline resolves its auxiliary caches relative to the working
    directory, so the test has to run inside one.
    """
    monkeypatch.chdir(tmp_path)
    make_coco_cache(Path("cache/fake_coco"), n_images=8, caps_per_image=5, dim=DIM)
    make_imagenet_cache(Path("cache/fake_imagenet"), n_images=12, n_classes=3,
                        n_templates=4, dim=DIM)
    return tmp_path


def test_cc3m_downstream_reaches_table1(workspace: Path) -> None:
    root = workspace / "out"
    cfg = _build_config(root)
    run_cc3m(cfg, stage="all")

    table = root / "table1.md"
    assert table.exists(), "the pipeline must end at table1.md"
    text = table.read_text()
    for method in METHODS:
        label = {
            "shared": "Shared SAE", "separated": "Modality-Specific SAEs",
            "iso_align": "Iso-Energy Alignment", "group_sparse": "Group-Sparse",
            "ours": "Post-hoc Alignment (Ours)",
        }[method]
        assert label in text, f"{label} is missing from the table"
    assert (root / "table1.tex").exists()

    # Every method trained except `ours`, which re-uses the separated model.
    for method in METHODS:
        if method == "ours":
            continue
        assert (root / "seed0" / method / "final" / "config.json").exists()
    assert not (root / "seed0" / "ours" / "final").exists()

    # The panel is the one artifact the alignment reads.
    panel = root / "seed0" / "ours" / "panel.npz"
    assert panel.exists()
    meta = json.loads((root / "seed0" / "ours" / "panel.json").read_text())
    assert meta["alive_rule"] == "fire_count >= 1 on the full train split"
    assert meta["n_samples"] == meta["n_split_rows"], "the panel must use every pair"
    assert (root / "seed0" / "ours" / "perm.npz").exists()

    # Four evaluation files per method, named the way table1 expects them.
    for method in METHODS:
        eval_dir = root / "seed0" / "eval" / method
        for name in ("recon_coco.json", "recon_imagenet.json",
                     "retrieval.json", "zeroshot.json"):
            assert (eval_dir / name).exists(), f"{method}/{name} is missing"
    zs = json.loads((root / "seed0" / "eval" / "ours" / "zeroshot.json").read_text())
    assert zs["variant"] == "raw"
    assert zs["kept_latents"] == zs["total_latents"], "the raw variant filters nothing"


def test_rerunning_the_pipeline_changes_nothing(workspace: Path) -> None:
    """Idempotence: a second run must not rebuild what already exists."""
    root = workspace / "out"
    cfg = _build_config(root)
    run_cc3m(cfg, stage="all")
    panel = root / "seed0" / "ours" / "panel.npz"
    before = panel.stat().st_mtime_ns
    checkpoint = root / "seed0" / "separated" / "final" / "config.json"
    ckpt_before = checkpoint.stat().st_mtime_ns

    run_cc3m(cfg, stage="all")
    assert panel.stat().st_mtime_ns == before
    assert checkpoint.stat().st_mtime_ns == ckpt_before


def test_extract_stage_skips_a_complete_cache(workspace: Path) -> None:
    """No encoder is ever loaded when the cache is already there."""
    cfg = _build_config(workspace / "out")
    run_cc3m(cfg, stage="extract")
    assert not (workspace / "out" / "seed0").exists()


def test_multi_density_pipeline_reaches_the_figure(workspace: Path) -> None:
    """The Figure 2 pipeline runs on the same fake COCO cache."""
    from src.pipelines.multi_density import run as run_density

    cfg = Config()
    cfg.kind = "multi_density"
    cfg.models = [ModelConfig(key="fake", backend="transformers", hidden_size=DIM)]
    cfg.cache = CacheConfig(cache_dir="cache/{key}_coco", dataset="coco", split="train")
    cfg.training = TrainingConfig(lr=1e-3, num_epochs=1, batch_size=8, k=2,
                                  latent_size=8, warmup_ratio=0.5, device="cpu", seeds=[0])
    root = workspace / "density"
    cfg.output = OutputConfig(root=str(root))

    run_density(cfg, stage="all")
    assert (root / "multi_density.pdf").exists()
    assert (root / "multi_density.png").exists()
    assert (root / "figure2_caption.md").exists()
    assert (root / "figure2_bin_stats.md").exists()
    assert (root / "fake" / "panel.npz").exists()

    stats = json.loads((root / "figure2_bin_stats.json").read_text())
    model_stats = next(iter(stats.values()))
    # No filter: every ordered (i, j) latent pair takes part.
    assert model_stats["n_pairs"] == 4 * 4


def test_eval_without_the_perm_stage_says_so(workspace: Path) -> None:
    """`--stage eval` after only extract and train must name the missing stage.

    Without the panel, `ours` is just the separated checkpoint, and the
    retrieval call would fail on a bare missing-file error after the other four
    methods had already been evaluated.
    """
    root = workspace / "out"
    cfg = _build_config(root)
    run_cc3m(cfg, stage="train")
    with pytest.raises(FileNotFoundError, match="--stage perm"):
        run_cc3m(cfg, stage="eval")


def test_a_stage_the_kind_does_not_have_is_refused(workspace: Path) -> None:
    """A silent success would read as "the figure step ran"."""
    from src.pipelines.multi_density import run as run_density

    cfg = _build_config(workspace / "out")
    with pytest.raises(ValueError, match="does not exist for kind 'cc3m_downstream'"):
        run_cc3m(cfg, stage="density")

    dens = Config()
    dens.kind = "multi_density"
    dens.models = [ModelConfig(key="fake", backend="transformers", hidden_size=DIM)]
    dens.cache = CacheConfig(cache_dir="cache/{key}_coco", dataset="coco", split="train")
    dens.output = OutputConfig(root=str(workspace / "density"))
    with pytest.raises(ValueError, match="does not exist for kind 'multi_density'"):
        run_density(dens, stage="table")


def test_a_quick_check_panel_is_not_reused_as_the_full_one(workspace: Path) -> None:
    """A panel built with max_samples > 0 must be rebuilt, not adopted."""
    from src.alignment import build_panel, save_panel
    from src.models import TwoSidedTopKSAE

    root = workspace / "out"
    cfg = _build_config(root)
    run_cc3m(cfg, stage="train")

    ours = root / "seed0" / "ours"
    ours.mkdir(parents=True, exist_ok=True)
    model = TwoSidedTopKSAE.from_pretrained(root / "seed0" / "separated" / "final")
    quick = build_panel(model=model, cache_dir=cfg.cache.cache_dir, split="train",
                        batch_size=8, device="cpu", max_samples=4)
    save_panel(ours / "panel.npz", quick)
    save_panel(ours / "perm.npz", quick)
    assert json.loads((ours / "panel.json").read_text())["max_samples"] == 4

    run_cc3m(cfg, stage="perm")
    meta = json.loads((ours / "panel.json").read_text())
    assert meta["max_samples"] == 0
    assert meta["n_samples"] == meta["n_split_rows"], "the panel must cover every pair"

"""The post-rebuttal pipeline, end to end on tiny fixtures.

This is a scaffold: it exercises the parts of the pipeline that exist before
any analysis has been ported, namely the stage names, the registry, the extra
co-activation panels and the report. As analyses are registered, extend
`test_the_rebuttal_stage_runs_every_registered_analysis` rather than writing a
separate smoke test per analysis.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.alignment import load_panel
from src.pipelines.post_rebuttal import STAGES
from src.pipelines.post_rebuttal import run as post_rebuttal_run
from src.rebuttal import registry
from src.rebuttal.common import Setting
from src.rebuttal.registry import Analysis, analyses_for
from src.utils.config import (
    CacheConfig,
    Config,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    RebuttalConfig,
    TrainingConfig,
)
from tests.conftest import make_coco_cache, make_two_sided_sae


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_every_registered_analysis_is_well_formed() -> None:
    """Each row names a module that imports and a subset of the real settings."""
    import importlib

    from src.rebuttal.common import SETTING_TAGS

    names = [a.name for a in registry.ANALYSES]
    assert len(names) == len(set(names)), f"duplicate analysis names: {names}"
    for analysis in registry.ANALYSES:
        assert analysis.settings, f"{analysis.name} applies to no setting"
        unknown = set(analysis.settings) - set(SETTING_TAGS)
        assert not unknown, f"{analysis.name} names unknown settings {unknown}"
        module = importlib.import_module(analysis.module_path)
        assert callable(getattr(module, "run", None)), \
            f"{analysis.module_path} has no run()"


def test_analyses_for_filters_by_name_and_by_setting(monkeypatch) -> None:
    rows = [
        Analysis("both_settings", "src.rebuttal.common", ("coco_k8", "cc3m_k32")),
        Analysis("coco_only", "src.rebuttal.common", ("coco_k8",)),
    ]
    monkeypatch.setattr(registry, "ANALYSES", rows)
    assert [a.name for a in analyses_for("coco_k8")] == ["both_settings", "coco_only"]
    assert [a.name for a in analyses_for("cc3m_k32")] == ["both_settings"]
    assert [a.name for a in analyses_for("coco_k8", ["all"])] == \
        ["both_settings", "coco_only"]
    assert [a.name for a in analyses_for("coco_k8", ["coco_only"])] == ["coco_only"]


def test_analyses_for_refuses_a_name_nobody_registered(monkeypatch) -> None:
    """A typo in the config must not read as a deliberately empty run."""
    monkeypatch.setattr(registry, "ANALYSES", [])
    with pytest.raises(ValueError, match="unknown analyses"):
        analyses_for("coco_k8", ["no_such_analysis"])


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def test_a_stage_from_another_pipeline_is_refused(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    with pytest.raises(ValueError, match="does not exist for kind 'post_rebuttal'"):
        post_rebuttal_run(cfg, stage="extract")


def test_the_stage_names_are_the_four_deliverables_plus_all() -> None:
    assert STAGES == ("all", "figure2", "table1", "rebuttal", "report")


# --------------------------------------------------------------------------- #
# rebuttal stage
# --------------------------------------------------------------------------- #
def test_the_rebuttal_stage_builds_the_extra_panels(tmp_path: Path) -> None:
    """Two same-modality panels, one different-caption panel, one noise floor."""
    cfg = _tiny_config(tmp_path)
    _place_checkpoints(cfg, tmp_path)
    post_rebuttal_run(cfg, stage="rebuttal")

    panels = tmp_path / "out" / "rebuttal" / "coco_k8" / "panels"
    for name in ("img_img.npz", "txt_txt.npz", "txt_txt_diffcap.npz", "img_txt_null.npz"):
        assert (panels / name).exists(), f"{name} was not built"

    # The sidecar has to record how each panel was built, or a later run cannot
    # tell a noise-floor panel from the real one.
    null_meta = load_panel(panels / "img_txt_null.npz")["_meta"]
    assert null_meta["pairing"] == "img_txt"
    assert null_meta["shuffle_seed"] == cfg.rebuttal.null_seed
    same_meta = load_panel(panels / "img_img.npz")["_meta"]
    assert same_meta["pairing"] == "img_img"
    assert same_meta["shuffle_seed"] == 0
    assert same_meta["max_samples"] == 0

    # CC3M carries one caption per image, so it gets no different-caption panel.
    cc3m_panels = tmp_path / "out" / "rebuttal" / "cc3m_k32" / "panels"
    assert not (cc3m_panels / "txt_txt_diffcap.npz").exists()


def test_the_rebuttal_stage_records_which_checkpoints_it_read(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    _place_checkpoints(cfg, tmp_path)
    post_rebuttal_run(cfg, stage="rebuttal")
    import json

    manifest = json.loads(
        (tmp_path / "out" / "rebuttal" / "coco_k8" / "setting.json").read_text())
    assert manifest["tag"] == "coco_k8"
    assert manifest["k"] == 2
    assert manifest["ckpt_a"].endswith("clip_b32/final")


def test_the_rebuttal_stage_runs_every_registered_analysis(tmp_path: Path, monkeypatch) -> None:
    """A registered analysis is called once per setting it applies to, then skipped.

    Extend this as analyses land: the contract under test is the one every
    ported module has to satisfy, that `run(setting, out_dir=..., device=...,
    **knobs)` writes `<out_dir>/<name>.json` and is not called again once that
    file exists.
    """
    calls: list[tuple[str, str, float, int]] = []

    def fake_run(setting: Setting, *, out_dir, device, **knobs) -> dict:
        calls.append((setting.tag, device, knobs["tau"], knobs["n_boot"]))
        payload = {"setting": setting.tag, "rows": 0}
        (Path(out_dir) / "fake_analysis.json").write_text('{"rows": 0}')
        (Path(out_dir) / "fake_analysis.md").write_text(
            "# Fake analysis\n\nMeasured nothing, over 0 rows.\n")
        return payload

    module = types.ModuleType("tests_fake_analysis")
    module.run = fake_run
    monkeypatch.setitem(sys.modules, "tests_fake_analysis", module)
    monkeypatch.setattr(registry, "ANALYSES", [
        Analysis("fake_analysis", "tests_fake_analysis", ("coco_k8", "cc3m_k32")),
    ])

    cfg = _tiny_config(tmp_path)
    _place_checkpoints(cfg, tmp_path)
    post_rebuttal_run(cfg, stage="rebuttal")
    assert [c[0] for c in calls] == ["coco_k8", "cc3m_k32"]
    assert calls[0][2] == pytest.approx(cfg.rebuttal.tau)
    assert calls[0][3] == cfg.rebuttal.n_boot

    # Idempotent: the json is on disk, so a second pass does not call it again.
    post_rebuttal_run(cfg, stage="rebuttal")
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# report stage
# --------------------------------------------------------------------------- #
def test_the_report_names_a_missing_analysis_instead_of_dropping_it(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, "ANALYSES", [
        Analysis("never_ran", "src.rebuttal.common", ("coco_k8",)),
    ])
    cfg = _tiny_config(tmp_path)
    post_rebuttal_run(cfg, stage="report")
    text = (tmp_path / "out" / "post_rebuttal_results.md").read_text()
    assert "## Setting coco_k8" in text
    assert "### never_ran" in text
    assert "Missing" in text
    assert "—" not in text


def test_the_report_inlines_each_analysis_report_under_one_top_heading(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, "ANALYSES", [
        Analysis("fake_analysis", "src.rebuttal.common", ("coco_k8",)),
    ])
    cfg = _tiny_config(tmp_path)
    out_dir = tmp_path / "out" / "rebuttal" / "coco_k8"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fake_analysis.md").write_text(
        "# Fake analysis\n\nMeasured over 512 rows.\n\n## Results\n\nnothing\n")
    post_rebuttal_run(cfg, stage="report")
    text = (tmp_path / "out" / "post_rebuttal_results.md").read_text()
    assert text.startswith("# Post-rebuttal measurements\n")
    assert text.count("\n# ") == 0          # the report's own title became the section
    assert "### Fake analysis" in text      # its title, not its module name
    assert "#### Results" in text           # its own sections pushed under it
    assert "Measured over 512 rows." in text


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tiny_config(tmp_path: Path) -> Config:
    """A post-rebuttal config over two tiny caches, everything on the CPU.

    The COCO cache carries two captions per photograph, which the
    different-caption pairing needs; the CC3M-shaped cache carries one, which
    is why that pairing is not built for it.
    """
    make_coco_cache(tmp_path / "coco", n_images=8, caps_per_image=2, dim=16, seed=0)
    make_coco_cache(tmp_path / "cc3m", n_images=8, caps_per_image=1, dim=16, seed=1)

    model = ModelConfig(key="clip_b32", backend="transformers", hidden_size=16)
    training = TrainingConfig(num_epochs=1, batch_size=4, k=2, latent_size=8,
                              device="cpu", seed=0, seeds=[0, 1, 2])

    figure2 = Config(
        kind="multi_density", models=[model],
        cache=CacheConfig(cache_dir=str(tmp_path / "coco"), dataset="coco", split="train"),
        training=training,
        output=OutputConfig(root=str(tmp_path / "out" / "coco_clip_b32")),
    )
    table1 = Config(
        kind="cc3m_downstream", model=model,
        cache=CacheConfig(cache_dir=str(tmp_path / "cc3m"), dataset="cc3m", split="train"),
        training=training,
        methods=[MethodConfig(name=n) for n in
                 ("shared", "separated", "iso_align", "group_sparse", "ours")],
        output=OutputConfig(root=str(tmp_path / "out" / "cc3m_clip_b32")),
    )
    return Config(
        kind="post_rebuttal",
        output=OutputConfig(root=str(tmp_path / "out")),
        rebuttal=RebuttalConfig(tau=0.4, null_seed=7, n_boot=16,
                                settings=["coco_k8", "cc3m_k32"],
                                analyses=["all"], coco_seed_b=1),
        figure2=figure2, table1=table1,
    )


def _place_checkpoints(cfg: Config, tmp_path: Path) -> None:
    """Save the four checkpoints the rebuttal stage expects to find or train.

    Model A of each setting stands in for what Figure 2 and Table 1 wrote;
    model B of the CC3M setting stands in for its seed-1 run. The COCO model B
    is deliberately left out, so that the stage trains it, which is what it does
    in production.
    """
    dim = cfg.figure2.models[0].hidden_size
    L = cfg.figure2.training.latent_size
    k = cfg.figure2.training.k
    for seed, path in enumerate([
        Path(cfg.figure2.output.root) / "clip_b32" / "final",
        Path(cfg.table1.output.root) / "seed0" / "separated" / "final",
        Path(cfg.table1.output.root) / "seed1" / "separated" / "final",
    ]):
        make_two_sided_sae(dim=dim, latent_size=L, k=k, seed=seed).save_pretrained(path)

"""The post-rebuttal pipeline, end to end on tiny fixtures, on the CPU.

Every stage runs for real: the Figure 2 pipeline trains a model and builds its
co-activation panel, the Table 1 pipeline trains four methods for two seeds and
evaluates them, the rebuttal stage trains the second COCO model, builds the
extra panels and runs every registered analysis, and the report stage gathers
the result. Nothing is downloaded and no encoder is loaded: the three embedding
caches are written by the test, already complete, so every extraction step
skips, and the COCO object annotations come from a fixture file.

The caches are deliberately shaped the way the real ones are. The COCO cache
carries two captions per photograph, which the different-caption pairing needs,
and a large test split, because the COCO-80 analyses measure that split and need
each object category to have enough positives in each half of it. The
CC3M-shaped cache carries one caption per image, which is why no
different-caption panel is built for it.
"""

from __future__ import annotations

import json
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
    EvalConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    RebuttalConfig,
    TrainingConfig,
)
from tests.conftest import (
    make_coco_cache,
    make_coco_instances,
    make_imagenet_cache,
    make_two_sided_sae,
)

#: Embedding width of every fixture cache.
DIM = 16

#: Total latent budget, so 4 latents per modality, of which 2 are active.
LATENT_TOTAL = 8
K = 2

#: Photographs of the fixture COCO cache, and how many of them are its test
#: split. The COCO-80 analyses drop an object category with fewer than 50
#: positives in a half of that split, and the fixture annotation file gives each
#: photograph one large object and one small one drawn from ten categories, so
#: 1,600 test photographs leave roughly 80 positives per half under the area
#: condition and twice that without it.
COCO_IMAGES = 1_700
COCO_TEST_IMAGES = 1_600
COCO_CAPS = 2

#: Photographs of the fixture CC3M-shaped cache, one caption each.
CC3M_IMAGES = 120

#: Every analysis registered for at least one setting, in reporting order.
ALL_ANALYSES = [a.name for a in registry.ANALYSES]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_every_registered_analysis_is_well_formed() -> None:
    """Each row names a module that imports and a subset of the real settings."""
    import importlib

    from src.rebuttal.common import SETTING_TAGS

    names = [a.name for a in registry.ANALYSES]
    assert names, "no analysis is registered"
    assert len(names) == len(set(names)), f"duplicate analysis names: {names}"
    for analysis in registry.ANALYSES:
        assert analysis.settings, f"{analysis.name} applies to no setting"
        unknown = set(analysis.settings) - set(SETTING_TAGS)
        assert not unknown, f"{analysis.name} names unknown settings {unknown}"
        module = importlib.import_module(analysis.module_path)
        assert callable(getattr(module, "run", None)), \
            f"{analysis.module_path} has no run()"
        assert registry.question_for(analysis.name) != analysis.name, \
            f"{analysis.name} has no question recorded in registry.QUESTIONS"


def test_the_registry_holds_every_analysis_in_reporting_order() -> None:
    assert ALL_ANALYSES == [
        "same_modality_control",
        "coco80_correspondence",
        "coco80_heterogeneity",
        "match_confidence",
        "correlation_bands",
        "one_to_many_span",
        "one_to_many_splitting",
        "alignment_ceiling",
        "stability_conditioned",
        "confidence_ablation",
        "alignment_methods",
    ]
    coco_only = [a.name for a in analyses_for("coco_k8")]
    assert "correlation_bands" not in coco_only, \
        "Figure 2 already writes the per-band statistics for the COCO setting"
    assert "confidence_ablation" not in coco_only
    assert "alignment_methods" not in coco_only


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
# end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def finished_run(tmp_path_factory, request) -> dict:
    """Run every stage once, and hand the paths to the tests that read them.

    Module scoped because the whole pipeline runs here; each test below reads
    one part of what it produced rather than running it again.
    """
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    tmp_path = tmp_path_factory.mktemp("post_rebuttal")
    monkeypatch.chdir(tmp_path)
    _write_caches(tmp_path)
    _point_the_annotations_at_the_fixture(monkeypatch, tmp_path)

    cfg = _tiny_config(tmp_path)
    post_rebuttal_run(cfg, stage="all")
    return {"tmp": tmp_path, "cfg": cfg, "root": tmp_path / "out"}


def test_the_report_names_every_analysis_of_every_setting(finished_run: dict) -> None:
    report = finished_run["root"] / "post_rebuttal_results.md"
    assert report.exists(), "the report stage wrote no post_rebuttal_results.md"
    text = report.read_text()
    assert text.startswith("# Post-rebuttal measurements\n")
    assert "Missing" not in text, "an analysis produced no report"

    for name in ALL_ANALYSES:
        assert f"### {registry.question_for(name)}" in text, \
            f"{name} has no section in the report"
        assert f"`{name}`" in text, f"{name} is never named by its own file"

    # The inventory, then the two paper deliverables, then the two settings.
    assert "## Files this run produced" in text
    assert "## Table 1" in text
    assert "## Figure 2" in text
    assert "## Setting coco_k8" in text
    assert "## Setting cc3m_k32" in text
    assert text.count("\n# ") == 0, "an inlined report kept its own level-1 title"
    assert "—" not in text, "an em dash reached the report"


def test_no_analysis_recorded_a_failure(finished_run: dict) -> None:
    """A failing analysis leaves <name>.error.txt behind; none may exist."""
    errors = sorted(p.as_posix() for p in finished_run["root"].rglob("*.error.txt"))
    assert errors == [], f"analyses failed: {errors}"


def test_every_analysis_wrote_its_json_and_its_report(finished_run: dict) -> None:
    root = finished_run["root"] / "rebuttal"
    for analysis in registry.ANALYSES:
        for tag in analysis.settings:
            out_dir = root / tag
            assert (out_dir / f"{analysis.name}.json").exists(), \
                f"{tag}/{analysis.name}.json is missing"
            assert (out_dir / f"{analysis.name}.md").exists(), \
                f"{tag}/{analysis.name}.md is missing"


def test_the_extra_panels_exist(finished_run: dict) -> None:
    """Two same-modality panels, one different-caption panel, one noise floor."""
    root = finished_run["root"]
    panels = root / "rebuttal" / "coco_k8" / "panels"
    for name in ("img_img.npz", "txt_txt.npz", "txt_txt_diffcap.npz",
                 "img_txt_null.npz"):
        assert (panels / name).exists(), f"{name} was not built"

    null_meta = load_panel(panels / "img_txt_null.npz")["_meta"]
    assert null_meta["pairing"] == "img_txt"
    assert null_meta["shuffle_seed"] == finished_run["cfg"].rebuttal.null_seed
    same_meta = load_panel(panels / "img_img.npz")["_meta"]
    assert same_meta["pairing"] == "img_img"
    assert same_meta["shuffle_seed"] == 0
    assert same_meta["max_samples"] == 0

    # CC3M carries one caption per image, so it gets no different-caption panel.
    cc3m_panels = root / "rebuttal" / "cc3m_k32" / "panels"
    assert (cc3m_panels / "img_img.npz").exists()
    assert (cc3m_panels / "txt_txt.npz").exists()
    assert (cc3m_panels / "img_txt_null.npz").exists()
    assert not (cc3m_panels / "txt_txt_diffcap.npz").exists()


def test_the_two_paper_deliverables_were_produced(finished_run: dict) -> None:
    root = finished_run["root"]
    assert (root / "cc3m_clip_b32" / "table1.md").exists()
    assert (root / "cc3m_clip_b32" / "table1.tex").exists()
    assert (root / "coco_clip_b32" / "multi_density.pdf").exists()
    assert (root / "coco_clip_b32" / "figure2_bin_stats.md").exists()
    # The second COCO model is the rebuttal stage's own training run.
    assert (root / "rebuttal" / "coco_k8" / "seed1" / "final" / "config.json").exists()


def test_the_run_records_which_checkpoints_it_read(finished_run: dict) -> None:
    manifest = json.loads(
        (finished_run["root"] / "rebuttal" / "coco_k8" / "setting.json").read_text())
    assert manifest["tag"] == "coco_k8"
    assert manifest["k"] == K
    assert manifest["ckpt_a"].endswith("clip_b32/final")


def test_rerunning_every_stage_rebuilds_nothing(finished_run: dict, monkeypatch) -> None:
    """Idempotence: the second pass must touch no finished file."""
    monkeypatch.chdir(finished_run["tmp"])
    _point_the_annotations_at_the_fixture(monkeypatch, finished_run["tmp"])
    root = finished_run["root"]
    watched = [
        root / "rebuttal" / "coco_k8" / "match_confidence.json",
        root / "rebuttal" / "cc3m_k32" / "alignment_methods.json",
        root / "rebuttal" / "coco_k8" / "panels" / "img_img.npz",
        root / "cc3m_clip_b32" / "seed0" / "separated" / "final" / "config.json",
    ]
    before = {p: p.stat().st_mtime_ns for p in watched}
    post_rebuttal_run(finished_run["cfg"], stage="all")
    for path, stamp in before.items():
        assert path.stat().st_mtime_ns == stamp, f"{path} was rebuilt"


# --------------------------------------------------------------------------- #
# failure handling
# --------------------------------------------------------------------------- #
def test_one_failing_analysis_does_not_stop_the_others(tmp_path: Path,
                                                       monkeypatch) -> None:
    """The traceback is recorded, the rest still run, the process exits non-zero."""
    ran: list[str] = []

    def fails(setting: Setting, *, out_dir, device, **knobs) -> dict:
        raise RuntimeError("this analysis cannot be computed here")

    def works(setting: Setting, *, out_dir, device, **knobs) -> dict:
        ran.append(setting.tag)
        (Path(out_dir) / "works.json").write_text('{"rows": 0}')
        (Path(out_dir) / "works.md").write_text(
            "# Works\n\nMeasured nothing, over 0 rows.\n")
        return {"rows": 0}

    for name, fn in (("tests_fails", fails), ("tests_works", works)):
        module = types.ModuleType(name)
        module.run = fn
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(registry, "ANALYSES", [
        Analysis("fails", "tests_fails", ("coco_k8",)),
        Analysis("works", "tests_works", ("coco_k8",)),
    ])

    monkeypatch.chdir(tmp_path)
    _write_caches(tmp_path)
    cfg = _tiny_config(tmp_path)
    _place_checkpoints(cfg)
    with pytest.raises(SystemExit, match="coco_k8/fails"):
        post_rebuttal_run(cfg, stage="rebuttal")

    out_dir = tmp_path / "out" / "rebuttal" / "coco_k8"
    error = out_dir / "fails.error.txt"
    assert error.exists(), "the traceback was not recorded"
    assert "this analysis cannot be computed here" in error.read_text()
    assert ran == ["coco_k8"], "the analysis after the failing one did not run"
    assert (out_dir / "works.json").exists()


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


def test_the_report_inlines_each_analysis_report_under_its_question(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(registry, "ANALYSES", [
        Analysis("match_confidence", "src.rebuttal.match_confidence", ("coco_k8",)),
    ])
    cfg = _tiny_config(tmp_path)
    out_dir = tmp_path / "out" / "rebuttal" / "coco_k8"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "match_confidence.md").write_text(
        "# Match confidence\n\nMeasured over 512 rows.\n\n## Results\n\nnothing\n")
    post_rebuttal_run(cfg, stage="report")
    text = (tmp_path / "out" / "post_rebuttal_results.md").read_text()
    assert text.startswith("# Post-rebuttal measurements\n")
    assert text.count("\n# ") == 0          # no inlined report kept its own title
    assert f"### {registry.question_for('match_confidence')}" in text
    assert "#### Results" in text           # its own sections pushed under it
    assert "Measured over 512 rows." in text


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_caches(tmp_path: Path) -> None:
    """The three embedding caches and the fixture object annotations."""
    make_coco_cache(tmp_path / "cache" / "clip_b32_coco", n_images=COCO_IMAGES,
                    caps_per_image=COCO_CAPS, dim=DIM, seed=0,
                    n_val_images=2, n_test_images=COCO_TEST_IMAGES)
    make_coco_cache(tmp_path / "cache" / "clip_b32_cc3m", n_images=CC3M_IMAGES,
                    caps_per_image=1, dim=DIM, seed=1)
    make_imagenet_cache(tmp_path / "cache" / "clip_b32_imagenet", n_images=12,
                        n_classes=3, n_templates=4, dim=DIM)
    annotations = tmp_path / "cache" / "coco_annotations"
    for name in ("instances_val2014.json", "instances_train2014.json"):
        make_coco_instances(annotations / name, n_images=COCO_IMAGES)


def _point_the_annotations_at_the_fixture(monkeypatch, tmp_path: Path) -> None:
    """Replace the COCO annotation download with the fixture files on disk."""
    annotations = tmp_path / "cache" / "coco_annotations"

    def fixture_annotations(cache_dir="cache/coco_annotations") -> dict[str, Path]:
        return {name: annotations / name
                for name in ("instances_val2014.json", "instances_train2014.json")}

    monkeypatch.setattr("src.rebuttal.coco80_labels.ensure_coco_annotations",
                        fixture_annotations)


def _tiny_config(tmp_path: Path) -> Config:
    """A post-rebuttal config over the fixture caches, everything on the CPU."""
    model = ModelConfig(key="clip_b32", backend="transformers", hidden_size=DIM)
    training = dict(lr=1e-3, num_epochs=1, batch_size=64, k=K,
                    latent_size=LATENT_TOTAL, warmup_ratio=0.5, device="cpu")

    figure2 = Config(
        kind="multi_density", models=[model],
        cache=CacheConfig(cache_dir="cache/{key}_coco", dataset="coco", split="train"),
        training=TrainingConfig(seed=0, seeds=[0], **training),
        output=OutputConfig(root=str(tmp_path / "out" / "coco_clip_b32")),
    )
    table1 = Config(
        kind="cc3m_downstream", model=model,
        cache=CacheConfig(cache_dir="cache/clip_b32_cc3m", dataset="cc3m",
                          split="train"),
        training=TrainingConfig(seed=0, seeds=[0, 1], **training),
        methods=[MethodConfig(name=n) for n in
                 ("shared", "separated", "iso_align", "group_sparse", "ours")],
        eval=EvalConfig(recon=True, retrieval=True, zeroshot=True,
                        zeroshot_variant="raw"),
        output=OutputConfig(root=str(tmp_path / "out" / "cc3m_clip_b32")),
    )
    return Config(
        kind="post_rebuttal",
        output=OutputConfig(root=str(tmp_path / "out")),
        rebuttal=RebuttalConfig(tau=0.4, null_seed=7, n_boot=20,
                                settings=["coco_k8", "cc3m_k32"],
                                analyses=["all"], coco_seed_b=1),
        figure2=figure2, table1=table1,
    )


def _place_checkpoints(cfg: Config) -> None:
    """Save the checkpoints the failure test needs, without training anything."""
    for seed, path in enumerate([
        Path(cfg.figure2.output.root) / "clip_b32" / "final",
        Path(cfg.table1.output.root) / "seed0" / "separated" / "final",
        Path(cfg.table1.output.root) / "seed1" / "separated" / "final",
    ]):
        make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K,
                           seed=seed).save_pretrained(path)

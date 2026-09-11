"""The shared ground under every rebuttal analysis.

Covers the summary statistics, the bootstrap, the markdown helpers, and the
setting builder reading the real post-rebuttal config, because that is the one
place where a path typo would send every analysis at a checkpoint that does not
exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.rebuttal.common import (
    SETTING_TAGS,
    bootstrap_ci,
    describe,
    fmt,
    load_panel_or_raise,
    matched_distance,
    md_table,
    panel_build_command,
    pct,
    settings_from_config,
    unit_decoder,
    write_json,
    write_md,
)
from src.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #
def test_describe_reports_the_six_percentiles_and_the_count() -> None:
    values = np.arange(101, dtype=np.float64)  # 0 .. 100
    d = describe(values)
    assert d["n"] == 101
    assert d["mean"] == pytest.approx(50.0)
    assert d["median"] == pytest.approx(50.0)
    assert d["p05"] == pytest.approx(5.0)
    assert d["p25"] == pytest.approx(25.0)
    assert d["p75"] == pytest.approx(75.0)
    assert d["p95"] == pytest.approx(95.0)


def test_describe_of_nothing_reports_only_the_count() -> None:
    """An empty input must not hand back a mean that was never computed."""
    assert describe(np.array([])) == {"n": 0}


# --------------------------------------------------------------------------- #
# bootstrap_ci
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=400)
    point, lo, hi = bootstrap_ci(values, stat=np.median, n_boot=500, seed=0)
    assert point == pytest.approx(float(np.median(values)))
    assert lo < point < hi
    assert hi - lo < 0.05


def test_bootstrap_ci_is_reproducible_and_seed_dependent() -> None:
    values = np.linspace(0.0, 1.0, 200)
    a = bootstrap_ci(values, n_boot=200, seed=1)
    b = bootstrap_ci(values, n_boot=200, seed=1)
    c = bootstrap_ci(values, n_boot=200, seed=2)
    assert a == b
    assert a[0] == c[0]          # the point estimate does not depend on the seed
    assert a[1:] != c[1:]        # the interval does


def test_bootstrap_ci_takes_the_mean_as_well_as_the_median() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0])
    med, _, _ = bootstrap_ci(values, stat=np.median, n_boot=100, seed=0)
    mean, _, _ = bootstrap_ci(values, stat=np.mean, n_boot=100, seed=0)
    assert med == pytest.approx(0.0)
    assert mean == pytest.approx(2.5)


def test_bootstrap_ci_of_one_value_has_no_interval() -> None:
    point, lo, hi = bootstrap_ci([0.3], n_boot=100, seed=0)
    assert point == pytest.approx(0.3)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_of_nothing_is_all_nan() -> None:
    point, lo, hi = bootstrap_ci([], n_boot=100, seed=0)
    assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# matched_distance
# --------------------------------------------------------------------------- #
def test_matched_distance_uses_only_the_usable_rows() -> None:
    Wa = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    Wb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    perm = np.array([0, 1, 2])
    usable = np.array([True, False, True])
    d = matched_distance(Wa, Wb, perm, usable)
    # Row 0 is its partner exactly (distance 0); row 2 is orthogonal to its
    # partner (distance 1); row 1 is excluded.
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# markdown helpers
# --------------------------------------------------------------------------- #
def test_md_table_renders_a_header_a_rule_and_one_line_per_row() -> None:
    table = md_table(["Comparison", "Latent pairs (count)"],
                     [["image against text", "73"], ["image against image", "513"]])
    lines = table.splitlines()
    assert lines[0] == "| Comparison | Latent pairs (count) |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| image against text | 73 |"
    assert len(lines) == 4


def test_md_table_refuses_a_row_of_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="row 0 has 1 cells"):
        md_table(["a", "b"], [["only one"]])


def test_fmt_and_pct_never_print_a_nan_or_a_vanishing_percentage() -> None:
    assert fmt(0.12345) == "0.123"
    assert fmt(0.12345, nd=1) == "0.1"
    assert fmt(1234) == "1,234"
    assert fmt(float("nan")) == "n/a"
    assert fmt(None) == "n/a"
    assert pct(0.692) == "69.2%"
    assert pct(0.002) == "0.200%"     # a chance rate stays visible
    assert pct(0.0) == "0%"
    assert pct(None) == "n/a"


def test_write_md_puts_one_top_heading_over_the_paragraphs_and_tables(tmp_path: Path) -> None:
    table = md_table(["Quantity", "Value"], [["rows measured", "512"]])
    path = write_md(tmp_path / "demo.md", "Demo measurement",
                    ["Measured over 512 rows."], [("Results", table)])
    text = path.read_text()
    assert text.startswith("# Demo measurement\n")
    assert text.count("\n# ") == 0        # exactly one level-1 heading
    assert "## Results" in text
    assert "Measured over 512 rows." in text
    assert "| rows measured | 512 |" in text
    assert "—" not in text                # no em dashes anywhere


def test_write_json_serializes_numpy_scalars(tmp_path: Path) -> None:
    import json

    path = write_json(tmp_path / "demo.json",
                      {"n": np.int64(7), "median": np.float32(0.5),
                       "values": np.array([1.0, 2.0])})
    payload = json.loads(path.read_text())
    assert payload == {"n": 7, "median": 0.5, "values": [1.0, 2.0]}


# --------------------------------------------------------------------------- #
# unit_decoder
# --------------------------------------------------------------------------- #
def test_unit_decoder_returns_unit_norm_rows(tmp_path: Path) -> None:
    from tests.conftest import make_two_sided_sae

    model = make_two_sided_sae(dim=16, latent_size=8, k=2)
    ckpt = tmp_path / "final"
    model.save_pretrained(ckpt)
    for side in ("image", "text"):
        W = unit_decoder(ckpt, side)
        assert W.shape == (4, 16)          # total budget 8, so 4 per side
        assert np.allclose(np.linalg.norm(W, axis=1), 1.0, atol=1e-5)


def test_unit_decoder_refuses_a_side_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="image.*text"):
        unit_decoder(tmp_path, "vision")


# --------------------------------------------------------------------------- #
# settings_from_config
# --------------------------------------------------------------------------- #
def test_settings_from_the_shipped_config_name_both_trained_points() -> None:
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    assert cfg.kind == "post_rebuttal"
    settings = settings_from_config(cfg)
    assert list(settings) == list(SETTING_TAGS)

    coco = settings["coco_k8"]
    assert coco.dataset == "coco"
    assert coco.split == "train"
    assert coco.k == 8 and coco.num_epochs == 30 and coco.latent_size == 8192
    assert coco.latents_per_side == 4096
    assert str(coco.cache_dir) == "cache/clip_b32_coco"
    assert str(coco.ckpt_a) == "outputs/post_rebuttal/coco_clip_b32/clip_b32/final"
    assert str(coco.ckpt_b) == "outputs/post_rebuttal/rebuttal/coco_k8/seed1/final"
    assert str(coco.panel_img_txt) == "outputs/post_rebuttal/coco_clip_b32/clip_b32/panel.npz"
    assert str(coco.panels_dir) == "outputs/post_rebuttal/rebuttal/coco_k8/panels"
    assert str(coco.out_dir) == "outputs/post_rebuttal/rebuttal/coco_k8"
    assert coco.baselines == {}

    cc3m = settings["cc3m_k32"]
    assert cc3m.dataset == "cc3m"
    assert cc3m.k == 32 and cc3m.num_epochs == 10
    assert str(cc3m.cache_dir) == "cache/clip_b32_cc3m"
    assert str(cc3m.ckpt_a) == "outputs/post_rebuttal/cc3m_clip_b32/seed0/separated/final"
    assert str(cc3m.ckpt_b) == "outputs/post_rebuttal/cc3m_clip_b32/seed1/separated/final"
    assert str(cc3m.panel_img_txt) == "outputs/post_rebuttal/cc3m_clip_b32/seed0/ours/panel.npz"
    assert str(cc3m.coco_cache) == "cache/clip_b32_coco"
    assert set(cc3m.baselines) == {"shared", "iso_align", "group_sparse", "noalign"}
    assert str(cc3m.baselines["shared"]) == \
        "outputs/post_rebuttal/cc3m_clip_b32/seed0/shared/final"
    # "noalign" is model A scored with the identity permutation, so it points
    # at the same checkpoint rather than at one of its own.
    assert cc3m.baselines["noalign"] == cc3m.ckpt_a


def test_settings_honour_the_configured_subset() -> None:
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    cfg.rebuttal.settings = ["cc3m_k32"]
    assert list(settings_from_config(cfg)) == ["cc3m_k32"]


def test_setting_panel_path_points_at_the_right_file() -> None:
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    coco = settings_from_config(cfg)["coco_k8"]
    assert coco.panel_path("img_txt") == coco.panel_img_txt
    assert coco.panel_path("img_img") == coco.panels_dir / "img_img.npz"
    assert coco.panel_path("txt_txt_diffcap") == coco.panels_dir / "txt_txt_diffcap.npz"


def test_a_setting_tag_nobody_defined_is_refused(monkeypatch) -> None:
    """A typo in the config must not read as a deliberately empty run.

    An unknown analysis name already raises; an unknown setting used to be
    dropped in silence, which ended the whole rebuttal stage successfully with
    nothing measured and no section in the report.
    """
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    cfg.rebuttal.settings = ["coco_k9"]
    with pytest.raises(ValueError, match="unknown settings.*coco_k9"):
        settings_from_config(cfg)


# --------------------------------------------------------------------------- #
# load_panel_or_raise
# --------------------------------------------------------------------------- #
def test_a_missing_panel_names_the_stage_that_actually_builds_it() -> None:
    """The rebuttal stage builds the extra panels and nothing else.

    The image-to-text panel is written by the stage that produced the
    deliverable it belongs to, so pointing a reader at the rebuttal stage for a
    missing one sends them round a loop that rebuilds nothing.
    """
    extra = Path("outputs/post_rebuttal/rebuttal/cc3m_k32/panels/img_img.npz")
    assert panel_build_command(extra).endswith("--stage rebuttal")

    cc3m = Path("outputs/post_rebuttal/cc3m_clip_b32/seed0/ours/panel.npz")
    assert panel_build_command(cc3m).endswith("--stage table1")

    coco = Path("outputs/post_rebuttal/coco_clip_b32/clip_b32/panel.npz")
    assert panel_build_command(coco).endswith("--stage figure2")


def test_a_missing_panel_raises_with_that_command_in_the_message(
        tmp_path: Path) -> None:
    missing = tmp_path / "cc3m_clip_b32" / "seed0" / "ours" / "panel.npz"
    with pytest.raises(FileNotFoundError) as caught:
        load_panel_or_raise(missing)
    message = str(caught.value)
    assert "--stage table1" in message
    assert "--stage rebuttal" not in message


def test_settings_need_both_pipeline_configs() -> None:
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    cfg.table1 = None
    with pytest.raises(ValueError, match="figure2 and table1"):
        settings_from_config(cfg)


# --------------------------------------------------------------------------- #
# ensure_coco_annotations
# --------------------------------------------------------------------------- #
def test_ensure_coco_annotations_downloads_nothing_when_both_files_are_there(
        tmp_path: Path, monkeypatch) -> None:
    """The 241 MB download must not be repeated on every run of an analysis."""
    from src.rebuttal import common

    def explode(*_args, **_kwargs):
        raise AssertionError("the annotations were already on disk")

    monkeypatch.setattr(common, "_download_resumable", explode)
    ann_dir = tmp_path / "coco_annotations"
    ann_dir.mkdir()
    for name in common.COCO_INSTANCE_FILES:
        (ann_dir / name).write_text('{"images": [], "annotations": []}')
    paths = common.ensure_coco_annotations(ann_dir)
    assert set(paths) == set(common.COCO_INSTANCE_FILES)
    assert all(p.exists() for p in paths.values())


def test_ensure_coco_annotations_unpacks_an_already_downloaded_zip(
        tmp_path: Path, monkeypatch) -> None:
    """Only the validation instances are pulled out, under the directory itself.

    The archive is deleted once it has been unpacked and the 333 MB training
    instances are never written, because nothing in this repository opens
    either of them and the operating guide sizes the disk for what is kept.
    """
    import zipfile

    from src.rebuttal import common

    monkeypatch.setattr(common, "_download_resumable", lambda *a, **k: None)
    ann_dir = tmp_path / "coco_annotations"
    ann_dir.mkdir()
    zip_path = ann_dir / "annotations_trainval2014.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in ("instances_val2014.json", "instances_train2014.json"):
            zf.writestr(f"annotations/{name}", '{"images": []}')
        zf.writestr("annotations/captions_val2014.json", "{}")
    paths = common.ensure_coco_annotations(ann_dir)
    assert (ann_dir / "instances_val2014.json").read_text() == '{"images": []}'
    assert not (ann_dir / "captions_val2014.json").exists()
    assert not (ann_dir / "instances_train2014.json").exists()
    assert not zip_path.exists(), "the archive was kept after it was unpacked"
    assert set(paths) == {"instances_val2014.json"}


# --------------------------------------------------------------------------- #
# the shipped config
# --------------------------------------------------------------------------- #
def test_the_post_rebuttal_config_carries_the_stated_knobs() -> None:
    cfg = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    assert cfg.rebuttal.tau == pytest.approx(0.4)
    assert cfg.rebuttal.null_seed == 7
    # 2000 is the number the paper's own rebuttal scripts drew.
    assert cfg.rebuttal.n_boot == 2000
    # Left unset, so each analysis keeps the batch size its own author chose.
    assert cfg.rebuttal.batch_size is None
    assert cfg.rebuttal.coco_seed_b == 1
    assert cfg.rebuttal.analyses == ["all"]
    assert cfg.output.root == "outputs/post_rebuttal"
    # The two nested configs are parsed exactly as they would be on their own.
    assert cfg.figure2.kind == "multi_density"
    assert cfg.figure2.training.k == 8
    assert cfg.table1.kind == "cc3m_downstream"
    assert cfg.table1.training.resolved_seeds() == [0, 1, 2]

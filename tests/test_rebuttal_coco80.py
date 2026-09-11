"""The three COCO-80 label analyses, on fixture annotations and a fixture cache.

Nothing here downloads anything: `annotations_dir` is pointed at a directory
holding the fixture `instances_val2014.json`, so `ensure_coco_annotations` finds
both files already present and returns immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.alignment.panel import save_panel
from src.rebuttal import coco80_correspondence as corr
from src.rebuttal import coco80_heterogeneity as het
from src.rebuttal import coco80_labels as lab
from src.rebuttal.coco80_synonyms import COCO_80
from src.rebuttal.common import Setting
from tests.conftest import (
    COCO_TEST_CATEGORIES,
    make_coco_cache,
    make_coco_instances,
    make_two_sided_sae,
)

#: Every photograph of the fixture cache is in the test split of this build, so
#: the label matrix is not left empty by the split restriction. Forty
#: photographs is enough for several object categories to have a positive in
#: both halves, which the comparison across categories needs.
N_IMAGES = 40
CAPS_PER_IMAGE = 3
DIM = 16
LATENT_TOTAL = 16
K = 2


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _all_in_test_split(cache_dir: Path) -> None:
    """Move every key of the fixture cache into the test split.

    The COCO-80 analyses read the COCO test split, and `make_coco_cache` puts
    most of its photographs into train, which would leave two photographs to
    measure. Rewriting splits.json is simpler than reproducing the cache writer.
    """
    path = cache_dir / "splits.json"
    splits = json.loads(path.read_text())
    every = [k for keys in splits.values() for k in keys]
    path.write_text(json.dumps({"train": [], "val": [], "test": every}))


@pytest.fixture
def coco_world(tmp_path: Path) -> dict:
    """A COCO cache whose photographs all carry fixture annotations."""
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=N_IMAGES, caps_per_image=CAPS_PER_IMAGE,
                    dim=DIM, seed=0)
    _all_in_test_split(cache_dir)
    annotations = tmp_path / "annotations"
    make_coco_instances(annotations / "instances_val2014.json", n_images=N_IMAGES)
    # ensure_coco_annotations wants both instance files present before it will
    # skip the download; the second one is never read by these analyses.
    make_coco_instances(annotations / "instances_train2014.json", n_images=N_IMAGES)
    return {"cache_dir": cache_dir, "annotations": annotations, "tmp": tmp_path}


def _setting(world: dict, *, out_dir: Path, with_baselines: bool) -> Setting:
    """A Setting whose checkpoints and panel exist, over the fixture cache."""
    tmp = world["tmp"]
    ckpt_a = tmp / "models" / "a" / "final"
    ckpt_b = tmp / "models" / "b" / "final"
    model_a = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=0)
    model_a.save_pretrained(ckpt_a)
    make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=1).save_pretrained(ckpt_b)

    per_side = LATENT_TOTAL // 2
    panel_path = tmp / "panels" / "panel.npz"
    rng = np.random.default_rng(0)
    save_panel(panel_path, {
        "C": rng.normal(size=(per_side, per_side)).astype(np.float32),
        "perm": np.roll(np.arange(per_side, dtype=np.int64), 1),
        "usable": np.ones(per_side, dtype=bool),
        "alive_image": np.ones(per_side, dtype=bool),
        "alive_text": np.ones(per_side, dtype=bool),
        "fire_count_image": np.ones(per_side, dtype=np.int64),
        "fire_count_text": np.ones(per_side, dtype=np.int64),
        "rate_image": np.full(per_side, 0.5),
        "rate_text": np.full(per_side, 0.5),
        "n_samples": np.int64(N_IMAGES * CAPS_PER_IMAGE),
        "_meta": {"pairing": "img_txt", "split": "test"},
    })

    baselines: dict[str, Path] = {}
    if with_baselines:
        from src.models import TopKSAE, TopKSAEConfig

        for name in ("shared", "iso_align", "group_sparse"):
            path = tmp / "models" / name / "final"
            TopKSAE(TopKSAEConfig(hidden_size=DIM, latent_size=per_side, k=K,
                                  normalize_decoder=True)).save_pretrained(path)
            baselines[name] = path
        baselines["noalign"] = ckpt_a

    out_dir.mkdir(parents=True, exist_ok=True)
    return Setting(
        tag="cc3m_k32" if with_baselines else "coco_k8",
        dataset="coco",
        cache_dir=world["cache_dir"],
        split="test",
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        panel_img_txt=panel_path,
        panels_dir=tmp / "panels",
        out_dir=out_dir,
        baselines=baselines,
        coco_cache=world["cache_dir"],
        k=K,
        latent_size=LATENT_TOTAL,
        num_epochs=1,
    )


def _knobs(world: dict, **extra) -> dict:
    """The knobs every call in this file shares: no download, tiny thresholds."""
    knobs = {
        "annotations_dir": world["annotations"],
        "min_count": 1,
        "min_support": 0.0,
        "n_null": 16,
        "n_boot": 16,
        "batch_size": 8,
        "tau": 0.4,
        "null_seed": 7,
    }
    knobs.update(extra)
    return knobs


# --------------------------------------------------------------------------- #
# (a) the label matrix
# --------------------------------------------------------------------------- #
def test_the_label_matrix_covers_the_test_split_photographs(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_labels"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    labels = lab.load_or_build(setting, out_dir, min_count=1,
                               annotations_dir=coco_world["annotations"])

    assert labels.n_images == N_IMAGES
    assert labels.n_captions == N_IMAGES * CAPS_PER_IMAGE
    assert labels.area.shape == (N_IMAGES, 80)
    assert labels.no_area.shape == (N_IMAGES, 80)
    assert sorted(labels.image_ids.tolist()) == list(range(N_IMAGES))
    # Each photograph carries two annotations in the fixture, one covering 25
    # percent of the frame and one covering 1 percent, so the area condition
    # keeps exactly one of the two.
    assert labels.no_area.sum(axis=1).tolist() == [2] * N_IMAGES
    assert labels.area.sum(axis=1).tolist() == [1] * N_IMAGES
    # Only categories the fixture annotation file names can ever be positive.
    named = {name for _cid, name in COCO_TEST_CATEGORIES}
    positive = {COCO_80[c] for c in np.where(labels.no_area.any(axis=0))[0]}
    assert positive <= named


def test_the_halves_are_stable_and_split_the_photographs(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_halves"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    labels = lab.load_or_build(setting, out_dir, min_count=1,
                               annotations_dir=coco_world["annotations"])
    assert set(labels.half.tolist()) <= {0, 1}
    assert lab.image_half(7) == lab.image_half(7)
    assert [lab.image_half(int(i)) for i in labels.image_ids] == labels.half.tolist()


def test_the_labels_are_cached_and_reread_identically(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_cache"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    first = lab.load_or_build(setting, out_dir, min_count=1,
                              annotations_dir=coco_world["annotations"])
    assert (out_dir / "coco80_labels.json").exists()
    second = lab.load_or_build(setting, out_dir, min_count=1,
                               annotations_dir=coco_world["annotations"])
    assert np.array_equal(first.area, second.area)
    assert np.array_equal(first.no_area, second.no_area)
    assert np.array_equal(first.half, second.half)
    assert np.array_equal(first.caption_rows, second.caption_rows)
    assert first.caption_text == second.caption_text


def test_the_label_analysis_writes_its_json_and_its_report(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_run"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    payload = lab.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    assert (out_dir / "coco80_labels.json").exists()
    assert (out_dir / "coco80_labels.md").exists()
    assert payload["n_images"] == N_IMAGES
    assert payload["split"] == "test"
    assert set(payload["counts"]) == {"area_filtered", "no_area"}
    text = (out_dir / "coco80_labels.md").read_text()
    assert "—" not in text
    assert "COCO-80 object labels" in text


# --------------------------------------------------------------------------- #
# the separation statistic
# --------------------------------------------------------------------------- #
def test_a_perfect_ranking_scores_one() -> None:
    """A coordinate that fires on the positives and on nothing else."""
    n_samples = 10
    labels = np.zeros((n_samples, 1), dtype=bool)
    labels[:4, 0] = True
    samp = np.arange(4, dtype=np.int64)
    lat = np.zeros(4, dtype=np.int64)
    val = np.array([0.5, 0.6, 0.7, 0.8])
    auc, support = corr.auc_matrix(samp, lat, val, labels, n_samples, 1, 0.0)
    assert auc[0, 0] == pytest.approx(1.0)
    assert support[0, 0] == pytest.approx(1.0)


def test_a_reversed_ranking_scores_zero() -> None:
    """The same coordinate firing on the negatives instead."""
    n_samples = 10
    labels = np.zeros((n_samples, 1), dtype=bool)
    labels[:4, 0] = True
    samp = np.arange(4, 10, dtype=np.int64)
    lat = np.zeros(6, dtype=np.int64)
    val = np.linspace(0.5, 1.0, 6)
    auc, _ = corr.auc_matrix(samp, lat, val, labels, n_samples, 1, 0.0)
    assert auc[0, 0] == pytest.approx(0.0)


def test_ties_score_one_half() -> None:
    """Two ways for every comparison to be a tie, both scoring 0.5."""
    n_samples = 10
    labels = np.zeros((n_samples, 1), dtype=bool)
    labels[:4, 0] = True

    # A coordinate that never fires: every sample ties at zero.
    empty = np.zeros(0, dtype=np.int64)
    auc, _ = corr.auc_matrix(empty, empty.copy(), np.zeros(0), labels, n_samples, 1, 0.0)
    assert auc[0, 0] == pytest.approx(0.5)

    # A coordinate that fires on every sample with the identical value.
    samp = np.arange(n_samples, dtype=np.int64)
    lat = np.zeros(n_samples, dtype=np.int64)
    val = np.full(n_samples, 0.25)
    auc, _ = corr.auc_matrix(samp, lat, val, labels, n_samples, 1, 0.0)
    assert auc[0, 0] == pytest.approx(0.5)


def test_the_separation_score_is_the_pairwise_definition_with_ties_at_one_half() -> None:
    """Repeated activation values get half credit, not sort order.

    The paper's script counted a negative by its position in the stable sort,
    so a positive took either the full point or none against a negative holding
    the identical value. The rule here is the standard one, checked against the
    pairwise definition of the area under the ROC curve on inputs built to
    repeat values exactly.
    """
    def pairwise(scores: np.ndarray, y: np.ndarray) -> float:
        pos, neg = scores[y], scores[~y]
        u = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return float(u) / (len(pos) * len(neg))

    rng = np.random.default_rng(0)
    for _ in range(50):
        n_samples = 30
        dense = rng.choice([0.0, 0.0, 0.5, 0.5, 1.0, 2.0], size=n_samples)
        y = rng.integers(0, 2, size=n_samples).astype(bool)
        if y.all() or not y.any():
            continue
        firing = np.where(dense > 0)[0].astype(np.int64)
        auc, _ = corr.auc_matrix(firing, np.zeros(firing.size, dtype=np.int64),
                                 dense[firing], y[:, None], n_samples, 1, 0.0)
        assert auc[0, 0] == pytest.approx(pairwise(dense, y))


def test_a_coordinate_below_the_support_floor_is_taken_out_of_the_maximum() -> None:
    n_samples = 100
    labels = np.zeros((n_samples, 1), dtype=bool)
    labels[:50, 0] = True
    samp = np.array([0], dtype=np.int64)          # one positive out of fifty
    lat = np.zeros(1, dtype=np.int64)
    val = np.array([9.0])
    auc, support = corr.auc_matrix(samp, lat, val, labels, n_samples, 1, 0.05)
    assert support[0, 0] == pytest.approx(0.02)
    assert auc[0, 0] == -np.inf


def test_rank_of_counts_from_one_and_gives_ties_the_best_rank() -> None:
    values = np.array([0.9, 0.5, 0.9, 0.1])
    assert corr.rank_of(values, 0) == 1
    assert corr.rank_of(values, 2) == 1
    assert corr.rank_of(values, 1) == 3
    assert corr.rank_of(values, 3) == 4


# --------------------------------------------------------------------------- #
# (b) the agreement test
# --------------------------------------------------------------------------- #
def test_the_agreement_test_writes_one_entry_per_arm_and_variant(
        coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_corr"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=True)
    payload = corr.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))

    assert (out_dir / "coco80_correspondence.json").exists()
    assert (out_dir / "coco80_correspondence.md").exists()
    assert set(payload["arms"]) == {"ours", "noalign", "shared", "iso_align",
                                    "group_sparse"}
    for arm, entry in payload["arms"].items():
        assert set(entry["variants"]) == {"area_filtered", "no_area"}
        for variant in entry["variants"].values():
            assert variant["n_categories"] >= 1, f"{arm} scored no category"
            assert 0.0 <= variant["agree_at_1"] <= 1.0
            controls = variant["controls"]
            assert 0.0 <= controls["image_self_agreement"] <= 1.0
            assert controls["chance_hit_at_1"] > 0.0
    # The arm with no permutation reads coordinate i against coordinate i, so it
    # puts forward as many candidates as there are coordinates alive on both
    # sides, and the arm with the permutation uses the panel's usable rows.
    assert payload["arms"]["noalign"]["variants"]["area_filtered"][
        "n_candidate_coordinates"] == LATENT_TOTAL // 2

    text = (out_dir / "coco80_correspondence.md").read_text()
    assert "—" not in text
    assert "Agreement at rank 1, against every reference" in text
    assert "Every alignment arm on the identical test" in text


def test_the_agreement_test_runs_with_only_the_one_arm(coco_world, tmp_path) -> None:
    """A setting with no baselines measures the modality-specific arm alone."""
    out_dir = tmp_path / "out_corr_nobase"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    payload = corr.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    assert set(payload["arms"]) == {"ours"}
    assert (out_dir / "coco80_correspondence.md").exists()


def test_the_agreement_test_is_idempotent(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_corr_again"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    first = corr.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    stamp = (out_dir / "coco80_correspondence.json").stat().st_mtime_ns
    second = corr.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    assert (out_dir / "coco80_correspondence.json").stat().st_mtime_ns == stamp
    assert second["arms"].keys() == first["arms"].keys()


# --------------------------------------------------------------------------- #
# (c) heterogeneity from labels
# --------------------------------------------------------------------------- #
def test_the_heterogeneity_analysis_writes_every_comparison(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_het"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    payload = het.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))

    assert (out_dir / "coco80_heterogeneity.json").exists()
    assert (out_dir / "coco80_heterogeneity.md").exists()
    assert set(payload["variants"]) == {"area_filtered", "no_area"}
    head = payload["variants"]["area_filtered"]
    assert head["n_categories"] >= 1
    for key in ("within_image_two_halves", "cross_modal_same_category",
                "cross_modal_different_category", "random_unit_vectors",
                "coactivation_partner_of_the_same_image_coordinate"):
        entry = head["comparisons"].get(key)
        assert entry and entry["n"] > 0, f"{key} was not measured"
        assert -1.0001 <= entry["cosine_median"] <= 1.0001

    # Random unit vectors in a 16-dimensional space average to about zero.
    assert abs(head["comparisons"]["random_unit_vectors"]["cosine_mean"]) < 0.2
    text = (out_dir / "coco80_heterogeneity.md").read_text()
    assert "—" not in text
    assert "One row per object category" in text


def test_the_heterogeneity_analysis_says_why_no_category_survived(
        coco_world, tmp_path) -> None:
    """Demanding more positives than exist names the conditions, not a KeyError.

    Nothing may be written either, so that the next run starts from an empty
    directory rather than from a json that holds no measurement.
    """
    out_dir = tmp_path / "out_het_empty"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    with pytest.raises(ValueError, match="no object category survived"):
        het.run(setting, out_dir=out_dir, device="cpu",
                **_knobs(coco_world, min_count=10_000))
    assert not (out_dir / "coco80_heterogeneity.json").exists()
    assert not (out_dir / "coco80_heterogeneity.md").exists()


def test_the_heterogeneity_analysis_is_idempotent(coco_world, tmp_path) -> None:
    out_dir = tmp_path / "out_het_again"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    het.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    stamp = (out_dir / "coco80_heterogeneity.json").stat().st_mtime_ns
    het.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    assert (out_dir / "coco80_heterogeneity.json").stat().st_mtime_ns == stamp


def test_the_two_analyses_share_one_label_file(coco_world, tmp_path) -> None:
    """Both read the same cached labels rather than rebuilding their own."""
    out_dir = tmp_path / "out_shared_labels"
    setting = _setting(coco_world, out_dir=out_dir, with_baselines=False)
    corr.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    stamp = (out_dir / "coco80_labels.json").stat().st_mtime_ns
    het.run(setting, out_dir=out_dir, device="cpu", **_knobs(coco_world))
    assert (out_dir / "coco80_labels.json").stat().st_mtime_ns == stamp

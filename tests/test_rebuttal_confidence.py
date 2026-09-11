"""The three matching-confidence analyses, on panels small enough to check by hand.

The panels here are written by hand rather than built from a model, so that every
matched correlation is a number the test chose: that is what lets the band edges,
the cumulative shares and the noise floor be asserted exactly. One of them is a
stand-in for the shuffled panel the rebuttal stage builds, and it carries
deliberately different correlations from the real one, so a noise floor computed
on the wrong panel cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.alignment import save_panel
from src.rebuttal import confidence_ablation, correlation_bands, match_confidence
from src.rebuttal.common import Setting
from tests.conftest import make_coco_cache, make_two_sided_sae

#: Matched correlations of the real panel, one per latent, chosen so that four
#: different bands are occupied: [0.8, 1.0], [0.4, 0.6), [0.0, 0.2) and the row
#: for negative correlations.
REAL_DIAGONAL = (0.95, 0.50, 0.15, -0.05)

#: Matched correlations of the shuffled panel. Three are near zero and one is
#: moderate, which puts its 99th percentile well below the real panel's.
NULL_DIAGONAL = (0.01, 0.02, 0.03, 0.30)

DIM = 16
TOTAL_LATENTS = 8          # the TOTAL budget; a side holds four
LATENTS_PER_SIDE = TOTAL_LATENTS // 2
K = 2
N_SAMPLES = 400


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _panel_payload(diagonal, *, pairing: str, shuffle_seed: int, seed: int) -> dict:
    """A panel whose matched correlations are exactly `diagonal`.

    Off-diagonal correlations are drawn small and positive so that they cannot
    overtake a diagonal entry, and the assignment stored is the identity, which
    makes `C[i, perm[i]]` the diagonal by construction.
    """
    gen = np.random.default_rng(seed)
    n = len(diagonal)
    C = gen.uniform(0.0, 0.05, size=(n, n)).astype(np.float32)
    for i, value in enumerate(diagonal):
        C[i, i] = np.float32(value)
    fire = np.full(n, N_SAMPLES // 4, dtype=np.int64)
    return {
        "C": C,
        "perm": np.arange(n, dtype=np.int64),
        "usable": np.ones(n, dtype=bool),
        "alive_image": np.ones(n, dtype=bool),
        "alive_text": np.ones(n, dtype=bool),
        "fire_count_image": fire,
        "fire_count_text": fire,
        "rate_image": (fire / N_SAMPLES).astype(np.float64),
        "rate_text": (fire / N_SAMPLES).astype(np.float64),
        "n_samples": np.int64(N_SAMPLES),
        "_meta": {
            "pairing": pairing,
            "split": "train",
            "n_samples": N_SAMPLES,
            "max_samples": 0,
            "shuffle_seed": shuffle_seed,
            "alive_rule": "fire_count >= 1 on the full train split",
        },
    }


@pytest.fixture
def setting(tmp_path: Path) -> Setting:
    """A setting with a checkpoint, a COCO cache, a real panel and a null panel."""
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=8, caps_per_image=2, dim=DIM, seed=0)

    ckpt = tmp_path / "model_a" / "final"
    make_two_sided_sae(dim=DIM, latent_size=TOTAL_LATENTS, k=K, seed=0).save_pretrained(ckpt)

    panels_dir = tmp_path / "panels"
    panel_path = tmp_path / "panel.npz"
    save_panel(panel_path, _panel_payload(REAL_DIAGONAL, pairing="img_txt",
                                          shuffle_seed=0, seed=1))
    save_panel(panels_dir / "img_txt_null.npz",
               _panel_payload(NULL_DIAGONAL, pairing="img_txt", shuffle_seed=7, seed=2))

    return Setting(
        tag="coco_k8",
        dataset="coco",
        cache_dir=cache_dir,
        split="train",
        ckpt_a=ckpt,
        ckpt_b=tmp_path / "model_b" / "final",
        panel_img_txt=panel_path,
        panels_dir=panels_dir,
        out_dir=tmp_path / "out",
        baselines={},
        coco_cache=cache_dir,
        k=K,
        latent_size=TOTAL_LATENTS,
        num_epochs=1,
    )


def _run_match_confidence(setting: Setting) -> dict:
    return match_confidence.run(setting, out_dir=setting.out_dir, device="cpu",
                                tau=0.4, n_boot=16, null_seed=7)


def _band(rows: list[dict], label: str) -> dict:
    for row in rows:
        if row["band"] == label:
            return row
    raise AssertionError(f"no band {label!r} in {[r['band'] for r in rows]}")


# --------------------------------------------------------------------------- #
# match_confidence: files
# --------------------------------------------------------------------------- #
def test_match_confidence_writes_its_four_files(setting: Setting) -> None:
    payload = _run_match_confidence(setting)
    for suffix in (".json", ".md", ".pdf", ".png"):
        path = setting.out_dir / f"match_confidence{suffix}"
        assert path.exists(), f"match_confidence{suffix} was not written"
        assert path.stat().st_size > 0
    assert payload["n_matched_usable"] == len(REAL_DIAGONAL)
    text = (setting.out_dir / "match_confidence.md").read_text()
    assert "—" not in text, "the report must contain no em dash"
    assert text.startswith("# Match confidence, coco_k8")
    assert f"{N_SAMPLES:,}" in text


def test_match_confidence_is_idempotent(setting: Setting) -> None:
    """A second call returns what is on disk instead of measuring again."""
    _run_match_confidence(setting)
    out_json = setting.out_dir / "match_confidence.json"
    marked = json.loads(out_json.read_text())
    marked["analysis"] = "already on disk"
    out_json.write_text(json.dumps(marked))
    again = _run_match_confidence(setting)
    assert again["analysis"] == "already on disk"


# --------------------------------------------------------------------------- #
# match_confidence: bands
# --------------------------------------------------------------------------- #
def test_the_coarse_bands_are_the_figure_2_edges(setting: Setting) -> None:
    """Edges 0, 0.2, 0.4, 0.6, 0.8, 1.0, plus one row below zero, nothing dropped."""
    payload = _run_match_confidence(setting)
    rows = payload["correlation_bands_0p2"]
    assert [r["band"] for r in rows] == [
        "[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]",
        "below 0.0",
    ]
    assert _band(rows, "[0.8, 1.0]")["n_pairs"] == 1        # 0.95
    assert _band(rows, "[0.4, 0.6)")["n_pairs"] == 1        # 0.50
    assert _band(rows, "[0.0, 0.2)")["n_pairs"] == 1        # 0.15
    assert _band(rows, "below 0.0")["n_pairs"] == 1         # -0.05
    assert _band(rows, "[0.2, 0.4)")["n_pairs"] == 0
    assert _band(rows, "[0.6, 0.8)")["n_pairs"] == 0
    assert sum(r["n_pairs"] for r in rows) == len(REAL_DIAGONAL)
    assert sum(r["share_of_matched_pairs"] for r in rows) == pytest.approx(1.0)


def test_every_occupied_coarse_band_carries_a_cosine_distance(setting: Setting) -> None:
    """The distance is a real cosine distance of the checkpoint's own decoders."""
    payload = _run_match_confidence(setting)
    for row in payload["correlation_bands_0p2"]:
        if row["n_pairs"]:
            assert 0.0 <= row["cosine_distance_median"] <= 2.0
            assert 0.0 <= row["cosine_distance_mean"] <= 2.0
        else:
            assert "cosine_distance_median" not in row


def test_the_fine_bands_run_from_the_top_and_accumulate_to_one(setting: Setting) -> None:
    payload = _run_match_confidence(setting)
    rows = payload["correlation_bands_0p1"]
    assert rows[0]["band"] == "[0.9, 1.0]"
    assert rows[-1]["band"] == "below 0.0"
    assert [r["band"] for r in rows].count("[0.5, 0.6)") == 1
    assert _band(rows, "[0.9, 1.0]")["n_pairs"] == 1
    assert _band(rows, "[0.5, 0.6)")["n_pairs"] == 1
    assert _band(rows, "[0.1, 0.2)")["n_pairs"] == 1
    assert _band(rows, "below 0.0")["n_pairs"] == 1
    assert rows[-1]["cumulative_share_from_the_top"] == pytest.approx(1.0)
    # The cumulative column never decreases as the bands get weaker.
    shares = [r["cumulative_share_from_the_top"] for r in rows]
    assert shares == sorted(shares)


# --------------------------------------------------------------------------- #
# match_confidence: noise floor
# --------------------------------------------------------------------------- #
def test_the_noise_floor_is_computed_on_the_shuffled_panel_only(setting: Setting) -> None:
    """Reading the real panel instead would give a floor three times as high."""
    payload = _run_match_confidence(setting)
    floor = payload["noise_floor"]
    expected = float(np.percentile(np.array(NULL_DIAGONAL), 99))
    assert floor["floor"] == pytest.approx(expected)
    assert floor["n"] == len(NULL_DIAGONAL)
    assert floor["shuffle_seed"] == 7
    wrong = float(np.percentile(np.array(REAL_DIAGONAL), 99))
    assert floor["floor"] != pytest.approx(wrong)
    # 0.15 and -0.05 sit below the floor; 0.95 and 0.50 do not.
    assert floor["n_real_matches_below_floor"] == 2
    assert floor["share_of_real_matches_below_floor"] == pytest.approx(0.5)


def test_a_missing_null_panel_names_the_stage_that_builds_it(setting: Setting) -> None:
    (setting.panels_dir / "img_txt_null.npz").unlink()
    with pytest.raises(FileNotFoundError, match="--stage rebuttal"):
        _run_match_confidence(setting)


# --------------------------------------------------------------------------- #
# match_confidence: ambiguity and reciprocity
# --------------------------------------------------------------------------- #
def test_ambiguity_and_reciprocity_are_shares_of_the_matched_pairs(
        setting: Setting) -> None:
    """Three of the four assigned partners are the row's own highest candidate.

    The fourth is the row assigned a correlation of -0.05, which every small
    positive off-diagonal entry of that row beats, so that row's first choice
    lies elsewhere and the pair is not mutual.
    """
    payload = _run_match_confidence(setting)
    rec = payload["reciprocity"]
    assert rec["n_mutual_first_choice"] == 3
    assert rec["share_mutual_first_choice"] == pytest.approx(0.75)
    amb = payload["ambiguity"]
    assert amb["n"] == len(REAL_DIAGONAL)
    assert 0.0 <= amb["share_runner_up_within_10_percent"] <= 1.0
    assert amb["best_minus_second_best"]["n"] == len(REAL_DIAGONAL)


# --------------------------------------------------------------------------- #
# correlation_bands
# --------------------------------------------------------------------------- #
def test_correlation_bands_covers_every_latent_pair_with_no_filter(
        setting: Setting) -> None:
    payload = correlation_bands.run(setting, out_dir=setting.out_dir, device="cpu",
                                    tau=0.4, n_boot=16, null_seed=7)
    stats = payload["statistics"]
    assert stats["n_pairs"] == LATENTS_PER_SIDE * LATENTS_PER_SIDE
    assert payload["bin_edges"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert [b["bin"] for b in stats["bins"]] == [
        "[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0]",
    ]
    counted = sum(b["n"] for b in stats["bins"]) + stats["n_negative_correlation_excluded"]
    assert counted == stats["n_pairs"]
    # One entry of the hand-written matrix is 0.95 and one is 0.50.
    assert _bin(stats, "[0.8, 1.0]")["n"] == 1
    assert _bin(stats, "[0.4, 0.6)")["n"] == 1
    assert stats["n_negative_correlation_excluded"] == 1


def test_correlation_bands_writes_its_files(setting: Setting) -> None:
    correlation_bands.run(setting, out_dir=setting.out_dir, device="cpu",
                          tau=0.4, n_boot=16, null_seed=7)
    for suffix in (".json", ".md"):
        assert (setting.out_dir / f"correlation_bands{suffix}").exists()
    text = (setting.out_dir / "correlation_bands.md").read_text()
    assert "—" not in text
    assert "no alive mask" in text


def _bin(stats: dict, label: str) -> dict:
    for row in stats["bins"]:
        if row["bin"] == label:
            return row
    raise AssertionError(f"no bin {label!r}")


# --------------------------------------------------------------------------- #
# confidence_ablation
# --------------------------------------------------------------------------- #
def test_confidence_ablation_scores_every_cutoff_and_its_control(
        setting: Setting, monkeypatch) -> None:
    """Four latents is below the real floor of ten, so the floor is lowered here."""
    monkeypatch.setattr(confidence_ablation, "MIN_COORDINATES", 1)
    payload = confidence_ablation.run(setting, out_dir=setting.out_dir, device="cpu",
                                      tau=0.4, n_boot=16, null_seed=7)
    rows = payload["by_cutoff"]
    assert [r["cutoff"] for r in rows] == list(confidence_ablation.CUTOFFS)
    # Matched correlations 0.95, 0.50, 0.15 and -0.05 against the cutoffs
    # 0.0, 0.1, 0.2, 0.3, 0.4 and 0.6.
    counts = [r["n_coordinates"] for r in rows]
    assert counts == [3, 3, 2, 2, 2, 1]
    assert counts == sorted(counts, reverse=True)
    for row in rows:
        assert set(row["I2T"]) == {"R@1", "R@5", "R@10"}
        assert set(row["shuffled_partners"]) == {"T2I", "I2T"}
        for direction in ("T2I", "I2T"):
            for value in row[direction].values():
                assert 0.0 <= value <= 1.0


def test_confidence_ablation_does_not_score_a_cutoff_with_too_few_coordinates(
        setting: Setting) -> None:
    """The real floor of ten leaves nothing scorable on a four-latent panel."""
    payload = confidence_ablation.run(setting, out_dir=setting.out_dir, device="cpu",
                                      tau=0.4, n_boot=16, null_seed=7)
    for row in payload["by_cutoff"]:
        assert "I2T" not in row
        assert "fewer than 10 coordinates" in row["note"]


def test_confidence_ablation_writes_its_files(setting: Setting, monkeypatch) -> None:
    monkeypatch.setattr(confidence_ablation, "MIN_COORDINATES", 1)
    payload = confidence_ablation.run(setting, out_dir=setting.out_dir, device="cpu",
                                      tau=0.4, n_boot=16, null_seed=7)
    for suffix in (".json", ".md"):
        assert (setting.out_dir / f"confidence_ablation{suffix}").exists()
    text = (setting.out_dir / "confidence_ablation.md").read_text()
    assert "—" not in text
    assert "image-to-text recall at 1 (percent)" in text
    # Two photographs and four captions are held out by the fixture cache.
    assert payload["n_images"] == 2
    assert payload["n_captions"] == 4
    assert payload["split"] == "test"


def test_confidence_ablation_is_idempotent(setting: Setting, monkeypatch) -> None:
    monkeypatch.setattr(confidence_ablation, "MIN_COORDINATES", 1)
    confidence_ablation.run(setting, out_dir=setting.out_dir, device="cpu",
                            tau=0.4, n_boot=16, null_seed=7)
    out_json = setting.out_dir / "confidence_ablation.json"
    marked = json.loads(out_json.read_text())
    marked["analysis"] = "already on disk"
    out_json.write_text(json.dumps(marked))
    again = confidence_ablation.run(setting, out_dir=setting.out_dir, device="cpu",
                                    tau=0.4, n_boot=16, null_seed=7)
    assert again["analysis"] == "already on disk"

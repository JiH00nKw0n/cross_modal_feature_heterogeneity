"""Conditioning the cross-modal distance on cross-run stability, on tiny panels.

The arithmetic is short; the parts worth testing are the filters. A concept
enters only when it is alive on both sides of the image-to-image comparison and
usable in the image-to-text panel, the stability score is the cosine of the
assignment that maximizes total decoder similarity, and the block that needs
both endpoints of a pair to be reproducible appears only when the text-to-text
panel exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.alignment.panel import build_panel, load_panel, save_panel
from src.rebuttal import stability_conditioned as sc
from src.rebuttal.common import Setting, unit_decoder
from tests.conftest import make_coco_cache, make_two_sided_sae

DIM = 16
LATENT_TOTAL = 8  # 4 latents per side
K = 2


def _build_setting(tmp_path: Path, *, with_txt_txt: bool = True) -> Setting:
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=10, caps_per_image=2, dim=DIM, seed=5)

    model_a = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=0)
    model_b = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=1)
    ckpt_a = tmp_path / "model_a" / "final"
    ckpt_b = tmp_path / "model_b" / "final"
    model_a.save_pretrained(ckpt_a)
    model_b.save_pretrained(ckpt_b)

    panels_dir = tmp_path / "panels"
    panel_img_txt = tmp_path / "figure2" / "panel.npz"

    def _panel(path: Path, pairing: str, other) -> None:
        payload = build_panel(
            model=model_a, cache_dir=cache_dir, split="train", batch_size=32,
            device="cpu", max_samples=0, pairing=pairing, model_b=other,
            ckpt_a=str(ckpt_a), ckpt_b=str(ckpt_b) if other is not None else None,
        )
        save_panel(path, payload)

    _panel(panel_img_txt, "img_txt", None)
    _panel(panels_dir / "img_img.npz", "img_img", model_b)
    if with_txt_txt:
        _panel(panels_dir / "txt_txt.npz", "txt_txt", model_b)

    return Setting(
        tag="coco_k8",
        dataset="coco",
        cache_dir=cache_dir,
        split="train",
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        panel_img_txt=panel_img_txt,
        panels_dir=panels_dir,
        out_dir=tmp_path / "out",
        coco_cache=cache_dir,
        k=K,
        latent_size=LATENT_TOTAL,
        num_epochs=1,
    )


def test_it_writes_the_json_and_the_report(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    out = tmp_path / "out"
    report = sc.run(setting, out_dir=out, device="cpu", n_boot=25)

    assert (out / "stability_conditioned.json").exists()
    assert (out / "stability_conditioned.md").exists()
    assert np.isfinite(report["mean_stability"])
    assert report["co_activation_min"] == 0.6
    text = (out / "stability_conditioned.md").read_text()
    assert "—" not in text, "no em dash is allowed in a report"
    assert "img_txt" not in text, "the report must not print internal panel codes"


def test_a_concept_counts_only_when_it_is_usable_in_both_comparisons(
        tmp_path: Path) -> None:
    """The concept count is the stability pairs filtered by the panel's usable mask."""
    setting = _build_setting(tmp_path)
    report = sc.run(setting, out_dir=tmp_path / "out", device="cpu", n_boot=10)

    ii = load_panel(setting.panel_path("img_img"))
    it = load_panel(setting.panel_img_txt)
    Wa = unit_decoder(setting.ckpt_a, "image")
    Wb = unit_decoder(setting.ckpt_b, "image")
    stab = sc.geometry_stability(Wa, Wb,
                                 np.asarray(ii["alive_image"], dtype=bool),
                                 np.asarray(ii["alive_text"], dtype=bool))
    usable = np.asarray(it["usable"], dtype=bool)
    expected = int(usable[stab["rows"]].sum())

    assert report["n_concepts"] == expected
    assert report["n_concepts"] <= int(usable.sum())
    assert report["same_modality_distance_all"]["n"] == expected
    assert report["cross_modal_distance_all"]["n"] == expected


def test_stability_is_the_cosine_of_the_best_total_assignment(tmp_path: Path) -> None:
    """Two identical dictionaries have to score a stability of one everywhere."""
    Wa = np.eye(4)
    alive = np.ones(4, dtype=bool)
    out = sc.geometry_stability(Wa, Wa[[2, 0, 3, 1]], alive, alive)
    assert out["mean_stability"] == pytest.approx(1.0)
    np.testing.assert_array_equal(out["partner"], np.array([1, 3, 0, 2]))


def test_every_quantile_and_decile_cut_is_reported(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    report = sc.run(setting, out_dir=tmp_path / "out", device="cpu", n_boot=10)

    assert list(report["by_stability_quantile"]) == [
        "top_1pct", "top_5pct", "top_10pct", "top_25pct", "top_50pct", "top_100pct",
    ]
    counts = [e["n"] for e in report["by_stability_quantile"].values()]
    assert counts == sorted(counts), "a wider cut cannot hold fewer concepts"
    assert counts[-1] == report["n_concepts"]

    # Stability falls as the cut widens, because the concepts are ordered by it.
    medians = [e["stability_median"] for e in report["by_stability_quantile"].values()]
    assert medians == sorted(medians, reverse=True)

    assert report["by_decile"], "the decile curve must be reported"
    assert sum(e["n"] for e in report["by_decile"].values()) == report["n_concepts"]


def test_the_both_endpoint_block_needs_the_text_to_text_panel(tmp_path: Path) -> None:
    with_panel = _build_setting(tmp_path / "with")
    report = sc.run(with_panel, out_dir=tmp_path / "with" / "out", device="cpu",
                    n_boot=10)
    block = report["stable_and_corresponding"]
    assert block, "the block must be reported when the text-to-text panel exists"
    assert set(block["grid"]) == {
        f"stability>={s}, c>={c}"
        for s in sc.GRID_STABILITY for c in sc.GRID_CORRELATION
    }
    # The loosest grid cell still asks for a non-negative correlation, so it
    # holds at most the pairs that could be scored on both endpoints.
    unfiltered = block["grid"]["stability>=0.0, c>=0.0"]
    assert unfiltered["n"] <= block["n_pairs_scored"]
    assert block["grid"]["stability>=0.9, c>=0.6"]["n"] <= unfiltered["n"]

    without = _build_setting(tmp_path / "without", with_txt_txt=False)
    thin = sc.run(without, out_dir=tmp_path / "without" / "out", device="cpu",
                  n_boot=10)
    assert thin["stable_and_corresponding"] == {}
    text = (tmp_path / "without" / "out" / "stability_conditioned.md").read_text()
    assert "text-to-text panel is not available" in text


def test_the_pairing_rule_and_modality_shares_add_up_to_the_whole_gap(
        tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    report = sc.run(setting, out_dir=tmp_path / "out", device="cpu", n_boot=10)
    ov = report["operator_vs_modality"]
    if ov["attributable_to_operator"] is None:
        pytest.skip("no concept had a partner under both pairing rules")
    total = ov["cross_modal_matched_by_coactivation"] - ov["same_modality"]
    assert (ov["attributable_to_operator"] + ov["attributable_to_modality"]
            == pytest.approx(total, abs=1e-9))


def test_a_missing_image_to_image_panel_says_how_to_build_it(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    setting.panel_path("img_img").unlink()
    with pytest.raises(FileNotFoundError, match="--stage rebuttal"):
        sc.run(setting, out_dir=tmp_path / "out2", device="cpu", n_boot=10)


def test_running_twice_reuses_the_json_instead_of_recomputing(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    out = tmp_path / "out"
    sc.run(setting, out_dir=out, device="cpu", n_boot=10)

    json_path = out / "stability_conditioned.json"
    payload = json.loads(json_path.read_text())
    payload["marker"] = "untouched"
    json_path.write_text(json.dumps(payload))

    again = sc.run(setting, out_dir=out, device="cpu", n_boot=10)
    assert again.get("marker") == "untouched"

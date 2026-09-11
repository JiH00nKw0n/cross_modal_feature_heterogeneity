"""The same-modality control, on tiny panels built from tiny SAEs.

The measurement is one line of arithmetic over the panel's own assignment, so
what these tests check is the plumbing around it: that a panel missing from
disk is skipped rather than silently counted, that only the rows marked usable
reach the statistics, and that all four output files are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.alignment.panel import build_panel, load_panel, save_panel
from src.rebuttal import same_modality_control as smc
from src.rebuttal.common import Setting, unit_decoder
from tests.conftest import make_coco_cache, make_two_sided_sae

DIM = 16
LATENT_TOTAL = 8  # 4 latents per side
K = 2


def _build_setting(tmp_path: Path, *, with_same_modality: bool = True,
                   kill_image_latent: bool = False) -> Setting:
    """A COCO-shaped setting with real panels, small enough to run in a second.

    `kill_image_latent` drives one image latent of model A permanently negative
    so that it never fires. That latent is then dead on the image side of every
    panel, which is what lets a test see the difference between the rows the
    assignment hands a partner to and the rows that are allowed into a summary.
    """
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=10, caps_per_image=2, dim=DIM, seed=3)

    model_a = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=0)
    model_b = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=K, seed=1)
    if kill_image_latent:
        import torch

        with torch.no_grad():
            model_a.image_sae.encoder.weight[0].zero_()
            model_a.image_sae.encoder.bias[0] = -100.0

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
    if with_same_modality:
        for pairing in ("img_img", "txt_txt", "txt_txt_diffcap"):
            _panel(panels_dir / f"{pairing}.npz", pairing, model_b)

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


def test_it_writes_json_markdown_and_both_figure_formats(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    out = tmp_path / "out"
    payload = smc.run(setting, out_dir=out, device="cpu", tau=0.4, n_boot=25, min_bin=2)

    for suffix in (".json", ".md", ".pdf", ".png"):
        assert (out / f"same_modality_control{suffix}").exists(), suffix
    assert payload["random_null_distance"] == 1.0
    assert payload["dim"] == DIM
    text = (out / "same_modality_control.md").read_text()
    assert "—" not in text, "no em dash is allowed in a report"
    assert "img_txt" not in text, "the report must not print internal panel codes"


def test_every_available_panel_is_measured_and_a_missing_one_is_named(
        tmp_path: Path) -> None:
    """All four panels when they exist; only the image-to-text panel otherwise."""
    full = _build_setting(tmp_path / "full")
    payload = smc.run(full, out_dir=tmp_path / "full" / "out", device="cpu", n_boot=10)
    assert list(payload["panels"]) == [
        "img_img", "txt_txt", "txt_txt_diffcap", "img_txt",
    ]
    assert payload["panels_not_available"] == []

    thin = _build_setting(tmp_path / "thin", with_same_modality=False)
    thin_payload = smc.run(thin, out_dir=tmp_path / "thin" / "out", device="cpu",
                           n_boot=10)
    assert list(thin_payload["panels"]) == ["img_txt"]
    assert len(thin_payload["panels_not_available"]) == 3
    assert thin_payload["paired_img_txt_minus_img_img"]["n_latents"] == 0


def test_only_usable_rows_enter_the_statistics(tmp_path: Path) -> None:
    """A latent that never fires is given a partner by the assignment, and dropped.

    The check recomputes the distance from the checkpoints for exactly the rows
    the panel marks usable and compares it against what the module reported.
    """
    setting = _build_setting(tmp_path, kill_image_latent=True)
    payload = smc.run(setting, out_dir=tmp_path / "out", device="cpu", n_boot=10)

    panel = load_panel(setting.panel_img_txt)
    usable = np.asarray(panel["usable"], dtype=bool)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    assert not usable[0], "the silenced latent must not be usable"
    assert usable.sum() < usable.size, "the fixture must leave at least one row out"

    Wi = unit_decoder(setting.ckpt_a, "image")
    Wt = unit_decoder(setting.ckpt_a, "text")
    rows = np.where(usable)[0]
    expected = 1.0 - (Wi[rows] * Wt[perm[rows]]).sum(axis=1)

    entry = payload["panels"]["img_txt"]
    assert entry["n_usable"] == int(usable.sum())
    assert entry["distance"]["n"] == int(usable.sum())
    assert entry["matched_correlation"]["n"] == int(usable.sum())
    assert entry["distance"]["median"] == pytest.approx(float(np.median(expected)),
                                                        abs=1e-6)
    assert entry["distance"]["mean"] == pytest.approx(float(np.mean(expected)),
                                                      abs=1e-6)


def test_the_bands_partition_the_matched_pairs_with_a_correlation_of_at_least_zero(
        tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    payload = smc.run(setting, out_dir=tmp_path / "out", device="cpu", n_boot=10)

    entry = payload["panels"]["img_img"]
    assert list(entry["bins"]) == ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)",
                                   "[0.6,0.8)", "[0.8,1.0]"]
    counted = sum(b["n_pairs"] for b in entry["bins"].values())
    assert counted + entry["n_below_bands"] == entry["n_usable"]


def test_the_two_thresholds_are_the_headline_and_the_tau_knob(tmp_path: Path) -> None:
    """A higher threshold can only keep fewer pairs, and tau names the second one."""
    setting = _build_setting(tmp_path)
    payload = smc.run(setting, out_dir=tmp_path / "out", device="cpu",
                      tau=0.3, headline_c=0.7, n_boot=10)
    assert payload["headline_c"] == 0.7
    assert payload["fallback_c"] == 0.3
    for entry in payload["panels"].values():
        assert entry["headline"]["threshold"] == 0.7
        assert entry["fallback"]["threshold"] == 0.3
        assert entry["headline"]["n_pairs"] <= entry["fallback"]["n_pairs"]
        # One matched partner per latent means a pair is a latent.
        assert entry["headline"]["n_pairs"] == entry["headline"]["n_rows"]


def test_running_twice_reuses_the_json_instead_of_recomputing(tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    out = tmp_path / "out"
    smc.run(setting, out_dir=out, device="cpu", n_boot=10)

    json_path = out / "same_modality_control.json"
    payload = json.loads(json_path.read_text())
    payload["marker"] = "untouched"
    json_path.write_text(json.dumps(payload))

    again = smc.run(setting, out_dir=out, device="cpu", n_boot=10)
    assert again.get("marker") == "untouched"


def test_the_paired_difference_uses_the_latents_both_panels_share(
        tmp_path: Path) -> None:
    setting = _build_setting(tmp_path)
    payload = smc.run(setting, out_dir=tmp_path / "out", device="cpu",
                      headline_c=0.0, n_boot=10)

    paired = payload["paired_img_txt_minus_img_img"]
    ii = load_panel(setting.panel_path("img_img"))
    it = load_panel(setting.panel_img_txt)
    shared = (np.asarray(ii["usable"], dtype=bool)
              & np.asarray(it["usable"], dtype=bool))
    # With a threshold of zero every usable row with a non-negative matched
    # correlation qualifies, so the shared count is an upper bound.
    assert paired["n_latents"] <= int(shared.sum())
    if paired["n_latents"]:
        assert paired["threshold"] == 0.0
        assert np.isfinite(paired["median_difference"])

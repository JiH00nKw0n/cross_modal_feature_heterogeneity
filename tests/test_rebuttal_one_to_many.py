"""The two one-to-many analyses, on hand-built inputs whose answers are known.

The span analysis is pure geometry, so its inputs are chosen to make the
orthogonal distance analytic: text directions that span the image direction give
exactly zero, text directions orthogonal to it give exactly one.

The splitting analysis counts co-firing over a streamed cache, so its answers
are checked against the same counts taken densely, in one shot, from the same
checkpoint and the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.cache_io import load_stacked, split_rows
from src.data.paired_dataset import normalize_np
from src.rebuttal import one_to_many_span as span_mod
from src.rebuttal import one_to_many_splitting as split_mod
from src.rebuttal.common import Setting
from tests.conftest import make_coco_cache, make_two_sided_sae


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_panel(path: Path, *, C: np.ndarray, alive_image: np.ndarray,
                 alive_text: np.ndarray, n_samples: int,
                 fire_image: np.ndarray | None = None,
                 fire_text: np.ndarray | None = None) -> Path:
    """A panel.npz holding exactly the fields the analyses read."""
    L_a, L_b = C.shape
    fire_image = np.where(alive_image, 1, 0).astype(np.int64) if fire_image is None \
        else fire_image.astype(np.int64)
    fire_text = np.where(alive_text, 1, 0).astype(np.int64) if fire_text is None \
        else fire_text.astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        C=C.astype(np.float32),
        perm=np.arange(L_a, dtype=np.int64) % L_b,
        usable=(alive_image & alive_text[np.arange(L_a) % L_b]).astype(bool),
        alive_image=alive_image.astype(bool),
        alive_text=alive_text.astype(bool),
        fire_count_image=fire_image,
        fire_count_text=fire_text,
        rate_image=(fire_image / max(n_samples, 1)).astype(np.float64),
        rate_text=(fire_text / max(n_samples, 1)).astype(np.float64),
        n_samples=np.int64(n_samples),
    )
    return path


def _setting(tmp_path: Path, *, cache_dir: Path, ckpt: Path, panel: Path,
             tag: str = "coco_k8") -> Setting:
    return Setting(
        tag=tag, dataset="coco", cache_dir=cache_dir, split="train",
        ckpt_a=ckpt, ckpt_b=ckpt, panel_img_txt=panel,
        panels_dir=tmp_path / "panels", out_dir=tmp_path / "out",
        coco_cache=cache_dir, k=2, latent_size=8, num_epochs=1,
    )


def _checkpoint_with_directions(path: Path, Wi: np.ndarray, Wt: np.ndarray) -> Path:
    """Save a two-sided checkpoint whose decoder rows are exactly Wi and Wt."""
    dim = Wi.shape[1]
    model = make_two_sided_sae(dim=dim, latent_size=2 * Wi.shape[0], k=2, seed=0)
    with torch.no_grad():
        model.image_sae.W_dec.copy_(torch.as_tensor(Wi, dtype=torch.float32))
        model.text_sae.W_dec.copy_(torch.as_tensor(Wt, dtype=torch.float32))
    model.save_pretrained(path)
    return path


def _dense_firing(ckpt: Path, cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Firing indicators (rows, latents) for both sides, computed in one shot.

    The independent answer the streamed counts are checked against.
    """
    from src.eval.eval_utils import load_sae

    model = load_sae(ckpt, "separated").eval()
    cache = load_stacked(cache_dir, mmap=False)
    rows = split_rows(cache, split)
    out = []
    for sae, table in ((model.image_sae, cache["image"]), (model.text_sae, cache["text"])):
        x = torch.from_numpy(normalize_np(table[rows]))
        with torch.no_grad():
            z = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
        out.append((z.dense_latents.squeeze(1).float().numpy() != 0))
    return out[0], out[1]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.sum(a & b))
    union = float(np.sum(a | b))
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# the projection itself
# --------------------------------------------------------------------------- #
def test_a_direction_inside_the_span_has_zero_orthogonal_distance() -> None:
    phi = np.array([1.0, 0.0, 0.0, 0.0])
    Psi = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    e = span_mod.explained_fraction(phi, Psi)
    assert e == pytest.approx(1.0, abs=1e-9)
    assert np.sqrt(max(1.0 - e, 0.0)) == pytest.approx(0.0, abs=1e-6)


def test_a_direction_orthogonal_to_the_span_has_distance_one() -> None:
    phi = np.array([0.0, 0.0, 1.0, 0.0])
    Psi = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    e = span_mod.explained_fraction(phi, Psi)
    assert e == pytest.approx(0.0, abs=1e-9)
    assert np.sqrt(max(1.0 - e, 0.0)) == pytest.approx(1.0, abs=1e-9)


def test_a_direction_at_forty_five_degrees_splits_its_energy_in_half() -> None:
    phi = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)
    Psi = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    e = span_mod.explained_fraction(phi, Psi)
    assert e == pytest.approx(0.5, abs=1e-9)
    assert np.sqrt(1.0 - e) == pytest.approx(np.sqrt(0.5), abs=1e-9)


def test_the_partner_rule_uses_the_signed_correlation_not_its_magnitude() -> None:
    """A strongly negative correlation is not a partner."""
    C = np.array([
        [0.9, 0.8, -0.95, 0.1],     # two partners at 0.4
        [0.9, -0.9, -0.9, -0.9],    # one partner only, so no group
    ], dtype=np.float32)
    alive_image = np.ones(2, dtype=bool)
    alive_text = np.ones(4, dtype=bool)
    groups = span_mod.find_groups(C, alive_image, alive_text, 0.4)
    assert len(groups) == 1
    i, partners = groups[0]
    assert i == 0
    assert partners.tolist() == [0, 1]           # ordered most correlated first


def test_a_dead_text_latent_is_never_a_partner() -> None:
    C = np.full((2, 4), 0.9, dtype=np.float32)
    alive_image = np.ones(2, dtype=bool)
    alive_text = np.array([True, False, False, True])
    groups = span_mod.find_groups(C, alive_image, alive_text, 0.4)
    assert [p.tolist() for _i, p in groups] == [[0, 3], [0, 3]]


# --------------------------------------------------------------------------- #
# span: end to end
# --------------------------------------------------------------------------- #
def _span_fixture(tmp_path: Path) -> Setting:
    """Two groups: one whose partners span the image direction, one orthogonal.

    Image latent 0 points along axis 0 and its partners are axes 0 and 1, so its
    orthogonal distance is exactly 0. Image latent 1 points along axis 3, which
    no partner touches, so its distance is exactly 1.
    """
    eye = np.eye(4, dtype=np.float32)
    Wi = np.stack([eye[0], eye[3], eye[2], eye[1]])
    Wt = np.stack([eye[0], eye[1], eye[2], eye[3]])
    ckpt = _checkpoint_with_directions(tmp_path / "ckpt", Wi, Wt)

    C = np.full((4, 4), 0.05, dtype=np.float32)
    C[0, 0] = 0.9
    C[0, 1] = 0.8
    C[1, 0] = 0.7
    C[1, 1] = 0.6
    panel = _write_panel(
        tmp_path / "panel.npz", C=C,
        alive_image=np.ones(4, dtype=bool), alive_text=np.ones(4, dtype=bool),
        n_samples=1000,
    )
    make_coco_cache(tmp_path / "coco", n_images=8, caps_per_image=2, dim=4, seed=0)
    return _setting(tmp_path, cache_dir=tmp_path / "coco", ckpt=ckpt, panel=panel)


def test_span_reports_the_analytic_distances_of_the_two_groups(tmp_path: Path) -> None:
    setting = _span_fixture(tmp_path)
    payload = span_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                           tau=0.4, n_boot=32, n_draws=2, seed=0)

    assert payload["n_groups"] == 2
    assert payload["per_group"]["image_latent"] == [0, 1]
    assert payload["per_group"]["n_partners"] == [2, 2]

    # One group is fully explained and one not at all, so the two distances are
    # 0 and 1 and their median is 0.5.
    full = np.array(payload["per_group"]["all_partners"])
    assert full[0] == pytest.approx(1.0, abs=1e-6)
    assert full[1] == pytest.approx(0.0, abs=1e-6)
    dist = payload["orthogonal_distance"]["all_partners"]
    assert dist["median"] == pytest.approx(0.5, abs=1e-6)
    assert dist["n"] == 2

    # The strongest partner alone covers image latent 0 completely and image
    # latent 1 not at all, so the same two values come back.
    top1 = np.array(payload["per_group"]["strongest_partner_only"])
    assert top1.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)

    assert payload["threshold_rule"] == "signed correlation C[i,j] >= tau (not |C|)"
    assert payload["group_share_of_alive_image"] == pytest.approx(0.5)
    assert payload["analytic_random_subspace"] == pytest.approx(2 / 4)
    assert payload["frac_groups_explained_above_half"] == pytest.approx(0.5)


def test_span_writes_a_self_contained_report_with_no_em_dash(tmp_path: Path) -> None:
    setting = _span_fixture(tmp_path)
    span_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                 tau=0.4, n_boot=32, n_draws=2, seed=0)
    text = (tmp_path / "out" / "one_to_many_span.md").read_text()
    assert text.startswith("# ")
    assert "—" not in text
    assert "signed correlation" in text
    assert "Groups measured (count)" in text
    assert "Text partners in the group (count)" in text
    # Every arm is named in words, not by its json key.
    for label in span_mod.ARM_LABELS.values():
        assert label in text
    assert "all_partners" not in text


def test_span_skips_when_its_json_already_exists(tmp_path: Path) -> None:
    setting = _span_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "one_to_many_span.json").write_text('{"n_groups": 99}')
    payload = span_mod.run(setting, out_dir=out, device="cpu", tau=0.4,
                           n_boot=8, n_draws=1, seed=0)
    assert payload["n_groups"] == 99
    assert not (out / "one_to_many_span.md").exists()


def test_span_reports_no_group_rather_than_failing(tmp_path: Path) -> None:
    setting = _span_fixture(tmp_path)
    payload = span_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                           tau=0.99, n_boot=8, n_draws=1, seed=0)
    assert payload["n_groups"] == 0
    text = (tmp_path / "out" / "one_to_many_span.md").read_text()
    assert "No image latent has two or more text partners" in text


# --------------------------------------------------------------------------- #
# splitting: end to end against dense counts
# --------------------------------------------------------------------------- #
def _splitting_fixture(tmp_path: Path, *, tau_partners: tuple[int, int] = (0, 1)):
    """A tiny cache, a real checkpoint, and a panel that forces one group.

    Returns the setting, the dense firing indicators and the image latent the
    group was built on, so a test can compute the same Jaccard and coverage
    numbers without the streaming pass. The image latent is the first one that
    actually fires, because a latent the Top-K never selects is dead and the
    analysis would correctly refuse to build a group on it.
    """
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=12, caps_per_image=3, dim=8, seed=5)
    ckpt = tmp_path / "ckpt"
    make_two_sided_sae(dim=8, latent_size=8, k=2, seed=1).save_pretrained(ckpt)

    fired_i, fired_t = _dense_firing(ckpt, cache_dir, "train")
    n_rows = fired_i.shape[0]
    alive_image = fired_i.any(axis=0)
    alive_text = fired_t.any(axis=0)
    img_latent = int(np.where(alive_image)[0][0])

    L = fired_i.shape[1]
    C = np.full((L, L), 0.05, dtype=np.float32)
    a, b = tau_partners
    C[img_latent, a] = 0.9
    C[img_latent, b] = 0.7
    panel = _write_panel(
        tmp_path / "panel.npz", C=C, alive_image=alive_image, alive_text=alive_text,
        n_samples=n_rows,
        fire_image=fired_i.sum(axis=0), fire_text=fired_t.sum(axis=0),
    )
    setting = _setting(tmp_path, cache_dir=cache_dir, ckpt=ckpt, panel=panel)
    return setting, fired_i, fired_t, img_latent


def test_splitting_jaccard_matches_the_same_sets_counted_densely(tmp_path: Path) -> None:
    setting, fired_i, fired_t, img_latent = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.4, n_random_pairs=8, batch_size=4, seed=0)

    assert payload["n_groups"] == 1
    row = payload["per_group"][0]
    assert row["image_latent"] == img_latent
    assert row["n_partners"] == 2

    expected = _jaccard(fired_t[:, 0], fired_t[:, 1])
    assert payload["jaccard_within_group"]["n"] == 1
    assert payload["jaccard_within_group"]["median"] == pytest.approx(expected, abs=1e-9)
    assert row["jaccard_median"] == pytest.approx(expected, abs=1e-9)


def test_splitting_coverage_matches_the_same_sets_counted_densely(tmp_path: Path) -> None:
    setting, fired_i, fired_t, img_latent = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.4, n_random_pairs=8, batch_size=4, seed=0)

    S = fired_i[:, img_latent]
    union = fired_t[:, 0] | fired_t[:, 1]
    co_any = float(np.sum(S & union))
    co_top = float(np.sum(S & fired_t[:, 0]))

    row = payload["per_group"][0]
    assert row["strongest_share_of_cofiring"] == pytest.approx(co_top / co_any, abs=1e-9)
    assert row["cofiring_share_of_image_firing"] == pytest.approx(
        co_any / float(S.sum()), abs=1e-9)


def test_splitting_streams_the_same_firing_counts_the_panel_holds(tmp_path: Path) -> None:
    """The streamed pass and the panel must have seen the same rows."""
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.4, n_random_pairs=8, batch_size=4, seed=0)
    agree = payload["fire_count_agreement"]
    assert agree["checked"] is True
    assert agree["max_abs_difference_image_latent_fire_count"] == 0.0
    assert agree["max_abs_difference_text_latent_fire_count"] == 0.0


def test_splitting_refuses_a_panel_built_on_a_different_number_of_rows(
        tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    # Rewrite the panel claiming a row count the split cannot produce.
    data = dict(np.load(setting.panel_img_txt))
    data["n_samples"] = np.int64(7)
    np.savez(setting.panel_img_txt, **data)
    with pytest.raises(ValueError, match="was built on 7"):
        split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                      tau=0.4, n_random_pairs=4, batch_size=4, seed=0)


def test_splitting_sweeps_the_threshold_and_the_counts_only_fall(tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.4, n_random_pairs=8, batch_size=4, seed=0)
    sweep = payload["tau_sweep"]
    assert [r["tau"] for r in sweep] == list(split_mod.DEFAULT_TAU_SWEEP)
    counts = [r["n_groups"] for r in sweep]
    assert counts == sorted(counts, reverse=True)
    # The one group is above 0.7 on its weaker partner, so it survives to 0.5
    # and is gone by the time the threshold passes 0.7.
    assert sweep[-1]["n_groups"] == 1


def test_splitting_random_pairs_are_two_different_alive_text_latents(
        tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.4, n_random_pairs=64, batch_size=4, seed=0)
    jr = payload["jaccard_random_pairs"]
    assert jr["n"] > 0
    assert 0.0 <= jr["median"] <= 1.0
    # A pair of a latent with itself would score 1 by construction, so the mean
    # cannot be pinned at 1 unless the draw is broken.
    assert jr["mean"] < 1.0


def test_splitting_writes_a_self_contained_report_with_no_em_dash(tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                  tau=0.4, n_random_pairs=8, batch_size=4, seed=0)
    text = (tmp_path / "out" / "one_to_many_splitting.md").read_text()
    assert text.startswith("# ")
    assert "—" not in text
    assert "Correlation threshold" in text
    assert "Median Jaccard overlap of the firing sets (0 to 1)" in text
    assert "two alive text latents drawn at random" in text
    assert "jaccard_within_group" not in text


def test_splitting_skips_when_its_json_already_exists(tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "one_to_many_splitting.json").write_text('{"n_groups": 42}')
    payload = split_mod.run(setting, out_dir=out, device="cpu", tau=0.4,
                            n_random_pairs=4, batch_size=4, seed=0)
    assert payload["n_groups"] == 42
    assert not (out / "one_to_many_splitting.md").exists()


def test_splitting_reports_the_sweep_even_when_no_group_clears_the_threshold(
        tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    payload = split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                            tau=0.95, n_random_pairs=4, batch_size=4, seed=0)
    assert payload["n_groups"] == 0
    text = (tmp_path / "out" / "one_to_many_splitting.md").read_text()
    assert "Correlation threshold" in text
    assert "no image latent has two or more partners" in text


def test_both_analyses_leave_a_json_a_reader_can_load(tmp_path: Path) -> None:
    setting, _fi, _ft, _img = _splitting_fixture(tmp_path)
    split_mod.run(setting, out_dir=tmp_path / "out", device="cpu",
                  tau=0.4, n_random_pairs=8, batch_size=4, seed=0)
    payload = json.loads((tmp_path / "out" / "one_to_many_splitting.json").read_text())
    assert payload["analysis"] == "one_to_many_splitting"
    assert payload["setting"]["tag"] == "coco_k8"
    assert payload["alive_rule"] == "fire_count >= 1 on the full train split"

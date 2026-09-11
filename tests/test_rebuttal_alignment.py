"""The two alignment analyses: the ceiling comparison and the operator swap.

Three things are checked. The transforms are checked against cases whose answer
is known in advance, so a sign error or a transposed matrix cannot pass: an
orthogonal Procrustes fit on a pair of point sets related by a known rotation
has to return that rotation, and greedy matching on a correlation matrix that is
a permuted identity has to return that permutation. The third check runs both
modules end to end on a tiny fixture setting and asserts that each writes its
json and its markdown, that the markdown is self-contained, and that a second
call does no work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.alignment import build_panel, save_panel
from src.rebuttal import alignment_ceiling, alignment_methods
from src.rebuttal.common import Setting
from tests.conftest import make_coco_cache, make_two_sided_sae

DIM = 16
LATENT_SIZE = 32          # total budget; a modality-specific model holds 16
K = 2


# --------------------------------------------------------------------------- #
# the transforms, against known answers
# --------------------------------------------------------------------------- #
def test_procrustes_recovers_a_rotation_it_was_shown() -> None:
    """Points rotated by a known orthogonal matrix must return that matrix."""
    rng = np.random.default_rng(0)
    d = 8
    X = rng.normal(size=(200, d))
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Y = X @ Q

    R = alignment_ceiling.orthogonal_map(X, Y)
    assert R.shape == (d, d)
    np.testing.assert_allclose(R, Q, atol=1e-8)
    np.testing.assert_allclose(R @ R.T, np.eye(d), atol=1e-8)
    # Applying it lines the two sets up exactly, so every cosine is one.
    cos = alignment_ceiling.applied_cosine(X, Y / np.linalg.norm(Y, axis=1, keepdims=True), R)
    np.testing.assert_allclose(cos, np.ones(len(X)), atol=1e-8)


def test_procrustes_on_the_latent_cross_covariance_recovers_a_rotation() -> None:
    """The same check for the operator fitted on a cross-covariance matrix.

    `procrustes_map` is handed the image-to-text cross-covariance and returns a
    matrix applied to the TEXT side, so with text latents built as
    `z_image @ Q` it has to return `Q` transposed back, that is the map carrying
    text onto image.
    """
    rng = np.random.default_rng(1)
    d = 6
    Zi = rng.normal(size=(500, d))
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Zt = Zi @ Q
    Sit = (Zi - Zi.mean(0)).T @ (Zt - Zt.mean(0)) / len(Zi)

    R = alignment_methods.procrustes_map(Sit)
    np.testing.assert_allclose(Zt @ R, Zi, atol=1e-8)


def test_a_linear_map_recovers_a_known_linear_map() -> None:
    rng = np.random.default_rng(2)
    d = 5
    X = rng.normal(size=(400, d))
    M = rng.normal(size=(d, d))
    A = alignment_ceiling.linear_map(X, X @ M, ridge=1e-9)
    np.testing.assert_allclose(A, M, atol=1e-5)


def test_greedy_matching_recovers_a_permuted_identity() -> None:
    """A correlation matrix that is a permuted identity has one obvious answer."""
    rng = np.random.default_rng(3)
    n = 12
    truth = rng.permutation(n)
    C = np.full((n, n), 0.01)
    C[np.arange(n), truth] = 1.0
    alive = np.ones(n, dtype=bool)

    np.testing.assert_array_equal(alignment_methods.greedy_perm(C, alive, alive), truth)
    np.testing.assert_array_equal(alignment_methods.hungarian_perm(C, alive, alive), truth)


def test_greedy_matching_never_uses_a_dead_row_or_column() -> None:
    rng = np.random.default_rng(4)
    n = 10
    C = rng.normal(size=(n, n))
    alive_a = np.ones(n, dtype=bool)
    alive_b = np.ones(n, dtype=bool)
    alive_a[3] = False
    alive_b[7] = False

    perm = alignment_methods.greedy_perm(C, alive_a, alive_b)
    assert perm[3] == 0, "a dead row must keep its placeholder partner"
    assert 7 not in set(perm[alive_a].tolist()), "a dead column was assigned to a live row"


def test_the_chance_oracle_directions_are_unit_length() -> None:
    v = alignment_ceiling.random_unit(64, 9, np.random.default_rng(5))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.ones(64), atol=1e-10)


# --------------------------------------------------------------------------- #
# end to end on a fixture setting
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_setting(tmp_path: Path) -> Setting:
    """One setting whose cache, checkpoints and panel are all tiny and real.

    The panel is built by the repository's own builder rather than faked, so the
    analyses read the same keys, the same alive rule and the same assignment
    they read in production.
    """
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=12, caps_per_image=3, dim=DIM, seed=0)

    ckpt_a = tmp_path / "models" / "seed0" / "final"
    ckpt_b = tmp_path / "models" / "seed1" / "final"
    make_two_sided_sae(dim=DIM, latent_size=LATENT_SIZE, k=K, seed=0).save_pretrained(ckpt_a)
    make_two_sided_sae(dim=DIM, latent_size=LATENT_SIZE, k=K, seed=1).save_pretrained(ckpt_b)

    from src.models import TwoSidedTopKSAE

    panel_path = tmp_path / "panels" / "panel.npz"
    payload = build_panel(
        model=TwoSidedTopKSAE.from_pretrained(ckpt_a), cache_dir=cache_dir,
        split="train", batch_size=8, device="cpu", pairing="img_txt",
        ckpt_a=ckpt_a,
    )
    save_panel(panel_path, payload)

    return Setting(
        tag="coco_k8", dataset="coco", cache_dir=cache_dir, split="train",
        ckpt_a=ckpt_a, ckpt_b=ckpt_b,
        panel_img_txt=panel_path, panels_dir=tmp_path / "panels",
        out_dir=tmp_path / "out", coco_cache=cache_dir,
        k=K, latent_size=LATENT_SIZE, num_epochs=1,
    )


def test_alignment_ceiling_writes_its_json_and_report(tiny_setting: Setting) -> None:
    out_dir = tiny_setting.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = alignment_ceiling.run(tiny_setting, out_dir=out_dir, device="cpu",
                                    n_boot=16, seed=0)

    out_json = out_dir / "alignment_ceiling.json"
    out_md = out_dir / "alignment_ceiling.md"
    assert out_json.exists() and out_md.exists()
    on_disk = json.loads(out_json.read_text())
    assert on_disk["analysis"] == "alignment_ceiling"
    assert on_disk["n_usable_pairs"] == payload["n_usable_pairs"] > 0

    # A cosine distance lives in [0, 2], and the oracle can never be further
    # away than the partner the assignment picked.
    assert 0.0 <= payload["oracle_distance"]["median"] <= 2.0
    assert payload["oracle_distance"]["median"] <= payload["matched_distance"]["median"] + 1e-9
    assert payload["global_transform"]["n_fit"] + payload["global_transform"]["n_eval"] == \
        payload["n_usable_pairs"]

    text = out_md.read_text()
    assert text.startswith("# ")
    assert "—" not in text, "no em dashes in a report"
    assert "no transform, the raw pair of directions" in text
    assert "best rotation (orthogonal Procrustes)" in text


def test_alignment_ceiling_is_idempotent(tiny_setting: Setting, monkeypatch) -> None:
    out_dir = tiny_setting.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    alignment_ceiling.run(tiny_setting, out_dir=out_dir, device="cpu", n_boot=8)

    def explode(*_a, **_k):  # pragma: no cover
        raise AssertionError("the analysis recomputed instead of skipping")

    monkeypatch.setattr(alignment_ceiling, "unit_decoder", explode)
    again = alignment_ceiling.run(tiny_setting, out_dir=out_dir, device="cpu", n_boot=8)
    assert again["analysis"] == "alignment_ceiling"


def test_alignment_methods_writes_every_operator(tiny_setting: Setting) -> None:
    out_dir = tiny_setting.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = alignment_methods.run(
        tiny_setting, out_dir=out_dir, device="cpu",
        max_fit_samples=0, sinkhorn_eps=(0.05,), sinkhorn_iters=20,
        cca_dims=(2, 4), conf_cutoffs=(), batch_size=8,
    )

    out_json = out_dir / "alignment_methods.json"
    out_md = out_dir / "alignment_methods.md"
    assert out_json.exists() and out_md.exists()

    keys = list(payload["results"])
    assert keys[0] == "hungarian_coactivation", "the paper's method leads the table"
    for expected in ("greedy_coactivation", "hungarian_decoder_cosine",
                     "sinkhorn_eps0.05", "procrustes_rotation", "cca_d2", "cca_d4"):
        assert expected in keys, f"{expected} was not scored"

    for key, row in payload["results"].items():
        for k in (1, 5, 10):
            assert 0.0 <= row[f"I2T R@{k}"] <= 1.0, key
            assert 0.0 <= row[f"T2I R@{k}"] <= 1.0, key
        assert row["I2T R@1"] <= row["I2T R@5"] <= row["I2T R@10"], key
        assert row["T2I R@1"] <= row["T2I R@5"] <= row["T2I R@10"], key

    text = out_md.read_text()
    assert "—" not in text, "no em dashes in a report"
    assert "Hungarian on co-activation (the paper's method)" in text
    assert "image finds caption, recall at 1" in text


def test_alignment_methods_hungarian_row_uses_the_panel_permutation(
        tiny_setting: Setting) -> None:
    """The paper's row must be scored with the permutation the panel stored.

    Re-scoring the panel's permutation by hand has to land on the same recalls,
    which is what proves the row is not quietly matching again on its own.
    """
    from src.alignment import load_panel
    from src.eval import eval_utils

    out_dir = tiny_setting.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = alignment_methods.run(
        tiny_setting, out_dir=out_dir, device="cpu",
        max_fit_samples=0, sinkhorn_eps=(), cca_dims=(), conf_cutoffs=(), batch_size=8,
    )

    import torch

    perm = np.asarray(load_panel(tiny_setting.panel_img_txt)["perm"], dtype=np.int64)
    model = eval_utils.load_sae(tiny_setting.ckpt_a, "separated")
    ds = eval_utils.load_paired_split(tiny_setting.coco_cache, "test")
    pair_img_idx, img_rows, gt_caps = alignment_methods.coco_test_rows(ds.keys)
    dev = torch.device("cpu")
    # Method "ours" is exactly "separated" plus the saved permutation on the
    # text side, so this is the path src/eval/retrieval.py takes for the paper's
    # numbers.
    z_img = eval_utils.encode_image(model, ds.image[img_rows], "ours", dev)
    z_txt = eval_utils.encode_text(model, ds.text, "ours", dev, perm=perm)
    expected = alignment_methods.recalls(z_img, z_txt, pair_img_idx, gt_caps)

    got = payload["results"]["hungarian_coactivation"]
    for key, value in expected.items():
        assert got[key] == pytest.approx(value), key


def test_alignment_methods_is_idempotent(tiny_setting: Setting, monkeypatch) -> None:
    out_dir = tiny_setting.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    alignment_methods.run(
        tiny_setting, out_dir=out_dir, device="cpu",
        max_fit_samples=0, sinkhorn_eps=(), cca_dims=(), conf_cutoffs=(), batch_size=8,
    )

    def explode(*_a, **_k):  # pragma: no cover
        raise AssertionError("the analysis recomputed instead of skipping")

    monkeypatch.setattr(alignment_methods, "second_moments", explode)
    again = alignment_methods.run(tiny_setting, out_dir=out_dir, device="cpu")
    assert again["analysis"] == "alignment_methods"


def test_the_centered_covariances_match_a_direct_computation() -> None:
    """The streaming moments must give the same covariances as numpy in one go."""
    rng = np.random.default_rng(6)
    n, L = 50, 7
    Zi = rng.normal(size=(n, L))
    Zt = rng.normal(size=(n, L))
    acc = {"sum_i": Zi.sum(0), "sum_t": Zt.sum(0),
           "ii": Zi.T @ Zi, "tt": Zt.T @ Zt, "it": Zi.T @ Zt, "n": float(n)}
    Sii, Stt, Sit = alignment_methods.centered_covariances(acc)
    np.testing.assert_allclose(Sii, np.cov(Zi.T, bias=True), atol=1e-10)
    np.testing.assert_allclose(Stt, np.cov(Zt.T, bias=True), atol=1e-10)
    direct = (Zi - Zi.mean(0)).T @ (Zt - Zt.mean(0)) / n
    np.testing.assert_allclose(Sit, direct, atol=1e-10)

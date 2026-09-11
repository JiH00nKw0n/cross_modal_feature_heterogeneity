"""The three panel rules: alive, correlation, and the Hungarian assignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.alignment.panel import (
    ALIVE_RULE,
    accumulate_cross_stats,
    build_panel,
    hungarian_alive,
    load_panel,
    panel_mismatch,
    pearson_from_stats,
    save_panel,
)
from src.data.cache_io import load_stacked, save_stacked
from tests.conftest import make_two_sided_sae

DIM = 16
LATENT_TOTAL = 8          # 4 latents per side
PER_SIDE = LATENT_TOTAL // 2


def _cache_with_signal(tmp_path: Path, n: int = 400, dim: int = DIM) -> Path:
    """A paired cache whose text row is a rotation of its image row.

    A deterministic relationship between the two sides is what lets the test
    assert that the assignment recovers a planted permutation rather than
    matching noise.
    """
    gen = np.random.default_rng(3)
    image = gen.normal(size=(n, dim)).astype(np.float32)
    text = np.roll(image, shift=1, axis=1).astype(np.float32)
    keys = [f"{i}_0" for i in range(n)]
    cache_dir = tmp_path / "signal"
    save_stacked(cache_dir, image_emb=image, text_emb=text, keys=keys,
                 splits={"train": keys}, meta={"dim": dim})
    return cache_dir


def test_pearson_matches_numpy_corrcoef() -> None:
    """The streamed accumulators reproduce a materialized correlation to 1e-5."""
    gen = np.random.default_rng(11)
    n, la, lb = 500, 6, 5
    za = gen.normal(size=(n, la)).astype(np.float32)
    zb = (za[:, :lb] * 0.7 + gen.normal(size=(n, lb)) * 0.3).astype(np.float32)

    stats = {
        "sum_a": za.sum(0).astype(np.float64), "sum_b": zb.sum(0).astype(np.float64),
        "sumsq_a": (za ** 2).sum(0).astype(np.float64),
        "sumsq_b": (zb ** 2).sum(0).astype(np.float64),
        "cross": (za.T @ zb).astype(np.float64),
        "n": n,
    }
    C = pearson_from_stats(stats)
    expected = np.corrcoef(za.T, zb.T)[:la, la:]
    np.testing.assert_allclose(C, expected, atol=1e-5)


def test_zero_variance_column_gives_zero_not_nan() -> None:
    """Correlation rule: a latent that never varies contributes 0, never NaN."""
    n = 50
    za = np.zeros((n, 2), dtype=np.float64)
    za[:, 0] = np.arange(n)
    zb = np.zeros((n, 2), dtype=np.float64)
    zb[:, 0] = np.arange(n)
    stats = {"sum_a": za.sum(0), "sum_b": zb.sum(0),
             "sumsq_a": (za ** 2).sum(0), "sumsq_b": (zb ** 2).sum(0),
             "cross": za.T @ zb, "n": n}
    C = pearson_from_stats(stats)
    assert np.isfinite(C).all()
    assert C[1, 1] == 0.0
    assert np.isclose(C[0, 0], 1.0)


def test_hungarian_recovers_a_planted_permutation() -> None:
    """Matching rule: the assignment finds the permutation that maximizes C."""
    L = 6
    planted = np.array([3, 0, 5, 1, 4, 2])
    C = np.full((L, L), -0.5)
    C[np.arange(L), planted] = 0.9
    alive = np.ones(L, dtype=bool)
    out = hungarian_alive(C, alive, alive)
    np.testing.assert_array_equal(out["perm"], planted)
    assert out["n_usable"] == L


def test_hungarian_keeps_the_sign_of_the_correlation() -> None:
    """A strongly anti-correlated pair must not be matched as if it were positive."""
    C = np.array([[-0.99, 0.10], [0.05, 0.80]])
    alive = np.ones(2, dtype=bool)
    out = hungarian_alive(C, alive, alive)
    assert out["perm"][0] == 1 or out["matched_c"][0] > 0


def test_dead_rows_cannot_take_an_alive_partner() -> None:
    """Matching rule: a dead latent is pushed to BIG_NEG and marked unusable."""
    L = 4
    C = np.zeros((L, L))
    C[0, 0] = 0.9
    C[1, 1] = 0.8
    alive_a = np.array([True, True, False, False])
    alive_b = np.array([True, True, False, False])
    out = hungarian_alive(C, alive_a, alive_b)
    assert out["perm"][0] == 0 and out["perm"][1] == 1
    assert out["usable"].tolist() == [True, True, False, False]
    assert out["n_usable"] == 2


def _cache_text_equals_image(tmp_path: Path, n: int = 400, dim: int = DIM) -> Path:
    """A cache whose text row is a copy of its image row.

    With the same vector on both sides, the only thing that can separate the
    two latent streams is a difference between the two dictionaries.
    """
    gen = np.random.default_rng(13)
    vecs = gen.normal(size=(n, dim)).astype(np.float32)
    keys = [f"{i}_0" for i in range(n)]
    cache_dir = tmp_path / "same"
    save_stacked(cache_dir, image_emb=vecs, text_emb=vecs, keys=keys,
                 splits={"train": keys}, meta={"dim": dim})
    return cache_dir


def test_build_panel_recovers_a_hidden_text_permutation(tmp_path: Path) -> None:
    """End to end: permute the text SAE's columns, the panel undoes it.

    The two sides are given the same weights, so latent i on one side means the
    same thing as latent i on the other. Reordering the text side's rows is
    then a permutation the assignment has to invert exactly.
    """
    cache_dir = _cache_text_equals_image(tmp_path)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=5)
    planted = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        model.text_sae.encoder.weight.copy_(model.image_sae.encoder.weight[planted])
        model.text_sae.encoder.bias.copy_(model.image_sae.encoder.bias[planted])
        model.text_sae.W_dec.copy_(model.image_sae.W_dec[planted])
        model.text_sae.b_dec.copy_(model.image_sae.b_dec)

    payload = build_panel(model=model, cache_dir=cache_dir, split="train",
                          batch_size=64, device="cpu", max_samples=0)
    perm = payload["perm"]
    usable = payload["usable"]
    # Text column j holds image latent planted[j], so image latent i sits at
    # the position where planted equals i, which is argsort(planted)[i].
    inverse = np.argsort(planted.numpy())
    assert usable.any()
    np.testing.assert_array_equal(perm[usable], inverse[usable])


def test_a_dead_latent_is_not_alive_and_its_row_is_zero(tmp_path: Path) -> None:
    """Alive rule: fire_count >= 1. A latent forced never to fire fails it."""
    cache_dir = _cache_with_signal(tmp_path, n=200)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=1, seed=7)
    with torch.no_grad():
        # Drive one image latent's pre-activation permanently negative, so ReLU
        # zeroes it and top-1 never selects it.
        model.image_sae.encoder.weight[0].zero_()
        model.image_sae.encoder.bias[0] = -100.0

    payload = build_panel(model=model, cache_dir=cache_dir, split="train",
                          batch_size=64, device="cpu", max_samples=0)
    assert payload["fire_count_image"][0] == 0
    assert not payload["alive_image"][0]
    assert payload["rate_image"][0] == 0.0
    np.testing.assert_array_equal(payload["C"][0], np.zeros(PER_SIDE, dtype=np.float32))
    assert not payload["usable"][0]


def test_shuffle_seed_changes_the_pairing(tmp_path: Path) -> None:
    """A non-zero shuffle_seed deranges the B side, so the panel differs."""
    cache_dir = _cache_with_signal(tmp_path, n=300)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=9)
    paired = build_panel(model=model, cache_dir=cache_dir, split="train",
                         batch_size=64, device="cpu", max_samples=0)
    shuffled = build_panel(model=model, cache_dir=cache_dir, split="train",
                           batch_size=64, device="cpu", max_samples=0, shuffle_seed=17)
    assert not np.allclose(paired["C"], shuffled["C"])
    # Firing is a property of one side's own inputs, so it must not move.
    np.testing.assert_array_equal(paired["fire_count_image"], shuffled["fire_count_image"])


def test_max_samples_subsamples_evenly(tmp_path: Path) -> None:
    """max_samples = 0 means every pair; a positive value takes a linspace."""
    cache_dir = _cache_with_signal(tmp_path, n=300)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=4)
    full = build_panel(model=model, cache_dir=cache_dir, split="train",
                       batch_size=64, device="cpu", max_samples=0)
    part = build_panel(model=model, cache_dir=cache_dir, split="train",
                       batch_size=64, device="cpu", max_samples=50)
    assert int(full["n_samples"]) == 300
    assert int(part["n_samples"]) == 50


def test_panel_saves_and_loads_with_its_sidecar(tmp_path: Path) -> None:
    cache_dir = _cache_with_signal(tmp_path, n=120)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=2)
    payload = build_panel(model=model, cache_dir=cache_dir, split="train",
                          batch_size=64, device="cpu", max_samples=0,
                          ckpt_a="fake/ckpt")
    out = tmp_path / "panel.npz"
    save_panel(out, payload)
    assert (tmp_path / "panel.json").exists()

    back = load_panel(out)
    for key in ("C", "perm", "usable", "alive_image", "alive_text",
                "fire_count_image", "fire_count_text", "rate_image", "rate_text",
                "n_samples"):
        assert key in back, key
    assert back["_meta"]["alive_rule"] == ALIVE_RULE
    assert back["_meta"]["pairing"] == "img_txt"
    assert back["_meta"]["ckpt_a"] == "fake/ckpt"


def test_accumulate_matches_a_materialized_pass(tmp_path: Path) -> None:
    """The streaming accumulator agrees with encoding everything at once."""
    cache_dir = _cache_with_signal(tmp_path, n=128)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=6)
    cache = load_stacked(cache_dir)
    rows = np.arange(128, dtype=np.int64)
    stats = accumulate_cross_stats(model.image_sae, model.text_sae,
                                   cache["image"], cache["text"], rows, rows,
                                   batch_size=16, device="cpu")
    C_stream = pearson_from_stats(stats)

    from src.data.paired_dataset import normalize_np

    with torch.no_grad():
        xi = torch.from_numpy(normalize_np(np.asarray(cache["image"])))
        xt = torch.from_numpy(normalize_np(np.asarray(cache["text"])))
        za = model.image_sae(hidden_states=xi.unsqueeze(1),
                             return_dense_latents=True).dense_latents.squeeze(1).numpy()
        zb = model.text_sae(hidden_states=xt.unsqueeze(1),
                            return_dense_latents=True).dense_latents.squeeze(1).numpy()
    la = za.shape[1]
    expected = np.corrcoef(za.T, zb.T)[:la, la:]
    expected = np.nan_to_num(expected, nan=0.0)
    np.testing.assert_allclose(C_stream, expected, atol=1e-5)


def test_panel_mismatch_names_the_first_difference(tmp_path: Path) -> None:
    """A panel is only reusable when its sidecar records the same rules."""
    cache_dir = _cache_with_signal(tmp_path, n=120)
    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=2)
    full = build_panel(model=model, cache_dir=cache_dir, split="train",
                       batch_size=64, device="cpu", max_samples=0)
    assert panel_mismatch(full, split="train", max_samples=0, shuffle_seed=0) is None

    quick = build_panel(model=model, cache_dir=cache_dir, split="train",
                        batch_size=64, device="cpu", max_samples=50)
    why = panel_mismatch(quick, split="train", max_samples=0, shuffle_seed=0)
    assert why is not None and "max_samples" in why

    noise = build_panel(model=model, cache_dir=cache_dir, split="train",
                        batch_size=64, device="cpu", max_samples=0, shuffle_seed=7)
    why = panel_mismatch(noise, split="train", max_samples=0, shuffle_seed=0)
    assert why is not None and "shuffle_seed" in why

    why = panel_mismatch({"C": np.zeros((2, 2))}, split="train")
    assert why is not None and "sidecar" in why


def test_the_panel_records_the_declared_split_size(tmp_path: Path) -> None:
    """A split key that keys.json does not carry must stay visible in the sidecar."""
    cache_dir = _cache_with_signal(tmp_path, n=60)
    import json as _json

    splits_path = cache_dir / "splits.json"
    splits = _json.loads(splits_path.read_text())
    splits["train"] = splits["train"] + ["a-key-that-is-not-in-keys-json"]
    splits_path.write_text(_json.dumps(splits))

    model = make_two_sided_sae(dim=DIM, latent_size=LATENT_TOTAL, k=2, seed=2)
    payload = build_panel(model=model, cache_dir=cache_dir, split="train",
                          batch_size=64, device="cpu", max_samples=0)
    meta = payload["_meta"]
    assert meta["n_split_rows"] == 61, "the sidecar must report what the split declares"
    assert meta["n_split_rows_resolved"] == 60
    assert meta["n_samples"] == 60

"""Reconstruction and zero-shot, on caches whose answer is fixed in advance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.cache_io import load_imagenet_cache, save_stacked
from src.eval import eval_utils
from src.eval.recon import _imagenet_pairs, run as run_recon
from src.eval.zeroshot import run as run_zeroshot
from src.models import TopKSAE, TopKSAEConfig
from tests.conftest import make_imagenet_cache

DIM = 8


def _identity_sae(tmp_path: Path) -> Path:
    """z = relu(x) and x_hat = relu(x): the reconstruction error is known."""
    cfg = TopKSAEConfig(hidden_size=DIM, latent_size=DIM, k=DIM, normalize_decoder=False)
    sae = TopKSAE(cfg)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(DIM))
        sae.encoder.bias.zero_()
        sae.W_dec.copy_(torch.eye(DIM))
        sae.b_dec.zero_()
    ckpt = tmp_path / "final"
    sae.save_pretrained(ckpt)
    return ckpt


def test_recon_of_a_nonnegative_input_is_zero(tmp_path: Path) -> None:
    """A relu autoencoder rebuilds a non-negative input exactly."""
    n = 10
    vecs = np.abs(np.random.default_rng(0).normal(size=(n, DIM))).astype(np.float32)
    keys = [f"{i}_0" for i in range(n)]
    cache_dir = tmp_path / "coco"
    save_stacked(cache_dir, image_emb=vecs, text_emb=vecs, keys=keys,
                 splits={"test": keys}, meta={"dim": DIM})

    result = run_recon(ckpt=_identity_sae(tmp_path), method="shared",
                       cache_dir=cache_dir, output=tmp_path / "recon.json",
                       dataset="coco", split="test", device="cpu")
    assert result["recon_error"] < 1e-10
    assert result["n"] == n


def test_recon_is_the_paper_formula(tmp_path: Path) -> None:
    """recon = 0.5 * mean over pairs of (||x - x_hat||^2 + ||y - y_hat||^2)."""
    n = 12
    gen = np.random.default_rng(2)
    image = gen.normal(size=(n, DIM)).astype(np.float32)
    text = gen.normal(size=(n, DIM)).astype(np.float32)
    keys = [f"{i}_0" for i in range(n)]
    cache_dir = tmp_path / "coco"
    save_stacked(cache_dir, image_emb=image, text_emb=text, keys=keys,
                 splits={"test": keys}, meta={"dim": DIM})

    result = run_recon(ckpt=_identity_sae(tmp_path), method="shared",
                       cache_dir=cache_dir, output=tmp_path / "recon.json",
                       dataset="coco", split="test", device="cpu")

    ds = eval_utils.load_paired_split(cache_dir, "test")
    expected_img = float((ds.image - torch.relu(ds.image)).pow(2).sum(-1).mean())
    expected_txt = float((ds.text - torch.relu(ds.text)).pow(2).sum(-1).mean())
    assert result["recon_image"] == pytest.approx(expected_img, rel=1e-5)
    assert result["recon_text"] == pytest.approx(expected_txt, rel=1e-5)
    assert result["recon_error"] == pytest.approx(0.5 * (expected_img + expected_txt),
                                                  rel=1e-5)


def test_imagenet_recon_draws_one_template_of_the_true_class(tmp_path: Path) -> None:
    """The paper pairs each validation image with a random template of its class."""
    make_imagenet_cache(tmp_path, n_images=20, n_classes=4, n_templates=5, dim=DIM)
    inet = eval_utils.load_imagenet_split(tmp_path)
    picked = _imagenet_pairs(inet, seed=0)
    assert picked.shape == (20, DIM)

    # Every picked row has to be one of that image's own class's templates.
    grid = inet.text.reshape(inet.n_classes, inet.n_templates, -1)
    for i, label in enumerate(inet.labels):
        options = grid[int(label)]
        assert any(torch.allclose(picked[i], opt) for opt in options)


def test_imagenet_recon_is_reproducible_under_a_seed(tmp_path: Path) -> None:
    make_imagenet_cache(tmp_path, n_images=16, n_classes=4, n_templates=5, dim=DIM)
    inet = eval_utils.load_imagenet_split(tmp_path)
    a = _imagenet_pairs(inet, seed=3)
    b = _imagenet_pairs(inet, seed=3)
    c = _imagenet_pairs(inet, seed=4)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_zeroshot_raw_keeps_every_latent(tmp_path: Path) -> None:
    """The raw variant applies no mask; Table 1 reports this one."""
    make_imagenet_cache(tmp_path / "inet", n_images=12, n_classes=3, n_templates=4, dim=DIM)
    result = run_zeroshot(ckpt=_identity_sae(tmp_path), method="shared",
                          cache_dir=tmp_path / "inet", output=tmp_path / "zs.json",
                          variant="raw", device="cpu")
    assert result["variant"] == "raw"
    assert result["metric"] == "zeroshot_top1"
    assert result["kept_latents"] == result["total_latents"]
    assert "max_fire_rate" not in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n_classes"] == 3 and result["n_templates"] == 4


def test_zeroshot_filtered_can_drop_latents(tmp_path: Path) -> None:
    """The filtered variant removes columns that fire on too many images."""
    make_imagenet_cache(tmp_path / "inet", n_images=12, n_classes=3, n_templates=4, dim=DIM)
    result = run_zeroshot(ckpt=_identity_sae(tmp_path), method="shared",
                          cache_dir=tmp_path / "inet", output=tmp_path / "zsf.json",
                          variant="filtered", max_fire_rate=0.1, device="cpu")
    assert result["variant"] == "filtered"
    assert result["metric"] == "zeroshot_top1_filtered"
    assert result["max_fire_rate"] == 0.1
    # relu of a standard normal fires on about half the rows, so a 0.1 cap
    # removes essentially every column.
    assert result["kept_latents"] < result["total_latents"]


def test_class_prototypes_average_the_templates(tmp_path: Path) -> None:
    """Prototype c is the L2-normalized mean of class c's templates."""
    from src.eval.zeroshot import _class_prototypes

    make_imagenet_cache(tmp_path, n_images=6, n_classes=3, n_templates=4, dim=DIM)
    inet = eval_utils.load_imagenet_split(tmp_path)
    protos = _class_prototypes(inet)
    assert protos.shape == (3, DIM)
    raw = load_imagenet_cache(tmp_path)["text"]
    manual = torch.from_numpy(np.array(raw[:4], dtype=np.float32))
    manual = manual / manual.norm(dim=-1, keepdim=True)
    manual = manual.mean(0)
    manual = manual / manual.norm()
    torch.testing.assert_close(protos[0], manual, rtol=1e-5, atol=1e-6)

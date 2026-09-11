"""Retrieval on a cache whose answer is known by construction.

When the caption embedding equals its own image embedding and the same SAE
encodes both sides, the correct partner scores a cosine of exactly 1 and every
other candidate scores less, so both directions must return Recall@1 of 1.0.
Anything lower means the protocol, not the data, lost the answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.cache_io import save_stacked
from src.eval.retrieval import run as run_retrieval
from src.models import TopKSAE, TopKSAEConfig

DIM = 16


def _identity_sae(tmp_path: Path) -> Path:
    """A TopKSAE that passes its input through: z = relu(x), x_hat = relu(x).

    Making the encoder the identity and keeping every latent removes the SAE
    from the question, so the test measures the ranking rule alone.
    """
    cfg = TopKSAEConfig(hidden_size=DIM, latent_size=DIM, k=DIM,
                        normalize_decoder=False)
    sae = TopKSAE(cfg)
    with torch.no_grad():
        sae.encoder.weight.copy_(torch.eye(DIM))
        sae.encoder.bias.zero_()
        sae.W_dec.copy_(torch.eye(DIM))
        sae.b_dec.zero_()
    ckpt = tmp_path / "final"
    sae.save_pretrained(ckpt)
    return ckpt


def _cache_text_equals_image(cache_dir: Path, n_images: int = 12) -> None:
    gen = np.random.default_rng(0)
    vecs = gen.normal(size=(n_images, DIM)).astype(np.float32)
    keys = [f"{i}_0" for i in range(n_images)]
    save_stacked(cache_dir, image_emb=vecs, text_emb=vecs, keys=keys,
                 splits={"test": keys}, meta={"dim": DIM, "dataset": "coco"})


def test_identical_embeddings_give_perfect_recall(tmp_path: Path) -> None:
    cache_dir = tmp_path / "coco"
    _cache_text_equals_image(cache_dir)
    ckpt = _identity_sae(tmp_path)
    out = tmp_path / "retrieval.json"

    result = run_retrieval(ckpt=ckpt, method="shared", cache_dir=cache_dir,
                           output=out, split="test", device="cpu")
    assert result["I2T"]["R@1"] == 1.0
    assert result["T2I"]["R@1"] == 1.0
    assert result["n_images"] == 12
    assert result["n_captions"] == 12
    # A cosine of 1 is reached only by the true partner, so nothing is tied.
    assert result["T2I_ties"]["mean_tie_size"] == 1.0
    assert json.loads(out.read_text())["I2T"]["R@10"] == 1.0


def test_unique_images_come_from_the_key_prefix(tmp_path: Path) -> None:
    """Five captions of one photo must collapse to one retrieval candidate."""
    gen = np.random.default_rng(1)
    n_images, caps = 6, 5
    img_vecs = gen.normal(size=(n_images, DIM)).astype(np.float32)
    image = np.repeat(img_vecs, caps, axis=0)
    text = gen.normal(size=(n_images * caps, DIM)).astype(np.float32)
    keys = [f"{i}_{c}" for i in range(n_images) for c in range(caps)]
    cache_dir = tmp_path / "coco"
    save_stacked(cache_dir, image_emb=image, text_emb=text, keys=keys,
                 splits={"test": keys}, meta={"dim": DIM, "dataset": "coco"})

    result = run_retrieval(ckpt=_identity_sae(tmp_path), method="shared",
                           cache_dir=cache_dir, output=tmp_path / "r.json",
                           split="test", device="cpu")
    assert result["n_images"] == n_images
    assert result["n_captions"] == n_images * caps
    for direction in ("I2T", "T2I"):
        for k in ("R@1", "R@5", "R@10"):
            assert 0.0 <= result[direction][k] <= 1.0


def test_ranking_is_pessimistic_about_ties(tmp_path: Path) -> None:
    """Every candidate tied with the ground truth counts against it.

    All images share one embedding, so every candidate ties. A pessimistic rank
    puts the ground truth last, which is Recall@1 of 0; an optimistic rule would
    report 1.0 and hide a collapsed representation.
    """
    n = 8
    vec = np.ones((n, DIM), dtype=np.float32)
    keys = [f"{i}_0" for i in range(n)]
    cache_dir = tmp_path / "coco"
    save_stacked(cache_dir, image_emb=vec, text_emb=vec, keys=keys,
                 splits={"test": keys}, meta={"dim": DIM, "dataset": "coco"})

    result = run_retrieval(ckpt=_identity_sae(tmp_path), method="shared",
                           cache_dir=cache_dir, output=tmp_path / "r.json",
                           split="test", device="cpu")
    assert result["T2I"]["R@1"] == 0.0
    assert result["T2I_ties"]["mean_tie_size"] == float(n)

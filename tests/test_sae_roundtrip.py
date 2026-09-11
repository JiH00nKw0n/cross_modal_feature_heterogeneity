"""A trained checkpoint survives save_pretrained and from_pretrained unchanged.

There is one SAE implementation in this repository. This proves that what the
trainer writes is exactly what the evaluators and the panel builder read back,
so a difference between a training-time number and an evaluation-time number
cannot be blamed on the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.eval import eval_utils
from src.models import TwoSidedTopKSAE, TwoSidedTopKSAEConfig

DIM = 12
LATENT_TOTAL = 8


def _train_two_steps(model: TwoSidedTopKSAE, steps: int = 2) -> None:
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(steps):
        img = torch.randn(16, DIM)
        txt = torch.randn(16, DIM)
        out = model(image_embeds=img, text_embeds=txt)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        model.set_decoder_norm_to_unit_norm()


def test_checkpoint_round_trip_is_bit_identical(tmp_path: Path) -> None:
    torch.manual_seed(1)
    cfg = TwoSidedTopKSAEConfig(hidden_size=DIM, latent_size=LATENT_TOTAL, k=2,
                                normalize_decoder=True)
    model = TwoSidedTopKSAE(cfg)
    _train_two_steps(model)
    model.eval()

    ckpt = tmp_path / "final"
    model.save_pretrained(ckpt)
    reloaded = TwoSidedTopKSAE.from_pretrained(ckpt).eval()

    probe_img = torch.randn(7, DIM)
    probe_txt = torch.randn(7, DIM)
    with torch.no_grad():
        a = model(image_embeds=probe_img, text_embeds=probe_txt)
        b = reloaded(image_embeds=probe_img, text_embeds=probe_txt)
    torch.testing.assert_close(a.image_output, b.image_output, rtol=0, atol=0)
    torch.testing.assert_close(a.text_output, b.text_output, rtol=0, atol=0)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)


def test_latent_size_is_the_total_budget_split_in_half() -> None:
    cfg = TwoSidedTopKSAEConfig(hidden_size=DIM, latent_size=LATENT_TOTAL, k=2)
    model = TwoSidedTopKSAE(cfg)
    assert model.image_sae.latent_size == LATENT_TOTAL // 2
    assert model.text_sae.latent_size == LATENT_TOTAL // 2


def test_eval_utils_loads_the_same_class(tmp_path: Path) -> None:
    """eval_utils reads checkpoints from src.models, not a second copy."""
    cfg = TwoSidedTopKSAEConfig(hidden_size=DIM, latent_size=LATENT_TOTAL, k=2)
    model = TwoSidedTopKSAE(cfg)
    ckpt = tmp_path / "final"
    model.save_pretrained(ckpt)
    for method in ("separated", "ours"):
        loaded = eval_utils.load_sae(ckpt, method)
        assert isinstance(loaded, TwoSidedTopKSAE)


def test_dense_latents_have_exactly_k_nonzeros() -> None:
    cfg = TwoSidedTopKSAEConfig(hidden_size=DIM, latent_size=LATENT_TOTAL, k=3)
    model = TwoSidedTopKSAE(cfg).eval()
    x = torch.randn(5, DIM)
    with torch.no_grad():
        out = model.image_sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
    dense = out.dense_latents.squeeze(1)
    assert dense.shape == (5, LATENT_TOTAL // 2)
    assert (dense != 0).sum(dim=1).max().item() <= 3

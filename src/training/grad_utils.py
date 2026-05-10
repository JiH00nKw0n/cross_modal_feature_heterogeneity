"""Decoder-direction gradient projection for SAE training.

Mirrors the synthetic pipeline (`src/runners/synthetic_trainers._project_sae_gradients`)
so that real-data trainers obtain the same EleutherAI-style decoder discipline:
remove the gradient component parallel to each `W_dec[k]` row before the
optimizer step, then `SAENormCallback` snaps rows back to unit norm post-step.
"""

from __future__ import annotations


def project_sae_decoder_grads(model) -> None:
    """Zero out the W_dec.grad component parallel to current W_dec rows.

    Handles both single SAEs (`model.W_dec`) and two-sided SAEs
    (`model.image_sae.W_dec`, `model.text_sae.W_dec`). Skipped when
    `cfg.weight_tie` is True, since decoder weights share storage with the
    encoder and projection would corrupt encoder gradients.
    """
    if hasattr(model, "image_sae"):  # TwoSidedTopKSAE
        for sae in (model.image_sae, model.text_sae):
            if getattr(getattr(sae, "cfg", None), "weight_tie", False):
                continue
            if getattr(sae, "W_dec", None) is not None and sae.W_dec.grad is not None:
                sae.remove_gradient_parallel_to_decoder_directions()
        return
    if getattr(model, "W_dec", None) is not None and model.W_dec.grad is not None:
        if getattr(getattr(model, "cfg", None), "weight_tie", False):
            return
        model.remove_gradient_parallel_to_decoder_directions()

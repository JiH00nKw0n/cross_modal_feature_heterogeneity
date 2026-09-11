"""Trainer callbacks for SAE training."""

from __future__ import annotations

from transformers import TrainerCallback


class SAENormCallback(TrainerCallback):
    """Post-step: set decoder rows to unit norm.

    Always applied unconditionally (``normalize_decoder=True`` is the only
    mode we support).
    """

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if hasattr(model, "image_sae"):  # TwoSidedTopKSAE
            model.image_sae.set_decoder_norm_to_unit_norm()
            model.text_sae.set_decoder_norm_to_unit_norm()
        else:
            model.set_decoder_norm_to_unit_norm()

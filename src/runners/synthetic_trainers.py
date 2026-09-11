"""HF Trainer subclasses for synthetic Theorem 2 experiments.

Separate from ``trainer.py`` to avoid modifying the existing file.

Trainers:
- ``SingleReconTrainer``: one shared SAE, recon only (method: single_recon)
- ``TwoReconTrainer``: TwoSidedTopKSAE, recon only (method: two_recon)
- ``PairedAuxTrainer``: shared SAE + aux loss (methods: group_sparse, trace_align, iso_align)

All include SAE-specific gradient projection in ``training_step`` and
should be used with ``SAENormCallback`` for post-step decoder normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from transformers import Trainer, TrainerCallback

from src.configs.experiment import MethodConfig
from src.metrics.alignment import compute_latent_correlation
from src.training.losses import (
    auxiliary_alignment_loss,
    barlow_twins_aux_loss_masked,
    naive_diag_aux_loss_masked,
    slot_infonce_loss,
)


def _batch_cross_corr_detached(z_i: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Per-batch standardized cross-correlation (no gradient)."""
    with torch.no_grad():
        zi = z_i - z_i.mean(dim=0, keepdim=True)
        zt = z_t - z_t.mean(dim=0, keepdim=True)
        si = zi.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
        st = zt.pow(2).mean(dim=0).sqrt().clamp(min=1e-6)
        n_batch = zi.shape[0]
        return (zi / si).t() @ (zt / st) / max(n_batch, 1)
from src.training.permutation import (
    apply_latent_permutation,
    greedy_permutation_match_full,
    partitioned_hungarian_on_unsettled,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _project_sae_gradients(model) -> None:
    """Remove gradient components parallel to decoder directions.

    Skipped when ``cfg.weight_tie`` is True: W_dec shares memory with
    encoder.weight, so projecting W_dec.grad would also mutate encoder.grad.
    """
    if hasattr(model, "image_sae"):  # TwoSidedTopKSAE
        for sae in (model.image_sae, model.text_sae):
            if getattr(sae.cfg, "weight_tie", False):
                continue
            if sae.W_dec.grad is not None:
                sae.remove_gradient_parallel_to_decoder_directions()
    elif hasattr(model, "W_dec") and model.W_dec.grad is not None:
        if getattr(getattr(model, "cfg", None), "weight_tie", False):
            return
        model.remove_gradient_parallel_to_decoder_directions()


def _sae_training_step(trainer: Trainer, model, inputs) -> torch.Tensor:
    """Common training_step with gradient projection."""
    model.train()
    loss = trainer.compute_loss(model, inputs)
    if trainer.args.gradient_accumulation_steps > 1:
        loss = loss / trainer.args.gradient_accumulation_steps
    loss.backward()
    _project_sae_gradients(model)
    return loss.detach()


# ------------------------------------------------------------------
# SingleReconTrainer (method: single_recon)
# ------------------------------------------------------------------


class SingleReconTrainer(Trainer):
    """Shared SAE forwarded twice (image + text), mean recon loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)
        out_i = model(hidden_states=hs_i)
        out_t = model(hidden_states=hs_t)
        loss = out_i.recon_loss + out_t.recon_loss
        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# TwoReconTrainer (method: two_recon)
# ------------------------------------------------------------------


class TwoReconTrainer(Trainer):
    """TwoSidedTopKSAE: disjoint image/text SAEs, recon only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)
        out_i = model.image_sae(hidden_states=hs_i)
        out_t = model.text_sae(hidden_states=hs_t)
        loss = out_i.recon_loss + out_t.recon_loss
        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# PairedAuxTrainer (methods: group_sparse, trace_align, iso_align)
# ------------------------------------------------------------------


class PairedAuxTrainer(Trainer):
    """Shared SAE + configurable auxiliary loss on dense paired latents.

    *aux_fn(z_img, z_txt) -> scalar* and *aux_weight* are passed at init.
    """

    def __init__(self, *args, aux_fn=None, aux_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_fn = aux_fn
        self.aux_weight = aux_weight
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)

        need_dense = self.aux_fn is not None
        out_i = model(hidden_states=hs_i, return_dense_latents=need_dense)
        out_t = model(hidden_states=hs_t, return_dense_latents=need_dense)

        loss = out_i.recon_loss + out_t.recon_loss
        if self.aux_fn is not None and need_dense:
            z_i = out_i.dense_latents.squeeze(1)
            z_t = out_t.dense_latents.squeeze(1)
            aux = self.aux_fn(z_i, z_t)
            loss = loss + self.aux_weight * aux

        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# OursTrainer (method: ours — Algorithm 1, two-stage)
# ------------------------------------------------------------------


class PermutationCallback(TrainerCallback):
    """Fires once at epoch ``k_align`` to permute latent indices."""

    def __init__(self, ours_trainer: "OursTrainer") -> None:
        self.ours = ours_trainer
        self._done = False

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        completed = round(state.epoch)
        if self._done or completed != self.ours.k_align:
            return
        self._done = True

        device = next(model.parameters()).device
        batch_size = args.per_device_train_batch_size

        logger.info("Permutation matching at epoch %d (stage 1 -> 2)", completed)
        C = compute_latent_correlation(
            model.image_sae, model.text_sae,
            self.ours.train_img, self.ours.train_txt,
            batch_size, device,
        )
        P_I, P_T, ordered = greedy_permutation_match_full(C)

        # Unwrap accelerator optimizer if needed
        opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
        apply_latent_permutation(model.image_sae, P_I, optimizer=opt)
        apply_latent_permutation(model.text_sae, P_T, optimizer=opt)

        m_S = self.ours.m_S
        rho = self.ours.rho
        m_S_hat = int((ordered > rho).sum())
        self.ours.diagnostics = {
            "m_S_hat_rho": m_S_hat,
            "m_S_supplied": m_S,
            "first_drop_below_rho": float(ordered[m_S_hat]) if m_S_hat < len(ordered) else float("nan"),
            "top_mS_diag_mean": float(ordered[:m_S].mean()) if m_S > 0 else float("nan"),
            "ordered_diag_head": [float(x) for x in ordered[:8]],
            "k_align": self.ours.k_align,
            "lambda_aux": self.ours.lambda_aux,
            "aux_norm": self.ours.aux_norm,
        }
        logger.info("Permutation done: m_S_hat=%d, top diag=%s",
                     m_S_hat, [f"{x:.3f}" for x in ordered[:4]])
        self.ours._stage = 2


class OursTrainer(Trainer):
    """Algorithm 1: stage-aware Trainer with mid-training permutation.

    Stage 1 (epoch < k_align): recon loss only on TwoSidedTopKSAE.
    Stage 2 (epoch >= k_align): recon + lambda_aux * alignment loss.

    Use with ``PermutationCallback(trainer)`` and ``SAENormCallback``.
    """

    def __init__(
        self, *args,
        k_align: int = 6,
        lambda_aux: float = 1.0,
        m_S: int = 512,
        aux_norm: str = "group",
        train_img: Optional[torch.Tensor] = None,
        train_txt: Optional[torch.Tensor] = None,
        rho: float = 0.3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_align = k_align
        self.lambda_aux = lambda_aux
        self.m_S = m_S
        self.aux_norm = aux_norm
        self.train_img = train_img
        self.train_txt = train_txt
        self.rho = rho
        self._stage = 1
        self.diagnostics: dict[str, Any] = {}
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)

        need_dense = self._stage == 2
        out_i = model.image_sae(hidden_states=hs_i, return_dense_latents=need_dense)
        out_t = model.text_sae(hidden_states=hs_t, return_dense_latents=need_dense)

        loss = out_i.recon_loss + out_t.recon_loss

        if self._stage == 2 and need_dense:
            z_i = out_i.dense_latents.squeeze(1)
            z_t = out_t.dense_latents.squeeze(1)
            aux = auxiliary_alignment_loss(z_i, z_t, self.m_S, norm=self.aux_norm)
            loss = loss + self.lambda_aux * aux

        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ------------------------------------------------------------------
# AuxAlignmentTrainer (method: paired_aux_alignment)
# ------------------------------------------------------------------


class AuxAlignmentTrainer(Trainer):
    """Generic trainer for ablating (loss form, hungarian schedule, revive).

    Internal state:
      - ``self.variant`` : ``MethodConfig`` carrying aux_loss / hungarian_schedule
        / revive_dead / rho0 / lambda_aux / barlow_lambda_off / k_align.
      - ``self.frozen_mask`` : bool tensor [n] - slots in F (positive InfoNCE
        pressure). Maintained by ``AuxAlignmentCallback``.
      - ``self.fire_count_i`` / ``self.fire_count_t`` : long tensor [n], number
        of non-zero activations during the current epoch (for dead detection).
        Reset by callback after each on_epoch_end.
      - ``self.model.log_tau`` : nn.Parameter (only when aux_loss == "infonce").
        Attached to model so it joins ``model.parameters()`` and is optimised.
      - ``self.diagnostics`` : updated each epoch by callback.
    """

    def __init__(
        self,
        *args,
        variant_cfg: MethodConfig,
        train_img: Optional[torch.Tensor] = None,
        train_txt: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.variant = variant_cfg
        self.train_img = train_img
        self.train_txt = train_txt
        self.diagnostics: dict[str, Any] = {}
        self.model_accepts_loss_kwargs = False

        device = next(self.model.parameters()).device
        n = int(self.model.image_sae.latent_size)
        self.frozen_mask: torch.Tensor = torch.zeros(n, dtype=torch.bool, device=device)
        self.fire_count_i: torch.Tensor = torch.zeros(n, dtype=torch.long, device=device)
        self.fire_count_t: torch.Tensor = torch.zeros(n, dtype=torch.long, device=device)
        # Alive masks used to restrict InfoNCE negative pool. Initialized to
        # "all alive" (no masking) so the first epoch is unrestricted; updated
        # by AuxAlignmentCallback.on_epoch_end using the prior epoch's fire counts.
        self.alive_mask_i: torch.Tensor = torch.ones(n, dtype=torch.bool, device=device)
        self.alive_mask_t: torch.Tensor = torch.ones(n, dtype=torch.bool, device=device)
        # EMA of batch cross-correlation. Updated every step; used by the
        # callback to make gate/Hungarian decisions at epoch end without an
        # extra forward pass over the full train set.
        self.ema_C: Optional[torch.Tensor] = None
        self.ema_momentum: float = 0.99

        if variant_cfg.aux_loss == "infonce":
            init_log_tau = float(np.log(1.0 / 0.07))
            self.model.log_tau = torch.nn.Parameter(
                torch.tensor(init_log_tau, dtype=torch.float32, device=device)
            )

    def _aux_loss_value(self, z_i: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        loss_kind = self.variant.aux_loss
        if loss_kind == "none":
            return z_i.new_tensor(0.0)
        # No frozen slots yet -> aux loss is 0 (auto warm-up).
        if not bool(self.frozen_mask.any()):
            return z_i.new_tensor(0.0)
        if loss_kind == "naive_diag":
            return naive_diag_aux_loss_masked(z_i, z_t, self.frozen_mask)
        if loss_kind == "barlow":
            return barlow_twins_aux_loss_masked(
                z_i, z_t, self.frozen_mask,
                lambda_off=float(self.variant.barlow_lambda_off),
            )
        if loss_kind == "infonce":
            return slot_infonce_loss(
                z_i, z_t, self.frozen_mask, self.model.log_tau,
                alive_mask_i=self.alive_mask_i,
                alive_mask_t=self.alive_mask_t,
            )
        raise ValueError(f"Unknown aux_loss '{loss_kind}'")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        hs_i = inputs["image_embeds"].unsqueeze(1)
        hs_t = inputs["text_embeds"].unsqueeze(1)

        out_i = model.image_sae(hidden_states=hs_i, return_dense_latents=True)
        out_t = model.text_sae(hidden_states=hs_t, return_dense_latents=True)
        z_i = out_i.dense_latents.squeeze(1)
        z_t = out_t.dense_latents.squeeze(1)

        loss = out_i.recon_loss + out_t.recon_loss
        aux = self._aux_loss_value(z_i, z_t)
        loss = loss + float(self.variant.lambda_aux) * aux

        with torch.no_grad():
            self.fire_count_i += (z_i > 0).sum(dim=0).to(self.fire_count_i.dtype)
            self.fire_count_t += (z_t > 0).sum(dim=0).to(self.fire_count_t.dtype)
            # EMA of batch cross-correlation -- eliminates the full-data forward
            # pass previously performed at on_epoch_end for schedule=per_epoch.
            if self.variant.hungarian_schedule == "per_epoch_partitioned":
                batch_C = _batch_cross_corr_detached(z_i, z_t)
                if self.ema_C is None:
                    self.ema_C = batch_C
                else:
                    m = self.ema_momentum
                    self.ema_C.mul_(m).add_(batch_C, alpha=1.0 - m)

        return (loss, (out_i, out_t)) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        return _sae_training_step(self, model, inputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


def _reinit_sae_slots(sae, idx: np.ndarray) -> None:
    """Re-init encoder + decoder columns at ``idx`` to small random values."""
    if idx.size == 0:
        return
    device = sae.W_dec.device
    dtype = sae.W_dec.dtype
    d = sae.W_dec.shape[1]
    with torch.no_grad():
        new_dec = torch.randn(idx.size, d, device=device, dtype=dtype)
        new_dec = new_dec / new_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sae.W_dec.data[idx] = new_dec
        new_enc = torch.randn(idx.size, d, device=device, dtype=dtype) * 0.01
        sae.encoder.weight.data[idx] = new_enc
        sae.encoder.bias.data[idx] = 0.0


class AuxAlignmentCallback(TrainerCallback):
    """End-of-epoch hook: gate -> Hungarian -> (optional revive).

    Behaviour split by ``trainer.variant.hungarian_schedule``:

    * ``"once"`` : fires only once, at epoch ``trainer.variant.k_align``.
      Greedy permutation on full C, applied; ``frozen_mask`` set to top-m_S_hat
      contiguous prefix and never changes again.

    * ``"per_epoch_partitioned"`` : fires every epoch.
      ``frozen_mask`` = (C.diag() > rho0); partitioned Hungarian on the
      unsettled submatrix; only unsettled positions of the text side permuted.

    Revive (``trainer.variant.revive_dead``):
      Slots that fired 0 times during the epoch and are NOT in F are
      re-initialised AFTER permutation.
    """

    def __init__(self, trainer: Any) -> None:
        # `trainer` is duck-typed: must expose `variant`, `frozen_mask`,
        # `fire_count_i`, `fire_count_t`, `train_img`, `train_txt`, `diagnostics`.
        # Both AuxAlignmentTrainer (synthetic) and RealAuxAlignmentTrainer satisfy this.
        self.trainer = trainer
        self._once_done = False

    def on_epoch_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        v = self.trainer.variant
        schedule = v.hungarian_schedule
        if schedule == "none":
            return

        completed = int(round(state.epoch))
        if schedule == "once":
            if self._once_done or completed != int(v.k_align):
                return

        device = next(model.parameters()).device
        batch_size = args.per_device_train_batch_size

        if schedule == "per_epoch_partitioned" and self.trainer.ema_C is not None:
            # Use EMA of batch C (updated each step) to avoid an expensive
            # full-data forward pass every epoch. Stale by O(window) batches
            # but sufficient for gate/Hungarian decisions.
            C = self.trainer.ema_C.detach().cpu().numpy().astype(np.float64)
            logger.info("Using ema_C for per_epoch_partitioned schedule (no extra forward pass)")
        else:
            # Once-mode (or ema_C not yet initialized): do the full forward pass.
            C = compute_latent_correlation(
                model.image_sae, model.text_sae,
                self.trainer.train_img, self.trainer.train_txt,
                batch_size, device,
            )
        diag_before = np.diag(C).copy()

        opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer

        if schedule == "once":
            P_I, P_T, ordered = greedy_permutation_match_full(C)
            apply_latent_permutation(model.image_sae, P_I, optimizer=opt)
            apply_latent_permutation(model.text_sae, P_T, optimizer=opt)
            m_S_hat = int((ordered > float(v.rho0)).sum())
            n = int(model.image_sae.latent_size)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            if m_S_hat > 0:
                mask[:m_S_hat] = True
            self.trainer.frozen_mask = mask
            self._once_done = True
            self.trainer.diagnostics.update({
                "m_S_hat_rho": m_S_hat,
                "rho0": float(v.rho0),
                "ordered_diag_head": [float(x) for x in ordered[:8]],
                "ordered_diag_tail": [float(x) for x in ordered[-4:]],
                "epoch_when_fired": completed,
            })
            logger.info(
                "Once-Hungarian@epoch%d: m_S_hat=%d, top diag=%s",
                completed, m_S_hat, [f"{x:.3f}" for x in ordered[:4]],
            )
        else:
            mask_arr = (diag_before > float(v.rho0))
            unsettled_idx = np.where(~mask_arr)[0]
            if unsettled_idx.size > 0:
                _, P_T_full = partitioned_hungarian_on_unsettled(C, unsettled_idx)
                apply_latent_permutation(model.text_sae, P_T_full, optimizer=opt)
            new_mask = torch.from_numpy(mask_arr.astype(bool)).to(device=device)
            self.trainer.frozen_mask = new_mask
            self.trainer.diagnostics.setdefault("m_S_hat_per_epoch", []).append(int(mask_arr.sum()))
            logger.info(
                "PerEpoch-Hungarian@epoch%d: |F|=%d, |U|=%d",
                completed, int(mask_arr.sum()), int((~mask_arr).sum()),
            )

        if bool(v.revive_dead):
            mask_np = self.trainer.frozen_mask.detach().cpu().numpy()
            dead_i_idx = np.where(
                (self.trainer.fire_count_i.detach().cpu().numpy() == 0) & (~mask_np)
            )[0]
            dead_t_idx = np.where(
                (self.trainer.fire_count_t.detach().cpu().numpy() == 0) & (~mask_np)
            )[0]
            _reinit_sae_slots(model.image_sae, dead_i_idx)
            _reinit_sae_slots(model.text_sae, dead_t_idx)
            self.trainer.diagnostics.setdefault("revived_per_epoch", []).append(
                {"img": int(dead_i_idx.size), "txt": int(dead_t_idx.size)}
            )

        # Snapshot "alive" masks from this epoch's fire counts BEFORE resetting.
        # These feed into next epoch's InfoNCE negative pool.
        self.trainer.alive_mask_i = (self.trainer.fire_count_i > 0).clone()
        self.trainer.alive_mask_t = (self.trainer.fire_count_t > 0).clone()

        self.trainer.fire_count_i.zero_()
        self.trainer.fire_count_t.zero_()

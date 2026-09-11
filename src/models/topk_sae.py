"""TopKSAE + TwoSidedTopKSAE — minimal SAE used by the new train_method dispatch.

Reproduces the SAE used in the paper:
- TopK activation (k non-zero latents per sample)
- Decoder column unit-norm enforced via callback / step hook
- AuxK loss for dead-feature revival (Gao 2024)

Only two classes; VL/Matryoshka/BatchTopK variants intentionally dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import einops
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_sae import TopKSAEConfig, TwoSidedTopKSAEConfig


def _decoder_impl(top_indices: Tensor, top_activations: Tensor, decoder_weight: Tensor) -> Tensor:
    """Sparse decode: scatter top-k acts into dense, then project with W_dec."""
    dense = top_activations.new_zeros(top_activations.shape[:-1] + (decoder_weight.shape[0],))
    dense = dense.scatter_(dim=-1, index=top_indices, src=top_activations)
    return dense @ decoder_weight


def _sanitize_topk_acts(acts: Tensor) -> Tensor:
    return torch.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class SAEOutput(ModelOutput):
    output: Tensor
    latent_activations: Optional[Tensor] = None
    latent_indices: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    auxk_loss: Optional[Tensor] = None
    dense_latents: Optional[Tensor] = None


@dataclass
class TwoSidedSAEOutput(ModelOutput):
    loss: Optional[Tensor] = None
    recon_loss: Optional[Tensor] = None
    recon_loss_image: Optional[Tensor] = None
    recon_loss_text: Optional[Tensor] = None
    image_output: Optional[Tensor] = None
    text_output: Optional[Tensor] = None


class TopKSAE(PreTrainedModel):
    """Single-modality TopK SAE."""

    config_class = TopKSAEConfig
    base_model_prefix = "sae"

    def __init__(self, config: TopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        self.hidden_size = config.hidden_size
        self.latent_size = config.latent_size or self.hidden_size * config.expansion_factor

        self.encoder = nn.Linear(self.hidden_size, self.latent_size)
        self.encoder.bias.data.zero_()

        if self.cfg.weight_tie:
            self.W_dec = self.encoder.weight
        else:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(self.hidden_size))
        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        sae_in = x.to(self.dtype) - self.b_dec
        return nn.functional.relu(self.encoder(sae_in))

    def select_topk(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        return latents.topk(self.cfg.k, sorted=False)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec)
        return y + self.b_dec

    def forward(
        self,
        hidden_states: Tensor,
        dead_mask: Optional[Tensor] = None,
        return_dense_latents: bool = False,
    ) -> SAEOutput:
        x = hidden_states
        pre_acts = self.pre_acts(x)
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        residual = sae_out - x

        total_variance = (x - x.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            cfg_k_aux = getattr(self.cfg, "k_aux", None)
            k_aux = int(cfg_k_aux) if cfg_k_aux is not None else x.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_acts = _sanitize_topk_acts(auxk_acts)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = scale * (e_hat - residual).pow(2).sum() / total_variance
            auxk_loss = torch.nan_to_num(auxk_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        recon_loss = residual.pow(2).sum() / total_variance

        dense_latents = None
        if return_dense_latents:
            dense_latents = top_acts.new_zeros(top_acts.shape[:-1] + (self.latent_size,))
            dense_latents.scatter_(dim=-1, index=top_indices, src=top_acts)

        return SAEOutput(
            output=sae_out,
            latent_activations=top_acts,
            latent_indices=top_indices,
            recon_loss=recon_loss,
            auxk_loss=auxk_loss,
            dense_latents=dense_latents,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec.grad is not None
        parallel = einops.einsum(
            self.W_dec.grad, self.W_dec.data,
            "latent hidden, latent hidden -> latent",
        )
        self.W_dec.grad -= einops.einsum(
            parallel, self.W_dec.data,
            "latent, latent hidden -> latent hidden",
        )


class TwoSidedTopKSAE(PreTrainedModel):
    """Two disjoint TopKSAEs sharing no parameters: one image, one text."""

    config_class = TwoSidedTopKSAEConfig
    base_model_prefix = "two_sided_topk_sae"

    def __init__(self, config: TwoSidedTopKSAEConfig):
        super().__init__(config)
        self.cfg = config
        per_side = config.latent_size_per_side
        sub_config = TopKSAEConfig(
            hidden_size=config.hidden_size,
            latent_size=per_side,
            expansion_factor=1,
            normalize_decoder=config.normalize_decoder,
            k=config.k,
            weight_tie=getattr(config, "weight_tie", False),
            k_aux=getattr(config, "k_aux", None),
        )
        self.image_sae = TopKSAE(sub_config)
        self.text_sae = TopKSAE(TopKSAEConfig(**sub_config.to_dict()))
        self.post_init()

    def _init_weights(self, module: nn.Module):
        return

    def forward(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        **_: dict,
    ) -> TwoSidedSAEOutput:
        hs_i = image_embeds.unsqueeze(1) if image_embeds.dim() == 2 else image_embeds
        hs_t = text_embeds.unsqueeze(1) if text_embeds.dim() == 2 else text_embeds
        out_i = self.image_sae(hidden_states=hs_i)
        out_t = self.text_sae(hidden_states=hs_t)
        recon = (out_i.recon_loss + out_t.recon_loss) / 2
        return TwoSidedSAEOutput(
            loss=recon, recon_loss=recon,
            recon_loss_image=out_i.recon_loss, recon_loss_text=out_t.recon_loss,
            image_output=out_i.output, text_output=out_t.output,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.image_sae.set_decoder_norm_to_unit_norm()
        self.text_sae.set_decoder_norm_to_unit_norm()


__all__ = ["TopKSAE", "TwoSidedTopKSAE", "SAEOutput", "TwoSidedSAEOutput"]

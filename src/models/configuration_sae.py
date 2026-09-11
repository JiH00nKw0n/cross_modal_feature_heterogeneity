from __future__ import annotations

from transformers import PretrainedConfig


class TopKSAEConfig(PretrainedConfig):
    """
    Configuration for the TopKSAE model.

    This mirrors the settings used in the multimodal-sae implementation, while
    presenting them in a Hugging Face `PretrainedConfig` so the model can be
    saved/loaded consistently with the Transformers ecosystem.
    """

    model_type = "topk_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        weight_tie: bool = False,
        k_aux: int | None = None,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Input feature width of the activations to be autoencoded.
            latent_size: Explicit latent size; if 0, use hidden_size * expansion_factor.
            expansion_factor: Multiplier for latent width when latent_size is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            k: Number of non-zero latent activations (top-k) to keep per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            weight_tie: Whether to tie encoder and decoder weights (W_dec = W_enc).
            k_aux: AuxK top-k for dead-feature revival. If None, defaults to
                ``hidden_size // 2`` (Gao et al. 2024 heuristic).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.expansion_factor = expansion_factor
        self.normalize_decoder = normalize_decoder
        self.latent_size = latent_size
        self.k = k
        self.multi_topk = multi_topk
        self.weight_tie = weight_tie
        self.k_aux = k_aux


class TwoSidedTopKSAEConfig(PretrainedConfig):
    """
    Configuration for TwoSidedTopKSAE.

    Holds two independent TopKSAE stacks (image side + text side), each with
    ``latent_size // 2`` latents. Used for the real-data α-diagnostic
    experiment (Diagnostic A/B) where image and text CLIP embeddings are
    encoded by disjoint decoders so per-modality atoms are preserved.
    """

    model_type = "two_sided_topk_sae"

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 8192,
        k: int = 8,
        normalize_decoder: bool = True,
        multi_topk: bool = False,
        k_aux: int | None = None,
        weight_tie: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if latent_size % 2 != 0:
            raise ValueError(f"latent_size must be even, got {latent_size}")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.multi_topk = multi_topk
        self.k_aux = k_aux
        self.weight_tie = weight_tie

    @property
    def latent_size_per_side(self) -> int:
        return self.latent_size // 2


__all__ = ["TopKSAEConfig", "TwoSidedTopKSAEConfig"]

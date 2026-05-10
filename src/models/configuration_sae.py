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


class BatchTopKSAEConfig(TopKSAEConfig):
    """
    Configuration for the BatchTopKSAE model.

    BatchTopK uses a batch-level top-k selection. The same activation rule is
    applied in both training and evaluation.
    """

    model_type = "batch_topk_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        use_batch_topk_in_eval: bool = True,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Input feature width of the activations to be autoencoded.
            latent_size: Explicit latent size; if 0, use hidden_size * expansion_factor.
            expansion_factor: Multiplier for latent width when latent_size is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            use_batch_topk_in_eval: Kept for backward compatibility. BatchTopK
                always uses batch-level top-k regardless of this flag.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            hidden_size=hidden_size,
            latent_size=latent_size,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        self.use_batch_topk_in_eval = use_batch_topk_in_eval


class MatryoshkaSAEConfig(BatchTopKSAEConfig):
    """
    Configuration for MatryoshkaSAE.

    Matryoshka splits the dictionary into groups, and trains each prefix of
    groups to reconstruct well without access to later groups.
    """

    model_type = "matryoshka_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        group_sizes: list[int] = None,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        active_groups: int | None = None,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Input feature width of the activations to be autoencoded.
            group_sizes: Sizes of each Matryoshka group (sum = total latents).
            latent_size: Explicit latent size; if 0, use sum(group_sizes).
            expansion_factor: Multiplier for latent width when latent_size is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            active_groups: Number of prefix groups to activate (defaults to all).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            hidden_size=hidden_size,
            latent_size=latent_size,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        if group_sizes is None:
            group_sizes = []
        self.group_sizes = group_sizes
        self.active_groups = active_groups


class VLTopKSAEConfig(TopKSAEConfig):
    """
    Configuration for VLTopKSAE.

    The dictionary is split into [visual | shared | text] subspaces whose
    proportions are controlled by ``vl_split_ratio`` (default 1:2:1).
    """

    model_type = "vl_topk_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        vl_split_ratio: list[int] | None = None,
        block_top_k: list[int] | None = None,
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
            vl_split_ratio: [visual, shared, text] proportions (default [1, 2, 1]).
            block_top_k: [modality_specific, shared] ratio used to split k across
                blocks when selecting top-k latents in VLTopKSAE.
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            hidden_size=hidden_size,
            latent_size=latent_size,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        self.vl_split_ratio = vl_split_ratio or [1, 2, 1]
        self.block_top_k = block_top_k


class VLBatchTopKSAEConfig(BatchTopKSAEConfig):
    """
    Configuration for VLBatchTopKSAE.

    The dictionary is split into [visual | shared | text] subspaces whose
    proportions are controlled by ``vl_split_ratio`` (default 1:2:1).
    """

    model_type = "vl_batch_topk_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        vl_split_ratio: list[int] | None = None,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Input feature width of the activations to be autoencoded.
            latent_size: Explicit latent size; if 0, use hidden_size * expansion_factor.
            expansion_factor: Multiplier for latent width when latent_size is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            vl_split_ratio: [visual, shared, text] proportions (default [1, 2, 1]).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            hidden_size=hidden_size,
            latent_size=latent_size,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            k=k,
            multi_topk=multi_topk,
            **kwargs,
        )
        self.vl_split_ratio = vl_split_ratio or [1, 2, 1]


class VLMatryoshkaSAEConfig(MatryoshkaSAEConfig):
    """
    Configuration for VLMatryoshkaSAE.

    The dictionary is split into [visual | shared | text] subspaces whose
    proportions are controlled by ``vl_split_ratio`` (default 1:2:1),
    and Matryoshka prefix reconstruction losses are applied.
    """

    model_type = "vl_matryoshka_sae"

    def __init__(
        self,
        hidden_size: int = 4096,
        group_sizes: list[int] = None,
        latent_size: int = 131072,
        expansion_factor: int = 32,
        normalize_decoder: bool = True,
        k: int = 256,
        multi_topk: bool = False,
        active_groups: int | None = None,
        shared_group_sizes: list[int] | None = None,
        shared_active_groups: int | None = None,
        vl_split_ratio: list[int] | None = None,
        **kwargs,
    ):
        """
        Args:
            hidden_size: Input feature width of the activations to be autoencoded.
            group_sizes: Sizes of each Matryoshka group (sum = total latents).
            latent_size: Explicit latent size; if 0, use sum(group_sizes).
            expansion_factor: Multiplier for latent width when latent_size is 0.
            normalize_decoder: Whether to normalize decoder rows to unit norm.
            k: Target average number of active latents per sample.
            multi_topk: Whether to compute Multi-TopK FVU in the forward pass.
            active_groups: Number of prefix groups to activate (defaults to all).
            shared_group_sizes: Sizes of shared-subspace Matryoshka groups. If None,
                defaults to a single shared group.
            shared_active_groups: Number of shared groups to activate (defaults to all).
            vl_split_ratio: [visual, shared, text] proportions (default [1, 2, 1]).
            **kwargs: Additional config args passed to PretrainedConfig.
        """
        super().__init__(
            hidden_size=hidden_size,
            latent_size=latent_size,
            group_sizes=group_sizes,
            expansion_factor=expansion_factor,
            normalize_decoder=normalize_decoder,
            k=k,
            multi_topk=multi_topk,
            active_groups=active_groups,
            **kwargs,
        )
        if group_sizes is None:
            group_sizes = []
        self.shared_group_sizes = shared_group_sizes
        self.shared_active_groups = shared_active_groups
        self.vl_split_ratio = vl_split_ratio or [1, 2, 1]


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


class TwoSidedBatchTopKSAEConfig(PretrainedConfig):
    """Two independent BatchTopKSAE stacks, each ``latent_size // 2`` wide."""

    model_type = "two_sided_batch_topk_sae"

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 8192,
        k: int = 8,
        normalize_decoder: bool = True,
        multi_topk: bool = False,
        weight_tie: bool = False,
        use_batch_topk_in_eval: bool = True,
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
        self.weight_tie = weight_tie
        self.use_batch_topk_in_eval = use_batch_topk_in_eval

    @property
    def latent_size_per_side(self) -> int:
        return self.latent_size // 2


__all__ = [
    "TopKSAEConfig",
    "BatchTopKSAEConfig",
    "MatryoshkaSAEConfig",
    "VLTopKSAEConfig",
    "VLBatchTopKSAEConfig",
    "VLMatryoshkaSAEConfig",
    "TwoSidedTopKSAEConfig",
    "TwoSidedBatchTopKSAEConfig",
]

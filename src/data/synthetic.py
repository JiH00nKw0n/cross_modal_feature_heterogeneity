"""Simplified paired data generator for Theorem 2 experiments.

Delegates to ``SyntheticTheoryFeatureBuilder`` with hardcoded defaults
for options that are fixed across all current experiments:

- ``coeff_dist="exponential"``
- ``cmin=0.0``
- ``min_active=1``
- ``strategy="gradient"``
- ``shared_coeff_mode="independent"``
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.datasets.synthetic_theory_feature import SyntheticTheoryFeatureBuilder


class SyntheticPairedBuilder:
    """Thin facade over ``SyntheticTheoryFeatureBuilder``."""

    def __init__(
        self,
        n_shared: int = 1024,
        n_image: int = 512,
        n_text: int = 512,
        representation_dim: int = 256,
        sparsity: float = 0.99,
        beta: float = 1.0,
        obs_noise_std: float = 0.05,
        max_interference: float = 0.1,
        alpha_target: float = 0.8,
        num_train: int = 50_000,
        num_eval: int = 10_000,
        seed: int = 0,
    ) -> None:
        shared_mode = "identical" if alpha_target >= 0.99 else "range"
        alpha_lo = max(0.0, alpha_target - 0.01) if shared_mode == "range" else 0.7
        alpha_hi = min(1.0, alpha_target + 0.01) if shared_mode == "range" else 0.9

        self._inner = SyntheticTheoryFeatureBuilder(
            n_image=n_image,
            n_shared=n_shared,
            n_text=n_text,
            representation_dim=representation_dim,
            sparsity=sparsity,
            beta=beta,
            obs_noise_std=obs_noise_std,
            max_interference=max_interference,
            shared_mode=shared_mode,
            alpha_target=alpha_target,
            alpha_lo=alpha_lo,
            alpha_hi=alpha_hi,
            num_train=num_train,
            num_eval=num_eval,
            num_test=num_eval,
            seed=seed,
            # Hardcoded
            coeff_dist="exponential",
            cmin=0.0,
            min_active=1,
            strategy="gradient",
            shared_coeff_mode="independent",
            verbose=False,
        )

    def build(self) -> dict[str, dict[str, np.ndarray]]:
        """Return ``{"train": {...}, "eval": {...}}`` numpy splits."""
        full = self._inner.build_numpy_dataset()
        return {"train": full["train"], "eval": full["eval"]}

    # --- Properties delegated to the inner builder ---

    @property
    def phi_S(self) -> np.ndarray:
        return self._inner.phi_S

    @property
    def psi_S(self) -> np.ndarray:
        return self._inner.psi_S

    @property
    def phi_I(self) -> Optional[np.ndarray]:
        return self._inner.phi_I

    @property
    def psi_T(self) -> Optional[np.ndarray]:
        return self._inner.psi_T

    @property
    def mean_shared_cosine_similarity(self) -> float:
        return self._inner.mean_shared_cosine_similarity

    @property
    def std_shared_cosine_similarity(self) -> float:
        return self._inner.std_shared_cosine_similarity

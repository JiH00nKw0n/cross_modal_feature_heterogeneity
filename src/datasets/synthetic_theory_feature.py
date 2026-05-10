"""
Synthetic theory feature dataset builder.

Generates paired image/text representations following the paper's notation:

    z = [z_I, z_S, z_T]^T in R^n_+

    Phi = [Phi_I | Phi_S | 0     ] in R^{m x n}
    Psi = [0     | Psi_S | Psi_T ] in R^{m x n}

    x = Phi z = Phi_I z_I + Phi_S z_S          (image embedding)
    y = Psi z = Psi_S z_S + Psi_T z_T          (text embedding)

When shared_mode="identical", Psi_S = Phi_S.
When shared_mode="contrastive", Psi_S is optimized from Phi_S via gradient
descent so that the symmetric InfoNCE loss on the full embeddings matches
a target value, while keeping column-wise max interference bounded.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import PrivateAttr, field_validator

from src.common.registry import registry
from src.datasets.base import BaseBuilder
from src.datasets.synthetic_feature import (
    SyntheticFeatureDatasetBuilder,
    _normalize_columns,
    _max_offdiag_dot,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SyntheticTheoryFeatureBuilder",
]


def _safe_float_array(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


def _symmetric_infonce(
    image_reps: torch.Tensor,
    text_reps: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Args:
        image_reps: (N, m) image embeddings.
        text_reps: (N, m) text embeddings.
        logit_scale: scalar log-scale parameter (used as exp(logit_scale)).

    Returns:
        Scalar loss value.
    """
    image_norm = F.normalize(image_reps, dim=-1)
    text_norm = F.normalize(text_reps, dim=-1)
    sim = text_norm @ image_norm.T * logit_scale.exp()
    targets = torch.arange(sim.shape[0], device=sim.device)
    caption_loss = F.cross_entropy(sim, targets)
    image_loss = F.cross_entropy(sim.T, targets)
    return (caption_loss + image_loss) / 2.0


@registry.register_builder("SyntheticTheoryFeatureBuilder")
class SyntheticTheoryFeatureBuilder(BaseBuilder):
    """
    Synthetic paired multimodal dataset following the paper's generative model.

    Dictionary matrices use paper notation:
        Phi_I (m x n_I), Phi_S (m x n_S), Psi_S (m x n_S), Psi_T (m x n_T).
    """

    # Dimensions
    n_image: int = 32           # n_I
    n_shared: int = 64          # n_S
    n_text: int = 32            # n_T
    representation_dim: int = 768   # m

    # Sparsity (uniform across all blocks)
    sparsity: float = 0.999
    min_active: int = 1

    # Coefficient distribution: "exponential" or "relu_gaussian"
    # exponential: c = cmin + Exp(beta)
    # relu_gaussian: c = ReLU(mu + sigma * N(0,1))
    #   (SynthSAEBench, Chanin & Garriga-Alonso 2026)
    coeff_dist: Literal["exponential", "relu_gaussian"] = "exponential"
    cmin: float = 0.0
    beta: float = 1.0
    coeff_mu: float = 4.5
    coeff_sigma: float = 0.5

    # Observation noise: Gaussian noise added to x, y embeddings after
    # sparse combination. 0 = clean embeddings (default).
    obs_noise_std: float = 0.0

    # Dictionary generation
    max_interference: float = 0.3
    strategy: Literal["gradient", "random", "sdp"] = "gradient"
    eps_margin: float = 1e-3
    lambda_neg: float = 1e-3
    dict_lr: float = 0.05
    dict_max_iters: int = 2000
    dict_tol: float = 1e-4

    # Shared coefficient mode: "identical" = same z_S for both modalities,
    # "independent" = same support (which features active) but magnitudes
    # sampled independently from cmin + Exp(beta) for each modality.
    shared_coeff_mode: Literal["identical", "independent"] = "identical"

    # Shared mode
    shared_mode: Literal["identical", "alpha", "range"] = "identical"
    alpha_target: float = 0.8          # target mean diag cosine between Phi_S and Psi_S
    alpha_lo: float = 0.7             # lower bound for per-atom cosine (range mode)
    alpha_hi: float = 0.9             # upper bound for per-atom cosine (range mode)
    calibration_lr: float = 0.005
    calibration_max_iters: int = 2000
    calibration_tol: float = 0.005

    # Samples per split
    num_train: int = 50_000
    num_eval: int = 10_000
    num_test: int = 10_000

    # General
    seed: int = 0
    verbose: bool = False

    # Private state
    _phi_I: Optional[np.ndarray] = PrivateAttr(default=None)
    _phi_S: Optional[np.ndarray] = PrivateAttr(default=None)
    _psi_S: Optional[np.ndarray] = PrivateAttr(default=None)
    _psi_T: Optional[np.ndarray] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    # Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("n_image", "n_text")
    @classmethod
    def _validate_nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v

    @field_validator("n_shared", "representation_dim", "num_train", "num_eval", "num_test")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"value must be positive, got {v}")
        return v

    @field_validator("sparsity")
    @classmethod
    def _validate_sparsity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {v}")
        return v

    @field_validator("min_active")
    @classmethod
    def _validate_min_active(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"min_active must be >= 0, got {v}")
        return v

    @field_validator("beta")
    @classmethod
    def _validate_beta(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"beta must be > 0, got {v}")
        return v

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Total latent dimensionality n = n_I + n_S + n_T."""
        return self.n_image + self.n_shared + self.n_text

    @property
    def phi_I(self) -> Optional[np.ndarray]:
        """Image-private dictionary Phi_I (m x n_I). None if n_I = 0."""
        self._ensure_dictionaries()
        return self._phi_I

    @property
    def phi_S(self) -> np.ndarray:
        """Image-side shared dictionary Phi_S (m x n_S)."""
        self._ensure_dictionaries()
        assert self._phi_S is not None
        return self._phi_S

    @property
    def psi_S(self) -> np.ndarray:
        """Text-side shared dictionary Psi_S (m x n_S)."""
        self._ensure_dictionaries()
        assert self._psi_S is not None
        return self._psi_S

    @property
    def psi_T(self) -> Optional[np.ndarray]:
        """Text-private dictionary Psi_T (m x n_T). None if n_T = 0."""
        self._ensure_dictionaries()
        return self._psi_T

    @property
    def phi_full(self) -> np.ndarray:
        """Full image mapping [Phi_I | Phi_S | 0] (m x n)."""
        self._ensure_dictionaries()
        parts = []
        if self._phi_I is not None:
            parts.append(self._phi_I)
        parts.append(self._phi_S)
        if self.n_text > 0:
            parts.append(np.zeros((self.representation_dim, self.n_text), dtype=np.float32))
        return np.concatenate(parts, axis=1)

    @property
    def psi_full(self) -> np.ndarray:
        """Full text mapping [0 | Psi_S | Psi_T] (m x n)."""
        self._ensure_dictionaries()
        parts = []
        if self.n_image > 0:
            parts.append(np.zeros((self.representation_dim, self.n_image), dtype=np.float32))
        parts.append(self._psi_S)
        if self._psi_T is not None:
            parts.append(self._psi_T)
        return np.concatenate(parts, axis=1)

    @property
    def _per_col_cosine(self) -> np.ndarray:
        """Per-atom cosine cos(Phi_S[:,i], Psi_S[:,i]) array of shape (n_S,)."""
        self._ensure_dictionaries()
        assert self._phi_S is not None and self._psi_S is not None
        phi_cols = self._phi_S / np.linalg.norm(self._phi_S, axis=0, keepdims=True).clip(min=1e-12)
        psi_cols = self._psi_S / np.linalg.norm(self._psi_S, axis=0, keepdims=True).clip(min=1e-12)
        return (phi_cols * psi_cols).sum(axis=0)

    @property
    def max_shared_cosine_similarity(self) -> float:
        """max_i cos(Phi_S[:,i], Psi_S[:,i])."""
        return float(self._per_col_cosine.max())

    @property
    def min_shared_cosine_similarity(self) -> float:
        """min_i cos(Phi_S[:,i], Psi_S[:,i])."""
        return float(self._per_col_cosine.min())

    @property
    def mean_shared_cosine_similarity(self) -> float:
        """mean_i cos(Phi_S[:,i], Psi_S[:,i])."""
        return float(self._per_col_cosine.mean())

    @property
    def std_shared_cosine_similarity(self) -> float:
        """std_i cos(Phi_S[:,i], Psi_S[:,i])."""
        return float(self._per_col_cosine.std())

    @property
    def actual_contrastive_loss(self, num_samples: int = 2048, logit_scale_val: float = 2.6592) -> float:
        """Compute contrastive loss on a fresh sample with given logit_scale."""
        self._ensure_dictionaries()
        seed_seq = np.random.SeedSequence(self.seed + 9999)
        split = self._build_numpy_split(num_samples, seed_seq)
        img = torch.from_numpy(split["image_representation"])
        txt = torch.from_numpy(split["text_representation"])
        logit_scale = torch.tensor(logit_scale_val)
        with torch.no_grad():
            loss = _symmetric_infonce(img, txt, logit_scale)
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Public methods                                                      #
    # ------------------------------------------------------------------ #

    def build_numpy_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Build numpy splits for fast experiment loops."""
        self._ensure_dictionaries()
        seeds = np.random.SeedSequence(self.seed).spawn(3)
        return {
            "train": self._build_numpy_split(self.num_train, seeds[0]),
            "eval": self._build_numpy_split(self.num_eval, seeds[1]),
            "test": self._build_numpy_split(self.num_test, seeds[2]),
        }

    def build_dataset(self):
        """Build HF DatasetDict (not primary interface; use build_numpy_dataset)."""
        from datasets import Dataset, DatasetDict
        splits = self.build_numpy_dataset()
        def _to_hf(split: dict[str, np.ndarray]) -> Dataset:
            data = {k: v.tolist() for k, v in split.items()}
            return Dataset.from_dict(data)
        return DatasetDict({k: _to_hf(v) for k, v in splits.items()})

    # ------------------------------------------------------------------ #
    # Dictionary generation                                               #
    # ------------------------------------------------------------------ #

    def _ensure_dictionaries(self) -> None:
        if self._phi_S is not None:
            return

        seeds = np.random.SeedSequence(self.seed).spawn(5)

        # Phi_I: image-private dictionary
        if self.n_image > 0:
            self._phi_I = _safe_float_array(
                self._generate_dictionary(self.n_image, int(seeds[0].generate_state(1)[0]))
            )
        else:
            self._phi_I = None

        # Phi_S: image-side shared dictionary
        self._phi_S = _safe_float_array(
            self._generate_dictionary(self.n_shared, int(seeds[1].generate_state(1)[0]))
        )

        # Psi_T: text-private dictionary
        if self.n_text > 0:
            self._psi_T = _safe_float_array(
                self._generate_dictionary(self.n_text, int(seeds[2].generate_state(1)[0]))
            )
        else:
            self._psi_T = None

        # Psi_S: text-side shared dictionary
        if self.shared_mode == "identical":
            self._psi_S = self._phi_S.copy()
        elif self.shared_mode == "alpha":
            self._psi_S = self._calibrate_psi_S_alpha()
        elif self.shared_mode == "range":
            self._psi_S = self._calibrate_psi_S_range()
        else:
            raise ValueError(f"Unknown shared_mode: {self.shared_mode}")

        if self.verbose:
            logger.info(
                "Generated dictionaries: n_I=%d n_S=%d n_T=%d m=%d shared_mode=%s",
                self.n_image, self.n_shared, self.n_text,
                self.representation_dim, self.shared_mode,
            )
            if self.shared_mode in ("alpha", "range"):
                logger.info(
                    "  shared_cos: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    self.min_shared_cosine_similarity,
                    self.max_shared_cosine_similarity,
                    self.mean_shared_cosine_similarity,
                    self.std_shared_cosine_similarity,
                )

    def _generate_dictionary(self, num_features: int, seed: int) -> np.ndarray:
        """Generate a dictionary matrix (m x num_features) with controlled interference."""
        if self.strategy == "random":
            rng = np.random.default_rng(seed)
            w = rng.standard_normal((self.representation_dim, num_features))
            return _normalize_columns(w)

        builder = SyntheticFeatureDatasetBuilder(
            feature_latent_dim=num_features,
            representation_dim=self.representation_dim,
            num_train=1,
            num_val=1,
            num_test=1,
            sparsity=0.99,
            min_active=0,
            max_interference=self.max_interference,
            strategy=self.strategy,
            eps_margin=self.eps_margin,
            lambda_neg=self.lambda_neg,
            lr=self.dict_lr,
            max_iters=self.dict_max_iters,
            tol=self.dict_tol,
            seed=seed,
            return_ground_truth=False,
            verbose=False,
        )
        w = builder._get_or_generate_wp().copy()
        return w

    # ------------------------------------------------------------------ #
    # Contrastive calibration of Psi_S                                    #
    # ------------------------------------------------------------------ #

    def _calibrate_psi_S_alpha(self) -> np.ndarray:
        """Optimize Psi_S so that mean_i cos(Phi_S[:,i], Psi_S[:,i]) ≈ alpha_target.

        Returns:
            psi_S as numpy array.
        """
        assert self._phi_S is not None
        psi_S = torch.tensor(self._phi_S.copy(), dtype=torch.float32, requires_grad=True)
        phi_S_t = torch.from_numpy(self._phi_S).float()

        optimizer = torch.optim.Adam([psi_S], lr=self.calibration_lr)
        target = self.alpha_target

        for step in range(self.calibration_max_iters):
            optimizer.zero_grad()

            # Mean diagonal cosine between Phi_S and Psi_S
            phi_normed = phi_S_t / phi_S_t.norm(dim=0, keepdim=True).clamp(min=1e-12)
            psi_normed = psi_S / psi_S.norm(dim=0, keepdim=True).clamp(min=1e-12)
            diag_cos = (phi_normed * psi_normed).sum(dim=0)  # (n_S,)
            mean_alpha = diag_cos.mean()

            total_loss = (mean_alpha - target) ** 2
            total_loss.backward()
            optimizer.step()

            # Normalize columns
            with torch.no_grad():
                psi_S.data = psi_S.data / psi_S.data.norm(dim=0, keepdim=True).clamp(min=1e-12)

            if self.verbose and step % 200 == 0:
                logger.info(
                    "  calibrate step=%d mean_alpha=%.4f target=%.4f",
                    step, mean_alpha.item(), target,
                )

            if abs(mean_alpha.item() - target) < self.calibration_tol:
                if self.verbose:
                    logger.info("  alpha calibration converged at step %d", step)
                break

        result = _safe_float_array(psi_S.detach().numpy())
        if self.verbose:
            logger.info(
                "  Psi_S alpha calibrated: mean_alpha=%.4f target=%.4f interf=%.4f",
                mean_alpha.item(), target, _max_offdiag_dot(result),
            )
        return result

    def _calibrate_psi_S_range(self) -> np.ndarray:
        """Optimize Psi_S so that ALL atoms satisfy alpha_lo <= cos(Phi_S[:,i], Psi_S[:,i]) <= alpha_hi.

        Strategy: target the midpoint (lo+hi)/2 per atom with hinge penalty
        for atoms outside the range. Initialize with noise proportional to
        distance from target to break symmetry.

        Returns:
            psi_S as numpy array.
        """
        assert self._phi_S is not None
        lo, hi = self.alpha_lo, self.alpha_hi

        # Special case: identical
        if lo >= 1.0 and hi >= 1.0:
            return self._phi_S.copy()

        mid = (lo + hi) / 2.0
        phi_S_t = torch.from_numpy(self._phi_S).float()

        # Initialize with noise proportional to desired perturbation
        # More noise for lower target ranges
        rng = np.random.default_rng(self.seed + 7777)
        noise_scale = max(0.1, 1.0 - mid)
        noise = rng.standard_normal(self._phi_S.shape).astype(np.float32) * noise_scale
        psi_init = self._phi_S.copy() + noise
        psi_init = _normalize_columns(psi_init)
        psi_S = torch.tensor(psi_init, dtype=torch.float32, requires_grad=True)

        optimizer = torch.optim.Adam([psi_S], lr=self.calibration_lr)

        for step in range(self.calibration_max_iters):
            optimizer.zero_grad()

            phi_normed = phi_S_t / phi_S_t.norm(dim=0, keepdim=True).clamp(min=1e-12)
            psi_normed = psi_S / psi_S.norm(dim=0, keepdim=True).clamp(min=1e-12)
            diag_cos = (phi_normed * psi_normed).sum(dim=0)  # (n_S,)

            # Per-atom loss: target midpoint + hinge for out-of-range
            midpoint_loss = (diag_cos - mid).pow(2).mean()
            below = torch.clamp(lo - diag_cos, min=0.0)
            above = torch.clamp(diag_cos - hi, min=0.0)
            hinge_loss = (below.pow(2) + above.pow(2)).mean()
            total_loss = midpoint_loss + 10.0 * hinge_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                psi_S.data = psi_S.data / psi_S.data.norm(dim=0, keepdim=True).clamp(min=1e-12)

            cos_min = diag_cos.min().item()
            cos_max = diag_cos.max().item()

            if self.verbose and step % 200 == 0:
                logger.info(
                    "  calibrate_range step=%d cos=[%.4f, %.4f] mean=%.4f target=[%.4f, %.4f] loss=%.6f",
                    step, cos_min, cos_max, diag_cos.mean().item(), lo, hi, total_loss.item(),
                )

            if cos_min >= lo - self.calibration_tol and cos_max <= hi + self.calibration_tol:
                if self.verbose:
                    logger.info("  range calibration converged at step %d", step)
                break

        result = _safe_float_array(psi_S.detach().numpy())
        final_cos = diag_cos.detach()
        if self.verbose:
            logger.info(
                "  Psi_S range calibrated: cos=[%.4f, %.4f] mean=%.4f std=%.4f target=[%.4f, %.4f] interf=%.4f",
                final_cos.min().item(), final_cos.max().item(),
                final_cos.mean().item(), final_cos.std().item(),
                lo, hi, _max_offdiag_dot(result),
            )
        return result

    # ------------------------------------------------------------------ #
    # Sparse coefficient sampling                                         #
    # ------------------------------------------------------------------ #

    def _sample_block(
        self,
        num_samples: int,
        num_features: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample sparse coefficients and masks for one latent block.

        Returns:
            (values, mask) each of shape (num_samples, num_features).
        """
        if num_features == 0:
            return (
                np.zeros((num_samples, 0), dtype=np.float32),
                np.zeros((num_samples, 0), dtype=np.float32),
            )

        base_prob = 1.0 - self.sparsity
        probs = np.full(num_features, base_prob, dtype=np.float64)
        mask = (rng.random((num_samples, num_features)) < probs).astype(np.float32)

        if self.min_active > 0 and self.min_active <= num_features:
            row_counts = mask.sum(axis=1)
            deficient = np.where(row_counts < self.min_active)[0]
            for idx in deficient:
                active = int(row_counts[idx])
                need = self.min_active - active
                inactive = np.where(mask[idx] == 0.0)[0]
                chosen = rng.choice(inactive, size=need, replace=False)
                mask[idx, chosen] = 1.0

        coeffs = self._sample_coefficients(rng, num_samples, num_features)
        values = _safe_float_array(mask * coeffs)
        return values, mask

    def _sample_coefficients(
        self, rng: np.random.Generator, num_samples: int, num_features: int,
    ) -> np.ndarray:
        if self.coeff_dist == "exponential":
            return self.cmin + rng.exponential(self.beta, size=(num_samples, num_features))
        elif self.coeff_dist == "relu_gaussian":
            raw = self.coeff_mu + self.coeff_sigma * rng.standard_normal(
                size=(num_samples, num_features)
            )
            return np.maximum(raw, 0.0)
        else:
            raise ValueError(f"Unknown coeff_dist '{self.coeff_dist}'; expected 'exponential' or 'relu_gaussian'.")

    # ------------------------------------------------------------------ #
    # Split building                                                      #
    # ------------------------------------------------------------------ #

    def _build_numpy_split(
        self,
        num_samples: int,
        seed_seq: np.random.SeedSequence,
    ) -> dict[str, np.ndarray]:
        return self._build_numpy_split_with_dict_override(
            num_samples=num_samples,
            seed_seq=seed_seq,
            psi_S_override=None,
        )

    def _build_numpy_split_with_dict_override(
        self,
        num_samples: int,
        seed_seq: np.random.SeedSequence,
        psi_S_override: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed_seq)

        image_values, image_mask = self._sample_block(num_samples, self.n_image, rng)
        text_values, text_mask = self._sample_block(num_samples, self.n_text, rng)

        if self.shared_coeff_mode == "independent":
            _, shared_mask = self._sample_block(num_samples, self.n_shared, rng)
            coeffs_img = self._sample_coefficients(rng, num_samples, self.n_shared)
            coeffs_txt = self._sample_coefficients(rng, num_samples, self.n_shared)
            shared_values_img = _safe_float_array(shared_mask * coeffs_img)
            shared_values_txt = _safe_float_array(shared_mask * coeffs_txt)
        else:
            shared_values, shared_mask = self._sample_block(num_samples, self.n_shared, rng)
            shared_values_img = shared_values
            shared_values_txt = shared_values

        psi_S_use = psi_S_override if psi_S_override is not None else self._psi_S
        assert psi_S_use is not None

        # x = Phi_I @ z_I + Phi_S @ z_S^img
        assert self._phi_S is not None
        image_rep = shared_values_img @ self._phi_S.T
        if self._phi_I is not None and self.n_image > 0:
            image_rep = image_rep + image_values @ self._phi_I.T

        # y = Psi_S @ z_S^txt + Psi_T @ z_T
        text_rep = shared_values_txt @ psi_S_use.T
        if self._psi_T is not None and self.n_text > 0:
            text_rep = text_rep + text_values @ self._psi_T.T

        if self.obs_noise_std > 0:
            image_rep = image_rep + rng.normal(0.0, self.obs_noise_std, size=image_rep.shape)
            text_rep = text_rep + rng.normal(0.0, self.obs_noise_std, size=text_rep.shape)

        split: dict[str, np.ndarray] = {
            "idx": np.arange(num_samples, dtype=np.int64),
            "image_representation": _safe_float_array(image_rep),
            "text_representation": _safe_float_array(text_rep),
            "shared_ground_truth": shared_mask,
            "image_private_ground_truth": image_mask,
            "text_private_ground_truth": text_mask,
            "_shared_values_img": shared_values_img,
            "_shared_values_txt": shared_values_txt,
            "_image_values": image_values,
            "_text_values": text_values,
        }

        return split

"""
Synthetic feature dataset builder.

Combines Linear Representation Bench (Appendix C) with Toy Models of
Superposition feature importance for SAE evaluation/training.

Pipeline: Wp generation -> importance calculation -> sparse x sampling -> xp = Wp @ x
"""

import logging
from typing import Literal, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from pydantic import PrivateAttr, field_validator

from src.common.registry import registry
from src.datasets.base import BaseBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "SyntheticFeatureDatasetBuilder",
    "_normalize_columns",
    "_max_offdiag_dot",
]


def _normalize_columns(W: np.ndarray) -> np.ndarray:
    """Project each column of W to unit norm."""
    norms = np.linalg.norm(W, axis=0, keepdims=True).clip(min=1e-12)
    return W / norms


def _max_offdiag_dot(W: np.ndarray) -> float:
    """Max positive off-diagonal dot product between columns of W (Appendix C.1)."""
    G = W.T @ W
    np.fill_diagonal(G, -np.inf)
    return float(np.max(G))


@registry.register_builder("SyntheticFeatureDatasetBuilder")
class SyntheticFeatureDatasetBuilder(BaseBuilder):
    """
    Synthetic feature dataset builder for SAE evaluation/training.

    Generates a projection matrix Wp with bounded column interference,
    samples sparse feature vectors x with optional importance weighting,
    and produces representations xp = Wp @ x as a DatasetDict.
    """

    # Dimensions
    feature_latent_dim: int = 1000          # n
    representation_dim: int = 768           # n_p

    # Samples per split
    num_train: int = 100_000
    num_val: int = 10_000
    num_test: int = 10_000

    # Sparsity (Appendix C.2)
    sparsity: float = 0.99                  # S = avg P(x_d = 0)
    cmin: float = 0.0
    beta: float = 1.0

    # Interference (Appendix C.1)
    max_interference: float = 0.1
    strategy: Literal["gradient", "sdp"] = "gradient"

    # Gradient params
    eps_margin: float = 1e-3
    lambda_neg: float = 1e-3
    lr: float = 0.05
    max_iters: int = 2000
    tol: float = 1e-4

    # SDP params
    sdp_restarts: int = 10
    sdp_refine_steps: int = 2000

    # Importance -- coefficient scaling
    scale_by_importance_coefficient: bool = False
    importance_coefficient_decay: float = 0.9   # I_i = decay^i, 1.0=uniform

    # Importance -- probability scaling
    scale_by_importance_probability: bool = False
    importance_probability_decay: float = 0.9   # I_i = decay^i, 1.0=uniform

    # General
    min_active: int = 0                     # minimum active features per sample (0=no constraint)
    seed: int = 0
    return_ground_truth: bool = True
    verbose: bool = False

    # Private
    _wp: Optional[np.ndarray] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    #  Validators                                                         #
    # ------------------------------------------------------------------ #

    @field_validator("strategy")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        if v not in {"gradient", "sdp"}:
            raise ValueError(f"strategy must be 'gradient' or 'sdp', got '{v}'")
        return v

    @field_validator("sparsity")
    @classmethod
    def _validate_sparsity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {v}")
        return v

    @field_validator("importance_coefficient_decay")
    @classmethod
    def _validate_coeff_decay(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"importance_coefficient_decay must be in (0, 1], got {v}")
        return v

    @field_validator("importance_probability_decay")
    @classmethod
    def _validate_prob_decay(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"importance_probability_decay must be in (0, 1], got {v}")
        return v

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def wp(self) -> Optional[np.ndarray]:
        """Cached projection matrix Wp (n_p x n)."""
        return self._wp

    @property
    def actual_max_interference(self) -> float:
        """Actual max off-diagonal dot product of Wp columns."""
        if self._wp is None:
            raise ValueError("Wp not generated yet -- call build_dataset() first")
        return _max_offdiag_dot(self._wp)

    @property
    def importance_coefficient(self) -> np.ndarray:
        """Coefficient importance vector (n,), mean-normalized to 1."""
        return self._compute_importance(self.importance_coefficient_decay)

    @property
    def importance_probability(self) -> np.ndarray:
        """Probability importance vector (n,), mean-normalized to 1."""
        return self._compute_importance(self.importance_probability_decay)

    # ------------------------------------------------------------------ #
    #  Public methods                                                     #
    # ------------------------------------------------------------------ #

    def build_dataset(self) -> DatasetDict:
        """Generate train/val/test splits as a DatasetDict."""
        self._get_or_generate_wp()
        seeds = np.random.SeedSequence(self.seed).spawn(3)
        return DatasetDict({
            "train": self._build_split(self.num_train, seeds[0]),
            "val": self._build_split(self.num_val, seeds[1]),
            "test": self._build_split(self.num_test, seeds[2]),
        })

    def save_wp(self, path: str) -> None:
        """Save Wp matrix to .npy file."""
        if self._wp is None:
            raise ValueError("Wp not generated yet -- call build_dataset() first")
        np.save(path, self._wp)
        if self.verbose:
            logger.info(f"Saved Wp to {path}")

    def load_wp(self, path: str) -> None:
        """Load Wp matrix from .npy file."""
        loaded = np.load(path)
        expected = (self.representation_dim, self.feature_latent_dim)
        if loaded.shape != expected:
            raise ValueError(f"Loaded Wp shape {loaded.shape} != expected {expected}")
        self._wp = loaded
        if self.verbose:
            logger.info(
                f"Loaded Wp from {path}, max interference={self.actual_max_interference:.6f}"
            )

    def visualize_features_3d(
        self,
        show: bool = True,
        save_path: Optional[str] = None,
        top_k_pairs: int = 3,
        figsize: tuple[int, int] = (10, 10),
    ) -> None:
        """Interactive 3D visualization of feature directions on the unit sphere.

        Only works when representation_dim == 3.

        Args:
            show: Open interactive matplotlib window (supports mouse rotation).
            save_path: If set, save figure to this path.
            top_k_pairs: Number of highest-dot pairs to highlight.
            figsize: Figure size.
        """
        if self._wp is None:
            raise ValueError("Wp not generated yet -- call build_dataset() first")
        if self.representation_dim != 3:
            raise ValueError(
                f"3D visualization requires representation_dim=3, got {self.representation_dim}"
            )

        import matplotlib
        if show:
            matplotlib.use("macosx")
        import matplotlib.pyplot as plt

        n = self.feature_latent_dim
        cols = self._wp.T  # (n, 3)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # wireframe unit sphere
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        ax.plot_wireframe(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            alpha=0.08, color="gray", linewidth=0.5,
        )

        # feature arrows
        cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
        colors = cmap(np.linspace(0, 1, n))
        for i in range(n):
            x, y, z = cols[i]
            ax.quiver(
                0, 0, 0, x, y, z,
                color=colors[i], arrow_length_ratio=0.08, linewidth=2.5,
            )
            ax.text(
                x * 1.15, y * 1.15, z * 1.15,
                f"f{i}", fontsize=10, fontweight="bold", color=colors[i], ha="center",
            )

        # importance ring (radius proportional to importance)
        if self.scale_by_importance_coefficient or self.scale_by_importance_probability:
            imp = (
                self.importance_coefficient
                if self.scale_by_importance_coefficient
                else self.importance_probability
            )
            imp_norm = imp / imp.max()
            for i in range(n):
                ax.scatter(
                    *cols[i], s=imp_norm[i] * 200 + 20,
                    color=colors[i], alpha=0.3, edgecolors="none",
                )

        # highlight top-k highest dot pairs
        G = self._wp.T @ self._wp
        np.fill_diagonal(G, -np.inf)
        pairs = sorted(
            ((G[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
            reverse=True,
        )
        for dot_val, i, j in pairs[:top_k_pairs]:
            ax.plot(
                [cols[i][0], cols[j][0]],
                [cols[i][1], cols[j][1]],
                [cols[i][2], cols[j][2]],
                "--", color="red", alpha=0.5, linewidth=1.2,
            )
            mid = (cols[i] + cols[j]) / 2
            mid = mid / np.linalg.norm(mid).clip(min=1e-12) * 0.55
            ax.text(*mid, f"{dot_val:.3f}", fontsize=8, color="red", alpha=0.8)

        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_zlim([-1.3, 1.3])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(
            f"{n} features in 3D  (M={self.actual_max_interference:.4f})",
            fontsize=14,
        )
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            if self.verbose:
                logger.info(f"Saved 3D figure to {save_path}")
        if show:
            plt.show()

    # ------------------------------------------------------------------ #
    #  Wp generation                                                      #
    # ------------------------------------------------------------------ #

    def _get_or_generate_wp(self) -> np.ndarray:
        if self._wp is not None:
            return self._wp
        if self.verbose:
            logger.info(
                f"Generating Wp ({self.representation_dim}x{self.feature_latent_dim}) "
                f"via {self.strategy}, target delta={self.max_interference}"
            )
        self._wp = (
            self._generate_wp_gradient()
            if self.strategy == "gradient"
            else self._generate_wp_sdp()
        )
        if self.verbose:
            logger.info(
                f"Wp done -- actual max interference={_max_offdiag_dot(self._wp):.6f}"
            )
        return self._wp

    def _generate_wp_gradient(self) -> np.ndarray:
        """
        Appendix C.1: soft-thresholded interference loss (eq 27) minimization
        with column unit-norm projection after each gradient step (eq 28).
        """
        rng = np.random.default_rng(self.seed)
        n, n_p, delta = self.feature_latent_dim, self.representation_dim, self.max_interference

        W = _normalize_columns(rng.standard_normal((n_p, n)))
        thresh = delta - self.eps_margin

        for it in range(self.max_iters):
            G = W.T @ W                        # (n, n)
            np.fill_diagonal(G, 0.0)

            # masks: positive above threshold + all positive pairs
            mask_above = G > thresh
            mask_pos = G > 0.0

            # dL/dG (off-diagonal, symmetric)
            D = np.zeros_like(G)
            D[mask_above] += 2.0 * (G[mask_above] - thresh)        # d/dG (G-thresh)^2
            D[mask_pos] += 2.0 * self.lambda_neg * G[mask_pos]     # d/dG lambda*G^2
            D = 0.5 * (D + D.T)
            np.fill_diagonal(D, 0.0)

            # dL/dW = 2 W @ D  (for G = W^T W with symmetric D)
            grad = 2.0 * (W @ D)

            W_new = _normalize_columns(W - self.lr * grad)
            max_dot = _max_offdiag_dot(W_new)

            if self.verbose and it % 200 == 0:
                logger.info(f"  iter {it}: max_offdiag_dot={max_dot:.6f}")

            if max_dot <= delta + self.tol:
                W = W_new
                if self.verbose:
                    logger.info(f"  converged iter {it}, max_dot={max_dot:.6f}")
                break

            W = W_new

        return W

    def _generate_wp_sdp(self) -> np.ndarray:
        """Welch -> SDP alternating projection -> low-rank factorization -> L-BFGS-B."""
        from scipy.optimize import minimize as scipy_minimize

        n, n_p, delta = self.feature_latent_dim, self.representation_dim, self.max_interference
        best_W: Optional[np.ndarray] = None
        best_coh = float("inf")

        for r in range(self.sdp_restarts):
            rng = np.random.default_rng(self.seed + r)
            W = _normalize_columns(rng.standard_normal((n_p, n)))
            G = W.T @ W

            # alternating projections: PSD cone + diag=1 + off-diag clipping
            for _ in range(50):
                np.fill_diagonal(G, 0.0)
                G = np.clip(G, -delta, delta)
                np.fill_diagonal(G, 1.0)

                eigvals, eigvecs = np.linalg.eigh(G)
                eigvals = np.maximum(eigvals, 0.0)
                G = (eigvecs * eigvals) @ eigvecs.T

                diag_sqrt = np.sqrt(np.diag(G)).clip(min=1e-12)
                G /= np.outer(diag_sqrt, diag_sqrt)

            # low-rank factorization: top n_p eigenvalues
            eigvals, eigvecs = np.linalg.eigh(G)
            top = np.argsort(eigvals)[-n_p:]
            W = (eigvecs[:, top] * np.sqrt(np.maximum(eigvals[top], 0.0))).T  # (n_p, n)
            W = _normalize_columns(W)

            # L-BFGS-B refinement
            thresh = delta - self.eps_margin
            lam = self.lambda_neg

            def _objective(w_flat: np.ndarray) -> tuple[float, np.ndarray]:
                Wl = _normalize_columns(w_flat.reshape(n_p, n))
                Gl = Wl.T @ Wl
                np.fill_diagonal(Gl, 0.0)
                above = Gl > thresh
                pos = Gl > 0.0
                D = np.zeros_like(Gl)
                D[above] += 2.0 * (Gl[above] - thresh)
                D[pos] += 2.0 * lam * Gl[pos]
                D = 0.5 * (D + D.T)
                np.fill_diagonal(D, 0.0)
                pen = np.maximum(Gl - thresh, 0.0)
                loss = float(np.sum(pen[above] ** 2)) + lam * float(np.sum(Gl[pos] ** 2))
                return loss, (2.0 * Wl @ D).ravel()

            res = scipy_minimize(
                _objective, W.ravel(), method="L-BFGS-B", jac=True,
                options={"maxiter": self.sdp_refine_steps, "ftol": 1e-12},
            )
            W = _normalize_columns(res.x.reshape(n_p, n))
            coh = _max_offdiag_dot(W)

            if self.verbose:
                logger.info(f"  SDP restart {r}: max_coh={coh:.6f}")
            if coh < best_coh:
                best_coh, best_W = coh, W.copy()

        return best_W  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Importance & sampling                                              #
    # ------------------------------------------------------------------ #

    def _compute_importance(self, decay: float) -> np.ndarray:
        """Compute decay-based importance vector, normalized to mean=1."""
        raw = decay ** np.arange(self.feature_latent_dim)
        return raw / raw.mean()

    def _sample_sparse_coefficients(
        self, num_samples: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample sparse feature vectors (Appendix C.2) with optional importance scaling.

        Returns (x, mask) where x is (num_samples, n) and mask is (num_samples, n).
        """
        n = self.feature_latent_dim
        base_prob = 1.0 - self.sparsity

        # per-feature activation probability
        if self.scale_by_importance_probability:
            probs = np.clip(self.importance_probability * base_prob, 0.0, 1.0)
        else:
            probs = np.full(n, base_prob)

        # Bernoulli mask
        mask = (rng.random((num_samples, n)) < probs).astype(np.float64)

        # enforce min_active: for deficient rows, sample extra features
        if self.min_active > 0:
            row_counts = mask.sum(axis=1)
            deficient = np.where(row_counts < self.min_active)[0]
            if len(deficient) > 0:
                # sampling weights: use activation probs (importance-aware)
                weights = probs / probs.sum()
                for idx in deficient:
                    active = np.where(mask[idx] == 1.0)[0]
                    need = self.min_active - len(active)
                    inactive = np.where(mask[idx] == 0.0)[0]
                    w = weights[inactive]
                    w = w / w.sum()
                    chosen = rng.choice(inactive, size=need, replace=False, p=w)
                    mask[idx, chosen] = 1.0

        # coefficients: cmin + Exp(beta)
        coeffs = self.cmin + rng.exponential(self.beta, size=(num_samples, n))

        if self.scale_by_importance_coefficient:
            coeffs *= self.importance_coefficient

        x = mask * coeffs
        return x, mask

    # ------------------------------------------------------------------ #
    #  Split building                                                     #
    # ------------------------------------------------------------------ #

    def _build_split(self, num_samples: int, seed_seq: np.random.SeedSequence) -> Dataset:
        """Build a single Dataset split."""
        rng = np.random.default_rng(seed_seq)
        x, ground_truth = self._sample_sparse_coefficients(num_samples, rng)
        representations = (self._wp @ x.T).T           # (samples, n_p)

        data: dict[str, list] = {
            "idx": list(range(num_samples)),
            "representation": representations.tolist(),
        }
        if self.return_ground_truth:
            data["ground_truth_feature"] = ground_truth.tolist()
            data["linear_coefficient"] = x.tolist()

        return Dataset.from_dict(data)

"""Row normalization utility shared across the codebase."""

from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row of *x* to unit L2 norm."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)

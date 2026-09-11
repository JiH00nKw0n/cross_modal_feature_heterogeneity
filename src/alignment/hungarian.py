"""Post-hoc Hungarian alignment for `Post-hoc Alignment (Ours)` (paper section 4.3).

This module is a thin compatibility wrapper. The rules and the computation live
in `src.alignment.panel`; all this does is give older call sites the
`build_perm` / `save_perm` / `load_perm` names they already use, while the
artifact that gets written is the same panel every other stage reads.

One permutation is built from the model's own training distribution and reused
for every downstream evaluation. Recomputing a permutation per evaluation
dataset would let each evaluation pick its own favourable matching, which
inflates the numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.alignment.panel import build_panel, load_panel, save_panel
from src.models import TwoSidedTopKSAE

logger = logging.getLogger(__name__)


def build_perm(
    *,
    model: TwoSidedTopKSAE,
    cache_dir: str | Path,
    split: str = "train",
    max_samples: int = 0,
    batch_size: int = 8192,
    device: str = "cuda",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the image-to-text panel and return its payload.

    `max_samples` defaults to 0, meaning every pair of the split: the alive
    rule and the correlation are computed on the FULL training split, not on a
    50,000-row sample.
    """
    return build_panel(
        model=model, cache_dir=cache_dir, split=split, batch_size=batch_size,
        device=device, max_samples=max_samples, pairing="img_txt", **kwargs,
    )


def save_perm(out_path: str | Path, payload: dict[str, Any]) -> None:
    """Write the payload to `out_path`, keeping the historical perm.npz name."""
    save_panel(out_path, payload)


def load_perm(path: str | Path) -> dict[str, Any]:
    """Read a panel or a legacy perm file; `perm` is always present."""
    payload = load_panel(path)
    if "perm" not in payload:
        raise KeyError(f"{path} has no 'perm' entry; entries present: {sorted(payload)}")
    return payload


def load_perm_array(path: str | Path) -> np.ndarray:
    """Just the permutation, for the evaluators that need nothing else."""
    return np.asarray(load_perm(path)["perm"], dtype=np.int64)


__all__ = ["build_perm", "save_perm", "load_perm", "load_perm_array"]

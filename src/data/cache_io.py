"""Single, simple cache format for paired (image, text) embeddings.

Layout under cache_dir/:
    image_embeddings.pt      torch.save dict[key]->Tensor[dim]  (legacy support)
    text_embeddings.pt       torch.save dict[key]->Tensor[dim]
    image_embeddings.npy     stacked NDArray[N, dim]            (preferred)
    text_embeddings.npy      stacked NDArray[N, dim]
    keys.json                list[str] of length N
    splits.json              {"train": [keys], "val": [keys], ...}
    meta.json                {"model_key": ..., "dim": ..., "dataset": ...}

Loaders mmap-friendly: prefer .npy when present (`load_stacked`), fall back to
the dict-of-tensors files for backwards compat with old caches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_stacked(
    cache_dir: str | Path,
    *,
    image_emb: np.ndarray,
    text_emb: np.ndarray,
    keys: list[str],
    splits: dict[str, list[str]],
    meta: dict[str, Any],
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "image_embeddings.npy", image_emb)
    np.save(cache_dir / "text_embeddings.npy", text_emb)
    with open(cache_dir / "keys.json", "w") as f:
        json.dump(keys, f)
    with open(cache_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_stacked(cache_dir: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    img_npy = cache_dir / "image_embeddings.npy"
    if img_npy.exists():
        with open(cache_dir / "keys.json") as f:
            keys = json.load(f)
        splits_path = cache_dir / "splits.json"
        splits = json.load(open(splits_path)) if splits_path.exists() else {"train": keys}
        meta = json.load(open(cache_dir / "meta.json")) if (cache_dir / "meta.json").exists() else {}
        return {
            "image": np.load(img_npy, mmap_mode="r" if mmap else None),
            "text": np.load(cache_dir / "text_embeddings.npy", mmap_mode="r" if mmap else None),
            "keys": keys,
            "splits": splits,
            "meta": meta,
        }
    # Legacy: dict-of-tensors .pt
    img_pt = torch.load(cache_dir / "image_embeddings.pt", map_location="cpu")
    txt_pt = torch.load(cache_dir / "text_embeddings.pt", map_location="cpu")
    keys = list(img_pt.keys())
    image = np.stack([img_pt[k].numpy() for k in keys])
    text = np.stack([txt_pt[k].numpy() for k in keys])
    splits_path = cache_dir / "splits.json"
    splits = json.load(open(splits_path)) if splits_path.exists() else {"train": keys}
    return {"image": image, "text": text, "keys": keys, "splits": splits, "meta": {}}


def append_state(state: dict, key: str, vec: torch.Tensor) -> None:
    """Backwards-compat: append to a dict-of-tensors cache (used by legacy extractors)."""
    state[key] = vec.detach().cpu()


def save_dict_state(state: dict, path: str | Path) -> None:
    torch.save(state, path)

"""Unified embedding extractor: COCO Karpathy / CC3M streaming / ImageNet.

All three datasets feed paired (image, text) pairs to an `Encoder` and write
to the canonical cache layout (see `cache_io.py`). Idempotent: skips keys
already present in the existing dict-of-tensors files (legacy compat).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from src.encoders import Encoder, load_encoder
from src.utils.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset adapters → (key, PIL.Image, caption)
# --------------------------------------------------------------------------- #
def _coco_karpathy(split: str = "train") -> Iterable[tuple[str, Image.Image, str]]:
    """COCO Karpathy via HuggingFace `yerevann/coco-karpathy`."""
    from datasets import load_dataset
    ds = load_dataset("yerevann/coco-karpathy", split=split, streaming=True)
    for ex in ds:
        img = ex["image"].convert("RGB") if hasattr(ex["image"], "convert") else None
        if img is None:
            continue
        for cap_idx, cap in enumerate(ex["sentences"]):
            yield f"{ex['cocoid']}_{cap_idx}", img, cap


def _cc3m_streaming(split: str = "train") -> Iterable[tuple[str, Image.Image, str]]:
    """CC3M via `pixparse/cc3m-wds` webdataset."""
    from datasets import load_dataset
    ds = load_dataset("pixparse/cc3m-wds", split=split, streaming=True)
    for row in ds:
        try:
            img_bytes = row["jpg"]["bytes"] if isinstance(row.get("jpg"), dict) else row["jpg"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            continue
        txt = row.get("txt") or ""
        if not txt:
            continue
        yield str(row["__key__"]), img, txt


def _imagenet(split: str = "train") -> Iterable[tuple[str, Image.Image, str]]:
    """ImageNet via HF `imagenet-1k`. Caption = class name."""
    from datasets import load_dataset
    ds = load_dataset("imagenet-1k", split=split, streaming=True)
    classnames = ds.features["label"].names if hasattr(ds, "features") else None
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB")
        cls = classnames[ex["label"]] if classnames else str(ex["label"])
        yield f"{i}", img, cls.replace("_", " ")


_DATASETS = {
    "coco": _coco_karpathy,
    "cc3m": _cc3m_streaming,
    "imagenet": _imagenet,
}


# --------------------------------------------------------------------------- #
# IterableDataset wrapper (so we can use DataLoader for batched PIL decode)
# --------------------------------------------------------------------------- #
class _PairStream(IterableDataset):
    def __init__(self, dataset: str, split: str, max_samples: int | None = None):
        super().__init__()
        self.fn = _DATASETS[dataset]
        self.split = split
        self.max_samples = max_samples

    def __iter__(self):
        n = 0
        for triple in self.fn(self.split):
            yield triple
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


def _collate(batch):
    return {
        "keys": [b[0] for b in batch],
        "imgs": [b[1] for b in batch],
        "txts": [b[2] for b in batch],
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_cache(
    *,
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    batch_size: int = 64,
    max_samples: int | None = None,
    num_workers: int = 2,
    device: str = "cuda",
) -> None:
    """Extract paired embeddings for `cache_cfg.dataset`/`cache_cfg.split` and save.

    Idempotent at the cache-dir level: if all 3 stacked files exist, exits.
    """
    cache_dir = Path(cache_cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_npy = cache_dir / "image_embeddings.npy"
    if img_npy.exists() and (cache_dir / "text_embeddings.npy").exists() and (cache_dir / "keys.json").exists():
        logger.info("[extract] cache exists at %s — skipping", cache_dir)
        return

    encoder: Encoder = load_encoder(model_cfg, device=device)
    stream = _PairStream(cache_cfg.dataset, cache_cfg.split, max_samples=max_samples)
    loader = DataLoader(stream, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, persistent_workers=False)

    img_chunks: list[np.ndarray] = []
    txt_chunks: list[np.ndarray] = []
    keys: list[str] = []
    pbar = tqdm(loader, desc=f"extract {cache_cfg.dataset}/{cache_cfg.split}")
    for batch in pbar:
        if not batch["keys"]:
            continue
        img_emb = encoder.encode_image(batch["imgs"]).numpy()
        txt_emb = encoder.encode_text(batch["txts"]).numpy()
        img_chunks.append(img_emb)
        txt_chunks.append(txt_emb)
        keys.extend(batch["keys"])

    image = np.concatenate(img_chunks, axis=0).astype(np.float32)
    text = np.concatenate(txt_chunks, axis=0).astype(np.float32)
    splits = {cache_cfg.split: keys}

    from src.data.cache_io import save_stacked
    save_stacked(
        cache_dir,
        image_emb=image, text_emb=text, keys=keys,
        splits=splits,
        meta={"model_key": model_cfg.key, "dim": int(image.shape[1]),
              "dataset": cache_cfg.dataset, "split": cache_cfg.split},
    )
    logger.info("[extract] wrote %d pairs to %s (dim=%d)", len(keys), cache_dir, image.shape[1])

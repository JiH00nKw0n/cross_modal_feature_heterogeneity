"""Fixtures that build tiny caches and checkpoints, so tests need no downloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.cache_io import save_imagenet_cache, save_stacked
from src.models import TwoSidedTopKSAE, TwoSidedTopKSAEConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def make_coco_cache(cache_dir: Path, *, n_images: int = 8, caps_per_image: int = 5,
                    dim: int = 16, seed: int = 0) -> dict:
    """A COCO-shaped paired cache: keys "{image_id}_{cap_idx}", train/val/test.

    The image row is duplicated across a photo's captions, which is the
    one-row-per-pair rule the real extractor follows.
    """
    gen = np.random.default_rng(seed)
    keys: list[str] = []
    images: list[np.ndarray] = []
    texts: list[np.ndarray] = []
    captions: dict[str, str] = {}
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_of = lambda i: "train" if i < n_images - 4 else ("val" if i < n_images - 2 else "test")  # noqa: E731

    for img_id in range(n_images):
        img_vec = gen.normal(size=dim).astype(np.float32)
        for cap in range(caps_per_image):
            key = f"{img_id}_{cap}"
            keys.append(key)
            images.append(img_vec)
            texts.append(gen.normal(size=dim).astype(np.float32))
            captions[key] = f"a caption {cap} of image {img_id}"
            splits[split_of(img_id)].append(key)

    save_stacked(
        cache_dir,
        image_emb=np.stack(images), text_emb=np.stack(texts), keys=keys,
        splits=splits,
        meta={"model_key": "fake", "dim": dim, "dataset": "coco", "n_pairs": len(keys)},
        captions=captions,
    )
    return {"keys": keys, "splits": splits, "dim": dim}


def make_imagenet_cache(cache_dir: Path, *, n_images: int = 12, n_classes: int = 3,
                        n_templates: int = 4, dim: int = 16, seed: int = 1) -> dict:
    """An ImageNet-shaped cache with a class-major text grid."""
    gen = np.random.default_rng(seed)
    image = gen.normal(size=(n_images, dim)).astype(np.float32)
    labels = (np.arange(n_images) % n_classes).astype(np.int64)
    text = gen.normal(size=(n_classes * n_templates, dim)).astype(np.float32)
    text_keys = [f"{c}_{t}" for c in range(n_classes) for t in range(n_templates)]
    save_imagenet_cache(
        cache_dir, image_emb=image, labels=labels, text_emb=text, text_keys=text_keys,
        meta={"model_key": "fake", "dim": dim, "dataset": "imagenet",
              "n_classes": n_classes, "n_templates": n_templates, "n_images": n_images},
    )
    return {"n_images": n_images, "n_classes": n_classes,
            "n_templates": n_templates, "dim": dim}


def make_two_sided_sae(dim: int = 16, latent_size: int = 8, k: int = 2,
                       seed: int = 0) -> TwoSidedTopKSAE:
    """A tiny TwoSidedTopKSAE. `latent_size` is the TOTAL budget, split in half."""
    torch.manual_seed(seed)
    cfg = TwoSidedTopKSAEConfig(hidden_size=dim, latent_size=latent_size, k=k,
                                normalize_decoder=True)
    return TwoSidedTopKSAE(cfg)


@pytest.fixture
def coco_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir)
    return cache_dir


@pytest.fixture
def imagenet_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "imagenet"
    make_imagenet_cache(cache_dir)
    return cache_dir


@pytest.fixture
def coco_cache_two_captions(tmp_path: Path) -> Path:
    """A COCO-shaped cache with exactly two captions per photograph.

    The different-caption pairing compares a model's text side against another
    model's text side while the two read different captions of one photo, so a
    cache with a single caption per image would make that pairing degenerate
    into the same-caption case. Two captions is the smallest number that keeps
    it meaningful.
    """
    cache_dir = tmp_path / "coco_two_captions"
    make_coco_cache(cache_dir, n_images=8, caps_per_image=2, seed=3)
    return cache_dir


#: A handful of COCO's 80 categories with their real, non-contiguous category
#: ids, so that a fixture annotation file maps onto the same index positions as
#: the genuine one. The ids come from COCO's own instances files.
COCO_TEST_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (17, "cat"), (18, "dog"),
    (62, "chair"), (63, "couch"), (67, "dining table"), (72, "tv"), (73, "laptop"),
]


def make_coco_instances(
    path: Path, *, n_images: int = 8, image_size: int = 100, seed: int = 0,
) -> dict:
    """Write an instances_val2014.json-shaped file over a few categories.

    Each image carries one or two object annotations drawn from
    `COCO_TEST_CATEGORIES`. Areas alternate between large and small relative to
    the frame, so a test can check that a salience rule ("the object must cover
    at least this fraction of the frame") actually drops the small ones: a large
    annotation covers 25 percent of the frame and a small one 1 percent.
    """
    gen = np.random.default_rng(seed)
    frame = image_size * image_size
    images = [{"id": i, "width": image_size, "height": image_size,
               "file_name": f"COCO_val2014_{i:012d}.jpg"} for i in range(n_images)]
    annotations = []
    ann_id = 0
    for i in range(n_images):
        picks = gen.choice(len(COCO_TEST_CATEGORIES), size=2, replace=False)
        for slot, cat_idx in enumerate(picks):
            cat_id, _name = COCO_TEST_CATEGORIES[int(cat_idx)]
            # Slot 0 is the large object, slot 1 the small one.
            area = 0.25 * frame if slot == 0 else 0.01 * frame
            side = float(np.sqrt(area))
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": int(cat_id),
                "area": float(area), "bbox": [0.0, 0.0, side, side], "iscrowd": 0,
            })
            ann_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name, "supercategory": "test"}
                       for cid, name in COCO_TEST_CATEGORIES],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


@pytest.fixture
def coco_instances_json(tmp_path: Path) -> Path:
    """Path to a fixture instances_val2014.json, no download needed."""
    path = tmp_path / "coco_annotations" / "instances_val2014.json"
    make_coco_instances(path)
    return path


def write_eval_json(root: Path, seed: int, method: str, name: str, payload: dict) -> None:
    """Drop one evaluation JSON into the tree table1 reads."""
    out = root / f"seed{seed}" / "eval" / method / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

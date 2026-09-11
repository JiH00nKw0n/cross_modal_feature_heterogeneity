"""The two cache layouts survive a write and a read unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.cache_io import (
    imagenet_cache_complete,
    load_captions,
    load_imagenet_cache,
    load_stacked,
    paired_cache_complete,
    save_stacked,
    split_rows,
)
from tests.conftest import make_coco_cache, make_imagenet_cache


def test_paired_round_trip(tmp_path: Path) -> None:
    image = np.arange(12, dtype=np.float32).reshape(4, 3)
    text = (image * -1).astype(np.float32)
    keys = ["7_0", "7_1", "9_0", "9_1"]
    splits = {"train": keys[:2], "test": keys[2:]}
    save_stacked(tmp_path, image_emb=image, text_emb=text, keys=keys, splits=splits,
                 meta={"model_key": "fake", "dim": 3}, captions={k: k for k in keys})

    assert paired_cache_complete(tmp_path)
    cache = load_stacked(tmp_path)
    np.testing.assert_array_equal(np.asarray(cache["image"]), image)
    np.testing.assert_array_equal(np.asarray(cache["text"]), text)
    assert cache["keys"] == keys
    assert cache["splits"] == splits
    assert cache["meta"]["dim"] == 3
    assert load_captions(tmp_path) == {k: k for k in keys}


def test_split_rows_are_the_split_order(tmp_path: Path) -> None:
    image = np.zeros((4, 2), dtype=np.float32)
    keys = ["a", "b", "c", "d"]
    save_stacked(tmp_path, image_emb=image, text_emb=image, keys=keys,
                 splits={"train": ["c", "a"], "test": ["d"]}, meta={})
    cache = load_stacked(tmp_path)
    np.testing.assert_array_equal(split_rows(cache, "train"), np.array([2, 0]))
    np.testing.assert_array_equal(split_rows(cache, "test"), np.array([3]))


def test_stored_embeddings_are_raw(tmp_path: Path) -> None:
    """Rule: the cache holds encoder output, so rows are not unit norm."""
    make_coco_cache(tmp_path, n_images=6, dim=8)
    image = np.asarray(load_stacked(tmp_path)["image"])
    norms = np.linalg.norm(image, axis=1)
    assert not np.allclose(norms, 1.0)


def test_coco_cache_duplicates_the_image_row_per_caption(tmp_path: Path) -> None:
    """Rule: one row per pair, the image row repeated across its captions."""
    make_coco_cache(tmp_path, n_images=3, caps_per_image=5, dim=8)
    cache = load_stacked(tmp_path)
    image = np.asarray(cache["image"])
    assert image.shape[0] == 15
    for img_id in range(3):
        block = image[img_id * 5:(img_id + 1) * 5]
        assert np.allclose(block, block[0])


def test_imagenet_cache_round_trip(tmp_path: Path) -> None:
    spec = make_imagenet_cache(tmp_path, n_images=10, n_classes=5, n_templates=3, dim=4)
    assert imagenet_cache_complete(tmp_path)
    cache = load_imagenet_cache(tmp_path)
    assert np.asarray(cache["image"]).shape == (10, 4)
    assert cache["labels"].shape == (10,)
    assert np.asarray(cache["text"]).shape == (15, 4)
    assert cache["text_keys"][0] == "0_0"
    assert cache["text_keys"][4] == "1_1"  # class-major: row c * n_templates + t
    assert cache["meta"]["n_classes"] == spec["n_classes"]


def test_incomplete_cache_is_reported_not_guessed(tmp_path: Path) -> None:
    (tmp_path / "keys.json").write_text(json.dumps(["a"]))
    assert not paired_cache_complete(tmp_path)
    try:
        load_stacked(tmp_path)
    except FileNotFoundError as exc:
        assert "image_embeddings.npy" in str(exc)
    else:
        raise AssertionError("an incomplete cache must not load")

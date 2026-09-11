"""Resuming an interrupted extraction must not lose or duplicate anything.

Both extractors are driven here against stub sources, so nothing is downloaded
and no encoder is loaded. The two regressions covered:

  COCO      the captions sidecar. A resumed pass skips the image groups it
            already encoded, so a captions dict built only from this attempt
            would be missing every pair written before the interruption.
  ImageNet  the resume offset. `groups_done` counts encoded images, so the
            iterator it is applied to has to be the one with undecodable rows
            already filtered out; slicing the raw stream re-encodes rows.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import extract
from src.utils.config import ModelConfig

DIM = 4


class _StubEncoder:
    """Maps an image to its red channel and a caption to its length."""

    dim = DIM
    device = torch.device("cpu")

    def encode_image(self, images):
        vals = [float(im.getpixel((0, 0))[0]) for im in images]
        return torch.tensor([[v] * DIM for v in vals], dtype=torch.float32)

    def encode_text(self, texts):
        return torch.tensor([[float(len(t))] * DIM for t in texts], dtype=torch.float32)


def _image(value: int) -> Image.Image:
    return Image.new("RGB", (2, 2), color=(value, 0, 0))


@pytest.fixture
def stub_env(monkeypatch):
    """Stub encoder plus a chunk writer that flushes on every row."""
    monkeypatch.setattr(extract, "load_encoder", lambda *a, **k: _StubEncoder())
    monkeypatch.setattr(
        extract, "_ChunkWriter",
        functools.partial(extract._ChunkWriter, chunk_size=1),
    )
    return ModelConfig(key="stub", backend="transformers", hidden_size=DIM)


def test_coco_resume_writes_every_caption(tmp_path: Path, stub_env, monkeypatch) -> None:
    """A COCO pass interrupted after two images still writes all captions."""
    state = {"fail_after": 2}

    def fake_groups(split: str):
        n = 4 if split == "train" else 1
        base = {"train": 0, "validation": 10, "test": 20}[split]
        for i in range(n):
            image_id = base + i
            if split == "train" and i >= state["fail_after"]:
                raise RuntimeError("stream interrupted")
            yield _image(image_id), [(f"{image_id}_0", f"caption of {image_id}")]

    monkeypatch.setattr(extract, "_iter_coco_groups", fake_groups)
    cache_dir = tmp_path / "coco"

    with pytest.raises(RuntimeError, match="stream interrupted"):
        extract.extract_coco(model_cfg=stub_env, cache_dir=cache_dir,
                             batch_size=1, device="cpu")

    state["fail_after"] = 99
    extract.extract_coco(model_cfg=stub_env, cache_dir=cache_dir,
                         batch_size=1, device="cpu")

    keys = json.loads((cache_dir / "keys.json").read_text())
    captions = json.loads((cache_dir / "captions.json").read_text())
    assert keys == ["0_0", "1_0", "2_0", "3_0", "10_0", "20_0"]
    assert sorted(captions) == sorted(keys), "the resumed pass lost captions"
    assert captions["0_0"] == "caption of 0"
    assert captions["3_0"] == "caption of 3"


def test_imagenet_resume_skips_decoded_rows_only(tmp_path: Path, stub_env, monkeypatch) -> None:
    """An undecodable source row must not shift the resume offset."""
    state = {"fail_after": 3}
    # Row 1 cannot be decoded, so the encoded rows are 10, 30, 40, 50.
    source = [
        {"image": _image(10), "label": 0},
        {"image": None, "label": 1},
        {"image": _image(30), "label": 3},
        {"image": _image(40), "label": 4},
        {"image": _image(50), "label": 5},
    ]

    def fake_load_dataset(*_a, **_k):
        def gen():
            for i, row in enumerate(source):
                if i >= state["fail_after"]:
                    raise RuntimeError("stream interrupted")
                yield row
        return gen()

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(extract, "_imagenet_prompts",
                        lambda: (["a", "bb"], ["0_0", "0_1"], 1, 2))
    cache_dir = tmp_path / "imagenet"

    with pytest.raises(RuntimeError, match="stream interrupted"):
        extract.extract_imagenet(model_cfg=stub_env, cache_dir=cache_dir,
                                 batch_size=1, device="cpu")

    state["fail_after"] = 99
    extract.extract_imagenet(model_cfg=stub_env, cache_dir=cache_dir,
                             batch_size=1, device="cpu")

    labels = np.load(cache_dir / "labels.npy")
    images = np.load(cache_dir / "image_embeddings.npy")
    np.testing.assert_array_equal(labels, np.array([0, 3, 4, 5], dtype=np.int64))
    np.testing.assert_array_equal(images[:, 0],
                                  np.array([10, 30, 40, 50], dtype=np.float32))


def test_caption_sidecar_ignores_a_truncated_final_line(tmp_path: Path) -> None:
    """A kill mid-write leaves a partial line; the rest must still be readable."""
    sidecar = extract._CaptionSidecar(tmp_path)
    sidecar.append([("a_0", "first"), ("b_0", "second")])
    with open(sidecar.path, "a") as f:
        f.write('{"k": "c_0", "c": "thi')
    assert sidecar.read() == {"a_0": "first", "b_0": "second"}

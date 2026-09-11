"""Checkpoint idempotence and the data loader's worker rule.

`train_method` skips a method whose checkpoint is already on disk. The question
these tests pin down is what "already on disk" means: `save_pretrained` writes
config.json before the weights, so a directory holding only config.json is an
interrupted save and has to be retrained, not skipped.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from src.training import trainer
from src.training.trainer import train_method
from src.utils.config import MethodConfig, TrainingConfig
from tests.conftest import make_coco_cache

DIM = 8


def _training() -> TrainingConfig:
    return TrainingConfig(lr=1e-3, num_epochs=1, batch_size=8, k=2, latent_size=8,
                          warmup_ratio=0.5, device="cpu", seeds=[0])


def _train(cache_dir: Path, save_dir: Path):
    return train_method(method=MethodConfig(name="separated"), training=_training(),
                        cache_dir=cache_dir, hidden_size=DIM, save_dir=save_dir,
                        seed=0, split="train")


def test_a_finished_checkpoint_is_reused(tmp_path: Path) -> None:
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=8, caps_per_image=5, dim=DIM)
    save_dir = tmp_path / "run"

    first = _train(cache_dir, save_dir)
    assert first.history, "the first call has to train"
    second = _train(cache_dir, save_dir)
    assert second.history == [], "a finished checkpoint must be loaded, not retrained"


def test_a_half_written_checkpoint_is_retrained(tmp_path: Path) -> None:
    """config.json without weights is an interrupted save, not a finished one."""
    cache_dir = tmp_path / "coco"
    make_coco_cache(cache_dir, n_images=8, caps_per_image=5, dim=DIM)
    save_dir = tmp_path / "run"

    _train(cache_dir, save_dir)
    final = save_dir / "final"
    for name in ("model.safetensors", "pytorch_model.bin"):
        (final / name).unlink(missing_ok=True)
    assert (final / "config.json").exists()

    again = _train(cache_dir, save_dir)
    assert again.history, "the interrupted save must be retrained, not reloaded"
    assert (final / "model.safetensors").exists()


def test_workers_follow_the_start_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fork shares the memory maps; spawn would pickle and materialize them."""
    big = trainer._WORKER_MIN_ROWS
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_k: "fork")
    assert trainer._dataloader_workers(big) == trainer._FORK_DATALOADER_WORKERS
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_k: "spawn")
    assert trainer._dataloader_workers(big) == 0


def test_a_small_split_stays_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda **_k: "fork")
    assert trainer._dataloader_workers(trainer._WORKER_MIN_ROWS - 1) == 0

"""The chunked writer behind extraction: bounded memory and a correct resume.

The CC3M pass writes about 2.87M rows, so an interruption partway through has
to be recoverable without re-encoding what is already on disk. These tests
drive `_ChunkWriter` and `_assemble` directly, so nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.extract import _assemble, _ChunkWriter

DIM = 4


def _feed(writer: _ChunkWriter, start: int, count: int) -> None:
    """Append `count` rows whose first column is the row's global index."""
    for i in range(start, start + count):
        row = np.full((1, DIM), float(i), dtype=np.float32)
        writer.add(row, row * -1, [f"k{i}"], groups=1)


def test_chunks_are_flushed_at_the_chunk_size(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 25)
    assert writer.n_parts == 2 and writer.rows_done == 20
    writer.close()
    assert writer.n_parts == 3 and writer.rows_done == 25
    assert writer.all_keys() == [f"k{i}" for i in range(25)]


def test_resume_keeps_the_finished_chunks(tmp_path: Path) -> None:
    """A restart picks up at the recorded row and source-record counts."""
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 23)          # two chunks flushed, three rows still buffered
    del writer

    resumed = _ChunkWriter(tmp_path, chunk_size=10)
    assert resumed.rows_done == 20, "the three buffered rows were never written"
    assert resumed.groups_done == 20
    _feed(resumed, 20, 5)
    resumed.close()
    assert resumed.all_keys() == [f"k{i}" for i in range(25)]


def test_a_missing_chunk_file_restarts_the_split(tmp_path: Path) -> None:
    """A half-written chunk set is thrown away rather than guessed at."""
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 20)
    (tmp_path / "part_000001_image.npy").unlink()

    resumed = _ChunkWriter(tmp_path, chunk_size=10)
    assert resumed.rows_done == 0
    assert resumed.n_parts == 0
    assert not list(tmp_path.glob("part_*"))


def test_a_progress_record_that_overcounts_restarts_the_split(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 20)
    (tmp_path / "progress.json").write_text(
        json.dumps({"rows": 999, "groups": 999, "n_parts": 2}))

    resumed = _ChunkWriter(tmp_path, chunk_size=10)
    assert resumed.rows_done == 0


def test_stale_chunks_beyond_the_record_are_removed(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 20)
    for name in ("part_000007_image.npy", "part_000007_text.npy"):
        np.save(tmp_path / name, np.zeros((10, DIM), dtype=np.float32))

    resumed = _ChunkWriter(tmp_path, chunk_size=10)
    assert resumed.rows_done == 20
    assert not (tmp_path / "part_000007_image.npy").exists()


def test_assemble_concatenates_in_chunk_order(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 25)
    writer.close()
    chunks = list(writer.iter_chunks())

    out = tmp_path / "image_embeddings.npy"
    _assemble([c[0] for c in chunks], out, total_rows=25, dim=DIM)
    arr = np.load(out, mmap_mode="r")
    assert arr.shape == (25, DIM)
    np.testing.assert_array_equal(arr[:, 0], np.arange(25, dtype=np.float32))


def test_assemble_refuses_a_wrong_row_count(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    _feed(writer, 0, 12)
    writer.close()
    chunks = list(writer.iter_chunks())
    try:
        _assemble([c[0] for c in chunks], tmp_path / "bad.npy", total_rows=99, dim=DIM)
    except RuntimeError as exc:
        assert "expected 99" in str(exc)
    else:
        raise AssertionError("a row-count mismatch must be reported, not written")


def test_row_and_key_counts_must_agree(tmp_path: Path) -> None:
    writer = _ChunkWriter(tmp_path, chunk_size=10)
    row = np.zeros((2, DIM), dtype=np.float32)
    try:
        writer.add(row, row, ["only-one-key"], groups=1)
    except ValueError as exc:
        assert "row count" in str(exc)
    else:
        raise AssertionError("mismatched rows and keys must be rejected")

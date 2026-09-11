"""The Table 1 renderer, on a results tree written by hand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.table1 import (
    METHOD_LABELS,
    collect,
    render,
    write_table1,
    zeroshot_filename,
)
from tests.conftest import write_eval_json

METHODS = ["shared", "separated", "ours"]


def _tree(root: Path, seeds=(0,), scale: float = 1.0) -> None:
    """Three methods per seed, with `ours` best on every column."""
    values = {
        "shared": {"recon": 0.90, "r1": 0.10, "zs": 0.20},
        "separated": {"recon": 0.50, "r1": 0.20, "zs": 0.25},
        "ours": {"recon": 0.50, "r1": 0.30, "zs": 0.30},
    }
    for seed in seeds:
        for method, v in values.items():
            bump = scale * seed * 0.01
            write_eval_json(root, seed, method, "recon_coco.json",
                            {"recon_error": v["recon"] + bump})
            write_eval_json(root, seed, method, "recon_imagenet.json",
                            {"recon_error": v["recon"] * 2 + bump})
            write_eval_json(root, seed, method, "retrieval.json", {
                "I2T": {"R@1": v["r1"] + bump, "R@5": v["r1"] + 0.1, "R@10": v["r1"] + 0.2},
                "T2I": {"R@1": v["r1"] - 0.01, "R@5": v["r1"] + 0.05, "R@10": v["r1"] + 0.15},
            })
            write_eval_json(root, seed, method, "zeroshot.json",
                            {"accuracy": v["zs"] + bump})


def test_collect_finds_every_column(tmp_path: Path) -> None:
    _tree(tmp_path)
    values = collect(tmp_path, METHODS, [0])
    assert all(len(values["ours"][ci]) == 1 for ci in range(9))
    assert values["ours"][0] == [0.50]
    assert values["ours"][8] == [0.30]


def test_single_seed_marks_best_and_second(tmp_path: Path) -> None:
    _tree(tmp_path)
    md = render(tmp_path, METHODS, [0], latex=False)
    lines = {row.split("|")[1].strip(): row for row in md.splitlines() if row.startswith("|")}
    ours = lines[METHOD_LABELS["ours"]]
    shared = lines[METHOD_LABELS["shared"]]
    # I to T R@1: ours 30.00 is best, separated 20.00 second, shared plain.
    assert "**30.00**" in ours
    assert "_20.00_" in lines[METHOD_LABELS["separated"]]
    assert "10.00" in shared and "**10.00**" not in shared


def test_a_tie_on_the_best_value_leaves_second_unmarked(tmp_path: Path) -> None:
    """Second best is only marked when its mean differs from the best."""
    _tree(tmp_path)
    md = render(tmp_path, METHODS, [0], latex=False)
    # COCO recon: separated and ours are both 0.5000, so no italic on that column.
    recon_cells = [row.split("|")[2].strip() for row in md.splitlines()
                   if row.startswith("| " + METHOD_LABELS["separated"])]
    assert recon_cells and not recon_cells[0].startswith("_")


def test_several_seeds_report_mean_and_std(tmp_path: Path) -> None:
    _tree(tmp_path, seeds=(0, 1, 2))
    md = render(tmp_path, METHODS, [0, 1, 2], latex=False)
    assert "±" in md
    assert "Seeds: 0, 1, 2" in md


def test_missing_files_render_as_a_gap(tmp_path: Path) -> None:
    write_eval_json(tmp_path, 0, "ours", "zeroshot.json", {"accuracy": 0.4})
    md = render(tmp_path, ["ours"], [0], latex=False)
    assert "--" in md
    assert "40.00" in md


def test_write_table1_emits_markdown_and_latex(tmp_path: Path) -> None:
    _tree(tmp_path)
    path = write_table1(tmp_path, METHODS, [0], title="unit test")
    assert path.name == "table1.md"
    assert (tmp_path / "table1.tex").exists()
    tex = (tmp_path / "table1.tex").read_text()
    assert r"\textbf{30.00}" in tex
    assert r"\underline{" in tex
    assert "unit test" in path.read_text()


def test_column_order_matches_the_paper(tmp_path: Path) -> None:
    _tree(tmp_path)
    header = render(tmp_path, METHODS, [0], latex=False).splitlines()[4]
    cols = [c.strip() for c in header.split("|") if c.strip()]
    assert cols == ["Method", "COCO Recon ↓", "I→T R@1", "I→T R@5", "I→T R@10",
                    "T→I R@1", "T→I R@5", "T→I R@10", "ImageNet Recon ↓",
                    "Zero-shot top-1 ↑"]


def test_the_zeroshot_column_reads_the_filtered_variant(tmp_path: Path) -> None:
    """`eval.zeroshot_variant: filtered` names the file zeroshot_filtered.json.

    The table has to read that name too, or the column renders as "--" for
    every method with no error and no warning, which is indistinguishable from
    an evaluation that never ran.
    """
    _tree(tmp_path)
    for method in METHODS:
        raw = tmp_path / "seed0" / "eval" / method / "zeroshot.json"
        payload = json.loads(raw.read_text())
        raw.unlink()
        write_eval_json(tmp_path, 0, method, "zeroshot_filtered.json", payload)

    values = collect(tmp_path, METHODS, [0])
    assert values["ours"][8] == [0.30]
    text = render(tmp_path, METHODS, [0], latex=False)
    assert "30.00" in text


def test_the_zeroshot_file_name_follows_the_variant() -> None:
    assert zeroshot_filename("raw") == "zeroshot.json"
    assert zeroshot_filename("filtered") == "zeroshot_filtered.json"
    with pytest.raises(ValueError, match="unknown zeroshot variant"):
        zeroshot_filename("everything")

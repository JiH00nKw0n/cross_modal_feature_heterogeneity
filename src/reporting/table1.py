"""Table 1: the downstream comparison of the five methods, as markdown and LaTeX.

Reads, for every seed and every method:

    <root>/seed{S}/eval/<method>/recon_coco.json      COCO reconstruction error
    <root>/seed{S}/eval/<method>/retrieval.json       COCO retrieval recalls
    <root>/seed{S}/eval/<method>/recon_imagenet.json  ImageNet reconstruction error
    <root>/seed{S}/eval/<method>/zeroshot.json        ImageNet zero-shot top-1
                                                      (zeroshot_filtered.json
                                                      when the run used the
                                                      filtered variant)

Column order is the paper's: COCO reconstruction (lower is better), I to T R@1,
R@5, R@10, T to I R@1, R@5, R@10, ImageNet reconstruction (lower is better),
zero-shot top-1 (higher is better).

Formatting rules. Recall and accuracy columns are multiplied by 100 and printed
with two decimals; reconstruction is printed with four decimals. With more than
one seed a cell shows "mean +/- std" over the seeds that produced a value, and
best and second best are decided by the mean alone. Best is bold, second best
is italic in markdown and underlined in LaTeX; second best is only marked when
its mean actually differs from the best, so two identical numbers do not get
ranked against each other.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Method key in the config -> the name printed in the table.
METHOD_LABELS = {
    "shared": "Shared SAE",
    "separated": "Modality-Specific SAEs",
    "iso_align": "Iso-Energy Alignment",
    "group_sparse": "Group-Sparse",
    "ours": "Post-hoc Alignment (Ours)",
}

#: Order the methods appear in, when the caller does not say.
DEFAULT_METHODS = ("shared", "separated", "iso_align", "group_sparse", "ours")


#: Result file name per `eval.zeroshot_variant`. The pipeline writes the name
#: this mapping gives and the table reads every name in it, so switching the
#: variant cannot leave the column empty. It lives here because it is the one
#: fact the writer of the file and the reader of it have to agree on.
ZEROSHOT_FILES = {"raw": "zeroshot.json", "filtered": "zeroshot_filtered.json"}


def zeroshot_filename(variant: str) -> str:
    """The zero-shot result file name for one variant; unknown names are refused."""
    try:
        return ZEROSHOT_FILES[variant]
    except KeyError:
        raise ValueError(
            f"unknown zeroshot variant {variant!r}; expected one of "
            f"{', '.join(sorted(ZEROSHOT_FILES))}"
        ) from None


@dataclass(frozen=True)
class Column:
    """One table column: where its number lives and how to print it.

    `sources` names the JSON files that may hold this column, tried in order,
    which is how the zero-shot column reads whichever variant the run wrote.
    `key` is the field inside (a tuple walks nested objects), `direction` says
    whether small or large is better, and `is_percent` marks the columns
    multiplied by 100.
    """

    sources: tuple[str, ...]
    key: Any
    md_label: str
    tex_label: str
    direction: str
    is_percent: bool


COLUMNS: tuple[Column, ...] = (
    Column(("recon_coco.json",), "recon_error", "COCO Recon ↓",
           r"COCO Recon $\downarrow$", "min", False),
    Column(("retrieval.json",), ("I2T", "R@1"), "I→T R@1", r"I$\rightarrow$T R@1", "max", True),
    Column(("retrieval.json",), ("I2T", "R@5"), "I→T R@5", "R@5", "max", True),
    Column(("retrieval.json",), ("I2T", "R@10"), "I→T R@10", "R@10", "max", True),
    Column(("retrieval.json",), ("T2I", "R@1"), "T→I R@1", r"T$\rightarrow$I R@1", "max", True),
    Column(("retrieval.json",), ("T2I", "R@5"), "T→I R@5", "R@5", "max", True),
    Column(("retrieval.json",), ("T2I", "R@10"), "T→I R@10", "R@10", "max", True),
    Column(("recon_imagenet.json",), "recon_error", "ImageNet Recon ↓",
           r"ImageNet Recon $\downarrow$", "min", False),
    Column(tuple(ZEROSHOT_FILES.values()), "accuracy", "Zero-shot top-1 ↑",
           r"Zero-shot $\uparrow$", "max", True),
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def _get(payload: dict | None, key: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(key, tuple):
        cur: Any = payload
        for part in key:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur if isinstance(cur, (int, float)) else None
    value = payload.get(key)
    return value if isinstance(value, (int, float)) else None


def collect(root: str | Path, methods: Sequence[str], seeds: Sequence[int],
            ) -> dict[str, dict[int, list[float]]]:
    """Per method, per column index, the values found across seeds.

    A missing file contributes nothing, so a partially finished run still
    renders and the gaps show up as "--".
    """
    root = Path(root)
    values: dict[str, dict[int, list[float]]] = {m: {i: [] for i in range(len(COLUMNS))}
                                                 for m in methods}
    for seed in seeds:
        seed_dir = root / f"seed{seed}" / "eval"
        for method in methods:
            cache: dict[str, dict | None] = {}
            for ci, col in enumerate(COLUMNS):
                for source in col.sources:
                    if source not in cache:
                        cache[source] = _load_json(seed_dir / method / source)
                    v = _get(cache[source], col.key)
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        values[method][ci].append(float(v))
                        break
    return values


def _mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _rank_marks(means: list[tuple[str, float | None]], direction: str) -> dict[str, str]:
    """Mark the best and, when it differs, the second best method of one column."""
    items = [(m, v) for m, v in means if v is not None]
    marks = {m: "" for m, _ in means}
    if not items:
        return marks
    items.sort(key=lambda x: x[1], reverse=(direction == "max"))
    marks[items[0][0]] = "best"
    if len(items) > 1 and items[0][1] != items[1][1]:
        marks[items[1][0]] = "second"
    return marks


def _fmt_number(v: float, col: Column) -> str:
    if col.is_percent:
        return f"{100 * v:.2f}"
    return f"{v:.4f}"


def _fmt_cell(vals: list[float], col: Column, mark: str, latex: bool) -> str:
    mean, std = _mean_std(vals)
    if mean is None:
        return "--"
    body = _fmt_number(mean, col)
    if std is not None:
        body = f"{body} ± {_fmt_number(std, col)}" if not latex \
            else rf"{body} $\pm$ {_fmt_number(std, col)}"
    if latex:
        if mark == "best":
            return r"\textbf{" + body + "}"
        if mark == "second":
            return r"\underline{" + body + "}"
        return body
    if mark == "best":
        return f"**{body}**"
    if mark == "second":
        return f"_{body}_"
    return body


def render(root: str | Path, methods: Sequence[str], seeds: Sequence[int],
           *, latex: bool, title: str = "") -> str:
    """Render Table 1 as markdown or as a LaTeX tabular."""
    values = collect(root, methods, seeds)
    marks: dict[int, dict[str, str]] = {}
    for ci, col in enumerate(COLUMNS):
        means = [(m, _mean_std(values[m][ci])[0]) for m in methods]
        marks[ci] = _rank_marks(means, col.direction)

    if latex:
        lines = [r"\begin{tabular}{l" + "c" * len(COLUMNS) + "}", r"\toprule"]
        header = ["Method"] + [c.tex_label for c in COLUMNS]
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")
        for m in methods:
            cells = [METHOD_LABELS.get(m, m)]
            for ci, col in enumerate(COLUMNS):
                cells.append(_fmt_cell(values[m][ci], col, marks[ci][m], latex=True))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines) + "\n"

    lines = [f"# {title or 'Table 1'}", ""]
    n_seeds = len(seeds)
    lines.append(
        f"Seeds: {', '.join(str(s) for s in seeds)} "
        f"({n_seeds} seed{'s' if n_seeds != 1 else ''}). "
        "Cells with several seeds show the mean and the sample standard deviation. "
        "Recall and accuracy are percentages. Bold marks the best method in a column "
        "and italic the second best, ranked by the mean."
    )
    lines.append("")
    header = ["Method"] + [c.md_label for c in COLUMNS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in methods:
        row = [METHOD_LABELS.get(m, m)]
        for ci, col in enumerate(COLUMNS):
            row.append(_fmt_cell(values[m][ci], col, marks[ci][m], latex=False))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_table1(root: str | Path, methods: Sequence[str] = DEFAULT_METHODS,
                 seeds: Sequence[int] = (0,), *, title: str = "") -> Path:
    """Write `<root>/table1.md` and `<root>/table1.tex`; return the markdown path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    md_path = root / "table1.md"
    md_path.write_text(render(root, methods, seeds, latex=False, title=title))
    (root / "table1.tex").write_text(render(root, methods, seeds, latex=True, title=title))
    logger.info("[table1] wrote %s and %s", md_path, root / "table1.tex")
    return md_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="output root holding seed*/eval/<method>/")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--title", default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = write_table1(args.root, args.methods, args.seeds, title=args.title)
    print(path.read_text())


if __name__ == "__main__":
    main()

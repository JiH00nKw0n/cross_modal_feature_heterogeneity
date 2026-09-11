"""Which rebuttal analysis runs on which setting.

This list is the only place the pipeline learns that an analysis exists. An
author porting one analysis writes `src/rebuttal/<name>.py` and stops there;
the person integrating the work adds the row here. Keeping the two jobs apart
is what lets several analyses be ported at the same time without any of them
touching a shared file.

A row is (name, module_path, settings), where

  name          the analysis name, which is also the stem of the files it
                writes: `<out_dir>/<name>.json`, `<name>.md` and, when it draws
                one, `<name>.pdf` plus `<name>.png`.
  module_path   the importable module holding `run(setting, *, out_dir, device,
                **knobs)`.
  settings      the setting tags the analysis applies to, a subset of
                ("coco_k8", "cc3m_k32"). An analysis that needs two captions of
                one photo runs on "coco_k8" only, because CC3M carries a single
                caption per image.

The pipeline runs the rows in order, so an analysis that reads another one's
json must appear after it.
"""

from __future__ import annotations

from typing import NamedTuple


class Analysis(NamedTuple):
    """One registered analysis. See the module docstring for the fields."""

    name: str
    module_path: str
    settings: tuple[str, ...]


#: Every registered analysis, in run order. Filled by the integrator.
ANALYSES: list[Analysis] = []


def analyses_for(tag: str, wanted: list[str] | tuple[str, ...] | None = None) -> list[Analysis]:
    """The registered analyses that apply to one setting, in run order.

    `wanted` filters by name. None, an empty list, or a list holding the single
    entry "all" means every analysis that applies to the setting. A name in
    `wanted` that is not registered raises, rather than being skipped, because a
    typo in a config would otherwise read as a deliberately empty run.
    """
    known = {a.name for a in ANALYSES}
    if wanted and list(wanted) != ["all"]:
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise ValueError(
                f"unknown analyses {unknown}; registered names are {sorted(known)}"
            )
        selected = set(wanted)
    else:
        selected = known
    return [a for a in ANALYSES if a.name in selected and tag in a.settings]


__all__ = ["Analysis", "ANALYSES", "analyses_for"]

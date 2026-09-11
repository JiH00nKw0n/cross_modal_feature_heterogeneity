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


#: The question each analysis answers, as a full sentence. The report stage
#: prints it as the heading above that analysis's section, so a reader of the
#: combined document learns what was measured before reading any number.
#: Keyed by analysis name.
QUESTIONS: dict[str, str] = {
    "same_modality_control":
        "Is the reported gap larger than the variation between two ordinary "
        "training runs?",
    "coco80_correspondence":
        "Do matched latents point at the same concept, judged against labels "
        "the model had no part in producing?",
    "coco80_heterogeneity":
        "How large is the gap when it is measured from those labels alone, "
        "with no co-activation anywhere?",
    "match_confidence":
        "How strong and how unambiguous are the matches?",
    "correlation_bands":
        "How does the distance between two feature directions change across "
        "co-activation bands?",
    "one_to_many_span":
        "What happens when the correspondence is one to many rather than one "
        "to one?",
    "one_to_many_splitting":
        "Is the gap explained by one concept being split across several "
        "latents?",
    "alignment_ceiling":
        "Can a better matching, or one global transform, remove the gap?",
    "stability_conditioned":
        "Does the gap remain on the concepts that two independent runs both "
        "recover?",
    "confidence_ablation":
        "How does retrieval change when only the most confident matches are "
        "kept?",
    "alignment_methods":
        "Do other post-hoc alignment methods do better?",
}

#: Every registered analysis, in run order.
#:
#: "coco_k8" is the paper's Figure 2 point (COCO, 8 active latents per input,
#: 30 epochs); "cc3m_k32" is its Table 1 point (CC3M, 32 active latents per
#: input, 10 epochs). Three analyses run on one setting only:
#:
#:   correlation_bands    cc3m_k32 only. For coco_k8 the Figure 2 stage already
#:                        writes the same per-band statistics, as
#:                        figure2_bin_stats.md.
#:   confidence_ablation  cc3m_k32 only. It scores held-out COCO retrieval,
#:                        which is not held out for a model trained on COCO.
#:   alignment_methods    cc3m_k32 only, for the same reason.
ANALYSES: list[Analysis] = [
    Analysis("same_modality_control", "src.rebuttal.same_modality_control",
             ("coco_k8", "cc3m_k32")),
    Analysis("coco80_correspondence", "src.rebuttal.coco80_correspondence",
             ("coco_k8", "cc3m_k32")),
    Analysis("coco80_heterogeneity", "src.rebuttal.coco80_heterogeneity",
             ("coco_k8", "cc3m_k32")),
    Analysis("match_confidence", "src.rebuttal.match_confidence",
             ("coco_k8", "cc3m_k32")),
    Analysis("correlation_bands", "src.rebuttal.correlation_bands",
             ("cc3m_k32",)),
    Analysis("one_to_many_span", "src.rebuttal.one_to_many_span",
             ("coco_k8", "cc3m_k32")),
    Analysis("one_to_many_splitting", "src.rebuttal.one_to_many_splitting",
             ("coco_k8", "cc3m_k32")),
    Analysis("alignment_ceiling", "src.rebuttal.alignment_ceiling",
             ("coco_k8", "cc3m_k32")),
    Analysis("stability_conditioned", "src.rebuttal.stability_conditioned",
             ("coco_k8", "cc3m_k32")),
    Analysis("confidence_ablation", "src.rebuttal.confidence_ablation",
             ("cc3m_k32",)),
    Analysis("alignment_methods", "src.rebuttal.alignment_methods",
             ("cc3m_k32",)),
]


def question_for(name: str) -> str:
    """The question one analysis answers, or its own name when none is recorded."""
    return QUESTIONS.get(name, name)


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


__all__ = ["Analysis", "ANALYSES", "QUESTIONS", "analyses_for", "question_for"]

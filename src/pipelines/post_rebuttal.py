"""Post-rebuttal pipeline: the two paper deliverables, then the extra analyses.

Stages (selected with `--stage`):

  figure2   The COCO density figure. Delegates to `src.pipelines.multi_density`
            on `cfg.figure2`, which trains the seed-0 modality-specific model
            and writes its image-to-text co-activation panel.
  table1    The CC3M downstream table. Delegates to
            `src.pipelines.cc3m_downstream` on `cfg.table1`, which trains every
            method for every seed and writes each seed's panel.
  rebuttal  Trains the second COCO model, the independent run the same-modality
            comparisons need; builds the extra co-activation panels; then runs
            every registered analysis on every configured setting.
  report    Gathers an inventory of every file produced, Table 1, the Figure 2
            per-band statistics and each analysis report into one file at
            `<root>/post_rebuttal_results.md`.
  all       The four above, in that order.

Every stage is idempotent. A checkpoint or a panel that already exists is not
rebuilt, and a panel on disk is reused only when its sidecar says it was built
under the rules being asked for. An analysis is skipped only when it left a
complete result behind, which means its json, its report and no record of a
failure. An analysis that died after writing its json is run again from the
start rather than counted as done.

One analysis that raises does not stop the rest. Its traceback is written to
`<out_dir>/<name>.error.txt`, the remaining analyses still run, the report is
still written, and the process then exits non-zero naming every analysis that
failed.

The extra panels, all on the FULL training split of their setting, written to
`<root>/rebuttal/<tag>/panels/`:

  img_img.npz           image side of model A against image side of model B,
                        both reading the same photograph. Measures how far two
                        independent training runs of one modality land apart.
  txt_txt.npz           text side of model A against text side of model B, both
                        reading the same caption.
  txt_txt_diffcap.npz   the same two text sides reading two different captions
                        of one photograph, which adds input mismatch while
                        keeping the modality fixed. Built for coco_k8 only,
                        because CC3M carries a single caption per image.
  img_txt_null.npz      model A's own image and text sides with the pairing
                        destroyed by a row shuffle, which is the noise floor.

The image-to-text panel itself is not built here: Figure 2 and Table 1 already
wrote it, and rebuilding it would risk two versions of the paper's matching.
"""

from __future__ import annotations

import importlib
import json
import logging
import time
import traceback
from pathlib import Path

from src.alignment import build_panel, load_panel, panel_mismatch, save_panel
from src.rebuttal.common import Setting, settings_from_config
from src.rebuttal.registry import Analysis, analyses_for, question_for
from src.utils.config import Config, MethodConfig

logger = logging.getLogger(__name__)

#: Stages this pipeline understands. `run.py` offers the union of every
#: pipeline's stages, so a name meant for another kind is refused here rather
#: than silently doing nothing and exiting successfully.
STAGES = ("all", "figure2", "table1", "rebuttal", "report")

#: Extra panels per setting: pairing name -> whether it needs a second model.
#: `txt_txt_diffcap` needs a cache with more than one caption per image, so it
#: is restricted to the settings listed in `_DIFFCAP_SETTINGS`.
_EXTRA_PAIRINGS = ("img_img", "txt_txt", "txt_txt_diffcap")

#: Settings whose cache holds more than one caption per photograph.
_DIFFCAP_SETTINGS = ("coco_k8",)

#: File name of the noise-floor panel.
_NULL_PANEL = "img_txt_null.npz"

#: Name of the combined report the report stage writes at the output root.
REPORT_NAME = "post_rebuttal_results.md"

#: File suffixes the inventory of the combined report lists. Checkpoints and
#: panels are left out: they are inputs to the numbers, not readable results,
#: and a panel is hundreds of megabytes.
_INVENTORY_SUFFIXES = (".md", ".pdf", ".png", ".json", ".tex")

#: Weight files that mark a directory as a trained checkpoint. Every small file
#: beside one of these, such as a checkpoint's config.json, is left out of the
#: inventory for the same reason the weights themselves are.
_CHECKPOINT_WEIGHTS = ("model.safetensors", "pytorch_model.bin")

#: How each inventory suffix is described in the table.
_INVENTORY_KINDS = {
    ".md": "report, readable text",
    ".pdf": "figure, vector",
    ".png": "figure, raster image",
    ".json": "numbers, machine readable",
    ".tex": "table, LaTeX source",
}


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "post_rebuttal", f"Wrong kind: {cfg.kind}"
    if stage not in STAGES:
        raise ValueError(
            f"stage {stage!r} does not exist for kind 'post_rebuttal'; "
            f"it understands {', '.join(STAGES)}"
        )
    if cfg.figure2 is None or cfg.table1 is None:
        raise ValueError(
            "a post_rebuttal config must carry both figure2 and table1; "
            "pull them in with !ref from configs/post_rebuttal/"
        )
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    if stage in ("all", "figure2"):
        from src.pipelines.multi_density import run as figure2_run

        logger.info("[post_rebuttal] figure2 -> %s", cfg.figure2.output.root)
        figure2_run(cfg.figure2, stage="all")

    if stage in ("all", "table1"):
        from src.pipelines.cc3m_downstream import run as table1_run

        logger.info("[post_rebuttal] table1 -> %s", cfg.table1.output.root)
        table1_run(cfg.table1, stage="all")

    failures: list[tuple[str, str, Path]] = []
    if stage in ("all", "rebuttal"):
        failures = _rebuttal_stage(cfg)

    if stage in ("all", "report"):
        _report_stage(cfg)

    if failures:
        # The report was written first on purpose: it names every analysis that
        # produced no file, so the failures are visible in the document as well
        # as in this message. The non-zero exit is what a batch runner reads.
        for tag, name, path in failures:
            logger.error("[post_rebuttal] FAILED %s / %s, traceback in %s",
                         tag, name, path)
        raise SystemExit(
            f"{len(failures)} analysis run(s) failed: "
            + ", ".join(f"{tag}/{name}" for tag, name, _ in failures)
        )


# --------------------------------------------------------------------------- #
# rebuttal stage
# --------------------------------------------------------------------------- #
def _rebuttal_stage(cfg: Config) -> list[tuple[str, str, Path]]:
    """Train model B, build the extra panels, run the analyses.

    Returns one entry per analysis that raised, as (setting tag, analysis name,
    path of the file holding its traceback). An empty list means every analysis
    that was asked for either ran or was skipped as already done.
    """
    settings = settings_from_config(cfg)
    if not settings:
        logger.warning("[post_rebuttal] cfg.rebuttal.settings selects no setting")
        return []

    _train_second_coco_model(cfg, settings.get("coco_k8"))

    failures: list[tuple[str, str, Path]] = []
    for tag, setting in settings.items():
        logger.info("[post_rebuttal] setting %s: %s", tag, setting.title())
        setting.out_dir.mkdir(parents=True, exist_ok=True)
        _write_setting_manifest(setting)
        _build_extra_panels(cfg, setting)
        failures += _run_analyses(cfg, setting,
                                  device=_training_for(cfg, setting).device)
    return failures


def _training_for(cfg: Config, setting: Setting):
    """The training block that produced this setting's checkpoints.

    A setting's panels have to be built with the same batch size and on the
    same device as the models they read, so the CC3M setting reads the Table 1
    config rather than the Figure 2 one. Reading the wrong block is invisible in
    the result and shows up only as a memory error on a machine whose batch size
    was lowered in the config that governs that corpus.
    """
    return cfg.table1.training if setting.dataset == "cc3m" else cfg.figure2.training


def _train_second_coco_model(cfg: Config, setting: Setting | None) -> None:
    """Train the COCO model B, with the same config as Figure 2's model A.

    Only the seed differs, which is the point: the same-modality comparison
    asks how far two independent runs of one modality land apart, so everything
    except the seed has to be held fixed.
    """
    if setting is None:
        return
    from src.training.trainer import train_method

    save_dir = setting.ckpt_b.parent
    logger.info("[post_rebuttal] COCO model B (seed %d) -> %s",
                cfg.rebuttal.coco_seed_b, save_dir)
    train_method(
        method=MethodConfig(name="separated"),
        training=cfg.figure2.training,
        cache_dir=setting.cache_dir,
        hidden_size=cfg.figure2.models[0].hidden_size,
        save_dir=save_dir,
        seed=int(cfg.rebuttal.coco_seed_b),
        split=setting.split,
    )


def _build_extra_panels(cfg: Config, setting: Setting) -> None:
    """Build the same-modality panels and the noise-floor panel for one setting.

    Skipped with a warning when a checkpoint is missing, so that a setting whose
    training has not finished does not stop the other setting's analyses.
    """
    from src.models import TwoSidedTopKSAE

    if not setting.ckpt_a.exists():
        logger.warning("[post_rebuttal] %s: missing model A at %s, no extra panels",
                       setting.tag, setting.ckpt_a)
        return
    setting.panels_dir.mkdir(parents=True, exist_ok=True)
    model_a = TwoSidedTopKSAE.from_pretrained(setting.ckpt_a)

    model_b = None
    if setting.ckpt_b.exists():
        model_b = TwoSidedTopKSAE.from_pretrained(setting.ckpt_b)
    else:
        logger.warning("[post_rebuttal] %s: missing model B at %s, skipping the "
                       "same-modality panels", setting.tag, setting.ckpt_b)

    pairings = [p for p in _EXTRA_PAIRINGS
                if p != "txt_txt_diffcap" or setting.tag in _DIFFCAP_SETTINGS]
    if model_b is not None:
        for pairing in pairings:
            _panel_once(
                cfg, setting, path=setting.panel_path(pairing), pairing=pairing,
                model=model_a, model_b=model_b, shuffle_seed=0,
                ckpt_a=setting.ckpt_a, ckpt_b=setting.ckpt_b,
            )

    _panel_once(
        cfg, setting, path=setting.panels_dir / _NULL_PANEL, pairing="img_txt",
        model=model_a, model_b=None, shuffle_seed=int(cfg.rebuttal.null_seed),
        ckpt_a=setting.ckpt_a, ckpt_b=None,
    )


def _panel_once(cfg: Config, setting: Setting, *, path: Path, pairing: str,
                model, model_b, shuffle_seed: int, ckpt_a, ckpt_b) -> None:
    """Build one panel unless a panel built under the same rules is on disk."""
    if path.exists():
        why = panel_mismatch(load_panel(path), split=setting.split, pairing=pairing,
                             max_samples=0, shuffle_seed=shuffle_seed)
        if why is None:
            logger.info("[post_rebuttal][skip] %s exists", path)
            return
        logger.warning("[post_rebuttal] rebuilding %s: %s", path, why)
    training = _training_for(cfg, setting)
    payload = build_panel(
        model=model, cache_dir=setting.cache_dir, split=setting.split,
        batch_size=training.batch_size,
        device=training.device,
        max_samples=0, shuffle_seed=shuffle_seed, pairing=pairing,
        model_b=model_b, ckpt_a=ckpt_a, ckpt_b=ckpt_b,
    )
    save_panel(path, payload)
    logger.info("[post_rebuttal] saved %s", path)


def _run_analyses(cfg: Config, setting: Setting, *,
                  device: str) -> list[tuple[str, str, Path]]:
    """Run every registered analysis that applies to this setting, in order.

    One analysis that raises does not stop the others: its traceback is written
    to `<out_dir>/<name>.error.txt` and it is returned as a failure, so that the
    run reaches every remaining measurement and still exits non-zero.

    An analysis is skipped only when it left a complete result behind: its json,
    its report, and no traceback from a previous attempt. Every analysis writes
    its json before its figure and its report, so a crash or a kill in between
    leaves a json that describes work nobody can read. Such a json is deleted
    and the analysis is run again, because the analyses themselves return the
    json on disk untouched when one is there and would otherwise never rewrite
    the missing report.

    Returns one (setting tag, analysis name, traceback path) per failure.
    """
    selected = analyses_for(setting.tag, list(cfg.rebuttal.analyses or ["all"]))
    if not selected:
        logger.info("[post_rebuttal] %s: no analysis registered", setting.tag)
        return []
    knobs = {
        "tau": float(cfg.rebuttal.tau),
        "n_boot": int(cfg.rebuttal.n_boot),
        "null_seed": int(cfg.rebuttal.null_seed),
    }
    if cfg.rebuttal.batch_size is not None:
        # Only passed when the config asks for it, so that leaving the key out
        # keeps each analysis on the batch size its own author chose.
        knobs["batch_size"] = int(cfg.rebuttal.batch_size)
    failures: list[tuple[str, str, Path]] = []
    for index, analysis in enumerate(selected, start=1):
        out_json = setting.out_dir / f"{analysis.name}.json"
        error_path = setting.out_dir / f"{analysis.name}.error.txt"
        incomplete = _incomplete_result(setting.out_dir, analysis.name)
        if out_json.exists() and incomplete is None:
            logger.info("[post_rebuttal][skip] %s exists", out_json)
            error_path.unlink(missing_ok=True)
            continue
        if out_json.exists():
            logger.warning(
                "[post_rebuttal] %s: %s left an unfinished result behind (%s), "
                "so %s is deleted and the analysis is run again",
                setting.tag, analysis.name, incomplete, out_json)
            out_json.unlink()
        logger.info("[post_rebuttal] %s: start %s (%d of %d)",
                    setting.tag, analysis.name, index, len(selected))
        started = time.perf_counter()
        try:
            module = importlib.import_module(analysis.module_path)
            fn = getattr(module, "run", None)
            if fn is None:
                raise AttributeError(
                    f"{analysis.module_path} has no "
                    "run(setting, *, out_dir, device, **knobs)"
                )
            fn(setting, out_dir=setting.out_dir, device=device, **knobs)
        except Exception:
            elapsed = time.perf_counter() - started
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                f"analysis: {analysis.name}\n"
                f"module: {analysis.module_path}\n"
                f"setting: {setting.tag}\n"
                f"elapsed_seconds: {elapsed:.1f}\n\n"
                + traceback.format_exc()
            )
            logger.exception("[post_rebuttal] %s: %s failed after %.1f s, "
                             "traceback written to %s",
                             setting.tag, analysis.name, elapsed, error_path)
            failures.append((setting.tag, analysis.name, error_path))
            continue
        elapsed = time.perf_counter() - started
        error_path.unlink(missing_ok=True)
        logger.info("[post_rebuttal] %s: end %s, %.1f s elapsed",
                    setting.tag, analysis.name, elapsed)
    return failures


def _incomplete_result(out_dir: Path, name: str) -> str | None:
    """Why the files of one analysis do not amount to a finished result.

    Returns None when they do. Otherwise returns one clause naming what is
    wrong, which the caller puts in its log line. Either the report is missing,
    or the previous attempt recorded a traceback next to the json.
    """
    md_path = out_dir / f"{name}.md"
    error_path = out_dir / f"{name}.error.txt"
    if not md_path.exists():
        return f"its report {md_path} is missing"
    if error_path.exists():
        return f"the previous attempt failed and left {error_path}"
    return None


def _write_setting_manifest(setting: Setting) -> None:
    """Record which checkpoints and panels this setting's numbers came from."""
    path = setting.out_dir / "setting.json"
    path.write_text(json.dumps(setting.as_dict(), indent=2))


# --------------------------------------------------------------------------- #
# report stage
# --------------------------------------------------------------------------- #
def _report_stage(cfg: Config) -> None:
    """Gather everything this configuration produced into one document.

    The document opens with an inventory of every readable file under the
    output root, so that a reader who received only this file knows what else
    exists and where. It then reproduces Table 1 and the per-band statistics of
    Figure 2, and then, per setting, every registered analysis report in
    registry order, each under a heading that states the question that analysis
    answers.

    An analysis that did not run is named as missing rather than left out, so
    that an absent measurement cannot be mistaken for one that was chosen
    against.

    Each inlined report keeps its own title as a line of its section, and its
    headings are pushed down so that they sit under the section they belong to.
    The combined document has one level-1 heading, one level-2 heading per
    part, and one level-3 heading per analysis.
    """
    out_root = Path(cfg.output.root)
    settings = settings_from_config(cfg)
    lines = [
        "# Post-rebuttal measurements",
        "",
        ("Every number below was produced by the analyses under `src/rebuttal/`, "
         "reading the checkpoints and co-activation panels that the Figure 2 and "
         "Table 1 pipelines wrote. Each section is reproduced verbatim from one "
         "report, which is also readable on its own at the path named under the "
         "section heading."),
        "",
        ("Three rules hold everywhere and are never re-decided by an individual "
         "analysis. A latent counts as alive when it fired at least once over "
         "the full training split, and there is no firing-rate threshold "
         "anywhere. The co-activation correlation between an image latent and a "
         "text latent is the signed Pearson correlation stored in the panel, "
         "never its absolute value. The matching between the two sides is the "
         "one Hungarian assignment stored in that same panel, restricted to the "
         "latents that are alive on both sides."),
        "",
    ]
    lines += _inventory_section(out_root)
    lines += _figure_and_table_sections(cfg)

    for tag, setting in settings.items():
        lines += [f"## Setting {tag}", "", setting.title(), ""]
        selected: list[Analysis] = analyses_for(tag, list(cfg.rebuttal.analyses or ["all"]))
        if not selected:
            lines += ["No analysis is registered for this setting.", ""]
            continue
        for analysis in selected:
            md_path = setting.out_dir / f"{analysis.name}.md"
            lines += [f"### {question_for(analysis.name)}", ""]
            if not md_path.exists():
                error_path = setting.out_dir / f"{analysis.name}.error.txt"
                note = (f" The run failed; its traceback is at `{error_path}`."
                        if error_path.exists() else "")
                lines += [
                    f"Missing: the analysis `{analysis.name}` produced no "
                    f"`{md_path}`.{note}", "",
                ]
                continue
            title, body = _split_title(md_path.read_text(), analysis.name)
            lines += [f"Analysis `{analysis.name}`, reported in `{md_path}` "
                      f"under the title \"{title}\".", ""]
            lines += body
            lines += [""]

    out_path = out_root / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    logger.info("[done] post_rebuttal -> %s", out_path)


def _split_title(text: str, fallback: str, shift: int = 2) -> tuple[str, list[str]]:
    """The report's own title, and its body with every heading pushed down.

    The title line is taken out of the body because the section it is placed
    under already names it, and a heading repeated twice in a row reads as two
    sections. Every remaining heading gains `shift` levels, so that the whole
    report sits inside the section that holds it. A report with no level-1
    heading keeps all of its lines and is filed under `fallback`, the analysis
    name.
    """
    lines = text.splitlines()
    title = fallback
    body: list[str] = []
    seen_title = False
    bump = "#" * int(shift)
    for line in lines:
        if not seen_title and line.startswith("# "):
            title = line[2:].strip()
            seen_title = True
            continue
        body.append(bump + line if line.startswith("#") else line)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return title, body


# --------------------------------------------------------------------------- #
# report: the inventory, and the two paper deliverables
# --------------------------------------------------------------------------- #
def _is_result_file(path: Path) -> bool:
    """Whether one file is a result a reader would open, rather than an input.

    A report, a figure, a LaTeX table and a json of numbers are results. Two
    kinds of json are not. One is the `config.json` written inside a trained
    checkpoint, recognised by the weight file beside it. The other is the
    sidecar of a co-activation panel, recognised by the `.npz` of the same name
    beside it. Both belong to artefacts the inventory says it does not list, so
    listing their small companion files would contradict it.
    """
    if not path.is_file() or path.suffix not in _INVENTORY_SUFFIXES:
        return False
    if path.name == REPORT_NAME:
        return False
    if path.suffix == ".json":
        if path.with_suffix(".npz").exists():
            return False
        if any((path.parent / w).exists() for w in _CHECKPOINT_WEIGHTS):
            return False
    return True


def _inventory_section(out_root: Path) -> list[str]:
    """A table of every readable file under the output root, with its size.

    Only reports, figures, numbers and LaTeX sources are listed. Checkpoints
    and co-activation panels are inputs to these numbers rather than results,
    and one panel is hundreds of megabytes, so neither appears, and neither do
    the small json files that sit inside a checkpoint or beside a panel.
    """
    rows: list[tuple[str, str, str]] = []
    if out_root.exists():
        for path in sorted(out_root.rglob("*")):
            if not _is_result_file(path):
                continue
            kilobytes = path.stat().st_size / 1024.0
            rows.append((
                f"`{path.relative_to(out_root).as_posix()}`",
                _INVENTORY_KINDS.get(path.suffix, path.suffix.lstrip(".")),
                f"{kilobytes:,.1f} kB",
            ))

    lines = [
        "## Files this run produced",
        "",
        (f"Every report, figure, number file and LaTeX table written under "
         f"`{out_root.as_posix()}`, {len(rows)} files in all. Each path in the "
         "table below is relative to that directory. The trained checkpoints "
         "and the co-activation panels are deliberately not listed: they are "
         "inputs to the numbers below rather than results, and one panel is "
         "hundreds of megabytes. The small json files that belong to those two, "
         "a checkpoint's `config.json` and a panel's sidecar, are left out for "
         "the same reason. This document itself is not listed either."),
        "",
    ]
    if not rows:
        lines += ["No file has been written yet.", ""]
        return lines
    lines += ["| File | What it holds | Size on disk |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += [""]
    return lines


def _figure_and_table_sections(cfg: Config) -> list[str]:
    """The Table 1 markdown and the Figure 2 per-band statistics, inlined."""
    lines: list[str] = []

    table1_path = Path(cfg.table1.output.root) / "table1.md"
    lines += ["## Table 1: reconstruction, retrieval and zero-shot accuracy per method",
              ""]
    if table1_path.exists():
        title, body = _split_title(table1_path.read_text(), "Table 1", shift=1)
        lines += [f"Reproduced from `{table1_path}`, written under the title "
                  f"\"{title}\".", ""]
        lines += body + [""]
    else:
        lines += [f"Missing: the Table 1 stage produced no `{table1_path}`.", ""]

    fig_root = Path(cfg.figure2.output.root)
    stats_path = fig_root / "figure2_bin_stats.md"
    pdf_path = fig_root / "multi_density.pdf"
    lines += ["## Figure 2: distance between two feature directions, per co-activation band",
              ""]
    if stats_path.exists():
        title, body = _split_title(stats_path.read_text(), "Figure 2", shift=1)
        drawn = (f" The figure itself is at `{pdf_path}`."
                 if pdf_path.exists() else "")
        lines += [f"Reproduced from `{stats_path}`, written under the title "
                  f"\"{title}\".{drawn}", ""]
        lines += body + [""]
    else:
        lines += [f"Missing: the Figure 2 stage produced no `{stats_path}`.", ""]
    return lines

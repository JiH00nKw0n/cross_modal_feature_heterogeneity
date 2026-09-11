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
  report    Gathers each analysis report into one file at
            `<root>/post_rebuttal_results.md`.
  all       The four above, in that order.

Every stage is idempotent. A checkpoint, a panel or an analysis json that
already exists is not rebuilt, and a panel on disk is reused only when its
sidecar says it was built under the rules being asked for.

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
from pathlib import Path

from src.alignment import build_panel, load_panel, panel_mismatch, save_panel
from src.rebuttal.common import Setting, settings_from_config
from src.rebuttal.registry import Analysis, analyses_for
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

    if stage in ("all", "rebuttal"):
        _rebuttal_stage(cfg)

    if stage in ("all", "report"):
        _report_stage(cfg)


# --------------------------------------------------------------------------- #
# rebuttal stage
# --------------------------------------------------------------------------- #
def _rebuttal_stage(cfg: Config) -> None:
    settings = settings_from_config(cfg)
    if not settings:
        logger.warning("[post_rebuttal] cfg.rebuttal.settings selects no setting")
        return
    device = cfg.figure2.training.device

    _train_second_coco_model(cfg, settings.get("coco_k8"))

    for tag, setting in settings.items():
        logger.info("[post_rebuttal] setting %s: %s", tag, setting.title())
        setting.out_dir.mkdir(parents=True, exist_ok=True)
        _write_setting_manifest(setting)
        _build_extra_panels(cfg, setting)
        _run_analyses(cfg, setting, device=device)


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
    payload = build_panel(
        model=model, cache_dir=setting.cache_dir, split=setting.split,
        batch_size=cfg.figure2.training.batch_size,
        device=cfg.figure2.training.device,
        max_samples=0, shuffle_seed=shuffle_seed, pairing=pairing,
        model_b=model_b, ckpt_a=ckpt_a, ckpt_b=ckpt_b,
    )
    save_panel(path, payload)
    logger.info("[post_rebuttal] saved %s", path)


def _run_analyses(cfg: Config, setting: Setting, *, device: str) -> None:
    """Run every registered analysis that applies to this setting, in order."""
    selected = analyses_for(setting.tag, list(cfg.rebuttal.analyses or ["all"]))
    if not selected:
        logger.info("[post_rebuttal] %s: no analysis registered", setting.tag)
        return
    knobs = {
        "tau": float(cfg.rebuttal.tau),
        "n_boot": int(cfg.rebuttal.n_boot),
        "null_seed": int(cfg.rebuttal.null_seed),
    }
    for analysis in selected:
        out_json = setting.out_dir / f"{analysis.name}.json"
        if out_json.exists():
            logger.info("[post_rebuttal][skip] %s exists", out_json)
            continue
        logger.info("[post_rebuttal] %s: running %s", setting.tag, analysis.name)
        try:
            module = importlib.import_module(analysis.module_path)
        except ImportError as exc:
            raise ImportError(
                f"analysis {analysis.name!r} is registered as "
                f"{analysis.module_path!r}, which does not import: {exc}"
            ) from exc
        fn = getattr(module, "run", None)
        if fn is None:
            raise AttributeError(
                f"{analysis.module_path} has no run(setting, *, out_dir, device, **knobs)"
            )
        fn(setting, out_dir=setting.out_dir, device=device, **knobs)


def _write_setting_manifest(setting: Setting) -> None:
    """Record which checkpoints and panels this setting's numbers came from."""
    path = setting.out_dir / "setting.json"
    path.write_text(json.dumps(setting.as_dict(), indent=2))


# --------------------------------------------------------------------------- #
# report stage
# --------------------------------------------------------------------------- #
def _report_stage(cfg: Config) -> None:
    """Concatenate every analysis report into one document.

    An analysis that did not run is named as missing rather than left out, so
    that an absent measurement cannot be mistaken for one that was chosen
    against.

    Each report keeps its own title, which becomes the section heading, and its
    remaining headings are pushed down two levels so that they sit under it.
    The combined document therefore has one level-1 heading, one level-2
    heading per setting, and one level-3 heading per analysis.
    """
    out_root = Path(cfg.output.root)
    settings = settings_from_config(cfg)
    lines = [
        "# Post-rebuttal measurements",
        "",
        ("Every number below was produced by the analyses under `src/rebuttal/`, "
         "reading the checkpoints and co-activation panels that the Figure 2 and "
         "Table 1 pipelines wrote. Each section is reproduced verbatim from that "
         "analysis's own report, which is also readable on its own at the path "
         "named under the section heading."),
        "",
    ]
    for tag, setting in settings.items():
        lines += [f"## Setting {tag}", "", setting.title(), ""]
        selected: list[Analysis] = analyses_for(tag, list(cfg.rebuttal.analyses or ["all"]))
        if not selected:
            lines += ["No analysis is registered for this setting.", ""]
            continue
        for analysis in selected:
            md_path = setting.out_dir / f"{analysis.name}.md"
            if not md_path.exists():
                lines += [
                    f"### {analysis.name}", "", f"`{md_path}`", "",
                    f"Missing: this analysis did not produce {md_path}.", "",
                ]
                continue
            title, body = _split_title(md_path.read_text(), analysis.name)
            lines += [f"### {title}", "", f"`{md_path}`", ""]
            lines += body
            lines += [""]

    out_path = out_root / "post_rebuttal_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    logger.info("[done] post_rebuttal -> %s", out_path)


def _split_title(text: str, fallback: str) -> tuple[str, list[str]]:
    """The report's own title, and its body with every heading pushed down two.

    The title line is taken out of the body because it becomes the section
    heading, which would otherwise appear twice in a row. A report with no
    level-1 heading keeps all of its lines and is filed under `fallback`, the
    analysis name.
    """
    lines = text.splitlines()
    title = fallback
    body: list[str] = []
    seen_title = False
    for line in lines:
        if not seen_title and line.startswith("# "):
            title = line[2:].strip()
            seen_title = True
            continue
        body.append("##" + line if line.startswith("#") else line)
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return title, body

"""Rebuttal analyses: the measurements the reviewers asked for.

Each analysis is one module `src/rebuttal/<name>.py` exposing

    run(setting, *, out_dir, device, **knobs) -> dict

which writes `<out_dir>/<name>.json` holding every number, `<out_dir>/<name>.md`
holding an English report that stands on its own, and, where the paper drew a
figure, `<out_dir>/<name>.pdf` and `<out_dir>/<name>.png`. A run is idempotent:
it returns the loaded json and writes nothing when the json already exists.

`src/rebuttal/common.py` holds everything shared, above all the `Setting`
dataclass that names the checkpoints and panels of one trained configuration.
`src/rebuttal/registry.py` lists which analyses run on which settings; the
pipeline reads that list and nothing else.
"""

from src.rebuttal.common import (
    SETTING_TAGS,
    SETTING_TITLES,
    Setting,
    bootstrap_ci,
    describe,
    ensure_coco_annotations,
    fmt,
    load_panel_or_raise,
    matched_distance,
    md_table,
    pct,
    settings_from_config,
    unit_decoder,
    write_json,
    write_md,
)

__all__ = [
    "Setting",
    "settings_from_config",
    "SETTING_TAGS",
    "SETTING_TITLES",
    "load_panel_or_raise",
    "unit_decoder",
    "matched_distance",
    "describe",
    "bootstrap_ci",
    "fmt",
    "pct",
    "md_table",
    "write_md",
    "write_json",
    "ensure_coco_annotations",
]

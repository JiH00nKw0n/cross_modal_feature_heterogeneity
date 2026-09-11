"""Figure 2's per-band table, for whichever setting is being analysed.

Figure 2 draws, for one model, the decoder cosine distance of EVERY ordered pair
of latents (i, j) with i an image latent and j a text latent, split into five
bins by the pair's co-activation correlation, and writes the per-bin counts and
statistics next to the figure. That table exists for the COCO setting because
the figure is drawn there; the CC3M setting has no figure and therefore had no
such table. This analysis produces it for either setting from the setting's own
image-to-text panel, so the two can be read side by side.

The three rules of the figure apply here unchanged, and they differ on purpose
from the rules the matched-pair analyses follow:

No filter. All (i, j) pairs take part. There is no alive mask, no Hungarian
matching and no correlation threshold. A latent that never fires has correlation
0 against every partner and therefore lands in the lowest band, which is what
makes that band the reference the others are read against.

Bands. Edges are 0, 0.2, 0.4, 0.6, 0.8, 1.0, with the top band closed on the
right so that a perfectly correlated pair is kept. Pairs with a negative
correlation fall below every band; their count is reported separately.

Exactness. The statistics are computed on the full band. The subsampling Figure
2 applies is only for the kernel density estimate it draws, and no estimate is
drawn here.

The band arithmetic is `src.plotting.multi_density.bin_statistics`, called
directly, so this table and the figure's own table can never disagree.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.plotting.multi_density import BIN_EDGES, all_pairs, bin_statistics
from src.rebuttal.common import (
    Setting,
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Stem of the two files this analysis writes.
NAME = "correlation_bands"


def _table(stats: dict[str, Any]) -> str:
    body = []
    for band in stats["bins"]:
        if not band["n"]:
            body.append([band["bin"], fmt(0), pct(0.0), "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        d = band["cosine_distance"]
        body.append([
            band["bin"],
            fmt(band["n"]),
            pct(band["share_of_pairs"]),
            fmt(d["mean"], 4),
            fmt(d["median"], 4),
            fmt(d["std"], 4),
            fmt(d["min"], 4),
            fmt(d["max"], 4),
        ])
    a = stats["all_pairs"]["cosine_distance"]
    body.append([
        "all pairs, every correlation",
        fmt(stats["n_pairs"]),
        pct(1.0),
        fmt(a["mean"], 4),
        fmt(a["median"], 4),
        fmt(a["std"], 4),
        fmt(a["min"], 4),
        fmt(a["max"], 4),
    ])
    return md_table(
        ["co-activation correlation of the latent pair",
         "latent pairs (count)",
         "share of all latent pairs (percent)",
         "decoder cosine distance, mean",
         "decoder cosine distance, median",
         "decoder cosine distance, standard deviation",
         "decoder cosine distance, minimum",
         "decoder cosine distance, maximum"],
        body,
    )


def _correlation_table(stats: dict[str, Any]) -> str:
    body = []
    for band in stats["bins"]:
        if not band["n"]:
            body.append([band["bin"], fmt(0), "n/a", "n/a"])
            continue
        c = band["correlation"]
        body.append([band["bin"], fmt(band["n"]), fmt(c["mean"], 4), fmt(c["median"], 4)])
    c = stats["all_pairs"]["correlation"]
    body.append(["all pairs, every correlation", fmt(stats["n_pairs"]),
                 fmt(c["mean"], 4), fmt(c["median"], 4)])
    return md_table(
        ["co-activation correlation of the latent pair",
         "latent pairs (count)",
         "co-activation correlation, mean",
         "co-activation correlation, median"],
        body,
    )


def _paragraphs(setting: Setting, payload: dict[str, Any]) -> list[str]:
    stats = payload["statistics"]
    lead = (
        f"Setting {setting.tag}: {setting.title()}. Every ordered pair of latents "
        f"(i, j), with i one of the {payload['n_latents_image']:,} image latents and j "
        f"one of the {payload['n_latents_text']:,} text latents, contributes one value: "
        "the cosine distance 1 minus the cosine between image latent i's decoder "
        "direction and text latent j's decoder direction, both normalized to unit "
        f"length. That is {stats['n_pairs']:,} pairs. Each pair is placed in a band by "
        "its co-activation correlation, accumulated over the "
        f"{payload['n_samples']:,} image-caption pairs of the {setting.split} split."
    )
    rules = (
        "No filter is applied, which is deliberate and is what separates this table "
        "from the matched-pair analyses: there is no alive mask, no Hungarian matching "
        "and no correlation threshold. A latent that never fires has a correlation of 0 "
        "against every partner, so it lands in the lowest band. Bands are closed on the "
        "left and open on the right except for the top band, which is closed on both "
        "sides. Pairs whose correlation is negative fall below the lowest band and "
        "appear in no row of the first table; there are "
        f"{stats['n_negative_correlation_excluded']:,} of them. Every statistic is "
        "computed on the full band."
    )
    return [lead, rules]


def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        **knobs: Any) -> dict[str, Any]:
    """Per-band statistics of the decoder cosine distance over all latent pairs.

    Reads the setting's image-to-text panel and model A's two decoders. Writes
    `correlation_bands.json` and `correlation_bands.md` into `out_dir` and
    returns the payload. Skips the work when the json is already there.

    `device` and the pipeline's knobs (`tau`, `n_boot`, `null_seed`) are
    accepted and unused: this analysis reads a finished panel and two decoder
    matrices, and draws no bootstrap.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return json.loads(out_json.read_text())

    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    C = np.asarray(panel["C"])
    W_img = unit_decoder(setting.ckpt_a, "image")
    W_txt = unit_decoder(setting.ckpt_a, "text")

    c_all, dist_all = all_pairs(C, W_img, W_txt)
    stats = bin_statistics(c_all, dist_all)
    logger.info("[%s] %d latent pairs in %d bands", NAME, stats["n_pairs"],
                len(stats["bins"]))

    payload = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "n_samples": int(panel["n_samples"]),
        "n_latents_image": int(C.shape[0]),
        "n_latents_text": int(C.shape[1]),
        "n_alive_image": int(np.asarray(panel["alive_image"], dtype=bool).sum()),
        "n_alive_text": int(np.asarray(panel["alive_text"], dtype=bool).sum()),
        "bin_edges": list(BIN_EDGES),
        "statistics": stats,
    }
    write_json(out_json, payload)
    write_md(out_dir / f"{NAME}.md",
             f"Decoder cosine distance by co-activation band, {setting.tag}",
             _paragraphs(setting, payload),
             [("Decoder cosine distance per correlation band", _table(stats)),
              ("Co-activation correlation within each band", _correlation_table(stats))])
    logger.info("[%s] wrote %s", NAME, out_dir / f"{NAME}.md")
    return payload


__all__ = ["run", "NAME"]

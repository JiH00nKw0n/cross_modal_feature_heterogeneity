"""How much of an image feature direction the span of its text partners covers.

One reading of the measured cross-modal distance is that it is an artifact of
feature splitting: the text dictionary may cut a single concept into several
coordinates, so that no single text direction can match the image direction even
though the several of them together cover it. If that were the whole story, the
image direction would lie inside the subspace spanned by all of the text
directions the image latent co-activates with.

The quantity measured here is the orthogonal distance from the image feature
direction to that subspace. Write the image decoder direction as `x`, the
subspace spanned by the group's text decoder directions as `S`, and the
orthogonal projection of `x` onto `S` as `P_S x`. Then

    d_perp(x, S) = || x - P_S x ||.

Decoder rows are unit norm, so `|| P_S x ||^2 + d_perp^2 = 1` holds exactly and
the distance and the explained energy share `e = || P_S x ||^2` are two readings
of one number, related by `d_perp = sqrt(1 - e)`.

An explained share is meaningless without a control, because a subspace of
dimension N covers roughly N/d of any direction by chance. Four control arms are
reported next to the measured one. The one that tests the splitting reading is
"strongest partner plus random text dictionary directions": it keeps the
strongest partner and replaces the remaining partners with directions drawn at
random from the text dictionary, so the comparison asks whether the additional
partners contribute more than arbitrary directions do. Two looser arms anchor
the scale, all N drawn at random from the text dictionary and all N drawn as
random unit vectors, the latter of which should land on the analytic value N/d.

Ported from `scripts/real_alpha/analyze_1toN_span.py` of the paper repository.
The measurement is unchanged; the data loading is not ported, because this
repository reads groups from the one co-activation panel.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.rebuttal.common import (
    Setting,
    bootstrap_ci,
    describe,
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: File stem; the json, the md and the report section all carry this name.
NAME = "one_to_many_span"

#: Exactly how a text latent qualifies as a partner. Recorded in the json so a
#: reader never has to guess whether the correlation was taken in absolute
#: value. It was not: a negative correlation does not make a partner.
THRESHOLD_RULE = "signed correlation C[i,j] >= tau (not |C|)"

#: The five subspaces each image direction is projected onto, in report order.
#: Key is the json field; value is the phrase used in the markdown table.
ARM_LABELS = {
    "all_partners": "span of all text partners of the group",
    "strongest_partner_only": "the single most correlated text partner",
    "strongest_partner_plus_random_atoms":
        "the most correlated partner plus random text dictionary directions",
    "random_text_atoms": "random text dictionary directions",
    "random_unit_directions": "random unit directions",
}


# --------------------------------------------------------------------------- #
# The projection
# --------------------------------------------------------------------------- #
def explained_fraction(phi: np.ndarray, Psi: np.ndarray) -> float:
    """Share of a unit direction's energy captured by the span of Psi's rows.

    `phi` is one unit-norm direction of length d, `Psi` is (N, d) with unit-norm
    rows. Returns `|| P_S phi ||^2` in [0, 1], where S is the span of those rows.

    A QR factorization rather than a normal-equation pseudo-inverse: partner
    directions can be close to linearly dependent, and QR stays well behaved
    there.
    """
    q, _r = np.linalg.qr(np.asarray(Psi, dtype=np.float64).T)  # (d, N) orthonormal
    return float(np.clip(np.sum((q.T @ np.asarray(phi, dtype=np.float64)) ** 2), 0.0, 1.0))


def find_groups(
    C: np.ndarray, alive_image: np.ndarray, alive_text: np.ndarray, tau: float,
) -> list[tuple[int, np.ndarray]]:
    """Every alive image latent with two or more alive text partners above tau.

    A partner is a text latent whose signed co-activation correlation with the
    image latent is at least `tau`; the correlation is never taken in absolute
    value. Partners come back ordered from the most correlated down, so that
    `partners[0]` is the one a one-to-one match would keep. Groups come back in
    increasing image latent index.
    """
    text_idx = np.where(alive_text)[0]
    groups: list[tuple[int, np.ndarray]] = []
    for i in np.where(alive_image)[0]:
        partners = text_idx[C[i, text_idx] >= tau]
        if partners.size >= 2:
            groups.append((int(i), partners[np.argsort(-C[i, partners])]))
    return groups


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(
    setting: Setting,
    *,
    out_dir: str | Path,
    device: str = "cpu",
    tau: float = 0.4,
    n_boot: int = 1000,
    n_draws: int = 20,
    seed: int = 0,
    **knobs: Any,
) -> dict[str, Any]:
    """Measure the orthogonal distance to the text partner span, and report it.

    Writes `<out_dir>/one_to_many_span.json` and `.md`, and returns the payload.
    Skips both when the json already exists.

    tau       Correlation at or above which a text latent counts as a partner.
    n_boot    Bootstrap resamples over groups, for the 95 percent intervals.
    n_draws   Random draws averaged per group in each of the three control arms.
    seed      Seed of the draws and of the bootstrap.
    device    Unused; the whole measurement is a projection of stored decoder
              rows and runs on the CPU. Accepted so that the pipeline can pass
              one signature to every analysis.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        import json

        return json.loads(out_json.read_text())

    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    C = np.asarray(panel["C"], dtype=np.float32)
    alive_image = np.asarray(panel["alive_image"], dtype=bool)
    alive_text = np.asarray(panel["alive_text"], dtype=bool)
    n_samples = int(panel["n_samples"])

    Wi = unit_decoder(setting.ckpt_a, "image")
    Wt = unit_decoder(setting.ckpt_a, "text")
    dim = int(Wi.shape[1])
    text_idx = np.where(alive_text)[0]
    n_alive_image = int(alive_image.sum())

    groups = find_groups(C, alive_image, alive_text, float(tau))
    logger.info("[%s] %s: %d groups over %d alive image latents at tau=%.2f",
                NAME, setting.tag, len(groups), n_alive_image, tau)

    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "ckpt": str(setting.ckpt_a),
        "n_samples": n_samples,
        "tau": float(tau),
        "threshold_rule": THRESHOLD_RULE,
        "alive_rule": "fire_count >= 1 on the full train split",
        "dim": dim,
        "n_alive_image": n_alive_image,
        "n_alive_text": int(alive_text.sum()),
        "n_groups": len(groups),
        "n_boot": int(n_boot),
        "n_draws": int(n_draws),
        "seed": int(seed),
    }

    if not groups:
        payload["note"] = (
            f"no image latent has two or more text partners at tau={tau}"
        )
        write_json(out_json, payload)
        _write_report(out_dir, setting, payload)
        return payload

    arms = _measure_arms(groups, Wi, Wt, text_idx, dim, n_draws=int(n_draws), seed=int(seed))
    sizes = np.array([p.size for _i, p in groups], dtype=np.int64)

    explained = {k: _with_ci(v, n_boot=int(n_boot), seed=int(seed)) for k, v in arms.items()}
    distance = {
        k: _with_ci(np.sqrt(np.clip(1.0 - v, 0.0, None)), n_boot=int(n_boot), seed=int(seed))
        for k, v in arms.items()
    }

    full = arms["all_partners"]
    top1 = arms["strongest_partner_only"]
    control = arms["strongest_partner_plus_random_atoms"]

    payload.update({
        "group_share_of_alive_image": float(len(groups) / max(n_alive_image, 1)),
        "group_size_histogram": {str(k): int(v)
                                 for k, v in sorted(Counter(sizes.tolist()).items())},
        "explained": explained,
        "orthogonal_distance": distance,
        "by_group_size": _by_group_size(sizes, full),
        "analytic_random_subspace": float(sizes.mean() / dim),
        "marginal_gain_over_strongest": float(np.median(full - top1)),
        "marginal_gain_of_control": float(np.median(control - top1)),
        "unexplained_median": float(1.0 - np.median(full)),
        "frac_groups_explained_above_half": float(np.mean(full > 0.5)),
        "per_group": {
            "image_latent": [int(i) for i, _p in groups],
            "n_partners": sizes.tolist(),
            **{k: v.tolist() for k, v in arms.items()},
        },
    })

    write_json(out_json, payload)
    _write_report(out_dir, setting, payload)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _measure_arms(
    groups: list[tuple[int, np.ndarray]],
    Wi: np.ndarray,
    Wt: np.ndarray,
    text_idx: np.ndarray,
    dim: int,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Explained energy share per group, one array per arm, in group order."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ARM_LABELS}
    for i, partners in groups:
        phi = Wi[i]
        n = int(partners.size)
        out["all_partners"].append(explained_fraction(phi, Wt[partners]))
        out["strongest_partner_only"].append(explained_fraction(phi, Wt[partners[:1]]))

        # The pool a control draws from: alive text latents that are not already
        # partners of this group, so a control can never re-draw a partner.
        pool = np.setdiff1d(text_idx, partners, assume_unique=False)
        keep, atoms, dirs = [], [], []
        for _ in range(n_draws):
            filler = rng.choice(pool, size=n - 1, replace=False)
            keep.append(explained_fraction(phi, np.vstack([Wt[partners[:1]], Wt[filler]])))
            atoms.append(explained_fraction(phi, Wt[rng.choice(pool, size=n, replace=False)]))
            v = rng.standard_normal((n, dim))
            dirs.append(explained_fraction(phi, v / np.linalg.norm(v, axis=1, keepdims=True)))
        out["strongest_partner_plus_random_atoms"].append(float(np.mean(keep)))
        out["random_text_atoms"].append(float(np.mean(atoms)))
        out["random_unit_directions"].append(float(np.mean(dirs)))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _with_ci(values: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Percentile summary plus bootstrap intervals for the median and the mean."""
    s = describe(values)
    med, med_lo, med_hi = bootstrap_ci(values, np.median, n_boot=n_boot, seed=seed)
    mean, mean_lo, mean_hi = bootstrap_ci(values, np.mean, n_boot=n_boot, seed=seed + 1)
    s["median"] = med
    s["mean"] = mean
    s["median_ci95"] = [med_lo, med_hi]
    s["mean_ci95"] = [mean_lo, mean_hi]
    return s


def _by_group_size(sizes: np.ndarray, full: np.ndarray) -> list[dict[str, Any]]:
    """One row per number of text partners, using the all-partners subspace."""
    rows: list[dict[str, Any]] = []
    for n in sorted(set(int(v) for v in sizes.tolist())):
        sel = sizes == n
        e = full[sel]
        rows.append({
            "n_partners": int(n),
            "n_groups": int(sel.sum()),
            "explained_median": float(np.median(e)),
            "distance_median": float(np.median(np.sqrt(np.clip(1.0 - e, 0.0, None)))),
        })
    return rows


def _ci(pair: Any) -> str:
    """A 95 percent interval as one cell, "low to high"."""
    if not pair or len(pair) != 2:
        return "n/a"
    lo, hi = pair
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "n/a"
    return f"{lo:.3f} to {hi:.3f}"


def _write_report(out_dir: Path, setting: Setting, payload: dict[str, Any]) -> None:
    """Write the markdown report next to the json."""
    out_md = Path(out_dir) / f"{NAME}.md"
    title = "Orthogonal distance from an image feature direction to the span of its text partners"

    if payload["n_groups"] == 0:
        write_md(out_md, title, [
            setting.title() + ".",
            (f"No image latent has two or more text partners at a signed co-activation "
             f"correlation of {fmt(payload['tau'], 2)} or more, over the "
             f"{fmt(payload['n_alive_image'])} image latents that fire at least once on the "
             f"{fmt(payload['n_samples'])} pairs of the {setting.split} split. There is "
             f"therefore no group to project, and no table follows. The partner rule is "
             f"{payload['threshold_rule']}."),
        ])
        return

    ex, dd = payload["explained"], payload["orthogonal_distance"]
    rows = []
    for key, label in ARM_LABELS.items():
        a, b = dd[key], ex[key]
        rows.append([
            label,
            fmt(a["n"]),
            fmt(a["median"]),
            _ci(a["median_ci95"]),
            fmt(a["mean"]),
            _ci(a["mean_ci95"]),
            fmt(b["median"]),
        ])
    arm_table = md_table(
        ["Subspace the image direction is projected onto",
         "Groups measured (count)",
         "Median orthogonal distance (0 to 1)",
         "Median, 95 percent interval",
         "Mean orthogonal distance (0 to 1)",
         "Mean, 95 percent interval",
         "Median explained energy share (0 to 1)"],
        rows,
    )

    size_table = md_table(
        ["Text partners in the group (count)",
         "Groups of this size (count)",
         "Median orthogonal distance using all partners (0 to 1)",
         "Median explained energy share using all partners (0 to 1)"],
        [[fmt(r["n_partners"]), fmt(r["n_groups"]),
          fmt(r["distance_median"]), fmt(r["explained_median"])]
         for r in payload["by_group_size"]],
    )

    headline = md_table(
        ["Quantity", "Value"],
        [
            ["Image latents that fire at least once (count)", fmt(payload["n_alive_image"])],
            ["Text latents that fire at least once (count)", fmt(payload["n_alive_text"])],
            ["Groups with two or more text partners (count)", fmt(payload["n_groups"])],
            ["Share of alive image latents that form a group (percent)",
             pct(payload["group_share_of_alive_image"])],
            ["Median explained share added by the partners beyond the strongest one (0 to 1)",
             fmt(payload["marginal_gain_over_strongest"])],
            ["The same median for the control that replaces those partners with random "
             "text dictionary directions (0 to 1)",
             fmt(payload["marginal_gain_of_control"])],
            ["Median share of the image direction's energy left outside the span of all "
             "partners (percent)", pct(payload["unexplained_median"])],
            ["Groups whose all-partner span explains more than half the energy (percent)",
             pct(payload["frac_groups_explained_above_half"])],
            ["Chance explained share of a subspace of the average group's size, "
             "mean partners divided by embedding dimension (0 to 1)",
             fmt(payload["analytic_random_subspace"], 4)],
        ],
    )

    paragraphs = [
        setting.title() + ".",
        (f"For every image latent that fires at least once, the text latents whose signed "
         f"co-activation correlation with it is {fmt(payload['tau'], 2)} or more are its "
         f"partners; the rule is {payload['threshold_rule']}. An image latent with two or "
         f"more partners forms a group, and {fmt(payload['n_groups'])} of the "
         f"{fmt(payload['n_alive_image'])} alive image latents do "
         f"({pct(payload['group_share_of_alive_image'])}). The correlations come from the "
         f"co-activation panel built on all {fmt(payload['n_samples'])} pairs of the "
         f"{setting.split} split. Each group contributes one number per arm: the orthogonal "
         f"distance from its image decoder direction to the span of a set of "
         f"{fmt(payload['dim'])}-dimensional unit directions, "
         f"d = sqrt(max(1 - e, 0)) where e is the share of the image direction's energy that "
         f"the span explains. The three random arms average "
         f"{fmt(payload['n_draws'])} draws per group before the group enters the summary. "
         f"Intervals are 95 percent percentile bootstrap intervals over "
         f"{fmt(payload['n_boot'])} resamples of the groups."),
    ]

    write_md(out_md, title, paragraphs, tables=[
        ("Distance to each subspace", arm_table),
        ("By the number of text partners in the group", size_table),
        ("Counts and summary quantities", headline),
    ])


__all__ = ["run", "explained_fraction", "find_groups", "NAME", "THRESHOLD_RULE", "ARM_LABELS"]

"""Is the one-to-many structure feature splitting, or several distinct concepts?

The post-hoc alignment forces a one-to-one match, so a concept that the text
dictionary holds in several coordinates loses all but one of them. Two questions
follow, and both are answered here on one checkpoint so that they rest on the
same basis.

How much one-to-many structure there is. For every image latent that fires at
least once, count the text latents whose signed co-activation correlation with
it clears a threshold. Two or more of them is a one-to-many group. The threshold
has no principled value, so the whole sweep is reported rather than one point.

What the several partners are. Partners that stand for distinct concepts fire on
different inputs. Partners that are one concept cut across several coordinates
fire on largely the same inputs. Two measurements separate those. The first is
the Jaccard overlap between two partners' firing sets, that is the number of
pairs on which both fire divided by the number on which either fires, with pairs
of unrelated alive text latents measured the same way as the scale it is read
against. The second is how much of a group's co-firing the single most
correlated partner already covers, reported together with how much of the image
latent's own firing that co-firing set accounts for, because the first number
alone has a denominator that is itself a fraction of the whole.

The firing sets are taken over every pair of the training split, the same rows
the co-activation panel was built on. The latents are never materialized as an
(N, L) matrix: one streaming pass accumulates the co-firing counts that every
statistic above is a ratio of.

Ported from `scripts/real_alpha/analyze_1toN_splitting.py` of the paper
repository. The measurement is unchanged; the data loading is not ported,
because this repository streams the unified cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.data.cache_io import load_stacked, split_rows
from src.data.paired_dataset import normalize_np
from src.eval.eval_utils import load_sae
from src.rebuttal.common import (
    Setting,
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    write_json,
    write_md,
)
from src.rebuttal.one_to_many_span import THRESHOLD_RULE, find_groups

logger = logging.getLogger(__name__)

#: File stem; the json, the md and the report section all carry this name.
NAME = "one_to_many_splitting"

#: Correlation thresholds the group count is swept over, unless a caller
#: overrides them.
DEFAULT_TAU_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5)

#: Pairs of text latents whose co-firing is accumulated per matrix-free chunk.
_PAIR_CHUNK = 1024

#: Batches between progress lines while streaming.
_LOG_EVERY = 64


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(
    setting: Setting,
    *,
    out_dir: str | Path,
    device: str = "cpu",
    tau: float = 0.4,
    n_random_pairs: int = 5000,
    tau_sweep: Sequence[float] = DEFAULT_TAU_SWEEP,
    batch_size: int = 8192,
    seed: int = 0,
    **knobs: Any,
) -> dict[str, Any]:
    """Measure how much one-to-many structure there is, and what the partners are.

    Writes `<out_dir>/one_to_many_splitting.json` and `.md`, and returns the
    payload. Skips both when the json already exists.

    tau             Correlation at or above which a text latent counts as a
                    partner, for the detailed part of the analysis.
    tau_sweep       Thresholds the group count is reported at.
    n_random_pairs  Pairs of alive text latents drawn as the Jaccard scale.
    batch_size      Rows encoded per step of the streaming pass.
    seed            Seed of the random pair draw.
    device          Where the two encoders run. Falls back to the CPU when CUDA
                    was asked for and is not available.
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
    n_alive_image = int(alive_image.sum())
    panel_samples = int(panel["n_samples"])

    sweep = [_sweep_row(C, alive_image, alive_text, float(t), n_alive_image)
             for t in tau_sweep]
    for row in sweep:
        logger.info("[%s] tau=%.2f -> %d groups (%.1f%% of alive image latents)",
                    NAME, row["tau"], row["n_groups"],
                    100.0 * row["share_of_alive_image"])

    groups = find_groups(C, alive_image, alive_text, float(tau))
    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "ckpt": str(setting.ckpt_a),
        "dataset": setting.dataset,
        "n_samples": panel_samples,
        "alive_rule": "fire_count >= 1 on the full train split",
        "threshold_rule": THRESHOLD_RULE,
        "n_alive_image": n_alive_image,
        "n_alive_text": int(alive_text.sum()),
        "tau_sweep": sweep,
        "tau": float(tau),
        "n_groups": len(groups),
        "n_random_pairs": int(n_random_pairs),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }

    if not groups:
        payload["note"] = f"no image latent has two or more text partners at tau={tau}"
        write_json(out_json, payload)
        _write_report(out_dir, setting, payload)
        return payload

    # The pairs whose co-firing the streaming pass has to count: every pair of
    # partners inside a group, then the random pairs that give Jaccard a scale.
    within_a, within_b, within_group = _within_group_pairs(groups)
    rng = np.random.default_rng(int(seed))
    rand_a, rand_b = _random_pairs(np.where(alive_text)[0], int(n_random_pairs), rng)
    pair_a = np.concatenate([within_a, rand_a]) if rand_a.size else within_a
    pair_b = np.concatenate([within_b, rand_b]) if rand_b.size else within_b

    counts = _stream_cofiring(
        setting=setting,
        group_image=np.array([i for i, _p in groups], dtype=np.int64),
        group_partners=[p for _i, p in groups],
        pair_a=pair_a, pair_b=pair_b,
        batch_size=int(batch_size), device=device,
    )
    if counts["n"] != panel_samples:
        raise ValueError(
            f"streamed {counts['n']} rows of split {setting.split!r} but the panel at "
            f"{setting.panel_path('img_txt')} was built on {panel_samples}. The firing "
            "sets and the correlations would not describe the same rows. Rebuild the "
            "panel on the full split."
        )

    fire_text = counts["fire_text"]
    inter = counts["pair_inter"]
    union = fire_text[pair_a] + fire_text[pair_b] - inter
    jaccard = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64),
                        where=union > 0)
    keep = union > 0

    n_within = within_a.size
    jac_random = jaccard[n_within:][keep[n_within:]]

    # Per group: the median Jaccard among its own partner pairs, the share of
    # the group's co-firing the most correlated partner covers, and how large
    # that co-firing set is next to the image latent's own firing.
    co_any = counts["co_any"]
    co_top = counts["co_top"]
    fire_image = counts["fire_image_group"]
    rows: list[dict[str, Any]] = []
    cover: list[float] = []
    cofire_share: list[float] = []
    # A group whose image latent never fires together with any of its partners
    # has no coverage to report, and its partner pairs are left out of the
    # Jaccard pool as well, so that both tables describe the same groups.
    survivor = np.zeros(len(groups), dtype=bool)
    for g, (i, partners) in enumerate(groups):
        if fire_image[g] == 0 or co_any[g] == 0:
            continue
        survivor[g] = True
        sel = (within_group == g) & keep[:n_within]
        js = jaccard[:n_within][sel]
        rec = {
            "image_latent": int(i),
            "n_partners": int(partners.size),
            "jaccard_median": float(np.median(js)) if js.size else None,
            "strongest_share_of_cofiring": float(co_top[g] / co_any[g]),
            "cofiring_share_of_image_firing": float(co_any[g] / fire_image[g]),
        }
        rows.append(rec)
        cover.append(rec["strongest_share_of_cofiring"])
        cofire_share.append(rec["cofiring_share_of_image_firing"])

    jac_within = jaccard[:n_within][keep[:n_within] & survivor[within_group]]

    sizes = np.array([p.size for _i, p in groups], dtype=np.int64)
    payload.update({
        "group_size_histogram": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(sizes, return_counts=True))},
        "n_groups_measured": len(rows),
        "jaccard_within_group": _summarize(jac_within),
        "jaccard_random_pairs": _summarize(jac_random),
        "strongest_share_of_cofiring": _summarize(cover),
        "cofiring_share_of_image_firing": _summarize(cofire_share),
        "fire_count_agreement": counts["fire_count_agreement"],
        "per_group": rows,
    })

    write_json(out_json, payload)
    _write_report(out_dir, setting, payload)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


# --------------------------------------------------------------------------- #
# groups and pairs
# --------------------------------------------------------------------------- #
def _sweep_row(C, alive_image, alive_text, tau: float, n_alive_image: int) -> dict[str, Any]:
    """Group count, share and size at one correlation threshold."""
    g = find_groups(C, alive_image, alive_text, tau)
    sizes = np.array([p.size for _i, p in g], dtype=np.int64)
    return {
        "tau": float(tau),
        "n_groups": len(g),
        "share_of_alive_image": float(len(g) / max(n_alive_image, 1)),
        "mean_group_size": float(sizes.mean()) if sizes.size else 0.0,
        "max_group_size": int(sizes.max()) if sizes.size else 0,
    }


def _within_group_pairs(
    groups: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every unordered pair of partners inside every group.

    Returns the two latent index arrays and, for each pair, the position of the
    group it came from, so that a per-group median can be taken later.
    """
    a: list[int] = []
    b: list[int] = []
    owner: list[int] = []
    for g, (_i, partners) in enumerate(groups):
        p = partners.tolist()
        for x in range(len(p)):
            for y in range(x + 1, len(p)):
                a.append(int(p[x]))
                b.append(int(p[y]))
                owner.append(g)
    return (np.array(a, dtype=np.int64), np.array(b, dtype=np.int64),
            np.array(owner, dtype=np.int64))


def _random_pairs(
    candidates: np.ndarray, n_pairs: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """`n_pairs` pairs of two different alive text latents, drawn with repetition.

    The two members of one pair are always different latents; two draws may
    repeat a pair, which is what the paper's loop did and which leaves the
    distribution of Jaccard values unchanged.
    """
    if candidates.size < 2 or n_pairs <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    a = rng.choice(candidates, size=n_pairs, replace=True)
    b = rng.choice(candidates, size=n_pairs, replace=True)
    clash = a == b
    # Re-draw the collisions rather than dropping them, so the count is exact.
    for _ in range(16):
        if not clash.any():
            break
        b[clash] = rng.choice(candidates, size=int(clash.sum()), replace=True)
        clash = a == b
    ok = a != b
    return a[ok].astype(np.int64), b[ok].astype(np.int64)


# --------------------------------------------------------------------------- #
# the streaming pass
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _stream_cofiring(
    *,
    setting: Setting,
    group_image: np.ndarray,
    group_partners: list[np.ndarray],
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """One pass over the split, accumulating every count the ratios need.

    Nothing of size (rows, latents) is ever held. Per batch the two sides are
    encoded, the post-TopK latents are turned into a zero-or-one firing
    indicator, and four sets of counts are added up:

      fire_text            pairs on which each text latent fires
      fire_image_group     pairs on which each group's image latent fires
      pair_inter           pairs on which both members of a listed text pair fire
      co_any               pairs on which a group's image latent fires together
                           with at least one of its text partners
      co_top               the same, counting only the most correlated partner

    A union over a group's partners cannot be recovered from pairwise counts,
    which is why `co_any` is accumulated here rather than derived afterwards.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = load_sae(setting.ckpt_a, "separated").to(dev).eval()
    image_sae, text_sae = model.image_sae, model.text_sae
    L_img = int(image_sae.latent_size)
    L_txt = int(text_sae.latent_size)

    cache = load_stacked(setting.cache_dir, mmap=True)
    rows = split_rows(cache, setting.split)
    n = int(rows.shape[0])
    n_groups = int(group_image.shape[0])

    # Membership matrices: column g holds the partners of group g. One matrix
    # multiply then gives, for every row of the batch, whether any partner of g
    # fired, and another gives whether its most correlated partner fired.
    member = torch.zeros(L_txt, n_groups, device=dev)
    top = torch.zeros(L_txt, n_groups, device=dev)
    for g, partners in enumerate(group_partners):
        member[torch.as_tensor(np.asarray(partners, dtype=np.int64), device=dev), g] = 1.0
        top[int(partners[0]), g] = 1.0
    gi_idx = torch.as_tensor(group_image, device=dev)
    a_idx = torch.as_tensor(pair_a, device=dev)
    b_idx = torch.as_tensor(pair_b, device=dev)

    fire_text = np.zeros(L_txt, dtype=np.float64)
    fire_image = np.zeros(L_img, dtype=np.float64)
    fire_image_group = np.zeros(n_groups, dtype=np.float64)
    pair_inter = np.zeros(pair_a.shape[0], dtype=np.float64)
    co_any = np.zeros(n_groups, dtype=np.float64)
    co_top = np.zeros(n_groups, dtype=np.float64)

    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        idx = rows[s:e]
        xb = torch.from_numpy(normalize_np(cache["image"][idx])).to(dev)
        yb = torch.from_numpy(normalize_np(cache["text"][idx])).to(dev)
        zi = _fired(image_sae, xb)
        zt = _fired(text_sae, yb)

        fire_text += zt.sum(0).double().cpu().numpy()
        fire_image += zi.sum(0).double().cpu().numpy()

        gi = zi.index_select(1, gi_idx)                     # (B, n_groups)
        any_partner = (zt @ member > 0).float()
        top_partner = (zt @ top > 0).float()
        fire_image_group += gi.sum(0).double().cpu().numpy()
        co_any += (gi * any_partner).sum(0).double().cpu().numpy()
        co_top += (gi * top_partner).sum(0).double().cpu().numpy()

        for cs in range(0, a_idx.shape[0], _PAIR_CHUNK):
            ce = min(cs + _PAIR_CHUNK, a_idx.shape[0])
            both = zt.index_select(1, a_idx[cs:ce]) * zt.index_select(1, b_idx[cs:ce])
            pair_inter[cs:ce] += both.sum(0).double().cpu().numpy()

        if (b + 1) % _LOG_EVERY == 0:
            logger.info("[%s] %d/%d rows", NAME, e, n)

    return {
        "n": n,
        "fire_text": fire_text,
        "fire_image": fire_image,
        "fire_image_group": fire_image_group,
        "pair_inter": pair_inter,
        "co_any": co_any,
        "co_top": co_top,
        "fire_count_agreement": _fire_agreement(setting, fire_image, fire_text),
    }


@torch.no_grad()
def _fired(sae, x: torch.Tensor) -> torch.Tensor:
    """Zero-or-one firing indicator per latent, shape (B, L)."""
    out = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
    return (out.dense_latents.squeeze(1).float() != 0).float()


def _fire_agreement(setting: Setting, fire_image: np.ndarray,
                    fire_text: np.ndarray) -> dict[str, Any]:
    """How far the streamed firing counts sit from the panel's own counts.

    Both are counts of the same event over the same rows, so a non-zero
    difference means the two passes did not see the same rows and every ratio
    computed from them would be mixing two bases. Reported rather than asserted,
    so that the number is on record either way.
    """
    try:
        panel = load_panel_or_raise(setting.panel_path("img_txt"))
    except FileNotFoundError:
        return {"checked": False}
    d_img = np.abs(np.asarray(panel["fire_count_image"], dtype=np.float64) - fire_image)
    d_txt = np.abs(np.asarray(panel["fire_count_text"], dtype=np.float64) - fire_text)
    out = {
        "checked": True,
        "max_abs_difference_image_latent_fire_count": float(d_img.max()) if d_img.size else 0.0,
        "max_abs_difference_text_latent_fire_count": float(d_txt.max()) if d_txt.size else 0.0,
    }
    if out["max_abs_difference_image_latent_fire_count"] > 0 or \
            out["max_abs_difference_text_latent_fire_count"] > 0:
        logger.warning("[%s] streamed firing counts differ from the panel's by up to "
                       "%.0f (image) and %.0f (text)", NAME,
                       out["max_abs_difference_image_latent_fire_count"],
                       out["max_abs_difference_text_latent_fire_count"])
    return out


# --------------------------------------------------------------------------- #
# summaries and report
# --------------------------------------------------------------------------- #
def _summarize(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Median, mean, the 5th and 95th percentile, and the share below 0.1."""
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "share_below_0.1": float(np.mean(a < 0.1)),
    }


def _write_report(out_dir: Path, setting: Setting, payload: dict[str, Any]) -> None:
    out_md = Path(out_dir) / f"{NAME}.md"
    title = "One-to-many groups, and whether their text partners fire on the same inputs"

    sweep_table = md_table(
        ["Correlation threshold",
         "Image latents with two or more text partners (count)",
         "Share of alive image latents (percent)",
         "Mean text partners per group (count)",
         "Most text partners in one group (count)"],
        [[fmt(r["tau"], 2), fmt(r["n_groups"]), pct(r["share_of_alive_image"]),
          fmt(r["mean_group_size"], 2), fmt(r["max_group_size"])]
         for r in payload["tau_sweep"]],
    )

    if payload["n_groups"] == 0:
        write_md(out_md, title, [
            setting.title() + ".",
            (f"For every image latent that fires at least once, the text latents whose "
             f"signed co-activation correlation with it clears a threshold are its "
             f"partners; the rule is {payload['threshold_rule']}. The correlations come "
             f"from the co-activation panel built on all {fmt(payload['n_samples'])} pairs "
             f"of the {setting.split} split, over {fmt(payload['n_alive_image'])} alive "
             f"image latents and {fmt(payload['n_alive_text'])} alive text latents. At the "
             f"threshold of {fmt(payload['tau'], 2)} chosen for the detailed part, no image "
             f"latent has two or more partners, so only the sweep below is reported and no "
             f"firing sets were streamed."),
        ], tables=[("How many one-to-many groups there are, by threshold", sweep_table)])
        return

    jw = payload["jaccard_within_group"]
    jr = payload["jaccard_random_pairs"]
    cs = payload["strongest_share_of_cofiring"]
    cf = payload["cofiring_share_of_image_firing"]

    jaccard_table = md_table(
        ["Which two text latents are compared",
         "Median Jaccard overlap of the firing sets (0 to 1)",
         "Mean (0 to 1)", "5th percentile (0 to 1)", "95th percentile (0 to 1)",
         "Pairs overlapping by less than 0.1 (percent)",
         "Pairs measured (count)"],
        [
            ["two partners of the same image latent",
             fmt(jw.get("median")), fmt(jw.get("mean")), fmt(jw.get("p05")),
             fmt(jw.get("p95")), pct(jw.get("share_below_0.1")), fmt(jw.get("n", 0))],
            ["two alive text latents drawn at random",
             fmt(jr.get("median")), fmt(jr.get("mean")), fmt(jr.get("p05")),
             fmt(jr.get("p95")), pct(jr.get("share_below_0.1")), fmt(jr.get("n", 0))],
        ],
    )

    cover_table = md_table(
        ["Quantity, one value per group",
         "Median (percent)", "Mean (percent)",
         "5th percentile (percent)", "95th percentile (percent)",
         "Groups measured (count)"],
        [
            ["pairs covered by the most correlated partner alone, out of the pairs where "
             "the image latent fires together with any partner",
             pct(cs.get("median")), pct(cs.get("mean")), pct(cs.get("p05")),
             pct(cs.get("p95")), fmt(cs.get("n", 0))],
            ["pairs where the image latent fires together with any partner, out of all "
             "pairs where that image latent fires",
             pct(cf.get("median")), pct(cf.get("mean")), pct(cf.get("p05")),
             pct(cf.get("p95")), fmt(cf.get("n", 0))],
        ],
    )

    size_table = md_table(
        ["Text partners in the group (count)", "Groups of this size (count)"],
        [[fmt(int(k)), fmt(v)] for k, v in
         sorted(payload["group_size_histogram"].items(), key=lambda kv: int(kv[0]))],
    )

    paragraphs = [
        setting.title() + ".",
        (f"For every image latent that fires at least once, the text latents whose signed "
         f"co-activation correlation with it is at least the threshold are its partners; "
         f"the rule is {payload['threshold_rule']}. An image latent with two or more "
         f"partners forms a one-to-many group. The correlations come from the co-activation "
         f"panel built on all {fmt(payload['n_samples'])} pairs of the {setting.split} "
         f"split, over {fmt(payload['n_alive_image'])} alive image latents and "
         f"{fmt(payload['n_alive_text'])} alive text latents. The threshold used in the "
         f"detailed tables is {fmt(payload['tau'], 2)}. The group count there is "
         f"{fmt(payload['n_groups'])}, and the number of those groups whose image latent "
         f"fires together with at least one of its partners, which is what the coverage "
         f"table measures, is {fmt(payload['n_groups_measured'])}. The firing "
         f"sets were taken by streaming those same {fmt(payload['n_samples'])} pairs through "
         f"the image and text encoders of the checkpoint at {payload['ckpt']}, counting for "
         f"each pair of text latents the rows on which both fire. The random arm draws "
         f"{fmt(payload['n_random_pairs'])} pairs of two different alive text latents with "
         f"seed {fmt(payload['seed'])}."),
    ]

    write_md(out_md, title, paragraphs, tables=[
        ("How many one-to-many groups there are, by threshold", sweep_table),
        (f"Sizes of the groups at a threshold of {fmt(payload['tau'], 2)}", size_table),
        ("Do the partners fire on the same inputs", jaccard_table),
        ("What keeping only the most correlated partner covers", cover_table),
    ])


__all__ = ["run", "NAME", "DEFAULT_TAU_SWEEP"]

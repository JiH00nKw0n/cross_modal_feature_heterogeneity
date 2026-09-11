"""How close could the image and text directions possibly be brought together?

The paper reports a cosine distance between an image latent's decoder direction
and the direction of the text latent it was matched to, and that distance is
large. Two objections stand between that number and the paper's conclusion, and
both say the distance is an artifact of how the two dictionaries are compared
rather than a property of the dictionaries themselves. This module measures both
objections directly.

The first objection is that the matching is bad. Perhaps some assignment other
than the Hungarian one would pair each image direction with a text direction it
actually agrees with. The test here throws the matching away entirely and asks
for the single closest text direction anywhere in the text dictionary, which no
assignment procedure can beat, because every assignment picks one partner per
row and this takes the best partner available to that row with no one-to-one
constraint at all. The same search is then run against random unit directions,
which gives the cosine such a maximum would report by chance from taking a
maximum over thousands of candidates, and against the closed-form value
sqrt(2 ln m / d) for that chance maximum, where m is the number of candidates
and d the embedding dimension.

The second objection is that the two dictionaries agree up to one global
transform, in which case a per-latent distance would be a coordinate artifact.
The test fits a single transform on half of the matched pairs and scores it on
the other half, as a rotation (orthogonal Procrustes) and as an unconstrained
ridge-regularized linear map. A fourth arm fits a rotation on a random
re-pairing of the same vectors, which shows what the held-out score looks like
when the transform has nothing real to learn, so a reader can see that the
evaluation is not simply rewarding extra free parameters.

Ported from `scripts/real_alpha/analyze_alignment_ceiling.py` of the paper
repository. The statistics are unchanged; what changed is where the inputs come
from. Alive masks, the correlation matrix and the assignment are read from the
co-activation panel rather than recomputed, so this analysis cannot disagree
with any other about which latents are alive or which latent is matched to
which.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.rebuttal.common import (
    bootstrap_ci,
    describe,
    fmt,
    load_panel_or_raise,
    matched_distance,
    md_table,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Stem of the files this analysis writes.
NAME = "alignment_ceiling"

#: Rows of the oracle search handled per chunk, so the full
#: (alive image) x (alive text) cosine matrix is never held at once.
_ORACLE_CHUNK = 512


# --------------------------------------------------------------------------- #
# The two transforms
# --------------------------------------------------------------------------- #
def orthogonal_map(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The rotation that carries the rows of X onto the rows of Y most closely.

    This is the orthogonal Procrustes solution: the matrix R maximizing
    trace(R^T X^T Y) over all orthogonal R, obtained from the singular value
    decomposition of X^T Y as R = U V^T. Applied as `X @ R`.
    """
    u, _s, vt = np.linalg.svd(np.asarray(X, dtype=np.float64).T @ np.asarray(Y, dtype=np.float64))
    return u @ vt


def linear_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """The unconstrained least-squares map from X to Y, lightly regularized.

    Solves `(X^T X + ridge * I) A = X^T Y`, which is ordinary least squares with
    a small ridge term so that the system stays solvable when the fitted half of
    the pairs does not span the embedding space. Applied as `X @ A`.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ Y)


def applied_cosine(X: np.ndarray, Y: np.ndarray, M: np.ndarray | None) -> np.ndarray:
    """Cosine between each transformed row of X and the matching row of Y.

    `M` is None for the identity arm. Y is expected to hold unit rows already,
    which `unit_decoder` guarantees; the transformed X rows are renormalized
    here because a transform changes their length.
    """
    Z = np.asarray(X, dtype=np.float64) if M is None else np.asarray(X, dtype=np.float64) @ M
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return (Z * np.asarray(Y, dtype=np.float64)).sum(axis=1)


def random_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """`n` unit-norm directions drawn uniformly on the sphere in `d` dimensions."""
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _rowwise_max_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each row of A, the largest cosine against any row of B.

    Both inputs hold unit rows, so a dot product is a cosine. Rows of A are
    processed in chunks, which keeps the peak memory at
    `_ORACLE_CHUNK x len(B)` instead of `len(A) x len(B)`.
    """
    out = np.empty(A.shape[0], dtype=np.float64)
    for s in range(0, A.shape[0], _ORACLE_CHUNK):
        block = A[s:s + _ORACLE_CHUNK] @ B.T
        out[s:s + block.shape[0]] = block.max(axis=1)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(setting, *, out_dir: str | Path, device: str = "cpu",
        seed: int = 0, n_boot: int = 1000, ridge: float = 1e-3,
        **knobs: Any) -> dict[str, Any]:
    """Measure the two ceilings for one setting and write the report.

    Everything here is linear algebra over the decoder matrices, so `device` is
    accepted for a uniform call signature and not used: no model is run.

    seed    Seed of the random directions used for the chance oracle and of the
            split into a fitted half and a held-out half. Default 0, the value
            the paper's script used.
    n_boot  Bootstrap resamples behind each reported interval. Default 1000.
    ridge   Ridge term of the unconstrained linear map. Default 1e-3.

    Returns the payload it wrote to `<out_dir>/alignment_ceiling.json`, and
    skips the work when that file is already there.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return dict(json.loads(out_json.read_text()))

    rng = np.random.default_rng(int(seed))
    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    alive_i = np.asarray(panel["alive_image"], dtype=bool)
    alive_t = np.asarray(panel["alive_text"], dtype=bool)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    usable = np.asarray(panel["usable"], dtype=bool)

    Wi = unit_decoder(setting.ckpt_a, "image")
    Wt = unit_decoder(setting.ckpt_a, "text")
    dim = int(Wi.shape[1])
    logger.info("[%s] alive image %d, alive text %d, embedding dimension %d",
                NAME, int(alive_i.sum()), int(alive_t.sum()), dim)

    # ---- what the paper's matching attains ---------------------------------
    d_matched = matched_distance(Wi, Wt, perm, usable)
    matched_point, matched_lo, matched_hi = bootstrap_ci(
        d_matched, stat=np.median, n_boot=int(n_boot), seed=int(seed))

    # ---- ceiling 1: the best partner available, matching ignored -----------
    rows = np.where(alive_i)[0]
    cols = np.where(alive_t)[0]
    oracle_cos = _rowwise_max_cosine(Wi[rows], Wt[cols])
    oracle_point, oracle_lo, oracle_hi = bootstrap_ci(
        1.0 - oracle_cos, stat=np.median, n_boot=int(n_boot), seed=int(seed))

    n_candidates = int(cols.size)
    oracle_cos_null = _rowwise_max_cosine(Wi[rows], random_unit(n_candidates, dim, rng))
    analytic_null = float(np.sqrt(2.0 * np.log(max(n_candidates, 2)) / dim))

    # ---- ceiling 2: one global transform on held-out pairs ------------------
    use = np.where(usable)[0]
    X = Wi[use]
    Y = Wt[perm[use]]
    order = rng.permutation(len(use))
    half = len(use) // 2
    fit_idx, eval_idx = order[:half], order[half:]
    shuffled = rng.permutation(len(use))

    R = orthogonal_map(X[fit_idx], Y[fit_idx])
    A = linear_map(X[fit_idx], Y[fit_idx], ridge=float(ridge))
    R_null = orthogonal_map(X[fit_idx], Y[shuffled[fit_idx]])

    arms = {
        "no_transform": applied_cosine(X[eval_idx], Y[eval_idx], None),
        "best_rotation": applied_cosine(X[eval_idx], Y[eval_idx], R),
        "best_linear_map": applied_cosine(X[eval_idx], Y[eval_idx], A),
        "rotation_fitted_on_shuffled_pairs": applied_cosine(X[eval_idx], Y[eval_idx], R_null),
    }
    transform: dict[str, Any] = {
        "n_pairs": int(len(use)),
        "n_fit": int(len(fit_idx)),
        "n_eval": int(len(eval_idx)),
        "ridge": float(ridge),
    }
    for arm, values in arms.items():
        point, lo, hi = bootstrap_ci(values, stat=np.mean, n_boot=int(n_boot), seed=int(seed))
        transform[arm] = {"mean_cosine": float(np.mean(values)),
                          "ci_low": lo, "ci_high": hi}

    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "ckpt": str(setting.ckpt_a),
        "embedding_dim": dim,
        "n_alive_image": int(alive_i.sum()),
        "n_alive_text": int(alive_t.sum()),
        "n_usable_pairs": int(usable.sum()),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "matched_distance": describe(d_matched),
        "matched_distance_median_ci": [matched_point, matched_lo, matched_hi],
        "oracle_cosine": describe(oracle_cos),
        "oracle_distance": describe(1.0 - oracle_cos),
        "oracle_distance_median_ci": [oracle_point, oracle_lo, oracle_hi],
        "oracle_cosine_against_random_directions": describe(oracle_cos_null),
        "oracle_chance_analytic": analytic_null,
        "n_oracle_candidates": n_candidates,
        "global_transform": transform,
    }
    write_json(out_json, payload)
    _write_report(out_dir / f"{NAME}.md", setting, payload)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
_ARM_LABELS = (
    ("no_transform", "no transform, the raw pair of directions"),
    ("best_rotation", "best rotation (orthogonal Procrustes)"),
    ("best_linear_map", "best unconstrained linear map (ridge least squares)"),
    ("rotation_fitted_on_shuffled_pairs",
     "rotation fitted on randomly re-paired directions"),
)


def _write_report(path: Path, setting, payload: dict[str, Any]) -> None:
    m = payload["matched_distance"]
    oc = payload["oracle_cosine"]
    od = payload["oracle_distance"]
    on = payload["oracle_cosine_against_random_directions"]
    t = payload["global_transform"]

    intro = (
        f"The setting is {setting.title()}. Measured here are two ceilings on how "
        f"close the image and text decoder directions could be brought. The "
        f"dictionaries hold {payload['n_alive_image']:,} image latents and "
        f"{payload['n_alive_text']:,} text latents that fired at least once on the "
        f"{setting.split} split, in an embedding space of "
        f"{payload['embedding_dim']} dimensions, and the assignment pairs "
        f"{payload['n_usable_pairs']:,} image latents with a partner that is alive "
        f"as well. The first ceiling drops the matching and gives every alive image "
        f"direction the single closest text direction anywhere in the text "
        f"dictionary, a choice no assignment procedure can improve on, so it is "
        f"measured over all {payload['n_alive_image']:,} of them while the assigned "
        f"partner is measured over the {payload['n_usable_pairs']:,} matched pairs. "
        f"The second ceiling fits one global transform on {t['n_fit']:,} of those "
        f"matched pairs and scores it on the {t['n_eval']:,} pairs held out from the "
        f"fit. Intervals are 95 percent percentile bootstrap intervals over "
        f"{payload['n_boot']:,} resamples."
    )

    matched_tbl = md_table(
        ["quantity", "latents measured", "median", "mean",
         "5th percentile", "95th percentile"],
        [
            ["cosine distance between an image latent and its assigned text partner",
             fmt(m["n"]), fmt(m["median"]), fmt(m["mean"]), fmt(m["p05"]), fmt(m["p95"])],
            ["cosine distance from an image latent to the closest text direction anywhere",
             fmt(od["n"]), fmt(od["median"]), fmt(od["mean"]), fmt(od["p05"]), fmt(od["p95"])],
            ["cosine from an image latent to the closest text direction anywhere",
             fmt(oc["n"]), fmt(oc["median"]), fmt(oc["mean"]), fmt(oc["p05"]), fmt(oc["p95"])],
            [f"cosine from an image latent to the closest of "
             f"{payload['n_oracle_candidates']:,} random directions",
             fmt(on["n"]), fmt(on["median"]), fmt(on["mean"]), fmt(on["p05"]), fmt(on["p95"])],
        ],
    )

    mp, mlo, mhi = payload["matched_distance_median_ci"]
    op, olo, ohi = payload["oracle_distance_median_ci"]
    ci_tbl = md_table(
        ["quantity", "median", "95 percent interval, low", "95 percent interval, high"],
        [
            ["cosine distance between an image latent and its assigned text partner",
             fmt(mp), fmt(mlo), fmt(mhi)],
            ["cosine distance from an image latent to the closest text direction anywhere",
             fmt(op), fmt(olo), fmt(ohi)],
        ],
    )

    chance = (
        f"The closed-form cosine for the largest of "
        f"{payload['n_oracle_candidates']:,} random directions in "
        f"{payload['embedding_dim']} dimensions, sqrt(2 ln m / d), is "
        f"{fmt(payload['oracle_chance_analytic'])}. The measured value over the "
        f"same number of drawn random directions is "
        f"{fmt(on['median'])} at the median."
    )

    transform_tbl = md_table(
        ["transform fitted on half the matched pairs",
         "mean cosine on the held-out half",
         "95 percent interval, low", "95 percent interval, high"],
        [[label, fmt(t[key]["mean_cosine"]), fmt(t[key]["ci_low"]), fmt(t[key]["ci_high"])]
         for key, label in _ARM_LABELS],
    )

    provenance = (
        f"Decoder directions come from `{payload['ckpt']}`, unit-normalized row "
        f"by row. The alive masks and the assignment come from "
        f"`{payload['panel']}`; a latent counts as alive when it fired at least "
        f"once over the split the panel was built on, and no firing-rate "
        f"threshold is applied anywhere. The random directions and the split "
        f"into a fitted and a held-out half use seed {payload['seed']}. The "
        f"unconstrained linear map carries a ridge term of "
        f"{fmt(t['ridge'], 5)} on the normal equations."
    )

    write_md(
        path,
        "Ceilings on the cosine distance: a better matching, and one global transform",
        [intro, chance, provenance],
        [
            ("The assigned partner, the closest partner available, and chance",
             matched_tbl),
            ("The first two medians with bootstrap intervals", ci_tbl),
            ("One global transform, fitted on half the pairs and scored on the other half",
             transform_tbl),
        ],
    )


__all__ = ["run", "NAME", "orthogonal_map", "linear_map", "applied_cosine", "random_unit"]

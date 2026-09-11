"""Swap only the operator that links the two latent spaces, and retrieve.

The paper's method links a modality-specific image dictionary to its text
dictionary with a permutation: one image latent is assigned one text latent by
Hungarian matching on the co-activation correlation. That is one choice of
operator among several, and a reviewer asked whether a different one would do
better. This module puts six families of operator in that one slot, holds the
sparse autoencoders, the training data and the evaluation fixed, and scores each
one by cross-modal retrieval on the COCO test split.

The operators:

  Hungarian on co-activation      the paper's method. A one-to-one assignment
                                  maximizing the total correlation.
  greedy one-to-one               repeatedly take the highest remaining
                                  correlation whose row and column are both
                                  still free. A deliberately weaker matcher,
                                  which says whether the global optimality of
                                  the Hungarian assignment is doing any work.
  Hungarian on decoder cosine     the same assignment, with the cost matrix
                                  taken from the cosine between decoder
                                  directions instead of from co-activation.
  Sinkhorn entropic transport     the one-to-one constraint relaxed into a soft
                                  transport plan, so one image latent may draw
                                  on several text latents. The entropy weight
                                  epsilon controls how far the mass spreads;
                                  as epsilon falls the plan approaches the
                                  permutation.
  Procrustes rotation             the orthogonal matrix that best carries the
                                  text latent space onto the image one. Since a
                                  permutation is itself orthogonal, this is the
                                  same objective with the one-to-one constraint
                                  dropped.
  canonical correlation analysis  a pair of projections into a shared subspace
                                  of a chosen width. Unlike the others it does
                                  not preserve the identity of a coordinate at
                                  all.

Fairness. Procrustes and canonical correlation analysis carry free parameters
and would win trivially if fitted and scored on the same data, so every operator
is fitted on the training split and scored on the held-out COCO test split. The
fit reads the training split only through first and second moments of the two
latent streams, accumulated in one streaming pass, so the latent matrix is never
held and every operator sees the same information.

Ported from `scripts/real_alpha/eval_alignment_methods.py` of the paper
repository. The operators, their default settings and the retrieval protocol are
unchanged. What changed is where the shared inputs come from: the alive masks,
the correlation matrix and the Hungarian permutation are read from the
co-activation panel, which was built over the full training split, rather than
recomputed here. The streaming pass remains, because Procrustes and canonical
correlation analysis need the full within-modality covariance matrices, which
the panel does not store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.data.cache_io import load_stacked, split_rows
from src.data.paired_dataset import normalize_np
from src.eval import eval_utils
from src.rebuttal.common import (
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    unit_decoder,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Stem of the files this analysis writes.
NAME = "alignment_methods"

#: Cost given to a dead row or column, so it cannot take an alive latent's
#: partner. The same value the panel builder uses.
BIG_NEG = -1e9

#: Recall cut-offs reported in both directions.
RECALL_KS = (1, 5, 10)

#: Readable name of each operator, keyed by the result key it is stored under.
#: An operator whose key carries a numeric setting, such as the entropy weight
#: of a transport plan or the width of a canonical subspace, is labelled where it
#: is scored, since its label has to name that number.
METHOD_LABELS = {
    "hungarian_coactivation": "Hungarian on co-activation (the paper's method)",
    "greedy_coactivation": "greedy one-to-one on co-activation",
    "hungarian_decoder_cosine": "Hungarian with decoder cosine as the cost",
    "procrustes_rotation": "Procrustes rotation",
}


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def hungarian_perm(C: np.ndarray, alive_a: np.ndarray, alive_b: np.ndarray) -> np.ndarray:
    """One signed, alive-restricted Hungarian assignment on a cost matrix.

    The same rule the co-activation panel applies, repeated here because this
    module also matches on a second cost matrix, the decoder cosine, which the
    panel does not carry. Dead rows and dead columns are pushed to `BIG_NEG` so
    they cannot win an assignment, and the sign of the cost is kept.
    """
    Cm = np.array(C, dtype=np.float64, copy=True)
    Cm[~alive_a, :] = BIG_NEG
    Cm[:, ~alive_b] = BIG_NEG
    Cm = np.nan_to_num(Cm, nan=BIG_NEG, posinf=1.0, neginf=BIG_NEG)
    row, col = linear_sum_assignment(-Cm)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    perm[row] = col
    return perm


def greedy_perm(C: np.ndarray, alive_a: np.ndarray, alive_b: np.ndarray) -> np.ndarray:
    """Take the highest remaining correlation whose row and column are free.

    A weaker matcher than the Hungarian assignment on purpose: it maximizes each
    pair locally instead of maximizing the total, so comparing the two says
    whether global optimality changes the outcome.
    """
    ri, rt = np.where(alive_a)[0], np.where(alive_b)[0]
    sub = np.nan_to_num(np.asarray(C, dtype=np.float64)[np.ix_(ri, rt)], nan=BIG_NEG)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    order = np.dstack(np.unravel_index(np.argsort(-sub, axis=None), sub.shape))[0]
    used_r: set[int] = set()
    used_c: set[int] = set()
    limit = min(len(ri), len(rt))
    for a, b in order:
        a, b = int(a), int(b)
        if a in used_r or b in used_c:
            continue
        used_r.add(a)
        used_c.add(b)
        perm[ri[a]] = rt[b]
        if len(used_r) == limit:
            break
    return perm


def sinkhorn_plan(C: np.ndarray, alive_a: np.ndarray, alive_b: np.ndarray,
                  eps: float, iters: int, device: torch.device) -> np.ndarray:
    """Entropic optimal transport with uniform marginals over the alive block.

    Returns a full (L_image, L_text) plan whose dead rows and columns are zero.
    `eps` is the entropy weight: a small value concentrates the mass of each row
    on one column and approaches the permutation, a large one spreads it.
    """
    ri, rt = np.where(alive_a)[0], np.where(alive_b)[0]
    K = torch.as_tensor(np.nan_to_num(np.asarray(C, dtype=np.float32)[np.ix_(ri, rt)], nan=0.0),
                        dtype=torch.float32, device=device) / float(eps)
    n, m = K.shape
    f = torch.zeros(n, device=device)
    g = torch.zeros(m, device=device)
    log_a = -float(np.log(n))
    log_b = -float(np.log(m))
    for _ in range(int(iters)):
        f = log_a - torch.logsumexp(K + g[None, :], dim=1)
        g = log_b - torch.logsumexp(K + f[:, None], dim=0)
    T_sub = torch.exp(K + f[:, None] + g[None, :]).cpu().numpy()
    T = np.zeros(C.shape, dtype=np.float32)
    T[np.ix_(ri, rt)] = T_sub
    return T


def procrustes_map(Sit: np.ndarray) -> np.ndarray:
    """Orthogonal matrix carrying the text latent space onto the image one.

    Maximizes trace(R^T Sit^T) over orthogonal R, which is the objective the
    permutation solves with the one-to-one constraint replaced by
    orthogonality. `Sit` is the centered cross-covariance between image latents
    and text latents. Applied as `z_text @ R`.
    """
    u, _s, vt = np.linalg.svd(np.asarray(Sit, dtype=np.float64), full_matrices=False)
    return (u @ vt).T


def cca_maps(Sii: np.ndarray, Stt: np.ndarray, Sit: np.ndarray,
             dim: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Canonical directions of the two latent spaces, `dim` of them per side.

    Returns (A, B), the projections applied as `z_image @ A` and `z_text @ B`.
    `ridge` is relative: each within-modality covariance gets `ridge` times its
    own mean diagonal added, which keeps the Cholesky factorization defined when
    some latents are close to constant.
    """
    Sii = np.asarray(Sii, dtype=np.float64)
    Stt = np.asarray(Stt, dtype=np.float64)
    Sit = np.asarray(Sit, dtype=np.float64)
    ti = ridge * np.trace(Sii) / Sii.shape[0]
    tt = ridge * np.trace(Stt) / Stt.shape[0]
    Li = np.linalg.cholesky(Sii + ti * np.eye(Sii.shape[0]))
    Lt = np.linalg.cholesky(Stt + tt * np.eye(Stt.shape[0]))
    M = np.linalg.solve(Li, Sit)
    M = np.linalg.solve(Lt, M.T).T
    u, _s, vt = np.linalg.svd(M, full_matrices=False)
    A = np.linalg.solve(Li.T, u[:, :dim])
    B = np.linalg.solve(Lt.T, vt[:dim].T)
    return A, B


def decoder_cosine_matrix(ckpt: str | Path) -> np.ndarray:
    """Cosine between every image decoder direction and every text one."""
    return unit_decoder(ckpt, "image") @ unit_decoder(ckpt, "text").T


# --------------------------------------------------------------------------- #
# The one streaming pass over the fitting split
# --------------------------------------------------------------------------- #
def _resolve_device(device: str) -> torch.device:
    """The device to compute on, falling back to the processor when needed."""
    if device == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _subsample(rows: np.ndarray, max_samples: int) -> np.ndarray:
    """Evenly spaced subsample of the split's rows, or every row when 0."""
    n = int(rows.shape[0])
    if max_samples <= 0 or max_samples >= n:
        return rows
    return rows[np.linspace(0, n - 1, int(max_samples), dtype=np.int64)]


@torch.no_grad()
def second_moments(model, cache: dict[str, Any], rows: np.ndarray, *,
                   device: torch.device, batch_size: int = 8192) -> dict[str, np.ndarray]:
    """First and second moments of the two latent streams over `rows`.

    Every operator fitted here is a function of those moments, so the pass
    accumulates them and never holds the latents. For a per-side width of 4096
    the three second-moment matrices come to about 400 MB in float64, whatever
    the number of pairs.

    Returns sum_image, sum_text, image-image, text-text and image-text second
    moments, and the count of pairs, all uncentered.
    """
    L_i = int(model.image_sae.latent_size)
    L_t = int(model.text_sae.latent_size)
    model = model.to(device).eval()

    sum_i = np.zeros(L_i, dtype=np.float64)
    sum_t = np.zeros(L_t, dtype=np.float64)
    ii = np.zeros((L_i, L_i), dtype=np.float64)
    tt = np.zeros((L_t, L_t), dtype=np.float64)
    it = np.zeros((L_i, L_t), dtype=np.float64)

    image, text = cache["image"], cache["text"]
    n = int(rows.shape[0])
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        block = rows[s:e]
        xb = torch.from_numpy(normalize_np(image[block])).to(device)
        yb = torch.from_numpy(normalize_np(text[block])).to(device)
        zi = model.image_sae(hidden_states=xb.unsqueeze(1),
                             return_dense_latents=True).dense_latents.squeeze(1).float()
        zt = model.text_sae(hidden_states=yb.unsqueeze(1),
                            return_dense_latents=True).dense_latents.squeeze(1).float()
        sum_i += zi.sum(0).cpu().numpy().astype(np.float64)
        sum_t += zt.sum(0).cpu().numpy().astype(np.float64)
        ii += (zi.T @ zi).cpu().numpy().astype(np.float64)
        tt += (zt.T @ zt).cpu().numpy().astype(np.float64)
        it += (zi.T @ zt).cpu().numpy().astype(np.float64)
        if (b + 1) % 20 == 0:
            logger.info("[%s] moments %d/%d rows", NAME, e, n)
    return {"sum_i": sum_i, "sum_t": sum_t, "ii": ii, "tt": tt, "it": it,
            "n": np.float64(n)}


def centered_covariances(acc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered covariances (image-image, text-text, image-text) from the moments."""
    n = float(acc["n"])
    mi = acc["sum_i"] / n
    mt = acc["sum_t"] / n
    Sii = acc["ii"] / n - np.outer(mi, mi)
    Stt = acc["tt"] / n - np.outer(mt, mt)
    Sit = acc["it"] / n - np.outer(mi, mt)
    return Sii, Stt, Sit


# --------------------------------------------------------------------------- #
# Retrieval, the protocol of src/eval/retrieval.py
# --------------------------------------------------------------------------- #
def coco_test_rows(keys: Sequence[str]) -> tuple[np.ndarray, torch.Tensor, list[list[int]]]:
    """Unique images of a COCO split, and which captions belong to each.

    Returns the image position of every caption, the cache rows holding one copy
    of each image, and the caption positions of each image. The image row is
    duplicated across a photo's captions in the cache, so the first row of each
    image id is taken once.
    """
    image_ids = [eval_utils.coco_image_id(k) for k in keys]
    first_row: dict[str, int] = {}
    order: list[str] = []
    for row, iid in enumerate(image_ids):
        if iid not in first_row:
            first_row[iid] = row
            order.append(iid)
    id_to_idx = {iid: i for i, iid in enumerate(order)}
    pair_img_idx = np.array([id_to_idx[iid] for iid in image_ids], dtype=np.int64)
    img_rows = torch.as_tensor([first_row[iid] for iid in order], dtype=torch.long)
    gt_caps: list[list[int]] = [[] for _ in order]
    for cap_pos, img_idx in enumerate(pair_img_idx):
        gt_caps[int(img_idx)].append(cap_pos)
    return pair_img_idx, img_rows, gt_caps


def recalls(z_img: torch.Tensor, z_txt: torch.Tensor,
            pair_img_idx: np.ndarray, gt_caps: list[list[int]],
            chunk: int = 1024) -> dict[str, float]:
    """Recall at 1, 5 and 10 in both directions, with pessimistic ranking.

    The protocol of `src.eval.retrieval`, repeated here because that function
    accepts only a permutation on the text side and several operators in this
    module are not permutations. Text to image ranks every unique image for each
    caption; image to text takes the best rank the image achieves over its own
    captions. A rank counts every candidate scoring greater than or equal to the
    ground truth, minus one, so a tie against the ground truth pushes it down
    rather than up.
    """
    zi = eval_utils.normalize_rows(z_img)
    zt = eval_utils.normalize_rows(z_txt)

    t2i = np.empty(zt.shape[0], dtype=np.int64)
    t2i_tie = np.empty(zt.shape[0], dtype=np.int64)
    for s in range(0, zt.shape[0], chunk):
        scores = zt[s:s + chunk] @ zi.T
        gt = pair_img_idx[s:s + chunk]
        gt_scores = scores[np.arange(len(gt)), gt]
        t2i[s:s + chunk] = ((scores >= gt_scores[:, None]).sum(dim=1) - 1).cpu().numpy()
        t2i_tie[s:s + chunk] = (scores == gt_scores[:, None]).sum(dim=1).cpu().numpy()

    i2t = np.empty(zi.shape[0], dtype=np.int64)
    i2t_tie = np.empty(zi.shape[0], dtype=np.int64)
    for s in range(0, zi.shape[0], chunk):
        scores = zi[s:s + chunk] @ zt.T
        for row in range(scores.shape[0]):
            idx = s + row
            best = scores[row, gt_caps[idx]].max()
            i2t[idx] = int((scores[row] >= best).sum().item()) - 1
            i2t_tie[idx] = int((scores[row] == best).sum().item())

    out: dict[str, float] = {}
    for k in RECALL_KS:
        out[f"I2T R@{k}"] = float((i2t < k).mean())
        out[f"T2I R@{k}"] = float((t2i < k).mean())
    out["I2T tie at ground truth rate"] = float((i2t_tie > 1).mean())
    out["T2I tie at ground truth rate"] = float((t2i_tie > 1).mean())
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(setting, *, out_dir: str | Path, device: str = "cpu",
        max_fit_samples: int = 300_000,
        sinkhorn_eps: Iterable[float] = (0.01, 0.05, 0.1),
        sinkhorn_iters: int = 300,
        cca_dims: Iterable[int] = (256, 1024),
        conf_cutoffs: Iterable[float] = (0.1, 0.2),
        ridge: float = 1e-3,
        batch_size: int = 8192,
        eval_split: str = "test",
        **knobs: Any) -> dict[str, Any]:
    """Score every alignment operator on COCO retrieval and write the report.

    max_fit_samples  Pairs of the training split fed through the streaming pass
                     that produces the covariances Procrustes and canonical
                     correlation analysis are fitted on. Evenly spaced over the
                     split. 0 means every pair. Default 300,000, the number the
                     paper's script used.
    sinkhorn_eps     Entropy weights of the transport plan. Default
                     (0.01, 0.05, 0.1).
    sinkhorn_iters   Sinkhorn iterations. Default 300.
    cca_dims         Widths of the shared subspace. Default (256, 1024).
    conf_cutoffs     Also score the Hungarian permutation restricted to matches
                     whose correlation reaches this value, since the other
                     operators effectively drop weak coordinates. Default
                     (0.1, 0.2).
    ridge            Relative ridge on the within-modality covariances, for
                     canonical correlation analysis. Default 1e-3.
    batch_size       Pairs per batch in the streaming pass. Default 8192.
    eval_split       Split of the COCO cache used for retrieval. Default "test".

    Returns the payload it wrote to `<out_dir>/alignment_methods.json`, and
    skips the work when that file is already there.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return dict(json.loads(out_json.read_text()))

    dev = _resolve_device(device)
    model = eval_utils.load_sae(setting.ckpt_a, "separated")

    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    C = np.asarray(panel["C"], dtype=np.float64)
    alive_i = np.asarray(panel["alive_image"], dtype=bool)
    alive_t = np.asarray(panel["alive_text"], dtype=bool)
    perm = np.asarray(panel["perm"], dtype=np.int64)
    usable = np.asarray(panel["usable"], dtype=bool)
    logger.info("[%s] alive image %d, alive text %d, usable matches %d",
                NAME, int(alive_i.sum()), int(alive_t.sum()), int(usable.sum()))

    # ---- the one streaming pass, for the two operators that need covariances -
    cache = load_stacked(setting.cache_dir, mmap=True)
    fit_rows = _subsample(split_rows(cache, setting.split), int(max_fit_samples))
    logger.info("[%s] fitting the covariances on %d pairs of split %r",
                NAME, int(fit_rows.shape[0]), setting.split)
    acc = second_moments(model, cache, fit_rows, device=dev, batch_size=int(batch_size))
    Sii, Stt, Sit = centered_covariances(acc)

    # ---- the held-out COCO split -------------------------------------------
    ds = eval_utils.load_paired_split(setting.coco_cache, eval_split)
    pair_img_idx, img_rows, gt_caps = coco_test_rows(ds.keys)
    z_img = eval_utils.encode_image(model, ds.image[img_rows], "separated", dev,
                                    batch_size=2048)
    z_txt = eval_utils.encode_text(model, ds.text, "separated", dev, batch_size=2048)
    logger.info("[%s] held-out COCO %s: %d images, %d captions",
                NAME, eval_split, z_img.shape[0], z_txt.shape[0])

    results: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}

    def score(key: str, label: str, zi: torch.Tensor, zt: torch.Tensor) -> None:
        results[key] = recalls(zi, zt, pair_img_idx, gt_caps)
        labels[key] = label
        logger.info("[%s] %-46s I2T R@1 %.4f  T2I R@1 %.4f",
                    NAME, label, results[key]["I2T R@1"], results[key]["T2I R@1"])

    def reindexed(p: np.ndarray) -> torch.Tensor:
        return z_txt[:, torch.as_tensor(np.asarray(p), dtype=torch.long)]

    # 1. the paper's permutation, straight from the panel
    z_txt_perm = reindexed(perm)
    score("hungarian_coactivation", METHOD_LABELS["hungarian_coactivation"],
          z_img, z_txt_perm)

    # 2. the same permutation restricted to its stronger matches. The other
    #    operators shrink the working set on their own: canonical correlation
    #    analysis keeps a few hundred directions and Sinkhorn damps weak
    #    coordinates, while the permutation keeps every coordinate including the
    #    ones whose match is near noise. This arm gives it the same freedom.
    matched_c = np.where(usable, C[np.arange(len(perm)), perm], -np.inf)
    for cmin in conf_cutoffs:
        keep = torch.as_tensor(matched_c >= float(cmin), dtype=torch.bool)
        n_keep = int(keep.sum())
        if n_keep < 8:
            logger.warning("[%s] correlation cut-off %.2f keeps only %d coordinates, skipped",
                           NAME, float(cmin), n_keep)
            continue
        score(f"hungarian_coactivation_c{cmin}",
              f"Hungarian on co-activation, only matches with correlation at least "
              f"{fmt(cmin, 2)} ({n_keep:,} coordinates)",
              z_img[:, keep], z_txt_perm[:, keep])

    # 3. the weaker matcher, and the second cost matrix
    score("greedy_coactivation", METHOD_LABELS["greedy_coactivation"],
          z_img, reindexed(greedy_perm(C, alive_i, alive_t)))
    dec_perm = hungarian_perm(decoder_cosine_matrix(setting.ckpt_a), alive_i, alive_t)
    score("hungarian_decoder_cosine", METHOD_LABELS["hungarian_decoder_cosine"],
          z_img, reindexed(dec_perm))

    # 4. the relaxations
    for eps in sinkhorn_eps:
        T = sinkhorn_plan(C, alive_i, alive_t, float(eps), int(sinkhorn_iters), dev)
        Tn = T / np.clip(T.sum(axis=1, keepdims=True), 1e-12, None)
        score(f"sinkhorn_eps{eps}",
              f"Sinkhorn entropic transport, entropy weight {fmt(eps, 3)}",
              z_img, z_txt @ torch.as_tensor(Tn.T, dtype=torch.float32))

    R = procrustes_map(Sit)
    score("procrustes_rotation", METHOD_LABELS["procrustes_rotation"],
          z_img, z_txt @ torch.as_tensor(R, dtype=torch.float32))

    for d in cca_dims:
        d = int(d)
        if d > min(Sii.shape[0], Stt.shape[0]):
            logger.warning("[%s] %d canonical directions exceed the latent width, skipped",
                           NAME, d)
            continue
        A, B = cca_maps(Sii, Stt, Sit, d, float(ridge))
        score(f"cca_d{d}",
              f"canonical correlation analysis, {d:,} dimensions kept",
              z_img @ torch.as_tensor(A, dtype=torch.float32),
              z_txt @ torch.as_tensor(B, dtype=torch.float32))

    payload: dict[str, Any] = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "ckpt": str(setting.ckpt_a),
        "fit_split": setting.split,
        "fit_cache_dir": str(setting.cache_dir),
        "n_fit_pairs": int(fit_rows.shape[0]),
        "max_fit_samples": int(max_fit_samples),
        "eval_cache_dir": str(setting.coco_cache),
        "eval_split": eval_split,
        "n_eval_images": int(z_img.shape[0]),
        "n_eval_captions": int(z_txt.shape[0]),
        "n_alive_image": int(alive_i.sum()),
        "n_alive_text": int(alive_t.sum()),
        "n_usable_matches": int(usable.sum()),
        "sinkhorn_iters": int(sinkhorn_iters),
        "ridge": float(ridge),
        "labels": labels,
        "results": results,
    }
    write_json(out_json, payload)
    _write_report(out_dir / f"{NAME}.md", setting, payload)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _write_report(path: Path, setting, payload: dict[str, Any]) -> None:
    results = payload["results"]
    labels = payload["labels"]

    intro = (
        f"The setting is {setting.title()}. One operator links its image latent "
        f"space to its text latent space, and only that operator changes from row "
        f"to row below: the sparse autoencoders, the training data and the "
        f"evaluation are identical everywhere. Each operator is fitted on the "
        f"{payload['fit_split']} split and scored on the held-out COCO "
        f"{payload['eval_split']} split, {payload['n_eval_images']:,} images and "
        f"{payload['n_eval_captions']:,} captions, so the operators carrying free "
        f"parameters cannot buy their score by memorizing. The latent space holds "
        f"{payload['n_alive_image']:,} image latents and "
        f"{payload['n_alive_text']:,} text latents that fired at least once on the "
        f"training split, and the Hungarian assignment pairs "
        f"{payload['n_usable_matches']:,} of them with a partner that is alive as "
        f"well. Recall at k is the share of queries whose correct answer appears "
        f"in the top k, with ties against the correct answer counted against the "
        f"query rather than for it."
    )

    rows = []
    for key, r in results.items():
        rows.append([
            labels.get(key, key),
            pct(r["I2T R@1"]), pct(r["I2T R@5"]), pct(r["I2T R@10"]),
            pct(r["T2I R@1"]), pct(r["T2I R@5"]), pct(r["T2I R@10"]),
        ])
    table = md_table(
        ["alignment operator",
         "image finds caption, recall at 1",
         "image finds caption, recall at 5",
         "image finds caption, recall at 10",
         "caption finds image, recall at 1",
         "caption finds image, recall at 5",
         "caption finds image, recall at 10"],
        rows,
    )

    tie_table = md_table(
        ["alignment operator",
         "image queries with a tie at the correct caption",
         "caption queries with a tie at the correct image"],
        [[labels.get(key, key),
          pct(r["I2T tie at ground truth rate"]),
          pct(r["T2I tie at ground truth rate"])]
         for key, r in results.items()],
    )

    provenance = (
        f"The correlation matrix, the alive masks and the Hungarian permutation "
        f"come from `{payload['panel']}`, built over the full "
        f"{payload['fit_split']} split; a latent counts as alive when it fired at "
        f"least once there, and no firing-rate threshold is applied. Procrustes "
        f"and canonical correlation analysis need the full within-modality "
        f"covariance matrices, which the panel does not store, so one streaming "
        f"pass over {payload['n_fit_pairs']:,} evenly spaced pairs of "
        f"`{payload['fit_cache_dir']}` accumulates the first and second moments "
        f"of both latent streams. The ridge term added to each within-modality "
        f"covariance is {fmt(payload['ridge'], 5)} times its own mean diagonal, "
        f"and the transport plans run {payload['sinkhorn_iters']:,} Sinkhorn "
        f"iterations. Latents are taken from `{payload['ckpt']}`."
    )

    scope = (
        "A permutation names a partner for each coordinate, so a statement of "
        "the form \"image latent 137 corresponds to text latent 2891\" survives "
        "it. A rotation and a canonical projection do not: each output "
        "coordinate is a dense combination of every input coordinate, so no "
        "individual coordinate carries a concept afterwards. The table measures "
        "retrieval only, which is one property of an alignment and not the one "
        "the per-coordinate correspondence tests measure."
    )

    write_md(
        path,
        "Retrieval under six alignment operators, with everything else held fixed",
        [intro, provenance, scope],
        [
            ("COCO retrieval on the held-out split", table),
            ("How often the correct answer was tied with another candidate", tie_table),
        ],
    )


__all__ = [
    "run",
    "NAME",
    "hungarian_perm",
    "greedy_perm",
    "sinkhorn_plan",
    "procrustes_map",
    "cca_maps",
    "decoder_cosine_matrix",
    "second_moments",
    "centered_covariances",
    "recalls",
    "coco_test_rows",
    "METHOD_LABELS",
]

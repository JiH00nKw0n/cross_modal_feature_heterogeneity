"""The single co-activation panel every downstream stage reads.

One artifact, `panel.npz`, carries the co-activation correlation matrix, the
alive masks and the one Hungarian assignment, so that no two analyses can
disagree about which latents are alive or which latent is matched to which.

Three rules are implemented here and nowhere else.

Alive rule. A latent is alive when it fires at least once over the rows the
panel was built on, that is `fire_count >= 1`. There is no firing-rate
threshold anywhere in this repository. The rate is still reported, as
`fire_count / n_samples`, because it is useful to read, but nothing is filtered
by it.

Correlation rule. `C[i, j]` is the Pearson correlation between image latent `i`
and text latent `j` over the panel's rows, accumulated in one streaming pass in
float64. The (N, L) latent matrix is never materialized, which is what makes
the full CC3M training split (2.8M rows) tractable. A latent with zero variance
gives a correlation of 0 rather than NaN.

Matching rule. One signed, alive-restricted Hungarian assignment on C: dead
rows and dead columns are pushed to BIG_NEG = -1e9 so they cannot take an alive
latent's partner, then `linear_sum_assignment(-C_masked)` maximizes the total
correlation. The correlation is used signed; it is never passed through abs().

Who has to look at `usable`. The assignment is a full permutation of the latent
columns, so every row is given a partner, including a row that never fired and
whose partner is therefore arbitrary. `usable` marks the rows alive on both
sides. Every statistic computed over matched pairs, the cosine distance between
a latent and its partner above all, has to be restricted to those rows, because
averaging in the arbitrary partners drags the summary toward the value for
unmatched noise. The downstream evaluations are the deliberate exception: COCO
retrieval and ImageNet zero-shot reindex the whole text latent vector by
`perm`, with no mask, which is the paper's protocol and is what makes `ours`
and `separated` differ by the reindexing and by nothing else.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.data.cache_io import load_stacked, split_rows
from src.data.paired_dataset import normalize_np
from src.models import TwoSidedTopKSAE

logger = logging.getLogger(__name__)

#: Cost assigned to a dead row or column so it cannot win an assignment.
BIG_NEG = -1e9

#: The alive rule, written once so the sidecar and the docs cannot drift.
ALIVE_RULE = "fire_count >= 1 on the full train split"

PAIRINGS = ("img_txt", "img_img", "txt_txt", "txt_txt_diffcap")

#: Batches between progress lines while streaming.
_LOG_EVERY = 64


@torch.no_grad()
def _dense_latents(sae, x: torch.Tensor) -> torch.Tensor:
    """Post-TopK dense latents (B, L) for a batch of embeddings (B, dim)."""
    out = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
    return out.dense_latents.squeeze(1).float()


@torch.no_grad()
def accumulate_cross_stats(
    sae_a,
    sae_b,
    X: np.ndarray,
    Y: np.ndarray,
    rows_a: np.ndarray,
    rows_b: np.ndarray,
    batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Stream two embedding tables through two SAE sides and accumulate moments.

    Implements the correlation rule's streaming half. `X` and `Y` are the raw
    stacked tables (numpy arrays or memmaps); `rows_a` and `rows_b` select and
    order the rows fed to side A and side B. Rows are L2-normalized on the way
    in, matching how the SAEs were trained.

    Accumulators are float64 on the chosen device and are flushed to CPU
    periodically. The (N, L) latent matrix is never held. For a total latent
    budget of 8192 the per-side width is 4096, so `cross` is 4096 x 4096
    float64, which is 128 MB.

    Returns sum_a, sum_b, sumsq_a, sumsq_b, cross (L_a, L_b), fire_a, fire_b
    (counts of non-zero activations) and n, all on the CPU.
    """
    if rows_a.shape[0] != rows_b.shape[0]:
        raise ValueError(f"row counts differ: {rows_a.shape[0]} vs {rows_b.shape[0]}")
    dev = torch.device(device)
    sae_a = sae_a.to(dev).eval()
    sae_b = sae_b.to(dev).eval()
    L_a = int(sae_a.latent_size)
    L_b = int(sae_b.latent_size)
    n = int(rows_a.shape[0])

    sum_a = torch.zeros(L_a, dtype=torch.float64)
    sum_b = torch.zeros(L_b, dtype=torch.float64)
    sumsq_a = torch.zeros(L_a, dtype=torch.float64)
    sumsq_b = torch.zeros(L_b, dtype=torch.float64)
    cross = torch.zeros(L_a, L_b, dtype=torch.float64)
    fire_a = torch.zeros(L_a, dtype=torch.float64)
    fire_b = torch.zeros(L_b, dtype=torch.float64)

    # float32 partials on the compute device, flushed into the float64 totals.
    p_sa = torch.zeros(L_a, device=dev)
    p_sb = torch.zeros(L_b, device=dev)
    p_qa = torch.zeros(L_a, device=dev)
    p_qb = torch.zeros(L_b, device=dev)
    p_cross = torch.zeros(L_a, L_b, device=dev)
    p_fa = torch.zeros(L_a, device=dev)
    p_fb = torch.zeros(L_b, device=dev)
    flush_every = 64

    def _flush() -> None:
        nonlocal sum_a, sum_b, sumsq_a, sumsq_b, cross, fire_a, fire_b
        sum_a += p_sa.cpu().double(); p_sa.zero_()
        sum_b += p_sb.cpu().double(); p_sb.zero_()
        sumsq_a += p_qa.cpu().double(); p_qa.zero_()
        sumsq_b += p_qb.cpu().double(); p_qb.zero_()
        cross += p_cross.cpu().double(); p_cross.zero_()
        fire_a += p_fa.cpu().double(); p_fa.zero_()
        fire_b += p_fb.cpu().double(); p_fb.zero_()

    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n)
        xb = torch.from_numpy(normalize_np(X[rows_a[s:e]])).to(dev)
        yb = torch.from_numpy(normalize_np(Y[rows_b[s:e]])).to(dev)
        za = _dense_latents(sae_a, xb)
        zb = _dense_latents(sae_b, yb)
        p_sa += za.sum(0)
        p_sb += zb.sum(0)
        p_qa += (za * za).sum(0)
        p_qb += (zb * zb).sum(0)
        p_cross += za.T @ zb
        p_fa += (za != 0).float().sum(0)
        p_fb += (zb != 0).float().sum(0)
        if (b + 1) % flush_every == 0:
            _flush()
        if (b + 1) % _LOG_EVERY == 0:
            logger.info("[panel] %d/%d rows", e, n)
    _flush()

    return {
        "sum_a": sum_a.numpy(), "sum_b": sum_b.numpy(),
        "sumsq_a": sumsq_a.numpy(), "sumsq_b": sumsq_b.numpy(),
        "cross": cross.numpy(),
        "fire_a": fire_a.numpy().astype(np.int64),
        "fire_b": fire_b.numpy().astype(np.int64),
        "n": n,
    }


def pearson_from_stats(stats: dict[str, Any]) -> np.ndarray:
    """Pearson correlation (L_a, L_b) float64 from the streamed accumulators.

    Implements the correlation rule's closing half. A latent whose activation
    never varies has zero variance, and its entries come out as 0 rather than
    NaN, so the matrix is always finite and can go straight into the assignment.
    """
    n = float(stats["n"])
    if n <= 0:
        raise ValueError("cannot compute a correlation from zero rows")
    mean_a = stats["sum_a"] / n
    mean_b = stats["sum_b"] / n
    cov = stats["cross"] / n - np.outer(mean_a, mean_b)
    var_a = np.maximum(stats["sumsq_a"] / n - mean_a ** 2, 0.0)
    var_b = np.maximum(stats["sumsq_b"] / n - mean_b ** 2, 0.0)
    denom = np.maximum(np.outer(np.sqrt(var_a), np.sqrt(var_b)), 1e-12)
    C = cov / denom
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def hungarian_alive(
    C: np.ndarray, alive_a: np.ndarray, alive_b: np.ndarray,
) -> dict[str, Any]:
    """One signed, alive-restricted Hungarian assignment on C.

    Implements the matching rule. Dead rows and dead columns are set to
    BIG_NEG so they cannot take an alive latent's partner, then
    `linear_sum_assignment(-C_masked)` maximizes total correlation. The sign of
    C is kept; abs() is never applied.

    Returns the full-length `perm` (so `perm[i]` is the partner of left latent
    `i`), `usable` marking rows alive on both sides, and `matched_c`, the
    correlation of each row's assigned partner. Rows outside `usable` carry an
    arbitrary partner and must be excluded from any statistic computed over
    matched pairs; see the module docstring for the one place that applies the
    permutation in full instead.
    """
    Cm = np.array(C, dtype=np.float64, copy=True)
    Cm[~alive_a, :] = BIG_NEG
    Cm[:, ~alive_b] = BIG_NEG
    Cm = np.nan_to_num(Cm, nan=BIG_NEG, posinf=1.0, neginf=BIG_NEG)

    row, col = linear_sum_assignment(-Cm)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    perm[row] = col

    usable = alive_a.copy()
    usable[row] &= alive_b[col]
    matched_c = np.full(C.shape[0], np.nan, dtype=np.float64)
    matched_c[row] = C[row, col]

    return {
        "perm": perm,
        "usable": usable,
        "matched_c": matched_c,
        "n_alive_a": int(alive_a.sum()),
        "n_alive_b": int(alive_b.sum()),
        "n_usable": int(usable.sum()),
    }


# --------------------------------------------------------------------------- #
# Pairing definitions
# --------------------------------------------------------------------------- #
def _other_caption_rows(keys: Sequence[str], rows: np.ndarray) -> tuple[np.ndarray, int]:
    """For each pair, the row of a DIFFERENT caption of the same image.

    Only meaningful for a COCO-style cache, whose keys are "{image_id}_{cap_idx}".
    Captions of one image are ordered as they appear in the split, and the pair
    at position t takes position t+1 within that image, wrapping, so every pair
    gets a partner and no caption is paired with itself. An image with a single
    caption keeps its own caption; those rows are counted and reported, because
    for them the pairing degenerates to the same-input case.
    """
    by_image: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        key = keys[int(r)]
        image_id = key.rsplit("_", 1)[0]
        by_image.setdefault(image_id, []).append(i)

    partner = np.empty(rows.shape[0], dtype=np.int64)
    n_singleton = 0
    for idxs in by_image.values():
        if len(idxs) == 1:
            partner[idxs[0]] = idxs[0]
            n_singleton += 1
            continue
        for j, pos in enumerate(idxs):
            partner[pos] = idxs[(j + 1) % len(idxs)]
    logger.info("[panel] different-caption pairing: %d/%d rows had no alternative caption",
                n_singleton, rows.shape[0])
    return rows[partner], n_singleton


def _resolve_pairing(
    pairing: str,
    model_a: TwoSidedTopKSAE,
    model_b: TwoSidedTopKSAE,
    cache: dict[str, Any],
    rows: np.ndarray,
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """(sae_a, sae_b, table_a, table_b, rows_a, rows_b, n_singleton) for a pairing.

    img_txt          image SAE and text SAE of model A over each pair. This is
                     the paper's panel.
    img_img          image SAE of model A against image SAE of model B, both
                     reading the same image.
    txt_txt          text SAE of model A against text SAE of model B, both
                     reading the same caption.
    txt_txt_diffcap  text SAE of model A against text SAE of model B, reading
                     two different captions of the same photo.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"unknown pairing {pairing!r}; expected one of {PAIRINGS}")
    image, text = cache["image"], cache["text"]
    if pairing == "img_txt":
        return model_a.image_sae, model_a.text_sae, image, text, rows, rows, 0
    if pairing == "img_img":
        return model_a.image_sae, model_b.image_sae, image, image, rows, rows, 0
    if pairing == "txt_txt":
        return model_a.text_sae, model_b.text_sae, text, text, rows, rows, 0
    rows_b, n_singleton = _other_caption_rows(cache["keys"], rows)
    return model_a.text_sae, model_b.text_sae, text, text, rows, rows_b, n_singleton


def _subsample(rows: np.ndarray, max_samples: int) -> np.ndarray:
    """Evenly spaced subsample, the same rule the paper's scripts use.

    `max_samples == 0` means every row. A positive value takes
    `np.linspace(0, n - 1, max_samples)`, which keeps the sample spread over
    the whole split instead of taking a prefix.
    """
    n = int(rows.shape[0])
    if max_samples <= 0 or max_samples >= n:
        return rows
    sel = np.linspace(0, n - 1, max_samples, dtype=np.int64)
    return rows[sel]


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_panel(
    *,
    model: TwoSidedTopKSAE,
    cache_dir: str | Path,
    split: str = "train",
    batch_size: int = 8192,
    device: str = "cuda",
    max_samples: int = 0,
    shuffle_seed: int = 0,
    pairing: str = "img_txt",
    model_b: TwoSidedTopKSAE | None = None,
    ckpt_a: str | Path | None = None,
    ckpt_b: str | Path | None = None,
) -> dict[str, Any]:
    """Build the panel: correlation matrix, alive masks and one assignment.

    Applies all three rules stated at the top of this module. `max_samples = 0`
    means every pair of the split, which is what Table 1 and Figure 2 use; a
    positive value takes an evenly spaced subsample. `shuffle_seed != 0`
    permutes the B-side rows, destroying the pairing, which turns the panel
    into a noise floor: shuffling the rows and recomputing is the only valid
    way to do that, because permuting the columns of a finished correlation
    matrix leaves each row's maximum intact and the assignment simply finds it
    again.

    `pairing` selects which two SAE sides are compared; see `_resolve_pairing`.
    The non-default pairings need `model_b`, a second training run of the same
    architecture.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    other = (model_b if model_b is not None else model).to(dev).eval()

    cache = load_stacked(cache_dir, mmap=True)
    rows = split_rows(cache, split)
    # Two counts, because they can differ: `n_declared` is how many keys the
    # split claims, `n_resolved` how many of them were found in keys.json.
    # Recording only the resolved count would let the sidecar say the panel
    # covers the whole split when part of it was silently dropped.
    n_declared = int(len(cache["splits"][split]))
    n_resolved = int(rows.shape[0])
    if n_resolved != n_declared:
        logger.warning("[panel] split %r declares %d keys but only %d resolve to rows",
                       split, n_declared, n_resolved)
    rows = _subsample(rows, max_samples)

    sae_a, sae_b, table_a, table_b, rows_a, rows_b, n_singleton = _resolve_pairing(
        pairing, model, other, cache, rows,
    )

    if shuffle_seed:
        order = np.random.default_rng(shuffle_seed).permutation(rows_a.shape[0])
        rows_b = rows_b[order]
        logger.info("[panel] pairing destroyed with seed %d (noise floor)", shuffle_seed)

    logger.info("[panel] %s over %d/%d rows of split %r (device=%s)",
                pairing, rows_a.shape[0], n_resolved, split, dev)
    stats = accumulate_cross_stats(sae_a, sae_b, table_a, table_b,
                                   rows_a, rows_b, batch_size=batch_size, device=dev)
    C = pearson_from_stats(stats)

    fire_a = stats["fire_a"]
    fire_b = stats["fire_b"]
    alive_a = fire_a >= 1
    alive_b = fire_b >= 1
    n = int(stats["n"])
    logger.info("[panel] alive (%s): A %d/%d, B %d/%d",
                ALIVE_RULE, int(alive_a.sum()), alive_a.size,
                int(alive_b.sum()), alive_b.size)

    assign = hungarian_alive(C, alive_a, alive_b)
    usable = assign["usable"]
    matched = assign["matched_c"][usable]
    logger.info("[panel] usable matches %d, mean matched correlation %.4f",
                int(usable.sum()), float(matched.mean()) if matched.size else float("nan"))

    payload = {
        "C": C.astype(np.float32),
        "perm": assign["perm"].astype(np.int64),
        "usable": usable.astype(bool),
        "alive_image": alive_a.astype(bool),
        "alive_text": alive_b.astype(bool),
        "fire_count_image": fire_a.astype(np.int64),
        "fire_count_text": fire_b.astype(np.int64),
        "rate_image": (fire_a / max(n, 1)).astype(np.float64),
        "rate_text": (fire_b / max(n, 1)).astype(np.float64),
        "n_samples": np.int64(n),
    }
    payload["_meta"] = {
        "pairing": pairing,
        "ckpt_a": str(ckpt_a) if ckpt_a is not None else "",
        "ckpt_b": str(ckpt_b) if ckpt_b is not None else "",
        "cache_dir": str(cache_dir),
        "split": split,
        "n_samples": n,
        "n_split_rows": n_declared,
        "n_split_rows_resolved": n_resolved,
        "max_samples": int(max_samples),
        "shuffle_seed": int(shuffle_seed),
        "alive_rule": ALIVE_RULE,
        "n_alive_image": int(alive_a.sum()),
        "n_alive_text": int(alive_b.sum()),
        "n_usable": int(usable.sum()),
        "n_singleton_caption": int(n_singleton),
    }
    return payload


def save_panel(out_path: str | Path, payload: dict[str, Any]) -> None:
    """Write panel.npz plus the panel.json sidecar describing how it was built."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = payload.get("_meta", {})
    arrays = {k: v for k, v in payload.items() if not k.startswith("_")}
    np.savez(out_path, **arrays)
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)


def panel_mismatch(
    payload: dict[str, Any],
    *,
    split: str,
    pairing: str = "img_txt",
    max_samples: int = 0,
    shuffle_seed: int = 0,
) -> str | None:
    """Why an existing panel cannot stand in for the one being asked for.

    Returns None when the panel on disk was built under exactly the requested
    rules, and a sentence naming the first difference otherwise. A pipeline
    reuses `panel.npz` whenever the file exists, and the file alone does not say
    how it was built: a quick-check panel written with `max_samples > 0`, or a
    noise-floor panel written with a non-zero `shuffle_seed`, is the same shape
    as the real thing and would otherwise be picked up silently by the next full
    run. A panel with no `panel.json` sidecar cannot be checked at all and is
    reported as such.
    """
    meta = payload.get("_meta")
    if not meta:
        return "it has no panel.json sidecar, so how it was built cannot be checked"
    checks = (
        ("split", split), ("pairing", pairing),
        ("max_samples", int(max_samples)), ("shuffle_seed", int(shuffle_seed)),
    )
    for key, want in checks:
        if key not in meta:
            return f"its sidecar does not record {key}"
        got = meta[key]
        if isinstance(want, int):
            got = int(got)
        if got != want:
            return f"it was built with {key}={got!r}, not {key}={want!r}"
    return None


def load_panel(path: str | Path) -> dict[str, Any]:
    """Read panel.npz (and panel.json when present) back into a payload dict."""
    path = Path(path)
    data = np.load(path)
    payload: dict[str, Any] = {k: data[k] for k in data.files}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        with open(sidecar) as f:
            payload["_meta"] = json.load(f)
    return payload


__all__ = [
    "accumulate_cross_stats",
    "pearson_from_stats",
    "hungarian_alive",
    "build_panel",
    "save_panel",
    "load_panel",
    "panel_mismatch",
    "ALIVE_RULE",
    "BIG_NEG",
    "PAIRINGS",
]

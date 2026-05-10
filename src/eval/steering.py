"""Cross-modal SAE steering on COCO test split.

Protocol (Kaushik et al. 2026 §A.5, Härle 2025):
  1. Identify concept latent k_T in TEXT-side SAE (positive vs negative captions).
  2. Map to image-side k_I per variant (single dict, perm-aligned, or raw-paired).
  3. Encode an off-concept BASE IMAGE → densify → z_dense[..., k_I] += α·σ_c → decode.
  4. Measure Δcos(x_steered, t_c) where t_c = mean caption embedding for concept c.

Variants compared:
  shared / iso_align / group_sparse  — k_I = k_T (single dictionary)
  ours                              — k_I = k_T in aligned frame (Hungarian perm)
  separated                         — k_I = k_T raw (no slot map; expected to fail)

Outputs (per variant):
  steering_per_concept.csv  — concept, k_T, k_I, alpha, mean Δ_c, mean preserve, retrieval rank
  summary.json              — α-aggregated metrics M1, M3, M4 + concept-latent table
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.eval import eval_utils
from src.eval.coco_concepts import COCO_CONCEPTS as COCO_80

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, required=True, help="label only")
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True, help="CLIP COCO cache (paired embeds)")
    p.add_argument("--captions-json", type=str, required=True,
                   help="cache_dir/captions.json (auto-emitted by extract_coco)")
    p.add_argument("--perm", type=str, default=None, help="required when method=ours")
    p.add_argument("--alphas", type=str, default="0,0.5,1.0,2.0,4.0,8.0",
                   help="comma-separated α multipliers (in σ_c units)")
    p.add_argument("--n-base-images", type=int, default=100,
                   help="off-concept base images per concept")
    p.add_argument("--min-pos-captions", type=int, default=1,
                   help="skip concepts with fewer matching captions (dataset filter)")
    p.add_argument("--min-base-images", type=int, default=50,
                   help="skip concepts with fewer than this off-concept images")
    p.add_argument("--require-alive-image", action="store_true",
                   help="skip concepts whose image-side slot is dead (default off "
                        "so all variants share the same concept set).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def _l2(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _load_paired_tensors(cache_dir: str, split: str = "test"):
    """Return (image_ids, text_keys, img, txt, pairs).

    img: (M, H) one row per unique image
    txt: (N, H) one row per (img, cap) pair
    pairs: list[(image_id, cap_idx)] of length N
    """
    ds = eval_utils.load_pair_dataset(cache_dir, "coco", split)
    pairs = ds.pairs  # type: ignore[attr-defined]
    img_ids = [int(p[0]) for p in pairs]
    unique_img_ids = sorted(set(img_ids))
    img_cache = ds._image_dict  # type: ignore[attr-defined]
    txt_cache = ds._text_dict  # type: ignore[attr-defined]
    img = torch.stack([img_cache[iid] for iid in unique_img_ids], dim=0)
    txt_keys = [f"{int(p[0])}_{int(p[1])}" for p in pairs]
    txt = torch.stack([txt_cache[k] for k in txt_keys], dim=0)
    return (
        np.array(unique_img_ids, dtype=np.int64),
        txt_keys,
        img,
        txt,
        [(int(p[0]), int(p[1])) for p in pairs],
    )


def _load_captions_text(captions_json: str, pairs: list[tuple[int, int]]) -> list[str]:
    """Return caption text aligned with pairs (length N)."""
    with open(captions_json) as f:
        d = json.load(f)
    out: list[str] = []
    for iid, cidx in pairs:
        for sep in ("::", "_"):
            key = f"{iid}{sep}{cidx}"
            if key in d:
                out.append(d[key])
                break
        else:
            out.append("")
    return out


# ----------------------------------------------------------------------
# Concept matching
# ----------------------------------------------------------------------

def _concept_pattern(concept: str) -> re.Pattern:
    # Word-boundary, lowercase. For multi-word concepts use whole phrase.
    return re.compile(r"\b" + re.escape(concept.lower()) + r"\b")


def _match_captions(captions: list[str], concept: str) -> np.ndarray:
    """Boolean mask over caption indices that contain concept."""
    pat = _concept_pattern(concept)
    return np.array([bool(pat.search(c.lower())) for c in captions], dtype=bool)


def _image_has_concept(captions: list[str], pairs: list[tuple[int, int]],
                       concept: str) -> dict[int, bool]:
    """For each unique image id, True iff ANY of its 5 captions contains concept."""
    pat = _concept_pattern(concept)
    out: dict[int, bool] = {}
    for (iid, _), cap in zip(pairs, captions):
        if pat.search(cap.lower()):
            out[iid] = True
        elif iid not in out:
            out[iid] = False
    return out


# ----------------------------------------------------------------------
# Retrieval metrics
# ----------------------------------------------------------------------

KS = (1, 5, 10, 50, 100)


def _retrieval_metrics(scores: np.ndarray, rel_set: set[int]) -> dict:
    """Vectorized ranking metrics from (B, N) scores against rel_set indices.

    Means computed over the B queries (NaN if rel_set empty / no rel found).
        median_rank: median position (0-indexed) of relevant items per query,
                     then median across queries.
        map:         mean Average Precision (full ranking).
        mrr:         Mean Reciprocal Rank — 1 / (rank+1) of FIRST relevant.
        ndcg10:      NDCG at cutoff 10.
        p@K, r@K for K in {1,5,10,50,100}:
            P@K = |top-K ∩ rel| / K
            R@K = |top-K ∩ rel| / |rel|
    """
    nan = float("nan")
    empty = {"median_rank": nan, "map": nan, "mrr": nan, "ndcg10": nan}
    for K in KS:
        empty[f"p{K}"] = nan
        empty[f"r{K}"] = nan
    if not rel_set:
        return empty

    B, N = scores.shape
    rel_mask_global = np.zeros(N, dtype=bool)
    rel_mask_global[np.fromiter(rel_set, dtype=np.int64)] = True

    # (B, N) order: descending scores. argsort handles ties deterministically.
    order = np.argsort(-scores, axis=1, kind="stable")        # (B, N) int
    rel_at_rank = rel_mask_global[order]                       # (B, N) bool
    n_rel_per_q = rel_at_rank.sum(axis=1)                      # (B,)
    has_any = n_rel_per_q > 0
    if not has_any.any():
        return empty

    ranks_1d = np.arange(1, N + 1, dtype=np.float64)           # 1..N
    cum_rel = np.cumsum(rel_at_rank, axis=1).astype(np.float64)  # (B, N)
    prec_at_each = cum_rel / ranks_1d[None, :]                 # (B, N)

    # mAP per query = sum(prec_at_each * rel) / n_rel_in_query
    sum_prec = (prec_at_each * rel_at_rank).sum(axis=1)
    ap_per_q = np.where(n_rel_per_q > 0,
                         sum_prec / np.maximum(n_rel_per_q, 1), 0.0)

    # MRR: position of first True in each row (or 0 if no True).
    first_rel_rank = np.argmax(rel_at_rank, axis=1)            # (B,)
    mrr_per_q = np.where(has_any, 1.0 / (first_rel_rank + 1), 0.0)

    # NDCG@10
    K_NDCG = 10
    log_disc = 1.0 / np.log2(np.arange(2, K_NDCG + 2, dtype=np.float64))  # length K
    dcg = (rel_at_rank[:, :K_NDCG].astype(np.float64) * log_disc[None]).sum(axis=1)
    n_rel_topk = np.minimum(n_rel_per_q, K_NDCG)
    log_disc_cum = np.cumsum(log_disc)
    idcg = np.where(n_rel_topk > 0,
                     log_disc_cum[np.clip(n_rel_topk - 1, 0, K_NDCG - 1)],
                     0.0)
    ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)

    # Median rank: per-query median of rel positions, then median across queries.
    # Use cumulative-counts approach: position where cumsum reaches half.
    # For exactness, do per-query find — vectorize via boolean arr -> argwhere is row-uneven.
    # For B≤200 this is fast enough as a Python loop.
    med_ranks = []
    for b in range(B):
        if has_any[b]:
            ranks_b = np.flatnonzero(rel_at_rank[b])
            med_ranks.append(float(np.median(ranks_b)))
    median_rank = float(np.median(med_ranks)) if med_ranks else nan

    # P@K, R@K via cum_rel at index K-1
    out = {
        "median_rank": median_rank,
        "map": float(ap_per_q[has_any].mean()),
        "mrr": float(mrr_per_q[has_any].mean()),
        "ndcg10": float(ndcg[has_any].mean()),
    }
    for K in KS:
        if K > N:
            out[f"p{K}"] = nan
            out[f"r{K}"] = nan
            continue
        hits_k = cum_rel[:, K - 1]                              # (B,) hits in top-K
        out[f"p{K}"] = float((hits_k / K)[has_any].mean())
        out[f"r{K}"] = float((hits_k / np.maximum(n_rel_per_q, 1))[has_any].mean())
    return out


# ----------------------------------------------------------------------
# Steering primitives
# ----------------------------------------------------------------------

@torch.no_grad()
def _encode_image_dense(model, x: torch.Tensor, method: str,
                         device: torch.device, batch_size: int) -> torch.Tensor:
    """Encode image embeddings (M, H) → dense latents (M, L)."""
    return eval_utils.encode_image(model, x, method, device, batch_size)


@torch.no_grad()
def _decode_dense(model, method: str, z_dense: torch.Tensor) -> torch.Tensor:
    """Decode dense latents (B, L) via image-side decoder.

    For shared/aux: model.W_dec / model.b_dec.
    For separated/ours: model.image_sae.W_dec / image_sae.b_dec.
    """
    if method in ("shared", "aux"):
        sae = model
    else:
        sae = model.image_sae
    W = sae.W_dec  # (L, H)
    b = sae.b_dec  # (H,)
    return z_dense @ W + b


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run(
    *,
    ckpt: str | Path,
    method: str,
    cache_dir: str | Path,
    captions_json: str | Path,
    output_dir: str | Path,
    perm_path: str | Path | None = None,
    alphas: list[float] | tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    concepts: list[str] | None = None,
    n_base_images: int = 100,
    min_pos_captions: int = 1,
    min_base_images: int = 50,
    require_alive_image: bool = False,
    seed: int = 0,
    batch_size: int = 2048,
    device: str = "cuda",
    variant: str | None = None,
    cache_x_steered_npz: bool = False,
) -> dict:
    """Cross-modal steering across COCO concepts; writes CSV + summary.json."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if concepts is None:
        concepts = list(COCO_80)
    if variant is None:
        variant = method
    alphas = [float(a) for a in alphas]

    logger.info("[steering] loading ckpt=%s method=%s", ckpt, method)
    model = eval_utils.load_sae(ckpt, method)

    perm = None
    alive_image = None
    if method == "ours":
        if perm_path is None:
            raise ValueError("perm_path required for method='ours'")
        perm_npz = np.load(str(perm_path))
        perm = perm_npz["perm"]
        if "alive_image" in perm_npz.files:
            alive_image = perm_npz["alive_image"].astype(bool)

    image_ids, _text_keys, img_t, txt_t, pairs = _load_paired_tensors(str(cache_dir))
    M, N = img_t.shape[0], txt_t.shape[0]
    logger.info("[steering] unique_images=%d captions=%d", M, N)
    captions_text = _load_captions_text(str(captions_json), pairs)
    assert len(captions_text) == N

    t0 = time.time()
    z_T = eval_utils.encode_text(model, txt_t, method, dev,
                                  perm=perm, batch_size=batch_size).cpu()
    L = z_T.shape[1]
    logger.info("[steering] encoded text latents %s in %.1fs", tuple(z_T.shape), time.time() - t0)

    sigma_text = z_T.std(dim=0).clamp_min(1e-6)

    z_I = _encode_image_dense(model, img_t, method, dev, batch_size).cpu()
    fire_image = (z_I != 0).sum(dim=0).numpy()
    sigma_image = z_I.std(dim=0).clamp_min(1e-6)

    if method in ("shared", "aux"):
        W_dec_img = model.W_dec  # type: ignore[union-attr]
    else:
        W_dec_img = model.image_sae.W_dec  # type: ignore[union-attr]
    W_img_norms = W_dec_img.detach().cpu().norm(dim=1).clamp_min(1e-6).numpy()

    rng = np.random.default_rng(seed)
    concept_records: list[dict] = []
    concept_steering_rows: list[dict] = []
    cache_x_steered: list[dict] = []

    txt_norm_full = _l2(txt_t)
    img_norm_full = _l2(img_t)
    image_id_to_row = {int(iid): i for i, iid in enumerate(image_ids)}

    for concept in tqdm(concepts, desc="concept", unit="concept"):
        c_mask = _match_captions(captions_text, concept)
        n_pos = int(c_mask.sum())
        if n_pos < min_pos_captions:
            continue

        neg_pool = np.where(~c_mask)[0]
        n_neg = min(len(neg_pool), n_pos)
        neg_idx = rng.choice(neg_pool, size=n_neg, replace=False)
        pos_idx = np.where(c_mask)[0]

        z_pos = z_T[pos_idx]
        z_neg = z_T[neg_idx]
        diff = (z_pos.mean(dim=0) - z_neg.mean(dim=0)).numpy()
        k_T = int(np.argmax(diff))
        mean_diff_k = float(diff[k_T])
        sigma_c_text = float(sigma_text[k_T])

        # k_I = k_T for all four supported methods (ours uses perm-aligned text → image slot identity).
        k_I = k_T

        if require_alive_image:
            if method == "ours" and alive_image is not None and not bool(alive_image[k_I]):
                continue
            if int(fire_image[k_I]) == 0:
                continue
        is_alive = int(fire_image[k_I]) > 0

        img_concept = _image_has_concept(captions_text, pairs, concept)
        on_iids = [iid for iid in image_ids.tolist() if img_concept.get(int(iid), False)]
        if not on_iids:
            continue
        on_rows = np.array([image_id_to_row[int(iid)] for iid in on_iids])
        on_rows_set = set(on_rows.tolist())
        sigma_c_img = float(sigma_image[k_T])

        off_iids = [iid for iid in image_ids.tolist() if not img_concept.get(int(iid), False)]
        if len(off_iids) < min_base_images:
            continue

        concept_seed = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
        rng_img = np.random.default_rng(seed + concept_seed % 100000)
        chosen_iids = rng_img.choice(off_iids,
                                     size=min(n_base_images, len(off_iids)),
                                     replace=False)
        base_rows = np.array([image_id_to_row[int(iid)] for iid in chosen_iids])
        x_base = img_t[base_rows].to(dev)
        x_base_norm = _l2(x_base.cpu())

        z_base = _encode_image_dense(model, x_base.cpu(), method, dev,
                                     batch_size=batch_size).to(dev)

        W_norm_k = float(W_img_norms[k_I])
        for alpha in alphas:
            with torch.no_grad():
                z_mod = z_base.clone()
                z_mod[..., k_I] = z_mod[..., k_I] + alpha / W_norm_k
                x_steer = _decode_dense(model, method, z_mod)
                x_steer_norm = _l2(x_steer).cpu()

            preserve = (x_steer_norm * x_base_norm).sum(-1).numpy()
            scores_txt = (x_steer_norm @ txt_norm_full.T).numpy()
            scores_img = (x_steer_norm @ img_norm_full.T).numpy()
            mt = _retrieval_metrics(scores_txt, set(pos_idx.tolist()))
            mi = _retrieval_metrics(scores_img, on_rows_set)

            row = {
                "concept": concept,
                "k_T": k_T, "k_I": k_I,
                "alpha": alpha,
                "n_pos": n_pos, "n_neg": int(n_neg),
                "n_base": int(len(chosen_iids)),
                "sigma_c_text": sigma_c_text,
                "sigma_c_img": sigma_c_img,
                "image_slot_alive": int(is_alive),
                "mean_diff_k": mean_diff_k,
                "preserve_mean": float(preserve.mean()),
            }
            for k, v in mt.items():
                row[f"{k}_txt"] = v
            for k, v in mi.items():
                row[f"{k}_img"] = v
            concept_steering_rows.append(row)

            if cache_x_steered_npz:
                cache_x_steered.append({
                    "concept": concept, "alpha": alpha, "k_T": k_T, "k_I": k_I,
                    "embeds": x_steer_norm.numpy().astype(np.float32),
                    "base_image_rows": base_rows.astype(np.int64),
                })

        concept_records.append({
            "concept": concept,
            "k_T": k_T, "k_I": k_I,
            "n_pos": n_pos,
            "mean_diff_k": mean_diff_k,
            "sigma_c_text": sigma_c_text,
            "sigma_c_img": sigma_c_img,
        })

    csv_path = out_dir / "steering_per_concept.csv"
    if concept_steering_rows:
        cols = list(concept_steering_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(concept_steering_rows)

    summary: dict = {
        "variant": variant,
        "method": method,
        "split": "test",
        "alphas": alphas,
        "n_concepts": len(concept_records),
        "n_total_concept_alpha_rows": len(concept_steering_rows),
        "latent_size": int(L),
    }

    by_alpha: dict[float, list[dict]] = {a: [] for a in alphas}
    for r in concept_steering_rows:
        by_alpha[r["alpha"]].append(r)
    summary["per_alpha"] = {}

    def _mean_drop_nan(vals: list[float]) -> float:
        v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(v)) if v else float("nan")

    metric_keys = ["preserve_mean"]
    for side in ("img", "txt"):
        metric_keys.extend([f"median_rank_{side}", f"map_{side}",
                            f"mrr_{side}", f"ndcg10_{side}"])
        for K in (1, 5, 10, 50, 100):
            metric_keys.extend([f"p{K}_{side}", f"r{K}_{side}"])

    for a in alphas:
        rows = by_alpha[a]
        if not rows:
            continue
        agg: dict = {"n_concepts": len(rows)}
        for mk in metric_keys:
            agg[mk] = _mean_drop_nan([r.get(mk, float("nan")) for r in rows])
        summary["per_alpha"][f"{a}"] = agg

    summary["concepts"] = concept_records
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if cache_x_steered:
        n_rec = len(cache_x_steered)
        max_B = max(item["embeds"].shape[0] for item in cache_x_steered)
        H = cache_x_steered[0]["embeds"].shape[1]
        embeds_stacked = np.zeros((n_rec, max_B, H), dtype=np.float32)
        base_rows_stacked = np.full((n_rec, max_B), -1, dtype=np.int64)
        for i, item in enumerate(cache_x_steered):
            B = item["embeds"].shape[0]
            embeds_stacked[i, :B] = item["embeds"]
            base_rows_stacked[i, :B] = item["base_image_rows"]
        meta = [{"concept": x["concept"], "alpha": x["alpha"],
                 "k_T": x["k_T"], "k_I": x["k_I"]} for x in cache_x_steered]
        np.savez_compressed(
            out_dir / "x_steered_cache.npz",
            embeddings=embeds_stacked,
            base_rows=base_rows_stacked,
        )
        with open(out_dir / "x_steered_cache_meta.json", "w") as f:
            json.dump(meta, f)

    logger.info("[steering] DONE variant=%s concepts=%d rows=%d → %s",
                variant, len(concept_records), len(concept_steering_rows), out_dir)
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    alphas = [float(a) for a in args.alphas.split(",")]
    run(
        ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        captions_json=args.captions_json, output_dir=args.out,
        perm_path=args.perm, alphas=alphas, n_base_images=args.n_base_images,
        min_pos_captions=args.min_pos_captions, min_base_images=args.min_base_images,
        require_alive_image=args.require_alive_image, seed=args.seed,
        batch_size=args.batch_size, device=args.device, variant=args.variant,
    )


if __name__ == "__main__":
    main()

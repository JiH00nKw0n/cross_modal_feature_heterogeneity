"""COCO Karpathy test cross-modal retrieval in SAE latent space.

Protocol:
  - I→T: for each image, rank all captions by ``cos(z_I, z_T)``; Recall@k is
    1 if any of the 5 ground-truth captions sits in top-k.
  - T→I: for each caption, rank all unique images; Recall@k on the single
    ground-truth image.

All latents are extracted via ``eval_utils.encode_{image,text}``; ``ours``
applies the saved Hungarian perm on the text side.

Callable from a pipeline as ``run(...)`` or from the shell via ``main()``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.eval import eval_utils

logger = logging.getLogger(__name__)


def _recall_at_k(ranks: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    return {f"R@{k}": float((ranks < k).mean()) for k in ks}


def run(
    *,
    ckpt: str | Path,
    method: str,
    cache_dir: str | Path,
    output: str | Path,
    split: str = "test",
    perm_path: str | Path | None = None,
    batch_size: int = 2048,
    device: str = "cuda",
) -> dict:
    """Run COCO retrieval for one method; write the result JSON and return it."""
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[retrieval] loading ckpt=%s method=%s", ckpt, method)
    model = eval_utils.load_sae(ckpt, method)

    perm = None
    if method == "ours":
        if perm_path is None:
            raise ValueError("perm_path required for method='ours'")
        perm = np.load(str(perm_path))["perm"]

    ds = eval_utils.load_pair_dataset(cache_dir, "coco", split)
    logger.info("[retrieval] pairs=%d split=%s", len(ds), split)

    pairs = ds.pairs  # type: ignore[attr-defined]
    img_ids = [int(p[0]) for p in pairs]
    unique_img_ids = sorted(set(img_ids))
    id_to_idx = {iid: k for k, iid in enumerate(unique_img_ids)}
    pair_img_idx = np.array([id_to_idx[iid] for iid in img_ids], dtype=np.int64)

    img_cache = ds._image_dict  # type: ignore[attr-defined]
    txt_cache = ds._text_dict  # type: ignore[attr-defined]
    img = torch.stack([img_cache[iid] for iid in unique_img_ids], dim=0)
    txt = torch.stack([txt_cache[f"{int(p[0])}_{int(p[1])}"] for p in pairs], dim=0)

    z_img = eval_utils.encode_image(model, img, method, dev, batch_size)
    z_txt = eval_utils.encode_text(model, txt, method, dev, perm=perm, batch_size=batch_size)
    z_img = eval_utils.normalize_rows(z_img)
    z_txt = eval_utils.normalize_rows(z_txt)

    # T→I: pessimistic rank of each caption's ground-truth image.
    t2i_ranks = np.empty(z_txt.shape[0], dtype=np.int64)
    t2i_tie_size = np.empty(z_txt.shape[0], dtype=np.int64)
    chunk = 1024
    for s in range(0, z_txt.shape[0], chunk):
        scores = z_txt[s:s + chunk] @ z_img.T
        gt = pair_img_idx[s:s + chunk]
        gt_scores = scores[np.arange(len(gt)), gt]
        ge_count = (scores >= gt_scores[:, None]).sum(dim=1)
        eq_count = (scores == gt_scores[:, None]).sum(dim=1)
        t2i_ranks[s:s + chunk] = (ge_count - 1).cpu().numpy()
        t2i_tie_size[s:s + chunk] = eq_count.cpu().numpy()

    # I→T: best (lowest) pessimistic rank across the 5 ground-truth captions per image.
    gt_caps_per_img: list[list[int]] = [[] for _ in range(len(unique_img_ids))]
    for cap_p, img_idx in enumerate(pair_img_idx):
        gt_caps_per_img[int(img_idx)].append(cap_p)

    i2t_min_rank = np.empty(z_img.shape[0], dtype=np.int64)
    i2t_tie_size = np.empty(z_img.shape[0], dtype=np.int64)
    for s in range(0, z_img.shape[0], chunk):
        scores = z_img[s:s + chunk] @ z_txt.T
        for row in range(scores.shape[0]):
            img_idx = s + row
            gt_caps = gt_caps_per_img[img_idx]
            best_gt_score = scores[row, gt_caps].max()
            ge_count = int((scores[row] >= best_gt_score).sum().item())
            eq_count = int((scores[row] == best_gt_score).sum().item())
            i2t_min_rank[img_idx] = ge_count - 1
            i2t_tie_size[img_idx] = eq_count

    t2i = _recall_at_k(t2i_ranks)
    i2t = _recall_at_k(i2t_min_rank)
    logger.info("[retrieval] T→I %s", t2i)
    logger.info("[retrieval] I→T %s", i2t)

    result = {
        "method": method,
        "dataset": "coco",
        "split": split,
        "n_images": int(len(unique_img_ids)),
        "n_captions": int(len(pairs)),
        "latent_size": int(z_img.shape[1]),
        "T2I": t2i,
        "I2T": i2t,
        "T2I_ties": {
            "tie_at_gt_rate": float((t2i_tie_size > 1).mean()),
            "mean_tie_size": float(t2i_tie_size.mean()),
        },
        "I2T_ties": {
            "tie_at_gt_rate": float((i2t_tie_size > 1).mean()),
            "mean_tie_size": float(i2t_tie_size.mean()),
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("[retrieval] wrote %s", out_path)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        output=args.output, split=args.split, perm_path=args.perm,
        batch_size=args.batch_size, device=args.device,
    )


if __name__ == "__main__":
    main()

"""COCO Karpathy cross-modal retrieval in SAE latent space.

Protocol, unchanged from the paper's script:

  T to I: for each caption, rank every unique image by cos(z_T, z_I); Recall@k
          is 1 when the caption's own image is inside the top k.
  I to T: for each image, rank every caption; the image's rank is the BEST
          (lowest) rank over its ground-truth captions, so an image counts as
          retrieved when any one of its captions is found.

Ranking is pessimistic. A rank counts every candidate scoring greater than OR
equal to the ground truth, minus one, so a tie against the ground truth pushes
it down rather than up. Collapsed latents produce large tied blocks, and an
optimistic tie rule would report those as perfect retrieval. The size of the
tie block at the ground truth is reported alongside the recalls, so a reader
can see when that is happening.

Unique images come from the cache key prefix: a COCO key is
"{image_id}_{cap_idx}", and the image row is duplicated across a photo's
captions, so the first row of each image id is taken once.

Latents come from `eval_utils.encode_image` and `eval_utils.encode_text`;
`ours` applies the saved Hungarian permutation on the text side.
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


def _recall_at_k(ranks: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
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
        perm = eval_utils.load_perm(perm_path)

    ds = eval_utils.load_paired_split(cache_dir, split)
    logger.info("[retrieval] pairs=%d split=%s", len(ds), split)

    # Unique images: first row per image id, plus each caption's image position.
    image_ids = [eval_utils.coco_image_id(k) for k in ds.keys]
    first_row: dict[str, int] = {}
    order: list[str] = []
    for row, iid in enumerate(image_ids):
        if iid not in first_row:
            first_row[iid] = row
            order.append(iid)
    id_to_idx = {iid: i for i, iid in enumerate(order)}
    pair_img_idx = np.array([id_to_idx[iid] for iid in image_ids], dtype=np.int64)
    img_rows = torch.as_tensor([first_row[iid] for iid in order], dtype=torch.long)

    z_img = eval_utils.encode_image(model, ds.image[img_rows], method, dev, batch_size)
    z_txt = eval_utils.encode_text(model, ds.text, method, dev, perm=perm, batch_size=batch_size)
    z_img = eval_utils.normalize_rows(z_img)
    z_txt = eval_utils.normalize_rows(z_txt)

    # T to I: pessimistic rank of each caption's ground-truth image.
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

    # I to T: best pessimistic rank across the image's ground-truth captions.
    gt_caps_per_img: list[list[int]] = [[] for _ in range(len(order))]
    for cap_pos, img_idx in enumerate(pair_img_idx):
        gt_caps_per_img[int(img_idx)].append(cap_pos)

    i2t_min_rank = np.empty(z_img.shape[0], dtype=np.int64)
    i2t_tie_size = np.empty(z_img.shape[0], dtype=np.int64)
    for s in range(0, z_img.shape[0], chunk):
        scores = z_img[s:s + chunk] @ z_txt.T
        for row in range(scores.shape[0]):
            img_idx = s + row
            best_gt_score = scores[row, gt_caps_per_img[img_idx]].max()
            i2t_min_rank[img_idx] = int((scores[row] >= best_gt_score).sum().item()) - 1
            i2t_tie_size[img_idx] = int((scores[row] == best_gt_score).sum().item())

    t2i = _recall_at_k(t2i_ranks)
    i2t = _recall_at_k(i2t_min_rank)
    logger.info("[retrieval] T to I %s", t2i)
    logger.info("[retrieval] I to T %s", i2t)

    result = {
        "method": method,
        "dataset": "coco",
        "split": split,
        "n_images": int(len(order)),
        "n_captions": int(len(ds)),
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
    p.add_argument("--method", type=str, required=True,
                   choices=["shared", "separated", "aux", "ours", "iso_align", "group_sparse"])
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        output=args.output, split=args.split, perm_path=args.perm,
        batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()

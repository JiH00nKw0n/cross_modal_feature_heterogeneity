"""Reconstruction error of an SAE on a held-out split.

The paper's formula, reproduced exactly:

    recon = 0.5 * mean over pairs of ( ||x - x_hat||^2 + ||y - y_hat||^2 )

where x and y are the L2-normalized image and text embeddings and x_hat, y_hat
are each SAE's reconstruction of its own modality. The norm is the squared L2
norm summed over the embedding dimension, averaged over pairs, not divided by
the dimension.

Two datasets are supported and they pair their text side differently.

COCO. The pair is what the cache says it is: caption row i belongs to image row
i, on the split named by `split` (the paper uses the test split).

ImageNet. The cache is not paired, so the paper's dataset object builds a pair
per validation image by drawing ONE of the 80 templates of that image's true
class at random, and the reconstruction is measured on that draw. That is
reproduced here, with the draw made by a seeded generator (`template_seed`) so
the number is reproducible, which the paper's own run was not.

For `ours` the reconstruction is identical to `separated` by construction: the
Hungarian permutation only renames latent slots and cannot change what either
decoder outputs. It is computed and written anyway, so every method has a row.
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


@torch.no_grad()
def _mean_squared_error(sae, embeds: torch.Tensor, batch_size: int,
                        device: torch.device) -> float:
    """Mean over rows of the squared L2 reconstruction error."""
    sae.eval()
    sae.to(device)
    total = 0.0
    n = 0
    for s in range(0, embeds.shape[0], batch_size):
        chunk = embeds[s:s + batch_size].to(device).unsqueeze(1)
        out = sae(hidden_states=chunk)
        err = (chunk.squeeze(1) - out.output.squeeze(1)).pow(2).sum(dim=-1)
        total += float(err.sum().item())
        n += chunk.shape[0]
    return total / max(n, 1)


def _imagenet_pairs(inet: eval_utils.ImageNetSplit, seed: int) -> torch.Tensor:
    """One random template per validation image, drawn from its true class.

    Row `c * n_templates + t` of the text table is template `t` of class `c`,
    so the draw is a random `t` per image followed by that index arithmetic.
    """
    rng = np.random.default_rng(seed)
    t = rng.integers(0, inet.n_templates, size=inet.labels.shape[0])
    rows = inet.labels * inet.n_templates + t
    return inet.text[torch.as_tensor(rows, dtype=torch.long)]


def run(
    *,
    ckpt: str | Path,
    method: str,
    cache_dir: str | Path,
    output: str | Path,
    dataset: str = "coco",
    split: str = "test",
    batch_size: int = 2048,
    device: str = "cuda",
    template_seed: int = 0,
) -> dict:
    """Compute the reconstruction error for one method; write the JSON and return it."""
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[recon] loading ckpt=%s method=%s", ckpt, method)
    model = eval_utils.load_sae(ckpt, method)

    if dataset == "imagenet":
        inet = eval_utils.load_imagenet_split(cache_dir)
        img = inet.image
        txt = _imagenet_pairs(inet, template_seed)
        n = int(img.shape[0])
        split_name = "val"
    else:
        ds = eval_utils.load_paired_split(cache_dir, split)
        img, txt = ds.image, ds.text
        n = len(ds)
        split_name = split

    mse_img = _mean_squared_error(eval_utils.image_sae_of(model, method), img, batch_size, dev)
    mse_txt = _mean_squared_error(eval_utils.text_sae_of(model, method), txt, batch_size, dev)
    recon = 0.5 * (mse_img + mse_txt)
    logger.info("[recon] %s/%s: recon=%.4f (image %.4f, text %.4f)",
                dataset, split_name, recon, mse_img, mse_txt)

    result = {
        "method": method,
        "dataset": dataset,
        "split": split_name,
        "n": n,
        "recon_error": recon,
        "recon_image": mse_img,
        "recon_text": mse_txt,
    }
    if dataset == "imagenet":
        result["template_seed"] = int(template_seed)
        result["text_pairing"] = "one random template of the image's true class"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("[recon] wrote %s", out_path)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["shared", "separated", "aux", "ours", "iso_align", "group_sparse"])
    p.add_argument("--dataset", type=str, default="coco", choices=["coco", "cc3m", "imagenet"])
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--template-seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        output=args.output, dataset=args.dataset, split=args.split,
        batch_size=args.batch_size, device=args.device,
        template_seed=args.template_seed)


if __name__ == "__main__":
    main()

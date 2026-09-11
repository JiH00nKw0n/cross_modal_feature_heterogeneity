"""ImageNet-1K zero-shot classification in SAE latent space.

Pipeline:

  1. For each class c, average its 80 template embeddings in CLIP space,
     L2-normalize that average, then encode it through the text-side SAE (for
     `ours`, apply the permutation) to get the class prototype z_T^c.
  2. Encode each validation image through the image-side SAE to get z_I.
  3. Predict argmax_c cos(z_I, z_T^c) and report top-1 accuracy.

Two variants exist and they are not interchangeable.

    raw       every latent column takes part in the cosine. This is what the
              paper's Table 1 reports: the task named "zeroshot_raw" in the
              paper's configs runs `eval_imagenet_zeroshot.py`, which applies
              no mask and no filter at all. It is the default here.
    filtered  columns whose image-side firing rate exceeds `max_fire_rate`
              (default 0.5) are dropped from both the prototypes and the images
              before the cosine, so a latent that is on for half the dataset
              cannot dominate the similarity. This mirrors the paper's separate
              `eval_imagenet_zeroshot_filtered.py`.

The variant is recorded in the result JSON, and the pipeline writes the raw
variant to `zeroshot.json` and the filtered one to `zeroshot_filtered.json`, so
the two can never be mistaken for each other.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.paired_dataset import l2_normalize_rows
from src.eval import eval_utils

logger = logging.getLogger(__name__)

VARIANTS = ("raw", "filtered")


def _class_prototypes(inet: eval_utils.ImageNetSplit) -> torch.Tensor:
    """Mean of each class's templates in CLIP space, L2-normalized.

    The text table is class-major, so row `c * n_templates + t` is template `t`
    of class `c` and the mean is a reshape away.
    """
    grid = inet.text.reshape(inet.n_classes, inet.n_templates, -1)
    return l2_normalize_rows(grid.mean(dim=1))


def run(
    *,
    ckpt: str | Path,
    method: str,
    cache_dir: str | Path,
    output: str | Path,
    perm_path: str | Path | None = None,
    batch_size: int = 2048,
    variant: str = "raw",
    max_fire_rate: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Run ImageNet zero-shot for one method; write the result JSON and return it.

    `variant="raw"` reproduces the paper's Table 1 column. `variant="filtered"`
    drops latents whose image-side firing rate exceeds `max_fire_rate`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[zeroshot] loading ckpt=%s method=%s variant=%s", ckpt, method, variant)
    model = eval_utils.load_sae(ckpt, method)

    perm = None
    if method == "ours":
        if perm_path is None:
            raise ValueError("perm_path required for method='ours'")
        perm = eval_utils.load_perm(perm_path)

    inet = eval_utils.load_imagenet_split(cache_dir)
    protos_clip = _class_prototypes(inet)
    z_protos = eval_utils.encode_text(model, protos_clip, method, dev,
                                      perm=perm, batch_size=batch_size)
    z_val = eval_utils.encode_image(model, inet.image, method, dev, batch_size)
    y_val = inet.labels

    L = int(z_val.shape[1])
    fire_rate = (z_val != 0).float().mean(dim=0).cpu().numpy()
    if variant == "filtered":
        keep = fire_rate <= max_fire_rate
        logger.info("[zeroshot] keeping %d/%d latents (fire rate <= %.2f)",
                    int(keep.sum()), L, max_fire_rate)
        keep_t = torch.from_numpy(keep)
        z_val = z_val[:, keep_t]
        z_protos = z_protos[:, keep_t]
        kept = int(keep.sum())
    else:
        kept = L

    z_val = eval_utils.normalize_rows(z_val)
    z_protos = eval_utils.normalize_rows(z_protos)

    correct = 0
    bsz = 8192
    for s in range(0, z_val.shape[0], bsz):
        scores = z_val[s:s + bsz] @ z_protos.T
        pred = scores.argmax(dim=1).cpu().numpy()
        correct += int((pred == y_val[s:s + bsz]).sum())
    acc = correct / max(z_val.shape[0], 1)
    logger.info("[zeroshot] top-1 accuracy (%s): %.4f", variant, acc)

    result = {
        "method": method,
        "dataset": "imagenet",
        "metric": "zeroshot_top1" if variant == "raw" else "zeroshot_top1_filtered",
        "variant": variant,
        "accuracy": float(acc),
        "kept_latents": kept,
        "total_latents": L,
        "kept_fraction": float(kept / max(L, 1)),
        "n_val": int(z_val.shape[0]),
        "n_classes": int(inet.n_classes),
        "n_templates": int(inet.n_templates),
    }
    if variant == "filtered":
        result["max_fire_rate"] = float(max_fire_rate)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("[zeroshot] wrote %s", out_path)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["shared", "separated", "aux", "ours", "iso_align", "group_sparse"])
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--variant", type=str, default="raw", choices=list(VARIANTS))
    p.add_argument("--max-fire-rate", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        output=args.output, perm_path=args.perm, batch_size=args.batch_size,
        variant=args.variant, max_fire_rate=args.max_fire_rate, device=args.device)


if __name__ == "__main__":
    main()

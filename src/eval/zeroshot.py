"""ImageNet-1K zero-shot classification in SAE latent space.

Pipeline:
  1) For each class c (0..999): take the cached template text embeddings
     in CLIP space and MEAN them. L2-normalize. Encode through text-side SAE
     (and apply the perm for ``ours``) → class prototype z_T^c.
  2) For each val image: encode with image-side SAE → z_I.
  3) Predict: ``argmax_c cos(z_I, z_T^c)``.

The prototypes can be optionally filtered along latents whose image-side
fire rate exceeds ``max_fire_rate`` (default 0.5).

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


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _build_class_prototypes_clip(
    text_dict: dict[str, torch.Tensor], n_classes: int, n_templates: int,
) -> torch.Tensor:
    protos = []
    for c in range(n_classes):
        vecs = torch.stack(
            [text_dict[f"{c}_{t}"] for t in range(n_templates)], dim=0,
        )
        mean = vecs.mean(dim=0)
        protos.append(_l2_normalize(mean))
    return torch.stack(protos, dim=0)


def run(
    *,
    ckpt: str | Path,
    method: str,
    cache_dir: str | Path,
    output: str | Path,
    perm_path: str | Path | None = None,
    batch_size: int = 2048,
    n_classes: int = 1000,
    n_templates: int = 80,
    max_fire_rate: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Run ImageNet zero-shot for one method; write the result JSON and return it."""
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[zeroshot] loading ckpt=%s method=%s", ckpt, method)
    model = eval_utils.load_sae(ckpt, method)

    perm = None
    if method == "ours":
        if perm_path is None:
            raise ValueError("perm_path required for method='ours'")
        perm = np.load(str(perm_path))["perm"]

    text_dict_raw = torch.load(
        str(Path(cache_dir) / "text_embeddings.pt"), map_location="cpu",
    )
    text_dict = {str(k): v.to(torch.float32) for k, v in text_dict_raw.items()}
    for k, v in text_dict.items():
        text_dict[k] = _l2_normalize(v)

    protos_clip = _build_class_prototypes_clip(text_dict, n_classes, n_templates)
    z_protos = eval_utils.encode_text(model, protos_clip, method, dev,
                                      perm=perm, batch_size=batch_size)

    val_ds = eval_utils.load_pair_dataset(cache_dir, "imagenet", "val")
    img_va = torch.stack([val_ds[i]["image_embeds"] for i in range(len(val_ds))], dim=0)
    y_va = np.array([int(val_ds.pairs[i][1]) for i in range(len(val_ds))], dtype=np.int64)
    z_val = eval_utils.encode_image(model, img_va, method, dev, batch_size)

    L = z_val.shape[1]
    fire_rate = (z_val != 0).float().mean(dim=0).cpu().numpy()
    keep = fire_rate <= max_fire_rate
    logger.info("[zeroshot] keep %d/%d latents (fire_rate≤%.2f)",
                int(keep.sum()), L, max_fire_rate)
    keep_t = torch.from_numpy(keep).to(z_val.device)
    z_val = eval_utils.normalize_rows(z_val[:, keep_t])
    z_protos = eval_utils.normalize_rows(z_protos[:, keep_t])

    correct = 0
    bsz = 8192
    for s in range(0, z_val.shape[0], bsz):
        scores = z_val[s:s + bsz] @ z_protos.T
        pred = scores.argmax(dim=1).cpu().numpy()
        correct += int((pred == y_va[s:s + bsz]).sum())
    acc = correct / z_val.shape[0]
    logger.info("[zeroshot] val top-1 accuracy (filtered): %.4f", acc)

    result = {
        "method": method,
        "dataset": "imagenet",
        "metric": "zeroshot_top1",
        "accuracy": float(acc),
        "max_fire_rate": float(max_fire_rate),
        "kept_latents": int(keep.sum()),
        "total_latents": int(L),
        "kept_fraction": float(keep.mean()),
        "n_val": int(z_val.shape[0]),
        "n_classes": int(n_classes),
        "n_templates": int(n_templates),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("[zeroshot] wrote %s", out_path)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours"], required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--perm", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--n-templates", type=int, default=80)
    p.add_argument("--max-fire-rate", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(
        ckpt=args.ckpt, method=args.method, cache_dir=args.cache_dir,
        output=args.output, perm_path=args.perm,
        batch_size=args.batch_size, n_classes=args.n_classes,
        n_templates=args.n_templates, max_fire_rate=args.max_fire_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()

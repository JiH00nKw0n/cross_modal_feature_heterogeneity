"""Extract CLIP image/text embeddings for ImageNet-1K.

Uses HF streaming to avoid downloading the full dataset to disk.
Images: all images cached (no subsampling). Subsample at experiment time.
Texts: 80 OpenAI templates × 1000 classes = 80K embeddings.

Usage:
    export HF_TOKEN=...
    python scripts/real_alpha/extract_imagenet_cache.py \
        --model openai/clip-vit-base-patch32 \
        --cache-dir cache/clip_b32_imagenet \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "ILSVRC/imagenet-1k"
# Offset val image indices to avoid collision with train in the shared dict.
SPLIT_OFFSETS = {"train": 0, "validation": 2_000_000}
SPLIT_TOTALS = {"train": 1_281_167, "validation": 50_000}
HF_TO_OUR_SPLIT = {"train": "train", "validation": "val"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                   help="HF model id (transformers) or OpenCLIP arch name")
    p.add_argument("--pretrained", type=str, default="",
                   help="OpenCLIP pretrained tag (only used when --backend openclip)")
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_imagenet")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--flush-every", type=int, default=50000,
                   help="Save checkpoint every N new embeddings (default 50000 for ImageNet)")
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    return p.parse_args()


def load_state(path: Path) -> dict:
    if path.exists():
        logger.info("Loading existing cache %s", path)
        return torch.load(path, map_location="cpu")
    return {}


def save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _run_extract(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    hf_token = os.environ.get("HF_TOKEN")
    logger.info("Device: %s", device)

    backend = args.backend
    if backend == "openclip":
        import open_clip
        pretrained = args.pretrained or None
        logger.info("Loading OpenCLIP %s (pretrained=%s)", args.model, pretrained)
        oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=pretrained, device=device)
        oc_model.eval()
        oc_tokenizer = open_clip.get_tokenizer(args.model)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            emb_dim = int(oc_model.encode_image(dummy).shape[-1])
        siglip = False
        model = oc_model  # for downstream `del model` and reuse
        processor = None
        logger.info("OpenCLIP dim: %d", emb_dim)
    else:
        logger.info("Loading %s", args.model)
        model = AutoModel.from_pretrained(args.model).to(device).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(args.model)
        siglip = getattr(model.config, "model_type", "") == "siglip"
        if siglip:
            emb_dim = model.config.vision_config.hidden_size
        else:
            emb_dim = model.config.projection_dim
        oc_preprocess = None
        oc_tokenizer = None
        logger.info("Embedding dim: %d (siglip=%s)", emb_dim, siglip)

    image_dict: dict[int, torch.Tensor] = {
        int(k): v for k, v in load_state(cache_dir / "image_embeddings.pt").items()
    }
    text_dict: dict[str, torch.Tensor] = {
        str(k): v for k, v in load_state(cache_dir / "text_embeddings.pt").items()
    }

    # Load existing splits (built incrementally during image pass)
    splits_path = cache_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        splits = {}

    # -------- Image pass (streaming) --------
    for hf_split in args.splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        offset = SPLIT_OFFSETS[hf_split]
        total = SPLIT_TOTALS[hf_split]

        # Skip if this split is already fully cached
        if our_split in splits:
            split_ids = {p[0] for p in splits[our_split]}
            if split_ids.issubset(image_dict.keys()):
                logger.info("Image pass %s: SKIP (all %d cached)", hf_split, len(split_ids))
                continue

        n_cached_before = len(image_dict)
        logger.info("=== Image pass: %s (streaming, ~%d images, %d already cached) ===",
                     hf_split, total, n_cached_before)
        ds = load_dataset(DATASET_NAME, split=hf_split, streaming=True, token=hf_token)

        split_pairs: list[list[int]] = []
        pending_ids: list[int] = []
        pending_pils: list = []
        processed_since_flush = 0
        t_start = time.time()
        n_extracted = 0

        def flush_images() -> None:
            nonlocal pending_ids, pending_pils, processed_since_flush, n_extracted
            if not pending_pils:
                return
            with torch.no_grad():
                if backend == "openclip":
                    batch = torch.stack([oc_preprocess(p) for p in pending_pils]).to(device)
                    feats = model.encode_image(batch).float().cpu()
                else:
                    inputs = processor(images=pending_pils, return_tensors="pt")
                    vision_out = model.vision_model(
                        pixel_values=inputs["pixel_values"].to(device)
                    )
                    if siglip:
                        feats = vision_out.pooler_output.float().cpu()
                    else:
                        feats = model.visual_projection(vision_out.pooler_output).float().cpu()
            for iid, vec in zip(pending_ids, feats):
                image_dict[iid] = vec.clone()
            n_extracted += len(pending_ids)
            processed_since_flush += len(pending_ids)
            pending_ids = []
            pending_pils = []

        for counter, row in enumerate(tqdm(ds, desc=f"img-{hf_split}", total=total,
                                           miniters=100, smoothing=0.1)):
            img_idx = offset + counter
            class_idx = int(row["label"])
            split_pairs.append([img_idx, class_idx])

            if img_idx in image_dict:
                continue
            pending_ids.append(img_idx)
            pending_pils.append(row["image"].convert("RGB"))
            if len(pending_pils) >= args.batch_size:
                flush_images()
                if processed_since_flush >= args.flush_every:
                    elapsed = time.time() - t_start
                    rate = n_extracted / elapsed if elapsed > 0 else 0
                    remaining = (total - counter) / rate if rate > 0 else 0
                    logger.info(
                        "  checkpoint: %d/%d extracted, %.1f img/s, ETA %.0f min",
                        n_extracted, total, rate, remaining / 60,
                    )
                    save_state(image_dict, cache_dir / "image_embeddings.pt")
                    processed_since_flush = 0

        flush_images()
        save_state(image_dict, cache_dir / "image_embeddings.pt")
        splits[our_split] = split_pairs
        with open(splits_path, "w") as f:
            json.dump(splits, f)
        elapsed = time.time() - t_start
        logger.info(
            "Image pass %s done: %d extracted in %.1f min (%.1f img/s), total cached = %d",
            hf_split, n_extracted, elapsed / 60,
            n_extracted / elapsed if elapsed > 0 else 0, len(image_dict),
        )

    # -------- Text pass (templates × classes, no HF dataset needed) --------
    n_templates = len(OPENAI_IMAGENET_TEMPLATES)
    n_classes = len(IMAGENET_CLASSNAMES)
    expected_texts = n_classes * n_templates

    if len(text_dict) >= expected_texts:
        logger.info("Text pass: SKIP (all %d cached)", expected_texts)
    else:
        logger.info(
            "=== Text pass: %d classes × %d templates = %d texts ===",
            n_classes, n_templates, expected_texts,
        )

        pending_keys: list[str] = []
        pending_texts: list[str] = []
        t_txt_start = time.time()
        n_txt_extracted = 0

        def flush_texts() -> None:
            nonlocal pending_keys, pending_texts, n_txt_extracted
            if not pending_texts:
                return
            with torch.no_grad():
                if backend == "openclip":
                    tokens = oc_tokenizer(pending_texts).to(device)
                    feats = model.encode_text(tokens).float().cpu()
                else:
                    tok = processor(
                        text=pending_texts, return_tensors="pt",
                        padding="max_length" if siglip else True,
                        truncation=True, max_length=64 if siglip else 77,
                    )
                    input_ids = tok["input_ids"].to(device)
                    attention_mask = tok.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    text_out = model.text_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )
                    if siglip:
                        feats = text_out.pooler_output.float().cpu()
                    else:
                        feats = model.text_projection(text_out.pooler_output).float().cpu()
            for key, vec in zip(pending_keys, feats):
                text_dict[key] = vec.clone()
            n_txt_extracted += len(pending_keys)
            pending_keys = []
            pending_texts = []

        pbar = tqdm(total=expected_texts, desc="txt-templates", miniters=1000)
        for class_idx, class_name in enumerate(IMAGENET_CLASSNAMES):
            for tmpl_idx, tmpl in enumerate(OPENAI_IMAGENET_TEMPLATES):
                key = f"{class_idx}_{tmpl_idx}"
                pbar.update(1)
                if key in text_dict:
                    continue
                pending_keys.append(key)
                pending_texts.append(tmpl(class_name))
                if len(pending_texts) >= args.batch_size:
                    flush_texts()
        pbar.close()

        flush_texts()
        save_state(text_dict, cache_dir / "text_embeddings.pt")
        elapsed = time.time() - t_txt_start
        logger.info(
            "Text pass done: %d extracted in %.1f sec, total cached = %d",
            n_txt_extracted, elapsed, len(text_dict),
        )

    # -------- Meta --------
    meta = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "dim": emb_dim,
        "n_images": len(image_dict),
        "n_texts": len(text_dict),
        "n_classes": n_classes,
        "n_templates": n_templates,
        "class_names": list(IMAGENET_CLASSNAMES),
        "n_train_pairs": len(splits.get("train", [])),
        "n_val_pairs": len(splits.get("val", [])),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "meta.json written: n_images=%d, n_texts=%d, dim=%d",
        meta["n_images"], meta["n_texts"], meta["dim"],
    )


def extract(
    *,
    model_cfg,
    cache_dir: str,
    splits: list[str] | None = None,
    batch_size: int = 256,
    device: str = "cuda",
    flush_every: int = 50000,
) -> None:
    """Programmatic entry: ImageNet-1K embedding extraction (val images +
    1000 × 80 templated text prototypes)."""
    if splits is None:
        splits = ["validation"]
    backend = model_cfg.backend
    if backend == "openclip":
        model_str = model_cfg.arch
        pretrained = model_cfg.pretrained
    else:
        model_str = model_cfg.hf_id
        pretrained = ""
    args = argparse.Namespace(
        backend=backend, model=model_str, pretrained=pretrained,
        cache_dir=cache_dir, batch_size=batch_size, device=device,
        flush_every=flush_every, splits=splits,
    )
    os.makedirs(cache_dir, exist_ok=True)
    _run_extract(args)


def main() -> None:
    args = parse_args()
    _run_extract(args)


if __name__ == "__main__":
    main()

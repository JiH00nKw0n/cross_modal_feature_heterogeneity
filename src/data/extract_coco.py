"""Extract VLM image/text embeddings for COCO Karpathy.

Supports:
  - CLIPModel-based (CLIP, MetaCLIP) via transformers
  - SiglipModel-based (SigLIP2) via transformers
  - OpenCLIP models (DataComp, MobileCLIP) via --backend openclip

Usage:
    python scripts/real_alpha/extract_clip_coco_cache.py \
        --model openai/clip-vit-base-patch32 --cache-dir cache/clip_b32_coco
    python scripts/real_alpha/extract_clip_coco_cache.py \
        --backend openclip --model ViT-B-16 --pretrained datacomp_xl_s13b_b90k \
        --cache-dir cache/datacomp_b16_coco
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
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "namkha1032/coco-karpathy"
HF_TO_OUR_SPLIT = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--pretrained", type=str, default="",
                   help="OpenCLIP pretrained tag (e.g. datacomp_xl_s13b_b90k)")
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_coco")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--flush-every", type=int, default=4096)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    return p.parse_args()


def _is_siglip(model_name: str) -> bool:
    cfg = AutoConfig.from_pretrained(model_name)
    return cfg.model_type == "siglip"


def load_state(path: Path) -> dict:
    if path.exists():
        logger.info("Loading existing cache %s", path)
        return torch.load(path, map_location="cpu")
    return {}


def save_state(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _dump_captions_json(cache_dir: Path, hf_splits: list[str], expected_n: int) -> None:
    """Emit f"{image_id}_{cap_idx}" -> caption_text dict.

    Required by `eval_cross_modal_steering.py` (concept supervision) and
    several inspection / qualitative-viz scripts. Cheap to build (HF parquet
    iteration with the image column dropped, no model forward) so we always
    emit it as part of cache extraction. Two outputs:

      cache_dir/captions.json                 — per-cache, model-agnostic
      cache/coco_karpathy_captions.json       — global, written if absent
                                                 (legacy hardcoded path used
                                                 by inspect_*/probe_*/HTML viz)
    """
    per_cache = cache_dir / "captions.json"
    if per_cache.exists():
        try:
            with open(per_cache) as f:
                existing = json.load(f)
            if len(existing) >= expected_n:
                logger.info("captions.json up-to-date (%d entries) — skipping",
                            len(existing))
                return
        except Exception:
            pass

    captions: dict[str, str] = {}
    for hf_split in hf_splits:
        ds = load_dataset(DATASET_NAME, split=hf_split)
        if "image" in ds.column_names:
            ds = ds.remove_columns(["image"])
        for row in tqdm(ds, desc=f"caps-{hf_split}"):
            iid = int(row["image_id"])
            for ci, cap in enumerate(row["captions"]):
                captions[f"{iid}_{ci}"] = cap

    tmp = per_cache.with_suffix(per_cache.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(captions, f)
    tmp.replace(per_cache)
    logger.info("wrote %s (%d captions)", per_cache, len(captions))

    legacy = Path("cache/coco_karpathy_captions.json")
    if not legacy.exists():
        legacy.parent.mkdir(parents=True, exist_ok=True)
        tmp = legacy.with_suffix(legacy.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(captions, f)
        tmp.replace(legacy)
        logger.info("wrote %s (legacy global path)", legacy)


def build_splits(cache_dir: Path, hf_splits: list[str]) -> dict[str, list[tuple[str, int]]]:
    splits_path = cache_dir / "splits.json"
    if splits_path.exists():
        logger.info("splits.json already exists, loading")
        with open(splits_path, "r") as f:
            return json.load(f)

    out: dict[str, list] = {}
    for hf_split in hf_splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        logger.info("Building split list: %s", hf_split)
        ds = load_dataset(DATASET_NAME, split=hf_split)
        pairs: list = []
        for row in tqdm(ds, desc=f"split-{hf_split}"):
            image_id = str(row["image_id"])
            n_caps = len(row["captions"])
            for ci in range(n_caps):
                pairs.append([image_id, ci])
        out[our_split] = pairs

    with open(splits_path, "w") as f:
        json.dump(out, f)
    logger.info("Saved splits.json with %s entries", {k: len(v) for k, v in out.items()})
    return out


def _run_extract(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    model_name = args.model
    backend = args.backend

    if backend == "openclip":
        import open_clip
        pretrained = args.pretrained or None
        logger.info("Loading OpenCLIP %s (pretrained=%s)", model_name, pretrained)
        model, _, oc_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        model.eval()
        oc_tokenizer = open_clip.get_tokenizer(model_name)
        # Detect dim from a dummy forward
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            emb_dim = model.encode_image(dummy).shape[-1]
        siglip = False
        processor = None  # not used for openclip
        logger.info("OpenCLIP dim: %d", emb_dim)
    else:
        siglip = _is_siglip(model_name)
        logger.info("Loading %s (siglip=%s)", model_name, siglip)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(model_name)
        oc_preprocess = None
        oc_tokenizer = None
        if siglip:
            emb_dim = model.config.vision_config.hidden_size
        else:
            emb_dim = model.config.projection_dim
        logger.info("Embedding dim: %d", emb_dim)

    image_dict: dict[int, torch.Tensor] = {
        int(k): v for k, v in load_state(cache_dir / "image_embeddings.pt").items()
    }
    text_dict: dict[str, torch.Tensor] = {
        str(k): v for k, v in load_state(cache_dir / "text_embeddings.pt").items()
    }

    splits = build_splits(cache_dir, args.splits)

    # -------- Image pass --------
    for hf_split in args.splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        # Skip entire split if every image_id in this split is already cached.
        split_pairs = splits.get(our_split, [])
        split_iids = {int(p[0]) for p in split_pairs}
        if split_iids and split_iids.issubset(image_dict.keys()):
            logger.info("=== Image pass: %s (SKIP: all %d ids already cached) ===",
                        hf_split, len(split_iids))
            continue
        logger.info("=== Image pass: %s ===", hf_split)
        ds = load_dataset(DATASET_NAME, split=hf_split)
        pending_ids: list[int] = []
        pending_pils: list = []
        processed_since_flush = 0

        def flush_images():
            nonlocal pending_ids, pending_pils, processed_since_flush
            if not pending_pils:
                return
            with torch.no_grad():
                if backend == "openclip":
                    batch = torch.stack([oc_preprocess(p) for p in pending_pils]).to(device)
                    feats = model.encode_image(batch).float().cpu()
                elif siglip:
                    inputs = processor(images=pending_pils, return_tensors="pt")
                    out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
                    feats = out.pooler_output.float().cpu()
                else:
                    inputs = processor(images=pending_pils, return_tensors="pt")
                    vision_out = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
                    feats = model.visual_projection(vision_out.pooler_output).float().cpu()
            for iid, vec in zip(pending_ids, feats):
                image_dict[iid] = vec.clone()
            processed_since_flush += len(pending_ids)
            pending_ids = []
            pending_pils = []

        for row in tqdm(ds, desc=f"img-{hf_split}"):
            iid = int(row["image_id"])
            if iid in image_dict:
                continue
            pending_ids.append(iid)
            pending_pils.append(row["image"].convert("RGB"))
            if len(pending_pils) >= args.batch_size:
                flush_images()
                if processed_since_flush >= args.flush_every:
                    save_state(image_dict, cache_dir / "image_embeddings.pt")
                    processed_since_flush = 0
        flush_images()
        save_state(image_dict, cache_dir / "image_embeddings.pt")
        logger.info("image_dict size = %d", len(image_dict))

    # -------- Text pass --------
    for hf_split in args.splits:
        our_split = HF_TO_OUR_SPLIT[hf_split]
        # Skip entire split if every (image_id, caption_idx) is cached.
        split_pairs = splits.get(our_split, [])
        if split_pairs and all(f"{int(p[0])}_{int(p[1])}" in text_dict for p in split_pairs):
            logger.info("=== Text pass: %s (SKIP: all %d pairs already cached) ===",
                        hf_split, len(split_pairs))
            continue
        logger.info("=== Text pass: %s ===", hf_split)
        # Drop the image column so HF datasets doesn't PIL-decode every row.
        ds = load_dataset(DATASET_NAME, split=hf_split)
        if "image" in ds.column_names:
            ds = ds.remove_columns(["image"])
        pending_keys: list[str] = []
        pending_texts: list[str] = []
        processed_since_flush = 0

        def flush_texts():
            nonlocal pending_keys, pending_texts, processed_since_flush
            if not pending_texts:
                return
            with torch.no_grad():
                if backend == "openclip":
                    tokens = oc_tokenizer(pending_texts).to(device)
                    feats = model.encode_text(tokens).float().cpu()
                else:
                    tok = processor(
                        text=pending_texts, return_tensors="pt",
                        padding="max_length" if siglip else True, truncation=True,
                        max_length=64 if siglip else 77,
                    )
                    input_ids = tok["input_ids"].to(device)
                    attention_mask = tok.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    if siglip:
                        out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                        feats = out.pooler_output.float().cpu()
                    else:
                        text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                        feats = model.text_projection(text_out.pooler_output).float().cpu()
            for key, vec in zip(pending_keys, feats):
                text_dict[key] = vec.clone()
            processed_since_flush += len(pending_keys)
            pending_keys = []
            pending_texts = []

        for row in tqdm(ds, desc=f"txt-{hf_split}"):
            iid = int(row["image_id"])
            caps = row["captions"]
            for ci, cap in enumerate(caps):
                key = f"{iid}_{ci}"
                if key in text_dict:
                    continue
                pending_keys.append(key)
                pending_texts.append(cap)
                if len(pending_texts) >= args.batch_size:
                    flush_texts()
                    if processed_since_flush >= args.flush_every:
                        save_state(text_dict, cache_dir / "text_embeddings.pt")
                        processed_since_flush = 0
        flush_texts()
        save_state(text_dict, cache_dir / "text_embeddings.pt")
        logger.info("text_dict size = %d", len(text_dict))

    _dump_captions_json(cache_dir, args.splits, expected_n=len(text_dict))

    meta = {
        "clip_model": model_name,
        "dataset": DATASET_NAME,
        "dim": emb_dim,
        "n_images": len(image_dict),
        "n_texts": len(text_dict),
        "n_train_pairs": len(splits.get("train", [])),
        "n_val_pairs": len(splits.get("val", [])),
        "n_test_pairs": len(splits.get("test", [])),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("meta.json written: %s", meta)


def extract(
    *,
    model_cfg,
    cache_dir: str,
    splits: list[str] | None = None,
    batch_size: int = 256,
    device: str = "cuda",
    flush_every: int = 4096,
) -> None:
    """Programmatic entry: COCO Karpathy embedding extraction.

    `model_cfg` is a ``ModelConfig`` (from ``src.utils.config``); we read its
    ``backend`` / ``hf_id`` / ``arch`` / ``pretrained`` to drive the extractor.
    """
    if splits is None:
        splits = ["train", "validation", "test"]
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
    os.makedirs(args.cache_dir, exist_ok=True)
    _run_extract(args)


if __name__ == "__main__":
    main()

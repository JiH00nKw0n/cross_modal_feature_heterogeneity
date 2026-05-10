"""Extract CLIP image/text embeddings for pixparse/cc3m-wds (streaming + DataLoader).

CC3M is webdataset-format; each sample has `__key__`, `jpg`, `txt`.
We stream from HuggingFace (no full local download), parallelize JPEG decode
across DataLoader workers (each worker gets its own shard of the tar stream),
and feed decoded PIL batches to CLIP on GPU.

  cache_dir/
    image_embeddings.pt   # {__key__: Tensor(d,)}
    text_embeddings.pt    # {__key__: Tensor(d,)}
    splits.json           # {"train": [[__key__, 0], ...]}
    meta.json

Usage:
    python scripts/real_alpha/extract_clip_cc3m_cache.py \
        --model openai/clip-vit-base-patch32 \
        --cache-dir cache/clip_b32_cc3m \
        --batch-size 256 --num-workers 4 \
        --max-samples 0     # 0 = full CC3M (~3.3M)
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET = "pixparse/cc3m-wds"
HF_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="transformers",
                   choices=["transformers", "openclip"])
    p.add_argument("--model", type=str, default="openai/clip-vit-base-patch32",
                   help="HF model id (transformers) or OpenCLIP arch name (e.g. ViT-L-14)")
    p.add_argument("--pretrained", type=str, default="",
                   help="OpenCLIP pretrained tag (only used when --backend openclip)")
    p.add_argument("--cache-dir", type=str, default="cache/clip_b32_cc3m")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--flush-every", type=int, default=65536)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--hf-split", type=str, default=HF_SPLIT)
    p.add_argument("--num-workers", type=int, default=4)
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


def _transform(row):
    """Decode PIL image (triggered by .convert('RGB')) inside worker."""
    try:
        img = row["jpg"].convert("RGB")
    except Exception:
        return None
    txt = row.get("txt") or ""
    if not txt:
        return None
    return {"key": str(row["__key__"]), "img": img, "txt": txt}


def _load_model(args, device):
    """Returns (forward_image, forward_text, emb_dim, backend_kind).

    forward_image: list[PIL] -> Tensor[B, dim] (cpu, fp32)
    forward_text:  list[str] -> Tensor[B, dim] (cpu, fp32)
    backend_kind:  "openclip" | "siglip" | "clip" — for logging only
    """
    if args.backend == "openclip":
        import open_clip
        pretrained = args.pretrained or None
        logger.info("Loading OpenCLIP %s (pretrained=%s)", args.model, pretrained)
        model, _, oc_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=pretrained, device=device)
        model.eval()
        oc_tokenizer = open_clip.get_tokenizer(args.model)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            emb_dim = int(model.encode_image(dummy).shape[-1])

        @torch.no_grad()
        def fwd_img(pils):
            batch = torch.stack([oc_preprocess(p) for p in pils]).to(device)
            return model.encode_image(batch).float().cpu()

        @torch.no_grad()
        def fwd_txt(texts):
            tokens = oc_tokenizer(texts).to(device)
            return model.encode_text(tokens).float().cpu()

        return fwd_img, fwd_txt, emb_dim, "openclip"

    # transformers
    logger.info("Loading transformers %s", args.model)
    tmodel = AutoModel.from_pretrained(args.model).to(device).eval()
    tmodel.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model)
    siglip = getattr(tmodel.config, "model_type", "") == "siglip"
    if siglip:
        emb_dim = tmodel.config.vision_config.hidden_size
    else:
        emb_dim = tmodel.config.projection_dim

    @torch.no_grad()
    def fwd_img(pils):
        inputs = processor(images=pils, return_tensors="pt")
        v_out = tmodel.vision_model(pixel_values=inputs["pixel_values"].to(device))
        if siglip:
            return v_out.pooler_output.float().cpu()
        return tmodel.visual_projection(v_out.pooler_output).float().cpu()

    @torch.no_grad()
    def fwd_txt(texts):
        tok = processor(text=texts, return_tensors="pt",
                        padding="max_length" if siglip else True,
                        truncation=True, max_length=64 if siglip else 77)
        input_ids = tok["input_ids"].to(device)
        attn = tok.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        t_out = tmodel.text_model(input_ids=input_ids, attention_mask=attn)
        if siglip:
            return t_out.pooler_output.float().cpu()
        return tmodel.text_projection(t_out.pooler_output).float().cpu()

    return fwd_img, fwd_txt, emb_dim, ("siglip" if siglip else "clip")


def _collate(batch):
    batch = [b for b in batch if b is not None]
    return {
        "keys": [b["key"] for b in batch],
        "imgs": [b["img"] for b in batch],
        "txts": [b["txt"] for b in batch],
    }


def _run_extract(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    logger.info("Device: %s", device)

    fwd_img, fwd_txt, emb_dim, backend_kind = _load_model(args, device)
    logger.info("Embedding dim: %d (backend=%s)", emb_dim, backend_kind)

    image_dict: dict[str, torch.Tensor] = {
        str(k): v for k, v in load_state(cache_dir / "image_embeddings.pt").items()
    }
    text_dict: dict[str, torch.Tensor] = {
        str(k): v for k, v in load_state(cache_dir / "text_embeddings.pt").items()
    }
    keys_so_far: list[str] = list(image_dict.keys())
    already_n = len(keys_so_far)
    logger.info("Resuming from %d cached samples", already_n)

    logger.info("Streaming %s split=%s (workers=%d)", HF_DATASET, args.hf_split, args.num_workers)
    hf_ds = load_dataset(HF_DATASET, split=args.hf_split, streaming=True)
    # .map is lazy for streaming; the transform runs in DataLoader workers
    hf_ds = hf_ds.map(_transform)

    loader = DataLoader(
        hf_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=bool(args.num_workers),
    )

    n_processed = 0
    last_flush_at = 0
    t0 = time.time()
    last_log_at = 0
    # pixparse/cc3m-wds is a re-scrape of original CC3M (~3.3M) with dead URLs
    # removed. Actual train size is ~2.87M. Used only for ETA display.
    CC3M_TARGET = 2_870_000
    log_target = args.max_samples if args.max_samples else CC3M_TARGET

    def flush_checkpoint():
        save_state(image_dict, cache_dir / "image_embeddings.pt")
        save_state(text_dict, cache_dir / "text_embeddings.pt")
        # Write under the same split label we streamed from (train, validation, ...)
        # so downstream loaders pick up the correct split.
        with open(cache_dir / "splits.json", "w") as f:
            json.dump({args.hf_split: [[k, 0] for k in keys_so_far]}, f)

    @torch.no_grad()
    def forward_batch(batch):
        nonlocal n_processed
        keys = batch["keys"]
        imgs = batch["imgs"]
        txts = batch["txts"]
        if not keys:
            return

        # Filter out keys already in cache (resume support)
        mask = [k not in image_dict or k not in text_dict for k in keys]
        if not any(mask):
            return
        keys = [k for k, m in zip(keys, mask) if m]
        imgs = [x for x, m in zip(imgs, mask) if m]
        txts = [x for x, m in zip(txts, mask) if m]

        img_feats = fwd_img(imgs)
        txt_feats = fwd_txt(txts)

        for k, iv, tv in zip(keys, img_feats, txt_feats):
            image_dict[k] = iv.clone()
            text_dict[k] = tv.clone()
            keys_so_far.append(k)
        n_processed += len(keys)

    LOG_EVERY = max(1, int(args.flush_every // 4))  # progress log ~4× per flush

    def progress_line(tag: str) -> str:
        elapsed = time.time() - t0
        rate = n_processed / max(elapsed, 1e-6)
        remaining = max(0, log_target - n_processed)
        eta_sec = remaining / max(rate, 1e-6)
        return (f"[{tag}] n={n_processed}/{log_target} ({100*n_processed/max(log_target,1):.2f}%) "
                f"| rate={rate:.1f}/s | elapsed={elapsed/60:.1f}min | ETA={eta_sec/60:.1f}min "
                f"| wall_end={time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(time.time()+eta_sec))}")

    logger.info(progress_line("start"))
    # tqdm silenced in nohup; rely on periodic logger.info lines
    for batch in loader:
        forward_batch(batch)
        # Periodic progress log (every LOG_EVERY samples)
        if n_processed - last_log_at >= LOG_EVERY:
            logger.info(progress_line("prog"))
            last_log_at = n_processed
        if n_processed - last_flush_at >= args.flush_every:
            flush_checkpoint()
            logger.info(progress_line("flush"))
            last_flush_at = n_processed
        if args.max_samples and n_processed >= args.max_samples:
            break

    flush_checkpoint()
    logger.info(progress_line("done"))

    elapsed = time.time() - t0
    meta = {
        "clip_model": args.model,
        "dataset": HF_DATASET,
        "split": args.hf_split,
        "dim": emb_dim,
        "n_images": len(image_dict),
        "n_texts": len(text_dict),
        "elapsed_sec": round(elapsed, 1),
        "processed_this_run": int(n_processed),
        "avg_rate_samples_per_sec": round(n_processed / max(elapsed, 1e-6), 2),
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("meta.json: %s", meta)


def extract(
    *,
    model_cfg,
    cache_dir: str,
    splits: list[str] | None = None,
    batch_size: int = 256,
    device: str = "cuda",
    flush_every: int = 65536,
    max_samples: int = 0,
    num_workers: int = 4,
) -> None:
    """Programmatic entry: CC3M (pixparse/cc3m-wds) embedding extraction.

    ``splits`` is interpreted as the HF split list; only the first is used
    (CC3M is a single train/val pair; we forward ``splits[0]`` as ``hf_split``).
    """
    if splits is None:
        splits = ["train"]
    hf_split = splits[0]
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
        flush_every=flush_every, max_samples=max_samples,
        hf_split=hf_split, num_workers=num_workers,
    )
    os.makedirs(cache_dir, exist_ok=True)
    _run_extract(args)


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    _run_extract(args)


if __name__ == "__main__":
    main()

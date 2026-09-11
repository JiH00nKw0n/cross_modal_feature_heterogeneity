"""Do the weak matches cost anything?

The correlation distribution shows that a large share of matched pairs have weak
co-activation, which invites the follow-up question: if so many matches are
weak, is the alignment usable at all. This measures cross-modal retrieval while
keeping only the matches whose correlation clears a cutoff and discarding the
rest, at several cutoffs.

The comparison that would prove nothing is a random subset of coordinates of the
same size, because the high-confidence coordinates are also the frequently
firing ones and would win on activation mass alone. The control used here keeps
exactly the same coordinates and shuffles which text latent each one is paired
with. Same coordinates, same activation, wrong correspondence. Retrieval that
survives that shuffle was never measuring correspondence.

The retrieval protocol is the one Table 1 uses, from `src.eval.retrieval`:
image-to-text and text-to-image Recall@k on the COCO test split, with
pessimistic tie handling, so that a tie against the ground truth pushes the rank
down rather than up. Sparse latents leave many pairs tied at zero similarity and
an optimistic tie rule would report those as perfect retrieval. The only
departure from Table 1 is the coordinate mask: the latent vectors are restricted
to the kept coordinates before they are normalized and compared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.eval import eval_utils, retrieval
from src.rebuttal.common import (
    Setting,
    fmt,
    load_panel_or_raise,
    md_table,
    write_json,
    write_md,
)
from src.rebuttal.match_confidence import matched_correlation

logger = logging.getLogger(__name__)

#: Stem of the two files this analysis writes.
NAME = "confidence_ablation"

#: Correlation cutoffs, as in the paper's script. A matched pair is kept when
#: its co-activation correlation is at least the cutoff.
CUTOFFS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)

#: Fewer kept coordinates than this and retrieval is not scored, because the
#: recalls would be dominated by which handful of coordinates survived.
MIN_COORDINATES = 10

#: Split the retrieval runs on. Held out from SAE training in both settings.
EVAL_SPLIT = "test"

#: Rows scored per matrix multiplication, as in `src.eval.retrieval`.
CHUNK = 1024


def _ranks(z_img: torch.Tensor, z_txt: torch.Tensor, pair_img_idx: np.ndarray,
           gt_caps_per_img: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Pessimistic retrieval ranks both ways, the protocol `src.eval.retrieval` uses.

    Text to image: the rank of a caption's own image counts every image scoring
    greater than or equal to it, minus one. Image to text: the rank of an image
    is the best such rank over its ground-truth captions, so an image counts as
    retrieved when any one of its captions is found.
    """
    t2i = np.empty(z_txt.shape[0], dtype=np.int64)
    for s in range(0, z_txt.shape[0], CHUNK):
        scores = z_txt[s:s + CHUNK] @ z_img.T
        gt = pair_img_idx[s:s + CHUNK]
        gt_scores = scores[np.arange(len(gt)), gt]
        t2i[s:s + CHUNK] = ((scores >= gt_scores[:, None]).sum(dim=1) - 1).cpu().numpy()

    i2t = np.empty(z_img.shape[0], dtype=np.int64)
    for s in range(0, z_img.shape[0], CHUNK):
        scores = z_img[s:s + CHUNK] @ z_txt.T
        for row in range(scores.shape[0]):
            img_idx = s + row
            best = scores[row, gt_caps_per_img[img_idx]].max()
            i2t[img_idx] = int((scores[row] >= best).sum().item()) - 1
    return t2i, i2t


def _recalls(z_img: torch.Tensor, z_txt: torch.Tensor, pair_img_idx: np.ndarray,
             gt_caps_per_img: list[list[int]]) -> dict[str, dict[str, float]]:
    """Recall at 1, 5 and 10 in both directions, as fractions of 1."""
    t2i, i2t = _ranks(z_img, z_txt, pair_img_idx, gt_caps_per_img)
    return {"T2I": retrieval._recall_at_k(t2i), "I2T": retrieval._recall_at_k(i2t)}


def _unique_images(keys: list[str]) -> tuple[np.ndarray, torch.Tensor, list[list[int]]]:
    """Image rows, each caption's image position, and each image's captions.

    A COCO cache holds one row per image-caption pair and duplicates the image
    embedding across a photograph's captions, so the unique images are the first
    row of each image id, in the order the ids first appear.
    """
    image_ids = [eval_utils.coco_image_id(k) for k in keys]
    first_row: dict[str, int] = {}
    order: list[str] = []
    for row, iid in enumerate(image_ids):
        if iid not in first_row:
            first_row[iid] = row
            order.append(iid)
    id_to_idx = {iid: i for i, iid in enumerate(order)}
    pair_img_idx = np.array([id_to_idx[iid] for iid in image_ids], dtype=np.int64)
    img_rows = torch.as_tensor([first_row[iid] for iid in order], dtype=torch.long)
    gt_caps_per_img: list[list[int]] = [[] for _ in order]
    for cap_pos, img_idx in enumerate(pair_img_idx):
        gt_caps_per_img[int(img_idx)].append(cap_pos)
    return pair_img_idx, img_rows, gt_caps_per_img


def _table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        if "I2T" not in row:
            body.append([
                fmt(row["cutoff"], 1), fmt(row["n_coordinates"]),
                "not scored", "not scored", "not scored", "not scored", "not scored",
            ])
            continue
        body.append([
            fmt(row["cutoff"], 1),
            fmt(row["n_coordinates"]),
            fmt(100 * row["I2T"]["R@1"], 2),
            fmt(100 * row["I2T"]["R@5"], 2),
            fmt(100 * row["T2I"]["R@1"], 2),
            fmt(100 * row["T2I"]["R@5"], 2),
            fmt(100 * row["shuffled_partners"]["I2T"]["R@1"], 2),
        ])
    return md_table(
        ["matches kept, minimum co-activation correlation",
         "latent coordinates kept (count)",
         "image-to-text recall at 1 (percent)",
         "image-to-text recall at 5 (percent)",
         "text-to-image recall at 1 (percent)",
         "text-to-image recall at 5 (percent)",
         "image-to-text recall at 1 with the partners shuffled (percent)"],
        body,
    )


def _paragraphs(setting: Setting, payload: dict[str, Any]) -> list[str]:
    lead = (
        f"Setting {setting.tag}: {setting.title()}. Cross-modal retrieval is scored on "
        f"the COCO {payload['split']} split, {payload['n_images']:,} photographs and "
        f"{payload['n_captions']:,} captions, using the checkpoint the other analyses "
        "measure and the Hungarian assignment stored in its co-activation panel. That "
        f"assignment was built over {payload['n_samples']:,} pairs of the "
        f"{setting.split} split and holds {payload['n_matched_usable']:,} matched pairs "
        "whose two sides are both alive. For each cutoff, only the matched pairs whose "
        "co-activation correlation is at least the cutoff are kept, the text latent "
        "vector is reindexed so that a matched text latent lands on its image partner's "
        "column, both latent vectors are restricted to the kept coordinates and "
        "normalized to unit length, and the two sides are ranked against each other by "
        "cosine similarity."
    )
    protocol = (
        "Recall at k counts a caption as retrieved when its own photograph is inside "
        "the top k, and a photograph as retrieved when any one of its ground-truth "
        "captions is inside the top k. Ranking is pessimistic: a candidate tying with "
        "the ground truth is counted as beating it, so that the large tied blocks "
        "sparse latents produce are not read as perfect retrieval. This is the protocol "
        "the paper's main retrieval table uses; the coordinate mask is the only "
        "difference."
    )
    control = (
        "The last column is the control. It keeps exactly the same coordinates and "
        "permutes which text latent each kept image coordinate is paired with, using "
        f"seed {payload['shuffle_seed']}, so that the activation mass is unchanged and "
        "only the correspondence is destroyed."
    )
    return [lead, protocol, control]


def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        seed: int = 0, batch_size: int = 2048, **knobs: Any) -> dict[str, Any]:
    """Retrieval with only the confident matches kept, at several cutoffs.

    Writes `confidence_ablation.json` and `confidence_ablation.md` into
    `out_dir` and returns the payload. Skips the work when the json is already
    there.

    `seed` seeds the partner shuffle of the control column and `batch_size` the
    encoder passes. The pipeline's knobs (`tau`, `n_boot`, `null_seed`) are
    accepted and unused.
    """
    out_dir = Path(out_dir)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return json.loads(out_json.read_text())

    rng = np.random.default_rng(seed)
    dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")

    panel = load_panel_or_raise(setting.panel_path("img_txt"))
    perm = np.asarray(panel["perm"], dtype=np.int64)
    usable = np.asarray(panel["usable"], dtype=bool)
    matched_c = matched_correlation(panel)

    model = eval_utils.load_sae(setting.ckpt_a, "separated")
    ds = eval_utils.load_paired_split(setting.coco_cache, EVAL_SPLIT)
    pair_img_idx, img_rows, gt_caps_per_img = _unique_images(ds.keys)
    logger.info("[%s] COCO %s split: %d images, %d captions",
                NAME, EVAL_SPLIT, img_rows.numel(), len(ds))

    z_img_full = eval_utils.encode_image(model, ds.image[img_rows], "separated",
                                         dev, batch_size)
    z_txt_raw = eval_utils.encode_text(model, ds.text, "separated", dev,
                                       perm=None, batch_size=batch_size)
    z_txt_full = z_txt_raw[:, torch.as_tensor(perm, dtype=torch.long)]

    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        keep = usable & (matched_c >= cutoff)
        n_keep = int(keep.sum())
        row: dict[str, Any] = {"cutoff": float(cutoff), "n_coordinates": n_keep}
        if n_keep < MIN_COORDINATES:
            row["note"] = (f"fewer than {MIN_COORDINATES} coordinates survive this "
                           "cutoff, so retrieval was not scored")
            rows.append(row)
            logger.info("[%s] cutoff %.1f: %d coordinates, not scored",
                        NAME, cutoff, n_keep)
            continue
        mask = torch.as_tensor(keep, dtype=torch.bool)
        z_img = eval_utils.normalize_rows(z_img_full[:, mask])
        z_txt = eval_utils.normalize_rows(z_txt_full[:, mask])
        row.update(_recalls(z_img, z_txt, pair_img_idx, gt_caps_per_img))

        # Same coordinates, correspondence destroyed.
        idx = np.where(keep)[0]
        shuffled = perm.copy()
        shuffled[idx] = perm[rng.permutation(idx)]
        z_txt_shuf = z_txt_raw[:, torch.as_tensor(shuffled, dtype=torch.long)]
        z_txt_shuf = eval_utils.normalize_rows(z_txt_shuf[:, mask])
        row["shuffled_partners"] = _recalls(z_img, z_txt_shuf, pair_img_idx,
                                            gt_caps_per_img)
        rows.append(row)
        logger.info("[%s] cutoff %.1f: %d coordinates, I2T R@1 %.4f, shuffled %.4f",
                    NAME, cutoff, n_keep, row["I2T"]["R@1"],
                    row["shuffled_partners"]["I2T"]["R@1"])

    payload = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "panel": str(setting.panel_path("img_txt")),
        "eval_cache": str(setting.coco_cache),
        "split": EVAL_SPLIT,
        "n_images": int(img_rows.numel()),
        "n_captions": int(len(ds)),
        "n_samples": int(panel["n_samples"]),
        "n_matched_usable": int(usable.sum()),
        "shuffle_seed": int(seed),
        "min_coordinates_to_score": MIN_COORDINATES,
        "by_cutoff": rows,
    }
    write_json(out_json, payload)
    write_md(out_dir / f"{NAME}.md",
             f"Retrieval with only the confident matches kept, {setting.tag}",
             _paragraphs(setting, payload),
             [("Retrieval by correlation cutoff", _table(rows))])
    logger.info("[%s] wrote %s", NAME, out_dir / f"{NAME}.md")
    return payload


__all__ = ["run", "CUTOFFS", "MIN_COORDINATES", "EVAL_SPLIT", "NAME"]

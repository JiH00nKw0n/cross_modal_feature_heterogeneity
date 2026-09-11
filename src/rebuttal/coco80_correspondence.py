"""Do two matched latents stand for the same object, judged from outside the model?

The paper pairs an image latent with a text latent because the two fire on the
same inputs. Firing together has two possible causes, and co-activation cannot
tell them apart: the two latents may mean the same thing, or they may mean two
different things that happen to appear in the same photographs. A chair and a
dining table are the second case. Arguing from the correlation that the
correlation found a concept correspondence would be circular, so the judgement
has to come from outside the model. COCO's hand-drawn object annotations supply
it.

The test. Fix one of the 80 object categories. On the image side, find the
coordinate whose activation best separates photographs containing that object
from photographs that do not. On the text side, find the coordinate whose
activation best separates captions of such photographs from other captions.
Neither search looks at the other modality, neither search looks at the
correlation matrix, and the two searches read disjoint halves of the
photographs, so the only thing that can link their answers is the concept. Only
then is the learned permutation consulted, with one question: are those two
coordinates a matched pair? The share of categories where they are is the
agreement at rank 1.

Separation is measured by the area under the ROC curve, which is the probability
that the coordinate is more active on a positive sample than on a negative one,
with a tie counting as one half. Two simpler criteria were rejected. The mean
activation on the positives picks whichever coordinate fires most often, which
is a measure of firing rate rather than of concept. A t statistic diverges on a
coordinate that fires on three positives and on nothing else, which at a firing
rate under 1 percent is a common situation rather than a rare one. The area
under the curve is bounded in [0, 1], is unaffected by activation scale, and is
computed here exactly rather than by sampling. A coordinate must additionally
fire on at least `min_support` of a category's positives, 5 percent by default,
so that an almost never firing coordinate cannot win a category by accident.

Five references come with the headline number, because a bare agreement rate
cannot be read on its own: the rate expected by chance from the number of
candidate coordinates; a random permutation between the same two sets of picks;
a shuffle of the category labels, which exposes the degenerate case where a few
busy coordinates win everything; and the rate at which the image side agrees
with itself across the two halves of the photographs, which is the highest
agreement any cross-modal test of this model could reach.

Arms. Every method whose coordinates carry a concept identity is put through the
identical test. The modality-specific model with its learned permutation is the
paper's method. The same model with the identity permutation keeps two separate
dictionaries and removes only the learned map between them. The shared,
iso-energy and group-sparse arms train one dictionary for both modalities, so
their correspondence is the identity by construction and what is being tested is
their alignment loss.

Output written under the setting's out_dir:

    coco80_correspondence.json   every number, per arm and per label variant
    coco80_correspondence.md     the tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.cache_io import load_stacked
from src.data.paired_dataset import normalize_np
from src.eval.eval_utils import load_sae
from src.rebuttal.coco80_labels import (
    DEFAULT_MIN_COUNT,
    VARIANT_TITLES,
    VARIANTS,
    Coco80Labels,
    load_or_build,
    support_phrase,
    usable_categories,
)
from src.rebuttal.coco80_synonyms import COCO_80
from src.rebuttal.common import (
    Setting,
    fmt,
    load_panel_or_raise,
    md_table,
    pct,
    write_json,
    write_md,
)

logger = logging.getLogger(__name__)

#: Name of this analysis, and the stem of every file it writes.
NAME = "coco80_correspondence"

#: Smallest share of a category's positives a coordinate has to fire on before
#: its separation score is allowed to count.
DEFAULT_MIN_SUPPORT = 0.05

#: Random permutations drawn for the control.
DEFAULT_N_NULL = 1000

#: Embeddings encoded per forward pass.
DEFAULT_BATCH_SIZE = 4096

#: Arm name to the method label `src.eval.eval_utils.load_sae` expects.
ARM_METHOD = {
    "ours": "separated",
    "noalign": "separated",
    "shared": "shared",
    "iso_align": "aux",
    "group_sparse": "aux",
}

#: Arm name as a table cell, spelled out rather than abbreviated.
ARM_TITLES = {
    "ours": "Modality-specific dictionaries with the learned permutation",
    "noalign": "Modality-specific dictionaries with no permutation (identity)",
    "shared": "One shared dictionary",
    "iso_align": "One shared dictionary with the iso-energy alignment loss",
    "group_sparse": "One shared dictionary with the group-sparse loss",
}

#: Order the arms are reported in.
ARM_ORDER = ("ours", "iso_align", "shared", "group_sparse", "noalign")


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def resolve_device(device: str | torch.device) -> torch.device:
    """The device to encode on, falling back to the CPU when one is absent."""
    if isinstance(device, torch.device):
        return device
    name = str(device)
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("[%s] no CUDA device available, encoding on the CPU", NAME)
        return torch.device("cpu")
    if name.startswith("mps") and not torch.backends.mps.is_available():
        logger.warning("[%s] no MPS device available, encoding on the CPU", NAME)
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def sparse_latents(
    sae, table: np.ndarray, rows: np.ndarray, *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode selected rows of an embedding table and keep only the non-zeros.

    A top-k sparse autoencoder leaves all but k of its thousands of coordinates
    exactly zero, so the dense matrix would be almost entirely zeros. The three
    arrays returned hold the same information: for each non-zero activation, the
    index of the sample within `rows`, the index of the coordinate, and the
    activation value. Embeddings are L2-normalized on the way in, which is how
    the autoencoders were trained.
    """
    dev = resolve_device(device)
    sae = sae.to(dev).eval()
    samples: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, int(rows.shape[0]), int(batch_size)):
        block = rows[start:start + int(batch_size)]
        x = torch.from_numpy(normalize_np(table[block])).to(dev)
        z = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True)
        z = z.dense_latents.squeeze(1).float()
        nz = torch.nonzero(z, as_tuple=False)
        samples.append((nz[:, 0] + start).cpu().numpy())
        coords.append(nz[:, 1].cpu().numpy())
        values.append(z[nz[:, 0], nz[:, 1]].cpu().numpy())
    if not samples:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=np.float32)
    return (np.concatenate(samples).astype(np.int64),
            np.concatenate(coords).astype(np.int64),
            np.concatenate(values).astype(np.float64))


# --------------------------------------------------------------------------- #
# Separation
# --------------------------------------------------------------------------- #
def auc_matrix(
    samp: np.ndarray, lat: np.ndarray, val: np.ndarray,
    labels: np.ndarray, n_samples: int, n_latents: int, min_support: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Area under the ROC curve for every coordinate and category, plus support.

    Exact, not sampled. Every sample on which a coordinate does not fire ties at
    the bottom of that coordinate's ranking, and the contribution of that tied
    block has a closed form, so only the non-zero activations need sorting. Ties
    among equal non-zero activations are handled the same way, each tied pair
    counting as one half.

    Returns a (n_latents, n_categories) matrix of scores and a matrix of the
    share of each category's positives the coordinate fires on. A score whose
    support is below `min_support` is set to negative infinity, which takes it
    out of every maximum taken downstream.
    """
    n_cat = int(labels.shape[1])
    n_pos = labels.sum(axis=0).astype(np.float64)
    n_neg = float(n_samples) - n_pos

    auc = np.full((n_latents, n_cat), 0.5, dtype=np.float64)
    support = np.zeros((n_latents, n_cat), dtype=np.float64)

    if samp.size:
        order = np.argsort(lat, kind="stable")
        lat_s, samp_s, val_s = lat[order], samp[order], val[order]
        bounds = np.searchsorted(lat_s, np.arange(n_latents + 1))
        for coord in range(n_latents):
            lo, hi = int(bounds[coord]), int(bounds[coord + 1])
            if hi == lo:
                continue
            v = val_s[lo:hi]
            o = np.argsort(v, kind="stable")            # ascending activation
            v_sorted = v[o]
            pos = labels[samp_s[lo:hi][o]]              # (S, n_cat) bool
            neg = ~pos

            cum_incl = np.cumsum(neg, axis=0).astype(np.float64)
            starts = np.empty(v_sorted.shape[0], dtype=bool)
            starts[0] = True
            starts[1:] = v_sorted[1:] != v_sorted[:-1]
            group_of = np.cumsum(starts) - 1
            last = np.flatnonzero(np.append(starts[1:], True))
            through = cum_incl[last]
            before = np.vstack([np.zeros((1, n_cat)), through[:-1]])
            tied = through - before

            # negatives strictly below each positive, plus half of those tied
            u_nz = (pos * (before[group_of] + 0.5 * tied[group_of])).sum(axis=0)

            n_pos_nz = pos.sum(axis=0).astype(np.float64)
            n_neg_nz = neg.sum(axis=0).astype(np.float64)
            n_neg_zero = n_neg - n_neg_nz
            n_pos_zero = n_pos - n_pos_nz

            # every firing positive beats every non-firing negative, and the
            # non-firing samples all tie with one another
            u = u_nz + n_pos_nz * n_neg_zero + 0.5 * n_pos_zero * n_neg_zero
            auc[coord] = u / np.maximum(n_pos * n_neg, 1.0)
            support[coord] = n_pos_nz / np.maximum(n_pos, 1.0)

    return np.where(support >= float(min_support), auc, -np.inf), support


def rank_of(values: np.ndarray, target: int) -> int:
    """Rank of `target` when `values` is ordered from largest to smallest, from 1.

    Used in both directions, because where the image side's pick lands in the
    text side's ranking and where the text side's pick lands in the image side's
    ranking are different questions. At rank 1 they describe the same event: the
    two sides chose a matched pair.
    """
    return int((values > values[target]).sum()) + 1


def side_scores(samp: np.ndarray, lat: np.ndarray, val: np.ndarray,
                 labels: np.ndarray, mask: np.ndarray, n_latents: int,
                 min_support: float) -> tuple[np.ndarray, np.ndarray]:
    """Separation scores restricted to the samples `mask` selects."""
    if not mask.any():
        return (np.full((n_latents, labels.shape[1]), -np.inf),
                np.zeros((n_latents, labels.shape[1])))
    keep = mask[samp]
    remap = np.cumsum(mask) - 1
    return auc_matrix(remap[samp[keep]], lat[keep], val[keep],
                      labels[mask], int(mask.sum()), n_latents, min_support)


# --------------------------------------------------------------------------- #
# Candidate coordinates, per arm
# --------------------------------------------------------------------------- #
def _candidates(
    arm: str, setting: Setting, image_coords_fired: np.ndarray,
    text_coords_fired: np.ndarray, n_latents: int,
) -> dict[str, Any]:
    """The coordinate pairs an arm puts forward, and how they were chosen.

    For the two modality-specific arms a candidate is an image coordinate that
    the panel marks usable, meaning alive on both sides of the assignment, and
    its partner is the coordinate the permutation assigns it. The arm with no
    permutation uses the identity instead, so coordinate i on the image side is
    read against coordinate i on the text side. For the single-dictionary arms
    there is nothing to permute, and a candidate is any coordinate that fires at
    least once on the photographs and at least once on the captions.
    """
    if arm in ("ours", "noalign"):
        panel = load_panel_or_raise(setting.panel_img_txt)
        alive_image = np.asarray(panel["alive_image"], dtype=bool)
        alive_text = np.asarray(panel["alive_text"], dtype=bool)
        if arm == "noalign":
            usable = alive_image & alive_text
            image_coords = np.where(usable)[0]
            return {
                "image_coords": image_coords,
                "text_coords": image_coords.copy(),
                "rule": ("coordinates alive on both sides, read against the same "
                         "coordinate index on the other side"),
                "n_alive_image": int(alive_image.sum()),
                "n_alive_text": int(alive_text.sum()),
            }
        perm = np.asarray(panel["perm"], dtype=np.int64)
        usable = np.asarray(panel["usable"], dtype=bool)
        image_coords = np.where(usable)[0]
        return {
            "image_coords": image_coords,
            "text_coords": perm[image_coords],
            "rule": ("image coordinates the panel marks usable, each read "
                     "against the text coordinate the permutation assigns it"),
            "n_alive_image": int(alive_image.sum()),
            "n_alive_text": int(alive_text.sum()),
        }

    fires_image = np.zeros(n_latents, dtype=bool)
    fires_image[image_coords_fired] = True
    fires_text = np.zeros(n_latents, dtype=bool)
    fires_text[text_coords_fired] = True
    both = np.where(fires_image & fires_text)[0]
    return {
        "image_coords": both,
        "text_coords": both.copy(),
        "rule": ("coordinates of the single dictionary that fire at least once "
                 "on the photographs and at least once on the captions"),
        "n_alive_image": int(fires_image.sum()),
        "n_alive_text": int(fires_text.sum()),
    }


# --------------------------------------------------------------------------- #
# One arm, one label variant
# --------------------------------------------------------------------------- #
def _score_variant(
    *,
    arm: str,
    variant: str,
    labels: Coco80Labels,
    candidates: dict[str, Any],
    image_triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    text_triples: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_latents: int,
    min_count: int,
    min_support: float,
    n_null: int,
    seed: int,
) -> dict[str, Any]:
    """Run the agreement test for one arm on one label variant."""
    rng = np.random.default_rng(seed)
    Y = labels.matrix(variant)
    Y_cap = Y[labels.caption_owner]

    image_half_mask = labels.half == 0
    caption_in_text_half = (labels.half == 1)[labels.caption_owner]

    i_samp, i_lat, i_val = image_triples
    t_samp, t_lat, t_val = text_triples

    auc_image, _ = side_scores(i_samp, i_lat, i_val, Y, image_half_mask,
                               n_latents, min_support)
    auc_text, _ = side_scores(t_samp, t_lat, t_val, Y_cap, caption_in_text_half,
                              n_latents, min_support)
    # The ceiling repeats each side's own choice on the other half of the same
    # modality, at the same sample size.
    auc_image_other, _ = side_scores(i_samp, i_lat, i_val, Y, ~image_half_mask,
                                     n_latents, min_support)
    auc_text_other, _ = side_scores(t_samp, t_lat, t_val, Y_cap, ~caption_in_text_half,
                                    n_latents, min_support)

    image_coords = candidates["image_coords"]
    text_coords = candidates["text_coords"]
    n_candidates = int(image_coords.shape[0])

    scored = usable_categories(labels, variant, min_count)
    counts_image = Y[image_half_mask].sum(axis=0)
    counts_text = Y_cap[caption_in_text_half].sum(axis=0)

    rows: list[dict[str, Any]] = []
    rank_in_text: list[int] = []
    rank_in_image: list[int] = []
    agree_image: list[int] = []
    agree_text: list[int] = []
    picks_image: list[int] = []
    picks_text: list[int] = []

    for c in scored.tolist():
        a_image = auc_image[image_coords, c] if n_candidates else np.zeros(0)
        a_text = auc_text[text_coords, c] if n_candidates else np.zeros(0)
        if not np.isfinite(a_image).any() or not np.isfinite(a_text).any():
            continue
        i_star = int(np.argmax(a_image))
        j_star = int(np.argmax(a_text))
        picks_image.append(i_star)
        picks_text.append(j_star)

        r_text = rank_of(a_text, i_star)
        r_image = rank_of(a_image, j_star)
        rank_in_text.append(r_text)
        rank_in_image.append(r_image)

        other_image = auc_image_other[image_coords, c]
        other_text = auc_text_other[text_coords, c]
        agree_image.append(int(np.isfinite(other_image).any()
                               and int(np.argmax(other_image)) == i_star))
        agree_text.append(int(np.isfinite(other_text).any()
                              and int(np.argmax(other_text)) == j_star))

        rows.append({
            "category": COCO_80[c],
            "positive_photographs": int(counts_image[c]),
            "positive_captions": int(counts_text[c]),
            "image_coordinate": int(image_coords[i_star]),
            "text_coordinate_via_permutation": int(text_coords[i_star]),
            "text_coordinate_chosen": int(text_coords[j_star]),
            "rank_in_text": r_text,
            "rank_in_image": r_image,
            "separation_image": float(a_image[i_star]),
            "separation_text": float(a_text[j_star]),
            "image_agrees_with_its_other_half": agree_image[-1],
            "text_agrees_with_its_other_half": agree_text[-1],
        })

    n = len(rows)
    r_text_arr = np.asarray(rank_in_text, dtype=np.float64)
    r_image_arr = np.asarray(rank_in_image, dtype=np.float64)

    def at_k(ranks: np.ndarray, k: int) -> float:
        return float(np.mean(ranks <= k)) if ranks.size else float("nan")

    def summary(ranks: np.ndarray) -> dict[str, float]:
        return {
            "top1": at_k(ranks, 1),
            "top5": at_k(ranks, 5),
            "top10": at_k(ranks, 10),
            "median_rank": float(np.median(ranks)) if ranks.size else float("nan"),
            "mrr": float(np.mean(1.0 / ranks)) if ranks.size else float("nan"),
        }

    i_star_arr = np.asarray(picks_image, dtype=np.int64)
    j_star_arr = np.asarray(picks_text, dtype=np.int64)
    if n and n_candidates:
        null_hits = np.array([
            float(np.mean(rng.permutation(n_candidates)[i_star_arr] == j_star_arr))
            for _ in range(int(n_null))
        ])
        label_shuffle = float(np.mean(i_star_arr == j_star_arr[rng.permutation(n)]))
    else:
        null_hits = np.zeros(0)
        label_shuffle = float("nan")

    agree_at_1 = at_k(r_text_arr, 1)
    controls = {
        "chance_hit_at_1": (1.0 / n_candidates) if n_candidates else float("nan"),
        "random_permutation_hit_at_1_mean": (float(null_hits.mean())
                                             if null_hits.size else float("nan")),
        "random_permutation_hit_at_1_p95": (float(np.percentile(null_hits, 95))
                                            if null_hits.size else float("nan")),
        "p_value_vs_random_permutation": (float(np.mean(null_hits >= agree_at_1))
                                          if null_hits.size else float("nan")),
        "label_shuffle_hit_at_1": label_shuffle,
        "image_self_agreement": float(np.mean(agree_image)) if n else float("nan"),
        "text_self_agreement": float(np.mean(agree_text)) if n else float("nan"),
    }

    return {
        "arm": arm,
        "variant": variant,
        "n_categories": n,
        "n_candidate_coordinates": n_candidates,
        "candidate_rule": candidates["rule"],
        "n_alive_image": candidates["n_alive_image"],
        "n_alive_text": candidates["n_alive_text"],
        "min_count": int(min_count),
        "min_support": float(min_support),
        "agree_at_1": agree_at_1,
        "image_pick_in_text_ranking": summary(r_text_arr),
        "text_pick_in_image_ranking": summary(r_image_arr),
        "controls": controls,
        "distinct_image_coordinates_chosen": len({r["image_coordinate"]
                                                  for r in rows}),
        "categories_dropped": [COCO_80[c] for c in range(80)
                               if c not in set(scored.tolist())],
        "per_category": rows,
    }


def _score_arm(
    *,
    arm: str,
    setting: Setting,
    labels: Coco80Labels,
    device: str | torch.device,
    min_count: int,
    min_support: float,
    n_null: int,
    seed: int,
    batch_size: int,
    encoded: dict[tuple[str, str], tuple[Any, ...]] | None = None,
) -> dict[str, Any]:
    """Encode one arm's model once, then score both label variants with it.

    `encoded` memoizes the activations by checkpoint and method, so that the two
    arms that read the same checkpoint, the one with the learned permutation and
    the one with the identity, encode the data once between them.
    """
    ckpt = setting.ckpt_a if arm in ("ours", "noalign") else setting.baselines[arm]
    method = ARM_METHOD[arm]
    key = (str(ckpt), method)
    if encoded is None:
        encoded = {}
    if key in encoded:
        n_latents, image_triples, text_triples = encoded[key]
        logger.info("[%s] arm %s: reusing the activations of %s", NAME, arm, ckpt)
    else:
        logger.info("[%s] arm %s: loading %s as method %r", NAME, arm, ckpt, method)
        model = load_sae(ckpt, method)
        single = method not in ("separated", "ours")
        sae_image = model if single else model.image_sae
        sae_text = model if single else model.text_sae
        n_latents = int(sae_image.latent_size)

        cache = load_stacked(setting.coco_cache, mmap=True)
        logger.info("[%s] arm %s: encoding %d photographs and %d captions",
                    NAME, arm, labels.n_images, labels.n_captions)
        image_triples = sparse_latents(sae_image, cache["image"], labels.image_rows,
                                       batch_size=batch_size, device=device)
        text_triples = sparse_latents(sae_text, cache["text"], labels.caption_rows,
                                      batch_size=batch_size, device=device)
        encoded[key] = (n_latents, image_triples, text_triples)

    candidates = _candidates(arm, setting, np.unique(image_triples[1]),
                             np.unique(text_triples[1]), n_latents)
    logger.info("[%s] arm %s: %d candidate coordinate pairs",
                NAME, arm, int(candidates["image_coords"].shape[0]))

    out: dict[str, Any] = {"checkpoint": str(ckpt), "method": method,
                           "latents_per_side": n_latents, "variants": {}}
    for variant in VARIANTS:
        out["variants"][variant] = _score_variant(
            arm=arm, variant=variant, labels=labels, candidates=candidates,
            image_triples=image_triples, text_triples=text_triples,
            n_latents=n_latents, min_count=min_count, min_support=min_support,
            n_null=n_null, seed=seed,
        )
        got = out["variants"][variant]
        logger.info("[%s] arm %s, %s: agreement at rank 1 = %s over %d categories",
                    NAME, arm, variant, pct(got["agree_at_1"]), got["n_categories"])
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _direction_table(entry: dict[str, Any]) -> str:
    rows = []
    for key, label in (("image_pick_in_text_ranking",
                        "The image side's pick, ranked by the text side"),
                       ("text_pick_in_image_ranking",
                        "The text side's pick, ranked by the image side")):
        d = entry[key]
        rows.append([
            label,
            pct(d["top1"]),
            pct(d["top5"]),
            pct(d["top10"]),
            f"{fmt(d['median_rank'], 0)} of {entry['n_candidate_coordinates']:,}",
            fmt(d["mrr"]),
        ])
    return md_table(
        ["Which side picked and which side ranked",
         "Share of categories where the pick ranks 1st",
         "Share of categories where it ranks in the top 5",
         "Share of categories where it ranks in the top 10",
         "Median rank of the pick",
         "Mean reciprocal rank (1.0 is perfect)"],
        rows,
    )


def _reference_table(entry: dict[str, Any], baselines: dict[str, dict[str, Any]]) -> str:
    c = entry["controls"]
    rows = [[ARM_TITLES["ours"], pct(entry["agree_at_1"])]]
    for arm in ARM_ORDER:
        if arm == "ours" or arm not in baselines:
            continue
        rows.append([ARM_TITLES[arm], pct(baselines[arm]["agree_at_1"])])
    rows += [
        ["A random permutation between the same two sets of picks",
         pct(c["random_permutation_hit_at_1_mean"])],
        ["The category labels shuffled across the same picks",
         pct(c["label_shuffle_hit_at_1"])],
        [f"Chance, one coordinate out of {entry['n_candidate_coordinates']:,}",
         pct(c["chance_hit_at_1"])],
        [("The image side against the other half of the photographs "
          "(the highest agreement this test could reach)"),
         pct(c["image_self_agreement"])],
        ["The text side against the other half of the captions",
         pct(c["text_self_agreement"])],
    ]
    return md_table(
        ["What the image side's pick was checked against",
         "Share of categories where the two picks are a matched pair"],
        rows,
    )


def _arms_table(arms: dict[str, dict[str, Any]], variant: str) -> str:
    rows = []
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        e = arms[arm]["variants"][variant]
        d = e["image_pick_in_text_ranking"]
        rows.append([
            ARM_TITLES[arm],
            pct(e["agree_at_1"]),
            pct(d["top5"]),
            pct(d["top10"]),
            fmt(d["mrr"]),
            f"{e['n_candidate_coordinates']:,}",
            pct(e["controls"]["image_self_agreement"]),
            pct(e["controls"]["chance_hit_at_1"]),
        ])
    return md_table(
        ["Method",
         "Share of categories where the two picks are a matched pair",
         "Share where the image pick ranks in the text side's top 5",
         "Share where it ranks in the top 10",
         "Mean reciprocal rank (1.0 is perfect)",
         "Candidate coordinate pairs",
         "Image side against the other half of the photographs (ceiling)",
         "Chance rate for this many candidates"],
        rows,
    )


def _sensitivity_table(arms: dict[str, dict[str, Any]]) -> str:
    rows = []
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        a = arms[arm]["variants"]["area_filtered"]
        b = arms[arm]["variants"]["no_area"]
        rows.append([
            ARM_TITLES[arm],
            f"{a['n_categories']}",
            pct(a["agree_at_1"]),
            f"{b['n_categories']}",
            pct(b["agree_at_1"]),
        ])
    return md_table(
        ["Method",
         "Categories scored when the object must cover 5 percent of the frame",
         "Agreement at rank 1 under that condition",
         "Categories scored with no area condition",
         "Agreement at rank 1 with no area condition"],
        rows,
    )


def _write_report(setting: Setting, out_dir: Path, labels: Coco80Labels,
                  arms: dict[str, dict[str, Any]], min_support: float,
                  min_count: int, n_null: int) -> Path:
    head = arms["ours"]["variants"]["area_filtered"]
    head_no_area = arms["ours"]["variants"]["no_area"]
    c = head["controls"]

    intro = (
        f"Whether a matched pair of latents stands for the same object, judged "
        f"against COCO's hand-drawn object annotations. Measured on the "
        f"{labels.split} split of the COCO embedding cache at "
        f"{setting.coco_cache}: {labels.n_images:,} photographs and "
        f"{labels.n_captions:,} captions, split into two halves by the md5 of the "
        f"image id so that the image side and the text side never read the same "
        f"photograph. Of the 80 object categories, {head['n_categories']} have "
        f"{min_count} or more positives on each half separately and are scored; "
        f"the others are reported below as dropped. The model measured is "
        f"{setting.ckpt_a}. Its setting is {setting.title()}."
    )
    method = (
        f"For each category the image side ranks every candidate coordinate by "
        f"how well its activation separates photographs containing the object "
        f"from photographs that do not, and the text side ranks the same "
        f"candidates by how well they separate captions of such photographs from "
        f"other captions. Separation is the area under the ROC curve, computed "
        f"exactly, with a tie counting as one half, and "
        f"{support_phrase(min_support)}. Neither "
        f"side sees the other's data and neither side sees the correlation "
        f"matrix. The learned permutation enters only afterwards, to ask whether "
        f"the two picks are a matched pair. There are "
        f"{head['n_candidate_coordinates']:,} candidate coordinate pairs, chosen "
        f"as {head['candidate_rule']}."
    )
    result = (
        f"Agreement at rank 1 is {pct(head['agree_at_1'])} of the "
        f"{head['n_categories']} scored categories. Against "
        f"{n_null:,} random permutations of the same picks the p-value is "
        f"{fmt(c['p_value_vs_random_permutation'], 4)}. Those "
        f"{head['n_categories']} categories drew "
        f"{head['distinct_image_coordinates_chosen']} distinct image "
        f"coordinates between them."
    )
    dropped = head["categories_dropped"]
    dropped_line = (
        f"The {len(dropped)} categories with too few positives to score, under "
        f"the 5 percent area condition: {', '.join(dropped)}."
        if dropped else "Every one of the 80 categories had enough positives to score."
    )
    area_line = (
        f"Removing the area condition takes the scored categories from "
        f"{head['n_categories']} to {head_no_area['n_categories']} and the "
        f"agreement at rank 1 from {pct(head['agree_at_1'])} to "
        f"{pct(head_no_area['agree_at_1'])}, while the ceiling moves from "
        f"{pct(c['image_self_agreement'])} to "
        f"{pct(head_no_area['controls']['image_self_agreement'])}."
    )

    tables: list[tuple[str, str]] = [
        ((f"Where each side's pick lands in the other side's ranking "
          f"({VARIANT_TITLES['area_filtered']})"), _direction_table(head)),
        ("Agreement at rank 1, against every reference",
         _reference_table(head, {a: arms[a]["variants"]["area_filtered"]
                                 for a in arms if a != "ours"})),
    ]
    if len(arms) > 1:
        tables.append((("Every alignment arm on the identical test "
                        f"({VARIANT_TITLES['area_filtered']})"),
                       _arms_table(arms, "area_filtered")))
    tables.append(("Sensitivity to the area condition", _sensitivity_table(arms)))

    return write_md(
        Path(out_dir) / f"{NAME}.md",
        f"COCO-80 concept agreement, {setting.tag}",
        [intro, method, result, dropped_line, area_line],
        tables,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(setting: Setting, *, out_dir: str | Path, device: str = "cpu",
        **knobs: Any) -> dict[str, Any]:
    """Run the COCO-80 agreement test for every arm of one setting.

    Writes `<out_dir>/coco80_correspondence.json` and
    `<out_dir>/coco80_correspondence.md`, and skips both when the json exists.

    Arms. The modality-specific model with its learned permutation is always
    measured. Every other arm has to be named in `setting.baselines`, which for
    the CC3M setting holds the shared, iso-energy and group-sparse checkpoints
    plus "noalign", the same modality-specific model scored with the identity
    permutation, and which is empty for the COCO setting.

    Knobs, all optional: `min_support` (default 0.05), `min_count` (default 50),
    `n_null` (default 1000), `seed` (default 0), `batch_size` (default 4096),
    plus everything `coco80_labels.load_or_build` takes. Any other keyword,
    including the `tau`, `n_boot` and `null_seed` the pipeline always passes, is
    ignored.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{NAME}.json"
    if out_json.exists():
        logger.info("[%s][skip] %s exists", NAME, out_json)
        return json.loads(out_json.read_text())

    min_count = int(knobs.get("min_count", DEFAULT_MIN_COUNT))
    min_support = float(knobs.get("min_support", DEFAULT_MIN_SUPPORT))
    n_null = int(knobs.get("n_null", DEFAULT_N_NULL))
    seed = int(knobs.get("seed", 0))
    batch_size = int(knobs.get("batch_size", DEFAULT_BATCH_SIZE))

    labels = load_or_build(
        setting, out_dir,
        area_frac=float(knobs.get("area_frac", 0.05)),
        min_count=min_count,
        coco_split=str(knobs.get("coco_split", "test")),
        annotations_dir=knobs.get("annotations_dir", "cache/coco_annotations"),
        instances_file=str(knobs.get("instances_file", "instances_val2014.json")),
    )

    # "ours" is always measured; every other arm has to be named in
    # setting.baselines, which is empty for the COCO setting whose pipeline
    # trains the modality-specific method alone.
    wanted = {"ours", *setting.baselines}
    unknown = sorted(wanted - set(ARM_ORDER))
    if unknown:
        logger.warning("[%s] no test is defined for the baseline arms %s; "
                       "they are not measured", NAME, unknown)
        wanted -= set(unknown)
    encoded: dict[tuple[str, str], tuple[Any, ...]] = {}
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        if arm not in wanted:
            continue
        arms[arm] = _score_arm(
            arm=arm, setting=setting, labels=labels, device=device,
            min_count=min_count, min_support=min_support, n_null=n_null,
            seed=seed, batch_size=batch_size, encoded=encoded,
        )

    payload = {
        "analysis": NAME,
        "setting": setting.as_dict(),
        "labels": {
            "split": labels.split,
            "n_photographs": labels.n_images,
            "n_captions": labels.n_captions,
            "area_frac_threshold": labels.area_frac_threshold,
            "instances_path": labels.instances_path,
        },
        "min_count": min_count,
        "min_support": min_support,
        "n_null_permutations": n_null,
        "seed": seed,
        "arms": arms,
    }
    write_json(out_json, payload)
    _write_report(setting, out_dir, labels, arms, min_support, min_count, n_null)
    logger.info("[%s] wrote %s", NAME, out_json)
    return payload


__all__ = [
    "ARM_METHOD",
    "ARM_ORDER",
    "ARM_TITLES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_N_NULL",
    "NAME",
    "auc_matrix",
    "rank_of",
    "resolve_device",
    "run",
    "side_scores",
    "sparse_latents",
]

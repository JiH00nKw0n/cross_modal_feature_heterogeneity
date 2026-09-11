# Same Concept, Different Directions: Cross-Modal Feature Heterogeneity in Sparse Autoencoders

Reproducible code for the paper's four experimental deliverables: two synthetic
sparse-autoencoder sweeps, the multi-model decoder-cosine density figure
(Figure 2), and the CC3M-trained downstream comparison of five methods
(Table 1).

Everything runs through one entry point, `python run.py <config.yaml>`, with one
YAML per experiment.

---

## Setup

### Docker

```bash
bash scripts/docker_build.sh
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e CONFIG=configs/post_rebuttal/clip_b32.yaml \
  -v $PWD/cache:/app/repo/cache -v $PWD/outputs:/app/repo/outputs \
  vlm-sae
```

The container reads `CONFIG` (default `configs/post_rebuttal/clip_b32.yaml`,
which produces Table 1, Figure 2 and every rebuttal analysis in one run) and
`STAGE` (default `all`) from the environment, and runs
`python run.py "$CONFIG" --stage "$STAGE"`.
`HF_TOKEN` is required for the ImageNet-1K extraction, which is a gated dataset.

The mounted `cache/` directory holds two things, so size the volume for both.
`cache/hf` is the Hugging Face download cache, which `HF_HOME` points at so
that the downloads outlive `docker run --rm` and an interrupted extraction does
not fetch them again; COCO alone is about 20 GB there, since it is loaded
without streaming. Beside it sit the embedding caches this code writes, which
are roughly 2 GB for COCO, 13 GB for CC3M and 3 GB for ImageNet at 512
dimensions. `.dockerignore` keeps the local `.venv`, `cache/` and `outputs/`
out of the build context, so the image carries the source and the installed
dependencies and nothing else.

### Local

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/python run.py <config.yaml> [--stage all|extract|train|perm|eval|table|plot|density]
.venv/bin/python -m pytest tests/ -q
```

---

## The deliverables and how to produce them

| Deliverable | Command | Output |
|---|---|---|
| Everything below the synthetic sweeps, in one run | `bash scripts/run_post_rebuttal.sh all` | `outputs/post_rebuttal/post_rebuttal_results.md` |
| Table 1, five methods on CC3M, three seeds | `bash scripts/run_post_rebuttal.sh table1` | `outputs/post_rebuttal/cc3m_clip_b32/table1.md` and `table1.tex` |
| Figure 2, decoder cosine density on COCO | `bash scripts/run_post_rebuttal.sh figure2` | `outputs/post_rebuttal/coco_clip_b32/multi_density.pdf` |
| The eleven post-rebuttal analyses | `bash scripts/run_post_rebuttal.sh rebuttal` | `outputs/post_rebuttal/rebuttal/<setting>/<analysis>.md` |
| One archive of every report, figure, number file and table | `bash scripts/run_post_rebuttal.sh deliverables` | `outputs/post_rebuttal_deliverables.tar.gz` |
| Synthetic sweep over the angle between paired feature directions | `bash scripts/run_synthetic_alpha.sh` | `outputs/theorem2_alpha_sweep_l2/runs/<timestamp>/params/*.npz` |
| Synthetic sweep over the auxiliary-loss weight | `bash scripts/run_synthetic_lambda.sh` | `outputs/theorem2_lambda_sweep_l2/runs/<timestamp>/params/*.npz` |
| Figure 2 across several encoders at once | `bash scripts/run_multi_density.sh` | `outputs/multi_density/multi_density.pdf` |
| Table 1 for one of the other encoders | `bash scripts/run_cc3m.sh <model_key>` | `outputs/cc3m_<model_key>/table1.md` |

`<model_key>` is one of `clip_b32`, `clip_l14`, `openclip_b32`, `openclip_l14`,
`siglip2_base`, `siglip2_large`.

Every stage is idempotent. Rerunning a command skips any artifact that already
exists, so an interrupted run resumes rather than restarting.

---

## The post-rebuttal run

`configs/post_rebuttal/clip_b32.yaml` is the one configuration that produces
every non-synthetic deliverable together. It pulls in the two pipeline configs
beside it, `clip_b32_coco.yaml` for Figure 2 and `clip_b32_cc3m.yaml` for
Table 1, and adds its own `rebuttal:` block: the co-activation threshold `tau`,
the shuffle seed of the noise-floor panel, the number of bootstrap resamples,
which of the two settings to analyse, which analyses to run, and the training
seed of the second COCO model.

It runs in four stages, all idempotent: `figure2`, `table1`, `rebuttal` and
`report`. The `rebuttal` stage trains the second COCO model, builds the extra
co-activation panels (image against image, text against text, text against a
different caption, and the shuffled noise floor), then runs the eleven analyses
registered in `src/rebuttal/registry.py`. Each analysis writes
`<name>.json` with every number, `<name>.md` with a report that stands on its
own, and a figure where the measurement has one. An analysis that raises does
not stop the others: its traceback goes to `<name>.error.txt`, the rest still
run, and the process exits non-zero naming what failed. The `report` stage
gathers all of it, plus an inventory of every file produced, into
`outputs/post_rebuttal/post_rebuttal_results.md`.

`post_rebuttal_exp.md` at the repository root is the operating guide for that
run: prerequisites, disk budget, the exact Docker commands, how long each part
takes on one GPU, how to resume after an interruption, and what to send back.
It is written in Korean.

---

## The rules every stage obeys

These four rules are implemented in exactly one place each, and every number in
the paper comes out of them.

**Alive latent.** A latent is alive when it fires at least once over the rows
the panel was built on, that is `fire_count >= 1`. There is no firing-rate
threshold anywhere in this repository. The rate is reported as
`fire_count / n_samples` because it is informative, but nothing is filtered by
it. Implemented in `src/alignment/panel.py`.

**Sample count.** The co-activation panel is built on the FULL training split:
every pair, which is 566,435 for COCO train and about 2.87M for CC3M. The
`max_samples` argument exists and defaults to 0, which means every pair; a
positive value takes an evenly spaced subsample and is only there for quick
checks.

**Co-activation correlation.** `C[i, j]` is the Pearson correlation between
image latent `i` and text latent `j`, accumulated in one streaming float64 pass
so the (N, L) latent matrix is never materialized. A latent with zero variance
gives a correlation of 0, not NaN.

**Matching.** One signed, alive-restricted Hungarian assignment on `C`. Dead
rows and dead columns are set to -1e9 so they cannot take an alive latent's
partner, then `linear_sum_assignment(-C_masked)` maximizes total correlation.
The correlation is used signed; `abs()` is never applied.

The assignment is a full permutation, so every image latent gets a partner,
including one that never fired and whose partner is therefore arbitrary.
`usable` marks the rows alive on both sides. Any statistic computed over
matched pairs, above all the cosine distance between a latent and its partner,
has to be restricted to those rows, because the arbitrary partners drag the
summary toward the value for unmatched noise. The downstream evaluations are
the deliberate exception: COCO retrieval and ImageNet zero-shot reindex the
whole text latent vector by `perm` with no mask, which is the paper's protocol
and is what makes `ours` differ from `separated` by the reindexing and by
nothing else.

Neither pipeline trusts a `panel.npz` it finds on disk without reading
`panel.json` first. A panel built for a quick check (`max_samples > 0`) or as a
noise floor (`shuffle_seed != 0`) has exactly the same shape as the real one,
so a panel whose sidecar does not match the rules above is rebuilt rather than
reused.

Figure 2 deliberately uses none of this. It draws EVERY ordered latent pair
`(i, j)` with no alive mask, no Hungarian matching and no correlation
threshold. Dead latents have correlation 0 against every partner and therefore
land in the lowest bin, which is what makes that bin the reference the other
bins are read against. Each bin is subsampled uniformly at random, with a fixed
seed, to at most two million pairs before the kernel density estimate; the
statistics in `figure2_bin_stats.md` are computed on the full bin.

---

## The cache format

One layout, written by `src/data/extract.py` and read by `src/data/cache_io.py`.
Embeddings are stored RAW. The single normalization rule, the CLIP vector norm
with no epsilon, lives in `src/data/paired_dataset.py` and is applied at load
time by training, by the panel builder and by every evaluator.

Paired caches (COCO, CC3M) under `cache/<model_key>_<dataset>/`:

```
image_embeddings.npy   float32 (N, dim)   one row per PAIR; the image row is
                                          duplicated across a photo's captions
text_embeddings.npy    float32 (N, dim)   row i is the caption of image row i
keys.json              list[str] of length N
splits.json            {"train": [keys], "val": [keys], "test": [keys]}
meta.json              {"model_key", "dim", "dataset", "n_pairs", ...}
captions.json          {key: caption text}   (COCO only)
```

COCO keys are `"{image_id}_{cap_idx}"`; CC3M keys are the webdataset `__key__`.
Recovering an image id from a COCO key by splitting on the last underscore is
part of the format, not an implementation detail. COCO writes all three splits
into one cache dir, named `train` (113,287 images), `val` (5,000) and `test`
(5,000).

The ImageNet cache is not paired, because its images carry a class label and its
text is a fixed class-by-template grid:

```
image_embeddings.npy   float32 (50000, dim)
labels.npy             int64   (50000,)      class index of each image
text_embeddings.npy    float32 (80000, dim)  class-major: row c * 80 + t
text_keys.json         ["{class_idx}_{tmpl_idx}", ...]
meta.json              {"model_key", "dim", "n_classes", "n_templates", ...}
```

The text side is the 80 OpenAI templates from `open_clip.zero_shot_metadata`
applied to the 1000 class names.

Extraction is memory-safe and resumable. Embeddings go to fixed-size chunk files
under `cache_dir/parts/` and are assembled at the end with
`numpy.lib.format.open_memmap`, so peak memory is one chunk rather than the
whole table. A restart reads `progress.json`, keeps the chunks it covers and
skips exactly that many source records, counted after decoding so that an
undecodable image cannot shift the resume offset. COCO additionally keeps an
append-only caption log beside its chunks, which is what lets a resumed pass
still write a complete `captions.json`. A progress line carrying a rate, and an
ETA whenever the split's size is known, is printed every 30 seconds of wall
clock, so a pass smaller than one chunk still reports.

---

## The panel artifact

Everything downstream of training reads one file, `panel.npz`, plus its
`panel.json` sidecar.

```
C                  float32 (L_img, L_txt)  co-activation Pearson correlation
perm               int64   (L_img,)        Hungarian partner of each image latent
usable             bool    (L_img,)        alive on both sides; exclude the rest
                                           from matched-pair statistics
alive_image        bool    (L_img,)        fire_count >= 1
alive_text         bool    (L_txt,)
fire_count_image   int64   (L_img,)
fire_count_text    int64   (L_txt,)
rate_image         float64 (L_img,)        fire_count / n_samples, reported only
rate_text          float64 (L_txt,)
n_samples          int64                   rows the panel was built on
```

`panel.json` records the pairing, the checkpoint paths, the split, the sample
count (`n_samples`), how many keys the split declares (`n_split_rows`), how
many of them resolved to rows (`n_split_rows_resolved`, lower only when
keys.json is missing some), the shuffle seed and the alive rule in words. `perm.npz` is written
alongside it with the same contents, for call sites that still ask for that
name.

`training.latent_size` is the TOTAL latent budget, so 8192 gives 4096 latents
per modality and `C` is 4096 by 4096.

`src/alignment/panel.py` also supports the pairings the later rebuttal analyses
need (`img_img`, `txt_txt`, `txt_txt_diffcap`) and a `shuffle_seed` that
deranges the second side to produce a noise floor. Those are machinery here;
neither Table 1 nor Figure 2 uses them.

---

## What the CC3M pipeline does

1. **Extract** the paired CC3M embeddings into `cache/<model_key>_cc3m`.
2. **Train**, for every seed in `training.seeds`, the five methods into
   `<root>/seed{S}/<method>/final`. `Post-hoc Alignment (Ours)` trains nothing;
   it re-uses the `Modality-Specific SAEs` checkpoint.
3. **Panel** for that seed, from its `separated` checkpoint on the FULL CC3M
   training split, into `<root>/seed{S}/ours/panel.npz`.
4. **Evaluate**, extracting `cache/<model_key>_coco` and
   `cache/<model_key>_imagenet` on demand, into `<root>/seed{S}/eval/<method>/`
   as `recon_coco.json`, `retrieval.json`, `recon_imagenet.json` and
   `zeroshot.json`.
5. **Table** across all seeds, into `<root>/table1.md` and `<root>/table1.tex`.

Output tree:

```
outputs/post_rebuttal/cc3m_clip_b32/
├── table1.md, table1.tex, config.json
├── seed0/
│   ├── shared/final/, separated/final/, iso_align/final/, group_sparse/final/
│   ├── ours/{panel.npz, panel.json, perm.npz}
│   └── eval/<method>/{recon_coco.json, retrieval.json,
│                     recon_imagenet.json, zeroshot.json}
├── seed1/ ...
└── seed2/ ...
```

Table 1 columns, in order: COCO reconstruction (lower is better), image to text
R@1, R@5, R@10, text to image R@1, R@5, R@10, ImageNet reconstruction (lower is
better), zero-shot top-1. Recall and accuracy are multiplied by 100. With
several seeds a cell shows the mean and the sample standard deviation, and best
and second best are decided by the mean alone.

### The three evaluations

**Reconstruction** reports the paper formula,
`0.5 * mean over pairs of (||x - x_hat||^2 + ||y - y_hat||^2)`, on L2-normalized
embeddings. On COCO the pair is what the cache says it is. On ImageNet, which is
not paired, each validation image is paired with ONE template of its true class
drawn at random, which is what the paper's dataset object did; the draw is
seeded here (`eval.recon_template_seed`) so the number is reproducible.

**Retrieval** ranks pessimistically: a rank counts every candidate scoring
greater than or equal to the ground truth, so a tie against the ground truth
counts against it. Collapsed latents produce large tied blocks, and an
optimistic rule would report those as perfect retrieval. The tie-block size at
the ground truth is reported next to the recalls. Image to text takes the best
rank over an image's five captions.

**Zero-shot** has two variants. `raw` scores every latent column and is what
Table 1 reports; it is the default and writes `zeroshot.json`. `filtered` drops
columns whose image-side firing rate exceeds `eval.max_fire_rate` (0.5) and
writes `zeroshot_filtered.json`. The variant is recorded inside the JSON, so the
two can never be confused.

---

## Adding a vision-language model

1. Add `configs/models/<key>.yaml` with `key`, `backend` (`transformers` or
   `openclip`), `hf_id` or `arch` plus `pretrained`, `hidden_size`,
   `text_max_length`, `is_siglip` and `image_size`.
2. For the downstream table, add `configs/cc3m/overrides/<key>.yaml` that
   `!ref`s `_shared.yaml#training`, `#methods` and `#eval`.
3. Run `bash scripts/run_cc3m.sh <key>`.

---

## Layout

```
.
├── README.md, pyproject.toml, Dockerfile, docker/entrypoint.sh
├── post_rebuttal_exp.md    operating guide for the post-rebuttal run (Korean)
├── run.py                  single entry point, dispatched on the config's kind
├── run_synthetic_v2.py     driver for the two synthetic sweeps
├── scripts/                one wrapper per deliverable, docker_build, and
│                           collect_deliverables (packs the archive to send)
├── configs/
│   ├── post_rebuttal/      clip_b32.yaml (everything), plus the Figure 2 and
│   │                       Table 1 configs it pulls in
│   ├── cc3m/               _shared.yaml plus one override per encoder
│   ├── multi_density.yaml  Figure 2 across several encoders
│   ├── synthetic/          the two synthetic sweeps
│   └── models/             encoder definitions
├── src/
│   ├── pipelines/          synthetic_sweep | multi_density | cc3m_downstream |
│   │                       post_rebuttal (the four-stage run)
│   ├── rebuttal/           one module per analysis, common.py (the shared
│   │                       Setting and helpers), registry.py (what runs where)
│   ├── data/               cache_io (the format), extract (the only extractor),
│   │                       paired_dataset (the normalization rule), synthetic
│   ├── datasets/           synthetic data builders
│   ├── encoders/           transformers and open_clip backends
│   ├── models/             TopKSAE and TwoSidedTopKSAE, the only SAE code
│   ├── training/           trainer, losses, gradient utilities
│   ├── alignment/          panel (the four rules), hungarian (thin wrapper)
│   ├── eval/               recon | retrieval | zeroshot, each callable and a CLI
│   ├── reporting/          table1
│   ├── runners/, metrics/, common/, configs/   used by the synthetic sweeps
│   ├── plotting/           multi_density (Figure 2), the synthetic figures
│   └── utils/config.py     YAML loader with !ref
├── tests/                  pytest, CPU only, no downloads
└── outputs/, cache/        runtime artifacts, not committed
```

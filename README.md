# Same Concept, Different Directions: Cross-Modal Feature Heterogeneity in Sparse Autoencoders

Reproducible code for the four experimental deliverables: two synthetic SAE
sweeps, a multi-VLM decoder-cosine density study, and a CC3M-trained
downstream evaluation suite (COCO retrieval / ImageNet zero-shot /
cross-modal steering / MonoSemanticity score).

All experiments run via a single entry point — `python run.py <config.yaml>`
— with one YAML per experiment.

---

## Setup

### Docker (recommended)

```bash
bash scripts/docker_build.sh
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e CONFIG=configs/cc3m/overrides/clip_l14.yaml \
  -v $PWD/cache:/app/repo/cache -v $PWD/outputs:/app/repo/outputs \
  vlm-sae
```

The container reads `CONFIG` (required) and `STAGE` (optional, default
`all`) from the env, then runs `python run.py "$CONFIG" --stage "$STAGE"`.

### Local

```bash
pip install -e .
python run.py <config.yaml> [--stage all|extract|train|perm|eval|plot|density]
```

---

## Run the experiments

| # | Deliverable | Command | Output |
|---|---|---|---|
| 1 | Synthetic α-sweep | `bash scripts/run_synthetic_alpha.sh` | `outputs/theorem2_alpha_sweep_l2/runs/<ts>/params/*.npz` |
| 2 | Synthetic λ-sweep | `bash scripts/run_synthetic_lambda.sh` | `outputs/theorem2_lambda_sweep_l2/runs/<ts>/params/*.npz` |
| 3 | Multi-VLM density figure | `bash scripts/run_multi_density.sh` | `outputs/multi_density/multi_density.{pdf,svg,png}` |
| 4 | CC3M downstream (per model) | `bash scripts/run_cc3m.sh <model_key>` | `outputs/cc3m_<model_key>/{<method>/final, ours/perm.npz, eval/<method>/{retrieval.json, zeroshot.json, steering/summary.json, ms/ms_summary.json}}` |

`<model_key>` for the CC3M pipeline is one of:

```
clip_b32  clip_l14  openclip_b32  openclip_l14  siglip2_base  siglip2_large
```

All scripts forward extra arguments to `python run.py`, e.g.
`bash scripts/run_synthetic_alpha.sh --stage extract`.

---

## What the CC3M pipeline does (`--stage all`)

1. **Extract** — paired CC3M embeddings → `cache/<key>_cc3m`.
2. **Train** — five SAEs (`shared`, `separated`, `iso_align`, `group_sparse`,
   `ours` shares the `separated` checkpoint). Each saved to
   `<root>/<method>/final/`.
3. **Perm** — single text→image Hungarian permutation built from CC3M train
   embeddings, saved to `<root>/ours/perm.npz`.
4. **Eval** — extracts auxiliary caches if missing
   (`cache/<key>_coco` for retrieval/steering, `cache/<key>_imagenet` for
   zero-shot, `cache/<external_encoder>_cc3m` for the MS reference probe),
   then runs each evaluator per method. Results land in
   `<root>/eval/<method>/`.

The eval block is config-driven: see the `eval:` block in
`configs/cc3m/_shared.yaml` to toggle individual evaluators or change
`steering_alphas`, `external_encoder`, etc.

---

## Adding a new VLM

1. Drop a model definition under `configs/models/<key>.yaml` (see existing
   entries for the schema — `key`, `backend`, `hf_id` or `arch`/`pretrained`,
   `hidden_size`, `text_max_length`, `is_siglip`, `image_size`).
2. For the CC3M pipeline, add `configs/cc3m/overrides/<key>.yaml` that
   `!ref`s `_shared.yaml#training`, `#methods`, `#eval`.
3. Run: `bash scripts/run_cc3m.sh <key>`.

---

## Layout

```
.
├── README.md, pyproject.toml, Dockerfile, docker/entrypoint.sh
├── scripts/                — bash wrappers (one per deliverable + docker_build)
├── run.py                  — single entrypoint, kind-dispatched
├── run_synthetic_v2.py     — HF-Trainer driver for the synthetic sweeps
├── configs/
│   ├── synthetic/          — alpha_sweep, lambda_sweep
│   ├── multi_density.yaml  — 8-VLM density figure
│   ├── models/             — VLM definitions
│   └── cc3m/               — _shared.yaml + per-model overrides
├── src/
│   ├── pipelines/          — synthetic_sweep | multi_density | cc3m_downstream
│   ├── data/               — extract (multi-density), extract_coco / _imagenet / _cc3m (eval), cache_io, paired_dataset
│   ├── datasets/           — synthetic generators + cached_clip_pairs / cached_imagenet_pairs
│   ├── encoders/           — transformers + open_clip backends
│   ├── models/             — TopKSAE / TwoSidedTopKSAE (clean impl + HF-style wrappers)
│   ├── training/           — trainer + losses + callbacks
│   ├── alignment/          — Hungarian permutation builder
│   ├── runners/            — HF Trainer subclasses for synthetic experiments
│   ├── metrics/            — alignment, normalize, evaluate, synthetic_eval, canonical_perm
│   ├── eval/               — retrieval | zeroshot | steering | ms (each callable + CLI)
│   ├── plotting/           — alpha_sweep, lambda_sweep, multi_density, palette
│   ├── configs/, common/   — ExperimentConfig dataclasses + registry
│   └── utils/config.py     — YAML loader (anchors + !ref)
├── tests/                  — pytest scaffold (skip-stubs)
└── outputs/, cache/        — runtime artifacts (gitignored)
```

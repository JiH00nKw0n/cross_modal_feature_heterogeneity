#!/usr/bin/env bash
# The post-rebuttal deliverables for CLIP ViT-B/32.
#
#   figure2   Figure 2 on COCO    -> outputs/post_rebuttal/coco_clip_b32/multi_density.pdf
#   table1    Table 1 on CC3M     -> outputs/post_rebuttal/cc3m_clip_b32/table1.md
#   rebuttal  The extra analyses  -> outputs/post_rebuttal/rebuttal/<setting>/
#   report    One document        -> outputs/post_rebuttal/post_rebuttal_results.md
#   all       All four, in that order.
#
# figure2 and table1 run their own pipeline config directly, so either can be
# driven stage by stage (--stage extract, --stage train, and so on). rebuttal,
# report and all run the post-rebuttal config, which delegates to those two
# pipelines first and is idempotent, so nothing finished is redone.
#
# Usage: bash scripts/run_post_rebuttal.sh {all|figure2|table1|rebuttal|report} [--stage ...]
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
shift || true
case "$what" in
  table1)   exec python run.py configs/post_rebuttal/clip_b32_cc3m.yaml "$@" ;;
  figure2)  exec python run.py configs/post_rebuttal/clip_b32_coco.yaml "$@" ;;
  rebuttal) exec python run.py configs/post_rebuttal/clip_b32.yaml --stage rebuttal "$@" ;;
  report)   exec python run.py configs/post_rebuttal/clip_b32.yaml --stage report "$@" ;;
  all)      exec python run.py configs/post_rebuttal/clip_b32.yaml --stage all "$@" ;;
  *) echo "usage: $0 {all|figure2|table1|rebuttal|report} [--stage ...]" >&2; exit 2 ;;
esac

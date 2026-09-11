#!/usr/bin/env bash
# The two post-rebuttal deliverables for CLIP ViT-B/32.
#   table1  Table 1 on CC3M over three seeds -> outputs/post_rebuttal/cc3m_clip_b32/table1.md
#   figure2 Figure 2 on COCO                 -> outputs/post_rebuttal/coco_clip_b32/multi_density.pdf
# Usage: bash scripts/run_post_rebuttal.sh {table1|figure2|all} [--stage ...]
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
shift || true
case "$what" in
  table1)  exec python run.py configs/post_rebuttal/clip_b32_cc3m.yaml "$@" ;;
  figure2) exec python run.py configs/post_rebuttal/clip_b32_coco.yaml "$@" ;;
  all)
    python run.py configs/post_rebuttal/clip_b32_coco.yaml "$@"
    exec python run.py configs/post_rebuttal/clip_b32_cc3m.yaml "$@"
    ;;
  *) echo "usage: $0 {table1|figure2|all} [--stage ...]" >&2; exit 2 ;;
esac

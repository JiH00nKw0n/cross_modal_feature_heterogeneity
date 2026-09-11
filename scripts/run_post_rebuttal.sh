#!/usr/bin/env bash
# The post-rebuttal deliverables for CLIP ViT-B/32.
#
#   all           Every stage of configs/post_rebuttal/clip_b32.yaml, in order:
#                 Figure 2 on COCO, Table 1 on CC3M, the extra analyses, the
#                 combined report. This is the one command a full run needs.
#   figure2       Figure 2 alone   -> outputs/post_rebuttal/coco_clip_b32/multi_density.pdf
#   table1        Table 1 alone    -> outputs/post_rebuttal/cc3m_clip_b32/table1.md
#   rebuttal      The rebuttal stage of the same config: trains the second COCO
#                 model, builds the extra co-activation panels and runs every
#                 registered analysis -> outputs/post_rebuttal/rebuttal/<setting>/
#   report        The combined document alone
#                 -> outputs/post_rebuttal/post_rebuttal_results.md
#   deliverables  Packs the reports, figures, numbers and tables into
#                 outputs/post_rebuttal_deliverables.tar.gz
#
# figure2 and table1 run their own pipeline config directly, so either can be
# driven stage by stage (--stage extract, --stage train, and so on). rebuttal,
# report and all run the post-rebuttal config, which delegates to those two
# pipelines first. Every stage is idempotent, so nothing finished is redone and
# an interrupted run is resumed by repeating the same command.
#
# Usage:
#   bash scripts/run_post_rebuttal.sh {all|figure2|table1|rebuttal|report|deliverables} [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
what="${1:-all}"
shift || true
case "$what" in
  table1)       exec python run.py configs/post_rebuttal/clip_b32_cc3m.yaml "$@" ;;
  figure2)      exec python run.py configs/post_rebuttal/clip_b32_coco.yaml "$@" ;;
  rebuttal)     exec python run.py configs/post_rebuttal/clip_b32.yaml --stage rebuttal "$@" ;;
  report)       exec python run.py configs/post_rebuttal/clip_b32.yaml --stage report "$@" ;;
  all)          exec python run.py configs/post_rebuttal/clip_b32.yaml --stage all "$@" ;;
  deliverables) exec bash scripts/collect_deliverables.sh "$@" ;;
  *) echo "usage: $0 {all|figure2|table1|rebuttal|report|deliverables} [extra args]" >&2
     exit 2 ;;
esac

#!/usr/bin/env bash
# Multi-VLM density figure (paper Fig multi_density) across 8 base+large VLMs.
# Usage: bash scripts/run_multi_density.sh [--stage all|extract|train|perm|density]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python run.py configs/multi_density.yaml "$@"

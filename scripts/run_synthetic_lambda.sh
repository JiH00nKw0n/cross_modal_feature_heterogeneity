#!/usr/bin/env bash
# Fig 2 — synthetic λ-sweep at α=0.5 (Iso-Energy + Group-Sparse + Post-hoc).
# Usage: bash scripts/run_synthetic_lambda.sh [--stage all|extract|train|perm|eval|plot]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python run.py configs/synthetic/lambda_sweep.yaml "$@"

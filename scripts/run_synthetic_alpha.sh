#!/usr/bin/env bash
# Fig 1 — synthetic α-sweep (cos(phi_i, psi_i) ∈ {0, 0.2, ..., 1.0}).
# Usage: bash scripts/run_synthetic_alpha.sh [--stage all|extract|train|perm|eval|plot]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python run.py configs/synthetic/alpha_sweep.yaml "$@"

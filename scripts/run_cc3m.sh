#!/usr/bin/env bash
# CC3M downstream pipeline for one model variant.
# Usage: bash scripts/run_cc3m.sh <model_key> [--stage all|extract|train|perm|eval]
#   <model_key> ∈ {clip_b32, clip_l14, openclip_b32, openclip_l14, siglip2_base, siglip2_large}
set -euo pipefail
key="${1:?usage: $0 <model_key> [--stage ...]}"
shift || true
cd "$(dirname "$0")/.."
cfg="configs/cc3m/overrides/${key}.yaml"
if [ ! -f "$cfg" ]; then
  echo "no such config: $cfg" >&2
  echo "available:" >&2
  ls configs/cc3m/overrides/ >&2
  exit 2
fi
exec python run.py "$cfg" "$@"

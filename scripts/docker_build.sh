#!/usr/bin/env bash
# Build the GPU Docker image (defined by ./Dockerfile).
# Usage: bash scripts/docker_build.sh [extra docker build args]
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker build -f Dockerfile -t vlm-sae "$@" .

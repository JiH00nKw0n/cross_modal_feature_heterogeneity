#!/usr/bin/env bash
# Container entrypoint. Reads CONFIG and STAGE from env, dispatches to run.py.
set -euo pipefail

CONFIG=${CONFIG:?"set CONFIG=configs/<...>.yaml"}
STAGE=${STAGE:-all}

cd /app/repo
echo "[entrypoint] CONFIG=$CONFIG STAGE=$STAGE"
exec python run.py "$CONFIG" --stage "$STAGE"

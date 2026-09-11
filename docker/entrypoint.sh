#!/usr/bin/env bash
# Container entrypoint. Reads CONFIG and STAGE from the environment and hands
# them to run.py.
#
# CONFIG defaults to the post-rebuttal configuration, which produces Table 1,
# Figure 2 and every rebuttal analysis in one run. STAGE defaults to "all"; the
# stages that configuration understands are figure2, table1, rebuttal and
# report, and every one of them is idempotent, so a container that is stopped
# and started again resumes rather than repeating finished work.
set -uo pipefail

CONFIG=${CONFIG:-configs/post_rebuttal/clip_b32.yaml}
STAGE=${STAGE:-all}

cd /app/repo
echo "[entrypoint] CONFIG=$CONFIG STAGE=$STAGE"
python run.py "$CONFIG" --stage "$STAGE"
status=$?

# The pipeline exits non-zero when an analysis failed, having written its
# traceback next to the reports. Name the place to look before the container
# ends, because the exit status alone does not say which analysis it was.
if [[ $status -ne 0 ]]; then
  echo "[entrypoint] run.py exited with status $status." >&2
  echo "[entrypoint] A failed analysis leaves its traceback at" >&2
  echo "[entrypoint]   outputs/post_rebuttal/rebuttal/<setting>/<analysis>.error.txt" >&2
fi
exit $status

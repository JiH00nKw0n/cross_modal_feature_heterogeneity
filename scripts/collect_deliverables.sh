#!/usr/bin/env bash
# Pack everything readable under outputs/post_rebuttal/ into one archive.
#
#   bash scripts/collect_deliverables.sh [output_root] [archive]
#
# Defaults: output_root = outputs/post_rebuttal, archive =
# outputs/post_rebuttal_deliverables.tar.gz.
#
# What goes in: every .md report, every .pdf and .png figure, every .json
# holding the numbers behind a report, and every .tex table. What stays out:
# the .npz co-activation panels and the trained checkpoints (model.safetensors,
# pytorch_model.bin, optimizer state). Those are inputs to the numbers rather
# than results, and one panel alone is hundreds of megabytes, which is the
# difference between an archive that can be sent and one that cannot.
set -euo pipefail
cd "$(dirname "$0")/.."

root="${1:-outputs/post_rebuttal}"
archive="${2:-outputs/post_rebuttal_deliverables.tar.gz}"

if [[ ! -d "$root" ]]; then
  echo "$0: $root does not exist; run the pipeline first:" >&2
  echo "  bash scripts/run_post_rebuttal.sh all" >&2
  exit 2
fi

# NUL-separated so that a path with a space cannot split into two entries.
list="$(mktemp)"
trap 'rm -f "$list"' EXIT
find "$root" -type f \
  \( -name '*.md' -o -name '*.pdf' -o -name '*.png' -o -name '*.json' -o -name '*.tex' \) \
  -print0 | sort -z > "$list"

count=$(tr -cd '\0' < "$list" | wc -c | tr -d ' ')
if [[ "$count" -eq 0 ]]; then
  echo "$0: found no report, figure, number file or table under $root" >&2
  exit 3
fi

mkdir -p "$(dirname "$archive")"
tar --null -czf "$archive" -T "$list"

size=$(du -h "$archive" | cut -f1)
echo "[collect] $count files from $root"
echo "[collect] wrote $archive ($size)"

#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-experiments.csv}"        # pass a path or defaults to ./experiments.csv
MAX_PARALLEL="${MAX_PARALLEL:-2}"  # cap concurrency to avoid overuse (e.g., 1–4)

TOTAL=$(( $(wc -l < "$CSV") - 1 ))
if (( TOTAL <= 0 )); then
  echo "CSV has no data rows: $CSV"; exit 1
fi

echo "Submitting $TOTAL tasks from $CSV with concurrency cap %$MAX_PARALLEL"
sbatch --export=CSV="$CSV" --array=0-$((TOTAL-1))%${MAX_PARALLEL} array.sbatch

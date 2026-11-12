#!/usr/bin/env bash
# Evaluate every finished run (PR-random and NR-random sweeps), then aggregate.
# Usage: scripts/launch_eval.sh [JOBS] [EPISODES] [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${1:-30}
EPISODES=${2:-50}
OUT=${3:-runs}

for done in "$OUT"/*/DONE; do
  d=$(dirname "$done")
  [ -f "$d/eval.csv" ] && continue
  echo "python evaluate.py --run-dir $d --episodes $EPISODES > $d/eval.log 2>&1"
done | xargs -P "$JOBS" -I{} bash -c "{}"

python aggregate.py --runs "$OUT" --out "results/$(basename "$OUT")"

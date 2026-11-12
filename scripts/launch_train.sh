#!/usr/bin/env bash
# Run the Phase 0 training grid in parallel (one CPU core per run).
# Usage: scripts/launch_train.sh [JOBS] [TOTAL_TIMESTEPS] [OUT_DIR]
# Runs that already have a DONE marker are skipped, so re-running resumes the grid.
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${1:-30}
STEPS=${2:-1000000}
OUT=${3:-runs}
ENVS=${ENVS:-"Hopper-v5 Walker2d-v5"}
MODES=${MODES:-"none pr nr"}
SEEDS=${SEEDS:-"0 1 2 3 4"}

mkdir -p "$OUT/logs"
for env in $ENVS; do for mode in $MODES; do for seed in $SEEDS; do
  name="${env}__${mode}__s${seed}"
  [ -f "$OUT/$name/DONE" ] && continue
  echo "python train.py --env-id $env --mode $mode --seed $seed --total-timesteps $STEPS --out-dir $OUT > $OUT/logs/$name.log 2>&1"
done; done; done | xargs -P "$JOBS" -I{} bash -c "{}"

echo "training grid finished: $(ls "$OUT"/*/DONE 2>/dev/null | wc -l) runs done"

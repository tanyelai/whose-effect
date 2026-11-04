#!/usr/bin/env bash
# The maps inside the network (Section 6 and Appendix D.1), one experiments/scripts/inside.py run per model and task: the six small
# models on every task of the paper's sources (CausalGym's 29 from its snapshot; the 49 others from the data folders that
# experiments/matrix/inside_maps.tsv names: each model's own for Kumon, RAVEL and the sentiment templates, experiments/tasks/ for IOI,
# number-only agreement and SST), and the four large models on Section 6's three tasks (experiments/matrix/inside_sets.json). A map
# already there is skipped; N=<k> runs k maps at a time on one GPU.
# Usage: bash experiments/run/inside.sh [small|large]      (default: both)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
WHICH="${1:-both}"
one() {   # one <model> <task> [<data folder>]
  local m=$1 task=$2 data=${3:-} t; t=$(tag "$m")
  [ -f "experiments/results/inside/inside_${t}_${task}.json" ] && return 0
  limit
  run "inside_${t}_${task}" "$PY" -I experiments/scripts/inside.py "$task" --model="$m" --revision="$(rev "$m")" ${data:+--data="$data"} \
    --device="$DEV" &
}
set_tasks() { "$PY" -I -c "import json; print(' '.join(json.load(open('experiments/matrix/inside_sets.json'))['$1']['tasks']))"; }
if [ "$WHICH" != large ]; then
  for task in $(set_tasks small_causalgym); do for m in "${SMALL[@]}"; do one "$m" "$task"; done; done
  while IFS=$'\t' read -r t group task data rest; do
    for m in "${SMALL[@]}"; do [ "$(tag "$m")" = "$t" ] && one "$m" "$task" "$data"; done
  done < experiments/matrix/inside_maps.tsv
fi
if [ "$WHICH" != small ]; then
  for task in $(set_tasks large_npi_fg_agr); do for m in "${LARGE[@]}"; do one "$m" "$task"; done; done
fi
wait

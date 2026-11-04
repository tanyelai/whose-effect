#!/usr/bin/env bash
# CausalGym's own maps at every layer (Section 7.1): the inside maps of bash experiments/run/inside.sh cover the paper's
# even layers (inside.py's layers 1, 3, 5, ...) of CausalGym's 29 tasks in the six small models; these runs add the paper's odd
# layers (inside.py's 2, 4, ..., L - 2 for a model of L blocks) and move each output to experiments/results/inside_even/. A map already
# there is skipped; N=<k> runs k maps at a time on one GPU.
# Usage: bash experiments/run/causalgym_maps.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
declare -A LAYERS=([gpt2]=12 [pythia-160m]=12 [pythia-410m]=24 [pythia-1b]=16 [SmolLM2-360M]=32 [Qwen2.5-0.5B]=24)
mkdir -p experiments/results/inside_even/dirs
one() {   # one <model> <task>
  local m=$1 task=$2 t; t=$(tag "$m"); local stem="inside_${t}_${task}_even"
  [ -f "experiments/results/inside_even/$stem.json" ] && return 0
  limit
  { run "$stem" "$PY" -I experiments/scripts/inside.py "$task" --model="$m" --revision="$(rev "$m")" \
        --layers="$(seq -s, 2 2 $((LAYERS[$t] - 2)))" --tag=even --device="$DEV"
    if [ -f "experiments/results/inside/$stem.json" ]; then
      mv "experiments/results/inside/$stem.json" "experiments/results/inside/$stem.npz" experiments/results/inside_even/
      mv "experiments/results/inside/dirs/$stem.npz" experiments/results/inside_even/dirs/
    fi; } &
}
for task in $("$PY" -I -c "import json; print(' '.join(json.load(open('experiments/matrix/inside_sets.json'))['small_causalgym']['tasks']))"); do
  for m in "${SMALL[@]}"; do one "$m" "$task"; done
done
wait

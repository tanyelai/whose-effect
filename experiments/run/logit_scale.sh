#!/usr/bin/env bash
# The logits' scale of every model (experiments/results/logit_scale.json), which sets the instruments' float32 tolerance
# (experiments/lib/instrument_tol.py): inference on the test inputs, up to 10 rows per task for the six small models and Pythia-1.4b,
# 3 for the larger ones (the scale is the largest |label logit|, so fewer rows can only lower it and tighten the tolerance).
# Qwen2.5-32B's scale comes from its readout runs' saved label logits instead (experiments/score/score_all.sh derives it).
# Usage: bash experiments/run/logit_scale.sh      (bash experiments/run/data.sh first: the scale reads each model's own tasks)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
OUT=experiments/results/logit_scale.json; mkdir -p experiments/results
csv() { local IFS=,; echo "$*"; }
run logit_scale_small  "$PY" -I experiments/scripts/logit_scale.py --out="$OUT" --models="$(csv "${SMALL[@]}")" --rows=10
run logit_scale_larger "$PY" -I experiments/scripts/logit_scale.py --out="$OUT" --models="$(csv "${LARGE[@]}" "${SCALE[@]}")" --rows=3
run logit_scale_1.4b   "$PY" -I experiments/scripts/logit_scale.py --out="$OUT" --models=EleutherAI/pythia-1.4b --rows=10

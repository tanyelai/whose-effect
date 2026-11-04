#!/usr/bin/env bash
# The sentiment direction of Tigges et al. re-read with their own fitting protocol (Section 7.3): the six small
# models and Pythia-1.4b (their published model), three starts each, on their prompts.yaml (fetched by bash experiments/run/data.sh).
# An output already there is skipped.
# Usage: bash experiments/run/sentiment.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
YAML="$REPO/build/sources/tigges/prompts.yaml"
[ -f "$YAML" ] || { echo "no $YAML: run bash experiments/run/data.sh first (it fetches their prompts)"; exit 1; }
for s in 0 1 2; do
  for m in "${SMALL[@]}" EleutherAI/pythia-1.4b; do
    t=$(tag "$m"); [ -f "experiments/results/sentiment_reread_${t}_seed${s}.json" ] && continue
    run "sentiment_${t}_seed${s}" "$PY" -I experiments/scripts/sentiment_reread.py --model="$m" --revision="$(rev "$m")" --yaml="$YAML" \
      --save_rows=1 --seed="$s" --device="$DEV"
  done
done

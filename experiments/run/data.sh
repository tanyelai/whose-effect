#!/usr/bin/env bash
# Each model's own task data (experiments/data/<model>/), from the pinned sources (experiments/lib/pins.py): Kumon & Yanaka's constructions and
# the sentiment templates of Tigges et al. (the model's tokenizer only), RAVEL's attributes under both prompt conventions and
# RAVEL's continent task (the model's own predictions decide which entities it knows; on the GPU), and the RAVEL tasks whose
# prompts come in two forms in one region layout for their maps (experiments/scripts/ravel_regions.py). The task data that do not
# depend on the model are in experiments/tasks/ (experiments/tasks/README.md). Then experiments/run/check_data.py compares every built
# file with the data of the paper's runs.
# Usage: bash experiments/run/data.sh [<model> ...]      (default: every model of the paper; run experiments/run/download.py first)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=("${ALL[@]}")
SRC="$REPO/build/sources"; mkdir -p "$SRC"
pin() { "$PY" -I -c "import sys; sys.path.insert(0, 'experiments/lib'); import pins; print($1)"; }
if [ ! -d "$SRC/kumon" ]; then
  HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download as s; s('$(pin 'pins.KUMON[0]')', repo_type='dataset', revision='$(pin 'pins.KUMON[1]')', local_dir='$SRC/kumon')"
  rm -rf "$SRC/kumon/.cache"                     # the downloader's own metadata folder, not a construction
fi
if [ ! -f "$SRC/tigges/prompts.yaml" ]; then
  mkdir -p "$SRC/tigges"; curl -fsSL -o "$SRC/tigges.zip" "$(pin 'pins.TIGGES[0]')/archive/$(pin 'pins.TIGGES[1]').zip"
  unzip -q -o "$SRC/tigges.zip" -d "$SRC/tigges"; cp "$SRC"/tigges/eliciting-latent-sentiment-*/prompts.yaml "$SRC/tigges/prompts.yaml"
fi
"$PY" -I -c "import hashlib, sys; sys.path.insert(0, 'experiments/lib'); import pins
assert hashlib.sha256(open('$SRC/tigges/prompts.yaml', 'rb').read()).hexdigest() == pins.TIGGES_PROMPTS_SHA256, 'prompts.yaml is not the pinned one'"
CSV=$(IFS=,; echo "${MODELS[*]}")
DIRS=$("$PY" -I -c "import sys; sys.path.insert(0, 'experiments/lib'); import pins; from huggingface_hub import snapshot_download as s
print(','.join(s(m, revision=pins.MODELS[m], local_files_only=True) for m in sys.argv[1].split(',')))" "$CSV")
run kumon_data     "$PY" -I experiments/tasks/make_kumon.py "$SRC/kumon" --models="$CSV" --tokenizer_dirs="$DIRS"
run sentiment_data "$PY" -I experiments/tasks/make_sentiment_templates.py "$SRC/tigges/prompts.yaml" --models="$CSV" --tokenizer_dirs="$DIRS"
for m in "${MODELS[@]}"; do
  r=$(rev "$m"); t=$(tag "$m")
  run "ravel_${t}"     "$PY" -I experiments/scripts/filter_known_attributes.py --model="$m" --revision="$r" --start=0 --ood=1 --device="$DEV"
  run "ravel_sot_${t}" "$PY" -I experiments/scripts/filter_known_attributes.py --model="$m" --revision="$r" --start=1 --ood=1 --device="$DEV"
  run "continent_${t}" "$PY" -I experiments/scripts/filter_known.py --model="$m" --revision="$r" --device="$DEV"
done
# the RAVEL tasks whose prompts come in two forms, in one region layout for their maps inside the network
run ravel_regions "$PY" -I experiments/scripts/ravel_regions.py "${MODELS[@]}"
"$PY" -I experiments/run/check_data.py "${MODELS[@]}"

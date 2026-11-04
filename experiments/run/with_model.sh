#!/usr/bin/env bash
# For a machine whose persistent disk cannot hold the 7-32B weights: download one model at its pinned revision into a scratch
# Hugging Face cache (SCRATCH, e.g. a job's local disk), with every entry of the persistent cache linked in, run a command
# there, and delete the model's weights afterwards.
# Usage: SCRATCH=<local disk>/hf bash experiments/run/with_model.sh <hf id> <command ...>
#        e.g. SCRATCH=<local disk>/hf bash experiments/run/with_model.sh Qwen/Qwen2.5-32B bash experiments/run/readout.sh --models=Qwen/Qwen2.5-32B
set -euo pipefail
MODEL="${1:?usage: with_model.sh <hf id> <command ...>}"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; PY="${PY:-python3}"
SCRATCH="${SCRATCH:?SCRATCH=<a folder with room for the weights>}"; mkdir -p "$SCRATCH/hub"
PERSIST="${HF_HOME:-$HOME/.cache/huggingface}"
for e in "$PERSIST"/hub/*; do [ -e "$e" ] && [ ! -e "$SCRATCH/hub/$(basename "$e")" ] && ln -s "$e" "$SCRATCH/hub/$(basename "$e")"; done
export HF_HOME="$SCRATCH"
DIR="$SCRATCH/hub/models--${MODEL//\//--}"
cleanup() { rm -rf "$DIR"; echo "removed $MODEL from $SCRATCH"; }
trap cleanup EXIT
HF_HUB_OFFLINE=0 "$PY" -I "$REPO/experiments/run/download.py" "$MODEL"
HF_HUB_OFFLINE=1 "$@"

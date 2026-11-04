# Shared by the scripts of experiments/run/ (sourced, not run): the repository, the Python, the paper's models and their pinned revisions,
# and a runner that logs each command to build/logs/run/<name>.log and keeps going when one fails.
#   PY   the Python of the GPU environment (default: python3)        DEV  the device (default: cuda)
#   N    commands at a time on one GPU (default: 1)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
PY="${PY:-python3}"; DEV="${DEV:-cuda}"; N="${N:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG="$REPO/build/logs/run"; mkdir -p "$LOG"
reg() { "$PY" -I -c "import sys; sys.path.insert(0, 'experiments/matrix'); import registry; print($1)"; }
rev() { "$PY" -I -c "import sys; sys.path.insert(0, 'experiments/lib'); import pins; print(pins.MODELS['$1'])"; }
read -r -a SMALL <<< "$(reg "' '.join(registry.SMALL)")"
read -r -a LARGE <<< "$(reg "' '.join(registry.LARGE)")"
read -r -a SCALE <<< "$(reg "' '.join(registry.SCALE)")"
read -r -a QWEN32 <<< "$(reg "' '.join(registry.QWEN32)")"
ALL=("${SMALL[@]}" "${LARGE[@]}" "${SCALE[@]}" "${QWEN32[@]}")
tag() { echo "${1##*/}"; }
run() {   # run <name> <command ...>: one logged command
  local name=$1; shift; echo "start $name $(date +%T)"
  if "$@" > "$LOG/$name.log" 2>&1; then echo "done  $name $(date +%T)"; else echo "FAILED $name (exit $?): $LOG/$name.log"; fi
}
limit() { while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n || true; done; }   # before each background run: at most N at a time

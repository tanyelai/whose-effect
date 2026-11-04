#!/usr/bin/env bash
# The environment for the GPU runs on a Linux machine with an NVIDIA GPU: a venv with the paper's software (requirements.txt),
# a check that a torch kernel runs on the device (a recent GPU such as the B200 needs a torch built for CUDA 12.8 or later; if
# the check fails, the same torch version is reinstalled from PyTorch's CUDA indexes in turn), and OwnEffect's checks on the GPU.
# Usage: bash experiments/run/setup.sh [<venv dir>]      (default: .venv; then PY=<venv dir>/bin/python for the other scripts)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"; V="${1:-.venv}"
log() { echo "[$(date '+%F %T')] $*"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "no GPU visible: stop"; exit 1; }
PY=python3; command -v python3.12 > /dev/null && PY=python3.12
[ -x "$V/bin/python" ] || "$PY" -m venv "$V"
PIP="$V/bin/python -m pip"
$PIP install -q --upgrade pip
$PIP install -q -r requirements.txt
gpu_ok() { "$V/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA device"
x = torch.randn(256, 256, device="cuda"); y = (x @ x.T).sum().item()
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0), "| matmul ok", y == y)
PY
}
if ! gpu_ok; then
  TV=$("$V/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
  for cu in cu128 cu129 cu130; do
    log "reinstalling torch $TV from the $cu index"
    $PIP install -q --force-reinstall "torch==$TV" --index-url "https://download.pytorch.org/whl/$cu" \
      --extra-index-url https://pypi.org/simple && $PIP install -q -r requirements.txt && gpu_ok && break
  done
  gpu_ok || { echo "no torch build runs on this GPU: stop"; exit 1; }
fi
"$V/bin/python" -c "import transformers; print('transformers', transformers.__version__)"
log "OwnEffect's checks on the GPU"; "$V/bin/python" -I owneffect/test_owneffect.py cuda
log "done. Next: PY=$V/bin/python python -I experiments/run/download.py, then bash experiments/run/all.sh (README.md)"

#!/usr/bin/env bash
# The readout experiments (Section 5 and Appendix C.1) through the unit runner (experiments/matrix/run.py), package by package: DAS and
# the network's own interchange at the readout on every model, Gemma-2 again with its final-logit soft cap removed, the
# cross-entropy law (it reads the readout's outputs), and the rank sweep. A unit already done is skipped.
# Usage: bash experiments/run/readout.sh [--models=a,b] [--workers=N]      (several small-model units can share one GPU)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
for p in readout readout_uncapped readout_ce_law readout_rank; do
  "$PY" -I experiments/matrix/run.py --device="$DEV" --package="$p" "$@"
done

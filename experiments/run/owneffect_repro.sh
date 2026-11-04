#!/usr/bin/env bash
# OwnEffect (owneffect/) against the measurement scripts on GPT-2: the library recomputes the readout's and three maps'
# quantities from their saved directions and must give their saved values (Appendix F reports the largest differences).
# Usage: bash experiments/run/owneffect_repro.sh      (after bash experiments/run/readout.sh and bash experiments/run/inside.sh)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
run owneffect_repro_gpt2 "$PY" -I experiments/scripts/owneffect_repro.py --model=gpt2 --revision="$(rev gpt2)" --readout=causalgym \
  --inside=agr_sv_num_pp,npi_any_subj-relc,filler_gap_subj --data=causalgym --device="$DEV"

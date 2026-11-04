#!/usr/bin/env bash
# The NPI decomposition at the readout (Section 5.4 and Appendix C.6, Figure 11): for Kumon & Yanaka's eight NPI constructions, one-directional and
# mirrored, where the log-odds gain of the top and of cross-entropy DAS comes from. It reads the readout's saved directions
# (bash experiments/run/readout.sh first). An output already there is skipped.
# Usage: bash experiments/run/decomp.sh [<model> ...]      (default: every model of the paper)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
MODELS=("$@"); [ ${#MODELS[@]} -gt 0 ] || MODELS=("${ALL[@]}")
NPI=(NPIConditionals NPIDeterminermono NPIEmb NPIOnly NPIOnlyScope NPIQuantifiers NPISimple NPISuperlatives)
ONE=$(IFS=,; echo "${NPI[*]}"); MIR=$(printf '%s_mirrored,' "${NPI[@]}"); MIR=${MIR%,}
for m in "${MODELS[@]}"; do
  t=$(tag "$m")
  for g in kumon kumon_npi_mirrored; do
    [ -f "experiments/results/readout_decomp/readout_decomp_${t}_${g}.json" ] && continue
    tasks=$ONE; [ "$g" = kumon_npi_mirrored ] && tasks=$MIR
    run "decomp_${t}_${g}" "$PY" -I experiments/scripts/readout_decomp.py --model="$m" --revision="$(rev "$m")" --data="experiments/data/$t/kumon" \
      --group="$g" --tasks="$tasks" --device="$DEV"
  done
done

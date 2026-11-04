#!/usr/bin/env bash
# CausalGym's own code, run again (Appendix E.2). Their repository is cloned at its pinned commit
# (experiments/lib/pins.py) and run in its own environment (experiments/causalgym/rerun/requirements_frozen.txt), on the five Pythia sizes that
# CausalGym publishes up to 410m (14m, 31m, 70m, 160m and 410m), which their code loads at the Hub's current revision. The tasks
# are the 19 on which the released code reads shifted token positions, and two aligned controls. Each runs twice: as released,
# and with the two-line position fix (experiments/causalgym/rerun/position_fix.patch). Their test_all.py writes its logs to
# experiments/causalgym/rerun/<released|fixed>/logs/<released|fixed>/.
# Usage: bash experiments/run/causalgym_rerun.sh      (PY_CG: the Python of their environment, default experiments/causalgym/rerun/venv/bin/python)
set -euo pipefail
source "$(dirname "$0")/lib.sh"
D=experiments/causalgym/rerun; PY_CG="${PY_CG:-$REPO/$D/venv/bin/python}"
if [ ! -x "$PY_CG" ]; then
  python3.12 -m venv "$D/venv" && "$D/venv/bin/pip" install -q -r "$D/requirements_frozen.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128
  PY_CG="$REPO/$D/venv/bin/python"
fi
URL=$("$PY" -I -c "import sys; sys.path.insert(0, 'experiments/lib'); import pins; print(pins.CAUSALGYM_CODE[0])")
COMMIT=$("$PY" -I -c "import sys; sys.path.insert(0, 'experiments/lib'); import pins; print(pins.CAUSALGYM_CODE[1])")
TASKS="agr_gender agr_sv_num_subj-relc agr_sv_num_obj-relc agr_sv_num_pp agr_refl_num_subj-relc agr_refl_num_obj-relc agr_refl_num_pp
       garden_mvrr garden_mvrr_mod garden_npz_obj garden_npz_obj_mod garden_npz_v-trans garden_npz_v-trans_mod gss_subord
       gss_subord_subj-relc gss_subord_obj-relc gss_subord_pp cleft cleft_mod npi_any_subj-relc filler_gap_obj"
DS=""; for t in $TASKS; do DS+="syntaxgym/$t "; done
for cond in released fixed; do
  dir="$D/$cond"
  [ -d "$dir" ] || { git clone -q "$URL" "$dir"; git -C "$dir" checkout -q "$COMMIT"; }
  if [ "$cond" = fixed ] && git -C "$dir" apply --check "$REPO/$D/position_fix.patch" 2>/dev/null; then
    git -C "$dir" apply "$REPO/$D/position_fix.patch"
  fi
  for m in EleutherAI/pythia-14m EleutherAI/pythia-31m EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m; do
    run "causalgym_${cond}_$(tag "$m")" bash -c "cd '$dir' && HF_HUB_OFFLINE=0 '$PY_CG' test_all.py --model '$m' --folder '$cond' --datasets $DS"
  done
done

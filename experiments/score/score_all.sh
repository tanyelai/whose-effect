#!/usr/bin/env bash
# Every scorer, in order, from the runs' outputs (experiments/results) to the files the paper's scripts read. Each step's log goes
# to build/logs/score_<step>.log; the first failure stops the run. The steps of the optional experiments (Qwen2.5-32B's logit
# scale, the saved readout directions, CausalGym's maps at every layer, the sentiment re-read, CausalGym's own code) are skipped
# when their inputs are absent; only the last prints a note. ANY_CODE=1 scores outputs whatever code made them
# (experiments/lib/provenance.py).
# Usage: bash experiments/score/score_all.sh            (PY: the Python to use, default python3)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"; PY="${PY:-python3}"; LOG="$REPO/build/logs"; mkdir -p "$LOG"
step() { local name=$1; shift; echo "== $name $(date +%T)"
  "$@" > "$LOG/score_$name.log" 2>&1 || { echo "STOPPED at $name"; tail -20 "$LOG/score_$name.log"; exit 1; }; tail -1 "$LOG/score_$name.log"; }
have() { compgen -G "$1" > /dev/null; }

# the logits' scale of Qwen2.5-32B, when experiments/results/logit_scale.json lacks it, from its readout runs' saved label logits
# (the instruments' tolerance)
if have "experiments/results/readout/readout_Qwen2.5-32B_m_*.npz" && ! grep -q '"Qwen2.5-32B"' experiments/results/logit_scale.json 2>/dev/null; then
  step logit_scale_32b "$PY" -I experiments/score/logit_scale_from_saved.py --models=Qwen2.5-32B --hidden=5120
fi

# the readout (Section 5, App. C.1)
step readout              "$PY" -I experiments/score/score_readout.py --models=all
step readout_sentence     "$PY" -I experiments/score/score_readout_sentence.py
step ce_law               "$PY" -I experiments/score/score_readout_ce_law.py
step adequacy             "$PY" -I experiments/score/score_adequacy.py
step decomp               "$PY" -I experiments/score/score_readout_decomp.py
step scale                "$PY" -I experiments/score/score_scale.py
step rank                 "$PY" -I experiments/score/score_readout_rank.py --models=all
step cap_mechanism        "$PY" -I experiments/score/cap_mechanism.py
if have "experiments/results/readout/dirs/*.npz"; then step derive_from_dirs "$PY" -I experiments/score/derive_from_dirs.py; fi

# inside the network (Section 6, App. D.1): each set of maps, then its readings
for s in small_npi_fg_agr small_causalgym small_other small_all large_npi_fg_agr; do
  step "inside_$s" "$PY" -I experiments/score/score_inside_set.py --set=$s --again=1
done
for s in small_causalgym small_other small_all; do step "post_$s" "$PY" -I experiments/score/inside_set_post.py --set=$s; done
step post_large_npi_fg_agr     "$PY" -I experiments/score/inside_set_post.py --set=large_npi_fg_agr --descriptive=1
step offroute_small_npi_fg_agr "$PY" -I experiments/score/inside_offroute_direct.py
for s in small_npi_fg_agr small_all; do step "forms_$s" "$PY" -I experiments/score/inside_forms_post.py --set=$s; done
for s in small_causalgym small_other small_all; do step "excess_rank_$s" "$PY" -I experiments/score/inside_excess_rank.py --set=$s; done

# CausalGym's maps at every layer, the sentiment direction re-read, CausalGym's own code re-run (Section 7)
if have "experiments/results/inside_even/*.json"; then
  step causalgym_map      "$PY" -I experiments/score/score_causalgym_map.py
  step causalgym_map_post "$PY" -I experiments/score/causalgym_map_post.py
fi
if have "experiments/results/sentiment_reread_*_seed*.json"; then step sentiment "$PY" -I experiments/score/score_sentiment_reread.py; fi
if have "experiments/causalgym/rerun/released/logs/released/*.json" && have "experiments/causalgym/rerun/fixed/logs/fixed/*.json"; then
  step causalgym_rerun    "$PY" -I experiments/score/score_causalgym_rerun.py
else
  echo "== causalgym_rerun skipped: the re-run's outputs are absent (bash experiments/run/causalgym_rerun.sh)"
fi
echo "SCORED $(date +%T): next bash paper/build.sh"

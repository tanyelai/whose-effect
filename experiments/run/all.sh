#!/usr/bin/env bash
# Every GPU experiment of the paper, in order, then every scorer and the paper itself. The experiment steps skip the outputs
# already there, so a stopped run resumes; the steps for the data, the logit scale, CausalGym's own code, the OwnEffect check,
# the scorers and the paper run again. The 7-32B models need room for their weights (experiments/run/with_model.sh runs a step with one
# model on a scratch disk). The paper's Appendix B.5 gives the runs' hours on NVIDIA B200 GPUs.
# Usage: PY=<venv>/bin/python bash experiments/run/all.sh      (after bash experiments/run/setup.sh and python -I experiments/run/download.py)
set -euo pipefail
cd "$(dirname "$0")/../.."
bash experiments/run/data.sh              # each model's own task data, checked against the paper's
bash experiments/run/logit_scale.sh       # the instruments' float32 tolerance
bash experiments/run/readout.sh           # Section 5: the readout, Gemma-2 uncapped, the cross-entropy law, the rank sweep
bash experiments/run/decomp.sh            # the NPI decomposition at the readout
bash experiments/run/inside.sh            # Section 6: the maps inside the network
bash experiments/run/causalgym_maps.sh    # Section 7: CausalGym's maps at the even layers
bash experiments/run/sentiment.sh         # Section 7: the sentiment direction re-read
bash experiments/run/causalgym_rerun.sh   # Section 7: CausalGym's own code, as released and fixed
bash experiments/run/owneffect_repro.sh   # OwnEffect against the measurement scripts
bash experiments/score/score_all.sh       # every scorer
bash paper/build.sh               # the paper's numbers, tables, figures and PDF

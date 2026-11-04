"""The logits' scale (experiments/scripts/logit_scale.py's quantity) for a model whose readout runs saved their raw label logits but that
experiments/run/logit_scale.sh does not run (Qwen2.5-32B). logit_scale.py runs the model in float32 on the base and source texts of up
to --rows test rows of every task and keeps |logit| at the two label tokens of the last position; the readout runs save exactly
these logits for every test pair (<task>/raw_logits_b and _s, each pair's two labels, from the same float32 inference on the
GPU, experiments/results/readout/*.npz), in the rows' order. So S (the largest |label logit|) and the median are read from the first
--rows rows of every task the readout ran. No model is run.
Positive control, before anything is written: the same reading reproduces the stored S exactly in the six small models (10 rows
per task), leaving out the groups in NOT_IN_SCALE_RUN, whose tasks the stored scale was not measured on (asserted). In the other models with
saved runs the comparison is printed, not asserted: their stored S rests on RAVEL rows other than the first rows of the saved
runs. Gemma-2 is left out, since its saved logits are taken before its cap. Qwen2.5-32B's saved logits are on the rows
logit_scale.py reads.
Usage: python -I experiments/score/logit_scale_from_saved.py --models=Qwen2.5-32B --hidden=5120 [--rows=3]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
ROWS = int(A.get("rows", "3"))
OUT = R / "experiments/results/logit_scale.json"
U = 2.0 ** -24
NOT_IN_SCALE_RUN = ("ravel_attributes_model_specific", "ravel_attributes_sot_model_specific", "sentiment_templates_model_specific")   # groups the stored scale was not measured on


def read(tag, rows, skip=()):
    vals = []
    files = sorted(glob.glob(str(R / f"experiments/results/readout/readout_{tag}_m_*.npz")))
    assert files, f"no saved readout runs for {tag}"
    for f in files:
        if Path(f).stem.split("_m_", 1)[1] in skip:
            continue
        Z = np.load(f)
        for t in sorted({k.split("/raw_logits_")[0] for k in Z.files if "/raw_logits_b" in k}):
            for s in ("b", "s"):
                vals += list(np.abs(Z[f"{t}/raw_logits_{s}"][:rows].astype(np.float64)).ravel())
    vals.sort()
    return vals


store = json.load(open(OUT))
SMALL = ("gpt2", "pythia-160m", "pythia-410m", "SmolLM2-360M", "Qwen2.5-0.5B", "pythia-1b")
for tag, v in store["models"].items():                                       # the positive control
    if tag.startswith("gemma") or "derived_from" in v or not glob.glob(str(R / f"experiments/results/readout/readout_{tag}_m_*.npz")):
        continue                                                             # (Pythia-1.4b: the sentiment re-read only)
    S = read(tag, 10 if tag in SMALL else 3, skip=NOT_IN_SCALE_RUN)[-1]
    same = abs(S - v["max_abs_label_logit"]) <= 1e-4 * v["max_abs_label_logit"]
    assert same or tag not in SMALL, f"{tag}: saved {S} against stored {v['max_abs_label_logit']}"
    print(f"control {tag}: stored S {v['max_abs_label_logit']:.4f}, from the saved logits {S:.4f}"
          + ("" if same else " (RAVEL rows rebuilt after the scale run, a5cbc7e1)"))
assert all(t in store["models"] for t in SMALL), "the control's six small models"
for tag in A["models"].split(","):
    D = int(A["hidden"]); vals = read(tag, ROWS); gamma = D * U / (1 - D * U)
    store["models"][tag] = dict(model=f"Qwen/{tag}" if tag.startswith("Qwen") else tag, D=D, inputs=len(vals) // 4,
                                max_abs_label_logit=vals[-1], median_abs_label_logit=vals[len(vals) // 2], gamma_D=gamma,
                                bound=gamma * vals[-1], rows_per_task=ROWS,
                                derived_from="the readout runs' saved raw label logits (experiments/score/logit_scale_from_saved.py)")
    print(f"{tag}: S {vals[-1]:.4f}, median {vals[len(vals) // 2]:.4f}, D {D}, bound {gamma * vals[-1]:.3e}, inputs {len(vals) // 4}")
json.dump(store, open(OUT, "w"), indent=1)
print("written", OUT)

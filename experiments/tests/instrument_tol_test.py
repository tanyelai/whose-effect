"""Tests experiments/lib/instrument_tol.py with the logit scales in experiments/results/logit_scale.json. Needs that output of the runs (written
by experiments/run/logit_scale.sh); no model is loaded; exits 1 on a failure.
An instrument compares two float32 computations of one quantity. Its tolerance is k gamma_D S, the rounding of a dot product of
the model's width D at its largest |label logit| S, with k = 2 for a logit and 8 for an effect (App. C.1, "Numerical checks").
Checks: every model of SMALL and LARGE (experiments/matrix/registry.py) has a scale (the scale series and Qwen2.5-32B are not checked
here); the largest instrument values recorded in the runs, written in here, pass (Pythia-160m's readout, 1.6e-3 for a logit,
and its maps, 4.0e-3 for an effect; Pythia-410m's map at 1.83e-3); a patching error of a third of the model's median |T| fails
in each of the ten models of SMALL and LARGE; a model without a scale stops a real scorer and keeps 1e-3 in a synthetic test.
Usage: python -I experiments/tests/instrument_tol_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "experiments/results/logit_scale.json").is_file():
    print("SKIPPED: this check reads the models' logit scales (experiments/results/logit_scale.json); run bash experiments/run/logit_scale.sh first")
    sys.exit(0)
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/matrix"))
import instrument_tol  # noqa: E402
import registry  # noqa: E402
from instrument_tol import tol  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name, detail if not ok else "")
    if not ok:
        fails.append(name)


# each model's median |T|, written in here
MED_T = {"gpt2": 4.68, "pythia-160m": 4.16, "pythia-410m": 5.78, "SmolLM2-360M": 5.61, "Qwen2.5-0.5B": 5.63, "pythia-1b": 5.91,
         "Llama-3.1-8B": 5.51, "gemma-2-9b": 6.63, "Qwen2.5-7B": 6.02, "pythia-6.9b": 6.28}
tags = [Path(h).name for h in registry.SMALL + registry.LARGE]
check("every model of the paper's ten has a scale", all(t in instrument_tol.scales() for t in tags),
      str([t for t in tags if t not in instrument_tol.scales()]))
check("Pythia-160m's largest recorded readout deviation (1.6e-3, a logit) passes", 1.6e-3 <= tol("pythia-160m", "logit"))
check("Pythia-160m's largest recorded map deviation (4.0e-3, an effect) passes", 4.0e-3 <= tol("pythia-160m", "effect"))
check("Pythia-410m's map at 1.83e-3 (an effect) passes", 1.83e-3 <= tol("EleutherAI/pythia-410m", "effect"))
check("a patching error of a third of the median |T| fails in every model", all(MED_T[t] / 3 > tol(t, "effect") for t in MED_T),
      str({t: (round(MED_T[t] / 3, 2), round(tol(t, "effect"), 3)) for t in MED_T}))
check("a synthetic model keeps the absolute 1e-3", tol("m1", "effect", fake=True) == 1e-3)
try:
    tol("no-such-model", "logit"); stopped = False
except SystemExit:
    stopped = True
check("a model without a scale stops a real scorer", stopped)
print("\nINSTRUMENT TOLERANCE OK" if not fails else "INSTRUMENT TOLERANCE FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

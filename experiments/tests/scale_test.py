"""Tests experiments/score/score_scale.py on synthetic score_readout.json files. Synthetic data only (no model, no outputs of the runs);
exits 1 on a failure.
Within a family of three sizes s1 < s2 < s3 (Pythia, Qwen2.5, Llama; Appendix C.7), the scorer asks whether a larger model leaves
more room (rho = out-of-sample top / T; output key room_grows), whether it reports a larger excess (r = cross-entropy DAS / T;
excess_grows), and whether cross-entropy takes a smaller share of the room (sigma = r / rho; share_falls). Checks:
- a family whose room and excess grow with size and whose share falls passes all three tests;
- a family with no growth passes none;
- a family that grows from s1 to s3 but falls significantly from s2 to s3 is stopped by the middle-size guard;
- a task with a value <= 0 in the larger size enters as -inf and is listed, never dropped;
- a family with a model whose instruments failed is void (a fail).
Usage: python -I experiments/tests/scale_test.py
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "experiments/score/score_scale.py"
FAM = {"Pythia": ["pythia-410m", "pythia-1b", "pythia-6.9b"], "Qwen2.5": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B"],
       "Llama": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"]}


def row(task, r, rho):
    return dict(group="g", task=task, main=True, defined=True, logit_cap="capped", one_q=True, lam_T=rho * 1.05,
                excess=dict(ce_p1=dict(pt=r)), top_train_T=[rho, 0, 0], angle=dict(P=rho), objective=dict(p1=dict(sq_T=1.0)))


def build(spec, seed):
    rng = np.random.default_rng(seed)
    models, rows = {}, {}
    for fam, sizes in FAM.items():
        growth = spec.get(fam, (0.0, 0.0, 0.0))            # per-size log shifts of rho; sigma falls by half of it
        bases = rng.uniform(1.5, 4.0, 60)                   # one value per task, shared by the family's sizes (paired tasks)
        for j, m in enumerate(sizes):
            models[m] = dict(instruments="FAILED" if spec.get("fail") == m else "OK")
            rows[m] = []
            for i in range(60):
                base = bases[i]
                rho = base * math.exp(growth[j] + rng.normal(0, 0.02))
                sigma = 0.8 * math.exp(-0.5 * growth[j] + rng.normal(0, 0.02))
                r = rho * sigma
                if spec.get("nonpos") == m and i == 0:
                    r = -0.1
                rows[m].append(row(f"t{i}", r, rho))
    return dict(report=dict(models=models), rows=rows)


def run(spec, seed=0):
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "s.json", Path(d) / "o.json"
        json.dump(build(spec, seed), open(src, "w"))
        subprocess.run([sys.executable, "-I", str(SCORER), f"--fake={src}", f"--out={out}", "--boot=500"], check=True,
                       capture_output=True)
        return json.load(open(out))


ok = True
r = run({"Qwen2.5": (0.0, 0.15, 0.3), "Llama": (0.0, 0.0, 0.0)})
q, l = r["families"]["Qwen2.5"], r["families"]["Llama"]
print("growth:", q["room_grows"], q["excess_grows"], q["share_falls"], "| none:", l["room_grows"], l["excess_grows"], l["share_falls"])
ok &= q["room_grows"] and q["excess_grows"] and q["share_falls"] and not (l["room_grows"] or l["excess_grows"] or l["share_falls"])
r = run({"Qwen2.5": (0.0, 0.4, 0.2)})                          # up, then significantly down from s2 to s3
q = r["families"]["Qwen2.5"]
print("middle guard:", q["middle_guard_fired"], "room grows", q["room_grows"]); ok &= q["middle_guard_fired"]["rho"] and not q["room_grows"]
r = run({"Qwen2.5": (0.0, 0.15, 0.3), "nonpos": "Qwen2.5-7B"})
q = r["families"]["Qwen2.5"]
print("nonpositive listed:", q["nonpositive"], "common tasks", q["common_tasks"]); ok &= q["nonpositive"] == ["Qwen2.5-7B:g/t0"] and q["common_tasks"] == 60
r = run({"Llama": (0.0, 0.15, 0.3), "fail": "Llama-3.2-3B"})
print("void:", r["families"]["Llama"].get("void")); ok &= r["families"]["Llama"].get("void") is True
print("SCALE SCORER OK" if ok else "SCALE SCORER FAILED"); sys.exit(0 if ok else 1)

"""The readout readings that need the saved directions (experiments/results/readout/dirs/, 1.3 GB), reduced to one small file,
experiments/results/readout/derived_from_dirs.json, which paper/scripts/numbers_main.py reads.
Per model, group and task of every readout run of the models in experiments/matrix/registry.py's SMALL, LARGE, SCALE and QWEN32
(Gemma-2's runs without its cap are left out, as in the paper's numbers):
- the angle check of Section 4.3: from the saved midpoint factors A = W Q and B = Delta Q (A_i . B_i is each test pair's
  midpoint trace term, checked against the run's tr_mid), the angle between the class-signed mean gradient wbar = E[s w] and the
  class-signed mean difference E[s Delta], and between wbar and m = E[c Delta] with c_i = w_i . wbar / |wbar|^2 (Proposition
  3.4's weighting by each pair's coefficient along the shared direction), in degrees;
- the convergence check: |cos| between cross-entropy DAS's three starts, pairwise, after one pass and after ten;
- the squared error's check: |cos| of each start's per-pair squared-error direction, after one pass and after ten, with its
  first-order optimum (sq_sur).
Each dirs file is checked against the sha256 its run recorded (dirs_sha256 in the json), and its sha256 is written out.
Usage: python -I experiments/score/derive_from_dirs.py
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/matrix"))
import registry  # noqa: E402

RES = ROOT / "experiments/results/readout"


def deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))))


out = dict(source="experiments/results/readout/dirs/ (saved directions and midpoint factors) and the runs' npz files", models={})
for u in registry.units(registry.SMALL + registry.LARGE + registry.SCALE + registry.QWEN32, ["readout"], device="cuda"):
    m, g = registry.tag(u["model"]), u["group"]
    J = json.load(open(ROOT / u["output"])); fz = (ROOT / u["output"]).with_suffix(".npz"); fd = RES / "dirs" / fz.name
    h = hashlib.sha256(fd.read_bytes()).hexdigest()
    assert h == J["dirs_sha256"], f"{fd.name}: not the directions its run wrote"
    Z, Zd = np.load(fz), np.load(fd)
    tasks = {}
    for t, row in J["tasks"].items():
        if row.get("skipped"):
            continue
        A_, B_, s_ = Zd[f"{t}/A_mid"].astype(float), Zd[f"{t}/B_mid"].astype(float), Z[f"{t}/sig"]
        tr = Z[f"{t}/tr_mid"]
        assert np.allclose((A_ * B_).sum(1), tr, atol=1e-4 * max(1.0, float(np.abs(tr).max()))), f"{m} {t}: factors"
        wbar = (s_[:, None] * A_).mean(0); c_ = A_ @ wbar / (wbar @ wbar)
        cos = {}
        for p_ in (1, 10):
            v = [Zd[f"{t}/das_ce_p{p_}_s{s}"] / np.linalg.norm(Zd[f"{t}/das_ce_p{p_}_s{s}"]) for s in (0, 1, 2)]
            cos[f"p{p_}"] = [float(abs(v[0] @ v[1])), float(abs(v[0] @ v[2])), float(abs(v[1] @ v[2]))]
        # |cos| of the per-pair squared error's trained direction (each start, one and ten passes) with its first-order optimum
        sur = Zd[f"{t}/sq_sur"] / np.linalg.norm(Zd[f"{t}/sq_sur"])
        sq_cos = {f"p{p_}": [float(abs(sur @ Zd[f"{t}/das_sq_p{p_}_s{s}"]) / np.linalg.norm(Zd[f"{t}/das_sq_p{p_}_s{s}"])) for s in (0, 1, 2)]
                  for p_ in (1, 10)}
        tasks[t] = dict(angle_signed_deg=deg(wbar, (s_[:, None] * B_).mean(0)), angle_weighted_deg=deg(wbar, (c_[:, None] * B_).mean(0)),
                        start_cos=cos, sq_optimum_cos=sq_cos)
    out["models"].setdefault(m, {})[g] = dict(dirs_sha256=h, tasks=tasks)
dest = RES / "derived_from_dirs.json"
json.dump(out, open(dest, "w"), indent=1)
print(f"written {dest.relative_to(ROOT)}: {sum(len(v) for v in out['models'].values())} groups, "
      f"{sum(len(g['tasks']) for v in out['models'].values() for g in v.values())} tasks")

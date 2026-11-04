"""Tests experiments/score/score_causalgym_rerun.py: CausalGym's code as released against the same code with the two-line position fix,
on the tasks whose positions the released code reads shifted (Appendix E.2). Synthetic data only (CausalGym-format outputs with
known answers; no model, no outputs of the runs); exits 1 on a failure.
The scorer's readings, with their output keys: the instrument (on two aligned controls the deterministic methods give
identical rows in both conditions); the fix's change of the mean difference over vanilla, and that ratio's level against
vanilla (fix_change and its level); the released run against the published values (replication); DAS with the fix against vanilla and against
the best other method (das_with_fix).
Planted, for "pythia-160m": the two aligned controls identical in both conditions (the instrument holds); a shifted task where the
fix raises the mean difference from 0.3 to 0.9 of vanilla (the change above 0, the level short of vanilla), one where it changes
nothing (the change's interval holds 0); published values equal to the released run's on the first task and far from it on the
second (inside the interval and rounding to the released value, then outside); DAS twice vanilla with the fix (above vanilla and
above the best other method); a task whose fixed run read other pairs (named and left out); a task whose fixed run writes the
same pairs with its sites in another order (kept). For "pythia-410m" a control differs between the conditions: the instrument
fails and the model gets no readings.
Usage: python -I experiments/tests/causalgym_rerun_scorer_test.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["vanilla", "das", "mean", "probe_0", "probe_1", "pca", "kmeans", "lda", "random"]
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name, detail if not ok else "")
    if not ok:
        fails.append(name)


def write(folder, model, task, level, rng, noise=0.3, base_shift=0.0, reorder=False):
    """One output in CausalGym's format: per method, layers 0-2, positions 0-2, 100 pairs; level(method) the mean log odds."""
    data = []
    for m in METHODS:
        step = 99 if m == "das" else -1
        for L in range(3):
            for p in range(3):
                mu = level(m) * (1.0 if p == 1 else 0.5)
                for i in range(100):
                    o = float(mu + noise * np.sin(1 + i * 7 + L + 3 * p) + (rng.normal(0, 0.05) if m == "das" else 0.0))
                    data.append(dict(method=m, step=step, layer=L, pos=p, base_p_base=-1.0 - 0.01 * i + base_shift,
                                     base_p_src=-2.0 - 0.01 * i, p_src=o - 1.0 - 0.01 * i + 2.0 + 0.01 * i, p_base=0.0))
    if reorder:                                                        # the same rows, the sites written in another order
        data = sorted(data, key=lambda r: (-r["layer"], -r["pos"]))
    folder.mkdir(parents=True, exist_ok=True)
    json.dump(dict(metadata=dict(model=f"EleutherAI/{model}", dataset=f"syntaxgym/{task}"), data=data, vec=[]),
              open(folder / f"x__{model}__{task}__orig.json", "w"))


d = Path(tempfile.mkdtemp()); rng = np.random.default_rng(0)
base = {"vanilla": 4.0, "das": 6.0, "mean": 1.2, "probe_0": 2.0, "probe_1": 1.5, "pca": 1.0, "kmeans": 1.0, "lda": 0.1, "random": 0.0}
for model in ("pythia-160m", "pythia-410m"):
    for t in ("npi_any_subj-relc", "filler_gap_obj"):
        write(d / "released", model, t, lambda m: base[m], np.random.default_rng(1))
        ctl = (lambda m: base[m] + (0.5 if (model == "pythia-410m" and m == "mean") else 0.0))
        write(d / "fixed", model, t, ctl, np.random.default_rng(1))
    write(d / "released", model, "agr_gender", lambda m: {**base, "mean": 0.3 * base["vanilla"]}[m], rng)
    write(d / "fixed", model, "agr_gender", lambda m: {**base, "mean": 0.9 * base["vanilla"], "das": 2 * base["vanilla"]}[m], rng)
    write(d / "released", model, "cleft", lambda m: base[m], rng)
    write(d / "fixed", model, "cleft", lambda m: {**base, "das": 2 * base["vanilla"]}[m], rng)
    write(d / "released", model, "cleft_mod", lambda m: base[m], rng)               # its fixed run read other pairs
    write(d / "fixed", model, "cleft_mod", lambda m: base[m], rng, base_shift=0.5)
    write(d / "released", model, "agr_refl_num_pp", lambda m: base[m], np.random.default_rng(5))   # the same pairs, sites in another order
    write(d / "fixed", model, "agr_refl_num_pp", lambda m: base[m], np.random.default_rng(5), reorder=True)
# published values: none on the scorer's first run; then the first task's at the released run's points and the second's far off
json.dump({}, open(d / "published.json", "w"))
p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_causalgym_rerun.py"), f"--fake={d}"], capture_output=True, text=True)
check("scorer runs", p.returncode == 0, (p.stdout + p.stderr)[-800:])
S = json.load(open(d / "score_causalgym_rerun.json"))
rel = S["pythia-160m"]["tasks"]["agr_gender"]["released"]
pub = {"pythia-160m|main": {"agr_gender": {("probe0" if k == "probe_0" else "probe1" if k == "probe_1" else "rand" if k == "random" else k):
                                           round(v, 2) for k, v in rel.items()},
                            "cleft": {"mean": 99.0, "vanilla": 99.0, "das": 99.0}}}
json.dump(pub, open(d / "published.json", "w"))
p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_causalgym_rerun.py"), f"--fake={d}"], capture_output=True, text=True)
S = json.load(open(d / "score_causalgym_rerun.json")); a = S["pythia-160m"]
check("the instrument holds where the controls are identical", a["instrument"], str(a["instrument_controls_identical"]))
check("the fix raising the mean difference reads 'above'", a["tasks"]["agr_gender"]["fix_change"]["cls"] == "above", json.dumps(a["tasks"]["agr_gender"]["fix_change"]))
check("no change reads 'includes 0'", a["tasks"]["cleft"]["fix_change"]["cls"] == "includes 0", json.dumps(a["tasks"]["cleft"]["fix_change"]))
check("a published value at the released point lies inside its interval", a["tasks"]["agr_gender"]["replication"]["mean"]["inside"],
      json.dumps(a["tasks"]["agr_gender"]["replication"]["mean"]))
check("a published value far off lies outside", not a["tasks"]["cleft"]["replication"]["mean"]["inside"], json.dumps(a["tasks"]["cleft"]["replication"]["mean"]))
check("the fixed mean difference at 0.9 of vanilla reads 'short of vanilla'", a["tasks"]["agr_gender"]["fix_change"]["level"] == "short of vanilla",
      json.dumps(a["tasks"]["agr_gender"]["fix_change"]))
check("a published value equal to ours at two decimals is reproduced (rounding form)", a["tasks"]["agr_gender"]["replication"]["mean"]["rounds_to"]
      and not a["tasks"]["cleft"]["replication"]["mean"]["rounds_to"], json.dumps(a["tasks"]["agr_gender"]["replication"]["mean"]))
check("a task whose fixed run read other pairs is named and left out", a.get("pairs_differ") == ["cleft_mod"] and "cleft_mod" not in a["tasks"],
      str(a.get("pairs_differ")))
check("a task whose fixed run wrote the same pairs with its sites in another order is kept", "agr_refl_num_pp" in a["tasks"],
      str(sorted(a["tasks"])))
check("DAS at twice vanilla with the fix (two tasks) or 1.5 times (the reordered one): above vanilla and the best other method",
      a["summary"]["das_with_fix"] == {"above_vanilla": 3, "above_best_other": 3}, json.dumps(a["summary"]["das_with_fix"]))
b = S["pythia-410m"]
check("a control that differs between the conditions fails the instrument: no readings", not b["instrument"] and not b["tasks"],
      str(b["instrument_controls_identical"]))
print("\nCAUSALGYM RE-RUN SCORER OK" if not fails else "CAUSALGYM RE-RUN SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

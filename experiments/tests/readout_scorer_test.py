"""Tests experiments/score/score_readout.py: the scorer's tests at the readout (Section 5; App. C.1) and the checks that stop it.
Synthetic data only (readout outputs with known answers; no model, no registry, no outputs of the runs); exits 1 on a failure.
The scorer's tests used here, with their output keys: the midpoint trace's error against the base point's (trace); a DAS start
above the first-order top lambda_1, the ceiling (key ceiling); cross-entropy DAS above T where there is room, the excess (key excess), with the
two-label tasks without room recorded; the angle's prediction of lambda_1 / tr M (angle); cross-entropy above the squared error
(objective).

Cases, on mirrored test sets (each row with its mirror, one cluster):
- a clean model: CE between T and lambda_1, the out-of-sample top above T, the angle exact, the trace's midpoint error -1/2 of
  the base point's (the leading-order relation on mirrored sets), so Simpson's combination of the two traces is closest;
- a planted ceiling violation in one task (CE at 1.5 lambda_1 on one start);
- a task whose T is noise around 0 (not defined);
- a task with no room (top below T, DAS below T): the excess test's below-T clause does not count it, and since it is a
  two-label task it is recorded among those without room (the theory predicts room there);
- a model whose instrument fails (the model is not judged);
- a run whose resample checksum is not the regenerated one, and an npz that is not the one its json names (the scorer stops).
Usage: python -I experiments/tests/readout_scorer_test.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
from measure import boot_counts, clusters  # noqa: E402

B, SEED, SEEDS, BUD = 400, 95, (0, 1, 2), (1, 10)
DIRS = [f"das_{o}_p{p}_s{s}" for o in ("ce", "jsd", "sq", "eff") for p in BUD for s in SEEDS] + \
       ["top_exact", "top_train", "sq_sur", "b_hat_mid", "u_hat", "v1_mid", "b_hat", "bisector", "v1"]
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def task(rng, n, T0, lam, ce, sq, top=None, plant=None, bad_sha=False, inst=0.0):
    """One mirrored task: n/2 pairs, each with its mirror (equal values); lambda_1 = lam T on every resample."""
    half = rng.normal(T0, 0.1 * abs(T0) + 0.05, n // 2); T = np.repeat(half, 2)
    cl = clusters([(f"b{i}", f"s{i}") if k == 0 else (f"s{i}", f"b{i}") for i in range(n // 2) for k in range(2)])
    C = boot_counts(cl, B, SEED)
    lam_boot = np.full(B, lam * T.mean())
    top = lam if top is None else top
    eff = {d: T * 1.0 for d in DIRS}
    for d in DIRS:
        if d.startswith("das_ce"):
            eff[d] = T * ce
        elif d.startswith("das_eff"):
            eff[d] = T * min(ce + 0.1, top)
        elif d.startswith(("das_sq", "das_jsd")):
            eff[d] = T * sq
    eff["top_exact"] = T * lam; eff["top_train"] = T * top; eff["sq_sur"] = T * sq
    if plant:
        eff[plant] = T * lam * 1.5
    arrays = {"T": T, "g0": -np.abs(T) / 2, "sig": np.ones(n), "cluster": cl, "tr_mid": T * 0.98, "tr_base": T * 1.04,
              "lam_mid_boot": lam_boot, "lam_base_boot": lam_boot}
    for d, v in eff.items():
        arrays[f"e/{d}"] = v; arrays[f"fm/{d}"] = v; arrays[f"fb/{d}"] = v
    sha = hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16]
    c = 1 / (2 * (lam / 0.98) - 1)                                     # cos theta so that the angle gives lambda_1 / tr M exactly
    row = dict(T=float(T.mean()), lambda1=lam * float(T.mean()), lambda1_mid=lam * float(T.mean()), trM=1.04 * float(T.mean()),
               trM_mid=0.98 * float(T.mean()), label_pairs=1, n_test=n, n_train=400, top_exact=dict(replaced_by=None),
               grad_share_mid=1.0, boot_counts_sha="0" * 16 if bad_sha else sha, cos_v1_mid={d: 0.9 for d in DIRS},
               one_angle_mid=0.5 * (1 + 1 / c), one_angle=0.5 * (1 + 1 / c), boot_check=0.0,
               instruments=dict(train=dict(readout_vs_logits=inst, T_vs_forward=0.0), test=dict(readout_vs_logits=0.0, T_vs_forward=0.0)))
    return row, arrays


def write(d, model, group, tasks, break_npz=False):
    J = dict(data="causalgym", boot=dict(B=B, seed=SEED), tasks={}); Z = {}
    for name, (row, arr) in tasks.items():
        J["tasks"][name] = row
        Z.update({f"{name}/{k}": v for k, v in arr.items()})
    fz = d / f"readout_{model}_m_{group}.npz"
    np.savez(fz, **Z)
    J["npz_sha256"] = "0" * 64 if break_npz else hashlib.sha256(fz.read_bytes()).hexdigest()
    json.dump(J, open(d / f"readout_{model}_m_{group}.json", "w"))


def run(d):
    r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout.py"), f"--fake={d}"], capture_output=True, text=True)
    out = json.load(open(d / "score_readout.json")) if (d / "score_readout.json").is_file() else None
    return r, out


rng = np.random.default_rng(3)
d = Path(tempfile.mkdtemp())
json.dump([["sA", "sB"], []], open(d / "models.json", "w"))
write(d, "sA", "causalgym", {"t1": task(rng, 100, 4.0, 3.0, 2.5, 1.0), "t2": task(rng, 100, 3.0, 2.0, 1.6, 1.0),
                             "t5": task(rng, 100, 3.0, 4.0, 3.0, 1.0),
                             "t3": task(rng, 100, 2.0, 2.6, 2.0, 1.0, plant="das_ce_p1_s2"),   # 2.6: no tie with the others' median
                             "t4": task(rng, 100, 3.0, 0.4, 0.3, 1.0, top=0.4),
                             "t0": task(rng, 100, 0.0, 2.0, 1.5, 1.0)})
write(d, "sB", "causalgym", {"t1": task(rng, 100, 4.0, 3.0, 2.5, 1.0, inst=1e-2)})
r, out = run(d)
check("scorer runs", r.returncode == 0 and out is not None, (r.stdout + r.stderr)[-600:])
if out:
    S = out["report"]["models"]["sA"]["main"]
    check("T around 0 is not defined", S["not_defined"] == ["t0"], str(S["not_defined"]))
    check("the planted violation is the only one", S["ceiling"]["above_mid"] == 1 and S["ceiling"]["violations"][0][:2] == ["t3", "das_ce_p1_s2"],
          str(S["ceiling"]["violations"]))
    check("room: four tasks, one without", S["room"]["room"] == 4 and S["room"]["no_room"] == 1, str(S["room"]))
    x = S["excess"]["ce_p1"]
    check("inflation in every task with room; below T without room is not a failure",
          x["above_in_room"] == x["room"] == 4 and x["below_in_room"] == [] and x["below_in_no_room"] == "1/1", str(x))
    check("CE above squared error in every task with room", S["objective"]["p1"]["ce_above_sq"] == 4)
    angle_lam = S["angle"]["lam_over_tr"]
    check("an exact angle beats the other constructions' median overall and ranks the tasks",
          angle_lam["strong_form"] and angle_lam["per_task_worse"] == 0 and angle_lam["spearman"][0] > 0.99, str(angle_lam))
    check("excess: a one-question task without room is recorded", S["room"]["one_q_no_room"] == ["t4"], str(S["room"]))
    trace = S["trace"]
    check("trace: opposite signs and slope -1/2 on mirrored tasks", trace["opposite_where_both_nonzero"] == "5/5" and abs(trace["slope_mid_on_base_mirrored"] + 0.5) < 1e-9,
          str(trace))
    check("Simpson's combination is closest when the deviations are -1/2 of each other", trace["simpson_closest_where_both_nonzero"] == "5/5"
          and abs(trace["median_abs_dev_simpson_mirrored"]) < 1e-9, str(trace.get("simpson_closest_where_both_nonzero")))
    check("a failing instrument is not scored", out["report"]["models"]["sB"]["instruments"] == "FAILED")
d2 = Path(tempfile.mkdtemp()); json.dump([["sA"], []], open(d2 / "models.json", "w"))
write(d2, "sA", "causalgym", {"t1": task(rng, 100, 4.0, 3.0, 2.5, 1.0, bad_sha=True)})
r2, _ = run(d2)
check("a resample checksum that does not match stops the scorer", r2.returncode != 0 and "checksum" in (r2.stdout + r2.stderr))
d3 = Path(tempfile.mkdtemp()); json.dump([["sA"], []], open(d3 / "models.json", "w"))
write(d3, "sA", "causalgym", {"t1": task(rng, 100, 4.0, 3.0, 2.5, 1.0)}, break_npz=True)
r3, _ = run(d3)
check("an npz that is not the one its json names stops the scorer", r3.returncode == 2 and "not the npz" in r3.stdout)
for x in (d, d2, d3):
    shutil.rmtree(x)
print("\nREADOUT SCORER OK" if not fails else "READOUT SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

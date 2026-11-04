"""Tests experiments/score/score_adequacy.py: whether its test of Gemma-2's cap ranks as planted, and whether its reports run.
Synthetic data only (a synthetic score_readout.json and Gemma-2 readout outputs; no model, no outputs of the runs); exits 1 on
a failure.
The cap test (output key cap_rank) reads Gemma-2's fully mirrored main tasks. There nu, how far a swap moves the two labels' raw
logits relative to the cap c (median over test pairs, before any training), should rank delta, the cap's own change of
log(top/lambda_1): capped minus uncapped on the same task. The decision is a one-sided permutation test (p < 0.025)
(Appendix C.4, "Ranking the cap's shift before training"). The test must
- hold when the cap's own contribution grows with nu (e_cap = e_uncap + h(nu));
- fail when nu is unrelated (at most 4 of 60 trials);
- fail when nu drives the ceiling error equally with and without the cap (a curvature that is not the cap's; at most 1 of 10
  trials);
- fail when only the partly mirrored tasks are higher in both nu and the error (mirroring as a confound; the test reads only
  the fully mirrored tasks; at most 1 of 10 trials).
The descriptive reports must run without error: the cap's change of kappa (the midpoint trace's distance from T) against its
change of the ceiling's error (kappa_change); each number-only set against its parent (number_only_vs_parent, which must pair all three
number-only sets); kappa against the ceiling's error, and the readout gradients' spread against the angle's error, across
models (reports).
Usage: python -I experiments/tests/adequacy_test.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "experiments/score/score_adequacy.py"
MODELS = ["m1", "m2", "gemma-x"]
PARENTS = [f"agr_{i}" for i in range(3)]
TASKS = PARENTS + [p + "__num" for p in PARENTS] + [f"t{i}" for i in range(44)]


def row(task, e, kappa, rng, logit_cap="capped", mirrored="all", width=0.2):
    lam = 2.0; top = lam * float(np.exp(e))
    return dict(group="number_only" if task.endswith("__num") else "g", task=task, main=True, defined=True, logit_cap=logit_cap,
                one_q=True, T=float(rng.uniform(0.5, 3)), mirrored=mirrored, source="CausalGym", tr_mid_T=1 + kappa,
                tr_base_T=1 + 1.5 * kappa, grad_share_mid=float(1 - rng.uniform(0, 0.01)), lam_T=lam,
                top_exact_T=[top, top * (1 - width / 2), top * (1 + width / 2)], ceiling={"s0": dict(above=bool(e > 0.1))},
                angle=dict(P=2.0, x=dict(top_train=dict(pt=2.0))))


def build(case, seed):
    """A synthetic score_readout.json: per task, a capped row in every model and an uncapped row in Gemma-2 ("gemma-x");
    the number-only tasks and every sixth t-task are partly mirrored. Returns it with nu per task."""
    rng = np.random.default_rng(seed)
    mir = {t: ("part" if (t.endswith("__num") or (t.startswith("t") and int(t[1:]) % 6 == 0)) else "all") for t in TASKS}
    data = dict(report=dict(models={m: dict(instruments="OK") for m in MODELS}), rows={m: [] for m in MODELS})
    nu = {}
    for t in TASKS:
        n_ = float(rng.uniform(0.2, 1.5))
        e_un = abs(float(rng.normal(0.05, 0.03)))
        if case == "related":                               # the cap's own contribution grows with nu
            e_cap = e_un + 0.2 * n_ + abs(float(rng.normal(0, 0.02)))
        elif case == "same_in_both":                        # nu drives a curvature shared with and without the cap
            e_un = 0.2 * n_ + abs(float(rng.normal(0, 0.01))); e_cap = e_un + abs(float(rng.normal(0, 0.01)))
        elif case == "mirroring":                           # only the partly mirrored tasks are higher in both
            if mir[t] != "all":
                n_ += 1.0; e_cap = e_un + 0.5 + abs(float(rng.normal(0, 0.02)))
            else:
                e_cap = e_un + abs(float(rng.normal(0.1, 0.03)))
        else:                                               # unrelated
            e_cap = e_un + abs(float(rng.normal(0.1, 0.05)))
        nu[t] = n_
        w = float(rng.uniform(0.05, 0.6))
        for m in MODELS:
            e = e_cap if m == "gemma-x" else abs(float(rng.normal(0.03, 0.02)))
            data["rows"][m].append(row(t, e, float(rng.uniform(0, 0.1)), rng, mirrored=mir[t], width=w))
        data["rows"]["gemma-x"].append(row(t, e_un, float(rng.uniform(0, 0.1)), rng, logit_cap="uncapped", mirrored=mir[t], width=w))
    return data, nu


def write_runs(d, data, nu, seed):
    """Gemma-2's readout outputs with and without the cap, per group: raw label logits for base and source, two labels each
    moved by c nu / 2 with c = 30, so that the scorer's nu (median over test pairs of (|d o_s| + |d o_b|) / c) is nu; the exact
    top's three starts end within 0.2% of each other, so they agree."""
    rng = np.random.default_rng(seed)
    (Path(d) / "readout").mkdir()
    for logit_cap, suf in (("capped", ""), ("uncapped", "__uncapped")):
        for group in ("g", "number_only"):
            arrays, tasks = {}, {}
            for r in data["rows"]["gemma-x"]:
                if r["logit_cap"] != logit_cap or r["group"] != group:
                    continue
                b = rng.normal(40, 3, (50, 2)); arrays[f"{r['task']}/raw_logits_b"] = b
                arrays[f"{r['task']}/raw_logits_s"] = b + 30 * nu[r["task"]] / 2 * np.sign(rng.normal(size=(50, 2)))
                tasks[r["task"]] = dict(top_exact=dict(replaced_by=None,
                                                       starts=[dict(loss=-1.0), dict(loss=-0.999), dict(loss=-0.998)]))
            np.savez(Path(d) / "readout" / f"readout_gemma-x_m_{group}{suf}.npz", **arrays)
            json.dump(dict(cap=30.0, tasks=tasks), open(Path(d) / "readout" / f"readout_gemma-x_m_{group}{suf}.json", "w"))


def run(case, seed, boot=2000, perm=10000):
    with tempfile.TemporaryDirectory() as d:
        data, nu = build(case, seed)
        src, out = Path(d) / "s.json", Path(d) / "o.json"
        json.dump(data, open(src, "w"))
        write_runs(d, data, nu, seed)
        p = subprocess.run([sys.executable, "-I", str(SCORER), f"--fake={src}", f"--out={out}", f"--boot={boot}", f"--perm={perm}"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stderr[-2000:]); raise SystemExit("scorer failed")
        return json.load(open(out))


ok = True
r = run("related", 1)
print("cap ranking related:", r["cap_rank"]["holds"], round(r["cap_rank"]["main_test"]["pt"], 3), r["cap_rank"]["dropped"]); ok &= r["cap_rank"]["holds"]
print("   reports ran:", sorted(r["cap_rank"]["reports"]), "kappa_change" if "kappa_change" in r else "-", "number_only_vs_parent" if "number_only_vs_parent" in r else "-",
      sorted(r["reports"]))
ok &= "kappa_change" in r and "number_only_vs_parent" in r and r["number_only_vs_parent"]["n"] == 3
fp = sum(run("unrelated", s, boot=200, perm=1000)["cap_rank"]["holds"] for s in range(60))
print(f"cap ranking unrelated: holds in {fp}/60"); ok &= fp <= 4
fp = sum(run("same_in_both", s, boot=200, perm=1000)["cap_rank"]["holds"] for s in range(100, 110))
print(f"cap ranking same curvature with and without the cap: holds in {fp}/10"); ok &= fp <= 1
fp = sum(run("mirroring", s, boot=200, perm=1000)["cap_rank"]["holds"] for s in range(200, 210))
print(f"cap ranking mirroring confound: holds in {fp}/10"); ok &= fp <= 1
print("ADEQUACY SCORER OK" if ok else "ADEQUACY SCORER FAILED"); sys.exit(0 if ok else 1)

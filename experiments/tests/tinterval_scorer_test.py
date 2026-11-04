"""Tests experiments/score/score_readout_tinterval.py on synthetic readout outputs with known answers, and the coverage of its intervals
on simulated data. Synthetic data only (no model, no outputs of the runs); exits 1 on a failure.
The scorer reads the readout's counts without a minimum number of clusters: a ratio of means with the bias-reduced cluster
variance CR2 and Student's t on G* - 1 degrees of freedom (G* the effective number of clusters), under the mirror and the
sentence clusters, and for RAVEL also a two-way interval by the entity of each value (ratio_twoway, which experiments/lib/ravel_twoway.py
uses for the readout's RAVEL counts; Appendix B.4). It reads the excess (output key excess: cross-entropy DAS above T where there is
room) and cross-entropy above the squared error (objective).
Planted, in one model: a task with 50 mirror clusters and DAS at 1.5 T (above under both clusterings); a task with 10 clusters,
which score_readout.py's 20-cluster rule leaves undefined, with DAS at 1.5 T (defined and above here, named as a change); a
task whose rows all share one base input (one sentence cluster: undefined at the sentence level, counted); a null task with DAS
at T (not above); a task whose rows cannot be rebuilt (counted apart, not as undefined); a RAVEL task, read also two-way by the
entity of each value. The CR2 variance and the effective number of clusters are checked against their formulas by hand, and
the interval's coverage against 95% on simulated clustered ratios at balanced sizes and at unequal sentence-level sizes taken
from real RAVEL tasks (2,000 draws each; the CR1 interval with t on G - 1 is printed beside it for comparison). The two-way
interval's coverage is simulated on RAVEL-like designs (1,000 draws each), with the sentence clustering's beside it.
Usage: python -I experiments/tests/tinterval_scorer_test.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/score"))
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name, detail if not ok else "")
    if not ok:
        fails.append(name)


def task_arrays(name, n_clusters, das_over_T, rng, shared_base=False):
    """Rows in mirror pairs (two rows per cluster), T near 3, every DAS start at das_over_T times T plus noise; returns the
    npz entries and the test rows (base/src texts: distinct per pair unless shared_base)."""
    n = 2 * n_clusters; cl = np.repeat(np.arange(n_clusters), 2)
    T = 3.0 + rng.normal(0, 0.3, n_clusters)[cl] + rng.normal(0, 0.1, n)
    Z = {f"{name}/T": T, f"{name}/cluster": cl, f"{name}/e/top_train": 2.0 * T}
    for obj in ("ce", "eff", "sq"):
        for p in (1, 10):
            for s in (0, 1, 2):
                f = das_over_T if obj != "sq" else 1.0
                Z[f"{name}/e/das_{obj}_p{p}_s{s}"] = f * T + rng.normal(0, 0.05, n_clusters)[cl] + rng.normal(0, 0.02, n)
    rows = []
    for c in range(n_clusters):
        a, b = ("same", "same") if shared_base else (f"base {c}", f"src {c}")
        rows += [dict(base=[a], src=[b]), dict(base=[b], src=[a])]
    return Z, rows


d = Path(tempfile.mkdtemp()); rng = np.random.default_rng(0)
Z, srows = {}, []
spec = [("big", 50, 1.5, False, True), ("small", 10, 1.5, False, False), ("one", 25, 1.5, True, True), ("null", 50, 1.0, False, True),
        ("gone", 30, 1.5, False, True)]
for name, G, f, shared, boot_defined in spec:
    z, rows = task_arrays(name, G, f, rng, shared); Z.update(z)
    if name != "gone":                                                # "gone": its rows cannot be rebuilt
        (d / "fakedata" / name).mkdir(parents=True); json.dump(rows, open(d / "fakedata" / name / "test.json", "w"))
    srows.append(dict(group="g", task=name, main=True, logit_cap="capped", source="Fake", defined=boot_defined, room=boot_defined or None,
                      excess={k: dict(above=boot_defined and f > 1) for k in ("ce_p1", "ce_p10", "eff_p1", "eff_p10")} if boot_defined else {},
                      objective={p: dict(ce_above_sq=boot_defined and f > 1) for p in ("p1", "p10")} if boot_defined else {}))
np.savez(d / "readout_fake_m_g.npz", **Z)          # --fake reads the outputs from the folder itself
sha = hashlib.sha256((d / "readout_fake_m_g.npz").read_bytes()).hexdigest()
json.dump(dict(npz_sha256=sha, data="fakedata"), open(d / "readout_fake_m_g.json", "w"))
# a RAVEL task (prompts, value types; four entities per value, every entity under several prompts) for the two-way reading
zr, _ = task_arrays("rav", 30, 1.5, rng); rrows = []
for c in range(30):
    pr = f"Q{c % 15}: %s is in"; ea, eb = f"A{c % 4}", f"B{(c // 4) % 4}"
    for (x, tx), (y_, ty) in (((ea, "a"), (eb, "b")), ((eb, "b"), (ea, "a"))):
        rrows.append(dict(base=[f"Q{c % 15}:", " " + x, " is in"], src=[f"Q{c % 15}:", " " + y_, " is in"], prompt=pr,
                          base_type=tx, src_type=ty))
(d / "fakerav" / "rav").mkdir(parents=True); json.dump(rrows, open(d / "fakerav" / "rav" / "test.json", "w"))
np.savez(d / "readout_fake_m_ravel_attributes.npz", **zr)
json.dump(dict(npz_sha256=hashlib.sha256((d / "readout_fake_m_ravel_attributes.npz").read_bytes()).hexdigest(), data="fakerav"),
          open(d / "readout_fake_m_ravel_attributes.json", "w"))
srows.append(dict(group="ravel_attributes", task="rav", main=True, logit_cap="capped", source="RAVEL", defined=True, room=True,
                  excess={k: dict(above=True) for k in ("ce_p1", "ce_p10", "eff_p1", "eff_p10")}, objective={p: dict(ce_above_sq=True) for p in ("p1", "p10")}))
json.dump(dict(report=dict(models=dict(fake=dict(instruments="OK"))), rows=dict(fake=srows)), open(d / "score_readout.json", "w"))
p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout_tinterval.py"), f"--fake={d}"], capture_output=True, text=True)
check("scorer runs", p.returncode == 0, (p.stdout + p.stderr)[-1500:])
O = json.load(open(d / "score_readout_tinterval.json"))["models"]["fake"]
by = {t["task"]: t for t in O["tasks"]}
check("50 clusters, DAS at 1.5 T: above in both clusterings", by["big"]["mirror"]["excess"]["ce_p1"]["above"] and by["big"]["sentence"]["excess"]["ce_p1"]["above"])
check("10 clusters, left undefined by the 20-cluster rule: defined and above here, and named",
      by["small"]["mirror"]["defined"] and by["small"]["mirror"]["excess"]["ce_p1"]["above"]
      and ["g", "small"] in O["summary"]["mirror"]["excess"]["ce_p1"]["above_here_only"], json.dumps(O["summary"]["mirror"]["excess"]["ce_p1"]))
check("rows sharing one base input: one sentence cluster, undefined there, counted",
      by["one"]["sentence"]["G"] == 1 and not by["one"]["sentence"]["defined"] and O["summary"]["sentence"]["one_cluster"] == 1)
check("a task whose rows cannot be rebuilt is counted apart, not as undefined",
      "sentence" not in by["gone"] and [x[1] for x in O["summary"]["sentence"]["skipped"]] == ["gone"]
      and O["summary"]["sentence"]["tasks"] == len(by) - 1, json.dumps(O["summary"]["sentence"]["skipped"]))
check("RAVEL: the two-way reading by each value's entity is computed there (4 entities a value), and only there",
      "twoway" in by["rav"] and by["rav"]["twoway"]["G"] == 4 and "twoway" not in by["big"] and O["summary"]["twoway"]["tasks"] == 1,
      json.dumps({k: by["rav"].get("twoway", {}).get(k) for k in ("G", "G_eff", "defined")}))
check("a null task (DAS at T) is not above", not by["null"]["mirror"]["excess"]["ce_p1"]["above"])
check("objective: cross-entropy above squared error where DAS is at 1.5 T, not where it is at T",
      by["big"]["mirror"]["objective"]["p1"]["ce_above_sq"] and not by["null"]["mirror"]["objective"]["p1"]["ce_above_sq"])
# the formulas by hand
from scipy import stats  # noqa: E402
from score_readout_tinterval import ratio  # noqa: E402
y, x, cl = np.array([1.0, 2.0, 3.0, 5.0, 4.0, 6.0]), np.array([1.0, 1.0, 2.0, 2.0, 1.0, 3.0]), np.array([0, 0, 1, 1, 2, 2])
Y, X = np.array([3.0, 8.0, 10.0]), np.array([2.0, 4.0, 4.0]); Rh = Y.sum() / X.sum(); h = X / X.sum()
se = np.sqrt(np.sum((Y - Rh * X) ** 2 / (1 - h))) / X.sum(); Gs = 1 / np.sum(h ** 2)
got = ratio(y, x, cl)
check("the CR2 variance and the effective number of clusters equal their formulas",
      abs(got[0] - Rh) < 1e-12 and abs(got[3] - se) < 1e-12 and abs(got[5] - Gs) < 1e-12, str(got))


def cr1(yy, xx, cc):                                                   # the CR1 interval with t on G - 1, for comparison only
    ids, inv = np.unique(cc, return_inverse=True); G = len(ids)
    Yc, Xc = np.bincount(inv, weights=yy), np.bincount(inv, weights=xx); R_ = Yc.sum() / Xc.sum()
    s_ = np.sqrt(G / (G - 1) * np.sum((Yc - R_ * Xc) ** 2)) / abs(Xc.sum()); q = stats.t.ppf(0.975, G - 1)
    return R_ - q * s_, R_ + q * s_


# coverage on simulated clustered ratios: rows with a cluster effect in T and in the numerator, at balanced sizes and at the
# unequal sentence-level sizes of real RAVEL tasks (GPT-2's ravel_physical_object_size_json; Pythia-6.9b's ravel_physical_object_size)
for sizes in ([2] * 5, [2] * 10, [2] * 30, [54, 40, 4, 2], [50, 30, 6, 4, 4, 2, 2, 2]):
    cc = np.repeat(np.arange(len(sizes)), sizes); hit = cr1_hits = 0
    for _ in range(2000):
        tt = 3.0 + rng.normal(0, 0.6, len(sizes))[cc] + rng.normal(0, 0.3, len(cc))
        yy = 1.3 * tt + rng.normal(0, 0.8, len(sizes))[cc] + rng.normal(0, 0.4, len(cc))
        r = ratio(yy, tt, cc); hit += r[1] <= 1.3 <= r[2]; lo_, hi_ = cr1(yy, tt, cc); cr1_hits += lo_ <= 1.3 <= hi_
    check(f"cluster sizes {sizes[:8]}: the CR2 interval on G* - 1 covers {hit / 2000:.3f}, near 95% (2,000 draws; the earlier CR1 on G - 1: "
          f"{cr1_hits / 2000:.3f})", 0.92 <= hit / 2000 <= 0.99, f"{hit / 2000:.3f}")
# RAVEL-like designs (nA, nB entities of the two values, 15 prompts, 50 pairs drawn without replacement, each with its
# mirror; entity effects on T and on the numerator of both values): the two-way interval against the sentence clustering's
from score_readout_tinterval import ratio_twoway, sentence_clusters  # noqa: E402
for nA, nB in ((3, 3), (4, 8), (10, 10), (30, 30)):
    hit = hit_s = defined = 0
    combos = [(q, a_, b_) for q in range(15) for a_ in range(nA) for b_ in range(nB)]
    for _ in range(1000):
        pick = rng.choice(len(combos), size=min(50, len(combos)), replace=False)
        rws = [(*combos[k], m) for k in pick for m in (0, 1)]
        al, be, ga, de = (rng.normal(0, 0.5, n) for n in (nA, nB, nA, nB))
        xx = np.array([3 + al[a_] + be[b_] + rng.normal(0, 0.2) for _, a_, b_, _ in rws])
        yy = np.array([1.3 * xv + ga[a_] + de[b_] + rng.normal(0, 0.2) for xv, (_, a_, b_, _) in zip(xx, rws)])
        r = ratio_twoway(yy, xx, [f"a{a_}" for _, a_, _, _ in rws], [f"b{b_}" for _, _, b_, _ in rws])
        defined += r[1] is not None; hit += r[1] is not None and r[1] <= 1.3 <= r[2]
        keys = [(f"{q}|{(a_, b_)[m]}", f"{q}|{(b_, a_)[m]}") for q, a_, b_, m in rws]
        rs = ratio(yy, xx, sentence_clusters(keys)); hit_s += rs[1] is not None and rs[1] <= 1.3 <= rs[2]
    check(f"RAVEL-like design, {nA} and {nB} entities: the two-way interval covers {hit / max(defined, 1):.3f} (the sentence "
          f"clustering: {hit_s / 1000:.3f})", defined == 1000 and hit / defined >= 0.92, f"{hit / max(defined, 1):.3f}")
print("\nT-INTERVAL SCORER OK" if not fails else "T-INTERVAL SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

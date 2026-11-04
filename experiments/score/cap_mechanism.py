"""Gemma-2's soft cap, an exploratory reading (Appendix C.4, "Mechanism (exploratory)"): why the first-order description is looser with the final logit
soft cap, computed from the saved raw (pre-cap) label logits of the test pairs.

Per main-population task of gemma-2-9b, with the cap and without it (same pairs, same hidden states; the two runs differ only in
the cap c):
- depth: median over (pair, label) of |o_mid| / c, the raw label logits at the full swap's midpoint;
- bend: median of (|o_mid| - |o_lin|) / c, with o_lin = (o_base + o_source) / 2 the straight-line midpoint, and the share of
  (pair, label) where the midpoint lies further out (the final norm bends the raw-logit path);
- the cap's share of the trace's first-order error from the raw logits: log(G / F) with G the exact capped change of the log
  odds and F the midpoint value g'(o_mid) * (o_source - o_base) summed over the two labels (g = c tanh(o / c)), at the recorded
  midpoint and at the straight-line midpoint;
- the same share measured with the real gradients: log(T / tr M^mid) with the cap minus without it;
- delta = log(exact top / lambda_1), with the cap minus without it (the quantity experiments/score/score_adequacy.py tests).
Rank correlations over the fully mirrored tasks (score_adequacy.py's population) and per source; the same over the means of
the label families, and a permutation test within families.
Usage: python -I experiments/score/cap_mechanism.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = json.load(open(ROOT / "experiments/results/score_readout.json"))
RD = ROOT / "experiments/results/readout"
M = "gemma-2-9b"
INFLECTION = 0.6585                                   # |o|/c where tanh''' changes sign: 3 tanh^2 = 1


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1]) if len(a) > 2 else float("nan")


rows = {(r["group"], r["task"], r["logit_cap"]): r for r in S["rows"][M]}
main = [k[:2] for k, r in rows.items() if k[2] == "capped" and r["main"] and r["defined"]]
cache = {}


def run(group, logit_cap):
    key = (group, logit_cap)
    if key not in cache:
        stem = RD / f"readout_{M}_m_{group}{'__uncapped' if logit_cap == 'uncapped' else ''}"
        cache[key] = (json.load(open(stem.with_suffix(".json"))), np.load(stem.with_suffix(".npz")))
    return cache[key]


out = []
for g, t in main:
    rc, ru = rows[(g, t, "capped")], rows.get((g, t, "uncapped"))
    if ru is None or not ru["defined"]:
        continue
    J, Z = run(g, "capped")
    c = J["cap"]
    ob, os_, om = Z[f"{t}/raw_logits_b"], Z[f"{t}/raw_logits_s"], Z[f"{t}/raw_logits_m"]   # columns (y_s, y_b)
    Zu = run(g, "uncapped")[1]
    same = bool(np.array_equal(Zu[f"{t}/raw_logits_b"], ob) and np.array_equal(Zu[f"{t}/raw_logits_s"], os_))
    sgn = np.array([1.0, -1.0])
    cap = lambda o: c * np.tanh(o / c)  # noqa: E731
    slope = lambda o: 1 / np.cosh(o / c) ** 2  # noqa: E731
    do, lin = os_ - ob, (ob + os_) / 2
    G = ((cap(os_) - cap(ob)) * sgn).sum(1).sum()
    F_rec = ((slope(om) * do) * sgn).sum(1).sum()
    F_lin = ((slope(lin) * do) * sgn).sum(1).sum()
    rec = dict(group=g, task=t, source=rc["source"], mirrored=rc["mirrored"], raw_same_with_and_without_cap=same,
               depth=float(np.median(np.abs(om) / c)), past_inflection=float((np.abs(om) / c > INFLECTION).mean()),
               bend=float(np.median((np.abs(om) - np.abs(lin)) / c)), bend_outward=float((np.abs(om) > np.abs(lin)).mean()),
               # Lean cap_undershoot's hypotheses per (pair, label): the straight segment stays on one side past the inflection
               # (g' convex there) and the midpoint is bent outward (where sech^2 is smaller)
               conditions=float(((np.sign(ob) == np.sign(os_)) & (np.minimum(np.abs(ob), np.abs(os_)) / c > INFLECTION)
                                 & (np.abs(om) >= np.abs(lin))).mean()),
               cap_share_rec=math.log(G / F_rec) if G > 0 and F_rec > 0 else None,
               cap_share_lin=math.log(G / F_lin) if G > 0 and F_lin > 0 else None,
               cap_share_measured=math.log(1 / rc["tr_mid_T"]) - math.log(1 / ru["tr_mid_T"]),
               nu=float(np.median(np.abs(do).sum(1) / c)), move=float(np.median(np.abs(do) / c)),
               delta=math.log(rc["top_exact_T"][0] / rc["lam_T"]) - math.log(ru["top_exact_T"][0] / ru["lam_T"]))
    out.append(rec)

mir = [r for r in out if r["mirrored"] == "all"]
ok = [r for r in mir if r["cap_share_rec"] is not None and r["cap_share_lin"] is not None]
res = dict(tasks=len(out), mirrored=len(mir), raw_logits_same_with_and_without_cap=all(r["raw_same_with_and_without_cap"] for r in out),
           median_depth=float(np.median([r["depth"] for r in mir])),
           median_past_inflection=float(np.median([r["past_inflection"] for r in mir])),
           median_bend=float(np.median([r["bend"] for r in mir])),
           median_bend_outward=float(np.median([r["bend_outward"] for r in mir])),
           median_conditions=float(np.median([r["conditions"] for r in mir])),
           pooled_conditions=float(np.mean([r["conditions"] for r in mir])),
           min_bend_outward=float(min(r["bend_outward"] for r in mir)),
           median_cap_share_recorded_midpoint=float(np.median([r["cap_share_rec"] for r in ok])),
           median_cap_share_straight_midpoint=float(np.median([r["cap_share_lin"] for r in ok])),
           median_cap_share_measured=float(np.median([r["cap_share_measured"] for r in mir])),
           spearman_rawlogit_vs_measured=spearman([r["cap_share_rec"] for r in ok], [r["cap_share_measured"] for r in ok]),
           spearman_depth_delta=spearman([r["depth"] for r in mir], [r["delta"] for r in mir]),
           spearman_nu_delta=spearman([r["nu"] for r in mir], [r["delta"] for r in mir]),
           spearman_measured_share_delta=spearman([r["cap_share_measured"] for r in mir], [r["delta"] for r in mir]),
           median_delta=float(np.median([r["delta"] for r in mir])),
           spearman_nu_delta_non_causalgym=spearman([r["nu"] for r in mir if r["source"] != "CausalGym"],
                                                    [r["delta"] for r in mir if r["source"] != "CausalGym"]),
           n_non_causalgym=sum(r["source"] != "CausalGym" for r in mir),
           median_move_per_label=float(np.median([r["move"] for r in mir])),
           per_source={s: dict(n=len(v), spearman_nu_delta=spearman([r["nu"] for r in v], [r["delta"] for r in v]))
                       for s in sorted({r["source"] for r in mir}) for v in [[r for r in mir if r["source"] == s]]},
           rows=out)
# label families (the test rows' label pairs) of the mirrored tasks: is the ranking carried between families or within them?
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/matrix"))
import registry  # noqa: E402

CG = Path.home() / f".cache/huggingface/hub/datasets--aryaman--causalgym/snapshots/{registry.CAUSALGYM_REV}/test.json"
cg = json.load(open(CG)) if CG.is_file() else None
fam = {}
for r in mir:
    data = run(r["group"], "capped")[0]["data"]
    if data == "causalgym":
        if cg is None:
            continue
        tr = [x for x in cg if x["task"] == r["task"]]
    else:
        tr = json.load(open(ROOT / data / r["task"] / "test.json"))
    fam[(r["group"], r["task"])] = tuple(sorted(tuple(sorted(p)) for p in {frozenset((x["base_label"], x["src_label"])) for x in tr}))
if len(fam) == len(mir):
    fams = sorted(set(fam.values()))
    means = [(np.mean([r["nu"] for r in mir if fam[(r["group"], r["task"])] == F]),
              np.mean([r["delta"] for r in mir if fam[(r["group"], r["task"])] == F])) for F in fams]
    res["label_families"] = len(fams)
    res["spearman_family_means"] = spearman([a for a, _ in means], [b for _, b in means])
    rng = np.random.default_rng(7)
    nu_, de_ = np.array([r["nu"] for r in mir]), np.array([r["delta"] for r in mir])
    lab = np.array([fams.index(fam[(r["group"], r["task"])]) for r in mir])
    obs, hits, P = spearman(nu_, de_), 0, 20000
    for _ in range(P):
        x = nu_.copy()
        for k in np.unique(lab):
            idx = np.where(lab == k)[0]; x[idx] = nu_[rng.permutation(idx)]
        hits += spearman(x, de_) >= obs - 1e-12
    res["within_family_permutation_p"] = (hits + 1) / (P + 1)
else:
    res["label_families"] = None                       # CausalGym's pinned snapshot is not in the local cache

# the tanh expansion at the median depth and per-label move: exact midpoint-rule error of one label against its leading term
o_, m_ = res["median_depth"], res["median_move_per_label"]
res["tanh_midpoint_error_exact"] = (math.tanh(o_ + m_ / 2) - math.tanh(o_ - m_ / 2)) / (m_ / math.cosh(o_) ** 2) - 1
res["tanh_midpoint_error_leading"] = (3 * math.tanh(o_) ** 2 - 1) / 12 * m_ ** 2
json.dump(res, open(ROOT / "experiments/results/cap_mechanism.json", "w"), indent=1)
for k, v in res.items():
    if k != "rows":
        print(k, v if not isinstance(v, float) else round(v, 4))

"""The NPI decomposition's scorer (Figure 11; Section 5.4 and Appendix C.6).

Per model, over Kumon & Yanaka's NPI constructions: the share of a direction's log-odds effect that comes from lowering the base
label's logit, -mean(dz_b) / mean(dz_s - dz_b). Predictions (medians over tasks):
  1. one-directional sets (group kumon): the out-of-sample top's share > 1/2;
  2. one-directional sets: ce's share (its three starts' mean) minus the top's < 0;
  3. the gap (top - ce) is smaller on the mirrored sets (group kumon_npi_mirrored) than on the one-directional sets;
  4. (report) the base's median p_b on the one-directional sets, the theory's precondition (< 0.1).
Usage: python -I experiments/score/score_readout_decomp.py [--fake=<dir>]
"""
import json
import statistics as st
import sys
from pathlib import Path

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
ROOT = Path(__file__).resolve().parents[2]
RES = Path(A["fake"]) if A.get("fake") else ROOT / "experiments/results/readout_decomp"
CE = ("das_ce_p1_s0", "das_ce_p1_s1", "das_ce_p1_s2")


def shares(J):
    out = {}
    for t, r in J["tasks"].items():
        d = r["dirs"]
        if "top_train" not in d or d["top_train"]["share_lowering_b"] is None:
            continue
        ce = [d[k]["share_lowering_b"] for k in CE if k in d and d[k]["share_lowering_b"] is not None]
        if ce:
            out[t.replace("_mirrored", "")] = dict(top=d["top_train"]["share_lowering_b"], ce=st.mean(ce), p_b=r["median_p_b"])
    return out


models = sorted({f.stem.split("readout_decomp_", 1)[1].rsplit("_kumon", 1)[0] for f in RES.glob("readout_decomp_*_kumon*.json")})
report = {}
for m in models:
    f1, f2 = RES / f"readout_decomp_{m}_kumon.json", RES / f"readout_decomp_{m}_kumon_npi_mirrored.json"
    if not (f1.is_file() and f2.is_file()):
        report[m] = "missing an output"; continue
    one, mir = shares(json.load(open(f1))), shares(json.load(open(f2)))
    common = sorted(set(one) & set(mir))
    if not common:
        report[m] = "no task in both"; continue
    p1 = st.median(one[t]["top"] for t in common)
    p2 = st.median(one[t]["ce"] - one[t]["top"] for t in common)
    gap_one = st.median(one[t]["top"] - one[t]["ce"] for t in common); gap_mir = st.median(mir[t]["top"] - mir[t]["ce"] for t in common)
    report[m] = dict(tasks=len(common), top_share_one=p1, pred1=p1 > 0.5, ce_minus_top_one=p2, pred2=p2 < 0,
                     gap_one=gap_one, gap_mirrored=gap_mir, pred3=gap_mir < gap_one,
                     median_p_b_one=st.median(one[t]["p_b"] for t in common),
                     per_task={t: dict(one=one[t], mirrored=mir[t]) for t in common})
dest = RES / "score_readout_decomp.json"
json.dump(report, open(dest, "w"), indent=1)
for m, r in report.items():
    if isinstance(r, str):
        print(m, r); continue
    print(f"{m}: {r['tasks']} tasks; top's share of lowering ' ever' {r['top_share_one']:.2f} (pred 1 {r['pred1']}); ce - top "
          f"{r['ce_minus_top_one']:+.2f} (pred 2 {r['pred2']}); gap one-directional {r['gap_one']:.2f}, mirrored {r['gap_mirrored']:.2f} "
          f"(pred 3 {r['pred3']}); median p_b {r['median_p_b_one']:.3g}")
print(f"written {dest}")

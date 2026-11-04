"""Size within a family (Section 5.4 and Appendix C.7, Tables 6 and 7): within a family, does a larger model leave more room (key room_grows,
rho = out-of-sample top / T), report a larger excess (excess_grows, r = cross-entropy DAS / T), and does cross-entropy take a smaller
share of the top (share_falls, sigma = r / rho)?

Families (same tokenizer, data and labels): Pythia 410m, 1b, 6.9b; Qwen2.5 0.5B, 1.5B, 7B; Llama 3.2-1B, 3.2-3B, 3.1-8B. Per
family, the common task set: main-population tasks (T's interval above 0, defined) in all three sizes. Per task, log
differences between sizes; D_q(s_i, s_j) is their median, with a task bootstrap (B = 2000, seed 97, percentile 95%).
- The room grows in a family iff the intervals of D_rho(s1, s2) and D_rho(s1, s3) lie entirely above 0; the excess likewise for
  log r.
- The share falls iff the interval of D_sigma(s1, s3) lies entirely below 0.
- Middle-size guard: D(s2, s3) is reported for every quantity; if its interval lies entirely below 0 for the room or the
  excess, the family does not pass that quantity.
- Values at or below 0: log is undefined; a value <= 0 in the larger size enters as -inf, in the smaller as +inf; every such
  task is listed by name.
- A model without held instruments voids its family (a fail, reported).
- Qwen2.5's step from 7B to 32B: the same three questions on D(s3, s4) (key qwen32).
Reports: per size, the angle's P on two-label tasks and lambda_1/T on all tasks, and the median |sq/T - 1|; growth per data
source, and the step intervals with one source left out at a time.

Usage: python -I experiments/score/score_scale.py [--src=<score json>] [--fake=<score json>] [--out=<path>] [--boot=<B>]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
# the readout scores of every model, experiments/results/score_readout.json (score_readout.py --models=all); --src names another
# file, --fake a synthetic one (tests)
SRC = Path(A.get("src") or A.get("fake") or ROOT / "experiments/results/score_readout.json")
OUT = Path(A.get("out", ROOT / "experiments/results/score_scale.json"))
B, SEED = int(A.get("boot", 2000)), 97
FAMILIES = {"Pythia": ["pythia-410m", "pythia-1b", "pythia-6.9b"],
            "Qwen2.5": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B"],
            "Llama": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"]}


def logdiff(a, b):
    """log b - log a, with the module docstring's rule for values at or below 0 (b in the larger size)."""
    if a > 0 and b > 0:
        return math.log(b) - math.log(a)
    if b <= 0 and a > 0:
        return -math.inf
    if a <= 0 and b > 0:
        return math.inf
    return float("nan")                                  # both <= 0: undefined either way; listed


def med_ci(d, rng):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return dict(pt=None, lo=None, hi=None, n=0)
    boots = [float(np.median(d[rng.integers(0, len(d), len(d))])) for _ in range(B)]
    return dict(pt=float(np.median(d)), lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)), n=int(len(d)))


S = json.load(open(SRC))
models = S["report"]["models"]
rng = np.random.default_rng(SEED)
res = dict(families={})
for fam, sizes in FAMILIES.items():
    out = dict(sizes=sizes)
    missing = [m for m in sizes if m not in models]
    failed = [m for m in sizes if m in models and models[m].get("instruments") != "OK"]
    if missing or failed:
        out.update(void=True, missing=missing, instruments_failed=failed)
        res["families"][fam] = out
        continue
    rows = {m: {(r["group"], r["task"]): r for r in S["rows"][m] if r["main"] and r["defined"] and r.get("logit_cap", "capped") == "capped"}
            for m in sizes}
    common = sorted(set.intersection(*[set(v) for v in rows.values()]))
    out["common_tasks"] = len(common)
    q = {}
    for m in sizes:
        for k in common:
            r = rows[m][k]
            rr, rho = r["excess"]["ce_p1"]["pt"], r["top_train_T"][0]
            q[(m, k)] = dict(r=rr, rho=rho, sigma=(rr / rho) if rho > 0 else float("nan"))
    nonpos = sorted({f"{m}:{k[0]}/{k[1]}" for (m, k), v in q.items() if v["r"] <= 0 or v["rho"] <= 0})
    out["nonpositive"] = nonpos
    s1, s2, s3 = sizes
    D = {}
    for name in ("rho", "r", "sigma"):
        for a, b in ((s1, s2), (s1, s3), (s2, s3)):
            D[f"{name}:{a}->{b}"] = med_ci([logdiff(q[(a, k)][name], q[(b, k)][name]) for k in common], rng)
    out["D"] = D
    above = lambda d: d["lo"] is not None and d["lo"] > 0  # noqa: E731
    below = lambda d: d["hi"] is not None and d["hi"] < 0  # noqa: E731
    guard = {name: below(D[f"{name}:{s2}->{s3}"]) for name in ("rho", "r")}       # for the quantities expected to grow
    out["room_grows"] = bool(above(D[f"rho:{s1}->{s2}"]) and above(D[f"rho:{s1}->{s3}"]) and not guard["rho"])
    out["excess_grows"] = bool(above(D[f"r:{s1}->{s2}"]) and above(D[f"r:{s1}->{s3}"]) and not guard["r"])
    out["share_falls"] = bool(below(D[f"sigma:{s1}->{s3}"]))
    out["middle_guard_fired"] = guard
    out["falls_with_size_sigma"] = sum(logdiff(q[(s1, k)]["sigma"], q[(s3, k)]["sigma"]) < 0 for k in common)
    rep = {}
    for m in sizes:
        rs = [rows[m][k] for k in common]
        P = [r["angle"]["P"] for r in rs if r.get("one_q") and r.get("angle", {}).get("P")]
        rep[m] = dict(median_P=float(np.median(P)) if P else None, median_lambda1_over_T=float(np.median([r["lam_T"] for r in rs])),
                      median_abs_sq_minus_1=float(np.median([abs(r["objective"]["p1"]["sq_T"] - 1) for r in rs])),
                      median_r=float(np.median([q[(m, k)]["r"] for k in common])),
                      median_rho=float(np.median([q[(m, k)]["rho"] for k in common])))
    out["reports"] = rep
    # reports (descriptive; their own generator, so the tests' resamples are unchanged): growth per data source, and the step
    # intervals with one source left out at a time
    src = {k: rows[s1][k].get("source", "unknown") for k in common}
    per = {}
    for so in sorted(set(src.values())):
        ks = [k for k in common if src[k] == so]
        per[so] = dict(n=len(ks), **{f"{name}{tag}": float(np.median([logdiff(q[(a, k)][name], q[(b, k)][name]) for k in ks]))
                                    for name in ("rho", "r") for tag, (a, b) in (("12", (s1, s2)), ("13", (s1, s3)))})
    out["per_source"] = per
    loso = {}
    for so in sorted(set(src.values())):
        ks = [k for k in common if src[k] != so]; g2 = np.random.default_rng(SEED)
        loso[so] = {f"{name}:{a}->{b}": med_ci([logdiff(q[(a, k)][name], q[(b, k)][name]) for k in ks], g2)
                    for name in ("rho", "r") for a, b in ((s1, s2), (s1, s3))}
    out["leave_one_source_out"] = loso
    res["families"][fam] = out

# Qwen2.5-32B: the step 7B -> 32B, the series' only step between two untied-embedding sizes. The room grows (room_grows):
# D_rho(s3, s4)'s interval entirely above 0; the excess likewise (excess_grows); the share falls (share_falls): D_sigma(s3, s4)'s
# entirely below 0. The same quantities, rule for values at or below 0 and task bootstrap as the families above, over the tasks
# in the main population in both sizes; its own generator (seed 97), so the family tests above keep their resamples.
s3, s4 = "Qwen2.5-7B", "Qwen2.5-32B"
if all(m in models for m in (s3, s4)):
    out = dict(sizes=[s3, s4])
    failed = [m for m in (s3, s4) if models[m].get("instruments") != "OK"]
    if failed:
        out.update(void=True, instruments_failed=failed)
    else:
        rows = {m: {(r["group"], r["task"]): r for r in S["rows"][m] if r["main"] and r["defined"] and r.get("logit_cap", "capped") == "capped"}
                for m in (s3, s4)}
        common = sorted(set(rows[s3]) & set(rows[s4])); g4 = np.random.default_rng(SEED)
        q = {(m, k): dict(r=rows[m][k]["excess"]["ce_p1"]["pt"], rho=rows[m][k]["top_train_T"][0]) for m in (s3, s4) for k in common}
        for v in q.values():
            v["sigma"] = v["r"] / v["rho"] if v["rho"] > 0 else float("nan")
        D = {f"{name}:{s3}->{s4}": med_ci([logdiff(q[(s3, k)][name], q[(s4, k)][name]) for k in common], g4) for name in ("rho", "r", "sigma")}
        above = lambda d: d["lo"] is not None and d["lo"] > 0  # noqa: E731
        below = lambda d: d["hi"] is not None and d["hi"] < 0  # noqa: E731
        out.update(common_tasks=len(common), D=D, room_grows=bool(above(D[f"rho:{s3}->{s4}"])),
                   excess_grows=bool(above(D[f"r:{s3}->{s4}"])), share_falls=bool(below(D[f"sigma:{s3}->{s4}"])),
                   nonpositive=sorted({f"{m}:{k[0]}/{k[1]}" for (m, k), v in q.items() if v["r"] <= 0 or v["rho"] <= 0}),
                   reports={m: dict(median_r=float(np.median([q[(m, k)]["r"] for k in common])),
                                    median_rho=float(np.median([q[(m, k)]["rho"] for k in common])),
                                    median_lambda1_over_T=float(np.median([rows[m][k]["lam_T"] for k in common]))) for m in (s3, s4)})
    res["qwen32"] = out
F = res["families"]
json.dump(res, open(OUT, "w"), indent=1, default=float)
fmt = lambda d: "n/a" if d["pt"] is None else f"{d['pt']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n {d['n']})"  # noqa: E731
for fam, out in F.items():
    if out.get("void"):
        print(f"{fam}: VOID", out); continue
    print(f"== {fam}, {out['common_tasks']} common tasks; nonpositive: {out['nonpositive'][:5]}")
    for k, d in out["D"].items():
        print(f"   D {k}: {fmt(d)}")
    print(f"   room grows {out['room_grows']}  excess grows {out['excess_grows']}  share falls {out['share_falls']}  middle guard {out['middle_guard_fired']}")
if "qwen32" in res and not res["qwen32"].get("void"):
    o = res["qwen32"]; print(f"== Qwen2.5 7B -> 32B, {o['common_tasks']} common tasks")
    for k, d in o["D"].items():
        print(f"   D {k}: {fmt(d)}")
    print(f"   room grows' {o['room_grows']}  excess grows' {o['excess_grows']}  share falls' {o['share_falls']}")
print("written", OUT)

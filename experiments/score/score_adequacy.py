"""How adequate the first-order description is at the readout, and what Gemma-2's soft cap does to it (Appendix C.4). Reads only
experiments/results/score_readout.json and, for Gemma-2's raw label logits and start records, the readout outputs it names.

The test (key cap_rank), the cap's mechanism in Gemma-2: in its fully mirrored main tasks, nu (computed before training from the
raw logits) ranks the cap's own change of the ceiling's first-order error, delta = log(top/lambda_1) with the cap minus without
it on the same task (same pairs, hidden states, mirroring and final norm: what the runs with and without the cap share
cancels); decided by a one-sided permutation test (p < 0.025), the bootstrap interval reported. Reports: other forms of the
same rank correlation (cap_rank's reports); the cap's change of kappa against its change of e_ceil, which an identity couples
(kappa_change); each number-only set against its parent, with an exact sign-flip test (number_only_vs_parent; theory: Lean
mirror_pair_error_eq); kappa against e_ceil across models (kappa shares its test pairs with the outcome) and g against e_angle
(reports).

Per task: kappa = |tr M^mid / T - 1|, kappa_tb = |tr M^base - tr M^mid| / T, g = 1 - grad_share_mid (before training);
e_ceil = |log(exact top / lambda_1)|, start_above_lambda1 (a trained start above lambda_1), e_angle = |log P - log x_top| (after training);
nu = median over test pairs of (|d o_s| + |d o_b|) / c (raw label logits, base -> source; Gemma-2, c = 30).
Intervals: task bootstrap (B = 2000, seed 97); across models the task keys are resampled jointly (the same tasks recur in every
model).

Usage: python -I experiments/score/score_adequacy.py [--fake=<score json>] [--out=<path>] [--boot=<B, default 2000>]
       [--perm=<permutations, default 10000>]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
SRC = Path(A.get("fake", ROOT / "experiments/results/score_readout.json"))
OUT = Path(A.get("out", ROOT / "experiments/results/score_adequacy.json"))
B, SEED, MIN_N = int(A.get("boot", 2000)), 97, 5     # --boot only for the synthetic test's null trials
AGREE = 0.01                                          # the exact top's starts agree when within 1% of the best


def rankdata(x):
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):                                   # average ranks for ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    """Spearman of x and y with the rank of z partialled out (ranks residualised linearly on z's rank)."""
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    if rz.std() == 0:
        return spearman(x, y)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    ex, ey = res(rx), res(ry)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def quantities(r):
    q = dict(task=(r["group"], r["task"]), T=r.get("T"), kappa=abs(r["tr_mid_T"] - 1),
             kappa_tb=abs(r["tr_base_T"] - r["tr_mid_T"]),
             g=(1 - r["grad_share_mid"]) if r.get("grad_share_mid") is not None else None,
             mirrored=r.get("mirrored"), source=r.get("source", ""))
    top, lo, hi = r["top_exact_T"]
    lam = r["lam_T"]
    q["s_ceil"] = math.log(top / lam) if top > 0 and lam > 0 else None          # signed
    q["e_ceil"] = abs(q["s_ceil"]) if q["s_ceil"] is not None else None
    q["rel_width"] = math.log((hi - lo) / top) if top > 0 and hi is not None and lo is not None and hi > lo else None
    q["start_above_lambda1"] = any(v.get("above") for v in r.get("ceiling", {}).values() if isinstance(v, dict))
    angle_row = r.get("angle") or {}
    P, x = angle_row.get("P"), ((angle_row.get("x") or {}).get("top_train") or {}).get("pt")
    q["e_angle"] = abs(math.log(P) - math.log(x)) if r.get("one_q") and P and x and P > 0 and x > 0 else None
    return q


def strat(groups, stat, xk, yk, zk=None):
    """The statistic per model, weighted by its task count; groups: {model: [quantity dicts]}."""
    vals, w = [], []
    for qs in groups.values():
        qs = [q for q in qs if q.get(xk) is not None and q.get(yk) is not None and (zk is None or q.get(zk) is not None)]
        if len(qs) < MIN_N:
            continue
        a, b = [q[xk] for q in qs], [q[yk] for q in qs]
        v = stat(a, b, [q[zk] for q in qs]) if zk else stat(a, b)
        if not math.isnan(v):
            vals.append(v); w.append(len(qs))
    return float(np.average(vals, weights=w)) if vals else float("nan")


def boot_joint(groups, fn, rng):
    """Percentile interval of fn over resamples of the task keys, drawn jointly for every model."""
    keys = sorted({q["task"] for qs in groups.values() for q in qs})
    by = {m: {} for m in groups}
    for m, qs in groups.items():
        for q in qs:
            by[m].setdefault(q["task"], []).append(q)
    out = []
    for _ in range(B):
        draw = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        g2 = {m: [q for k in draw for q in by[m].get(k, [])] for m in groups}
        v = fn(g2)
        if not math.isnan(v):
            out.append(v)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))] if out else [None, None]


def perm_p(qs, xk, yk, zk, rng, n_perm=10000):
    """One-sided permutation p-value of Spearman(x, y), or of the partial Spearman given z: x permuted among tasks (exact under
    exchangeability)."""
    qs = [q for q in qs if q.get(xk) is not None and q.get(yk) is not None and (zk is None or q.get(zk) is not None)]
    if len(qs) < MIN_N:
        return None
    x, y = [q[xk] for q in qs], [q[yk] for q in qs]
    st = (lambda a: spearman(a, y)) if zk is None else (lambda a: partial_spearman(a, y, [q[zk] for q in qs]))
    obs = st(x)
    hits = sum(st(list(rng.permutation(x))) >= obs - 1e-12 for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def test(groups, stat, xk, yk, null, rng, zk=None):
    pt = strat(groups, stat, xk, yk, zk)
    lo, hi = boot_joint(groups, lambda g: strat(g, stat, xk, yk, zk), rng)
    n = sum(len([q for q in qs if q.get(xk) is not None and q.get(yk) is not None and (zk is None or q.get(zk) is not None)])
            for qs in groups.values())
    return dict(pt=pt, lo=lo, hi=hi, tasks=n, holds=lo is not None and lo > null)


S = json.load(open(SRC))
models = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]
gem = [m for m in models if m.startswith("gemma")]
rows = {m: [r for r in S["rows"][m] if r["main"] and r["defined"] and r.get("logit_cap", "capped") == "capped"] for m in models}
Q = {m: [quantities(r) for r in rs] for m, rs in rows.items()}
rng = np.random.default_rng(SEED)
res = dict(models=models, tasks={m: len(v) for m, v in Q.items()},
           missing_e_ceil={m: sum(q["e_ceil"] is None for q in v) for m, v in Q.items()})


def run_files(model, group, logit_cap):
    suf = "__uncapped" if logit_cap == "uncapped" else ""
    f = SRC.parent / "readout" / f"readout_{model}_m_{group}{suf}"
    return f.with_suffix(".json"), f.with_suffix(".npz")


def gemma_tasks(logit_cap):
    """Gemma's main tasks with the cap or without it (arm "capped" or "uncapped"): quantities, nu from the raw logits, and
    whether the exact top's starts agree."""
    out, cache = {}, {}
    main_keys = {q["task"] for q in Q[gem[0]]}
    for r in S["rows"][gem[0]]:
        k = (r["group"], r["task"])
        if not (r["defined"] and r.get("logit_cap", "capped") == logit_cap and k in main_keys):
            continue
        q = quantities(r)
        jf, nf = run_files(gem[0], r["group"], logit_cap)
        if r["group"] not in cache:
            cache[r["group"]] = (json.load(open(jf)) if jf.is_file() else None, np.load(nf) if nf.is_file() else None)
        J, Z = cache[r["group"]]
        c = (J or {}).get("cap")
        kb, ks = f"{r['task']}/raw_logits_b", f"{r['task']}/raw_logits_s"
        q["nu"] = float(np.median(np.abs(Z[ks] - Z[kb]).sum(1) / c)) if c and Z is not None and kb in Z and ks in Z else None
        te = ((J or {}).get("tasks", {}).get(r["task"], {}) or {}).get("top_exact") or {}
        ls = [-s["loss"] for s in te.get("starts") or [] if s.get("loss") is not None]
        q["starts_agree"] = (te.get("replaced_by") is None and bool(ls)
                             and (max(ls) - min(ls)) <= AGREE * abs(max(ls)))  # a replaced top has no agreeing starts
        out[k] = q
    return out


fmt = lambda t: "n/a" if not t or t.get("lo") is None else f"{t['pt']:.3f} [{t['lo']:.3f}, {t['hi']:.3f}] on {t['tasks']} tasks, holds={t['holds']}"  # noqa: E731

# ---- the test: does nu rank the cap's change delta? ----
# In Gemma-2's fully mirrored main tasks, nu ranks delta, the cap's change of log(top/lambda_1) on the same task (same pairs,
# hidden states, mirroring and final norm; only the cap differs).
if not gem:
    print("NOTE: no Gemma-2 model with held instruments; the cap's test and the kappa report are not computed")
else:
    by_logit_cap = {a: gemma_tasks(a) for a in ("capped", "uncapped")}
    both = []
    for k, q in by_logit_cap["capped"].items():
        u = by_logit_cap["uncapped"].get(k)
        if u is None:
            continue
        both.append(dict(task=k, nu=q["nu"], e_cap=q["e_ceil"], e_uncap=u["e_ceil"], mirrored=q["mirrored"],
                         rel_width=q["rel_width"], starts_agree=q["starts_agree"],
                         delta=(q["s_ceil"] - u["s_ceil"]) if q["s_ceil"] is not None and u["s_ceil"] is not None else None))
    usable = [p for p in both if p["nu"] is not None and p["delta"] is not None]
    mir_ = [p for p in usable if p["mirrored"] == "all"]
    # the cap's own contribution: delta = log(top/lambda_1) capped minus uncapped, same task (shared curvature cancels)
    res["cap_rank"] = dict(main_test=test({"g": mir_}, spearman, "nu", "delta", 0.0, rng),
                     dropped=dict(main_tasks=len(by_logit_cap["capped"]), capped_and_uncapped=len(both), usable=len(usable),
                                  mirrored=len(mir_)))
    res["cap_rank"]["main_test"]["perm_p"] = perm_p(mir_, "nu", "delta", None, rng, n_perm=int(A.get("perm", 10000)))
    res["cap_rank"]["holds"] = res["cap_rank"]["main_test"]["perm_p"] is not None and res["cap_rank"]["main_test"]["perm_p"] < 0.025   # the decision
    res["cap_rank"]["reports"] = dict(
        all_tasks=test({"g": usable}, spearman, "nu", "delta", 0.0, rng),
        mirrored_partial_given_uncapped=test({"g": mir_}, partial_spearman, "nu", "e_cap", 0.0, rng, zk="e_uncap"),
        delta_positive=dict(mirrored=sum(p["delta"] > 0 for p in mir_), of=len(mir_)),
        plain_capped=test({"g": usable}, spearman, "nu", "e_cap", 0.0, rng),
        plain_uncapped=test({"g": usable}, spearman, "nu", "e_uncap", 0.0, rng),
        given_noise=test({"g": usable}, partial_spearman, "nu", "e_cap", 0.0, rng, zk="rel_width"),
        mirrored_starts_agree=test({"g": [p for p in mir_ if p["starts_agree"]]}, spearman, "nu", "delta", 0.0, rng))
    if usable:
        mnu = float(np.median([p["nu"] for p in usable]))
        res["cap_rank"]["magnitude"] = dict(median_nu=mnu, median_e_cap=float(np.median([p["e_cap"] for p in usable])),
                                      median_e_uncap=float(np.median([p["e_uncap"] for p in usable])),
                                      bound_scale=(2 / 3) * (mnu / 2) ** 2)
    # a report: the cap's change of kappa against its change of e_ceil (an identity couples them)
    keys = [k for k in by_logit_cap["capped"] if k in by_logit_cap["uncapped"]
            and by_logit_cap["capped"][k]["e_ceil"] is not None and by_logit_cap["uncapped"][k]["e_ceil"] is not None]
    pairs = [dict(task=k, dk=by_logit_cap["capped"][k]["kappa"] - by_logit_cap["uncapped"][k]["kappa"],
                  de=by_logit_cap["capped"][k]["e_ceil"] - by_logit_cap["uncapped"][k]["e_ceil"]) for k in keys]
    if pairs:
        k1, n = sum(p["dk"] > 0 for p in pairs), len(pairs)
        res["kappa_change"] = dict(rank_correlation=test({"g": pairs}, spearman, "dk", "de", 0.0, rng),
                               sign_test=dict(cap_larger=k1, tasks=n, share=k1 / n,
                                       p=sum(math.comb(n, i) for i in range(k1, n + 1)) / 2 ** n),
                               e_ceil_larger_with_cap=sum(p["de"] > 0 for p in pairs))
# a report: each partly mirrored number-only set against its own parent, averaged over the models; exact sign flip
par = {}
for m, v in Q.items():
    by = {q["task"][1]: q for q in v}
    for name, q in by.items():
        if name.endswith("__num") and name[:-5] in by and q["e_ceil"] is not None and by[name[:-5]]["e_ceil"] is not None:
            par.setdefault(name, []).append(q["e_ceil"] - by[name[:-5]]["e_ceil"])
diffs = {k: float(np.mean(v)) for k, v in par.items()}
if diffs:
    d = np.array(list(diffs.values())); obs = d.sum(); n = len(d)
    flips = [np.dot(d, [1 if (b >> i) & 1 else -1 for i in range(n)]) for b in range(2 ** n)]
    res["number_only_vs_parent"] = dict(pairs=diffs, positive=int((d > 0).sum()), n=n,
                            p_one_sided=float(np.mean([f >= obs - 1e-15 for f in flips])), models_per_pair={k: len(v) for k, v in par.items()})

# ---- reports (descriptive: kappa shares its test pairs with the outcome) ----
for m, v in Q.items():
    for q in v:
        q["logT"] = math.log(q["T"]) if q.get("T") and q["T"] > 0 else None
withv = {m: v for m, v in Q.items() if 0 < sum(q["start_above_lambda1"] for q in v) < len(v)}
rep = dict(kappa_e_ceil=test(Q, spearman, "kappa", "e_ceil", 0.0, rng),
           kappa_auc_start_above=test(withv, auc, "kappa", "start_above_lambda1", 0.5, rng) if withv else None,
           kappa_e_ceil_given_logT=test(Q, partial_spearman, "kappa", "e_ceil", 0.0, rng, zk="logT"),
           kappa_e_ceil_gemma=test({m: Q[m] for m in gem}, spearman, "kappa", "e_ceil", 0.0, rng) if gem else None,
           kappa_tb_e_ceil=test(Q, spearman, "kappa_tb", "e_ceil", 0.0, rng),
           g_e_angle=test(Q, spearman, "g", "e_angle", 0.0, rng))
pool = [q for v in Q.values() for q in v if q["kappa"] is not None and q["e_ceil"] is not None]
rep["kappa_e_ceil_pooled"] = spearman([q["kappa"] for q in pool], [q["e_ceil"] for q in pool])
rep["per_model_median"] = {m: dict(kappa=float(np.median([q["kappa"] for q in v])),
                                   e_ceil=float(np.median([q["e_ceil"] for q in v if q["e_ceil"] is not None])))
                           for m, v in Q.items() if v}
res["reports"] = rep
json.dump(res, open(OUT, "w"), indent=1, default=float)

if "cap_rank" in res:
    print("the cap's test Gemma, mirrored tasks, Spearman(nu, delta = cap's change of log(top/lambda_1)):", fmt(res["cap_rank"]["main_test"]),
          "permutation p", res["cap_rank"]["main_test"]["perm_p"], "-> holds", res["cap_rank"]["holds"], res["cap_rank"]["dropped"])
    for k, t in res["cap_rank"]["reports"].items():
        print(f"   report {k}:", fmt(t) if isinstance(t, dict) and "pt" in t else t)
    print("   magnitude:", res["cap_rank"].get("magnitude"))
if "kappa_change" in res:
    print("kappa change: rank correlation", fmt(res["kappa_change"]["rank_correlation"]), "| sign test", res["kappa_change"]["sign_test"], "| e_ceil larger with the cap",
          res["kappa_change"]["e_ceil_larger_with_cap"])
if "number_only_vs_parent" in res:
    a7 = res["number_only_vs_parent"]; print(f"number-only against parent: number-only minus parent, {a7['positive']}/{a7['n']} positive, exact p {a7['p_one_sided']:.4f}")
print("reports: kappa_e_ceil", fmt(rep["kappa_e_ceil"]), "| A2", fmt(rep["kappa_e_ceil_gemma"]), "| A4", fmt(rep["kappa_auc_start_above"]), "| A5", fmt(rep["kappa_tb_e_ceil"]), "| A6", fmt(rep["kappa_e_ceil_given_logT"]),
      "| C", fmt(rep["g_e_angle"]), "| A3 pooled", round(rep["kappa_e_ceil_pooled"], 3))
print("written", OUT)

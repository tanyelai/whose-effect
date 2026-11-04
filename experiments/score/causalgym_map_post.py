"""Readings of CausalGym's rebuilt maps beside experiments/score/score_causalgym_map.py's (Section 7.1), from the same maps and resamples.
- The off-route lead, three ways. off_lead, the scorer's own off_lead: a layer's lead (DAS's best minus vanilla's best)
  counts when DAS's chosen cell is one where G*'s interval does not lie above 0. off_lead_above_gstar: it counts only where, in addition,
  the cell's paired interval of (DAS - G*)/T lies above 0 (off-route proper); off_lead_indeterminate: the rest of the off-route
  lead. Per model, the ratio of sums over its maps with the scorer's construction bootstrap; pooled over the scorer's pooled
  models likewise.
- The not-produced share, three forms: signed, N / X (the scorer's s_N); positive part, sum max(DAS - G*, 0) / X;
  conservative, the sum of DAS - G* over the layers whose paired interval lies above 0, / X. Per model and pooled, as above.
Checks: the off-route lead, N and X reproduce the scorer's per map (experiments/results/score_causalgym_map.json) within 1e-9 T.
The scorer's definitions (loading, provenance, degenerate cells, intervals, the construction bootstrap) are executed from its own
file, up to its main loop, so this reads exactly what it read.
Output: experiments/results/score_causalgym_map.post.json. Usage: python -I experiments/score/causalgym_map_post.py
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
SRC = (R / "experiments/score/score_causalgym_map.py").read_text()
cut = SRC.index("\nfor m, t in pairs:")
ns = {"__file__": str(R / "experiments/score/score_causalgym_map.py"), "__name__": "causalgym_map_defs"}
sys.argv = [sys.argv[0]]
exec(compile(SRC[:cut], "score_causalgym_map.py", "exec"), ns)              # definitions and inputs only; no map is scored here
S = json.load(open(R / "experiments/results/score_causalgym_map.json"))
judged = S["pooled"]["models"]


def readings(J1, Z1, J2, Z2, C, exclude):
    """Per layer at the point: DAS's chosen cell, the lead d - v, d - G*, and the two off-route classes of the chosen cell."""
    NL = max(c["layer"] for c in J1["cells"]); W = np.vstack([np.ones(C.shape[1]), C]).astype(float); w = W.sum(1)
    T = Z1["T"]; Tm = W @ T / w; raw = lambda x: W @ x / w  # noqa: E731
    E, G, Gs = {}, {}, {}
    for J, Z in ((J1, Z1), (J2, Z2)):
        for c in J["cells"]:
            k = (c["layer"], c["region"])
            if (J is J2 and c["layer"] == NL) or k in exclude:
                continue
            live = c.get("live", True); key = f"L{k[0]}r{k[1]}"; zero = np.zeros(len(T))
            get = lambda n: Z[f"{key}/{n}"] if live else zero  # noqa: E731
            E[k] = raw(np.mean([get(f"e/das_ce_s{s}_normal") for s in (0, 1, 2)], axis=0)); G[k] = raw(get("e/own_normal"))
            Gs[k] = raw(get("gstar_test")) if k[0] < NL else G[k]
    off = {k: bool(k[0] < NL and np.quantile(v[1:], 0.025) <= 0) for k, v in Gs.items()}
    out = []
    for L in sorted({k[0] for k in E}):
        kd = max((k for k in E if k[0] == L), key=lambda k: E[k][0]); kv = max((k for k in G if k[0] == L), key=lambda k: G[k][0])
        ex = (E[kd] - Gs[kd]) / Tm
        lo = ns["interval"](float(ex[0]), ex[1:], True)[0]
        out.append(dict(L=L, d=E[kd][0], v=G[kv][0], gs=Gs[kd][0], off=off[kd], above=lo is not None and lo > 0, Tm=Tm[0]))
    return out


per_map, groups = {}, {}
for key, rec in S["maps"].items():
    if not rec["instruments"]:
        continue
    m, t = key.split("/")
    J1, Z1, J2, Z2 = ns["load"](m, t)
    C = ns["boot_counts"](Z1["cluster"], J1["boot"]["B"], J1["boot"]["seed"])
    excl, _ = ns["degenerate"](J1, J2, t)
    rows = readings(J1, Z1, J2, Z2, C, excl); Tm = rows[0]["Tm"]
    X = sum(r["d"] - r["v"] for r in rows) / Tm
    off_lead = sum(r["d"] - r["v"] for r in rows if r["off"]) / Tm
    off_lead_above_gstar = sum(r["d"] - r["v"] for r in rows if r["off"] and r["above"]) / Tm
    N = sum(r["d"] - r["gs"] for r in rows) / Tm
    N_pos = sum(max(r["d"] - r["gs"], 0) for r in rows) / Tm
    N_cons = sum(r["d"] - r["gs"] for r in rows if r["above"]) / Tm
    prim = rec["primary"]
    assert abs(off_lead - prim["off_lead"]["pt"]) < 1e-9 and abs(N - prim["N"]["pt"]) < 1e-9 and abs(X - prim["X"]["pt"]) < 1e-9, \
        (key, off_lead, prim["off_lead"]["pt"], N, prim["N"]["pt"])
    per_map[key] = dict(X=X, off_lead=off_lead, off_lead_above_gstar=off_lead_above_gstar, off_lead_indeterminate=off_lead - off_lead_above_gstar, N=N,
                        N_positive=N_pos, N_conservative=N_cons,
                        layers_off=sum(r["off"] for r in rows), layers_off_above_gstar=sum(r["off"] and r["above"] for r in rows))
    groups.setdefault(m, {}).setdefault(ns["construction"](t), []).append(per_map[key])


def share(gr, num):
    return ns["boot_ratio_sums"]({c: [(v[num], v["X"]) for v in vs] for c, vs in gr.items()})


out = dict(models={}, pooled={})
for m, gr in sorted(groups.items()):
    out["models"][m] = {k: share(gr, k) for k in ("off_lead", "off_lead_above_gstar", "off_lead_indeterminate", "N", "N_positive", "N_conservative")}
    out["models"][m]["maps"] = sum(len(v) for v in gr.values())
pool = {}
for m in judged:
    for c, vs in groups.get(m, {}).items():
        pool.setdefault(c, []).extend(vs)
out["pooled"] = {k: share(pool, k) for k in ("off_lead", "off_lead_above_gstar", "off_lead_indeterminate", "N", "N_positive", "N_conservative")}
out["pooled"]["models"] = judged
out["per_map"] = per_map
json.dump(out, open(R / "experiments/results/score_causalgym_map.post.json", "w"), indent=1, default=float)
for m, v in list(out["models"].items()) + [("pooled", out["pooled"])]:
    print(f"{m:14s} s_off {v['off_lead'][0]:.2f} {v['off_lead'][1:]} | above G* {v['off_lead_above_gstar'][0]:.2f} "
          f"{v['off_lead_above_gstar'][1:]} | N/X signed {v['N'][0]:.2f}, positive {v['N_positive'][0]:.2f}, conservative {v['N_conservative'][0]:.2f}")

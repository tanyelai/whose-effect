"""The off-route class on small_npi_fg_agr, with the difference tested directly (Section 6; App. D.1). A cell whose DAS interval lies
above zero while G*'s does not (G* with its context chosen on the training pairs; key off) compares two significance tests;
this also tests the difference itself, DAS - G*, over T, on the map's own resamples, at each such cell (raw, and after
Benjamini-Hochberg per model at q = 0.025 within the class), which makes it off-route proper; and, at every zero-class cell
(G*'s interval not above zero), the squared error aimed at T and CDAS's divergence, against G* the same way.
Writes experiments/results/inside_sets/small_npi_fg_agr_offroute.post.json. Usage: python -I experiments/score/inside_offroute_direct.py
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "experiments/lib"))
from measure import bh, boot_counts, boot_ratio, interval, p_above  # noqa: E402

S = json.load(open(R / "experiments/results/inside_sets/small_npi_fg_agr.json"))
off, zero = [], []
for m, cs in S["cells"].items():
    for t in sorted({c["task"] for c in cs}):
        f = R / f"experiments/results/inside/inside_{m}_{t}.json"; J = json.load(open(f)); Z = np.load(f.with_suffix(".npz"))
        T = Z["T"]; Tm = float(T.mean()); C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"])
        for c in (c for c in cs if c["task"] == t):
            k = lambda n: Z[f"{c['cell']}/{n}"]
            gs = k("gstar_train"); ce = np.mean([k(f"e/das_ce_s{x}_normal") for x in (0, 1, 2)], axis=0)
            gpt, dpt = float(gs.mean() / Tm), float(ce.mean() / Tm)
            glo, ghi = interval(gpt, boot_ratio(gs, T, C)); dlo, dhi = interval(dpt, boot_ratio(ce, T, C))
            rd = boot_ratio(ce - gs, T, C); dd = float((ce - gs).mean() / Tm); ddlo, ddhi = interval(dd, rd)
            rec = dict(m=m, t=t, cell=c["cell"], g=(gpt, glo, ghi), d=(dpt, dlo, dhi), diff=(dd, ddlo, ddhi), p=p_above(dd, rd, 0.0))
            for nm in ("das_sq_T_s0_normal", "das_jsd_s0_normal"):
                x = k(f"e/{nm}"); xd = float((x - gs).mean() / Tm); xr = boot_ratio(x - gs, T, C); xlo, xhi = interval(xd, xr)
                xa = float(x.mean() / Tm); alo, ahi = interval(xa, boot_ratio(x, T, C))
                rec[nm] = dict(eff=(xa, alo, ahi), diff=(xd, xlo, xhi), p=p_above(xd, xr, 0.0))
            if glo <= 0:
                zero.append(rec)
                if dlo > 0:
                    off.append(rec)
n_bh = 0
for m in sorted({r["m"] for r in off}):
    n_bh += int(sum(bh([r["p"] for r in off if r["m"] == m], 0.025)))
out = dict(zero=len(zero), off=len(off), off_above_gstar_raw=int(sum(r["diff"][1] > 0 for r in off)), off_above_gstar_bh=n_bh,
           off_gstar_hi_median=float(np.median([r["g"][2] for r in off])), off_gstar_hi_max=float(max(r["g"][2] for r in off)),
           off_das_at_or_below_gstar_hi=int(sum(r["d"][0] <= r["g"][2] for r in off)),
           zero_above_gstar={nm: int(sum(r[nm]["diff"][1] > 0 for r in zero)) for nm in ("das_sq_T_s0_normal", "das_jsd_s0_normal")},
           zero_nonzero={nm: int(sum(r[nm]["eff"][1] > 0 for r in zero)) for nm in ("das_sq_T_s0_normal", "das_jsd_s0_normal")})
dest = R / "experiments/results/inside_sets/small_npi_fg_agr_offroute.post.json"
json.dump(out, open(dest, "w"), indent=1)
print(out)

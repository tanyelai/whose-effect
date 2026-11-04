"""Section 6's readings in summed form on a scored set of maps (App. D.5): the not-produced share, the off-route
class, and DAS above G over the carried cells. Read from the maps' own per-pair arrays and resamples (experiments/results/inside/, the
run's clusters, B and seed), on the set's population: the cells of <set>.post.json's cell_classes (live, not degenerate, in
maps that enter), or every scored cell where the set has no post file (small_npi_fg_agr).
1. The not-produced share, per model and pooled, three forms over the same resamples (a ratio of sums over cells, each cell over
   its map's mean T; one resample row b of every map at once, the maps being independent): positive part (pos),
   sum max(e - G*, 0) / sum e; signed, sum (e - G*) / sum e; conservative, sum (e - G*) over the cells whose paired interval of
   (e - G*)/T lies above 0, / sum e. e: cross-entropy DAS, its three starts' mean; G*: the best context chosen on the training
   pairs, measured on the test pairs. Beside each, the part through attention (each cell's part times the share of DAS's effect
   freezing removes, clipped to [0, 1], 0 where DAS's interval holds 0).
2. The off-route class, three ways, without a bar: (i) off-route, G*'s interval not above 0, DAS's above 0, and the paired
   interval of (DAS - G*)/T above 0; (ii) indeterminate, the first two but the difference's interval holding 0; (iii) carried by
   G*, its interval above 0. Benjamini-Hochberg per model (q = 0.025) within (i) + (ii) beside the raw count.
3. "Above G" over carried cells: of the cells whose G interval lies above 0 (the post scorer's class "carried"), those where DAS
   lies significantly above G after false-discovery control per model over the population, beside the count over every cell.
Per cell also sqT_nonzero: whether the interval of the squared error aimed at T (s = 0) excludes 0; paper/scripts/numbers_main.py
weights that objective's share through attention by it, as DAS's is weighted above.
Positive control: on small_npi_fg_agr it reproduces the off-route counts of small_npi_fg_agr_offroute.post.json
(experiments/score/inside_offroute_direct.py) and the per-model ranges of the positive-part share (0.56--0.65) and of its part through
attention (0.57--0.72) that Section 6 and App. D.1 report for the small models on those three tasks.
Writes experiments/results/inside_sets/<set>_forms.post.json. Usage: python -I experiments/score/inside_forms_post.py --set=<set>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "experiments/lib"))
from measure import bh, boot_counts, interval, p_above  # noqa: E402

A_ = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
SET = A_["set"]
S = json.load(open(R / f"experiments/results/inside_sets/{SET}.json"))
PF = R / f"experiments/results/inside_sets/{SET}.post.json"
POP = set(json.load(open(PF))["cell_classes"]) if PF.is_file() else {f"{c['model']}/{c['task']}/{c['cell']}" for cs in S["cells"].values() for c in cs}
SRC = lambda t: ("number-only" if t.endswith("__num") else "Kumon" if t.startswith(("NPI", "FG")) else "RAVEL" if t.startswith("ravel_")  # noqa: E731
                 else "sentiment" if t.startswith("sentiment_") else "SST" if t.startswith("sst_") else "IOI" if t.startswith("ioi_") else "CausalGym")

rec = []          # one record per cell of the population
chk = []          # the same cells, for the checks beside the share
boot = {}         # model -> {name: per-resample sums (B,)}
for m, cs in sorted(S["cells"].items()):
    cs = [c for c in cs if f"{m}/{c['task']}/{c['cell']}" in POP]
    # Benjamini-Hochberg per model over the population (the post scorer's rule): DAS above G, from the scorer's per-cell p-values
    rejG = bh([c["above_own_G"]["p"] for c in cs], 0.025)
    for c, r_ in zip(cs, rejG):
        c["_aboveG_bh"] = bool(r_)
    sums = None
    for t in sorted({c["task"] for c in cs}):
        f = R / f"experiments/results/inside/inside_{m}_{t}.json"; J = json.load(open(f)); fz = f.with_suffix(".npz")
        assert J["npz_sha256"] == hashlib.sha256(fz.read_bytes()).hexdigest(), f"{fz.name}: not the npz its json names"
        Z = np.load(fz); T = Z["T"].astype(float)
        C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"]).astype(float)
        assert hashlib.sha256(np.ascontiguousarray(C.astype(np.int64)).tobytes()).hexdigest()[:16] == J["boot_counts_sha"]
        Tb = C @ T / C.sum(1); Tm = float(T.mean())
        mc = [c for c in cs if c["task"] == t]
        k = lambda c, n: Z[f"{c['cell']}/{n}"].astype(float)  # noqa: E731
        E = np.stack([np.mean([k(c, f"e/das_ce_s{x}_normal") for x in (0, 1, 2)], axis=0) for c in mc], 1)    # pairs x cells
        G = np.stack([k(c, "gstar_train") for c in mc], 1)
        Eb = (C @ E) / C.sum(1)[:, None] / Tb[:, None]; Gb = (C @ G) / C.sum(1)[:, None] / Tb[:, None]          # B x cells, over T
        e_pt, g_pt = E.mean(0) / Tm, G.mean(0) / Tm
        for j, c in enumerate(mc):
            assert abs(e_pt[j] - c["ce_T"]) < 1e-9 and abs(g_pt[j] - c["Gstar_train_T"]) < 1e-9, (m, t, c["cell"])
        sd = np.array([np.clip(c["attention_share"]["s_das"], 0, 1) if c.get("ce_nonzero") and c["attention_share"].get("s_das") is not None else 0.0 for c in mc])
        # the checks beside the share: DAS with every attention pattern frozen and with only the readout row frozen, the network's
        # own interchange under both, and DAS swapped together with G*'s own second region (the cell alone where G* has none)
        jc = {f"L{x['layer']}r{x['region']}": x for x in J["cells"]}
        ctx = [jc[c["cell"]].get("context_train") is not None and f"{c['cell']}/ctx/own" in Z.files for c in mc]
        EF = np.stack([np.mean([k(c, f"e/das_ce_s{x}_frozen") for x in (0, 1, 2)], axis=0) for c in mc], 1)
        ER = np.stack([np.mean([k(c, f"e/das_ce_s{x}_row") for x in (0, 1, 2)], axis=0) for c in mc], 1)
        GN, GF, GR = (np.stack([k(c, f"e/own_{mode}") for c in mc], 1) for mode in ("normal", "frozen", "row"))
        EC = np.stack([np.mean([k(c, f"ctx/das_ce_s{x}") for x in (0, 1, 2)], axis=0) if ctx[j] else E[:, j] for j, c in enumerate(mc)], 1)
        ECb = (C @ EC) / C.sum(1)[:, None] / Tb[:, None]; EFb = (C @ EF) / C.sum(1)[:, None] / Tb[:, None]
        for j, c in enumerate(mc):
            dc = float((EC[:, j] - G[:, j]).mean() / Tm)
            chk.append(dict(m=m, ctx=ctx[j], gain=float((G[:, j] - GN[:, j]).mean() / Tm), p_ctx=p_above(dc, ECb[:, j] - Gb[:, j], 0.0),
                            e=float(E[:, j].mean() / Tm), ef=float(EF[:, j].mean() / Tm), er=float(ER[:, j].mean() / Tm),
                            gn=float(GN[:, j].mean() / Tm), gf=float(GF[:, j].mean() / Tm), gr=float(GR[:, j].mean() / Tm)))
        for j, c in enumerate(mc):
            glo, ghi = interval(g_pt[j], Gb[:, j]); dlo, dhi = interval(e_pt[j], Eb[:, j])
            dd = e_pt[j] - g_pt[j]; ddr = Eb[:, j] - Gb[:, j]; ddlo, ddhi = interval(dd, ddr)
            r_ = dict(m=m, t=t, src=SRC(t), cell=c["cell"], e=e_pt[j], g=g_pt[j], glo=glo, ghi=ghi, dlo=dlo, dhi=dhi,
                      diff=dd, difflo=ddlo, diffhi=ddhi, p=p_above(dd, ddr, 0.0), sd=sd[j],
                      carried=c["attention_share"]["own_defined"] and c["G_T"] > 0, aboveG_bh=c["_aboveG_bh"])
            # whether the T-aimed squared error's own effect has an interval off zero, which weights its share through attention
            xT = k(c, "e/das_sq_T_s0_normal"); xlo, xhi = interval(float(xT.mean() / Tm), (C @ xT) / C.sum(1) / Tb)
            r_["sqT_nonzero"] = bool(xlo > 0 or xhi < 0)
            if glo <= 0:          # the zero class: the objectives aimed at T or at the source run, against G* the same way
                for nm in ("das_sq_T_s0_normal", "das_jsd_s0_normal"):
                    x = k(c, f"e/{nm}"); xd = float((x - G[:, j]).mean() / Tm)
                    xr = (C @ x) / C.sum(1) / Tb - Gb[:, j]
                    r_[nm] = dict(diff=xd, difflo=interval(xd, xr)[0], eff=float(x.mean() / Tm), efflo=interval(float(x.mean() / Tm), (C @ x) / C.sum(1) / Tb)[0])
            rec.append(r_)
        cons = np.array([r_["difflo"] > 0 for r_ in rec[-len(mc):]])
        ef_pt, ec_pt = EF.mean(0) / Tm, EC.mean(0) / Tm
        red = lambda e_, ef_, g_: np.minimum(np.maximum(e_ - g_, 0), np.maximum(np.maximum(e_ - g_, 0) - np.maximum(ef_ - g_, 0), 0))  # noqa: E731
        part = dict(e=Eb.sum(1), pos=np.maximum(Eb - Gb, 0).sum(1), signed=(Eb - Gb).sum(1), cons=(Eb - Gb)[:, cons].sum(1),
                    pos_att=(np.maximum(Eb - Gb, 0) * sd).sum(1), signed_att=((Eb - Gb) * sd).sum(1), cons_att=((Eb - Gb) * sd)[:, cons].sum(1),
                    pos_red=red(Eb, EFb, Gb).sum(1), ec=ECb.sum(1), pos_ctx=np.maximum(ECb - Gb, 0).sum(1))
        pt = dict(e=e_pt.sum(), pos=np.maximum(e_pt - g_pt, 0).sum(), signed=(e_pt - g_pt).sum(), cons=(e_pt - g_pt)[cons].sum(),
                  pos_att=(np.maximum(e_pt - g_pt, 0) * sd).sum(), signed_att=((e_pt - g_pt) * sd).sum(), cons_att=((e_pt - g_pt) * sd)[cons].sum(),
                  pos_red=red(e_pt, ef_pt, g_pt).sum(), ec=ec_pt.sum(), pos_ctx=np.maximum(ec_pt - g_pt, 0).sum())
        if sums is None:
            sums = {n: [0.0, np.zeros(len(Tb))] for n in part}
        for n in part:
            sums[n][0] += pt[n]; sums[n][1] = sums[n][1] + part[n]
    boot[m] = sums


def forms(sums):
    out = {}
    for n in ("pos", "signed", "cons"):
        pt = sums[n][0] / sums["e"][0]; r_ = sums[n][1] / sums["e"][1]
        lo, hi = interval(pt, r_)
        pa = sums[n + "_att"][0] / sums[n][0] if sums[n][0] else float("nan")
        out[n] = dict(share=pt, lo=lo, hi=hi, through_attention_of_it=pa)
    return out


res = dict(set_name=SET, cells=len(rec), models={}, pooled={}, sources={})
for m in sorted(boot):
    rs = [r_ for r_ in rec if r_["m"] == m]
    zero = [r_ for r_ in rs if r_["glo"] <= 0]; cand = [r_ for r_ in zero if r_["dlo"] > 0]
    off = [r_ for r_ in cand if r_["difflo"] > 0]
    car = [r_ for r_ in rs if r_["carried"]]
    res["models"][m] = dict(live=len(rs), above_G_bh=sum(r_["aboveG_bh"] for r_ in rs), carried=len(car),
                            above_G_bh_in_carried=sum(r_["aboveG_bh"] for r_ in car), gstar_zero=len(zero),
                            off_or_indeterminate=len(cand), off_route=len(off), indeterminate=len(cand) - len(off),
                            off_route_bh=int(sum(bh([r_["p"] for r_ in cand], 0.025))) if cand else 0,
                            carried_by_gstar=len(rs) - len(zero), forms=forms(boot[m]))
pooled = {n: [sum(boot[m][n][0] for m in boot), sum(boot[m][n][1] for m in boot)] for n in next(iter(boot.values()))}
res["pooled"] = dict(forms=forms(pooled), **{k: sum(v[k] for v in res["models"].values())
                                            for k in ("live", "above_G_bh", "carried", "above_G_bh_in_carried", "gstar_zero",
                                                      "off_or_indeterminate", "off_route", "indeterminate", "off_route_bh", "carried_by_gstar")})


def share(sums, num, den):
    pt = sums[num][0] / sums[den][0]
    lo, hi = interval(pt, sums[num][1] / sums[den][1])
    return dict(share=float(pt), lo=float(lo), hi=float(hi))


def checks(cs, sums):
    """The checks beside the not-produced share and its attention part, over these cells (values over T, summed over cells):
    attention_direct, each cell's excess over G* with and without every attention pattern frozen (the reduction clipped to
    [0, excess], G* held fixed), against the weighting by the share of DAS's whole effect; not_produced_in_context, DAS swapped
    together with G*'s own second region (alone where G* has none) against G*; above_gstar_in_context, the cells where that
    difference's interval lies above zero (Benjamini-Hochberg per model, q = 0.025); excess_over_G_removed, 1 - sum(e' - G')/sum(e - G)
    with every pattern frozen and with only the readout position's row frozen from the site's block on; context_gain, G* - G where
    G* has a second region."""
    v = lambda key: np.array([c[key] for c in cs])  # noqa: E731
    e, ef, er, gn, gf, gr = (v(x) for x in ("e", "ef", "er", "gn", "gf", "gr"))
    gain = np.array([c["gain"] for c in cs if c["ctx"]])
    return dict(attention_direct=share(sums, "pos_red", "pos"), not_produced_in_context=share(sums, "pos_ctx", "ec"),
                excess_over_G_removed=dict(every_pattern=float(1 - (ef - gf).sum() / (e - gn).sum()),
                                           readout_row=float(1 - (er - gr).sum() / (e - gn).sum())),
                own_kept=dict(every_pattern=float(gf.sum() / gn.sum()), readout_row=float(gr.sum() / gn.sum())),
                context_cells=int(len(gain)),
                context_gain=dict(median=float(np.median(gain)), q90=float(np.quantile(gain, 0.9))) if len(gain) else None)


res["checks"] = dict(models={}, pooled=None)
for m in sorted(boot):
    cs = [c for c in chk if c["m"] == m]
    res["checks"]["models"][m] = dict(checks(cs, boot[m]), above_gstar_in_context=int(bh([c["p_ctx"] for c in cs], 0.025).sum()))
res["checks"]["pooled"] = dict(checks(chk, pooled), above_gstar_in_context=sum(v["above_gstar_in_context"] for v in res["checks"]["models"].values()))
# at the off-route cells (class (i) of point 2 above), the share of DAS's effect, summed over the cells, that freezing removes:
# every pattern, and only the readout position's row from the site's block on (Section 6.2: most of the excess through attention runs through that row)
assert len(chk) == len(rec)
_off = [c for c, r_ in zip(chk, rec) if r_["glo"] <= 0 and r_["dlo"] > 0 and r_["difflo"] > 0]
res["checks"]["off_route_removed"] = dict(cells=len(_off), every_pattern=float(1 - sum(c["ef"] for c in _off) / sum(c["e"] for c in _off)),
                                          readout_row=float(1 - sum(c["er"] for c in _off) / sum(c["e"] for c in _off)))

for s in sorted({r_["src"] for r_ in rec}):
    rs = [r_ for r_ in rec if r_["src"] == s]; car = [r_ for r_ in rs if r_["carried"]]
    e = np.array([r_["e"] for r_ in rs]); g = np.array([r_["g"] for r_ in rs])
    res["sources"][s] = dict(live=len(rs), above_G_bh=sum(r_["aboveG_bh"] for r_ in rs), carried=len(car),
                             above_G_bh_in_carried=sum(r_["aboveG_bh"] for r_ in car),
                             not_produced=dict(pos=float(np.maximum(e - g, 0).sum() / e.sum()), signed=float((e - g).sum() / e.sum())),
                             off_route=sum(r_["glo"] <= 0 and r_["dlo"] > 0 and r_["difflo"] > 0 for r_ in rs))

# the zero class cell by cell (G*'s interval not above 0), for the text's off-route readings: its class, G*'s and DAS's intervals,
# the difference's lower bound, and the two objectives aimed at T or at the source run against G*
res["zero_cells"] = {f"{r_['m']}/{r_['t']}/{r_['cell']}": dict(cls="off" if r_["dlo"] > 0 and r_["difflo"] > 0 else "indeterminate"
                                                               if r_["dlo"] > 0 else "quiet", g=[r_["g"], r_["glo"], r_["ghi"]],
                                                               das=[r_["e"], r_["dlo"], r_["dhi"]], difflo=r_["difflo"],
                                                               sqT=r_["das_sq_T_s0_normal"], jsd=r_["das_jsd_s0_normal"])
                     for r_ in rec if r_["glo"] <= 0}
res["sqT_nonzero"] = {f"{r_['m']}/{r_['t']}/{r_['cell']}": r_["sqT_nonzero"] for r_ in rec}
res["pooled"]["zero_above_gstar"] = {nm: int(sum(r_[nm]["difflo"] > 0 for r_ in rec if r_["glo"] <= 0)) for nm in ("das_sq_T_s0_normal", "das_jsd_s0_normal")}

# the positive control: small_npi_fg_agr's direct off-route test (inside_offroute_direct.py) and its positive-part shares
if SET == "small_npi_fg_agr":
    OR = json.load(open(R / "experiments/results/inside_sets/small_npi_fg_agr_offroute.post.json"))
    assert res["pooled"]["off_or_indeterminate"] == OR["off"] and res["pooled"]["gstar_zero"] == OR["zero"], "small_three's off-route cells"
    assert res["pooled"]["off_route"] == OR["off_above_gstar_raw"], "small_three's off-route cells above G*"
    assert res["pooled"]["zero_above_gstar"] == OR["zero_above_gstar"], "small_three's zero class against G*"
    pos = [v["forms"]["pos"]["share"] for v in res["models"].values()]; att = [v["forms"]["pos"]["through_attention_of_it"] for v in res["models"].values()]
    assert (f"{min(pos):.2f}", f"{max(pos):.2f}") == ("0.56", "0.65") and (f"{min(att):.2f}", f"{max(att):.2f}") == ("0.57", "0.72"), (pos, att)
    print("control: small_three's off-route counts and positive-part shares reproduced")
dest = R / f"experiments/results/inside_sets/{SET}_forms.post.json"
json.dump(res, open(dest, "w"), indent=1, default=float)
for m, v in res["models"].items():
    f_ = v["forms"]
    print(f"{m:14s} live {v['live']:5d} aboveG {v['above_G_bh']:5d} carried {v['above_G_bh_in_carried']}/{v['carried']} off {v['off_route']} "
          f"indet {v['indeterminate']} | pos {f_['pos']['share']:.3f} [{f_['pos']['lo']:.3f}, {f_['pos']['hi']:.3f}] signed "
          f"{f_['signed']['share']:.3f} [{f_['signed']['lo']:.3f}, {f_['signed']['hi']:.3f}] cons {f_['cons']['share']:.3f} "
          f"[{f_['cons']['lo']:.3f}, {f_['cons']['hi']:.3f}] att(pos) {f_['pos']['through_attention_of_it']:.3f}")
print("pooled", {k: v for k, v in res["pooled"].items() if k != "forms"}, {n: round(v["share"], 3) for n, v in res["pooled"]["forms"].items()})
for m, v in res["checks"]["models"].items():
    print(f"{m:14s} checks: attention direct {v['attention_direct']['share']:.3f} [{v['attention_direct']['lo']:.3f}, "
          f"{v['attention_direct']['hi']:.3f}], not produced in context {v['not_produced_in_context']['share']:.3f}, above G* in context "
          f"{v['above_gstar_in_context']}, excess over G removed: every pattern {v['excess_over_G_removed']['every_pattern']:.3f}, "
          f"readout row {v['excess_over_G_removed']['readout_row']:.3f}")
print("written", dest)

"""The scorer of the maps inside the network (Section 6; App. D.1): per cell and per model, tests trace-families on the maps of
experiments/scripts/inside.py. trace, the midpoint and base-point traces against the network's own interchange G; ceiling, each cross-entropy
start against the first-order top lambda_1 (Bonferroni over the three starts); excess_rank, the first-order excess (lambda_1 - tr M)/T
against DAS's measured excess over G; above_own, cross-entropy DAS against G, G*_train and G*_test (and in the best context); attention_share, the
shares of an effect that freezing attention removes; door_readings, the door (Proposition 8): (a) how well first order describes the
change of the attention score at the site, (b) whether the predicted opening ranks the share freezing removes, (c) whether the
site's attention weight rises most in block L; squared_error, the squared error aimed at G: (a) where it lands against G, (b) its route
against cross-entropy's; families, per task family. experiments/tests/inside_scorer_test.py tests it on synthetic maps.

Inputs: experiments/results/inside/inside_<model>_<task>[_<tag>].{json,npz} (experiments/scripts/inside.py), each npz checked against
the sha256 its json names, and every file each run executed against this tree (experiments/lib/provenance.py).
Every test uses live cells only (Delta != 0 in some row); a map enters with >= 20 training and test rows, >= 20 test clusters
and T's interval above 0. Per cell, on the test pairs: cluster-bootstrap resamples (a row and its mirror one cluster;
regenerated and checked against the run's checksum); every ratio a ratio of means on the same resample; basic intervals for
ratios with lambda_1. Effects in units of T. The ce estimator is the mean of its three starts' per-pair effects. Across cells,
the unit is the map: within-map rank correlation, averaged over maps, with a map-level bootstrap. Benjamini-Hochberg at
q = 0.025 on ceiling's and above_own's per-cell counts. Instruments (an unpatched run with attention frozen against the base run, compared
as logits; at the readout cell the network's own interchange against T, compared as effects): within float32's rounding bound
at the model's logit scale (experiments/lib/instrument_tol.py).
Usage: <venv python> -I experiments/score/score_inside.py [--models=<tag>,...] [--tasks=<task>,...] [--fake=<dir>]
"""
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
from measure import bh, boot_counts, boot_ratio, interval, p_above  # noqa: E402
from instrument_tol import tol  # noqa: E402
from provenance import not_this_code  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
RES = Path(A["fake"]) if A.get("fake") else ROOT / "experiments/results/inside"
Q, NB = 0.025, 2000


def family(task):
    return "npi/fg" if task.startswith(("npi_", "filler_gap_")) else ("agreement" if task.startswith("agr_") else "other")


def provenance(J):
    """None when the run is the full grid with the DAS directions trained together (stacked) and every file it executed
    (sha256 recorded by the run) is this tree's (experiments/lib/provenance.py); else why not."""
    if A.get("fake"):
        return None
    if not J.get("full_grid"):
        return f"not a full-grid stacked run (cells {J.get('cells_arg')}, stack {J.get('stack')})"
    return not_this_code(J.get("code_sha256"))


def load():
    out, stop = {}, []
    for f in sorted(RES.glob("inside_*.json")):
        J = json.load(open(f)); fz = f.with_suffix(".npz")
        if not fz.is_file() or J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
            stop.append(f"{fz.name}: not the npz its json names"); continue
        if why := provenance(J):
            stop.append(f"{f.name}: {why}"); continue
        m = f.stem[len("inside_"):].split("_" + J["task"], 1)[0]
        if A.get("models") and m not in A["models"].split(","):
            continue
        if A.get("tasks") and J["task"] not in A["tasks"].split(","):
            continue
        if (m, J["task"]) in out:
            stop.append(f"two outputs for {m} {J['task']} ({f.name}): keep one"); continue
        out[(m, J["task"])] = (J, np.load(fz))
    if stop:
        print("STOP:\n  " + "\n  ".join(stop)); sys.exit(2)
    return out


def rank(v):
    return np.argsort(np.argsort(v))


def sp(a, b):
    """Rank correlation on ordinal ranks (ties broken by order); NaN with fewer than three cells; 0 when either side is
    constant (no rank information)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def map_level(per_map, B=NB, seed=11):
    """Mean over maps of a per-map statistic, with a map-level bootstrap interval (NaN maps left out)."""
    v = np.array([x for x in per_map if not np.isnan(x)])
    if len(v) == 0:
        return [None, None, None, 0]
    idx = np.random.default_rng(seed).integers(0, len(v), size=(B, len(v))); m = v[idx].mean(1)
    return [float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), int(len(v))]


def share_removed(free, frozen, C):
    """1 - mean(frozen)/mean(free) per resample, and its point value."""
    r = 1 - boot_ratio(frozen, free, C)
    return float(1 - frozen.mean() / free.mean()), r


def cell_stats(J, Z, row, C):
    c = f"L{row['layer']}r{row['region']}"; k = lambda n: Z[f"{c}/{n}"]  # noqa: E731
    T = Z["T"]; Tm = float(T.mean())
    e = lambda n: k(f"e/{n}")  # noqa: E731
    ce = np.mean([e(f"das_ce_s{s}_normal") for s in (0, 1, 2)], axis=0); ce_fr = np.mean([e(f"das_ce_s{s}_frozen") for s in (0, 1, 2)], axis=0)
    G, G_fr = e("own_normal"), e("own_frozen")
    Gt, Gte = k("gstar_train"), k("gstar_test")
    R = lambda num, den=T, **kw: boot_ratio(num, den, C, **kw)  # noqa: E731
    s = dict(cell=c, layer=row["layer"], region=row["region"], door=row["door"], G_T=float(G.mean() / Tm),
             Gstar_train_T=float(Gt.mean() / Tm), Gstar_test_T=float(Gte.mean() / Tm), ce_T=float(ce.mean() / Tm))
    # trace
    trm, trb = k("tr_mid"), k("tr_base")
    s["trace"] = dict(tr_mid_over_G=float(trm.mean() / G.mean()) if abs(G.mean()) > 1e-9 else None,
                   tr_base_over_G=float(trb.mean() / G.mean()) if abs(G.mean()) > 1e-9 else None,
                   mid_closer=bool(abs((trm - G).mean()) < abs((trb - G).mean())))
    # ceiling: every ce start against lambda_1
    lam = k("lam_mid_boot"); ceiling, ps = {}, []
    for sd in (0, 1, 2):
        x = e(f"das_ce_s{sd}_normal"); pt = float(x.mean() / row["lambda1_mid"]) if row["lambda1_mid"] != 0 else np.nan
        r = R(x, None, den_boot=lam); lo, hi = interval(pt, r, basic=True)
        ceiling[sd] = dict(pt=pt, lo=lo, hi=hi, above=lo > 1); ps.append(p_above(pt, r) if not np.isnan(pt) else 1.0)
    s["ceiling"] = ceiling; s["ceiling_p"] = min(1.0, 3 * min(ps))
    # excess_rank: predicted and measured excess
    s["pred"] = float((row["lambda1_mid"] - row["trM_mid"]) / Tm)
    s["pred_v1"] = float((k("fm/v1_mid").mean() - row["trM_mid"]) / Tm)
    s["meas_ce"] = float((ce - G).mean() / Tm); s["meas_eff"] = float((e("das_eff_s0_normal") - G).mean() / Tm)
    # above_own: inflation against G, G*_train, G*_test (and in context)
    for name, ref in (("G", G), ("Gstar_train", Gt), ("Gstar_test", Gte)):
        r = R(ce - ref); pt = float((ce - ref).mean() / Tm); lo, hi = interval(pt, r)
        s[f"above_own_{name}"] = dict(pt=pt, lo=lo, hi=hi, above=lo > 0, below=hi < 0, p=p_above(pt, r, 0.0))
    if f"{c}/ctx/own" in Z.files:
        cc = np.mean([k(f"ctx/das_ce_s{sd}") for sd in (0, 1, 2)], axis=0); g_ctx = k("ctx/own")
        s["above_own_ctx"] = dict(meas=float((cc - g_ctx).mean() / Tm),
                           pred=float((row["lambda1_ctx"] - row["trM_ctx"]) / Tm) if row.get("lambda1_ctx") is not None else None)
    # attention_share: shares removed by freezing attention
    sd_pt, sd_r = share_removed(ce, ce_fr, C)
    Gr = R(G, np.ones_like(G)); own_defined = interval(float(G.mean()), Gr)[0] > 0 or interval(float(G.mean()), Gr)[1] < 0
    s["attention_share"] = dict(s_das=sd_pt, s_das_ci=list(interval(sd_pt, sd_r)), own_defined=own_defined,
                   s_das_row=share_removed(ce, np.mean([e(f"das_ce_s{x}_row") for x in (0, 1, 2)], axis=0), C)[0])
    if own_defined:
        so_pt, so_r = share_removed(G, G_fr, C); d = sd_r - so_r; lo, hi = interval(sd_pt - so_pt, d)
        s["attention_share"].update(s_own=so_pt, diff=[sd_pt - so_pt, lo, hi], das_more_through_attention=lo > 0)
    else:
        s["attention_share"]["das_more_through_attention"] = None
    s["attention_share"]["natural"] = {n: share_removed(e(f"{n}_normal"), e(f"{n}_frozen"), C)[0] if abs(e(f"{n}_normal").mean()) > 1e-9 else None
                          for n in ("u_hat", "b_hat", "bisector", "v1_mid")}
    cr = R(ce, np.ones_like(ce)); clo, chi = interval(float(ce.mean()), cr); s["ce_nonzero"] = clo > 0 or chi < 0
    aligned = Z["aligned"].astype(bool)
    if aligned.sum() >= 10 and abs(ce[aligned].mean()) > 1e-9:
        s["attention_share"]["s_das_aligned"] = float(1 - ce_fr[aligned].mean() / ce[aligned].mean())
        s["attention_share"]["not_produced_aligned"] = float(max(ce[aligned].mean() - Gte[aligned].mean(), 0) / Tm)
    # door_readings: the door
    if row["door"]:
        al, ad, an, dn = k("door/alpha"), k("door/a_dot_delta"), k("door/a_norm_delta_norm"), k("door/delta_norm")
        half = Z["half"].astype(bool); dso = k("door/dscore_own")
        def corr(x, y):
            x, y = np.ravel(x), np.ravel(y)
            return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        s["door_adequacy"] = dict(own_adequacy_corr=corr(dso, ad), das_first_corr=corr(k("door/dscore_das_ce_s0"), k("door/first_das_ce_s0")),
                        das_above_pair_ceiling=float(np.mean(k("door/dscore_das_ce_s0") > 0.5 * (ad + an) + 1e-4 * np.maximum(1.0, np.abs(0.5 * (ad + an))))),
                        own_adequacy_by_half=[corr(dso[h_], ad[h_]) for h_ in (half, ~half)])
        opens = []
        for hname, sel in (("half0", half), ("half1", ~half)):
            if f"{c}/door/range_lam_{hname}" in Z.files and sel.sum() >= 2:
                Lh = np.median(np.log(1 / al[sel]), axis=0)
                opens.append(np.max(k(f"door/range_lam_{hname}") - Lh))
        s["door_predictor"] = dict(pred_open=float(max(opens)) if opens else None, naive_delta=float(dn.mean()), naive_alpha=float(np.median(al, axis=0).max()))
        wb, wd = k("door/w_base"), k("door/w_das_ce_s0")
        rises = np.stack([(wd[half] - wb[half]).mean(0), (wd[~half] - wb[~half]).mean(0)])   # halves x blocks x heads
        flat = np.unravel_index(np.argmax(rises), rises.shape)
        s["door_attention_rise"] = dict(max_rise=float(rises.max()), argmax_in_block_L=bool(flat[1] == 0),
                        block_L_share=float(rises[:, 0].max() / rises.max()) if rises.max() > 0 else None)
    # squared_error: the objectives aimed at G. (a) Where each lands against G (Corollary 6(ii): sqm on G when lambda_D <= G <= lambda_1,
    # at the optimum), with the per-pair sq_G and the training-free u-hat and b-hat beside it; (b) the route, sqm s=0 against
    # ce s=0 (same start).
    w_ = C.sum(1); mG = C @ G / w_; land = {}
    for name, x in (("sqm", e("das_sqm_G_s0_normal")), ("sqG", np.mean([e(f"das_sq_G_s{x_}_normal") for x_ in (0, 1, 2)], axis=0)),
                    ("u_hat", e("u_hat_normal")), ("b_hat", e("b_hat_normal")), ("ce", ce)):
        r = (C @ x / w_ - mG) / Tm; pt = float((x.mean() - G.mean()) / Tm); lo, hi = interval(pt, r)
        land[name] = dict(pt=pt, lo=lo, hi=hi, contains0=lo <= 0 <= hi)
    in_range = row["lambdaD_mid"] <= float(G.mean()) <= row["lambda1_mid"]
    s["squared_error"] = dict(in_range=in_range, land=land)
    ce0, ce0_fr = e("das_ce_s0_normal"), e("das_ce_s0_frozen"); sq0, sq0_fr = e("das_sqm_G_s0_normal"), e("das_sqm_G_s0_frozen")
    if s["ce_nonzero"] and abs(sq0.mean()) > 1e-9:
        a_pt, a_r = share_removed(sq0, sq0_fr, C); b_pt, b_r = share_removed(ce0, ce0_fr, C)
        lo, hi = interval(a_pt - b_pt, a_r - b_r); s["squared_error"]["route_diff"] = [a_pt - b_pt, lo, hi]
    return s


data = load(); report = dict(maps={}, models={}); cells = {}
for (m, t), (J, Z) in sorted(data.items()):
    C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"])
    if hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16] != J["boot_counts_sha"]:
        raise SystemExit(f"{m} {t}: the regenerated resamples are not the run's (checksum)")
    T = Z["T"]; Tr = boot_ratio(T, np.ones_like(T), C); Tlo = interval(float(T.mean()), Tr)[0]
    inst = [J.get("frozen_no_patch_max_abs", np.inf)] + [c.get("own_vs_T_max_abs") for c in J["cells"] if c.get("own_vs_T_max_abs") is not None]
    fake = bool(A.get("fake"))      # the frozen unpatched run compares logits (k = 2), the own interchange at the readout an effect (k = 8)
    inst_ok = inst[0] <= tol(m, "logit", fake=fake) and all(v <= tol(m, "effect", fake=fake) for v in inst[1:])
    enters = J["n_train"] >= 20 and J["n_test"] >= 20 and J["n_clusters_test"] >= 20 and Tlo > 0
    report["maps"][f"{m}/{t}"] = dict(instruments="OK" if inst_ok else "FAILED", enters=enters,
                                      live=sum(c["live"] for c in J["cells"]), dead=sum(not c["live"] for c in J["cells"]))
    if not enters or report["maps"][f"{m}/{t}"]["instruments"] != "OK":
        continue
    cells.setdefault(m, []).extend(dict(cell_stats(J, Z, c, C), model=m, task=t, family=family(t)) for c in J["cells"] if c["live"])
def partial_sp(x, y, z):
    """Rank correlation of x and y given z: the ordinal ranks of x and y residualised on the rank of z, then correlated."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if len(x) < 4:
        return np.nan
    rx, ry, rz = rank(x).astype(float), rank(y).astype(float), rank(z).astype(float)
    if np.std(rz) == 0:
        return sp(x, y)
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)  # noqa: E731
    a, b = res(rx), res(ry)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0


def excess_rank_maps(bym, measured):
    rho = [sp([c["pred"] for c in v], [c[measured] for c in v]) for v in bym.values()]
    part = [partial_sp([c["pred"] for c in v], [c[measured] for c in v], [c["G_T"] for c in v]) for v in bym.values()]
    naive = [sp([c["G_T"] for c in v], [c[measured] for c in v]) for v in bym.values()]
    return rho, part, naive


def door_predictor_maps(bym):
    out = {k: [] for k in ("rho", "partial_G", "naive_delta", "naive_alpha", "naive_minus_G")}
    for v in bym.values():
        v = [c for c in v if c.get("door_predictor") and c["door_predictor"]["pred_open"] is not None and c["ce_nonzero"]]
        P, Y = [c["door_predictor"]["pred_open"] for c in v], [c["attention_share"]["s_das"] for c in v]
        out["rho"].append(sp(P, Y)); out["partial_G"].append(partial_sp(P, Y, [c["G_T"] for c in v]))
        out["naive_delta"].append(sp([c["door_predictor"]["naive_delta"] for c in v], Y))
        out["naive_alpha"].append(sp([c["door_predictor"]["naive_alpha"] for c in v], Y))
        out["naive_minus_G"].append(sp([-c["G_T"] for c in v], Y))
    return out


def summarise_maps(bym):
    """The strong forms over a set of maps (excess_rank and door_readings), each map one unit."""
    out = {"maps": len(bym)}
    for measured in ("meas_ce", "meas_eff"):
        rho, part, naive = excess_rank_maps(bym, measured)
        out[f"excess_rank_{measured}"] = dict(rho=map_level(rho), partial_given_G=map_level(part), rho_naive_G=map_level(naive))
        out[f"excess_rank_{measured}"]["strong_form"] = bool(out[f"excess_rank_{measured}"]["partial_given_G"][1] is not None and out[f"excess_rank_{measured}"]["partial_given_G"][1] > 0)
    d = door_predictor_maps(bym); best = [max(a, b, c_) if not any(np.isnan([a, b, c_])) else np.nan
                             for a, b, c_ in zip(d["naive_delta"], d["naive_alpha"], d["naive_minus_G"])]
    out["door_predictor"] = dict(rho=map_level(d["rho"]), partial_given_G=map_level(d["partial_G"]),
                      diff_vs_best_naive=map_level([a - b for a, b in zip(d["rho"], best)]),
                      naive=dict(delta=map_level(d["naive_delta"]), alpha=map_level(d["naive_alpha"]), minus_G=map_level(d["naive_minus_G"])))
    out["door_predictor"]["strong_form"] = bool(all(x[1] is not None and x[1] > 0 for x in (out["door_predictor"]["partial_given_G"], out["door_predictor"]["diff_vs_best_naive"])))
    rd = [np.mean([c["squared_error"]["route_diff"][0] for c in v if c["squared_error"].get("route_diff")]) if any(c["squared_error"].get("route_diff") for c in v) else np.nan
          for v in bym.values()]
    out["squared_error_route"] = map_level(rd); out["squared_error_route"].append(bool(out["squared_error_route"][2] is not None and out["squared_error_route"][2] < 0))
    return out


allmaps = {}
for m, cs in cells.items():
    tasks = sorted({c["task"] for c in cs}); bym = {(m, t): [c for c in cs if c["task"] == t] for t in tasks}
    allmaps.update(bym)
    dc = [c for c in cs if c.get("door_predictor") and c["door_predictor"]["pred_open"] is not None]
    pop7 = [c for c in cs if c["above_own_G"]["above"] and c["squared_error"]["in_range"]]
    by_layer = {}
    for c in cs:
        if c["trace"]["tr_mid_over_G"] is not None:
            by_layer.setdefault(c["layer"], []).append(c["trace"]["tr_mid_over_G"])
    report["models"][m] = dict(
        maps=len(tasks), live_cells=len(cs), per_model=summarise_maps(bym) if len(tasks) >= 10 else "fewer than 10 maps: pooled only",
        trace=dict(median_tr_mid_over_G=st.median([c["trace"]["tr_mid_over_G"] for c in cs if c["trace"]["tr_mid_over_G"] is not None] or [np.nan]),
                median_tr_base_over_G=st.median([c["trace"]["tr_base_over_G"] for c in cs if c["trace"]["tr_base_over_G"] is not None] or [np.nan]),
                mid_closer=sum(c["trace"]["mid_closer"] for c in cs), by_layer={l: st.median(v) for l, v in sorted(by_layer.items())}),
        ceiling=dict(cells_above=sum(any(v["above"] for v in c["ceiling"].values()) for c in cs), cells_above_bh=int(bh([c["ceiling_p"] for c in cs], Q).sum())),
        excess_rank_lowest_quarter=dict(meas_ce_median=st.median([c["meas_ce"] for c in cs if c["pred"] <= np.quantile([x["pred"] for x in cs], 0.25)] or [np.nan])),
        above_own={n: dict(above=sum(c[f"above_own_{n}"]["above"] for c in cs), below=sum(c[f"above_own_{n}"]["below"] for c in cs),
                    above_bh=int(bh([c[f"above_own_{n}"]["p"] for c in cs], Q).sum())) for n in ("G", "Gstar_train", "Gstar_test")},
        above_own_ctx=dict(cells=sum(1 for c in cs if c.get("above_own_ctx")),
                    spearman_meas_pred=sp([c["above_own_ctx"]["pred"] for c in cs if c.get("above_own_ctx") and c["above_own_ctx"]["pred"] is not None],
                                          [c["above_own_ctx"]["meas"] for c in cs if c.get("above_own_ctx") and c["above_own_ctx"]["pred"] is not None])),
        attention_share=dict(das_more_through_attention=sum(bool(c["attention_share"]["das_more_through_attention"]) for c in cs),
                own_defined=sum(c["attention_share"]["own_defined"] for c in cs),
                not_produced_share_gstar_test=float(sum(max(c["ce_T"] - c["Gstar_test_T"], 0) for c in cs) / max(sum(c["ce_T"] for c in cs), 1e-12)),
                not_produced_through_attention=float(sum(max(c["ce_T"] - c["Gstar_test_T"], 0) * min(max(c["attention_share"]["s_das"], 0), 1) for c in cs)
                                                     / max(sum(c["ce_T"] for c in cs), 1e-12)),
                not_produced_share_aligned=st.median([c["attention_share"]["not_produced_aligned"] for c in cs if c["attention_share"].get("not_produced_aligned") is not None] or [np.nan]),
                natural_median_share={n: st.median([c["attention_share"]["natural"][n] for c in cs if c["attention_share"]["natural"][n] is not None] or [np.nan])
                                      for n in ("u_hat", "b_hat", "bisector", "v1_mid")}),
        door_readings=dict(door_cells=len(dc),
                a=dict(own_adequacy_corr_median=st.median([c["door_adequacy"]["own_adequacy_corr"] for c in dc if not np.isnan(c["door_adequacy"]["own_adequacy_corr"])] or [np.nan]),
                       das_first_corr_median=st.median([c["door_adequacy"]["das_first_corr"] for c in dc if not np.isnan(c["door_adequacy"]["das_first_corr"])] or [np.nan]),
                       das_above_pair_ceiling_median=st.median([c["door_adequacy"]["das_above_pair_ceiling"] for c in dc] or [np.nan])),
                c=dict(argmax_in_block_L=sum(c["door_attention_rise"]["argmax_in_block_L"] for c in dc),
                       block_L_share_median=st.median([c["door_attention_rise"]["block_L_share"] for c in dc if c["door_attention_rise"]["block_L_share"] is not None] or [np.nan]))),
        squared_error=dict(population=len(pop7),
                land_contains0={n: sum(c["squared_error"]["land"][n]["contains0"] for c in pop7) for n in ("sqm", "sqG", "u_hat", "b_hat", "ce")},
                land_median_abs={n: st.median([abs(c["squared_error"]["land"][n]["pt"]) for c in pop7] or [np.nan]) for n in ("sqm", "sqG", "u_hat", "b_hat", "ce")}),
        families={fam: dict(cells=len([c for c in cs if c["family"] == fam]),
                      pred_median=st.median([c["pred"] for c in cs if c["family"] == fam] or [np.nan]),
                      pred_open_median=st.median([c["door_predictor"]["pred_open"] for c in dc if c["family"] == fam] or [np.nan]),
                      s_das_median=st.median([c["attention_share"]["s_das"] for c in cs if c["family"] == fam] or [np.nan]))
            for fam in ("npi/fg", "agreement", "other")})
report["pooled"] = summarise_maps(allmaps) if allmaps else None

def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items()}
    if isinstance(x, (np.bool_, np.integer)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


out = dict(report=strip(report), cells={m: [strip(c) for c in cs] for m, cs in cells.items()})
dest = RES / "score_inside.json" if A.get("fake") else ROOT / "experiments/results/score_inside.json"
json.dump(out, open(dest, "w"), indent=1, default=float)
for m, r in report["models"].items():
    print(f"\n== {m}: {r['maps']} maps, {r['live_cells']} live cells")
    for key in ("per_model", "trace", "ceiling", "excess_rank_lowest_quarter", "above_own", "above_own_ctx", "attention_share", "door_readings", "squared_error", "families"):
        print(f"  {key}: {json.dumps(strip(r[key]), default=float)}")
if report.get("pooled"):
    print(f"\n== pooled over every map ({report['pooled']['maps']}): {json.dumps(strip(report['pooled']), default=float)}")
print(f"\nwritten {dest}")

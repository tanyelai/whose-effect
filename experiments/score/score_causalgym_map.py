"""CausalGym's own aggregate on the rebuilt maps at every layer, and the part of DAS's lead over vanilla that the network does
not produce (Section 7.1 and Appendix E.1, Table 8). experiments/tests/causalgym_map_scorer_test.py tests it on synthetic maps with known
answers.
Inputs: per model and CausalGym task, the inside map's odd layers and readout cell (experiments/results/inside/inside_<m>_<t>.json,
.npz) and the even-layer run (experiments/results/inside_even/inside_<m>_<t>_even.json, .npz), both by experiments/scripts/inside.py.
Checks: every file each run executed is byte-identical in this tree (experiments/lib/provenance.py); the odd-layer map is the full grid
at the odd layers; the even run has exactly --layers=2,4,...,NL-2, the DAS directions trained together (stacked), no --cells;
both runs read the same test pairs (the clusters byte for byte, T within float32's scale for an effect,
experiments/lib/instrument_tol.py) and draw the same resamples; the readout cell agrees (the network's own interchange within the same
scale per pair; each ce start's mean within the range of the odd-layer run's three starts at that cell, the float32 effect
tolerance as the floor); the instruments hold (a map that fails them is listed under "excluded" and left out of its model's
shares and of the pooled share).
Per map, in units of T (ratio of means on each resample): per layer the best region of DAS (the mean of ce's three starts per
pair) and of vanilla (the network's own single swap); CausalGym's aggregate D, V, R = D / V, each start's own D_s, D-bar, R-bar;
the mean difference's aggregate U; the lead X = sum over layers of DAS at its best minus vanilla at its best, split exactly into
N (DAS beyond the network's own interchange in its best context, G*, at DAS's chosen cell) and P (that G* beyond vanilla's
best). G*'s context is chosen here on the test pairs, as CausalGym's aggregate chooses each method's region on the pairs it
scores; s_N_train_context gives N's share with the context chosen on the training pairs, as in Section 6. s_N = N / X; N's
share by region; N+ and the share of it that freezing attention removes. The selection is redone on every resample. Basic
intervals (2 x point - quantiles) for these maxima-built quantities, percentile beside them, and the resampling's bias and the
stability of each layer's choice.
Comparison with the published values (on the ten tasks the released code reads correctly, Pythia-160m/410m/1b; each map's
"published", and published_figure for the figure): vanilla equal to the published value at its two decimals; the published DAS inside the
95% range of our single-start aggregates; for Pythia-1b on npi_any_subj-relc, DAS's map aligned with the published figure's
darkness (Spearman at no shift above both one-layer shifts).
The off-route share s_off, the part of the lead from layers where DAS's chosen cell is one where the network's own
interchange in its best context does not lie above 0 (G*'s interval reaches 0 or lies below it): per model the ratio of the
sums over its tasks (off-route lead over lead) with a construction-level bootstrap (a task with its _mod twin), tested by its
interval lying above 0 (holds); pooled over the models likewise. This test is close to an existence test (a layer's part can
be negative where DAS's pick lies below vanilla's, but rarely), so its report also counts the constructions with an off-route
pick at the point and gives G*'s upper bound at those picks (T units: median, q90, max). A small true G* with a wide interval
is counted off-route. A cell whose true G* is 0 is counted carried (G*'s interval above 0) in one case in forty, and more
often because G*'s context is chosen on the same test pairs (up to about one in five with eight independent contexts); that
error lowers s_off. A map whose lead X is not established (X's basic interval reaches 0) has no shares of its own. s_N is
reported beside s_off. Descriptive: N's region and the off-route lead's region, each share with its interval; N+'s share
through attention with its interval.
Output: experiments/results/score_causalgym_map.json. Usage: python -I experiments/score/score_causalgym_map.py [--fake=<dir>] [--models=<tags>]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
sys.path.insert(0, str(ROOT))
from owneffect.stats import boot_counts  # noqa: E402   (the same function as experiments/lib/measure.py's; experiments/tests/owneffect_stats_identity_test.py checks they agree)

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = Path(A["fake"]) if A.get("fake") else None
ODD = FAKE / "inside" if FAKE else ROOT / "experiments/results/inside"
EVEN = FAKE / "inside_even" if FAKE else ROOT / "experiments/results/inside_even"
PUB = FAKE / "published.json" if FAKE else ROOT / "experiments/causalgym/causalgym_published_tables.json"
DARK = FAKE / "darkness.json" if FAKE else ROOT / "experiments/causalgym/causalgym_fig4_darkness.json"
CORRECT = ["filler_gap_embed_3", "filler_gap_embed_4", "filler_gap_hierarchy", "filler_gap_obj", "filler_gap_pp",
           "filler_gap_subj", "npi_any_obj-relc", "npi_any_subj-relc", "npi_ever_obj-relc", "npi_ever_subj-relc"]
PUBLISHED_SIZES = {"pythia-160m": "pythia-160m", "pythia-410m": "pythia-410m", "pythia-1b": "pythia-1b"}
NB = 2000
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/matrix"))
from regions import facts  # noqa: E402
from instrument_tol import tol  # noqa: E402
from provenance import not_this_code  # noqa: E402


def degenerate(J1, J2, task):
    """Cells left out of the choice in the primary reading: a region empty in some row of the task (the encoder then gives it
    the previous region's token), and cells whose best context is such a region (G* swaps that token too)."""
    if FAKE:
        rows = json.load(open(FAKE / "fake_rows" / f"{task}.json")) if (FAKE / "fake_rows" / f"{task}.json").is_file() else None
    else:
        import registry
        from huggingface_hub import hf_hub_download
        rows = {s: [r for r in json.load(open(hf_hub_download("aryaman/causalgym", f"{s}.json", repo_type="dataset",
                                                                revision=registry.CAUSALGYM_REV))) if r["task"] == task]
                for s in ("train", "test")}
    bad = set(facts(rows)["degenerate_regions"]) if rows else set()
    out = set()
    for J in (J1, J2):
        for c in J["cells"]:
            if c["region"] in bad or c.get("context_test") in bad:
                out.add((c["layer"], c["region"]))
    NL = max(c["layer"] for c in J1["cells"])
    return frozenset(k for k in out if k[0] < NL), sorted(bad)


def provenance(J):
    """None when every file the run executed (sha256 recorded by the run) is this tree's (experiments/lib/provenance.py)."""
    return None if FAKE else not_this_code(J.get("code_sha256"))


def load(m, t):
    """The odd-layer map and the even-layer run of one model and task, checked; (J1, Z1, J2, Z2) or a reason."""
    f1, f2 = ODD / f"inside_{m}_{t}.json", EVEN / f"inside_{m}_{t}_even.json"
    if not f1.is_file() or not f2.is_file():
        return f"missing {'odd' if not f1.is_file() else 'even'}-layer output"
    out = []
    for f in (f1, f2):
        J = json.load(open(f)); fz = f.with_suffix(".npz")
        if not fz.is_file() or hashlib.sha256(fz.read_bytes()).hexdigest() != J["npz_sha256"]:
            return f"{fz.name}: not the npz its json names"
        if why := provenance(J):
            return f"{f.name}: {why}"
        out += [J, np.load(fz)]
    J1, Z1, J2, Z2 = out
    NL = max(c["layer"] for c in J1["cells"])
    if not J1.get("full_grid") or J1.get("layers") != list(range(1, NL, 2)):
        return "the odd-layer map is not a full grid at the odd layers"
    if J2.get("layers") != list(range(2, NL - 1, 2)) or J2.get("cells_arg") or not J2.get("stack"):
        return f"the even run's layers {J2.get('layers')} are not 2..{NL - 2} by twos (or not stacked, or with --cells)"
    te = tol(J1.get("model", ""), "effect", fake=FAKE is not None)       # two runs' T and own interchange: float32's scale for an effect
    if not (np.array_equal(Z1["cluster"], Z2["cluster"]) and float(np.abs(Z1["T"] - Z2["T"]).max()) <= te):
        return "the two runs did not read the same test pairs"
    if (J1["boot"], J1["boot_counts_sha"]) != (J2["boot"], J2["boot_counts_sha"]):
        return "the two runs drew different resamples"
    rc = f"L{NL}r{J1['cells'][-1]['region']}"; Tm = float(Z1["T"].mean())
    if np.abs(Z1[f"{rc}/e/own_normal"] - Z2[f"{rc}/e/own_normal"]).max() > te:
        return "the readout cell's own interchange differs between the runs"
    # the trained starts: each start's difference between the runs within the range of the odd-layer run's own three starts at
    # that cell (changing the start is a far larger perturbation than floating-point order), with the float32 effect tolerance
    # as the floor where the starts agree exactly
    ref = [float(Z1[f"{rc}/e/das_ce_s{s}_normal"].mean()) for s in (0, 1, 2)]; yard = max(max(ref) - min(ref), te)
    for s in (0, 1, 2):
        if abs(ref[s] - float(Z2[f"{rc}/e/das_ce_s{s}_normal"].mean())) > yard:
            return f"the readout cell's ce start {s} differs between the runs by more than its starts' range ({yard / abs(Tm):.3f} T)"
    return J1, Z1, J2, Z2


def instruments_ok(J, Z):
    inst = [J.get("frozen_no_patch_max_abs", np.inf)] + [c.get("own_vs_T_max_abs") for c in J["cells"] if c.get("own_vs_T_max_abs") is not None]
    m, fake = J.get("model", ""), FAKE is not None                       # float32's scale at the model's logits:
    return inst[0] <= tol(m, "logit", fake=fake) and all(v <= tol(m, "effect", fake=fake) for v in inst[1:])  # logits k = 2, effects k = 8


def interval(pt, r, basic):
    lo, hi = float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
    return [2 * pt - hi, 2 * pt - lo] if basic else [lo, hi]


def per_map(J1, Z1, J2, Z2, C, exclude=frozenset()):
    """Every quantity of one map, on the point (all-ones weights, row 0) and every resample (rows 1..B). The choices of
    region are made on log odds (dividing by T after, so a resample with mean T <= 0 cannot invert them); `exclude` holds the
    degenerate cells, left out of the choice (the primary reading) or not (the sensitivity reading)."""
    NL = max(c["layer"] for c in J1["cells"]); cells = []
    for J, Z in ((J1, Z1), (J2, Z2)):
        for c in J["cells"]:
            if (J is J2 and c["layer"] == NL) or (c["layer"], c["region"]) in exclude:
                continue                                                  # the readout cell from the odd run; degenerate cells out
            if c.get("live", True) and f"L{c['layer']}r{c['region']}/e/own_normal" not in Z.files:
                raise KeyError(f"live cell L{c['layer']}r{c['region']} has no arrays")
            cells.append((c["layer"], c["region"], Z if c.get("live", True) else None))   # a dead cell: every effect 0
    W = np.vstack([np.ones(C.shape[1]), C]).astype(float); w = W.sum(1)
    T = Z1["T"]; Tm = W @ T / w                                                # (B+1,)
    raw = lambda x: W @ x / w                                                  # noqa: E731
    layers = sorted({L for L, _, _ in cells}); zero = np.zeros(len(T))
    get = lambda Z, c, k: Z[f"{c}/{k}"] if Z is not None else zero              # noqa: E731
    E, G, Gs, Gs_tr, Efr, U, Es = {}, {}, {}, {}, {}, {}, {s: {} for s in (0, 1, 2)}
    for L, r, Z in cells:
        c = f"L{L}r{r}"
        E[(L, r)] = raw(np.mean([get(Z, c, f"e/das_ce_s{s}_normal") for s in (0, 1, 2)], axis=0)); G[(L, r)] = raw(get(Z, c, "e/own_normal"))
        Gs[(L, r)] = raw(get(Z, c, "gstar_test")) if L < NL else G[(L, r)]
        Gs_tr[(L, r)] = raw(get(Z, c, "gstar_train")) if L < NL else G[(L, r)]
        Efr[(L, r)] = raw(np.mean([get(Z, c, f"e/das_ce_s{s}_frozen") for s in (0, 1, 2)], axis=0))
        U[(L, r)] = raw(get(Z, c, "e/u_hat_normal"))
        for s in (0, 1, 2):
            Es[s][(L, r)] = raw(get(Z, c, f"e/das_ce_s{s}_normal"))
    # off-route: G*'s interval does not lie above 0 (indistinguishable from 0, or below it); never the readout cell, where G* = T
    off = {k: bool(k[0] < NL and np.quantile(v[1:], 0.025) <= 0) for k, v in Gs.items()}
    nb = W.shape[0]; idx = np.arange(nb)
    def best(M, L):
        keys = [k for k in M if k[0] == L]; A_ = np.stack([M[k] for k in keys])   # (regions, B+1), log odds
        j = A_.argmax(0); return [keys[x] for x in j], A_[j, idx]
    D = V = Ug = N = P = Ntr = Xoff = Nplus_in = Natt_in = N_read = 0.0; Ds = {s: 0.0 for s in (0, 1, 2)}
    by_region, off_region, layer_rec = {}, {}, {}
    for L in layers:
        kd, d = best(E, L); _, v = best(G, L); _, u = best(U, L)
        gs = np.array([Gs[k][i] for i, k in enumerate(kd)]); gstr = np.array([Gs_tr[k][i] for i, k in enumerate(kd)])
        efr = np.array([Efr[k][i] for i, k in enumerate(kd)]); offd = np.array([off[k] for k in kd], float)
        D = D + d; V = V + v; Ug = Ug + u; N = N + (d - gs); P = P + (gs - v); Ntr = Ntr + (d - gstr); Xoff = Xoff + offd * (d - v)
        if L == NL:
            N_read = d - gs
        else:                                                             # through attention: the readout cell has no attention to freeze
            npos = np.maximum(d - gs, 0); Nplus_in = Nplus_in + npos
            sdas = np.divide(efr, d, out=np.ones_like(d), where=np.abs(d) > 1e-12); Natt_in = Natt_in + npos * np.clip(1 - sdas, 0, 1)
            for i, k in enumerate(kd):
                by_region.setdefault(k[1], np.zeros(nb))[i] += (d - gs)[i]
                off_region.setdefault(k[1], np.zeros(nb))[i] += offd[i] * (d - v)[i]     # the off-route lead, by DAS's region
        ex = (d - gs) / Tm
        layer_rec[L] = dict(cell=list(kd[0]), cell_off=bool(off[kd[0]]), stable=float(np.mean([k == kd[0] for k in kd[1:]])),
                            excess_over_Gstar_T=dict(pt=float(ex[0]), ci=interval(float(ex[0]), ex[1:], True)),
                            Gstar_hi_T=float(np.quantile(Gs[kd[0]][1:], 0.975) / Tm[0]) if off[kd[0]] else None)
        for s in (0, 1, 2):
            Ds[s] = Ds[s] + best(Es[s], L)[1]
    n = len(layers); X = N + P; Dbar = np.mean([Ds[s] for s in (0, 1, 2)], axis=0)
    N_in = N - N_read
    div = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)  # noqa: E731
    q = dict(D=D / n / Tm, V=V / n / Tm, R=D / V, Dbar=Dbar / n / Tm, Rbar=Dbar / V, U=Ug / n / Tm, X=X / Tm, N=N / Tm, P=P / Tm,
             off_lead=Xoff / Tm, s_N=div(N, X), s_P=div(P, X), s_off=div(Xoff, X), s_N_train_context=div(Ntr, X), N_inner=N_in / Tm,
             N_readout=N_read / Tm, s_N_inner=div(N_in, X - N_read),
             through_attention=Natt_in / np.where(Nplus_in > 0, Nplus_in, np.nan))
    raw_out = dict(D=D / n, V=V / n, Ds={s: Ds[s] / n for s in (0, 1, 2)}, R=D / V, Rbar=Dbar / V)
    region = {r: v / np.where(np.abs(N_in) > 1e-12, N_in, np.nan) for r, v in by_region.items()}   # inner layers' N by region
    region_off = {r: v / np.where(np.abs(Xoff) > 1e-12, Xoff, np.nan) for r, v in off_region.items()}   # the off-route lead by region
    return q, raw_out, (region, region_off), layer_rec, layers


def summarise(v, basic=True):
    pt, r = float(v[0]), v[1:][~np.isnan(v[1:])]
    return dict(pt=pt, ci=interval(pt, r, basic) if len(r) else [None, None], pct=interval(pt, r, False) if len(r) else [None, None],
                bias=float(r.mean() - pt) if len(r) else None)


def boot_ratio_sums(groups, B=NB, seed=22):
    """Sum of numerators over sum of denominators of per-task (num, den) pairs, with a bootstrap over groups (a construction's
    tasks, in every model, drawn together)."""
    keys = sorted(groups); num = np.array([sum(a for a, _ in groups[k]) for k in keys]); den = np.array([sum(b for _, b in groups[k]) for k in keys])
    i = np.random.default_rng(seed).integers(0, len(keys), size=(B, len(keys)))
    r = num[i].sum(1) / den[i].sum(1)
    return [float(num.sum() / den.sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975)), len(keys)]


def off_picks(recs):
    """The off-route picks at the point (inner layers): constructions with one, and G*'s upper bounds there (T units)."""
    ub = [L["Gstar_hi_T"] for v in recs.values() for L in v["primary"]["per_layer"].values() if L["cell_off"] and L["Gstar_hi_T"] is not None]
    task = lambda key: key.split("/")[-1]  # noqa: E731   (keys are tasks, or model/task when pooled)
    cons = {construction(task(t)) for t, v in recs.items() if any(L["cell_off"] for L in v["primary"]["per_layer"].values())}
    return dict(constructions_with_off_pick=len(cons), constructions=len({construction(task(t)) for t in recs}), picks=len(ub),
                Gstar_hi_T=dict(median=float(np.median(ub)), q90=float(np.quantile(ub, 0.9)), max=float(np.max(ub))) if ub else None)


pairs = sorted({(f.stem[len("inside_"):-len("_even")].split("_", 1)[0], f.stem[len("inside_"):-len("_even")].split("_", 1)[1])
                for f in EVEN.glob("inside_*_even.json")})
if A.get("models"):
    pairs = [p for p in pairs if p[0] in A["models"].split(",")]
pub = json.load(open(PUB)) if PUB.is_file() else {}
report, stop = dict(maps={}, models={}, pooled={}, published_figure={}, excluded={}), []


def construction(task):
    return task[:-len("_mod")] if task.endswith("_mod") else task             # a task and its _mod twin share their templates


for m, t in pairs:
    got = load(m, t)
    if isinstance(got, str):
        stop.append(f"{m} {t}: {got}"); continue
    J1, Z1, J2, Z2 = got
    C = boot_counts(Z1["cluster"], J1["boot"]["B"], J1["boot"]["seed"])
    if hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16] != J1["boot_counts_sha"]:
        stop.append(f"{m} {t}: the regenerated resamples are not the run's"); continue
    excl, bad_regions = degenerate(J1, J2, t)
    rec = dict(instruments=instruments_ok(J1, Z1) and instruments_ok(J2, Z2), degenerate_regions=bad_regions, degenerate_cells=len(excl))
    for reading, ex in (("primary", excl), ("sensitivity", frozenset())):
        q, raw, region, layer_rec, layers = per_map(J1, Z1, J2, Z2, C, exclude=ex)
        rec[reading] = dict(layers=len(layers), **{k: summarise(v) for k, v in q.items()},
                            raw=dict(D=float(raw["D"][0]), V=float(raw["V"][0]), Ds=[float(raw["Ds"][s][0]) for s in (0, 1, 2)],
                                     Ds_range=[float(np.quantile(np.concatenate([raw["Ds"][s][1:] for s in (0, 1, 2)]), x)) for x in (0.025, 0.975)],
                                     Rbar=summarise(raw["Rbar"])),
                            N_inner_by_region={str(r): summarise(v) for r, v in sorted(region[0].items())},
                            off_lead_by_region={str(r): summarise(v) for r, v in sorted(region[1].items())},
                            per_layer={str(L): v for L, v in layer_rec.items()})
        if ex == excl and not excl:
            rec["sensitivity"] = "same as primary"; break
    for reading in ("primary", "sensitivity"):
        if isinstance(rec.get(reading), dict):
            r_ = rec[reading]; r_["lead_established"] = bool(r_["X"]["ci"][0] is not None and r_["X"]["ci"][0] > 0)
            if not r_["lead_established"]:                               # a lead that may be 0 has no shares of its own
                for k in ("s_N", "s_P", "s_off", "s_N_train_context", "s_N_inner"):
                    r_[k] = dict(r_[k], pt=None, ci=[None, None], note="lead X not established")
    prim = rec["primary"]
    if m in PUBLISHED_SIZES and t in CORRECT and f"{PUBLISHED_SIZES[m]}|main" in pub:
        pv = pub[f"{PUBLISHED_SIZES[m]}|main"][t]; Rpub = pv["das"] / pv["vanilla"]
        rec["published"] = dict(vanilla=pv["vanilla"], das=pv["das"], vanilla_ours=prim["raw"]["V"],
                                vanilla_difference=prim["raw"]["V"] - pv["vanilla"],
                                das_inside_single_start_range=prim["raw"]["Ds_range"][0] <= pv["das"] <= prim["raw"]["Ds_range"][1],
                                R_published=Rpub, R_inside_Rbar_interval=bool(prim["raw"]["Rbar"]["ci"][0] <= Rpub <= prim["raw"]["Rbar"]["ci"][1]))
    report["maps"][f"{m}/{t}"] = rec
    if not rec["instruments"]:
        report["excluded"][f"{m}/{t}"] = "instruments"
if stop:
    print("STOP:\n  " + "\n  ".join(stop)); sys.exit(2)
# the alignment with the published figure: Pythia-1b on npi_any_subj-relc against its darkness (regions 1..8, paper layers 0..14)
if "pythia-1b/npi_any_subj-relc" in report["maps"] and DARK.is_file():
    J1, Z1, J2, Z2 = load("pythia-1b", "npi_any_subj-relc")
    dark = np.array(json.load(open(DARK))["das"]["143000"])[1:]                  # regions 1..8 x 16 paper layers
    NL = max(c["layer"] for c in J1["cells"]); Tm = float(Z1["T"].mean())
    ours = np.full((dark.shape[0], NL - 1), np.nan)
    for J, Z in ((J1, Z1), (J2, Z2)):
        for c in J["cells"]:
            if c["layer"] < NL and c["region"] - 1 < dark.shape[0] and c.get("live", True):
                ours[c["region"] - 1, c["layer"] - 1] = np.mean([Z[f"L{c['layer']}r{c['region']}/e/das_ce_s{s}_normal"].mean() for s in (0, 1, 2)]) / Tm
    def sp(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b); a, b = a[ok], b[ok]
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    s0 = sp(ours.ravel(), dark[:, :NL - 1].ravel())
    sm = sp(ours[:, 1:].ravel(), dark[:, :NL - 2].ravel()); spl = sp(ours[:, :-1].ravel(), dark[:, 1:NL - 1].ravel())
    report["published_figure"]["alignment_pythia_1b_npi"] = dict(no_shift=s0, ours_one_later=sm, ours_one_earlier=spl, aligned=s0 > max(sm, spl))
# s_off (tested: its interval above 0) and s_N (reported): per model over its maps whose instruments hold, and pooled
models = sorted({k.split("/")[0] for k in report["maps"]})
for m in models:
    recs = {k.split("/")[1]: v for k, v in report["maps"].items() if k.startswith(m + "/") and v["instruments"]}
    if not recs:
        report["models"][m] = dict(tasks=0); continue
    rs = lambda key, reading="primary": {construction(t): [((v[reading] if isinstance(v[reading], dict) else v["primary"])[key[0]]["pt"],  # noqa: E731
                                                           (v[reading] if isinstance(v[reading], dict) else v["primary"])[key[1]]["pt"])
                                                          for tt, v in recs.items() if construction(tt) == construction(t)] for t in recs}
    s_off = boot_ratio_sums(rs(("off_lead", "X"))); s_n = boot_ratio_sums(rs(("N", "X")))
    s_n_in = boot_ratio_sums({c: [(a, b - nr) for (a, b), (nr, _) in zip(v, rs(("N_readout", "X"))[c])] for c, v in rs(("N_inner", "X")).items()})
    report["models"][m] = dict(tasks=len(recs), s_off=s_off, s_off_above_0=bool(s_off[1] > 0),
                               s_off_sensitivity=boot_ratio_sums(rs(("off_lead", "X"), "sensitivity")), off_picks=off_picks(recs),
                               s_N=s_n, s_N_inner=s_n_in,
                               through_attention_median=float(np.nanmedian([v["primary"]["through_attention"]["pt"] for v in recs.values()])),
                               published=[{"task": t, **v["published"]} for t, v in recs.items() if "published" in v])
judged = [m for m in models if report["models"][m]["tasks"]]
groups = {}
for k, v in report["maps"].items():
    if k.split("/")[0] in judged and v["instruments"]:
        groups.setdefault(construction(k.split("/")[1]), []).append((v["primary"]["off_lead"]["pt"], v["primary"]["X"]["pt"]))
if groups:
    iv = boot_ratio_sums(groups)
    report["pooled"] = dict(models=judged, s_off=iv, s_off_above_0=bool(iv[1] > 0),
                            off_picks=off_picks({k: v for k, v in report["maps"].items() if k.split("/")[0] in judged and v["instruments"]}))


def strip(x):
    if isinstance(x, dict):
        return {str(k): strip(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, (np.bool_, np.integer)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


dest = (FAKE if FAKE else ROOT / "experiments/results") / "score_causalgym_map.json"
json.dump(strip(report), open(dest, "w"), indent=1, default=float)
for m, v in report["models"].items():
    if v.get("tasks"):
        print(f"{m}: {v['tasks']} tasks, s_off {v['s_off'][0]:.2f} [{v['s_off'][1]:.2f}, {v['s_off'][2]:.2f}] -> "
              f"{'s_off_above_0' if v['s_off_above_0'] else 'does not hold'}; s_N {v['s_N'][0]:.2f}")
for k, why in report["excluded"].items():
    print(f"  excluded {k}: {why}")
if report.get("pooled"):
    p = report["pooled"]; print(f"pooled ({len(p['models'])} models): s_off {p['s_off'][0]:.2f} [{p['s_off'][1]:.2f}, {p['s_off'][2]:.2f}]")
print(f"written {dest}")

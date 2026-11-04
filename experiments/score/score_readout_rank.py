"""The rank sweep's scorer: the readout's ceiling, excess and objective tests at rank k (Section 5.4), checked on synthetic
outputs by experiments/tests/rank_scorer_test.py.

Inputs: experiments/results/readout_rank/readout_rank_<model>_m_<group>.{json,npz} (experiments/scripts/readout_rank.py), each the output of its
unit in experiments/matrix/registry.py (registry.unverified checks the json; the npz must have the sha256 the run wrote into the json);
the readout's outputs of the same groups (experiments/results/readout/, for the twin check), verified the same way; and the readout's
scored rows (experiments/results/score_readout.json), which give the population and every task's two-label flag (one_q), construction,
source and trace deviation. A missing or unverified output stops the scorer (exit 2) unless --models narrows it to complete
models.
Population: the readout's main population (rows with main and defined, arm capped).
Per task, model and rank k < D, on the test pairs: cluster-bootstrap resamples regenerated from the saved cluster ids and seed
(checked against the run's checksum); every ratio a ratio of means on the same resample; K_k (the first-order rank-k top)
recomputed on every resample by the run, and basic (bias-correcting) intervals for every ratio that involves it. The ranks are
named by value (1, 2, 4, 16, 64), q (the number of distinct labels less one) and D-1. Estimators are the mean of an objective's
three starts' per-pair effects.
- ceiling_k, the ceiling at rank k (Theorem 5(iii)): e/K_k for every ce start and budget is "above" when its basic interval lies
  above 1. Strong form: in no task, start, budget or rank. Raw counts, and Benjamini-Hochberg at q = 0.025 over a model's
  tasks (each task one family, Bonferroni within it over its starts, budgets and ranks). Each cell keeps its landing (f/K_k)
  and adequacy (e/f), and each violation the task's trace deviation (the readout's trace test, trace).
- Room at rank k: the out-of-sample rank-k top's e/T interval lies above 1 ("no room": below 1).
- excess_k, the excess at rank k: the ce estimator over T, per budget. Strong form: above 1 in every task with room at rank k and
  entirely below 1 in none, at every rank and budget. Tasks with no room at rank one and room at rank k are counted apart
  (not at D-1, where the room is |lambda_D| and comes nearly by construction).
- objective_k, the objective decides at rank k: the ce estimator minus the sq estimator (the same three starts and orders) above 0 in
  every task with room at rank k, at every rank and budget (the readout's objective test at rank k).
- Across ranks: the counts at rank one (the readout's ceiling, excess and objective tests on this run) and above it, and the
  tasks whose test holds at rank one and fails at rank k, or the reverse; "beyond_one" strong forms leave rank one out.
- one_question_ratios (reported, not tested), two-label tasks: (K_k - lambda_1)/T; the out-of-sample top's rank-k over rank-one ratio (exact);
  e_ce,k / e_ce,1 and e_eff,k / e_eff,1 (per budget): median, quartiles, the largest upper bound, counts above and below 1.
- gains (reported, not tested), tasks with more than one label pair: the out-of-sample top's gain from rank one to rank k and
  DAS's, each over T with its interval; the counts where each lies above 0, and both; their Spearman correlation over tasks
  with a construction-level bootstrap. (The gains share the test pairs, and the top starts from eff, which shares its starts
  and orders with ce: hence no test.)
- medians (reported, not tested), medians per rank: e_ce/K_k (both budgets) and its landing f_ce/K_k; e_eff/K_k after ten passes;
  e_sq/T (both budgets); first-order adequacy (e/f) of the Ky Fan subspace and of ce; ce's tenth pass over its first; ce's
  overlap with the Ky Fan subspace.
- RAVEL (experiments/lib/ravel_twoway.py): at every rank the room, excess_k and objective_k use the two-way interval by each value's entity, with the
  mirror-cluster values beside them; ceiling_k, one_question_ratios and gains keep the mirror-cluster bootstrap.
Constructions as in the readout's scorer: a construction holds a test when every one of its tasks does. Tiers as there: a tier
is pooled only when every one of its models passes the instruments, over the tasks present in all of them.
Instruments, per model; a model whose instruments fail is not scored. The run's own: the readout reproduces the logits and T
within the instruments' float32 tolerance at the model's logit scale (2 gamma_D S for a logit and 8 gamma_D S for an effect;
experiments/lib/instrument_tol.py); the resampled K_k at unit counts lies within 1e-6 of the spectrum's scale and K_1 there equals
lambda_1; per rank, the Ky Fan subspace's first-order value equals K_k and no stored subspace's exceeds it, within 1e-6 of the
scale, and the top's training value is at least each start's. The twin check against the readout: the test pairs' T within
the effect tolerance (absolute) and the same clusters; K_1 on the resamples within 1e-6 of the scale of the readout's lambda_1;
at rank one the effects of the test and the training M's top eigenvectors within the effect tolerance (absolute) of the
readout's v1_test_mid and v1_mid; each of the eighteen trained directions (three objectives, three starts, two budgets) within
the range of the readout's own three starts of that objective and budget, with the float32 effect tolerance as the floor, a
direction beyond it passing if the same task trained alone (--stack=0,
experiments/results/readout_rank/readout_rank_<model>_m_<group>_stack0.json) is within it; both runs on an NVIDIA B200. Reported, not
gating: the out-of-sample top's difference from the readout's, and the first tenth of each direction's first-pass loss against
the readout's.
Output: experiments/results/score_readout_rank.json and the printed report.
Usage: <venv python> -I experiments/score/score_readout_rank.py [--models=small|large|small_large|all|<tag>,...] [--fake=<dir>] [--out=<path>]
       --fake reads synthetic outputs from <dir> without the registry (experiments/tests/rank_scorer_test.py only).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/matrix")); sys.path.insert(0, str(ROOT / "experiments/lib"))
from owneffect.stats import bh, boot_counts, interval, p_above  # noqa: E402
from instrument_tol import tol  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/score"))
from ravel_twoway import RAVEL_MAIN, apply_twoway_rank  # noqa: E402
from score_readout_sentence import test_rows  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = A.get("fake")
SEEDS, BUDGETS, Q, OBJS = (0, 1, 2), (1, 10), 0.025, ("ce", "sq", "eff")
TWIN_FITS = [(o, s) for o in OBJS for s in SEEDS]
TOL_BOOT, GPU = 1e-6, "NVIDIA B200"
GRID = ("1", "2", "4", "16", "64")

if FAKE:
    RES = Path(FAKE); READOUT = RES / "readout"; SMALL, LARGE = json.load(open(RES / "models.json"))[:2]; TAG = lambda m: m  # noqa: E731
    SCORED = json.load(open(RES / "score_readout.json"))
else:
    import registry  # noqa: E402
    RES = ROOT / "experiments/results/readout_rank"; READOUT = ROOT / "experiments/results/readout"; SMALL, LARGE = registry.SMALL, registry.LARGE
    TAG = registry.tag; SCORED = json.load(open(ROOT / "experiments/results/score_readout.json"))
sel = A.get("models", "all")
EXTRA = [] if A.get("fake") else registry.SCALE + registry.QWEN32   # every model measured: the sizes within the families and Qwen2.5-32B too
MODELS = ({"small": SMALL, "large": LARGE, "small_large": SMALL + LARGE, "all": SMALL + LARGE + EXTRA}.get(sel)
          or [m for m in SMALL + LARGE + EXTRA if TAG(m) in sel.split(",")])


_GPU = {}


def gpu_of(output):
    """The GPU named by the run record that wrote an output (experiments/logs/<run>/done.json and env.json), from one index."""
    if not _GPU and (ROOT / "experiments/logs").is_dir():
        for d in (ROOT / "experiments/logs").iterdir():
            if (d / "done.json").is_file() and (d / "env.json").is_file():
                g = json.load(open(d / "env.json")).get("gpu")
                for o in json.load(open(d / "done.json")).get("outputs", []):
                    _GPU[o["path"]] = g
    return _GPU.get(output)


def load():
    """{(model, group): dict(r=(json, npz), v=(json, npz), stack0=(json, npz) or None, gpu=[the rank run's GPU, the readout run's GPU])};
    stops on anything missing or unverified."""
    out, stop = {}, []

    def pair(f):
        if not f.is_file():
            return None
        J = json.load(open(f)); fz = f.with_suffix(".npz")
        if not fz.is_file() or J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
            stop.append(f"{fz.name}: not the npz its json names"); return None
        return J, np.load(fz)
    if FAKE:
        for f in sorted(RES.glob("readout_rank_*_m_*.json")):
            if f.stem.endswith("_stack0"):
                continue
            m, g = f.stem[len("readout_rank_"):].split("_m_", 1)
            if m in [TAG(x) for x in MODELS]:
                a, b = pair(f), pair(READOUT / f"readout_{m}_m_{g}.json")
                if a and b:
                    out[(m, g)] = dict(r=a, v=b, stack0=pair(RES / f"readout_rank_{m}_m_{g}_stack0.json"), gpu=[GPU, GPU])
                elif b is None:
                    stop.append(f"readout_{m}_m_{g}.json: missing")
    else:
        bad = registry.unverified(["readout_rank", "readout"], models=MODELS)
        stop += [f"{o}: {why}" for o, why in bad.items()]
        readout_units = {(registry.tag(u["model"]), u["group"]): u for u in registry.units(MODELS, ["readout"], device="cuda")}
        for u in registry.units(MODELS, ["readout_rank"], device="cuda"):
            key = (registry.tag(u["model"]), u["group"])
            if u["output"] in bad or key not in readout_units or readout_units[key]["output"] in bad:
                continue
            a, b = pair(ROOT / u["output"]), pair(ROOT / readout_units[key]["output"])
            s0 = ROOT / u["output"].replace(".json", "_stack0.json")
            s0p = pair(s0) if s0.is_file() else None
            if s0p and gpu_of(str(s0.relative_to(ROOT))) is None:
                stop.append(f"{s0.name}: no run record"); s0p = None
            if a and b:
                out[key] = dict(r=a, v=b, stack0=s0p, gpu=[gpu_of(u["output"]), gpu_of(readout_units[key]["output"])])
    if stop:
        print(f"STOP: {len(stop)} outputs are missing or not their unit's:\n  " + "\n  ".join(stop[:40]))
        sys.exit(2)
    return out


def ratio(num, den, C, point, den_boot=None, basic=False):
    """A ratio of means: point value, interval and the resampled values (the denominator resampled by the run when given)."""
    w = C.sum(1)
    r = (C @ num / w) / ((C @ den / w) if den_boot is None else den_boot)
    lo, hi = interval(point, r, basic=basic)
    return dict(pt=float(point), lo=float(lo), hi=float(hi)), r


def diff_over_T(a, b, T, C):
    """(mean a - mean b) / mean T on the same resample; percentile interval."""
    w = C.sum(1); r = (C @ (a - b) / w) / (C @ T / w)
    pt = float((a.mean() - b.mean()) / T.mean()); lo, hi = interval(pt, r)
    return dict(pt=pt, lo=float(lo), hi=float(hi))


def names_of(rr):
    """(name, k) for a task's ranks: grid values by value, q, D-1."""
    D, q = rr["D"], rr.get("rank_q")
    out = [(str(k), k) for k in rr["ranks"] if str(k) in GRID and k != D - 1]
    if q:
        out.append(("q", q))
    return out + [("D-1", D - 1)]


def twin(Jr, Zr, Jv, Zv, task, S0):
    rr, vt = Jr["tasks"][task], Jv["tasks"][task]
    T, Tv = Zr[f"{task}/T"].astype(float), Zv[f"{task}/T"].astype(float); Tm = float(T.mean())
    mean = lambda Z, k: float(Z[k].astype(float).mean())  # noqa: E731
    # the trained directions' twin: training on a GPU is not repeated bit for bit, and changing the start is a far larger
    # perturbation than reordering floating-point sums, so each start's difference from the readout is judged against the range
    # of the readout's own three starts (the same objective and budget), with the float32 effect tolerance as its floor where the
    # starts agree exactly
    te_ = tol(Jr.get("model", ""), "effect", fake=bool(FAKE))
    fits = {}
    for o in OBJS:
        for p in BUDGETS:
            ref = [mean(Zv, f"{task}/e/das_{o}_p{p}_s{s}") for s in SEEDS]
            yard = max(max(ref) - min(ref), te_)
            for s in SEEDS:
                d = abs(mean(Zr, f"{task}/k1/das_{o}_p{p}_s{s}/e") - ref[s])
                d0 = None
                if d > yard and S0 is not None and task in S0[0]["tasks"]:
                    d0 = abs(mean(S0[1], f"{task}/k1/das_{o}_p{p}_s{s}/e") - ref[s])
                fits[f"{o}_s{s}_p{p}"] = dict(diff=d / abs(Tm), alone=None if d0 is None else d0 / abs(Tm), yardstick=yard / abs(Tm),
                                              over_float32=d / te_,          # reported: a gap above numerics inside the range shows
                                              ok=d <= yard or (d0 is not None and d0 <= yard))
    rel = lambda a, b: abs(a - b) / max(abs(b), 1.0)  # noqa: E731
    eig = max(rel(mean(Zr, f"{task}/k1/kyfan/e"), mean(Zv, f"{task}/e/v1_test_mid")),
              rel(mean(Zr, f"{task}/k1/kyfan_oos/e"), mean(Zv, f"{task}/e/v1_mid")))
    eig_abs = max(abs(mean(Zr, f"{task}/k1/kyfan/e") - mean(Zv, f"{task}/e/v1_test_mid")),
                  abs(mean(Zr, f"{task}/k1/kyfan_oos/e") - mean(Zv, f"{task}/e/v1_mid")))
    top = abs(mean(Zr, f"{task}/k1/top_train/e") - mean(Zv, f"{task}/e/top_train")) / abs(Tm)
    loss = max(abs(rr["ranks_out"]["1"]["losses"][f"das_{o}_p1_s{s}"][0] - vt["losses"][f"das_{o}_p1_s{s}"][0])
               / max(abs(vt["losses"][f"das_{o}_p1_s{s}"][0]), 1e-12) for o, s in TWIN_FITS)
    ro = [rr["ranks_out"][str(k)]["instruments"] for k in rr["ranks"]]
    inst = dict(readout=max(rr["instruments"][s]["readout_vs_logits"] for s in ("train", "test")),
                T_forward=max(rr["instruments"][s]["T_vs_forward"] for s in ("train", "test")),
                boot_check=rr["instruments"]["boot_check"], lam_check=rr["instruments"]["lam_check"],
                kyfan_first_order=max(x["kyfan_first_order"] for x in ro), above_kyfan=max(x["above_kyfan"] for x in ro),
                top_start=min(x["top_start"] for x in ro),
                twin_T=float(np.max(np.abs(T - Tv) / np.maximum(np.abs(Tv), 1))) if len(T) == len(Tv) else float("inf"),
                twin_clusters=bool(len(T) == len(Tv) and np.array_equal(Zr[f"{task}/cluster"], Zv[f"{task}/cluster"])),
                twin_lam=float(np.max(np.abs(Zr[f"{task}/kyfan_mid_boot"][:, 0] - Zv[f"{task}/lam_mid_boot"])) / rr["scale"]),
                twin_eigvectors=eig, twin_eigvectors_abs=eig_abs, twin_top=top, twin_loss=loss, twin_fits=max(a["diff"] for a in fits.values()),
                twin_fits_ok=all(a["ok"] for a in fits.values()), fits_alone=sorted(k for k, a in fits.items() if a["alone"] is not None),
                twin_fits_over_float32=max(a["over_float32"] for a in fits.values()))
    # the logit and T identities within the instruments' float32 scale at the model's logits; the twin's T and eigenvector
    # effects, compared in logit units (absolute differences), within the same tolerance for an effect
    tl, te = tol(Jr.get("model", ""), "logit", fake=bool(FAKE)), tol(Jr.get("model", ""), "effect", fake=bool(FAKE))
    inst["twin_T_abs"] = float(np.max(np.abs(T - Tv))) if len(T) == len(Tv) else float("inf")
    inst["ok"] = bool(inst["readout"] <= tl and inst["T_forward"] <= te and inst["boot_check"] <= TOL_BOOT
                      and inst["lam_check"] <= TOL_BOOT and inst["kyfan_first_order"] <= TOL_BOOT and inst["above_kyfan"] <= TOL_BOOT
                      and inst["top_start"] >= -1e-9 and inst["twin_T_abs"] <= te and inst["twin_clusters"]
                      and inst["twin_lam"] <= TOL_BOOT and inst["twin_eigvectors_abs"] <= te and inst["twin_fits_ok"])
    # the top's and the first-pass loss's twins are reported, not gating: the deterministic twins above and the trained directions
    # establish that the two codes compute the same thing; neither has a measured yardstick
    return inst


def task_stats(Jr, Zr, Jv, Zv, task, S0):
    rr = Jr["tasks"][task]; z = lambda name: Zr[f"{task}/{name}"].astype(float)  # noqa: E731
    T, cl = z("T"), Zr[f"{task}/cluster"]; C = boot_counts(cl, Jr["boot"]["B"], Jr["boot"]["seed"]).astype(float)
    if hashlib.sha256(np.ascontiguousarray(C.astype(np.int64)).tobytes()).hexdigest()[:16] != rr["boot_counts_sha"]:
        raise SystemExit(f"{task}: the regenerated resamples are not the run's (checksum)")
    Tm, KB, ranks = float(T.mean()), z("kyfan_mid_boot"), rr["ranks"]
    out = dict(task=task, T=Tm, n=len(T), label_pairs=rr["label_pairs"], labels=rr.get("labels"), D=rr["D"], ranks=ranks,
               names=names_of(rr), inst=twin(Jr, Zr, Jv, Zv, task, S0), k={})
    est, tops = {}, {}
    for j, k in enumerate(ranks):
        e = lambda name: z(f"k{k}/{name}/e")  # noqa: E731
        fm = lambda name: z(f"k{k}/{name}/fm")  # noqa: E731
        K = rr["ranks_out"][str(k)]["K_mid_T"] * Tm
        r1 = {}
        for s in SEEDS:
            for p in BUDGETS:
                x, f = e(f"das_ce_p{p}_s{s}"), fm(f"das_ce_p{p}_s{s}")
                d, rb = ratio(x, None, C, x.mean() / K, den_boot=KB[:, j], basic=True)
                d.update(above=d["lo"] > 1, p=p_above(d["pt"], rb), landing=float(f.mean() / K),
                         adequacy=float(x.mean() / f.mean()) if f.mean() != 0 else None)
                r1[f"s{s}_p{p}"] = d
        tops[k] = e("top_train"); room, _ = ratio(tops[k], T, C, tops[k].mean() / Tm)
        est[k] = {o: {p: np.mean([e(f"das_{o}_p{p}_s{s}") for s in SEEDS], axis=0) for p in BUDGETS} for o in OBJS}
        fce = np.mean([fm(f"das_ce_p1_s{s}") for s in SEEDS], axis=0)
        sub = rr["ranks_out"][str(k)]["subspaces"]
        rec = dict(K_T=K / Tm, ceiling_k=r1, room=dict(room, room=room["lo"] > 1, no_room=room["hi"] < 1),
                   excess_k={f"p{p}": ratio(est[k]["ce"][p], T, C, est[k]["ce"][p].mean() / Tm)[0] for p in BUDGETS},
                   objective_k={f"p{p}": diff_over_T(est[k]["ce"][p], est[k]["sq"][p], T, C) for p in BUDGETS},
                   medians=dict(e_ce_over_K_p1=float(est[k]["ce"][1].mean() / K), e_ce_over_K_p10=float(est[k]["ce"][10].mean() / K),
                           landing_ce=float(fce.mean() / K), e_eff_over_K_p10=float(est[k]["eff"][10].mean() / K),
                           e_sq_T_p1=float(est[k]["sq"][1].mean() / Tm), e_sq_T_p10=float(est[k]["sq"][10].mean() / Tm),
                           adequacy_kyfan=float(e("kyfan").mean() / fm("kyfan").mean()) if fm("kyfan").mean() != 0 else None,
                           adequacy_ce=float(est[k]["ce"][1].mean() / fce.mean()) if fce.mean() != 0 else None,
                           ce10_over_ce1=float(est[k]["ce"][10].mean() / est[k]["ce"][1].mean()),
                           overlap_kyfan_ce=sub["das_ce_p1_s0"]["overlap_kyfan"]))
        out["k"][str(k)] = rec
    if 1 in ranks:
        K1 = rr["ranks_out"]["1"]["K_mid_T"] * Tm; j1 = ranks.index(1)
        for j, k in enumerate(ranks):
            if k == 1:
                continue
            K = rr["ranks_out"][str(k)]["K_mid_T"] * Tm; g_pt = (K - K1) / Tm
            lo, hi = interval(g_pt, (KB[:, j] - KB[:, j1]) / (C @ T / C.sum(1)), basic=True)
            out["k"][str(k)]["one_question_ratios"] = dict(K_gain_T=dict(pt=g_pt, lo=float(lo), hi=float(hi)),
                                         top_ratio=ratio(tops[k], tops[1], C, tops[k].mean() / tops[1].mean())[0],
                                         **{f"ratio_{o}_p{p}": ratio(est[k][o][p], est[1][o][p], C, est[k][o][p].mean() / est[1][o][p].mean())[0]
                                            for o in ("ce", "eff") for p in BUDGETS})
            out["k"][str(k)]["gains"] = dict(top_gain=diff_over_T(tops[k], tops[1], T, C),
                                         **{f"das_gain_p{p}": diff_over_T(est[k]["ce"][p], est[1]["ce"][p], T, C) for p in BUDGETS})
    return out


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and np.std(rx) > 0 and np.std(ry) > 0 else None


def dist(rows_k, get, hi_get=None, lo_get=None):
    x = np.array([get(r, k) for r, k in rows_k], float)
    d = dict(tasks=len(x), median=float(np.median(x)), q1=float(np.quantile(x, 0.25)), q3=float(np.quantile(x, 0.75)))
    if hi_get:
        d.update(max_hi=float(max(hi_get(r, k) for r, k in rows_k)), above=int(sum(lo_get(r, k) > 1 for r, k in rows_k)),
                 below=int(sum(hi_get(r, k) < 1 for r, k in rows_k)))
    return d


def summarise(rows):
    """Counts over task rows (each with model, construction, one_q, the trace deviation trace_dev_mid and the per-rank statistics)."""
    S = dict(tasks=len(rows), constructions=len({r["construction"] for r in rows}), ceiling_k={}, excess_k={}, objective_k={}, one_question_ratios={}, gains={}, medians={}, across={})
    order = list(GRID) + ["q", "D-1"]
    names = [n for n in order if any(n == nm for r in rows for nm, _ in r["names"])]
    viol = []
    sel = {n: [(r, str(k)) for r in rows for nm, k in r["names"] if nm == n] for n in names}
    for name in names:
        s_ = sel[name]
        S["ceiling_k"][name] = dict(tests=sum(len(r["k"][k]["ceiling_k"]) for r, k in s_), above=sum(d["above"] for r, k in s_ for d in r["k"][k]["ceiling_k"].values()))
        viol += [dict(model=r["model"], task=r["task"], rank=name, k=int(k), fit=a, pt=d["pt"], lo=d["lo"], landing=d["landing"],
                      adequacy=d["adequacy"], trace_dev_mid=r["trace_dev_mid"]) for r, k in s_ for a, d in r["k"][k]["ceiling_k"].items() if d["above"]]
        room = [(r, k) for r, k in s_ if r["k"][k]["room"]["room"]]
        apart = [(r, k) for r, k in room if "1" in r["k"] and r["k"]["1"]["room"]["no_room"]] if name not in ("1", "D-1") else []
        S["excess_k"][name] = {f"p{p}": dict(room=len(room), above=sum(r["k"][k]["excess_k"][f"p{p}"]["lo"] > 1 for r, k in room),
                                      below=sum(r["k"][k]["excess_k"][f"p{p}"]["hi"] < 1 for r, k in room),
                                      constructions_room=len({r["construction"] for r, k in room}),
                                      constructions_above=len({r["construction"] for r, k in room}
                                                              - {r["construction"] for r, k in room if not r["k"][k]["excess_k"][f"p{p}"]["lo"] > 1}),
                                      **(dict(no_room_at_one=len(apart), no_room_at_one_above=sum(r["k"][k]["excess_k"][f"p{p}"]["lo"] > 1 for r, k in apart))
                                         if name not in ("1", "D-1") else {}))
                         for p in BUDGETS}
        S["objective_k"][name] = {f"p{p}": dict(room=len(room), above=sum(r["k"][k]["objective_k"][f"p{p}"]["lo"] > 0 for r, k in room)) for p in BUDGETS}
        oq = [(r, k) for r, k in s_ if r["one_q"] and "one_question_ratios" in r["k"][k]]
        if oq:
            S["one_question_ratios"][name] = dict(K_gain_T=dist(oq, lambda r, k: r["k"][k]["one_question_ratios"]["K_gain_T"]["pt"]),
                                 **{q_: dist(oq, lambda r, k, q_=q_: r["k"][k]["one_question_ratios"][q_]["pt"], lambda r, k, q_=q_: r["k"][k]["one_question_ratios"][q_]["hi"],
                                             lambda r, k, q_=q_: r["k"][k]["one_question_ratios"][q_]["lo"])
                                    for q_ in ["top_ratio"] + [f"ratio_{o}_p{p}" for o in ("ce", "eff") for p in BUDGETS]})
        mq = [(r, k) for r, k in s_ if r["label_pairs"] > 1 and "gains" in r["k"][k]]
        if mq:
            tg = np.array([r["k"][k]["gains"]["top_gain"]["pt"] for r, k in mq])
            S["gains"][name] = dict(tasks=len(mq), top_up=sum(r["k"][k]["gains"]["top_gain"]["lo"] > 0 for r, k in mq))
            for p in BUDGETS:
                dg = np.array([r["k"][k]["gains"][f"das_gain_p{p}"]["pt"] for r, k in mq])
                cons = sorted({r["construction"] for r, k in mq}); rng = np.random.default_rng(95); boot = []
                for _ in range(2000):
                    pick = [i for c in rng.choice(len(cons), len(cons)) for i, (r, k) in enumerate(mq) if r["construction"] == cons[c]]
                    v = spearman(tg[pick], dg[pick]) if len(pick) > 2 else None
                    if v is not None:
                        boot.append(v)
                S["gains"][name][f"p{p}"] = dict(das_up=sum(r["k"][k]["gains"][f"das_gain_p{p}"]["lo"] > 0 for r, k in mq),
                                              both_up=sum(r["k"][k]["gains"]["top_gain"]["lo"] > 0 and r["k"][k]["gains"][f"das_gain_p{p}"]["lo"] > 0
                                                          for r, k in mq),
                                              spearman=spearman(tg, dg), spearman_ci=list(np.quantile(boot, [0.025, 0.975])) if len(boot) > 100 else None)
        S["medians"][name] = {q_: float(np.median(v)) if (v := [r["k"][k]["medians"][q_] for r, k in s_ if r["k"][k]["medians"][q_] is not None]) else None
                         for q_ in ("e_ce_over_K_p1", "e_ce_over_K_p10", "landing_ce", "e_eff_over_K_p10", "e_sq_T_p1", "e_sq_T_p10",
                                    "adequacy_kyfan", "adequacy_ce", "ce10_over_ce1", "overlap_kyfan_ce")}
        if name != "1":                                                # across ranks: tasks with room at rank one and at rank k
            both = [(r, k) for r, k in s_ if "1" in r["k"] and r["k"]["1"]["room"]["room"] and r["k"][k]["room"]["room"]]
            S["across"][name] = {f"p{p}": dict(room_at_both=len(both),
                                              excess_k_holds_at_one_fails_at_k=sum(r["k"]["1"]["excess_k"][f"p{p}"]["lo"] > 1 and not r["k"][k]["excess_k"][f"p{p}"]["lo"] > 1 for r, k in both),
                                              excess_k_fails_at_one_holds_at_k=sum(not r["k"]["1"]["excess_k"][f"p{p}"]["lo"] > 1 and r["k"][k]["excess_k"][f"p{p}"]["lo"] > 1 for r, k in both),
                                              objective_k_holds_at_one_fails_at_k=sum(r["k"]["1"]["objective_k"][f"p{p}"]["lo"] > 0 and not r["k"][k]["objective_k"][f"p{p}"]["lo"] > 0 for r, k in both),
                                              objective_k_fails_at_one_holds_at_k=sum(not r["k"]["1"]["objective_k"][f"p{p}"]["lo"] > 0 and r["k"][k]["objective_k"][f"p{p}"]["lo"] > 0 for r, k in both))
                                 for p in BUDGETS}
    S["ceiling_k_violations"] = viol
    holds = lambda keep: dict(  # noqa: E731
        ceiling_k=not [v for v in viol if keep(v["rank"])],
        excess_k=all(v[f"p{p}"]["above"] == v[f"p{p}"]["room"] and v[f"p{p}"]["below"] == 0 for n, v in S["excess_k"].items() if keep(n) for p in BUDGETS),
        objective_k=all(v[f"p{p}"]["above"] == v[f"p{p}"]["room"] for n, v in S["objective_k"].items() if keep(n) for p in BUDGETS))
    S["holds"] = holds(lambda n: True); S["holds_beyond_one"] = holds(lambda n: n != "1")
    return S


def bh_count(rows):
    """ceiling_k under false-discovery control: each task one family (Bonferroni over its starts, budgets and ranks), BH over tasks."""
    fam = [min(1.0, min(d["p"] for k in r["k"] for d in r["k"][k]["ceiling_k"].values()) * sum(len(r["k"][k]["ceiling_k"]) for k in r["k"])) for r in rows]
    return int(bh(fam, Q).sum()) if fam else 0


data = load()
pop = {}
for m in MODELS:
    for r in SCORED["rows"].get(TAG(m), []):
        if r["main"] and r["defined"] and r["logit_cap"] == "capped":
            pop[(TAG(m), r["group"], r["task"])] = r
report, allrows, missing = dict(models={}, tiers={}), {}, []
for m in MODELS:
    t = TAG(m); rows = []
    for (mm, g, task), pr in sorted(pop.items()):
        if mm != t:
            continue
        d = data.get((t, g))
        if d is None or task not in d["r"][0]["tasks"] or d["r"][0]["tasks"][task].get("skipped"):
            missing.append(f"{t}/{g}/{task}"); continue
        s = task_stats(*d["r"], *d["v"], task, d["stack0"])
        if g in RAVEL_MAIN and not FAKE:                    # RAVEL: the two-way interval by each value's entity (experiments/lib/ravel_twoway.py)
            apply_twoway_rank(s, d["r"][1], task, test_rows(d["v"][0]["data"], task, ROOT))
        s["inst"]["gpu"] = d["gpu"]; s["inst"]["ok"] = s["inst"]["ok"] and d["gpu"] == [GPU, GPU]
        dev = (pr.get("trace") or {}).get("dev_mid")
        s.update(model=t, group=g, one_q=pr["one_q"], construction=pr["construction"], source=pr["source"],
                 trace_dev_mid=dev[0] if isinstance(dev, list) else dev)
        rows.append(s)
    allrows[t] = rows
    if not rows:
        report["models"][t] = dict(instruments="NO DATA"); continue
    bad = [r["task"] for r in rows if not r["inst"]["ok"]]
    inst = {q_: max(r["inst"][q_] for r in rows) for q_ in ("readout", "T_forward", "boot_check", "lam_check", "kyfan_first_order",
                                                            "above_kyfan", "twin_T", "twin_T_abs", "twin_lam", "twin_eigvectors",
                                                            "twin_eigvectors_abs", "twin_top", "twin_loss", "twin_fits",
                                                            "twin_fits_over_float32")}
    inst.update(top_start=min(r["inst"]["top_start"] for r in rows), twin_clusters=all(r["inst"]["twin_clusters"] for r in rows),
                gpus=sorted({g_ for r in rows for g_ in r["inst"]["gpu"] if g_}), fits_alone={r["task"]: r["inst"]["fits_alone"] for r in rows if r["inst"]["fits_alone"]})
    if bad:
        report["models"][t] = dict(instruments="FAILED", bad=bad, worst=inst); continue
    report["models"][t] = dict(instruments="OK", worst=inst, main=summarise(rows), ceiling_k_bh_tasks=bh_count(rows),
                               per_source={src: summarise([r for r in rows if r["source"] == src]) for src in sorted({r["source"] for r in rows})})
for tier, ms in (("small", SMALL), ("large", LARGE)):
    tags = [TAG(m) for m in ms]
    if len(tags) < 2 or not all(report["models"].get(x, {}).get("instruments") == "OK" for x in tags):
        report["tiers"][tier] = "not pooled: a model of the tier is not scored (as in the readout's scorer)"; continue
    common_ = set.intersection(*[{(r["group"], r["task"]) for r in allrows[x]} for x in tags])
    report["tiers"][tier] = dict(models=tags, main=summarise([r for x in tags for r in allrows[x] if (r["group"], r["task"]) in common_]))
complete = not missing and len(report["models"]) == len(MODELS)
out_path = Path(A["out"]) if "out" in A else (RES / "score_readout_rank.json" if FAKE else ROOT / "experiments/results/score_readout_rank.json")
json.dump(dict(complete=complete, missing=missing, models_scored=[TAG(m) for m in MODELS], report=report,
               rows={t: rows for t, rows in allrows.items()}), open(out_path, "w"), indent=1, default=float)
for t, v in report["models"].items():
    if v.get("instruments") != "OK":
        print(f"{t}: instruments {v.get('instruments')} {v.get('bad', [])[:5]}"); continue
    M = v["main"]
    print(f"{t}: {M['tasks']} tasks; ceiling holds {M['holds']['ceiling_k']} ({len(M['ceiling_k_violations'])} above, BH tasks {v['ceiling_k_bh_tasks']}); "
          f"excess holds {M['holds']['excess_k']}; objective holds {M['holds']['objective_k']}; beyond rank one {M['holds_beyond_one']}")
for tier, v in report["tiers"].items():
    print(f"tier {tier}: {v if isinstance(v, str) else v['main']['holds']}")
print(f"complete: {complete}; missing {len(missing)}" + (f" (first: {missing[:5]})" if missing else ""))

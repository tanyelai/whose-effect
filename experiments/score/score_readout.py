"""The readout's scorer: the tests of Section 5 and Appendix C.1 on every task and model, and Gemma-2 with and without its cap
(Appendix C.4), checked on synthetic outputs by experiments/tests/readout_scorer_test.py.

Inputs: experiments/results/readout/readout_<model>_m_<group>[__uncapped].{json,npz} (experiments/scripts/readout.py). Every output must be the
output of its unit in experiments/matrix/registry.py (registry.unverified checks the json; the npz must have the sha256 the run wrote
into the json). A missing or unverified output stops the scorer (exit 2) unless --models narrows it to models that are complete.
Intervals come from a cluster bootstrap over the test pairs, a row and its mirror forming one cluster. The resamples are
regenerated from the saved cluster ids and seed and checked against the run's checksum. Every ratio is a ratio of means on the
same resample. The run recomputes lambda_1 on each resample, and every ratio that involves it uses the basic (bias-correcting)
interval. RAVEL's pairs share entities across clusters, so every RAVEL row (every group named ravel_attributes*, with or without
the start-of-text token, held-out prompts included) that the mirror-cluster rule defines is read again with the two-way interval
by the entity of each value (experiments/lib/ravel_twoway.py; Appendix B.4): T's interval, the room, the excess and cross-entropy against
the squared error. A row whose two-way T is not above 0 leaves the counts.
A task enters the main population when it is one of MAIN's, not skipped, has at least 20 test clusters, and its T's interval
lies above 0. Two-label tasks (one_q): one label pair and a class sign equal to the label sign (or signed by label). The DAS
estimator of a task is the mean of its three starts' per-pair effects; the ceiling is tested on every start and budget. "Room":
the out-of-sample top (fitted on training pairs, measured on test pairs) lies above T. p-values: normal approximation with the
bootstrap standard error; Benjamini-Hochberg at q = 0.025 (one-sided, matching the two-sided 95% intervals) over a model's
tasks, for the excess over its tasks with room. For the ceiling each task is one family, Bonferroni over its three starts and
two budgets.
Instruments, per model: the readout reproduces the logits and T within the float32 rounding scale at the model's logit scale
(2 gamma_D S for a logit, 8 gamma_D S for an effect; experiments/lib/instrument_tol.py), and the bootstrap's lambda_1 at unit counts
equals the direct one (within 1e-6 of the spectrum's scale). A model whose instruments fail is not scored. Where another
direction's test effect exceeds the exact top, the run replaces the top by it, and the scorer counts these tasks (top_replaced).
Output: experiments/results/score_readout.json (every task's statistics and every count) and the printed report.
Usage: <venv python> -I experiments/score/score_readout.py [--models=small|large|small_large|all|<tag>,...] [--fake=<dir>] [--out=<path>]
       (small: the six small models; large: the four large ones; small_large: both; all, the default: every model measured,
       the six small and the four large models, the sizes within their families and Qwen2.5-32B)
       --fake reads synthetic outputs from <dir> without the registry (experiments/tests/readout_scorer_test.py only).
"""
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/matrix"))
from measure import bh, boot_counts, boot_ratio, interval, p_above  # noqa: E402
from instrument_tol import tol  # noqa: E402
from readout_groups import construction  # noqa: E402
from ravel_twoway import apply_twoway  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/score"))
from score_readout_sentence import test_rows  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = A.get("fake")
SEEDS, BUDGETS, Q = (0, 1, 2), (1, 10), 0.025
MAIN = {   # group -> tasks in the main population (None: every task of the group), each task once
    "causalgym": None, "number_only": None, "kumon": None, "sst": None, "ioi_wang": None, "ioi_wang_label_signed": None,
    "ravel_attributes": None, "sentiment_templates": None,   # every template
    "ioi_ours_label_signed": None, "ioi_ours": lambda t: t == "ioi_varied",
    # each model's tasks beyond the groups above, which keep only the tasks of all six small models and the RAVEL attributes that
    # meet the 10-entity minimums
    "ravel_attributes_model_specific": None, "sentiment_templates_model_specific": None}
SOURCE = {"causalgym": "CausalGym", "number_only": "CausalGym number-only", "kumon": "Kumon & Yanaka", "sst": "SST",
          "ioi_wang": "IOI (Wang et al.)", "ioi_wang_label_signed": "IOI (Wang et al.)", "ravel_attributes": "RAVEL",
          "sentiment_templates": "Tigges et al.", "ioi_ours_label_signed": "IOI (ours)", "ioi_ours": "IOI (ours)",
          "ravel_attributes_model_specific": "RAVEL", "sentiment_templates_model_specific": "Tigges et al."}

if FAKE:
    RES = Path(FAKE); SMALL, LARGE = json.load(open(RES / "models.json"))[:2]; SCALE = []; TAG = lambda m: m  # noqa: E731
else:
    import registry  # noqa: E402
    RES = ROOT / "experiments/results/readout"; SMALL, LARGE = registry.SMALL, registry.LARGE; TAG = registry.tag
    SCALE = registry.SCALE + registry.QWEN32                # the sizes within the families above (App. C.7) and Qwen2.5-32B, same rules
sel = A.get("models", "all")
# all: every model measured; small_large: the six small and the four large models; or model tags, comma-separated
SELECTORS = {"small": SMALL, "large": LARGE, "small_large": SMALL + LARGE, "all": SMALL + LARGE + SCALE}
if sel in SELECTORS:
    MODELS = SELECTORS[sel]
else:
    if unknown := [w for w in sel.split(",") if w not in [TAG(m) for m in SMALL + LARGE + SCALE]]:
        raise SystemExit(f"--models: unknown {', '.join(unknown)}; give one of {', '.join(SELECTORS)} or model tags "
                         f"({', '.join(TAG(m) for m in SMALL + LARGE + SCALE)})")
    MODELS = [m for m in SMALL + LARGE + SCALE if TAG(m) in sel.split(",")]


def main_task(group, task):
    return group in MAIN and (MAIN[group] is None or MAIN[group](task))


def class_is_label(data, task):
    """The base class fixes the base label one to one in the task's rows."""
    if FAKE or data == "causalgym":
        return True
    if data == "fg":
        return None
    rows = json.load(open(ROOT / data / task / "test.json")) + json.load(open(ROOT / data / task / "train.json"))
    a, b = {}, {}
    for r in rows:
        a.setdefault(r["base_type"], set()).add(r["base_label"]); b.setdefault(r["base_label"], set()).add(r["base_type"])
    return all(len(v) == 1 for v in a.values()) and all(len(v) == 1 for v in b.values())


def load():
    """{(model, group, suffix): (json, npz)} for every unit of the selected models, suffix "uncapped" for Gemma-2 without its cap
    and "" otherwise; stops on anything missing or unverified."""
    out, stop = {}, []
    if FAKE:
        for f in sorted(RES.glob("readout_*.json")):
            m, rest = f.stem[len("readout_"):].split("_m_", 1)
            g, _, suf = rest.partition("__")
            if m in [TAG(x) for x in MODELS]:
                J = json.load(open(f)); fz = f.with_suffix(".npz")
                if J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
                    stop.append(f"{fz.name}: not the npz its json names")
                out[(m, g, suf)] = (J, np.load(fz))
    else:
        kinds = ["readout"] + (["readout_uncapped"] if "google/gemma-2-9b" in MODELS else [])
        bad = registry.unverified(kinds, models=MODELS)
        stop += [f"{o}: {why}" for o, why in bad.items()]
        for u in registry.units(MODELS, kinds, device="cuda"):
            if u["output"] in bad:
                continue
            f = ROOT / u["output"]; fz = f.with_suffix(".npz"); J = json.load(open(f))
            if not fz.is_file() or J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
                stop.append(f"{fz.relative_to(ROOT)}: not the npz its json names"); continue
            out[(registry.tag(u["model"]), u["group"], "uncapped" if u["kind"] == "readout_uncapped" else "")] = (J, np.load(fz))
    if stop:
        print(f"STOP: {len(stop)} readout outputs are missing or not their unit's:\n  " + "\n  ".join(stop[:40]))
        sys.exit(2)
    return out


def R(num, den, C, point=None, num_boot=None, den_boot=None, basic=False):
    """A ratio of means: point value, interval, resampled values and its one-sided p-value for "above 1"."""
    r = boot_ratio(num, den, C, num_boot=num_boot, den_boot=den_boot)
    pt = point if point is not None else float(np.mean(num) / np.mean(den))
    lo, hi = interval(pt, r, basic=basic)
    return dict(pt=pt, lo=lo, hi=hi, r=r, p=p_above(pt, r))


def task_stats(J, Z, task, signed):
    k = lambda name: Z[f"{task}/{name}"]  # noqa: E731
    row = J["tasks"][task]; T = k("T"); n = len(T); cl = k("cluster")
    C = boot_counts(cl, J["boot"]["B"], J["boot"]["seed"])
    if hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16] != row["boot_counts_sha"]:
        raise SystemExit(f"{task}: the regenerated resamples are not the run's (checksum)")
    sizes = np.bincount(np.unique(cl, return_inverse=True)[1])
    lam_m, lam_b, ones = k("lam_mid_boot"), k("lam_base_boot"), np.ones(n)
    e = lambda d: k(f"e/{d}")  # noqa: E731
    fm = lambda d: k(f"fm/{d}")  # noqa: E731
    avg = lambda obj, p: np.mean([e(f"das_{obj}_p{p}_s{s}") for s in SEEDS], axis=0)  # noqa: E731
    Tr = R(T, ones, C)
    s = dict(task=task, n=n, clusters=int(len(sizes)), mirrored="all" if np.all(sizes == 2) else ("none" if np.all(sizes == 1) else "part"),
             T=float(T.mean()), T_lo=Tr["lo"], T_hi=Tr["hi"], label_pairs=row["label_pairs"], one_q=row["label_pairs"] == 1 and bool(signed),
             lam_T=row["lambda1_mid"] / row["T"], tr_mid_T=row["trM_mid"] / row["T"], tr_base_T=row["trM"] / row["T"],
             top_replaced=row["top_exact"]["replaced_by"], grad_share_mid=row.get("grad_share_mid"))
    if J.get("cap"):                                                 # the raw (pre-cap) label logits against the cap, at the base point
        ra = np.abs(k("raw_logits_b")).max(1) / J["cap"]; s["raw_over_cap"] = [float(np.median(ra)), float(np.quantile(ra, 0.9))]
    s["defined"] = s["clusters"] >= 20 and s["T_lo"] > 0
    if not s["defined"]:
        return s
    # the trace identity's error at the midpoint and at the base point
    dm = R(k("tr_mid") - T, T, C); db = R(k("tr_base") - T, T, C)
    simp = R((k("tr_base") + 2 * k("tr_mid")) / 3 - T, T, C)          # Simpson's rule on mirrored sets: fifth order
    diff = np.abs(dm["r"]) - np.abs(db["r"])
    s["trace"] = dict(dev_mid=[dm["pt"], dm["lo"], dm["hi"]], dev_base=[db["pt"], db["lo"], db["hi"]],
                   dev_simpson=[simp["pt"], simp["lo"], simp["hi"]],
                   simpson_closest=abs(simp["pt"]) < min(abs(dm["pt"]), abs(db["pt"])),
                   mid_closer=bool(np.quantile(diff, 0.975) < 0), base_closer=bool(np.quantile(diff, 0.025) > 0),
                   both_nonzero=(dm["lo"] > 0 or dm["hi"] < 0) and (db["lo"] > 0 or db["hi"] < 0),
                   opposite=bool(np.sign(dm["pt"]) == -np.sign(db["pt"])),
                   adequacy={d: float(e(d).mean() / fm(d).mean()) if abs(fm(d).mean()) > 0 else None
                             for d in ("b_hat_mid", "u_hat", "v1_mid", "das_ce_p1_s0", "sq_sur")})
    top_tr = R(e("top_train"), T, C); top_ex = R(e("top_exact"), T, C)
    s["top_train_T"] = [top_tr["pt"], top_tr["lo"], top_tr["hi"]]; s["top_exact_T"] = [top_ex["pt"], top_ex["lo"], top_ex["hi"]]
    s["room"], s["no_room"] = top_tr["lo"] > 1, top_tr["hi"] < 1
    # the ceiling: every start's exact effect against the first-order top lambda_1
    ceiling, ps = {}, []
    for p in BUDGETS:
        for sd in SEEDS:
            d = f"das_ce_p{p}_s{sd}"
            rm = R(e(d), None, C, point=float(e(d).mean() / row["lambda1_mid"]), den_boot=lam_m, basic=True)
            rb = R(e(d), None, C, point=float(e(d).mean() / row["lambda1"]), den_boot=lam_b, basic=True)
            ceiling[d] = dict(pt=rm["pt"], lo=rm["lo"], hi=rm["hi"], above=rm["lo"] > 1, above_base=rb["lo"] > 1,
                         bias=float(np.mean(rm["r"]) - rm["pt"]), landing=float(fm(d).mean() / row["lambda1_mid"]),
                         adequacy=float(e(d).mean() / fm(d).mean()))
            ps.append(rm["p"])
    s["ceiling"] = ceiling; s["ceiling_p"] = min(1.0, min(ps) * len(ps))
    # the excess over T: the mean over starts, and each start
    excess = {}
    for obj in ("ce", "eff"):
        for p in BUDGETS:
            r = R(avg(obj, p), T, C)
            per = [R(e(f"das_{obj}_p{p}_s{sd}"), T, C) for sd in SEEDS]
            excess[f"{obj}_p{p}"] = dict(pt=r["pt"], lo=r["lo"], hi=r["hi"], above=r["lo"] > 1, below=r["hi"] < 1, p=r["p"],
                                     seeds_above=sum(x["lo"] > 1 for x in per), seeds_pt=[x["pt"] for x in per])
    s["excess"] = excess
    s["share_of_top"] = {f"{obj}_p{p}": dict(of_top_train=float(avg(obj, p).mean() / e("top_train").mean()),
                                   of_top_exact=float(avg(obj, p).mean() / e("top_exact").mean())) for obj in ("ce", "eff") for p in BUDGETS}
    s["share_of_top"]["cos_v1_mid"] = {d: row["cos_v1_mid"][d] for d in row["cos_v1_mid"] if d.startswith(("das_ce_p", "das_eff_p"))}
    # the objective decides: cross-entropy against squared error, and where squared error lands
    objective = {}
    for p in BUDGETS:
        dce, dsq = avg("ce", p), avg("sq", p)
        r = R(dce - dsq, T, C)
        xs = boot_ratio(dsq, T, C); xu = boot_ratio(e("sq_sur"), T, C)
        dd = np.abs(xs - xu) - np.abs(xs - 1)
        sur = R(e("sq_sur"), T, C)
        objective[f"p{p}"] = dict(ce_minus_sq=[r["pt"], r["lo"], r["hi"]], ce_above_sq=r["lo"] > 0, sq_T=float(dsq.mean() / T.mean()),
                           sur_T=[sur["pt"], sur["lo"], sur["hi"]], sur_differs=sur["lo"] > 1 or sur["hi"] < 1,
                           sur_better=bool(np.quantile(dd, 0.975) < 0), sur_worse=bool(np.quantile(dd, 0.025) > 0),
                           jsd_T=float(avg("jsd", p).mean() / T.mean()))
    s["objective"] = objective
    # interchange accuracy (per start, then averaged over starts)
    g0 = k("g0"); own = (g0 + T > 0).astype(float)
    acc = lambda x: (g0 + x > 0).astype(float)  # noqa: E731
    accuracy = {"own": float(own.mean())}
    for obj in ("ce", "sq", "eff"):
        accuracy[obj] = float(np.mean([acc(e(f"das_{obj}_p1_s{sd}")).mean() for sd in SEEDS]))
    for d in ("u_hat", "b_hat", "top_train"):
        accuracy[d] = float(acc(e(d)).mean())
    dd = np.mean([acc(e(f"das_ce_p1_s{sd}")) for sd in SEEDS], axis=0) - own
    b = (C @ dd) / C.sum(1)
    accuracy["ce_minus_own"] = [float(dd.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]
    s["accuracy"] = accuracy
    if s["one_q"]:
        s["natural"] = dict(first_b_hat=float(fm("b_hat_mid").mean() / k("tr_mid").mean()),
                       first_u_hat=float(fm("u_hat").mean() / k("tr_mid").mean()),
                       exact_b_hat=[R(e("b_hat_mid"), T, C)[q] for q in ("pt", "lo", "hi")],
                       exact_u_hat=[R(e("u_hat"), T, C)[q] for q in ("pt", "lo", "hi")])
        s["angle"] = dict(P=row.get("one_angle_mid"), P_base=row.get("one_angle"), x={})
        if s["angle"]["P"] is not None:
            for name, (num, den, kw, basic) in dict(
                    lam_over_tr=(None, None, dict(num_boot=lam_m, den_boot=(C @ k("tr_mid")) / C.sum(1)), True),
                    top_train=(e("top_train"), T, {}, False), ce_p1=(avg("ce", 1), T, {}, False),
                    ce_p10=(avg("ce", 10), T, {}, False)).items():
                pt = row["lambda1_mid"] / row["trM_mid"] if num is None else float(num.mean() / den.mean())
                r = boot_ratio(num, den, C, **kw)
                lo, hi = interval(pt, r, basic=basic)
                s["angle"]["x"][name] = dict(pt=pt, lo=lo, hi=hi, r=2 * pt - r if basic else r)   # reflected: lambda_1's bias removed
    return s


def count(rows, key):
    return sum(1 for r in rows if key(r))


def angle_summary(Q_rows, B=2000, seed=6):
    """The angle's prediction P against the median of the target over the other constructions (the strongest naive constant; in a tier pool the same
    construction in every model is left out too). Strong form, across tasks: a task-level bootstrap of the median of
    |log P - log x| - |log c - log x| with its interval below 0, and Spearman's rank correlation of P and x with its task-level
    interval above 0. Per-task better/worse counts (a paired bootstrap per task) are reported only."""
    out = {}
    for t in ("lam_over_tr", "top_train", "ce_p1", "ce_p10"):
        R_ = [r for r in Q_rows if r["angle"]["x"].get(t) and r["angle"]["x"][t]["lo"] > 0]
        if len(R_) < 3:
            out[t] = dict(tasks=len(R_)); continue
        xs = np.array([r["angle"]["x"][t]["pt"] for r in R_]); P = np.array([r["angle"]["P"] for r in R_])
        cons = np.array([r["construction"] for r in R_])
        better = worse = 0; dpt = []
        for i, r in enumerate(R_):
            others = xs[cons != cons[i]]
            if len(others) == 0:
                continue
            c = float(np.median(others)); xb = np.clip(r["angle"]["x"][t]["r"], 1e-9, None)
            dd = np.abs(np.log(P[i]) - np.log(xb)) - np.abs(np.log(c) - np.log(xb))
            better += np.quantile(dd, 0.975) < 0; worse += np.quantile(dd, 0.025) > 0
            dpt.append(abs(np.log(P[i] / xs[i])) - abs(np.log(c / xs[i])))
        dpt = np.array(dpt); rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(dpt), size=(B, len(dpt))); med = np.median(dpt[idx], axis=1)
        rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0  # noqa: E731
        ti = rng.integers(0, len(xs), size=(B, len(xs))); rhos = np.array([sp(P[k], xs[k]) for k in ti])
        lo_m, hi_m = float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))
        lo_r, hi_r = float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))
        out[t] = dict(tasks=len(R_), median_diff=[float(np.median(dpt)), lo_m, hi_m], angle_better_overall=hi_m < 0,
                      spearman=[sp(P, xs), lo_r, hi_r], rank_correlated=lo_r > 0, strong_form=hi_m < 0 and lo_r > 0,
                      per_task_better=int(better), per_task_worse=int(worse),
                      quartiles_abs_log_err_angle=[float(np.quantile(np.abs(np.log(P / xs)), q)) for q in (0.25, 0.5, 0.75)])
    return out


def summarise(name, rows):
    D = [r for r in rows if r["defined"]]
    out = dict(population=name, tasks=len(rows), defined=len(D), constructions=len({r["construction"] for r in D}),
               not_defined=[r["task"] for r in rows if not r["defined"]])
    if not D:
        return out
    mir = [r for r in D if r["mirrored"] == "all"]; non = [r for r in D if r["mirrored"] != "all"]   # partly mirrored: non-mirrored
    both = [r for r in mir if r["trace"]["both_nonzero"]]
    x = np.array([r["trace"]["dev_base"][0] for r in mir]); y = np.array([r["trace"]["dev_mid"][0] for r in mir])
    slope = float(x @ y / (x @ x)) if len(mir) and x @ x > 0 else None
    out["trace"] = dict(mirrored=len(mir), opposite_where_both_nonzero=f"{count(both, lambda r: r['trace']['opposite'])}/{len(both)}",
                     slope_mid_on_base_mirrored=slope, non_mirrored=len(non),
                     mid_closer_non_mirrored=count(non, lambda r: r["trace"]["mid_closer"]),
                     base_closer_non_mirrored=count(non, lambda r: r["trace"]["base_closer"]),
                     simpson_closest_where_both_nonzero=f"{count(both, lambda r: r['trace']['simpson_closest'])}/{len(both)}",
                     median_abs_dev_simpson_mirrored=st.median(abs(r["trace"]["dev_simpson"][0]) for r in mir) if mir else None,
                     median_abs_dev_mid=st.median(abs(r["trace"]["dev_mid"][0]) for r in D),
                     median_abs_dev_base=st.median(abs(r["trace"]["dev_base"][0]) for r in D),
                     range_dev_mid=[min(r["trace"]["dev_mid"][0] for r in D), max(r["trace"]["dev_mid"][0] for r in D)])
    viol = [(r["task"], d, round(v["pt"], 3)) for r in D for d, v in r["ceiling"].items() if v["above"]]
    out["ceiling"] = dict(tasks=len(D), tests=6 * len(D), above_mid=len(viol), tasks_above=len({t for t, _, _ in viol}),
                     tasks_above_bh=int(bh([r["ceiling_p"] for r in D], Q).sum()), violations=viol[:50],
                     above_base=count([v for r in D for v in r["ceiling"].values()], lambda v: v["above_base"]),
                     max_pt=max(v["pt"] for r in D for v in r["ceiling"].values()),
                     median_bias=st.median(v["bias"] for r in D for v in r["ceiling"].values()),
                     s0_only=count(D, lambda r: r["ceiling"]["das_ce_p1_s0"]["above"]))
    Rm = [r for r in D if r["room"]]; Rn = [r for r in D if r["no_room"]]
    out["room"] = dict(room=len(Rm), no_room=len(Rn), neither=len(D) - len(Rm) - len(Rn),
                       one_q_no_room=[r["task"] for r in Rn if r["one_q"]], one_q=count(D, lambda r: r["one_q"]),
                       one_q_room=count(Rm, lambda r: r["one_q"]))
    out["excess"] = {}
    for key in ("ce_p1", "ce_p10", "eff_p1", "eff_p10"):
        out["excess"][key] = dict(above_in_room=count(Rm, lambda r: r["excess"][key]["above"]), room=len(Rm),
                              above_bh=int(bh([r["excess"][key]["p"] for r in Rm], Q).sum()) if Rm else 0,
                              below_in_room=[r["task"] for r in Rm if r["excess"][key]["below"]],
                              below_in_no_room=f"{count(Rn, lambda r: r['excess'][key]['below'])}/{len(Rn)}",
                              all_seeds_above_in_room=count(Rm, lambda r: r["excess"][key]["seeds_above"] == 3),
                              median=st.median(r["excess"][key]["pt"] for r in D),
                              constructions_all_above=len({r["construction"] for r in Rm} - {r["construction"] for r in Rm if not r["excess"][key]["above"]}),
                              constructions_room=len({r["construction"] for r in Rm}))
    out["share_of_top"] = {key: dict(of_top_train=st.median(r["share_of_top"][key]["of_top_train"] for r in D),
                           of_top_exact=st.median(r["share_of_top"][key]["of_top_exact"] for r in D)) for key in ("ce_p1", "ce_p10", "eff_p1", "eff_p10")}
    out["objective"] = {}
    for p in ("p1", "p10"):
        Sd = [r for r in D if r["objective"][p]["sur_differs"]]
        out["objective"][p] = dict(room=len(Rm), ce_above_sq=count(Rm, lambda r: r["objective"][p]["ce_above_sq"]),
                            sq_T=[st.median(r["objective"][p]["sq_T"] for r in D), min(r["objective"][p]["sq_T"] for r in D), max(r["objective"][p]["sq_T"] for r in D)],
                            sur_T=[st.median(r["objective"][p]["sur_T"][0] for r in D), min(r["objective"][p]["sur_T"][0] for r in D), max(r["objective"][p]["sur_T"][0] for r in D)],
                            sur_differs_from_T=len(Sd), sur_better=count(Sd, lambda r: r["objective"][p]["sur_better"]),
                            sur_worse=count(Sd, lambda r: r["objective"][p]["sur_worse"]),
                            jsd_T=[st.median(r["objective"][p]["jsd_T"] for r in D), min(r["objective"][p]["jsd_T"] for r in D), max(r["objective"][p]["jsd_T"] for r in D)])
    Qr = [r for r in D if r["one_q"] and r.get("angle", {}).get("P") is not None]
    out["natural"] = dict(one_q=len(Qr), first_b_hat=[min(r["natural"]["first_b_hat"] for r in Qr), max(r["natural"]["first_b_hat"] for r in Qr)] if Qr else None,
                     exact_b_hat_contains_1=count(Qr, lambda r: r["natural"]["exact_b_hat"][1] <= 1 <= r["natural"]["exact_b_hat"][2]),
                     exact_u_hat_contains_1=count(Qr, lambda r: r["natural"]["exact_u_hat"][1] <= 1 <= r["natural"]["exact_u_hat"][2]))
    out["angle"] = angle_summary(Qr)
    out["accuracy"] = dict(own=st.median(r["accuracy"]["own"] for r in D), ce=st.median(r["accuracy"]["ce"] for r in D), sq=st.median(r["accuracy"]["sq"] for r in D),
                     ce_above_own=count(D, lambda r: r["accuracy"]["ce_minus_own"][1] > 0),
                     ce_below_own=count(D, lambda r: r["accuracy"]["ce_minus_own"][2] < 0))
    for label, sub in (("T_ge_0.5", [r for r in D if r["T"] >= 0.5]),):
        sr = [r for r in sub if r["room"]]
        out[f"sensitivity_{label}"] = dict(tasks=len(sub), ceiling_tasks_above=count(sub, lambda r: any(v["above"] for v in r["ceiling"].values())),
                                           excess_ce_p1_above_in_room=f"{count(sr, lambda r: r['excess']['ce_p1']['above'])}/{len(sr)}")
    out["cap_measures"] = dict(dev_mid=st.median(abs(r["trace"]["dev_mid"][0]) for r in D), ceiling_share=out["ceiling"]["tasks_above"] / len(D),
                    angle_error=(out["angle"].get("lam_over_tr", {}).get("quartiles_abs_log_err_angle") or [None, None])[1],   # the median |log P - log x|
                    grad_share=st.median(r["grad_share_mid"] for r in D if r["grad_share_mid"] is not None) if any(r["grad_share_mid"] is not None for r in D) else None,
                    b_hat_adequacy=st.median(abs(r["trace"]["adequacy"]["b_hat_mid"] - 1) for r in D if r["trace"]["adequacy"]["b_hat_mid"] is not None),
                    raw_over_cap=st.median(r["raw_over_cap"][0] for r in D if r.get("raw_over_cap")) if any(r.get("raw_over_cap") for r in D) else None)
    return out


def instruments(J):
    bad = []; tl, te = tol(J.get("model", ""), "logit", fake=bool(FAKE)), tol(J.get("model", ""), "effect", fake=bool(FAKE))
    for t, r in J["tasks"].items():
        if r.get("skipped"):
            continue
        for sd in ("train", "test"):
            i = r["instruments"][sd]
            if i["readout_vs_logits"] > tl or i["T_vs_forward"] > te:
                bad.append(f"{t} {sd} readout {i['readout_vs_logits']:.1e} T {i['T_vs_forward']:.1e}")
        if r["boot_check"] > 1e-6:
            bad.append(f"{t} bootstrap lambda_1 {r['boot_check']:.1e}")
    return bad


data = load()
report, allrows = dict(models={}, tiers={}, apart={}, gemma_cap=None,
                       ravel_interval="two-way by each value's entity for room, V3 and V7 (experiments/lib/ravel_twoway.py); mirror beside"), {}
for m in [TAG(x) for x in MODELS]:
    units = {(g, suf): v for (mm, g, suf), v in data.items() if mm == m}
    if not units:
        continue
    bad = [b for (g, suf), (J, Z) in units.items() for b in instruments(J)]
    rows = []
    for (g, suf), (J, Z) in sorted(units.items()):
        for t, r in J["tasks"].items():
            if r.get("skipped"):
                continue
            s = task_stats(J, Z, t, g.endswith("label_signed") or class_is_label(J["data"], t))
            s.update(model=m, group=g, logit_cap=suf or "capped", main=main_task(g, t) and not suf,
                     source=SOURCE.get(g), construction=construction(g, t))
            if g.startswith("ravel_attributes") and s["defined"] and not FAKE:     # every RAVEL row: its pairs share entities
                apply_twoway(s, Z, t, test_rows(J["data"], t, ROOT))   # RAVEL's interval, two-way by the entity of each value
            rows.append(s)
    allrows[m] = rows
    if bad:
        report["models"][m] = dict(instruments="FAILED", bad=bad[:20]); print(f"{m}: instruments FAILED (not scored):", bad[:5]); continue
    main = [r for r in rows if r["main"]]
    report["models"][m] = dict(instruments="OK", top_replaced=count(rows, lambda r: r.get("top_replaced")), main=summarise(m, main),
                               per_source={src: summarise(src, [r for r in main if r["source"] == src]) for src in sorted({r["source"] for r in main})})
    report["apart"][m] = {g: summarise(g, [r for r in rows if r["group"] == g and not r["main"] and r["logit_cap"] == "capped"])
                          for g in sorted({r["group"] for r in rows if not r["main"] and r["logit_cap"] == "capped"})}
for tier, ms in (("small", SMALL), ("large", LARGE)):
    tags = [TAG(x) for x in ms if TAG(x) in report["models"] and report["models"][TAG(x)]["instruments"] == "OK"]
    if not ms or len(tags) < len(ms):                              # a tier is summarised only when every model is scored
        continue
    common_ = set.intersection(*[{(r["group"], r["task"]) for r in allrows[m] if r["main"] and r["defined"]} for m in tags])
    report["tiers"][tier] = summarise(tier, [r for m in tags for r in allrows[m] if r["main"] and (r["group"], r["task"]) in common_])
    report["tiers"][tier]["tasks_in_all"] = len(common_)
g = "gemma-2-9b"
if g in allrows and any(r["logit_cap"] == "uncapped" for r in allrows[g]):
    ug = {(r["group"], r["task"]) for r in allrows[g] if r["logit_cap"] == "uncapped"}
    pick = lambda rows, logit_cap: [r for r in rows if r["logit_cap"] == logit_cap and (r["group"], r["task"]) in ug]  # noqa: E731
    cap_summaries = {"gemma capped": summarise("gemma capped", pick(allrows[g], "capped")),
         "gemma uncapped": summarise("gemma uncapped", pick(allrows[g], "uncapped"))}
    others = {m: summarise(m, pick(allrows[m], "capped")).get("cap_measures") for m in ("Llama-3.1-8B", "Qwen2.5-7B", "pythia-6.9b") if m in allrows}
    diag = {}
    for key in ("dev_mid", "ceiling_share", "angle_error", "grad_share", "b_hat_adequacy"):
        vals = [v[key] for v in others.values() if v and v.get(key) is not None]
        cap, unc = cap_summaries["gemma capped"].get("cap_measures", {}).get(key), cap_summaries["gemma uncapped"].get("cap_measures", {}).get(key)
        if not vals or cap is None or unc is None:
            continue
        lo, hi = min(vals), max(vals); outside = not (lo <= cap <= hi)
        edge = lo if cap < lo else hi
        diag[key] = dict(others=[lo, hi], capped=cap, uncapped=unc, capped_outside=outside, uncapped_inside=lo <= unc <= hi,
                            gap_closed=(cap - unc) / (cap - edge) if outside and cap != edge else None)
    out_keys = [k for k, v in diag.items() if v["capped_outside"]]
    report["gemma_cap"] = dict(summaries=cap_summaries, others=others, diagnostics=diag,
                           raw_over_cap=cap_summaries["gemma capped"].get("cap_measures", {}).get("raw_over_cap"),
                           supported=bool(out_keys) and all(diag[k]["uncapped_inside"] for k in out_keys),
                           rule="supported when Gemma-2 without its cap lies inside the other three families' range on every diagnostic "
                                "where Gemma-2 with its cap lies outside it")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "r"}
    if isinstance(x, (np.bool_, np.integer)):
        return x.item()
    return x


out = dict(complete=not FAKE, models_scored=list(report["models"]), report=strip(report),
           rows={m: [strip(r) for r in rows] for m, rows in allrows.items()})
dest = (RES / "score_readout.json") if FAKE else (ROOT / "experiments/results/score_readout.json")
if A.get("out"):                                                     # --out writes the score file elsewhere
    dest = Path(A["out"]) if Path(A["out"]).is_absolute() else ROOT / A["out"]
json.dump(out, open(dest, "w"), indent=1, default=float)
for m, v in report["models"].items():
    if v["instruments"] != "OK":
        continue
    S = v["main"]; print(f"\n== {m}: main population {S['tasks']} tasks ({S.get('constructions')} constructions), {S['defined']} "
                         f"defined; exact top replaced {v['top_replaced']}")
    if S["defined"] == 0:
        continue
    trace = S["trace"]
    print(f"  trace: mirrored {trace['mirrored']}: opposite signs {trace['opposite_where_both_nonzero']}, slope mid on base "
          f"{trace['slope_mid_on_base_mirrored']} (leading order -0.5); not fully mirrored {trace['non_mirrored']}: base closer "
          f"{trace['base_closer_non_mirrored']} (strong form: none), midpoint closer {trace['mid_closer_non_mirrored']}; median |dev| mid {trace['median_abs_dev_mid']:.4f}, "
          f"base {trace['median_abs_dev_base']:.4f}")
    print(f"  ceiling: tasks with a start above lambda_1(mid) {S['ceiling']['tasks_above']}/{S['ceiling']['tasks']} (BH {S['ceiling']['tasks_above_bh']}), "
          f"starts {S['ceiling']['above_mid']}/{S['ceiling']['tests']}; above lambda_1(base) {S['ceiling']['above_base']}; max {S['ceiling']['max_pt']:.3f}; "
          f"median resampling bias {S['ceiling']['median_bias']:.4f}")
    print(f"  room (out-of-sample top above T): {S['room']}  ;  one-question tasks without room {S['room']['one_q_no_room']}")
    for key, x in S["excess"].items():
        print(f"  V3 {key}: above T in {x['above_in_room']}/{x['room']} with room (BH {x['above_bh']}; all three starts "
              f"{x['all_seeds_above_in_room']}; constructions {x['constructions_all_above']}/{x['constructions_room']}); below T with room "
              f"{x['below_in_room']}; below T without room {x['below_in_no_room']}; median {x['median']:.2f} T")
    print(f"  share of the top (out-of-sample / in-sample): " + ", ".join(f"{k} {x['of_top_train']:.3f}/{x['of_top_exact']:.3f}" for k, x in S["share_of_top"].items()))
    print(f"  natural directions, one-question {S['natural']['one_q']}: exact b-hat interval contains T {S['natural']['exact_b_hat_contains_1']}, u-hat "
          f"{S['natural']['exact_u_hat_contains_1']}; first-order b-hat/tr {S['natural']['first_b_hat']}")
    for t, x in S["angle"].items():
        print(f"  angle against the leave-one-out median, target {t}: {x}")
    for p, x in S["objective"].items():
        print(f"  V7 {p}: CE above sq in {x['ce_above_sq']}/{x['room']} with room; sq/T {x['sq_T']}; first-order optimum/T {x['sur_T']}; "
              f"where it differs from T ({x['sur_differs_from_T']}): optimum closer {x['sur_better']}, T closer {x['sur_worse']}; jsd/T {x['jsd_T']}")
    print(f"  accuracy (medians): own {S['accuracy']['own']:.2f}, ce {S['accuracy']['ce']:.2f}, sq {S['accuracy']['sq']:.2f}; ce above own "
          f"{S['accuracy']['ce_above_own']}, below {S['accuracy']['ce_below_own']}")
for tier, S in report["tiers"].items():
    if "ceiling" not in S:   # a tier whose models share no defined main task
        print(f"\n== tier {tier}: no task in every model"); continue
    print(f"\n== tier {tier}: {S['tasks_in_all']} tasks in every model; ceiling: tasks above {S['ceiling']['tasks_above']}/{S['ceiling']['tasks']}; "
          f"excess ce_p1 above in room {S['excess']['ce_p1']['above_in_room']}/{S['excess']['ce_p1']['room']}")
if report["gemma_cap"]:
    print("\n== Gemma-2's cap:", json.dumps(strip(report["gemma_cap"]["diagnostics"]), default=float), "supported:", report["gemma_cap"]["supported"])
print(f"\nwritten {dest.relative_to(ROOT) if not FAKE else dest}")

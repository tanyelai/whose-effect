"""The re-run of CausalGym's own code (Appendix E.2, Figure 13), scored with intervals.

CausalGym's own code (at its pinned commit) on the five published Pythia sizes up to 410m, as released
(experiments/causalgym/rerun/released/logs/released/) and with the two-line position fix (experiments/causalgym/rerun/fixed/logs/fixed/),
on the 19 tasks whose positions the released code reads shifted and two aligned controls (experiments/run/causalgym_rerun.sh). Scores
follow its plot.py summary: per (method, layer, position) the mean log odds ratio over the evaluation pairs; the last training
step for DAS (step -1 for the rest); the maximum over positions; the mean over layers. Intervals: the evaluation pairs
resampled by mirror cluster (CausalGym's evaluation batches hold two pairs and their swaps, items j and j + 2 of every four: a
pair and its mirror are one cluster, 50 per task, as in our own measurements; B = 2000, seed 95), the same resample for every
method and for both conditions of a task; 95% percentile intervals. Every task's pairs are checked to be the same, in the same
order, in both conditions, from the intervention-free fields (base_p_base, base_p_src). No threshold decides anything.
- The instrument (exact): on the aligned controls the deterministic methods give identical rows in both conditions (the fix
  changes only the read positions, which coincide there); a size that fails gets no readings.
- The fix's change of the mean difference (key fix_change, with fix_level): per task, the mean difference over vanilla with the fix
  minus as released, with its interval (raised, lowered or unchanged), and the level itself, the fixed mean difference over
  vanilla against 1, vanilla's own score (short of vanilla, reaching it, or above); counts per size.
- Replication (key replication): per task and method, the released run against the published value (two decimals,
  experiments/causalgym/causalgym_published_tables.json), in two forms. Rounding: the published value is reproduced when ours
  rounds to it (|ours - published| <= 0.005), the yardstick for the deterministic methods, computed on the same seeded
  evaluation pairs. Interval: the published value inside the released run's cluster interval (reported; it measures pair
  sampling, not the run-to-run variance that separates two trainings of DAS). Counts per method and size, with the median
  absolute difference; every method is reported.
- DAS with the fix (key das_with_fix): per task, DAS minus vanilla and DAS minus the best other method on the same resample, each
  interval's class; counts per size.
Output: experiments/results/score_causalgym_rerun.json and a printed report. Usage: python -I experiments/score/score_causalgym_rerun.py [--fake=<dir with released/ and fixed/>]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = A.get("fake")
REL = Path(FAKE) / "released" if FAKE else REPO / "experiments/causalgym/rerun/released/logs/released"
FIX = Path(FAKE) / "fixed" if FAKE else REPO / "experiments/causalgym/rerun/fixed/logs/fixed"
PUB = json.load(open(Path(FAKE) / "published.json" if FAKE else REPO / "experiments/causalgym/causalgym_published_tables.json"))
SHIFTED = ["agr_gender", "agr_sv_num_subj-relc", "agr_sv_num_obj-relc", "agr_sv_num_pp", "agr_refl_num_subj-relc",
           "agr_refl_num_obj-relc", "agr_refl_num_pp", "garden_mvrr", "garden_mvrr_mod", "garden_npz_obj", "garden_npz_obj_mod",
           "garden_npz_v-trans", "garden_npz_v-trans_mod", "gss_subord", "gss_subord_subj-relc", "gss_subord_obj-relc",
           "gss_subord_pp", "cleft", "cleft_mod"]
CONTROLS = ["npi_any_subj-relc", "filler_gap_obj"]
DETERMINISTIC = ["vanilla", "mean", "probe_0", "probe_1", "kmeans", "lda"]
PUBNAME = {"probe_0": "probe0", "probe_1": "probe1", "random": "rand"}
B, SEED = 2000, 95


def load(folder, model):
    """task -> {(method, layer, pos): per-pair log odds ratios at the summarised step}, and the deterministic methods' rows."""
    out, rows, base = {}, {}, {}
    for f in sorted(glob.glob(str(folder / "*.json"))):
        j = json.load(open(f))
        if j["metadata"].get("model", "").split("/")[-1] != model:
            continue
        task = j["metadata"]["dataset"].split("/")[-1]
        cell = defaultdict(list); steps = defaultdict(set)
        for r in j["data"]:
            cell[(r["method"], r["step"], r["layer"], r["pos"])].append(r["base_p_base"] - r["base_p_src"] + r["p_src"] - r["p_base"])
            steps[r["method"]].add(r["step"])
        last = max(s for v in steps.values() for s in v)
        out[task] = {(m, l, p): np.array(v) for (m, s, l, p), v in cell.items() if s == last or s == -1}
        rows[task] = {k: v for k, v in cell.items() if k[0] in DETERMINISTIC}
        bs = defaultdict(list)                                       # the intervention-free fields per site, in pair order
        for r in j["data"]:
            if r["method"] == "vanilla":
                bs[(r["layer"], r["pos"], r["step"])].append((r["base_p_base"], r["base_p_src"]))
        base[task] = dict(bs)
    return out, rows, base


def summarise(cells, C):
    """Per method: the score (max over positions of the mean, mean over layers) on the point (C None) or on each resample."""
    by = defaultdict(lambda: defaultdict(list))
    for (m, l, _), v in cells.items():
        by[m][l].append(v.mean() if C is None else C @ v / C.sum(1))
    return {m: np.mean([np.max(np.stack(ps), axis=0) for ps in layers.values()], axis=0) for m, layers in by.items()}


def same_pairs(a, b):
    """The two conditions read the same evaluation pairs in the same order: at every site (layer, position, step) the
    intervention-free base fields agree exactly, pair by pair (the same computation on the same inputs, so exact equality is the
    test). The order of the sites in a file does not matter (the fixed code writes one task's sites in another order; the scorer
    groups by site)."""
    return set(a) == set(b) and all(a[k] == b[k] for k in a)


def iv(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def cls(lo_hi):
    return "above" if lo_hi[0] > 0 else "below" if lo_hi[1] < 0 else "includes 0"


models = sorted({json.load(open(f))["metadata"]["model"].split("/")[-1] for f in glob.glob(str(REL / "*.json"))})
report = {}
for model in models:
    rel, rrows, rbase = load(REL, model); fix, frows, fbase = load(FIX, model)
    pub = PUB.get(f"{model}|main", {})
    same = [all(np.array_equal(np.array(rrows[t][k]), np.array(frows[t].get(k, [np.nan]))) for k in rrows[t])
            for t in CONTROLS if t in rrows and t in frows]
    inst = bool(same) and all(same)
    rec = dict(instrument_controls_identical=same, instrument=inst, tasks={})
    report[model] = rec
    print(f"\n== {model}: the controls identical row by row: {same} -> {'OK' if inst else 'FAILED: no readings for this size'}")
    if not inst:
        continue
    for t in [t for t in SHIFTED if t in rel and t in fix]:
        if not same_pairs(rbase[t], fbase[t]):
            rec.setdefault("pairs_differ", []).append(t); continue
        n = len(next(iter(rel[t].values()))); ncl = n // 2
        cl = np.array([(i // 4) * 2 + (i % 2) for i in range(n)])              # a pair and its mirror (items j, j + 2 of four)
        Cc = np.random.default_rng(SEED).multinomial(ncl, np.full(ncl, 1 / ncl), size=B).astype(float)
        C = Cc[:, cl]                                                        # each pair weighted by its cluster's draw
        pr, pf = summarise(rel[t], None), summarise(fix[t], None)
        br, bf = summarise(rel[t], C), summarise(fix[t], C)
        r1 = bf["mean"] / bf["vanilla"] - br["mean"] / br["vanilla"]; lvl = bf["mean"] / bf["vanilla"] - 1.0
        others = [m for m in bf if m not in ("das", "vanilla")]
        best_other = np.max(np.stack([bf[m] for m in others]), axis=0)
        rec["tasks"][t] = dict(
            released=pr, fixed=pf,
            fix_change=dict(point=float(pf["mean"] / pf["vanilla"] - pr["mean"] / pr["vanilla"]), interval=iv(r1), cls=cls(iv(r1)),
                    mean_over_vanilla=[float(pr["mean"] / pr["vanilla"]), float(pf["mean"] / pf["vanilla"])],
                    level_minus_1=iv(lvl), level=("above vanilla" if iv(lvl)[0] > 0 else "short of vanilla" if iv(lvl)[1] < 0
                                                  else "reaches vanilla")),
            replication={m: dict(published=pub.get(t, {}).get(PUBNAME.get(m, m)), ours=float(pr[m]), interval=iv(br[m]),
                        rounds_to=(pub.get(t, {}).get(PUBNAME.get(m, m)) is not None
                                   and abs(float(pr[m]) - pub[t][PUBNAME.get(m, m)]) <= 0.005 + 1e-12),
                        inside=(pub.get(t, {}).get(PUBNAME.get(m, m)) is not None
                                and iv(br[m])[0] <= pub[t][PUBNAME.get(m, m)] <= iv(br[m])[1])) for m in pr},
            das_with_fix=dict(das_minus_vanilla=iv(bf["das"] - bf["vanilla"]), das_minus_best_other=iv(bf["das"] - best_other),
                    best_other=max(others, key=lambda m: pf[m])))
        rec["tasks"][t]["das_with_fix"].update(cls_vanilla=cls(rec["tasks"][t]["das_with_fix"]["das_minus_vanilla"]),
                                     cls_best_other=cls(rec["tasks"][t]["das_with_fix"]["das_minus_best_other"]))
    T = rec["tasks"]
    if not T:
        continue
    rec["summary"] = dict(
        tasks=len(T),
        fix_change={c: sum(v["fix_change"]["cls"] == c for v in T.values()) for c in ("above", "below", "includes 0")},
        fix_level={c: sum(v["fix_change"]["level"] == c for v in T.values()) for c in ("short of vanilla", "reaches vanilla", "above vanilla")},
        replication={m: dict(rounds_to=sum(v["replication"][m]["rounds_to"] for v in T.values() if m in v["replication"]),
                    inside=sum(v["replication"][m]["inside"] for v in T.values() if m in v["replication"]),
                    with_published=sum(v["replication"][m]["published"] is not None for v in T.values() if m in v["replication"]),
                    median_abs_diff=float(np.median([abs(v["replication"][m]["ours"] - v["replication"][m]["published"]) for v in T.values()
                                                     if m in v["replication"] and v["replication"][m]["published"] is not None] or [np.nan])))
            for m in sorted({m for v in T.values() for m in v["replication"]})},
        das_with_fix=dict(above_vanilla=sum(v["das_with_fix"]["cls_vanilla"] == "above" for v in T.values()),
                above_best_other=sum(v["das_with_fix"]["cls_best_other"] == "above" for v in T.values())))
    s = rec["summary"]
    print(f"  shifted tasks {s['tasks']}/19 (pairs differ: {rec.get('pairs_differ', [])}); the fix's change (fix - released, mean/vanilla): {s['fix_change']}; "
          f"level with the fix: {s['fix_level']}")
    print(f"  replication: published value reproduced (rounds to it) / inside the cluster interval: " + ", ".join(
        f"{m} {v['rounds_to']}/{v['inside']}/{v['with_published']} (median |diff| {v['median_abs_diff']:.2f})" for m, v in s["replication"].items()))
    print(f"  DAS with the fix: DAS above vanilla {s['das_with_fix']['above_vanilla']}/{s['tasks']}, above the best other method "
          f"{s['das_with_fix']['above_best_other']}/{s['tasks']}")


def strip(x):
    if isinstance(x, dict):
        return {str(k): strip(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


dest = Path(FAKE) / "score_causalgym_rerun.json" if FAKE else REPO / "experiments/results/score_causalgym_rerun.json"
json.dump(strip(report), open(dest, "w"), indent=1)
print(f"\nwritten {dest}")

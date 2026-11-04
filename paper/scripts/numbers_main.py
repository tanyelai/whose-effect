"""Every number the paper's text quotes, as LaTeX macros (paper/sections/numbers_main.tex), read from the scorers' outputs so
that no count is typed by hand. Each macro is a plain number, a range or a short phrase; the text adds the words. The asserts
check the patterns the text states in words, so the script stops when an output no longer supports a sentence.

The readout (Section 5 and App. C.1): experiments/results/score_readout.json, population all, in every model whose instruments held.
One row per (model, task) of the main population that is defined: at least 20 test clusters and T's interval above 0 (the
population of Section 5's figure, paper/scripts/fig_readout.py). Tiers by experiments/matrix/registry.py: small (SMALL, six models,
124M-1B) and large (LARGE, four, 7-9B); the three SCALE models and Qwen2.5-32B (QWEN32) are in neither tier. Beside it: the
negative-polarity decomposition (experiments/results/readout_decomp/), the cross-entropy law (experiments/results/readout_ce_law/), the rank
sweep (experiments/results/readout_rank/), size within a family (score_scale.json), Gemma-2's cap (score_adequacy.json,
cap_mechanism.json), the sentence-level bootstrap (score_readout_sentence.json) and the quantities derived from the saved
directions (experiments/results/readout/derived_from_dirs.json).
Inside the network (Section 6 and App. D.1): the sets of maps small_all, small_causalgym, small_other, small_npi_fg_agr and
large_npi_fg_agr (experiments/results/inside_sets/), with the maps' per-pair files (experiments/results/inside/).
The benchmarks (Section 7): score_causalgym_rerun.json and score_sentiment_reread.json.
Also read: the task data (experiments/data/, experiments/tasks/ and the pinned CausalGym snapshot in the Hugging Face cache), the run
records (experiments/logs/, for App. B.5) and OwnEffect's reproduction (experiments/results/owneffect_repro/, for App. F).
Usage: python -I paper/scripts/numbers_main.py [--out=<file>]
"""
import hashlib
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(figdata.ROOT / "experiments/matrix"))
import registry  # noqa: E402

OUT = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), figdata.ROOT / "paper/sections/numbers_main.tex"))
S = json.load(open(figdata.ROOT / "experiments/results/score_readout.json"))
D = json.load(open(figdata.ROOT / "experiments/results/readout_decomp/score_readout_decomp.json"))
MODELS = S["report"]["models"]
# Section 5's population is every model the readout measured (PAPER): the six small, the four large, the three SCALE models
# and Qwen2.5-32B. The tiers keep their members (small: six, large: four)
PAPER = {Path(h).name for h in registry.SMALL + registry.LARGE + registry.SCALE + registry.QWEN32}
OK = [m for m, v in MODELS.items() if v.get("instruments") == "OK" and m in PAPER]
TIER = {Path(h).name: ("small" if h in registry.SMALL else "large") for h in registry.SMALL + registry.LARGE}
rows = [r for m in OK for r in S["rows"][m] if r["main"] and r["defined"]]
# first order's tightness is read over the models whose final logits are not capped (FO); a capped model (Gemma-2, whose rows
# carry raw_over_cap) gets its own macros (gm*), for Appendix C.4
CAPPED = [m for m in OK if any(r.get("raw_over_cap") for r in S["rows"][m])]
FO = [m for m in OK if m not in CAPPED]
rows_fo = [r for r in rows if r["model"] in FO]
npi1 = lambda r: r["group"] == "kumon" and r["task"].startswith("NPI")  # noqa: E731  (Kumon & Yanaka's one-directional NPI tasks)
macros = {}


def put(name, value):
    assert name.isalpha(), name
    macros[name] = value


def rng(xs, fmt="{:.2f}"):
    lo, hi = fmt.format(min(xs)), fmt.format(max(xs))
    return lo if lo == hi else f"{lo}--{hi}"


def pct(x):
    return f"{100 * x:.1f}"


# Section 5's setting: the models and the model-task cases
put("roModels", len(OK))
put("roTasks", len(rows))

# the trace test (the scorer's trace; Section 5.2; Proposition 9 in App. A.1): on
# mirrored test sets the midpoint trace's error is -1/2 of the base point's to leading order. The cases where both errors are
# nonzero and of opposite sign, the per-model slopes, off mirrored sets the cases where the base point is closer, and the
# midpoint trace's distance from T
v1 = [MODELS[m]["main"]["trace"] for m in FO]
put("roMirrored", sum(v["mirrored"] for v in v1))
opp = [tuple(map(int, v["opposite_where_both_nonzero"].split("/"))) for v in v1]
put("roOppositeNum", sum(a for a, _ in opp)); put("roOppositeDen", sum(b for _, b in opp))
slopes = [v["slope_mid_on_base_mirrored"] for v in v1]
put("roSlopeHigh", f"${max(slopes):.2f}$"); put("roSlopeLow", f"${min(slopes):.2f}$")   # the per-model range of the slopes
# the same tasks task by task: the midpoint trace's error over the base point's, where both are nonzero (by their intervals, the
# population of roOppositeNum/Den), its median and quartiles
_trr = np.array([r["trace"]["dev_mid"][0] / r["trace"]["dev_base"][0] for r in rows if r["model"] in FO and r["mirrored"] == "all"
                 and r["trace"]["both_nonzero"]])
assert len(_trr) == sum(b for _, b in opp)
put("roTraceRatioMed", f"${np.median(_trr):.2f}$")
put("roTraceRatioQone", f"${np.quantile(_trr, 0.25):.2f}$"); put("roTraceRatioQthree", f"${np.quantile(_trr, 0.75):.2f}$")
_tdv = [100 * v["median_abs_dev_mid"] for v in v1]; _f = lambda x: f"{x:.2f}" if x < 0.1 else f"{x:.1f}"  # noqa: E731
put("roTraceDevMid", f"{_f(min(_tdv))}--{_f(max(_tdv))}")                     # below 0.1%, two decimals: never printed as 0.0

# room and the excess (excess; Section 5.1): a case has room when the out-of-sample top's interval lies above T;
# cross-entropy DAS (and the mean-effect objective) lies above T when the interval of its three starts' mean effect, one pass,
# lies above T
room = [r for r in rows if r["room"]]
ce_above = [r for r in room if r["excess"]["ce_p1"]["above"]]
ce_miss = [r for r in room if not r["excess"]["ce_p1"]["above"]]
eff_above = [r for r in room if r["excess"]["eff_p1"]["above"]]
put("roRoom", len(room))
# the misses are the one-directional NPI constructions and one other task, Pythia-160m's sentiment classification
_miss_other = [r for r in ce_miss if not npi1(r)]
assert [(r["model"], r["task"]) for r in _miss_other] == [("pythia-160m", "sentiment_classification")], "the text: the misses"
put("roCeMissNpi", len(ce_miss) - len(_miss_other))
# the classification miss (Section 5.4 and App. C.6): its three starts' effects at one pass, where one start
# had not trained, and its effect after ten passes, above T and within 2% of the out-of-sample top
_cls = _miss_other[0]
put("roClsSeedsOne", ", ".join(f"{x:.2f}" for x in _cls["excess"]["ce_p1"]["seeds_pt"][:-1]) + f" and {_cls['excess']['ce_p1']['seeds_pt'][-1]:.2f}")
put("roClsTenT", f"{_cls['excess']['ce_p10']['pt']:.2f}")
assert _cls["excess"]["ce_p10"]["above"] and abs(_cls["excess"]["ce_p10"]["pt"] / _cls["top_train_T"][0] - 1) < 0.02, "the text: at ten passes, at the top"
# the one-directional NPI cases: how many, and what separates the misses from the rest, the room (the out-of-sample top over T)
# or the base label's probability (p_b at the midpoint, experiments/results/readout_ce_law/); two-sided rank-sum tests. All of them
# have room
from scipy.stats import mannwhitneyu  # noqa: E402
_CLp = json.load(open(figdata.ROOT / "experiments/results/readout_ce_law/score_readout_ce_law.json"))
_pb = {(m, x["group"], x["task"]): x["p_b"] for m, v in _CLp.items() if isinstance(v, dict) and "per_task" in v for x in v["per_task"]}
_npi = [r for r in room if npi1(r)]
assert len([r for r in rows if npi1(r)]) == len(_npi), "the text: every one-directional NPI case has room"
_nm, _nh = [r for r in _npi if not r["excess"]["ce_p1"]["above"]], [r for r in _npi if r["excess"]["ce_p1"]["above"]]
put("npiCases", len(_npi))
put("npiMissTop", f"{np.median([r['top_train_T'][0] for r in _nm]):.1f}"); put("npiHitTop", f"{np.median([r['top_train_T'][0] for r in _nh]):.1f}")
put("npiRoomP", f"{mannwhitneyu([r['top_train_T'][0] for r in _nm], [r['top_train_T'][0] for r in _nh]).pvalue:.3f}")
_pbm, _pbh = [_pb[(r["model"], r["group"], r["task"])] for r in _nm], [_pb[(r["model"], r["group"], r["task"])] for r in _nh]
put("npiPbP", f"{mannwhitneyu(_pbm, _pbh).pvalue:.2f}")
assert mannwhitneyu([r['top_train_T'][0] for r in _nm], [r['top_train_T'][0] for r in _nh]).pvalue < 0.05 < mannwhitneyu(_pbm, _pbh).pvalue, "the text: the room separates the misses, p_b does not"
put("roCeAbove", len(ce_above)); put("roCeMiss", len(ce_miss))
put("roEffAbove", len(eff_above))
# the distinct tasks behind the cases (Section 5's setting); cross-entropy DAS above T over every case, with room or not, so
# that this count does not lean on the room, whose top search also starts from an effect-seeking run; and among the cases with
# room after ten passes
put("roDistinctTasks", len({(r["group"], r["task"]) for r in rows}))
put("roCeAboveAll", sum(r["excess"]["ce_p1"]["above"] for r in rows))
put("roCeAboveTen", sum(r["excess"]["ce_p10"]["above"] for r in room))
# RAVEL's prompts of one attribute and SST's two scaffolds are separate tasks (App. C.1, "the measurement"). Counted once per
# construction, the first row of each (model, construction) is kept, by the construction the scorer also uses for the angle
# test's baseline (angle)
_RAVEL_MAIN = ("ravel_attributes", "ravel_attributes_model_specific")       # RAVEL's groups in the main population


def _construction(r):
    # the scorer's own field (experiments/lib/readout_groups.py: RAVEL's plain, JSON and value forms of an attribute are one construction,
    # and all of SST's tasks are one)
    return r["construction"]
_seen, _dup = set(), []
for r in rows:
    k = (r["model"], _construction(r))
    (_dup.append(r) if k in _seen else _seen.add(k))
put("roDupRavel", sum(r["group"] in _RAVEL_MAIN for r in _dup)); put("roDupSst", sum(r["group"] == "sst" for r in _dup))
_once = [r for r in rows if not any(r is d for d in _dup)]
_room_once = [r for r in _once if r["room"]]
put("roRoomConstr", len(_room_once)); put("roCeAboveConstr", sum(r["excess"]["ce_p1"]["above"] for r in _room_once))
put("roCeMissConstr", sum(not r["excess"]["ce_p1"]["above"] for r in _room_once))
# what the population rule leaves out (App. C.1, "the population"): the main population's tasks with data that the rule drops,
# by reason (fewer than 20 test clusters; else T's interval not above 0), and where cross-entropy DAS lies on the ones with too
# few clusters (read from the per-pair arrays; no interval is valid there)
excl = [(m, r) for m in OK for r in S["rows"][m] if r["main"] and not r["defined"]]
few = [(m, r) for m, r in excl if r["clusters"] < 20]; flat = [(m, r) for m, r in excl if r["clusters"] >= 20]
put("roExcl", len(excl)); put("roExclFew", len(few)); put("roExclFlat", len(flat))
# the text names the flat ones, where the input does not move the output the way the labels assume
assert sorted((m, r["task"]) for m, r in flat) == sorted([("Qwen2.5-0.5B", "filler_gap_embed_4"), ("pythia-160m", "filler_gap_embed_4"),
                                                          ("gemma-2-9b", "ioi_wang_20"), ("gemma-2-9b", "ioi_wang_1"),
                                                          ("Llama-3.2-3B", "ravel_city_latitude_json")]), "the text: the flat tasks"
# the text names the ones with too few clusters: sentiment_res in Pythia-160m, 410m, 1b and 6.9b, four sentiment templates in
# SmolLM2-360M, the book-review template in GPT-2 and Gemma-2, and four RAVEL attributes (in Pythia-1b, Pythia-6.9b and two in
# Llama-3.2-1B)
_few_named = sorted([(m, "sentiment_res") for m in ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-6.9b")]
                    + [("gpt2", "sentiment_book"), ("gemma-2-9b", "sentiment_book"),
                       ("pythia-1b", "ravel_occupation_work_location"), ("pythia-6.9b", "ravel_occupation_industry"),
                       ("Llama-3.2-1B", "ravel_city_longitude_json"), ("Llama-3.2-1B", "ravel_occupation_industry")])
assert sorted((m, r["task"]) for m, r in few if m != "SmolLM2-360M") == _few_named \
       and sum(m == "SmolLM2-360M" and r["task"].startswith("sentiment_") for m, r in few) == 4, "the text: which tasks have too few clusters"
put("roExclFewClusters", f"{min(r['clusters'] for _, r in few)}--{max(r['clusters'] for _, r in few)}" if few else "--")
_ratios = []
for m, r in few:
    Z = np.load(figdata.ROOT / f"experiments/results/readout/readout_{m}_m_{r['group']}.npz", allow_pickle=True)
    t = r["task"]; ce = np.mean([Z[f"{t}/e/das_ce_p1_s{x}"] for x in (0, 1, 2)], axis=0)
    _ratios.append(float(ce.mean() / Z[f"{t}/T"].mean()))
put("roExclFewCe", rng(_ratios, "{:.1f}") if _ratios else "--")
# Gemma-2 in the same paragraph: its IOI tasks' T without the start-of-text token (the main population) and with it (the
# _sot groups, reported apart), and how many of the population's RAVEL tasks are built in each (known.json's report)
if "gemma-2-9b" in OK:
    _cap = lambda r: r.get("logit_cap", "capped") == "capped"  # noqa: E731  (Gemma-2 as released, with its logit cap)
    _gi = [r for r in S["rows"]["gemma-2-9b"] if r["group"] in ("ioi_wang", "ioi_wang_label_signed") and r["main"] and _cap(r)]
    _gs = [r for r in S["rows"]["gemma-2-9b"] if r["group"] in ("ioi_wang_sot", "ioi_wang_sot_label_signed") and _cap(r)]
    put("gmIoiRawT", " and ".join(f"${r['T']:.1f}$" for r in sorted(_gi, key=lambda r: r["T"])))
    put("gmIoiSotT", rng([r["T"] for r in _gs], "{:.1f}"))
    _kn = lambda folder: json.load(open(figdata.ROOT / f"experiments/data/gemma-2-9b/{folder}/known.json"))["report"]
    # counted over the population's RAVEL tasks (the main rows of every model), not over RAVEL's whole list
    _rav = {r["task"] for m in OK for r in S["rows"][m] if r["group"] in _RAVEL_MAIN and r["main"]}
    put("gmRavelPop", len(_rav))
    for folder, key in (("ravel_attributes", "gmRavelRaw"), ("ravel_attributes_sot", "gmRavelSot")):
        put(key, sum(_kn(folder)[t]["written"] for t in _rav))
# the same counts after false-discovery control over each model's tasks (Benjamini-Hochberg, q = 0.025), given beside the
# per-task intervals' counts in App. C.1
put("roCeAboveBH", sum(MODELS[m]["main"]["excess"]["ce_p1"]["above_bh"] for m in OK))
put("roEffAboveBH", sum(MODELS[m]["main"]["excess"]["eff_p1"]["above_bh"] for m in OK))
put("gmCeilingTasksBH", MODELS["gemma-2-9b"]["main"]["ceiling"]["tasks_above_bh"] if "gemma-2-9b" in OK else 0)
# per-model medians of cross-entropy DAS over T and of its share of the out-of-sample top (share_of_top), over all models;
# the published sources (number-only agreement and our IOI frame are not counted as sources); the two-label cases
put("roCeMedian", rng([MODELS[m]["main"]["excess"]["ce_p1"]["median"] for m in OK]))
put("roSources", len({r["source"] for r in rows if r["source"] not in ("CausalGym number-only", "IOI (ours)")}))
oneq = [r for r in rows if r["one_q"]]

# what the tests can show (Section 5.2): the exact top against lambda_1 (per-model medians, uncapped
# models), and the smallest shared fraction of the readout gradients at the midpoint over the two-label cases
put("roTopExactOfLam", rng([float(np.median([r["top_exact_T"][0] / r["lam_T"] for r in rows if r["model"] == m and r.get("lam_T")])) for m in FO]))
oneq_fo = [r for r in oneq if r["model"] in FO]
# the text: no two-label task has an angle from training pairs of 90 degrees or more (its prediction would be undefined)
assert all(r["angle"]["P"] is not None for r in rows if r["one_q"]), "a one-question task without the one-angle prediction"
put("roGradShareMin", f"{min(r['grad_share_mid'] for r in oneq_fo if r.get('grad_share_mid') is not None):.4f}")
# how much of the change the readout reads (Section 5.3: most of the change is not read): the angle
# theta from training pairs over the two-label cases of the uncapped models, from the prediction P = (1 + 1/cos theta) / 2
_th = np.array([math.degrees(math.acos(min(1.0, 1.0 / (2 * r["angle"]["P"] - 1)))) for r in oneq_fo])
put("roThetaMed", f"{np.median(_th):.1f}"); put("roThetaQone", f"{np.quantile(_th, 0.25):.1f}"); put("roThetaQthree", f"{np.quantile(_th, 0.75):.1f}")
put("roModelsFO", len(FO)); put("roOneQFO", len(oneq_fo))
# the cases without room (the top's interval below T), and those where cross-entropy DAS lies below T
nr = [r for r in rows if r["no_room"]]
put("roNoRoomTasks", len(nr))

# where cross-entropy DAS lands, as a share of the out-of-sample top (share_of_top; per-model medians)
put("roCeOfTop", rng([MODELS[m]["main"]["share_of_top"]["ce_p1"]["of_top_train"] for m in OK]))
# Figure 1 and its rule (App. C.1, "the example in Figure 1"): the running example figdata.FIG1, picked by the rule
# written there and rechecked by fig1_check. The one-angle prediction and its rest, the untrained mean difference, cross-entropy
# DAS (three starts) and its test pairs above T (the readout's per-pair arrays), the squared error, and cross-entropy DAS's
# percentile among the two-label cases of the uncapped models
_F1 = figdata.FIG1; _F1C = figdata.fig1_check(S)
assert _F1C["n"] == macros["roOneQFO"], "Figure 1's percentile is among the one-question tasks of the uncapped models"
_r1 = next(r for r in S["rows"][_F1["model"]] if r["group"] == _F1["group"] and r["task"] == _F1["task"])
_z1 = np.load(figdata.ROOT / f"experiments/results/readout/readout_{_F1['model']}_m_{_F1['group']}.npz")
_e1 = np.mean([_z1[f"{_F1['task']}/e/das_ce_p1_s{x}"].astype(float) for x in (0, 1, 2)], axis=0)
assert abs(_e1.mean() / _z1[f"{_F1['task']}/T"].astype(float).mean() - _r1["excess"]["ce_p1"]["pt"]) < 1e-6, "Figure 1's DAS"
put("figOnePred", f"{_r1['angle']['P']:.2f}")
put("figOneCe", f"{_r1['excess']['ce_p1']['pt']:.2f}")
put("figOneSq", f"{_r1['objective']['p1']['sq_T']:.2f}"); put("figOneMeanDiff", f"{_r1['natural']['exact_u_hat'][0]:.2f}")
put("figOneAboveT", int((_e1 > _z1[f"{_F1['task']}/T"].astype(float)).sum()))
put("figOnePct", f"{_F1C['pct']:.0f}"); put("figOneMedian", f"{_F1C['median']:.2f}")
put("figOneAboveTText", f"all {_r1['n']}" if macros["figOneAboveT"] == _r1["n"] else f"{macros['figOneAboveT']} of the {_r1['n']}")
# how far below the predicted top the example's cross-entropy DAS lands, against the same ratio over the two-label cases of the
# uncapped models (the rule picks by DAS's effect, not by this gap, so App. C.1 says where the gap sits), and where it
# lands after ten passes
_one_fo = [r for m in FO for r in S["rows"][m] if r["main"] and r["defined"] and r["one_q"] and r.get("angle", {}).get("P")]
_rat = np.array([r["excess"]["ce_p1"]["pt"] / r["angle"]["P"] for r in _one_fo]); _r1r = _r1["excess"]["ce_p1"]["pt"] / _r1["angle"]["P"]
assert len(_rat) == _F1C["n"]
put("figOneRatio", f"{_r1r:.3f}"); put("figOneRatioPct", f"{100 * np.mean(_rat <= _r1r):.0f}")   # three decimals: it sits next
put("figOneRatioMed", f"{np.median(_rat):.3f}")                                                  # to the lower quartile
put("figOneRatioQone", f"{np.quantile(_rat, 0.25):.3f}"); put("figOneRatioQthree", f"{np.quantile(_rat, 0.75):.3f}")
put("figOneTen", f"{_r1['excess']['ce_p10']['pt']:.2f}"); put("figOneTopOos", f"{_r1['top_train_T'][0]:.2f}")
# panel (a) of Section 5's figure (fig:readout): where cross-entropy lands as a share of the out-of-sample top, median over the
# panel's cases; and the per-model medians of that share after ten passes
put("roCeOfTopTen", rng([MODELS[m]["main"]["share_of_top"]["ce_p10"]["of_top_train"] for m in OK]))

# where the squared error lands (objective; Section 5.1; Appendix C.2, "The squared error ends near T"): from the same
# starts, the per-pair squared error aimed at T. Its distance from T (median and upper quartile over the cases) and the cases
# with room where cross-entropy DAS lies above it
sq = [r["objective"]["p1"]["sq_T"] for r in rows]
_dev = [abs(x - 1) for x in sq]                                       # each case's distance from T, in T
put("roSqMedianDev", pct(st.median(_dev))); put("roSqDevQthree", pct(float(np.quantile(_dev, 0.75))))
put("roCeAboveSq", sum(MODELS[m]["main"]["objective"]["p1"]["ce_above_sq"] for m in OK))
# the landing test (objective; it holds when in no task the squared error's first-order optimum is significantly worse than T as a
# predictor of where it lands): the tasks per model where T is significantly closer, which make it fail
put("roSqTCloser", rng([MODELS[m]["main"]["objective"]["p1"]["sur_worse"] for m in OK], "{:d}"))
# both sides of that test where the first-order optimum and T differ as predictors: the tasks per model where the optimum is
# significantly closer, each side's total and Gemma-2's share of T's; how far the trained direction lies from that optimum
# (median |cos| over the cases, its three starts averaged; experiments/results/readout/derived_from_dirs.json)
put("roSqOptCloser", rng([MODELS[m]["main"]["objective"]["p1"]["sur_better"] for m in OK], "{:d}"))
put("roSqOptCloserSum", sum(MODELS[m]["main"]["objective"]["p1"]["sur_better"] for m in OK))
put("roSqTCloserSum", sum(MODELS[m]["main"]["objective"]["p1"]["sur_worse"] for m in OK))
put("roSqDiffer", sum(MODELS[m]["main"]["objective"]["p1"]["sur_differs_from_T"] for m in OK))
put("roSqTCloserGemma", MODELS["gemma-2-9b"]["main"]["objective"]["p1"]["sur_worse"])
_DFD = json.load(open(figdata.ROOT / "experiments/results/readout/derived_from_dirs.json"))["models"]
put("roSqOptCos", f"{np.median([np.mean(_DFD[r['model']][r['group']]['tasks'][r['task']]['sq_optimum_cos']['p1']) for r in rows]):.2f}")
# after ten passes: the squared error's distance from T (median and upper quartile), on the sentiment templates after one and ten
# passes, and CDAS's per-model medians; the training steps per pass on the sentiment templates and on every other source
_dev10 = [abs(r["objective"]["p10"]["sq_T"] - 1) for r in rows]
put("roSqMedianDevTen", pct(st.median(_dev10))); put("roSqDevQthreeTen", pct(float(np.quantile(_dev10, 0.75))))
_sent = [r for r in rows if r["group"] in ("sentiment_templates", "sentiment_templates_model_specific")]
put("roSqSentOne", pct(st.median(abs(r["objective"]["p1"]["sq_T"] - 1) for r in _sent)))
put("roSqSentTen", pct(st.median(abs(r["objective"]["p10"]["sq_T"] - 1) for r in _sent)))
put("roJsdMedianTen", rng([MODELS[m]["main"]["objective"]["p10"]["jsd_T"][0] for m in OK]))
_RJ = {}
def _steps(r):
    k = (r["model"], r["group"])
    if k not in _RJ:
        _RJ[k] = json.load(open(figdata.ROOT / f"experiments/results/readout/readout_{r['model']}_m_{r['group']}.json"))["tasks"]
    return _RJ[k][r["task"]]["steps_per_pass"]
put("roSentSteps", f"{st.median(_steps(r) for r in _sent):.0f}")
put("roOtherSteps", f"{st.median(_steps(r) for r in rows if r['group'] not in ('sentiment_templates', 'sentiment_templates_model_specific')):.0f}")
# Gemma-2's trace test (trace; App. C.2 quotes it); the asserts check where the trace test and the ten-pass mean-effect test fail
_of = lambda s_: s_.replace("/", " of ")  # noqa: E731
if "gemma-2-9b" in OK:
    put("gmTraceOpposite", _of(MODELS["gemma-2-9b"]["main"]["trace"]["opposite_where_both_nonzero"]))
_trace_main = {m: MODELS[m]["main"]["trace"] for m in OK}
assert [m for m, v in _trace_main.items() if v["opposite_where_both_nonzero"].split("/")[0] != v["opposite_where_both_nonzero"].split("/")[1]] == ["pythia-160m", "gemma-2-9b"] \
       and [m for m, v in _trace_main.items() if v["base_closer_non_mirrored"]] == ["Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B"] \
       and all(v["base_closer_non_mirrored"] == 1 for m, v in _trace_main.items() if m.startswith("Llama")), "the text: where the trace test fails (one task in each Llama)"
_eff10 = [m for m in OK if MODELS[m]["main"]["excess"]["eff_p10"]["above_in_room"] < MODELS[m]["main"]["excess"]["eff_p10"]["room"]]
assert _eff10 == ["pythia-160m"], "after ten passes the mean-effect objective lies above T in every task with room but in Pythia-160m"
# test pairs that also occur in training (App. C.1, "pairs shared by training and test"): every task directory of the
# per-model data (experiments/data/<model>/<group>/<task>/) and of the model-independent data (experiments/tasks/<source>/<task>/) is
# scanned; the overlap is the same in every model (asserted), and only the tasks the text names have any. CausalGym's
# pairs come from its released splits, which share none.
_pairkey = lambda r: json.dumps([r.get("base"), r.get("src")], sort_keys=True)   # noqa: E731
def _overlap_of(d):
    tr, te = d / "train.json", d / "test.json"
    if not (tr.is_file() and te.is_file()):
        return None
    seen = {_pairkey(r) for r in json.load(open(tr))}; test = json.load(open(te))
    return (sum(_pairkey(r) in seen for r in test), len(test))
_ov = {}
for _d in sorted((figdata.ROOT / "experiments/data").glob("*/*/*")) + sorted((figdata.ROOT / "experiments/tasks").glob("*/*")):
    _n = _overlap_of(_d) if _d.is_dir() else None
    if _n and _n[0]:
        _ov.setdefault(_d.name, set()).add(_n)
assert all(len(v) == 1 for v in _ov.values()), f"an overlap that differs across models: {_ov}"
_ov = {k: next(iter(v)) for k, v in _ov.items()}
_KU = ("NPIDeterminermono", "NPIOnlyScope", "NPISimple"); _NO = sorted(k for k in _ov if k.endswith("__num"))
assert set(_ov) == set(_KU) | {t + "_mirrored" for t in _KU} | set(_NO), f"the text names every overlapping task: {sorted(_ov)}"
assert all(_ov[t + "_mirrored"] == (2 * _ov[t][0], 2 * _ov[t][1]) for t in _KU), "the text: twice as many in the mirrored forms"
assert len({_ov[t][1] for t in _KU}) == 1 and len({_ov[t][1] for t in _NO}) == 1
_kd, _ko = _ov["NPIDeterminermono"], _ov["NPIOnlyScope"]
_shares = sorted(((n / N, t) for t, (n, N) in _ov.items() if not t.endswith("_mirrored")), reverse=True)
assert {t for _, t in _shares[:2]} == {"NPIDeterminermono", "NPIOnlyScope"}, "the text: the two rebuilt tasks share the most pairs"
assert sorted((n for t, (n, N) in _ov.items() if not t.endswith("_mirrored")), reverse=True)[:2] == sorted([_kd[0], _ko[0]], reverse=True)
put("kuOverlapDet", _kd[0]); put("kuOverlapOnly", _ko[0]); put("kuOverlapSimple", _ov["NPISimple"][0]); put("kuOverlapN", _kd[1])
put("noOverlapTasks", len(_NO)); put("noOverlapLo", min(_ov[t][0] for t in _NO)); put("noOverlapHi", max(_ov[t][0] for t in _NO))
put("noOverlapN", _ov[_NO[0]][1])
def _ce_class(m, task):
    r = next(r for r in S["rows"][m] if r["task"] == task and r.get("logit_cap") in (None, "capped"))
    v = r["excess"]["ce_p1"]; return "below" if v["below"] else "above" if v["above"] else "includes"
_kc = [(_ce_class(m, t_), _ce_class(m, t_ + "_disjoint")) for m in OK for t_ in ("NPIDeterminermono", "NPIOnlyScope")]
put("kuSame", sum(a == b for a, b in _kc)); put("kuCases", len(_kc))
# the cases whose classification changes: some move from including T to above it, and one, Qwen2.5-1.5B's determiner
# construction, from above T to including it; the text names it
_kcm = [(m, t_, _ce_class(m, t_), _ce_class(m, t_ + "_disjoint")) for m in OK for t_ in ("NPIDeterminermono", "NPIOnlyScope")]
put("kuUp", sum((a, b) == ("includes", "above") for _, _, a, b in _kcm))
assert [(m, t_) for m, t_, a, b in _kcm if a != b and (a, b) != ("includes", "above")] == [("Qwen2.5-1.5B", "NPIDeterminermono")] \
       and _ce_class("Qwen2.5-1.5B", "NPIDeterminermono") == "above", "the text: one case moves from above T to including it"
# the held-out sets at the readout (Section 5.1, "Held-out evaluation", and App. C.3): per set, its defined
# cases, those whose top lies above T (room) and, among them, those where cross-entropy DAS lies above T, and cross-entropy's
# median effect over T (every model, Gemma-2 with its cap)
def _heldout(groups, pairs=None, models=OK):
    rs = [r for m in models for r in S["rows"][m] if r["group"] in groups and r.get("logit_cap") in (None, "capped") and r["defined"]
          and (pairs is None or (r["label_pairs"] == 1) == (pairs == 1))]
    room = [r for r in rs if r["room"]]
    return dict(n=len(rs), room=len(room), above=sum(r["excess"]["ce_p1"]["above"] for r in room),
                ce=st.median(r["excess"]["ce_p1"]["pt"] for r in rs) if rs else None)
_hoI2, _hoI20 = _heldout(("ioi_wang_ood", "ioi_wang_ood_label_signed"), 1), _heldout(("ioi_wang_ood", "ioi_wang_ood_label_signed"), 2)
_hoR, _hoK = _heldout(("ravel_attributes_ood", "ravel_attributes_model_specific_ood")), _heldout(("kumon_ood",))
_hoD, _hoS = _heldout(("kumon_disjoint",)), _heldout(("sentiment_transfer", "sentiment_transfer_more"))
_hoGR = _heldout(("ravel_attributes_sot", "ravel_attributes_sot_model_specific"), models=["gemma-2-9b"])
_hoGI2 = _heldout(("ioi_wang_sot", "ioi_wang_sot_label_signed"), 1, models=["gemma-2-9b"])
_hoGI20 = _heldout(("ioi_wang_sot", "ioi_wang_sot_label_signed"), 2, models=["gemma-2-9b"])
assert _hoI2["room"] == _hoI2["n"] and _hoI20["room"] == 0 and _hoGI20["room"] == 0, "the text: two names keep room, twenty have none"
assert _hoGR["above"] == _hoGR["room"] == _hoGR["n"] - 1 and _hoGI2["n"] == _hoGI2["room"] == _hoGI2["above"] == 1, "the text: DAS lies above T in every Gemma-2 RAVEL task with room, and one task has none"
put("hoIoiN", _hoI2["n"])
put("hoRavelRoom", _hoR["room"]); put("hoRavelAbove", _hoR["above"])
put("hoKumonN", _hoK["n"]); put("hoKumonRoom", _hoK["room"]); put("hoKumonAbove", _hoK["above"]); put("hoKumonCe", f"{_hoK['ce']:.2f}")
put("hoDisjointRoom", _hoD["room"]); put("hoDisjointAbove", _hoD["above"])
put("hoSentN", _hoS["n"]); put("hoSentRoom", _hoS["room"]); put("hoSentAbove", _hoS["above"]); put("hoSentCe", f"{_hoS['ce']:.2f}")
put("hoGemmaRavel", _hoGR["n"]); put("hoGemmaRavelAbove", _hoGR["above"]); put("hoGemmaIoiCe", f"{_hoGI2['ce']:.2f}")
# CDAS's divergence over T, per-model medians (Section 5.1)
put("roJsdMedian", rng([MODELS[m]["main"]["objective"]["p1"]["jsd_T"][0] for m in OK]))

# the natural directions (natural; Section 5.3): on the two-label cases of the uncapped models, the
# exact effect over T of the readout gradient b-hat
bh = [r["natural"]["exact_b_hat"][0] for r in rows_fo if r["one_q"] and r["natural"].get("exact_b_hat")]
put("roBhatMedian", f"{st.median(bh):.3f}"); put("roBhatQone", f"{np.quantile(bh, 0.25):.3f}"); put("roBhatQthree", f"{np.quantile(bh, 0.75):.3f}")

# the angle (angle; from training pairs, before training) against the out-of-sample top and cross-entropy DAS, in the uncapped
# models: Spearman's rank correlations per model, and the angle's median error on the top
angle_main = [MODELS[m]["main"]["angle"] for m in FO]
put("roAngleCeSpearman", rng([v["ce_p1"]["spearman"][0] for v in angle_main if v.get("ce_p1")]))
put("roAngleTopErr", rng([100 * (math.exp(v["top_train"]["quartiles_abs_log_err_angle"][1]) - 1) for v in angle_main if v.get("top_train")], "{:.1f}"))
# the angle test's comparison for cross-entropy DAS (angle): the angle against the best constant (the median of the other
# constructions), per model; its strong form (the median log error's interval below the constant's) holds in every uncapped
# model
assert all(v["ce_p1"]["strong_form"] for v in angle_main), "the angle (ce) does not hold in every uncapped model"
# the angle test reads log x, so a two-label case whose cross-entropy effect's interval is not above zero leaves it (as it
# leaves panel (b) of Section 5's figure)
_angle_out = [r for r in oneq_fo if not (r["angle"]["x"].get("ce_p1") and r["angle"]["x"]["ce_p1"]["lo"] > 0)]
assert sum(v["ce_p1"]["tasks"] for v in angle_main) + len(_angle_out) == macros["roOneQFO"], "the angle's tasks are the one-question tasks of the uncapped models"

# readout quantities and counterexamples (Section 4.3, App. A.5). (a) The measured angle against the one of Proposition 7
# under one shared readout direction, on the test pairs of every two-label case of the uncapped models, from the saved
# midpoint factors A = W Q and B = Delta Q (Q orthonormal; A_i . B_i is each pair's midpoint trace term, checked): the
# class-signed mean gradient wbar = E[s w] and difference E[s Delta], against wbar and m = E[c Delta] with c_i = w_i . wbar / |wbar|^2,
# each pair's coefficient along the shared direction, so that the proposition weights each difference by its gradient's size.
# (b) Cross-entropy DAS (three starts) against the top eigenvector of the training pairs' midpoint M: per-model medians of |cos|
# over the two-label cases (Section 5.3). (c) The label pairs of IOI with twenty names, the only tasks
# without room (App. A.5). (d) CDAS's divergence over T, its range over the cases (App. A.5, "Objectives aimed at a target").
# The two angles per task come from experiments/results/readout/derived_from_dirs.json, which experiments/score/derive_from_dirs.py reduces
# from the saved directions (experiments/results/readout/dirs/)
DFD = json.load(open(figdata.ROOT / "experiments/results/readout/derived_from_dirs.json"))["models"]
_dang = [abs(DFD[r["model"]][r["group"]]["tasks"][r["task"]]["angle_signed_deg"] - DFD[r["model"]][r["group"]]["tasks"][r["task"]]["angle_weighted_deg"])
         for r in oneq_fo]
assert len(_dang) == macros["roOneQFO"]
put("roAngleWeightMedian", f"{np.median(_dang):.2f}"); put("roAngleWeightMax", f"{max(_dang):.2f}")
# the largest angle on test pairs signed by class alone (derive_from_dirs takes no sign toward the change), over the same cases;
# readout.py turns the readout direction toward the change, so its angle from training pairs lies below 90 degrees by its form
_sang = [DFD[r["model"]][r["group"]]["tasks"][r["task"]]["angle_signed_deg"] for r in oneq_fo]
assert max(_sang) < 90, "the text: signed by class alone, no two-label angle reaches 90 degrees"
put("roAngleSignedMax", f"{max(_sang):.1f}")
put("roCosVOne", rng([st.median(np.mean([abs(r["share_of_top"]["cos_v1_mid"][f"das_ce_p1_s{s_}"]) for s_ in (0, 1, 2)])
                                for r in oneq if r["model"] == m) for m in OK]))
_ioi20 = [r for r in rows if r["task"] in ("ioi_varied", "ioi_wang_20")]
assert {id(r) for r in _ioi20} == {id(r) for r in rows if r["no_room"]}    # the counterexamples: these, and only these, lack room
put("roIoiTwentyLp", rng(sorted({r["label_pairs"] for r in _ioi20}), "{:d}"))
_jsd = [r["objective"]["p1"]["jsd_T"] for r in rows]
put("roJsdMin", f"${min(_jsd):.2f}$"); put("roJsdMax", f"{max(_jsd):.2f}")

# Gemma-2's cap (Section 5.2; Appendix C.4): its first-order numbers with the cap and
# without it (the rows with arm "uncapped"), the scorer's cap_rank (experiments/score/score_adequacy.py: in its fully mirrored tasks, does
# nu, computed before training from the raw logits, rank the cap's change delta of log(top / lambda_1)?) and the exploratory
# mechanism (experiments/score/cap_mechanism.py)
if CAPPED:
    g = CAPPED[0]; grows = [r for r in rows if r["model"] == g]; others = [r for r in rows if r["model"] in FO]
    unc = {(r["group"], r["task"]): r for r in S["rows"][g] if r.get("logit_cap") == "uncapped" and r["defined"]}
    put("gmTasks", len(grows)); put("gmTopAboveLam", sum(r["top_exact_T"][1] > r["lam_T"] for r in grows))   # by interval
    put("gmTopAboveLamUncapped", sum(unc[(r["group"], r["task"])]["top_exact_T"][1] > unc[(r["group"], r["task"])]["lam_T"]
                                     for r in grows if (r["group"], r["task"]) in unc))
    # how far, not how often: in the uncapped models the exact top sits at lambda_1 or just below it, so a strict count of
    # "above" counts ties at the ceiling; the per-model medians of top / lambda_1 and its largest value say how far
    _orat = {m: [r["top_exact_T"][0] / r["lam_T"] for r in others if r["model"] == m] for m in sorted({r["model"] for r in others})}
    put("gmOtherTopOfLam", f"{min(np.median(v) for v in _orat.values()):.2f}--{max(np.median(v) for v in _orat.values()):.2f}")
    put("gmOtherTopOfLamMax", f"{max(max(v) for v in _orat.values()):.2f}")
    put("gmTopOfLam", f"{np.median([r['top_exact_T'][0] / r['lam_T'] for r in grows]):.2f}")
    put("gmAngleTopErr", f"{100 * (math.exp(MODELS[g]['main']['angle']['top_train']['quartiles_abs_log_err_angle'][1]) - 1):.0f}")
    put("gmCeilingTasks", MODELS[g]["main"]["ceiling"]["tasks_above"])
    AD = json.load(open(figdata.ROOT / "experiments/results/score_adequacy.json"))
    put("gmCapRank", f"{AD['cap_rank']['main_test']['pt']:.2f}"); put("gmCapRankN", AD["cap_rank"]["main_test"]["tasks"])
    assert AD["cap_rank"]["main_test"]["perm_p"] < 0.001, "the text: one-sided permutation p < 0.001"
    put("gmDeltaPos", AD["cap_rank"]["reports"]["delta_positive"]["mirrored"])
    CM = json.load(open(figdata.ROOT / "experiments/results/cap_mechanism.json"))
    put("gmDepth", f"{CM['median_depth']:.1f}"); put("gmBend", f"{CM['median_bend']:.3f}")
    put("gmCapShare", f"{CM['median_cap_share_measured']:.3f}"); put("gmCapShareRaw", f"{CM['median_cap_share_recorded_midpoint']:.3f}")
    put("gmCapShareStraight", f"{CM['median_cap_share_straight_midpoint']:.3f}")
    put("gmCapShareRho", f"{CM['spearman_rawlogit_vs_measured']:.3f}"); put("gmDepthRho", f"{CM['spearman_depth_delta']:.2f}")
    put("gmDeltaMedian", f"{CM['median_delta']:.2f}"); put("gmCGRho", f"{CM['per_source']['CausalGym']['spearman_nu_delta']:.2f}")
    put("gmCGN", CM["per_source"]["CausalGym"]["n"]); put("gmNonCGRho", f"{CM['spearman_nu_delta_non_causalgym']:.2f}")
    put("gmNonCGN", CM["n_non_causalgym"]); put("gmMove", f"{CM['median_move_per_label']:.2f}")
    # the two sources' rank correlations compared (Fisher's z, two-sided)
    from scipy.stats import norm  # noqa: E402
    _r1, _n1, _r2, _n2 = CM["per_source"]["CausalGym"]["spearman_nu_delta"], CM["per_source"]["CausalGym"]["n"], CM["spearman_nu_delta_non_causalgym"], CM["n_non_causalgym"]
    put("gmSourceP", f"{2 * norm.sf(abs(np.arctanh(_r1) - np.arctanh(_r2)) / np.sqrt(1 / (_n1 - 3) + 1 / (_n2 - 3))):.2f}")
    put("gmTanhExact", f"{100 * CM['tanh_midpoint_error_exact']:.1f}"); put("gmTanhLead", f"{100 * CM['tanh_midpoint_error_leading']:.1f}")
    put("gmCondMedian", f"{100 * CM['median_conditions']:.0f}")
    if CM.get("label_families"):
        put("gmFamilies", CM["label_families"]); put("gmFamilyRho", f"{CM['spearman_family_means']:.2f}")
        put("gmWithinFamilyP", f"{CM['within_family_permutation_p']:.2f}")
    # the cap diagnostic (the scorer's gemma_cap, over the tasks measured with and without the cap): five measures with the cap,
    # without it, and their range in the other three large models (Llama-3.1-8B, Qwen2.5-7B, Pythia-6.9b), and how much of the
    # gap removing the cap closes
    HC = S["report"]["gemma_cap"]["diagnostics"]
    _gc = S["report"]["gemma_cap"]["summaries"]["gemma capped"]["ceiling"]   # Table 5's second row as counts
    assert f"{_gc['tasks_above'] / _gc['tasks']:.2f}" == f"{HC['ceiling_share']['capped']:.2f}", "Table 5: the share is these counts"
    put("gmCeilAllAbove", _gc["tasks_above"]); put("gmCeilAllTasks", _gc["tasks"])
    for key, name, fmt_ in (("dev_mid", "Dev", "{:.3f}"), ("ceiling_share", "Ceiling", "{:.2f}"), ("angle_error", "Angle", "{:.3f}"),
                            ("grad_share", "Grad", "{:.5f}"), ("b_hat_adequacy", "Bhat", "{:.3f}")):
        d = HC[key]
        put(f"gm{name}Cap", fmt_.format(d["capped"])); put(f"gm{name}Unc", fmt_.format(d["uncapped"]))
        put(f"gm{name}Others", rng(d["others"], fmt_))
        if d.get("gap_closed") is not None:
            put(f"gm{name}Closed", f"{min(100 * d['gap_closed'], 100):.1f}" if d["gap_closed"] < 1 else "100")

# accuracy (accuracy, descriptive; Section 7.2): per-model interchange accuracy of cross-entropy
# DAS, of the network's own interchange and of the squared error, and the tasks where DAS's lies above or below the network's
accuracy_main = [MODELS[m]["main"]["accuracy"] for m in OK]
put("roAccCe", rng([v["ce"] for v in accuracy_main])); put("roAccOwn", rng([v["own"] for v in accuracy_main])); put("roAccSq", rng([v["sq"] for v in accuracy_main]))
# accuracy where the effect exceeds (Section 7.2): of the cases where cross-entropy
# DAS lies above T by its interval, those whose accuracy lies above (below) the network's own by the interval of accuracy's
# ce_minus_own (the three starts averaged), and those where both accuracies are 1 (every test pair right), where accuracy
# cannot show the excess
_ab = [r for r in rows if r["excess"]["ce_p1"]["above"]]
assert len(_ab) == macros["roCeAboveAll"]
put("roAccUpCe", sum(r["accuracy"]["ce_minus_own"][1] > 0 for r in _ab)); put("roAccDownCe", sum(r["accuracy"]["ce_minus_own"][2] < 0 for r in _ab))
put("roAccSatCe", sum(r["accuracy"]["own"] >= 1 - 1e-12 and r["accuracy"]["ce"] >= 1 - 1e-12 for r in _ab))   # every test pair right

# the number-only control (Section 5.1, "A confound"): CausalGym's six number-agreement tasks
# re-paired so that only the subject's number changes (group number_only, in the main population), against the same tasks'
# CausalGym pairs, per model: the median of cross-entropy DAS over T (three starts averaged); how many number-only tasks put it
# above T by their interval; and the lemma-only pairs (group number_only_lemma, outside the population): their whole-input
# effect over the CausalGym pairs' T, averaged over every model and task
_cap = lambda r: r.get("logit_cap", "capped") == "capped"  # noqa: E731
_no, _par, _lem = [], [], []
for m in OK:
    R_ = {(r["group"], r["task"]): r for r in S["rows"][m] if _cap(r)}
    no_ = [r for r in rows if r["model"] == m and r["group"] == "number_only"]
    par_ = [R_[("causalgym", r["task"].removesuffix("__num"))] for r in no_]
    assert len(no_) == 6 and all(p["main"] and p["defined"] for p in par_), m
    _no.append(st.median(r["excess"]["ce_p1"]["pt"] for r in no_)); _par.append(st.median(r["excess"]["ce_p1"]["pt"] for r in par_))
    _lem += [R_[("number_only_lemma", r["task"].replace("__num", "__lex"))]["T"] / p["T"] for r, p in zip(no_, par_)]
put("noCe", rng(_no)); put("noCeParent", rng(_par))
put("noAbove", sum(r["excess"]["ce_p1"]["above"] for r in rows if r["group"] == "number_only"))
put("noTasks", sum(r["group"] == "number_only" for r in rows)); put("noLemma", f"{np.mean(_lem):.2f}".replace("-0.00", "0.00"))

# the NPI decomposition (experiments/score/score_readout_decomp.py; Section 5.4, "Where cross-entropy does not exceed T", and
# App. C.6), per model over Kumon & Yanaka's NPI constructions: the share of the top's log-odds gain that
# comes from lowering the base label, cross-entropy DAS's share below it, the gap on mirrored sets, the base label's median
# probability, and whether the scorer's three predictions hold in every model
dm = [m for m, v in D.items() if isinstance(v, dict)]
put("npiModels", len(dm))
put("npiTopShare", rng([D[m]["top_share_one"] for m in dm])); put("npiCeMinusTop", rng([-D[m]["ce_minus_top_one"] for m in dm]))
put("npiGapMirrored", rng([abs(round(D[m]["gap_mirrored"], 2)) for m in dm])); put("npiPb", rng([D[m]["median_p_b_one"] for m in dm], "{:.3f}"))
# on the mirrored sets, the per-model medians over the constructions as Figure 11 draws them: every model's two shares (top and
# cross-entropy DAS) but Gemma-2-9B's, and Gemma-2-9B's two
mir = {(m, d): st.median(v["mirrored"][d] for v in D[m]["per_task"].values()) for m in dm for d in ("top", "ce")}
put("npiMirRest", rng([x for (m, _), x in mir.items() if m != "gemma-2-9b"]))
put("npiMirGemmaTop", f"{mir['gemma-2-9b', 'top']:.2f}"); put("npiMirGemmaCe", f"{mir['gemma-2-9b', 'ce']:.2f}")
# on the one-directional constructions, in log odds (task medians), Gemma-2 against the other large models: cross-entropy's
# gain, the network's own T, the top, and the top's gain from lowering the base label (Section 5.4)
lg = [m for m in dm if TIER.get(m) == "large"]
if "gemma-2-9b" in lg:
    oth = [m for m in lg if m != "gemma-2-9b"]
    def _npi_abs(m):
        rs = [r for r in S['rows'][m] if r['group'] == 'kumon' and r['task'].startswith('NPI') and r.get('logit_cap', 'capped') == 'capped'
              and r['main'] and r['defined']]
        top = [r['top_train_T'][0] * r['T'] for r in rs]
        return dict(T=st.median(r['T'] for r in rs), ce=st.median(r['excess']['ce_p1']['pt'] * r['T'] for r in rs), top=st.median(top),
                    low=st.median(D[m]['per_task'][r['task']]['one']['top'] * t for r, t in zip(rs, top)))
    G = _npi_abs('gemma-2-9b'); O = [_npi_abs(m) for m in oth]
    for key, name, fmt in (('T', 'T', '{:.1f}'), ('top', 'TopAbs', '{:.0f}'), ('low', 'TopLower', '{:.0f}')):
        put(f"npi{name}Gemma", fmt.format(G[key])); put(f"npi{name}OtherLarge", rng([o[key] for o in O], fmt))
    put("npiCeGainLarge", rng([G['ce']] + [o['ce'] for o in O], '{:.1f}'))

# convergence (descriptive; Appendix C.2, "cross-entropy ends near its own optimum"): |cos| between cross-entropy DAS's three
# random starts, pairwise, after one pass and after ten (derived_from_dirs.json, from the readout's saved directions)
cs = {1: [], 10: []}
for m in OK:
    for g, rec in sorted(DFD[m].items()):
        for t in sorted({r["task"] for r in rows if r["model"] == m and r["group"] == g}):
            for p_ in (1, 10):
                cs[p_] += rec["tasks"][t]["start_cos"][f"p{p_}"]
put("cvStartCosTen", f"{np.median(cs[10]):.3f}")
put("cvStartCosOne", f"{np.median(cs[1]):.3f}")

# the sentence-level bootstrap (a sensitivity analysis, experiments/results/score_readout_sentence.json; App. C.1, "intervals when
# sentences repeat"): every row that shares a base or a source sentence is resampled together, and every changed reading is
# counted
SB = figdata.ROOT / "experiments/results/score_readout_sentence.json"
sb = json.load(open(SB))["models"]
sm = [sb[m]["summary"] for m in OK if m in sb]
put("sbTasks", sum(x["rebuilt"] for x in sm)); put("sbDefined", sum(x["defined"] for x in sm))
put("sbSqChanged", sum(len(x["objective"]["p1"]["sur_changed"]) for x in sm))
n3 = macros["sbSqChanged"]
# the direction of each change: none is lost (above under the main interval, not under sentence clusters); the ones gained
# are RAVEL tasks, whose main interval, two-way by entity, is the stricter one
_lost = [t for x in sm for t in x["excess"]["ce_p1"]["lost"] + x["objective"]["p1"]["lost"]]
_gain = [t for x in sm for t in x["excess"]["ce_p1"]["gained"] + x["objective"]["p1"]["gained"]]
assert not _lost and all(g.startswith("ravel_attributes") for g, _ in _gain), "the text: none lost, the gains are RAVEL's"
put("sbCeGained", sum(len(x["excess"]["ce_p1"]["gained"]) for x in sm)); put("sbCeSqGained", sum(len(x["objective"]["p1"]["gained"]) for x in sm))
# the text names every task whose squared-error landing classification changes, not only their count
_nm = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "pythia-1b": "Pythia-1b",
       "gemma-2-9b": "Gemma-2-9B", "Qwen2.5-7B": "Qwen2.5-7B", "Qwen2.5-0.5B": "Qwen2.5-0.5B", "SmolLM2-360M": "SmolLM2-360M",
       "Llama-3.1-8B": "Llama-3.1-8B", "pythia-6.9b": "Pythia-6.9b", "Qwen2.5-1.5B": "Qwen2.5-1.5B",
       "Llama-3.2-1B": "Llama-3.2-1B", "Llama-3.2-3B": "Llama-3.2-3B", "Qwen2.5-32B": "Qwen2.5-32B"}
_ch = "; ".join(f"{_nm.get(m, m)}: " + ", ".join(r"\texttt{" + task.replace("_", r"\_") + "}" for _g, task in sb[m]["summary"]["objective"]["p1"]["sur_changed"])
                for m in OK if m in sb and sb[m]["summary"]["objective"]["p1"]["sur_changed"])
put("sbSqChangedText", "no classification of where the squared error ends up changes" if n3 == 0 else f"{n3} classifications of where the squared error ends up change ({_ch})")
# cross-entropy's own optimum (experiments/results/readout_ce_law/, the scorer's sensitivity entry; Appendix C.2, "cross-entropy ends near its
# own optimum"): per model, the median |cos| between cross-entropy DAS after ten passes and cross-entropy's own best direction
# (searched from the top eigenvector of the training pairs' midpoint M, f_top from_v1_mid), and how much more effect it has
CL = figdata.ROOT / "experiments/results/readout_ce_law/score_readout_ce_law.json"
cl = json.load(open(CL))
cs_ = [cl[m]["sensitivity"]["f_top"]["from_v1_mid"]["median_cos_ce10"] for m in cl if m not in ("tiers",) and isinstance(cl[m], dict) and "sensitivity" in cl[m]]
if cs_:
    put("cvOptCos", rng(cs_, "{:.3f}"))
    f4 = {m: cl[m]["sensitivity"]["f_top"]["from_v1_mid"] for m in cl if m not in ("tiers",) and isinstance(cl[m], dict) and "sensitivity" in cl[m]}
    put("cvOptGain", rng([100 * (v["median_e_over_ce10"] - 1) for v in f4.values()], "{:.1f}"))
    lgm = [m for m in f4 if TIER.get(m) == "large" or m == "Qwen2.5-32B"]       # the 7-32B models
    if lgm and "p90_e_over_ce10" in f4[lgm[0]]:
        # the tail: the 90th percentile of the gain over every case, whatever its |cos|
        put("cvOptGainNinetyLarge", rng([100 * (f4[m]["p90_e_over_ce10"] - 1) for m in lgm], "{:.0f}"))
        put("cvOptMaxLarge", f"{100 * (max(f4[m]['max_e_over_ce10'] for m in lgm) - 1):.0f}")

# size within a family (experiments/results/score_scale.json; Section 5.4, "Model size", and App. C.7). The score's room_grows
# and excess_grows: does the larger size leave more room (rho = out-of-sample top / T) and report a larger excess (r = cross-entropy
# DAS / T)? Its share_falls: does cross-entropy take a smaller share of the room (sigma = r / rho)? D is the median
# over the family's common tasks of the per-task log ratio; the growth from the smallest to the largest size is written as the
# percentage exp(D) - 1
SS = figdata.ROOT / "experiments/results/score_scale.json"
ss = json.load(open(SS))["families"]
pc = lambda d: f"{100 * (math.exp(d['pt']) - 1):.0f}"  # noqa: E731
for fam, key in (("Qwen2.5", "Qwen"), ("Llama", "Llama"), ("Pythia", "Pythia")):
    F = ss[fam]; s1, _, s3 = F["sizes"]
    put(f"ss{key}Rho", pc(F["D"][f"rho:{s1}->{s3}"])); put(f"ss{key}R", pc(F["D"][f"r:{s1}->{s3}"]))
    put(f"ss{key}Sigma", f"{100 * (1 - math.exp(F['D'][f'sigma:{s1}->{s3}']['pt'])):.0f}")
    put(f"ss{key}SigmaFalls", F["falls_with_size_sigma"]); put(f"ss{key}Tasks", F["common_tasks"])
# the text: all three hold in Qwen2.5 and Pythia; in Llama the room and the share hold and the excess grows from the smallest
# to the largest size, its step from 1B to 3B including zero
assert all(ss[f][k] for f in ("Qwen2.5", "Pythia") for k in ("room_grows", "excess_grows", "share_falls")), "the text: all three in Qwen2.5 and Pythia"
_L = ss["Llama"]; _st = _L["D"][f"r:{_L['sizes'][0]}->{_L['sizes'][1]}"]
assert _L["room_grows"] and _L["share_falls"] and not _L["excess_grows"] and _st["lo"] <= 0 < _st["pt"], "the text: Llama's excess step 1B-3B"
assert _L["D"][f"r:{_L['sizes'][0]}->{_L['sizes'][2]}"]["lo"] > 0, "the text: Llama's excess grows from the smallest to the largest"
# Section 5: its ranges run from Pythia to Llama; in Qwen2.5 and Llama most of the growth comes at the step to the largest
# size, while Pythia grows at about the same rate per doubling of its parameters at both steps (counts from Biderman et al.,
# 2023, Table 1; within a sixth of each other)
_d = lambda f, k, a, b: ss[f]["D"][f"{k}:{ss[f]['sizes'][a]}->{ss[f]['sizes'][b]}"]["pt"]  # noqa: E731
_pn = (405_334_016, 1_011_781_632, 6_857_302_016)
for k in ("rho", "r"):
    _g = {f: _d(f, k, 0, 2) for f in ("Pythia", "Qwen2.5", "Llama")}
    assert min(_g, key=_g.get) == "Pythia" and max(_g, key=_g.get) == "Llama", f"the text: the {k} range runs Pythia-Llama"
    assert all(_d(f, k, 1, 2) > _d(f, k, 0, 1) for f in ("Pythia", "Qwen2.5", "Llama")), f"the text: {k} grows most at the largest size"
    _a, _b = (_d("Pythia", k, i, i + 1) / math.log2(_pn[i + 1] / _pn[i]) for i in (0, 1))
    assert abs(_b / _a - 1) < 1 / 6, f"the text: Pythia's {k} grows at about the same rate per doubling ({_a:.3f}, {_b:.3f})"
pc1 = lambda x: f"${100 * (math.exp(x) - 1):.1f}$"   # noqa: E731
put("ssLlamaRStep", pc1(_st["pt"])); put("ssLlamaRStepLo", pc1(_st["lo"])); put("ssLlamaRStepHi", pc1(_st["hi"]))
# Qwen2.5-32B (App. C.7): the same three tests on the step from 7B to 32B (room_grows, excess_grows,
# share_falls); the room's, the excess's and cross-entropy's share's change, median per-task percentages with their intervals
_q = json.load(open(SS))["qwen32"]; _s3, _s4 = "Qwen2.5-7B", "Qwen2.5-32B"
for key, nm in (("rho", "Rho"), ("r", "R")):
    d_ = _q["D"][f"{key}:{_s3}->{_s4}"]
    put(f"ssThirty{nm}", pc(d_)); put(f"ssThirty{nm}Lo", f"{100 * (math.exp(d_['lo']) - 1):.0f}"); put(f"ssThirty{nm}Hi", f"{100 * (math.exp(d_['hi']) - 1):.0f}")
_d = _q["D"][f"sigma:{_s3}->{_s4}"]
put("ssThirtySigma", f"{100 * (1 - math.exp(_d['pt'])):.0f}"); put("ssThirtyTasks", _q["common_tasks"])
put("ssThirtySigmaLo", f"{100 * (1 - math.exp(_d['hi'])):.1f}"); put("ssThirtySigmaHi", f"{100 * (1 - math.exp(_d['lo'])):.1f}")
assert _q["room_grows"] and _q["excess_grows"] and _q["share_falls"], "the text: all three hold from 7B to 32B"
# the text (Section 5.4 and App. C.7): the untrained top of the range, lambda1/T (median per size over the common tasks), grows at every step,
# in each family and from Qwen2.5-7B to 32B
for _f, _v in json.load(open(SS))["families"].items():
    _l = [_v["reports"][m]["median_lambda1_over_T"] for m in _v["sizes"]]
    assert all(b_ > a_ for a_, b_ in zip(_l, _l[1:])), f"the text: lambda1/T grows at every step in {_f} ({_l})"
# Appendix C.7 prints these medians, so the text's claim has its numbers
for _f, _key in (("Pythia", "Pythia"), ("Qwen2.5", "Qwen"), ("Llama", "Llama")):
    _l = [ss[_f]["reports"][m]["median_lambda1_over_T"] for m in ss[_f]["sizes"]]
    put(f"ssLam{_key}", f"{_l[0]:.2f}, {_l[1]:.2f} and {_l[2]:.2f}")
_l7, _l32 = (_q["reports"][m]["median_lambda1_over_T"] for m in (_s3, _s4))
assert _l32 > _l7, "the text: lambda1/T grows from Qwen2.5-7B to 32B"
put("ssLamThirty", f"{_l32:.2f}")
_l = [_q["reports"][m]["median_lambda1_over_T"] for m in (_s3, _s4)]
assert _l[1] > _l[0], f"the text: lambda1/T grows from 7B to 32B ({_l})"

# the rank sweep (experiments/score/score_readout_rank.py, experiments/results/score_readout_rank.json; Section 5.4): every model of Section 5's
# population, ranks 1, 2, 4, 16, 64, q (the labels minus one, where pairs ask several questions) and D-1, on the readout's
# pairs with its objectives, starts, budgets and intervals, first order at the midpoint. The sentences of Section 5.4 that state
# a pattern are checked here against the scorer's rows. The scorer's tests: ceiling_k, the ceiling K_k (the first-order rank-k
# top); excess_k, cross-entropy DAS's excess over T; objective_k, cross-entropy against the squared error; and its descriptive one_question_ratios, gains and
# medians: one question, several questions, and the medians per rank.
RV = figdata.ROOT / "experiments/results/score_readout_rank.json"
rv = json.load(open(RV)); assert rv["complete"], "the rank sweep is not complete"
RM, RROWS = rv["report"]["models"], rv["rows"]
RCAP = [m for m in RM if m in CAPPED]; RFO = [m for m in RM if m not in CAPPED]; assert len(RCAP) == 1 and sorted(RM) == sorted(OK)
ALLK = ["1", "2", "4", "16", "64", "q", "D-1"]; KS = ["2", "4", "16", "64"]
big = lambda n: f"{n:,}".replace(",", "{,}")                                        # noqa: E731
put("rkModels", len(RM))
# the ceiling (Theorem 5(iii)): no start at any rank and budget significantly above K_k outside the
# cap; under Gemma-2's cap, the starts above it per rank
assert sum(RM[m]["main"]["ceiling_k"][k]["above"] for m in RFO for k in ALLK) == 0, "the text: no start above K_k outside the cap"
put("rkCeilTests", big(sum(RM[m]["main"]["ceiling_k"][k]["tests"] for m in RFO for k in ALLK)))
put("rkCapAbove", rng([RM[m]["main"]["ceiling_k"][k]["above"] for m in RCAP for k in ["1"] + KS + ["D-1"]], "{:d}"))
put("rkCapTests", RM[RCAP[0]]["main"]["ceiling_k"]["1"]["tests"])
assert all(RM[m]["main"]["ceiling_k"][k]["tests"] == RM[RCAP[0]]["main"]["ceiling_k"]["1"]["tests"] for m in RCAP for k in ["1"] + KS + ["D-1"])
put("rkCapAboveQ", RM[RCAP[0]]["main"]["ceiling_k"]["q"]["above"]); put("rkCapTestsQ", RM[RCAP[0]]["main"]["ceiling_k"]["q"]["tests"])   # q: fewer tests
# the excess at the ranks the text quotes: cross-entropy (one pass) significantly above T, of the model-task cases with room at
# that rank (excess_k), and above the squared error aimed at T (objective_k); then the cases whose excess_k reading holds at rank one and fails at rank k,
# and the reverse
for k, nm in (("1", "One"), ("2", "Two"), ("4", "Four"), ("16", "Sixteen"), ("64", "SixtyFour"), ("D-1", "Dm")):
    put(f"rkAbove{nm}", sum(RM[m]["main"]["excess_k"][k]["p1"]["above"] for m in RM))
    put(f"rkRoom{nm}", sum(RM[m]["main"]["excess_k"][k]["p1"]["room"] for m in RM))
for k, nm in (("1", "One"), ("2", "Two"), ("4", "Four"), ("16", "Sixteen"), ("64", "SixtyFour")):
    put(f"rkCeSq{nm}", sum(RM[m]["main"]["objective_k"][k]["p1"]["above"] for m in RM))
_lost = {k: sum(RM[m]["main"]["across"][k]["p1"]["excess_k_holds_at_one_fails_at_k"] for m in RM) for k in ALLK[1:]}
_gain = {k: sum(RM[m]["main"]["across"][k]["p1"]["excess_k_fails_at_one_holds_at_k"] for m in RM) for k in ALLK[1:]}
# three cases with room lie above T at rank one and not at a larger rank, and the text names them: Pythia-410m's JSON prompt
# of RAVEL's object colours at rank 64, where the interval over its three entities widens, and two one-directional
# negative-polarity constructions of the Llama-3.2 models, whose lower bounds come to T
_lostc = sorted((m, r["task"], k) for m, rs in RROWS.items() for r in rs for k in KS + ["q"]
                if r["k"]["1"].get("room", {}).get("room") and r["k"]["1"].get("excess_k", {}).get("p1") and r["k"]["1"]["excess_k"]["p1"]["lo"] > 1
                and r["k"].get(k, {}).get("room", {}).get("room") and r["k"][k].get("excess_k", {}).get("p1") and r["k"][k]["excess_k"]["p1"]["lo"] <= 1)
assert sum(_lost[k] for k in KS + ["q"]) == 3 and _lostc == [("Llama-3.2-1B", "NPIOnly", "4"), ("Llama-3.2-3B", "NPIEmb", "64"),
                                                             ("pythia-410m", "ravel_physical_object_color_json", "64")], _lostc
_lr = [r for r in RROWS["pythia-410m"] if r["task"] == "ravel_physical_object_color_json"][0]["k"]["64"]["excess_k"]["p1"]
put("rkLostRavel", f"{_lr['pt']:.2f}\\,$T$, lower bound {_lr['lo']:.2f}")
_ln = [[r for r in RROWS[m] if r["task"] == t][0]["k"][k]["excess_k"]["p1"]["lo"] for m, t, k in _lostc[:2]]
put("rkLostNpi", " and ".join(f"{x:.2f}" for x in _ln))
put("rkGained", rng([_gain[k] for k in ALLK[1:]], "{:d}"))
_below = {(m, r["task"]) for m, rs in RROWS.items() for r in rs for k, v in r["k"].items()
          if v.get("room", {}).get("room") and v.get("excess_k", {}).get("p1") and v["excess_k"]["p1"]["hi"] < 1}
assert all(t.startswith("NPI") for _, t in _below), "the text: every case below T at some rank is a negative-polarity exception"
_na = lambda names: {(m, r["task"]) for m, rs in RROWS.items() for r in rs for nm, k in r["names"] if nm in names   # noqa: E731
                     and r["k"][str(k)]["room"]["room"] and r["k"][str(k)]["excess_k"]["p1"]["lo"] <= 1}
_miss1 = _na({"1"})
assert _na({"2", "4", "16", "64", "q"}) - _miss1 == {(m, t) for m, t, _ in _lostc}, \
       "the text: every case not above T at a larger rank is a rank-one miss or one of those three"
assert all(t.startswith("NPI") or (m, t) == ("pythia-160m", "sentiment_classification") for m, t in _miss1), "the rank-one misses"
# the two-label cases, one question (one_question_ratios; Proposition 7): the top, K_k and cross-entropy at rank k against
# rank one, uncapped models and Gemma-2 apart
_oq_ratios = lambda key, ms, ks: [RM[m]["main"]["one_question_ratios"][k][key]["median"] for m in ms for k in ks]   # noqa: E731
put("rkTopRatio", rng(_oq_ratios("top_ratio", RFO, KS), "{:.3f}")); put("rkTopRatioCap", rng(_oq_ratios("top_ratio", RCAP, KS), "{:.3f}"))
put("rkKGain", rng([100 * x for x in _oq_ratios("K_gain_T", RFO, KS)], "{:.1f}"))
put("rkCeRatio", rng(_oq_ratios("ratio_ce_p1", RFO, KS), "{:.2f}")); put("rkCeRatioSixtyFour", rng(_oq_ratios("ratio_ce_p1", RFO, ["64"]), "{:.2f}"))
# several questions: IOI with twenty names, without room at rank one, gains room beyond it and DAS lies above T there
_nr = {k: (sum(RM[m]["main"]["excess_k"][k]["p1"].get("no_room_at_one", 0) for m in RM),
           sum(RM[m]["main"]["excess_k"][k]["p1"].get("no_room_at_one_above", 0) for m in RM)) for k in ("16", "64", "q")}
assert len(set(_nr.values())) == 1 and _nr["16"][0] == _nr["16"][1], "the text: room appears and DAS is above T in all of them"
_ioi = {(m, r["task"], r["label_pairs"]) for m, rs in RROWS.items() for r in rs
        if r["k"]["1"]["room"]["no_room"] and r["k"].get("16", {}).get("room", {}).get("room")}   # no room at one: the interval below 1
assert {t for _, t, _ in _ioi} == {"ioi_varied", "ioi_wang_20"} and len(_ioi) == _nr["16"][0], "the text: IOI with twenty names"
put("rkIoiCases", _nr["16"][0]); put("rkIoiLp", rng(sorted({lp for _, _, lp in _ioi}), "{:d}"))
# limits (medians, per-model medians in the uncapped models): where cross-entropy lands against K_k, how far the mean-effect
# objective reaches after ten passes, ten passes against one, the squared error over T, and the Ky Fan subspace's adequacy
_medians = lambda key, ks: [RM[m]["main"]["medians"][k][key] for m in RFO for k in ks if RM[m]["main"]["medians"][k].get(key) is not None]  # noqa: E731
put("rkLanding", rng(_medians("e_ce_over_K_p1", ["1"] + KS))); put("rkLandingQ", rng(_medians("e_ce_over_K_p1", ["q"])))
put("rkReach", rng(_medians("e_eff_over_K_p10", ["1"] + KS))); put("rkReachQ", rng(_medians("e_eff_over_K_p10", ["q"])))
put("rkTenPass", rng([100 * (x - 1) for x in _medians("ce10_over_ce1", ALLK)], "{:.0f}"))
put("rkSqT", rng(_medians("e_sq_T_p1", ALLK))); put("rkAdequacy", rng(_medians("adequacy_kyfan", ALLK)))
# both budgets (the text: "at both budgets")
for k, nm in (("1", "One"), ("2", "Two"), ("4", "Four"), ("16", "Sixteen"), ("64", "SixtyFour")):
    put(f"rkAboveTen{nm}", sum(RM[m]["main"]["excess_k"][k]["p10"]["above"] for m in RM))
# the out-of-sample top can fall below the rank-one top: cases with room at rank one and at no rank from 2 to q
_fall = [(m, r["task"]) for m, rs in RROWS.items() for r in rs if r["k"]["1"]["room"]["room"]
         and not any(r["k"][str(k)]["room"]["room"] for nm, k in r["names"] if nm in ("2", "4", "16", "64", "q"))]
assert {m for m, _ in _fall} == {"pythia-160m"} and all(t.startswith("agr_sv_num") for _, t in _fall), "the text: Pythia-160m's agreement"
put("rkTopFalls", len(_fall))
# from the per-pair files (experiments/results/readout_rank/), uncapped models: the Ky Fan subspace's exact effect against K_k at
# every rank; on two-label cases, the share of DAS (start 0) along the training pairs' top direction at ranks 1 and 64; on
# several-question cases, K_k's growth from rank one to 16 on the test pairs against what the training pairs' Ky Fan subspace
# and the out-of-sample top carry to them, and DAS's gain; IOI with twenty names apart, below T at rank one and above at 16
_share, _grow, _oos, _toptr, _dasg, _ioi1, _ioi16, _kfx = (collections.defaultdict(list) for _ in range(8))
_kfabove, _kfn = collections.Counter(), collections.Counter()
for m in RFO:
    byg = collections.defaultdict(list)
    for r in RROWS[m]: byg[r["group"]].append(r)
    for g, rs in byg.items():
        z = np.load(figdata.ROOT / f"experiments/results/readout_rank/readout_rank_{m}_m_{g}.npz")
        J = json.load(open(figdata.ROOT / f"experiments/results/readout_rank/readout_rank_{m}_m_{g}.json"))
        for r in rs:
            t = r["task"]; T = float(z[f"{t}/T"].astype(float).mean()); nk = dict(r["names"])
            e = lambda k, a: float(z[f"{t}/k{k}/{a}/e"].astype(float).mean()) / T   # noqa: E731
            for nm, k in r["names"]:
                K = r["k"][str(k)]["K_T"]; x = e(k, "kyfan") / K
                _kfn[nm] += 1; _kfabove[nm] += x > 1; _kfx[nm].append(x)
            if r["one_q"]:
                for nm in ("1", "64"):
                    v = J["tasks"][t]["ranks_out"][str(nk[nm])]["subspaces"]["das_ce_p1_s0"].get("share_v1_train")
                    if v is not None: _share[(m, nm)].append(v)
            elif t in ("ioi_varied", "ioi_wang_20"):
                ce = lambda k: np.mean([e(k, f"das_ce_p1_s{s_}") for s_ in range(3)])   # noqa: E731
                _ioi1[m].append(ce(nk["1"])); _ioi16[m].append(ce(nk["16"]))
            else:
                _grow[m].append(r["k"][str(nk["16"])]["K_T"] - r["k"][str(nk["1"])]["K_T"])
                _oos[m].append(e(nk["16"], "kyfan_oos") - e(nk["1"], "kyfan_oos"))
                _toptr[m].append(e(nk["16"], "top_train") - e(nk["1"], "top_train"))
                _dasg[m].append(np.mean([e(nk["16"], f"das_ce_p1_s{s_}") - e(nk["1"], f"das_ce_p1_s{s_}") for s_ in range(3)]))
_med = lambda dd: [float(np.median(v)) for v in dd.values() if v]   # noqa: E731
put("rkShareOne", rng([float(np.median(v)) for (m, nm), v in _share.items() if nm == "1"]))
put("rkShareSixtyFour", rng([float(np.median(v)) for (m, nm), v in _share.items() if nm == "64"]))
put("rkMultiN", rng([len(v) for v in _grow.values()], "{:d}"))
srng = lambda xs: f"${min(xs):+.2f}$ to ${max(xs):+.2f}$"   # noqa: E731     signed, with a real minus
put("rkMultiGrow", rng(_med(_grow))); put("rkMultiOos", srng(_med(_oos))); put("rkMultiTop", srng(_med(_toptr)))
put("rkMultiDas", srng(_med(_dasg)))
assert max(_med(_oos)) < 0.1 < min(_med(_grow)), "the text: the growth on the test pairs does not carry from the training pairs"
put("rkIoiCeOne", rng(_med(_ioi1))); put("rkIoiCeSixteen", rng(_med(_ioi16)))
assert max(_med(_ioi1)) < 1 < min(_med(_ioi16)), "the text: below T at rank one, above at 16"
_kr = [k for k in ("2", "4", "16", "64")]
put("rkKfAbove", rng([_kfabove[k] for k in _kr], "{:d}")); put("rkKfN", rng([_kfn[k] for k in _kr], "{:d}"))
put("rkKfMax", f"{100 * (max(max(_kfx[k]) for k in _kr) - 1):.1f}")

# the cross-entropy law's direction test (the scorer's alignment, experiments/results/readout_ce_law/score_readout_ce_law.json): whether first
# order predicts the direction of cross-entropy DAS's off-top part. It is untestable in every model (the scorer's noise floor
# removes every task), as Appendix C.2, "cross-entropy ends near its own optimum", says
_CL = json.load(open(figdata.ROOT / "experiments/results/readout_ce_law/score_readout_ce_law.json"))
assert all(v["alignment"]["held"] is None for m, v in _CL.items() if m != "tiers" and "alignment" in v), "the alignment test with the floor is untestable everywhere"
# the same paragraph's last sentence: on fully mirrored test sets a pair and its mirror cancel the softmax
# terms, so first order for log p(y_s) is half the log odds' (M_f = M_z / 2): its optimum is the log-odds top and it predicts no
# off-top part (sin(v_f, top) is float32 residue there), while cross-entropy DAS lies off the top. Per model: the median over the
# law's population (main tasks with room, the one-directional NPI constructions aside, above the noise floor), its fully
# mirrored tasks, of the three starts' mean sin(ce, top)
_mir = {(m, r["group"], r["task"]) for m in OK for r in S["rows"][m] if r.get("mirrored") == "all"}
_sv, _sce = [], {}
for m, v in _CL.items():
    if m == "tiers" or "per_task" not in v:
        continue
    for x in v["per_task"]:
        if not (x["main"] and x["room"] and not x["npi1"] and x["noise_ok"] and (m, x["group"], x["task"]) in _mir):
            continue
        row = json.load(open(figdata.ROOT / f"experiments/results/readout_ce_law/readout_ce_law_{m}_m_{x['group']}.json"))["tasks"][x["task"]]
        _sv.append(row["sin_v_f"]); _sce.setdefault(m, []).append(np.mean([row["align"][f"s{k}"]["sin_ce"] for k in (0, 1, 2)]))
assert max(_sv) < 1e-4, f"first order's off-top part on mirrored sets is not rounding residue: {max(_sv)}"
put("ceSinCe", f"{min(np.median(v) for v in _sce.values()):.2f}--{max(np.median(v) for v in _sce.values()):.2f}")

# the first-order ranking (the scorer's excess_rank) on CausalGym's 29 tasks in the six small models (the set small_causalgym,
# experiments/results/inside_sets/small_causalgym.post.json; Section 6.1 and App. D.3): does the first-order
# excess rank where DAS exceeds the network better than the best untrained comparator chosen on each map? Over constructions;
# seven comparators (the layer among them) and, at door cells, nine (with the norm of the input change and that norm over the
# layer's mean). Both forms are written, since the ranking holds only where both hold
_cg29 = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_causalgym.post.json"))["pooled"]["with_sst"]
_cg29_rank, _cg29_rank_door = _cg29["strong"]["excess_rank_ce"]["construction_level"], _cg29["strong"]["excess_rank_door_ce"]["construction_level"]
_m = lambda x: f"${x:+.3f}$"  # noqa: E731   (a signed number in math mode: a minus, not a hyphen)
put("gcgConstr", _cg29["constructions"]); put("gcgRankSeven", _m(_cg29_rank[0])); put("gcgRankSevenLo", _m(_cg29_rank[1]))
put("gcgRankSevenHi", _m(_cg29_rank[2]))
assert _cg29_rank[1] > 0 and _cg29_rank_door[1] < 0 < _cg29_rank_door[2], "the appendix: the seven-comparator form holds, the door form does not"
# the ranking's parts (experiments/score/inside_excess_rank.py --set=small_causalgym, a descriptive reading map by map): first order's
# own rank correlation, the best single predictor's and the per-map best of seven's; at door cells, first order's there,
# its lead over the norm of the input change and over that norm relative to its layer, and Spearman across maps of the door
# cells' attention share with first order's door correlation
_cg29_rank_parts = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_causalgym_excess_rank.post.json"))["pooled"]
_dd = _cg29_rank_parts["door"]; _f2 = lambda x: f"{x:.2f}"  # noqa: E731
put("gcgThRho", _f2(_cg29_rank_parts["theory_rho"]["mean"])); put("gcgBestSingleRho", _f2(max(_cg29_rank_parts["rho"].values())))
put("gcgBestSevenRho", _f2(_cg29_rank_parts["best7_rho"])); put("gcgDoorRho", _f2(_dd["theory_rho_door"]["mean"]))
assert all(v["lo"] > 0 for v in _cg29_rank_parts["minus"].values()), "the text: first order beats each single predictor"
assert _dd["minus_D"]["lo"] > 0 and _dd["minus_D_layer"]["lo"] > 0 and _dd["spearman_attn_theory"][0] > 0 and _dd["spearman_attn_theory"][1] < 0.05, \
       "the text: at door cells first order beats the norm, and its correlation rises with the attention share"
assert abs(_dd["theory_rho_door"]["mean"] - _cg29_rank_parts["theory_rho"]["mean"]) < 0.05, "the text: about as well at door cells"

# the first-order ranking (excess_rank) on the full grid (the set small_all: 78 tasks in the six small models; Section 6.1, "Prediction
# before training", and App. D.3), the same readings as on small_causalgym: the form against the best of seven
# untrained predictors per map and, at door cells, of nine, over constructions and over maps; its parts
# (experiments/score/inside_excess_rank.py --set=small_all): first order's own mean rank correlation, each single predictor's, the
# per-map best of seven's, and at door cells first order's lead over the norm of the input change and over it relative to its
# layer
_grid = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json"))["pooled"]["with_sst"]
_grid_rank, _grid_rank_door = _grid["strong"]["excess_rank_ce"], _grid["strong"]["excess_rank_door_ce"]
put("gridMaps", _grid["maps"]); put("gridConstr", _grid["constructions"])
for nm_, v_, lv, K in (("Seven", _grid_rank, "construction_level", ""), ("Seven", _grid_rank, "map_level", "Map"),
                      ("Door", _grid_rank_door, "construction_level", "")):
    put(f"gridRank{nm_}{K}", _m(v_[lv][0])); put(f"gridRank{nm_}{K}Lo", _m(v_[lv][1])); put(f"gridRank{nm_}{K}Hi", _m(v_[lv][2]))
assert not _grid_rank["holds"] and _grid_rank["construction_level"][1] < 0 < _grid_rank["construction_level"][2] and _grid_rank["map_level"][2] < 0, \
    "the text: on the full grid first order does not beat the per-map best of seven (over maps it falls below it)"
assert not _grid_rank_door["holds"] and _grid_rank_door["construction_level"][2] < 0, "the text: at door cells it falls below the best of nine"
_grid_rank_parts = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all_excess_rank.post.json"))["pooled"]
assert _grid_rank_parts["maps"] == _grid["maps"]
put("gridThRho", _f2(_grid_rank_parts["theory_rho"]["mean"])); put("gridBestSingleRho", _f2(max(_grid_rank_parts["rho"].values())))
put("gridBestSingleName", {"|g|": r"$\mathbb{E}\,|G_i|$", "depth": "the layer",
                             "G": "$G$"}[max(_grid_rank_parts["rho"], key=_grid_rank_parts["rho"].get)])
put("gridBestSevenRho", _f2(_grid_rank_parts["best7_rho"])); put("gridDoorRho", _f2(_grid_rank_parts["door"]["theory_rho_door"]["mean"]))
put("gridMinusSingleLo", _m(min(v["lo"] for v in _grid_rank_parts["minus"].values())))
assert all(v["lo"] > 0 for v in _grid_rank_parts["minus"].values()), "the text: first order beats each single predictor on the full grid"
put("gridAttnRho", _m(_grid_rank_parts["door"]["spearman_attn_theory"][0])); assert _grid_rank_parts["door"]["spearman_attn_theory"][1] < 0.05
put("gridMinusD", _m(_grid_rank_parts["door"]["minus_D"]["mean"])); put("gridMinusDLo", _m(_grid_rank_parts["door"]["minus_D"]["lo"]))
put("gridMinusDHi", _m(_grid_rank_parts["door"]["minus_D"]["hi"]))
assert _grid_rank_parts["door"]["minus_D"]["lo"] > 0, "the text: at door cells first order beats the size of the input change"
# per model: where the form over all cells holds, and where the per-map best lies significantly above first order
_grid_models = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json"))["models"]
put("gridModelsBelow", sum(v["strong"]["excess_rank_ce"]["map_level"][2] < 0 for v in _grid_models.values()))
# the other sources alone (the set small_other, the 49 tasks besides CausalGym's), reported apart over constructions
_other_rank = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_other.post.json"))["pooled"]["with_sst"]["strong"]["excess_rank_ce"]["construction_level"]
put("gotherRankSevenC", _m(_other_rank[0])); put("gotherRankSevenCLo", _m(_other_rank[1])); put("gotherRankSevenCHi", _m(_other_rank[2]))

# inside the network (Section 6 and App. D.1): the set small_all, the full grid of 78 tasks in the six
# small models. The population is the post scorer's (small_all.post.json's cell_classes: live cells that are not degenerate, in
# maps that enter), and the post scorer recounts false discovery over it per model. Section 6's figures draw the running
# example's map (paper/scripts/fig_guide_b.py, fig_guide_d.py, fig_guide_f.py).
IN_SET = "small_all"
SI = figdata.ROOT / f"experiments/results/inside_sets/{IN_SET}.json"
si = json.load(open(SI))
_post = json.load(open(figdata.ROOT / f"experiments/results/inside_sets/{IN_SET}.post.json")); pmods = _post["models"]
_POP = set(_post["cell_classes"])
si["cells"] = {m: [c for c in cs if f"{m}/{c['task']}/{c['cell']}" in _POP] for m, cs in si["cells"].items()}
put("inMaps", _post["pooled"]["with_sst"]["maps"]); put("inModels", len(pmods))
put("inTasks", len({c["task"] for cs in si["cells"].values() for c in cs}))
put("inExcludedMaps", len(_post["excluded_maps"])); put("inDegenerate", _post["degenerate_cells"])
cells = [c for m in si["cells"] for c in si["cells"][m]]
assert len(cells) == len(_POP) == sum(v["counts"]["live_cells"] for v in pmods.values())
put("inLive", len(cells)); put("inCeAboveG", sum(v["counts"]["above_own"]["G"]["above_bh"] for v in pmods.values()))
# cross-entropy DAS above the network's own interchange G (the scorer's above_own), and above G*, its best context, chosen on the
# training pairs and measured on the test pairs (chosen on the pairs it is read on, a maximum over contexts is pushed up;
# above_own's Gstar_train); the count with the test-chosen context is written beside it (App. D.1)
put("inCeAboveGstar", sum(v["counts"]["above_own"]["Gstar_train"]["above_bh"] for v in pmods.values()))
put("inCeAboveGstarTest", sum(v["counts"]["above_own"]["Gstar_test"]["above_bh"] for v in pmods.values()))
sys.path.insert(0, str(figdata.ROOT / "experiments/lib")); from measure import bh as _bh, boot_counts  # noqa: E402
_sig = [c for m in si["cells"] for c, r in zip(si["cells"][m], _bh([c["above_own_G"]["p"] for c in si["cells"][m]], 0.025)) if r]
assert len(_sig) == macros["inCeAboveG"]           # the same cells the post scorer counted
# the symmetric forms (experiments/score/inside_forms_post.py, small_all_forms.post.json): the above-G count over the carried cells
# (G's interval above 0), the three-way off-route class, and the not-produced share in three forms with intervals: pos,
# sum max(e - G*, 0) / sum e; signed, sum (e - G*) / sum e; cons, the same over the cells whose excess over G* is significant
_FM = json.load(open(figdata.ROOT / f"experiments/results/inside_sets/{IN_SET}_forms.post.json"))
assert _FM["pooled"]["above_G_bh"] == macros["inCeAboveG"] and _FM["cells"] == len(cells), "the forms read the same cells"
_fpm = {m: v["forms"] for m, v in _FM["models"].items()}
for n, K in (("pos", "Pos"), ("signed", "Signed"), ("cons", "Cons")):
    put(f"inNp{K}", rng([v[n]["share"] for v in _fpm.values()]))
# Section 6's sentence covers the three forms at once: the range of the per-model shares over all three, and the band their
# per-model 95% intervals lie in (App. D.5)
put("inNpAll", rng([v[n]["share"] for v in _fpm.values() for n in ("pos", "signed", "cons")]))
put("inNpBandLo", f"{min(v[n]['lo'] for v in _fpm.values() for n in ('pos', 'signed', 'cons')):.2f}")
put("inNpBandHi", f"{max(v[n]['hi'] for v in _fpm.values() for n in ('pos', 'signed', 'cons')):.2f}")
assert min(v[n]["lo"] for v in _fpm.values() for n in ("pos", "signed", "cons")) >= 0.495, "the text: about half or more in every form"
# the small models on the three tasks the large ones were mapped on (the set small_npi_fg_agr), for the large models' like-for-like
# comparison (Section 6.1)
_F1a = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_npi_fg_agr_forms.post.json"))
put("inThreeLive", _F1a["pooled"]["live"]); put("inThreeAboveG", _F1a["pooled"]["above_G_bh"])
put("inThreeNotProduced", rng([v["forms"]["pos"]["share"] for v in _F1a["models"].values()]))
put("inThreeThroughAttn", rng([v["forms"]["pos"]["through_attention_of_it"] for v in _F1a["models"].values()]))
# the per-source reading (Section 6.1): the syntactic sources (CausalGym, Kumon,
# number-only) and the others, each source's share of cells where DAS lies above G after false-discovery control per model
_SRCS = _FM["sources"]
_syn = ("CausalGym", "Kumon", "number-only"); _oth = ("RAVEL", "IOI", "SST", "sentiment")
assert set(_SRCS) == set(_syn) | set(_oth), sorted(_SRCS)
put("inAboveSynRange", rng([100 * _SRCS[k]["above_G_bh"] / _SRCS[k]["live"] for k in _syn], "{:.0f}"))
put("inAboveOther", rng([100 * _SRCS[k]["above_G_bh"] / _SRCS[k]["live"] for k in _oth], "{:.0f}"))
# how far DAS lies above G at those cells, in T: the median and quartiles
_ex = np.array([c["ce_T"] - c["G_T"] for c in _sig])
put("inExcessMed", f"{np.median(_ex):.2f}"); put("inExcessQone", f"{np.quantile(_ex, 0.25):.2f}"); put("inExcessQthree", f"{np.quantile(_ex, 0.75):.2f}")
# effect-seeking DAS above the first-order top (the scorer's ceiling; Section 6.1 and App. D.3)
put("inCeilBH", sum(v["counts"]["ceiling"]["above_bh"] for v in pmods.values()))
# the counting rules the text states (App. D.1, "cells and quantities"): raw per-cell intervals beside the
# false-discovery counts
put("inCeAboveGRaw", sum(v["counts"]["above_own"]["G"]["above"] for v in pmods.values()))
put("inCeAboveGstarRaw", sum(v["counts"]["above_own"]["Gstar_train"]["above"] for v in pmods.values()))
put("inCeilRaw", sum(any(x["above"] for x in c["ceiling"].values()) for c in cells))
_ceiling_cells = [c for m in si["cells"] for c, r in zip(si["cells"][m], _bh([c["ceiling_p"] for c in si["cells"][m]], 0.025)) if r]
assert len(_ceiling_cells) == macros["inCeilBH"]
# Section 6.2, the share through attention, with no threshold on G/T. The share of DAS's summed effect beyond the
# network's own in its best context, and that part's share through attention (each cell's excess weighted by the
# share of DAS's effect freezing removes, clipped to [0, 1], 0 where DAS's interval holds 0), per model; where in G* the
# not-produced effect lies (the G* below which half of it lies, a quantile of the data); and the rank correlation of G* with
# the share freezing removes, per model, over the cells where DAS's interval excludes 0
_np, _at, _rho = [], [], []
from scipy.stats import spearmanr as _spr  # noqa: E402
for m in sorted(si["cells"]):
    cs = si["cells"][m]; e = np.array([c["ce_T"] for c in cs]); g = np.array([c["Gstar_train_T"] for c in cs])
    sd = np.array([c["attention_share"]["s_das"] if c.get("ce_nonzero") and c["attention_share"].get("s_das") is not None else 0.0 for c in cs])
    ex = np.maximum(e - g, 0); _np.append(ex.sum() / e.sum()); _at.append((np.clip(sd, 0, 1) * ex).sum() / ex.sum())
    cz = [c for c in cs if c.get("ce_nonzero") and c["ce_T"] > 0 and c["attention_share"].get("s_das") is not None]
    _rho.append(_spr([c["Gstar_train_T"] for c in cz], [c["attention_share"]["s_das"] for c in cz])[0])
assert max(_rho) < 0, "the text: the less the network carries, the more of DAS's effect runs through attention"
put("inThroughAttnOfNotProduced", rng(_at)); put("inAttnRho", f"${min(_rho):.2f}$ to ${max(_rho):.2f}$")
# the large models (the set large_npi_fg_agr, experiments/results/inside_sets/large_npi_fg_agr.json and .post.json; Section 6.1): the four large models on Section 6's three tasks, described without tests (three maps per model); the same
# counts and shares, computed as above, beside the small models'. Gemma-2, whose logits are capped, is given apart and enters
# no pooled number.
_large_set = json.load(open(figdata.ROOT / "experiments/results/inside_sets/large_npi_fg_agr.json")); _large_post = json.load(open(figdata.ROOT / "experiments/results/inside_sets/large_npi_fg_agr.post.json"))
assert sorted(_large_set["cells"]) == ["Llama-3.1-8B", "Qwen2.5-7B", "gemma-2-9b", "pythia-6.9b"] and not _large_post["excluded_maps"]
assert {c["task"] for m in _large_set["cells"] for c in _large_set["cells"][m]} == {"npi_any_subj-relc", "filler_gap_pp", "agr_sv_num_subj-relc"}
_bnp, _bat = {}, {}
for m in sorted(_large_set["cells"]):
    cs = _large_set["cells"][m]; e = np.array([c["ce_T"] for c in cs]); g = np.array([c["Gstar_train_T"] for c in cs])
    sd = np.array([c["attention_share"]["s_das"] if c.get("ce_nonzero") and c["attention_share"].get("s_das") is not None else 0.0 for c in cs])
    ex = np.maximum(e - g, 0); _bnp[m] = ex.sum() / e.sum(); _bat[m] = (np.clip(sd, 0, 1) * ex).sum() / ex.sum()
_unc = [m for m in _bnp if m != "gemma-2-9b"]
put("inBigLive", sum(_large_post["models"][m]["counts"]["live_cells"] for m in _bnp))
put("inBigCeAboveG", sum(_large_post["models"][m]["counts"]["above_own"]["G"]["above_bh"] for m in _bnp))
put("inBigNotProduced", rng(list(_bnp.values()))); put("inBigThroughAttn", rng([_bat[m] for m in _unc]))
put("inBigNotProducedUnc", rng([_bnp[m] for m in _unc]))
put("inBigGemmaNotProduced", f"{_bnp['gemma-2-9b']:.2f}"); put("inBigGemmaThroughAttn", f"{_bat['gemma-2-9b']:.2f}")
_e = np.array([c["ce_T"] for c in cells]); _g = np.array([c["Gstar_train_T"] for c in cells]); _x = np.maximum(_e - _g, 0)
_o = np.argsort(_g); put("inHalfExcessGstar", f"{_g[_o][np.searchsorted(np.cumsum(_x[_o]) / _x.sum(), 0.5)]:.2f}")
# the per-cell means over T (each map's mean_T in experiments/results/inside/): the objectives, the untrained directions and
# freezing, at every cell of the population
raw = []
scored = {(c["model"], c["task"]) for c in cells}                  # the maps the score covers, no others
for f in sorted((figdata.ROOT / "experiments/results/inside").glob("inside_*.json")):
    mt_ = [(m, t) for m, t in scored if f.name == f"inside_{m}_{t}.json"]
    if not mt_:
        continue
    for c in json.load(open(f))["cells"]:
        if c.get("live") and f"{mt_[0][0]}/{mt_[0][1]}/L{c['layer']}r{c['region']}" in _POP:
            mt = c["mean_T"]
            raw.append(dict(G=mt["own_normal"], G_fr=mt["own_frozen"],
                            ce=np.mean([mt[f"das_ce_s{x}_normal"] for x in (0, 1, 2)]), ce_fr=np.mean([mt[f"das_ce_s{x}_frozen"] for x in (0, 1, 2)]),
                            sqT=mt["das_sq_T_s0_normal"], sqT_fr=mt["das_sq_T_s0_frozen"],
                            sqG=np.mean([mt[f"das_sq_G_s{x}_normal"] for x in (0, 1, 2)]),
                            jsd=mt["das_jsd_s0_normal"], jsd_fr=mt["das_jsd_s0_frozen"], bis=mt["bisector_normal"],
                            uh=mt["u_hat_normal"], bh=mt["b_hat_normal"], key=(*mt_[0], f"L{c['layer']}r{c['region']}")))
sc_ = {(c["model"], c["task"], c["cell"]): c for c in cells}
assert len(raw) == len(sc_) and all(r["key"] in sc_ for r in raw)
for r in raw:
    r["Gs"] = sc_[r["key"]]["Gstar_train_T"]
A_ = lambda k: np.array([r[k] for r in raw])  # noqa: E731
G_, Gfr_ = A_("G"), A_("G_fr")
# the target decides (Section 6.3 and App. D.6): the objectives
# over every live cell, without classes, each one's median distance from G and a least-squares line on G; cross-entropy and
# the G-aimed squared error from three starts, the T-aimed squared error and the divergence from the first, and u-hat
# (u-hat outside IOI, whose class sign follows the base type and does not fix IOI's answer, as for its sums below)
_noi = np.array([not r["key"][1].startswith("ioi_") for r in raw])
for k, K in (("ce", "Ce"), ("sqT", "SqT"), ("sqG", "SqG"), ("uh", "Uhat")):
    m_ = _noi if k == "uh" else np.ones(len(raw), bool)
    v_, G_k = A_(k)[m_], G_[m_]; sl_, ic_ = np.linalg.lstsq(np.vstack([G_k, np.ones_like(G_k)]).T, v_, rcond=None)[0]
    if k in ("ce", "sqT"):                                          # the text quotes these two's distance from G
        put(f"in{K}MinusG", f"{abs(np.median(v_ - G_k)):.2f}" if abs(np.median(v_ - G_k)) < 0.005 else f"{np.median(v_ - G_k):.2f}")
    put(f"in{K}Slope", f"{sl_:.2f}"); put(f"in{K}Intercept", f"{abs(ic_):.2f}" if abs(ic_) < 0.005 else f"{ic_:.2f}")
# the T-aimed squared error's excess over G*, and its share through attention, weighted as the route's share above: the
# share at each cell, zero where the arm's own interval holds zero (experiments/score/inside_forms_post.py, "sqT_nonzero")
_nzT = np.array([_FM["sqT_nonzero"]["/".join(r["key"])] for r in raw], float)
_xT = np.maximum(A_("sqT") - A_("Gs"), 0)
_sT = _nzT * np.nan_to_num(np.clip(1 - A_("sqT_fr") / np.where(np.abs(A_("sqT")) > 1e-9, A_("sqT"), np.nan), 0, 1))
put("inSqTAttn", f"{(_sT * _xT).sum() / _xT.sum():.2f}")
# freezing over every live cell, summed (Section 6.2): what the network's own
# interchange keeps, what freezing removes of DAS's effect and of its excess over G
put("inOwnKeep", f"{Gfr_.sum() / G_.sum():.2f}")
put("inCeAttnAll", f"{1 - A_('ce_fr').sum() / A_('ce').sum():.2f}")
put("inExcessAttnAll", f"{1 - (A_('ce_fr') - Gfr_).sum() / (A_('ce') - G_).sum():.2f}")
# the checks beside the share (experiments/results/inside_sets/small_all_forms.post.json, "checks"): the attention part compared directly,
# DAS swapped in G*'s own context, freezing only the readout position's attention, and what G*'s second region adds
_CK = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all_forms.post.json"))["checks"]
assert f"{_CK['pooled']['excess_over_G_removed']['every_pattern']:.2f}" == f"{1 - (A_('ce_fr') - Gfr_).sum() / (A_('ce') - G_).sum():.2f}"
put("inAttDirect", rng([v["attention_direct"]["share"] for v in _CK["models"].values()]))
put("inNpCtx", rng([v["not_produced_in_context"]["share"] for v in _CK["models"].values()]))
put("inAboveGstarCtx", _CK["pooled"]["above_gstar_in_context"])
put("inRowExcess", f"{_CK['pooled']['excess_over_G_removed']['readout_row']:.2f}")
# at the off-route cells: the share of DAS's summed effect that freezing only the readout position's row removes (App. D.5:
# most of the off-route effect runs through that row)
_orr = _CK["off_route_removed"]; assert _orr["readout_row"] > 0.5
put("inOffRowRemoved", f"{_orr['readout_row']:.2f}")
put("inCtxGainMed", f"{_CK['pooled']['context_gain']['median']:.3f}"); put("inCtxGainQnine", f"{_CK['pooled']['context_gain']['q90']:.2f}")
# the untrained directions over G, summed over the live cells (weighted by G): the bisector of u-hat and b-hat, u-hat (the mean
# difference) and b-hat (the readout gradient)
# (IOI's maps leave the natural directions' sums: their class sign follows the base type, which does not fix IOI's answer)
# (and the cells whose bisector or b-hat is undefined, a zero mean gradient, leave all three sums; fewer than ten, asserted)
_nio = np.array([not r["key"][1].startswith("ioi_") for r in raw]) & np.isfinite(A_("bis")) & np.isfinite(A_("bh"))
assert (~np.isfinite(A_("bis"))).sum() < 10
for k, K in (("bis", "Bis"), ("uh", "Uhat"), ("bh", "Bhat")):
    put(f"in{K}Sum", f"{A_(k)[_nio].sum() / G_[_nio].sum():.2f}")
# off-route (Section 6), three ways without a threshold (experiments/score/inside_forms_post.py): G*'s interval not above 0, DAS's
# above 0, and the paired interval of (DAS - G*)/T above 0; the indeterminate rest beside. How small G* is there (its upper
# bound), how large DAS is, and what freezing removes of it, summed; the zero class (G*'s interval not above 0) for the
# objectives aimed at T or at the source run: their medians, their attention shares and their largest effect (Section 6.3 and App. D.6)
_ZC = _FM["zero_cells"]; _X = {tuple(k.rsplit("/", 2)): v for k, v in _ZC.items()}
_z = [r for r in raw if r["key"] in _X]; _off = [r for r in _z if _X[r["key"]]["cls"] == "off"]
assert len(_z) == _FM["pooled"]["gstar_zero"] and len(_off) == _FM["pooled"]["off_route"]
put("inOffRoute", len(_off)); put("inZero", len(_z))
put("inOffRouteIndet", _FM["pooled"]["indeterminate"]); put("inOffRouteBH", _FM["pooled"]["off_route_bh"])
# a few maps of the full grid have a small T (such as Pythia-160m's filler-gap embeddings), so the extremes of a ratio over T
# say little: the median of G*'s upper bound and its 90th percentile
_gh = [_X[r["key"]]["g"][2] for r in _off]
put("inOffRouteGstarHi", f"{np.median(_gh):.3f}"); put("inOffRouteGstarHiNinety", f"{np.quantile(_gh, 0.9):.3f}")
# the off-route cells where G*'s whole interval lies below zero (and the same over the zero class), and those with DAS above T
put("inOffRouteBelowZero", sum(g_ < 0 for g_ in _gh))
put("inOffRouteDasMed", f"{np.median([r['ce'] for r in _off]):.2f}"); put("inOffRouteDasNinety", f"{np.quantile([r['ce'] for r in _off], 0.9):.2f}")
put("inOffRouteAttn", f"{1 - sum(r['ce_fr'] for r in _off) / sum(r['ce'] for r in _off):.2f}")
put("inZeroSqT", f"{np.median([r['sqT'] for r in _z]):.2f}"); put("inZeroJsd", f"{np.median([r['jsd'] for r in _z]):.2f}")
put("inZeroSqTAttn", f"{1 - sum(r['sqT_fr'] for r in _z) / sum(r['sqT'] for r in _z):.2f}")
put("inZeroJsdAttn", f"{1 - sum(r['jsd_fr'] for r in _z) / sum(r['jsd'] for r in _z):.2f}")
put("inZeroMax", f"{max(max(r['sqT'], r['jsd']) for r in _z):.2f}")
# the text: almost all of these effects lie below T; the few at or above it are counted
put("inZeroAtT", sum(max(r["sqT"], r["jsd"]) >= 1 for r in _z))
assert macros["inZeroAtT"] < 0.01 * len(_z), "the text: almost all of them lie below T"
# the zero class's other objectives against G* by the difference's interval (as the off-route class reads DAS)
put("inZeroSqTAboveGstar", _FM["pooled"]["zero_above_gstar"]["das_sq_T_s0_normal"])
put("inZeroJsdAboveGstar", _FM["pooled"]["zero_above_gstar"]["das_jsd_s0_normal"])
# and by the off-route rule of Section 6 (raw intervals, as DAS's off-route count): their own interval above 0 as well
put("inZeroSqTOff", sum(v["sqT"]["efflo"] > 0 and v["sqT"]["difflo"] > 0 for v in _ZC.values()))
put("inZeroJsdOff", sum(v["jsd"]["efflo"] > 0 and v["jsd"]["difflo"] > 0 for v in _ZC.values()))
assert macros["inZeroSqTOff"] <= macros["inZeroSqTAboveGstar"] and macros["inZeroJsdOff"] <= macros["inZeroJsdAboveGstar"]
# the cells where DAS exceeds the first-order top (ceiling): how much the network carries there and what freezing removes
_ceiling_keys = {(c["model"], c["task"], c["cell"]) for c in _ceiling_cells}; _ceiling_rows = [r for r in raw if r["key"] in _ceiling_keys]
put("inCeilGstarMed", f"{np.median([r['Gs'] for r in _ceiling_rows]):.2f}")
put("inCeilAttn", f"{1 - sum(r['ce_fr'] for r in _ceiling_rows) / sum(r['ce'] for r in _ceiling_rows):.2f}")
# accuracy where the effect exceeds (Section 7.2): at the cells where cross-entropy DAS lies above G (the above_own count, inCeAboveG), DAS's accuracy (its three starts'
# per-pair accuracies averaged) against the network's own interchange's at the same cell, by the interval of their
# difference on the map's own resamples (the readout's rule, accuracy; the maps' per-pair arrays in experiments/results/inside/), and
# where both accuracies are 1 (every test pair right)
_cnt = dict(up=0, down=0, sat=0); _acc_ce, _acc_own = [], []
for (m, t) in sorted({(c["model"], c["task"]) for c in _sig}):
    J_ = json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{m}_{t}.json"))
    Z_ = np.load(figdata.ROOT / f"experiments/results/inside/inside_{m}_{t}.npz")
    C_ = boot_counts(Z_["cluster"], J_["boot"]["B"], J_["boot"]["seed"])
    assert hashlib.sha256(np.ascontiguousarray(C_, dtype=np.int64).tobytes()).hexdigest()[:16] == J_["boot_counts_sha"]
    for c in [c for c in _sig if (c["model"], c["task"]) == (m, t)]:
        a_ce = np.mean([Z_[f"{c['cell']}/acc/das_ce_s{x}_normal"] for x in (0, 1, 2)], axis=0); a_own = Z_[f"{c['cell']}/acc/own_normal"]
        b_ = (C_ @ (a_ce - a_own)) / C_.sum(1)
        _cnt["up"] += np.quantile(b_, 0.025) > 0; _cnt["down"] += np.quantile(b_, 0.975) < 0
        _cnt["sat"] += a_ce.mean() >= 1 - 1e-12 and a_own.mean() >= 1 - 1e-12              # every test pair right
        _acc_ce.append(a_ce.mean()); _acc_own.append(a_own.mean())
assert len(_acc_ce) == macros["inCeAboveG"]
put("inAccUpCe", int(_cnt["up"])); put("inAccDownCe", int(_cnt["down"])); put("inAccSatCe", int(_cnt["sat"]))
put("inAccCeMed", f"{np.median(_acc_ce):.2f}"); put("inAccOwnMed", f"{np.median(_acc_own):.2f}")
# where first order holds (the scorer's trace; App. D.3): per model the median midpoint trace over G and
# how often it is closer to G than the base-point trace, over the population's cells; the largest per-layer median overshoot;
# the exact effect over its first-order value, per model the median over cells of the ratio of their means (as the text reads it:
# a ratio of sums is dominated by the few large cells), for the untrained directions (the mean difference, the readout gradient,
# their bisector, M's top eigenvector) and cross-entropy DAS; cells where the direction is undefined (a zero mean gradient) or
# its first-order effect is exactly 0 leave
_trace_cells = {m: [c["trace"] for c in cs if c.get("trace") and c["trace"].get("tr_mid_over_G") is not None] for m, cs in si["cells"].items()}
put("inTraceMid", rng([float(np.median([x["tr_mid_over_G"] for x in v])) for v in _trace_cells.values()]))
put("inTraceMidCloser", rng([100 * np.mean([bool(x["mid_closer"]) for x in v]) for v in _trace_cells.values()], "{:.0f}"))
_ov = []
for m, cs in si["cells"].items():
    byL = {}
    for c in cs:
        if c.get("trace") and c["trace"].get("tr_mid_over_G") is not None:
            byL.setdefault(c["layer"], []).append(c["trace"]["tr_mid_over_G"])
    _ov.append(max(float(np.median(v)) for v in byL.values()) - 1)
put("inTraceOvershoot", rng([100 * x for x in _ov], "{:.0f}"))
_fom = {}
for (m, t) in sorted({(c["model"], c["task"]) for c in cells}):
    Z_ = np.load(figdata.ROOT / f"experiments/results/inside/inside_{m}_{t}.npz")
    for c in (c for c in si["cells"][m] if c["task"] == t):
        for k_ in ("u_hat", "b_hat", "bisector", "v1_mid", "das_ce"):
            if k_ == "das_ce":
                ex_ = np.mean([Z_[f"{c['cell']}/e/das_ce_s{x}_normal"].mean() for x in (0, 1, 2)])
                fo_ = np.mean([Z_[f"{c['cell']}/fm/das_ce_s{x}"].mean() for x in (0, 1, 2)])
            else:
                ex_, fo_ = Z_[f"{c['cell']}/e/{k_}_normal"].mean(), Z_[f"{c['cell']}/fm/{k_}"].mean()
            if np.isfinite(ex_) and np.isfinite(fo_) and fo_ != 0:
                _fom.setdefault((m, k_), []).append(float(ex_ / fo_))
put("inFoUntrained", rng([float(np.median(_fom[(m, k_)])) for m in si["cells"] for k_ in ("u_hat", "b_hat", "bisector", "v1_mid")]))
put("inFoDas", rng([float(np.median(_fom[(m, "das_ce")])) for m in si["cells"]]))
# the route by family (Section 6.2 and App. D.5; the families of experiments/score/inside_set_post.py): negative
# polarity and filler-gap are CausalGym's npi_ and filler_gap_ tasks and Kumon's NPI and FG constructions, agreement is
# CausalGym's agr_ tasks and number-only agreement. Per model, the median share of DAS's effect that freezing removes (attention_share's
# s_das) over the population's cells where it is defined (DAS's interval excludes 0); and the post scorer's reading per
# source (more on NPI and filler-gap than on agreement, in every model)
_fam = lambda t: "npifg" if t.startswith(("npi_", "filler_gap_", "NPI", "FG")) else "agr" if t.startswith("agr_") else None  # noqa: E731
_sd = lambda m, f: [c["attention_share"]["s_das"] for c in si["cells"][m] if _fam(c["task"]) == f and c.get("ce_nonzero") and c["attention_share"].get("s_das") is not None]  # noqa: E731
put("inSdasNpiFg", rng([float(np.median(_sd(m, "npifg"))) for m in si["cells"]]))
put("inSdasAgr", rng([float(np.median(_sd(m, "agr"))) for m in si["cells"]]))
_oth = lambda m: [c["attention_share"]["s_das"] for c in si["cells"][m] if c["task"].startswith(("ravel_", "ioi_", "sst_", "sentiment_"))  # noqa: E731
                  and c.get("ce_nonzero") and c["attention_share"].get("s_das") is not None]
put("inSdasOther", rng([float(np.median(_oth(m))) for m in si["cells"]]))
assert all(_post["npi_fg_reading"][k]["holds_every_model"] for k in ("causalgym", "kumon_number_only")), "the text: in every model, in both sources"
# the first-order description of Proposition 8 (the door) in the first block that reads the site (the scorer's door_adequacy;
# App. D.4): per model the median over the population's door cells (the scorer's own median counts degenerate
# cells)
put("inDoorFirstOrder", rng([float(np.median([c["door_adequacy"]["own_adequacy_corr"] for c in si["cells"][m] if c.get("door") and c.get("door_adequacy")
                                              and c["door_adequacy"].get("own_adequacy_corr") is not None])) for m in si["cells"]]))
# the share of door cells whose largest rise of attention is in the first block that reads the site, per model, from the
# post scorer over the population
put("inDoorBlockLShare", rng([100 * v["door_block"]["argmax_in_block_L"] / v["door_block"]["door_cells"] for v in pmods.values()], "{:.0f}"))

# OwnEffect against the measurement scripts on GPT-2 (experiments/scripts/owneffect_repro.py; App. F): the tasks, maps,
# cells and effects compared, and the largest absolute differences
OE = figdata.ROOT / "experiments/results/owneffect_repro/owneffect_repro_gpt2.json"
oe = json.load(open(OE)); tot = oe["total"]
def _sci(x):
    e = int(np.floor(np.log10(x))); m = x / 10 ** e; m = round(m + 0.05, 1) if round(m, 1) < m else round(m, 1)
    return f"${m:g}\\times10^{{{e}}}$"
put("oeTasks", len(oe["readout"])); put("oeMaps", len(oe["inside"]))
put("oeCells", tot["inside: lambda_1 (midpoint)"][1])
put("oeEffects", f"{tot['readout: exact effect of a saved direction, per pair'][1]:,}".replace(",", "{,}"))
put("oeReadoutMax", _sci(max(tot[k][0] for k in ("readout: T per test pair", "readout: exact effect of a saved direction, per pair"))))
put("oeSpecMax", _sci(max(tot[k][0] for k in ("readout: lambda_1 of the test M (midpoint)", "readout: tr M (midpoint)", "readout: tr M (base point)"))))
put("oeAngleMax", _sci(max(tot[k][0] for k in ("readout: cos theta (training pairs)", "readout: readout direction's share (test pairs)"))))
# the context OwnEffect chooses for G* on the training pairs is the saved one at every live cell
assert tot["inside: G*'s context chosen on the training pairs, 0 where it is the saved one"] == [0.0, tot["inside: lambda_1 (midpoint)"][1]]
put("oeInsideMax", _sci(max(tot[k][0] for k in ("inside: own swap, per pair", "inside: G* on the test pairs, per pair",
                                                 "inside: G* in the training pairs' context (context=), per pair",
                                                 "inside: a saved direction's effect, per pair", "inside: the same with attention frozen, per pair"))))
put("oeInsideLam", _sci(tot["inside: lambda_1 (midpoint)"][0]))
put("oeInsideWD", _sci(tot["inside: w . Delta per pair (midpoint)"][0]))
put("oeOneQ", tot["readout: cos theta (training pairs)"][1])

# the maps of the other sources (the set small_other, experiments/results/inside_sets/small_other.json, the scorer's live cells): where
# cross-entropy DAS lies above the network's own interchange, after false-discovery control over each model's cells as the
# scorer counts it, by source; the assert checks the abstract's inside sentence (above at most cells in every source)
sys.path.insert(0, str(figdata.ROOT / "experiments/lib"))
from measure import bh as _bh  # noqa: E402
_other = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_other.json"))
_src2 = lambda t: ("number-only" if t.endswith("__num") else "Kumon" if t.startswith(("NPI", "FG")) else "RAVEL" if t.startswith("ravel_")  # noqa: E731
                   else "sentiment" if t.startswith("sentiment_") else "SST" if t.startswith("sst_") else "IOI" if t.startswith("ioi_") else "other")
_n2, _a2 = {}, {}
for m, cs in _other["cells"].items():
    _f = _bh([c["above_own_G"]["p"] for c in cs], 0.025)
    assert int(_f.sum()) == _other["report"]["models"][m]["above_own"]["G"]["above_bh"], "the scorer's own count"
    for c, x in zip(cs, _f):
        k = _src2(c["task"]); _n2[k] = _n2.get(k, 0) + 1; _a2[k] = _a2.get(k, 0) + bool(x)
assert set(_n2) == {"Kumon", "number-only", "RAVEL", "IOI", "SST", "sentiment"}, "the text names six sources"
assert min(_a2[k] / _n2[k] for k in _n2) > 0.5, "the abstract: above at most cells in every source"

# the sentiment re-run's test prompts per model, their range over the seven models (experiments/results/score_sentiment_reread.json;
# App. B.1)
_SR = json.load(open(figdata.ROOT / "experiments/results/score_sentiment_reread.json"))["models"]
put("srTestPrompts", rng([v["n_test"] for v in _SR.values()], "{:d}"))
# the max-type band resamples blocks of two neighbouring prompts: their number per model
put("srBlocks", rng([int(np.ceil(v["n_test"] / 2)) for v in _SR.values()], "{:d}"))

# appendix numbers: the sentiment templates' distinct sentences, over the main population's templates that reuse them (App. C.1,
# "intervals when sentences repeat"); the maps of small_causalgym; how long a map took, from the run records of the maps
# small_causalgym and small_other score (experiments/logs/inside_*/done.json, the latest run per map; App. B.5)
_sn, _snt = [], set()
for m in OK:
    for r in S["rows"][m]:
        if r["main"] and r["defined"] and r["group"] in ("sentiment_templates", "sentiment_templates_model_specific"):
            R_ = json.load(open(figdata.ROOT / f"experiments/data/{m}/sentiment_templates/{r['task']}/test.json"))
            d_ = {"".join(x["base"]) for x in R_} | {"".join(x["src"]) for x in R_}
            if len(d_) < len(R_):                                  # fewer distinct sentences than pairs
                _sn.append(len(d_)); _snt.add(r["task"])
put("snSentRange", rng(_sn, "{:d}"))
# the templates that reuse sentences are exactly the sentiment templates whose sentence clusters fall below 20 in the
# sentence-level bootstrap (the text: "not those templates"); the others keep their intervals under it
_und = {r["task"] for m in OK if m in sb for r in sb[m]["tasks"] if str(r.get("group", "")).startswith("sentiment_templates")
        and not r["defined"]}
assert _und == _snt, (sorted(_und), sorted(_snt))
# the other sources alone (the set small_other): which forms of the first-order reading hold there (checked, not written)
_other_strong = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_other.post.json"))["pooled"]["with_sst"]["strong"]
# the text (App. D.4, "Where the attention to the site rises"): on the main set small_all too, the door predictor does not rank the cells better
# than the best of the three simple predictors (the norm of the input change, the attention weight, -G), map level, pooled
_dd = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json"))["pooled"]["with_sst"]["strong"]["door_predictor_diff"]["map_level"]
assert _dd[2] < 0, f"the text: the door predictor does not beat the best of the three simple predictors on small_all ({_dd})"
assert not any(_other_strong[k]["holds"] for k in ("excess_rank_ce", "excess_rank_door_ce", "door_predictor_diff", "squared_error_route")) and _other_strong["squared_error_route_median"]["holds"], \
    "small_other: the first-order ranking, its door form, the door predictor and the squared error's route form fail; the route form's median version holds"
_mp = set()
for P_ in ("small_causalgym", "small_other"):
    _mp |= {f"experiments/results/inside/inside_{k.replace('/', '_')}.json" for k in json.load(open(figdata.ROOT / f"experiments/results/inside_sets/{P_}.post.json"))["excess_rank_per_map"]}
_last = {}
for d_ in (figdata.ROOT / "experiments/logs").glob("inside_*/done.json"):
    j_ = json.load(open(d_))
    for o in j_["outputs"]:
        o = str(o["path"] if isinstance(o, dict) else o)
        if o in _mp and (o not in _last or j_["finished"] > _last[o][0]):
            _last[o] = (j_["finished"], j_["seconds"] / 3600)
assert len(_last) == len(_mp), f"run records for {len(_last)} of {len(_mp)} maps"
_hh = [v[1] for v in _last.values()]
put("inMapHours", f"{min(_hh):.2f}--{max(_hh):.1f}"); put("inMapHoursMed", f"{np.median(_hh):.1f}")

# CausalGym's own code on the GPU, as released and with the position fix, on the 19 shifted tasks and two aligned controls,
# Pythia 14M-410M (experiments/score/score_causalgym_rerun.py, its readings with intervals; Appendix E.2). The re-run scorer's readings:
# the instrument (the deterministic methods give identical rows in both conditions on the aligned controls); fix_change, how the fix
# changes the mean difference over vanilla, and fix_level, where the fixed mean difference lies against vanilla; replication, the tasks
# whose released score rounds to the published value, per method; das_with_fix, DAS above vanilla and above the best other method
CG_RERUN = json.load(open(figdata.ROOT / "experiments/results/score_causalgym_rerun.json"))
_cg_rerun_models = ["pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m"]
assert sorted(CG_RERUN) == sorted(_cg_rerun_models) and all(CG_RERUN[m]["instrument"] and all(CG_RERUN[m]["instrument_controls_identical"]) for m in _cg_rerun_models), "the re-run's instruments at every size"
# the methods whose released scores must round to the published ones on all 19 tasks; LDA is reported apart (cgrLdaRepro)
_REPRO_METHODS = ("vanilla", "mean", "probe_0", "probe_1", "pca", "kmeans")
_repro = [m for m in _cg_rerun_models if all(v["rounds_to"] == v["with_published"] == 19 for k, v in CG_RERUN[m]["summary"]["replication"].items() if k in _REPRO_METHODS)]
assert _repro == ["pythia-70m", "pythia-160m", "pythia-410m"], "the text: the released code reproduces the published scores at 70M-410M"
assert all(CG_RERUN[m]["summary"]["replication"]["vanilla"]["rounds_to"] <= 1 for m in ("pythia-14m", "pythia-31m")), "the text: not at 14M and 31M (replaced weights)"
put("cgrLdaRepro", rng([CG_RERUN[m]["summary"]["replication"]["lda"]["rounds_to"] for m in _repro], "{:d}"))
put("cgrFixAbove", rng([CG_RERUN[m]["summary"]["fix_change"]["above"] for m in _cg_rerun_models], "{:d}"))
put("cgrReach", rng([CG_RERUN[m]["summary"]["fix_level"]["reaches vanilla"] for m in _cg_rerun_models], "{:d}"))
assert all(CG_RERUN[m]["summary"]["das_with_fix"] == {"above_vanilla": 19, "above_best_other": 19} for m in _cg_rerun_models), "the text: DAS above vanilla and every other method on all 19"
_rv = lambda m, cond, k: float(np.median([t[cond][k] / t[cond]["vanilla"] for t in CG_RERUN[m]["tasks"].values() if k in t[cond]]))  # noqa: E731
put("cgrMeanRel", rng([_rv(m, "released", "mean") for m in _cg_rerun_models])); put("cgrMeanFixed", rng([_rv(m, "fixed", "mean") for m in _cg_rerun_models]))
put("cgrProbeFixed", rng([_rv(m, "fixed", k) for m in _cg_rerun_models for k in ("probe_0", "probe_1") if any(k in t["fixed"] for t in CG_RERUN[m]["tasks"].values())]))
put("cgrPcaKmFixed", rng([_rv(m, "fixed", k) for m in _cg_rerun_models for k in ("pca", "kmeans")]))
put("cgrDasFixed", rng([_rv(m, "fixed", "das") for m in _cg_rerun_models], "{:.1f}"))
_cl = CG_RERUN["pythia-160m"]["tasks"]["cleft"]["fixed"]
put("cgrCleftFour", rng([_cl[k] for k in ("mean", "pca", "kmeans", "probe_0", "probe_1")])); put("cgrCleftVanilla", f"{_cl['vanilla']:.2f}")

# where a swap reads the source at another absolute position (the maps' per-pair aligned flag: equal length and equal region
# ends): CausalGym's 29 tasks in the six small models' four tokenizers (Pythia's three share one), on the test pairs
# (App. B.2). The tasks misaligned on every pair are the ones with a region empty
# in some row (experiments/lib/regions.py), read from the pinned CausalGym snapshot
TOKZ = {"gpt2": "gpt2", "pythia-160m": "pythia", "pythia-410m": "pythia", "pythia-1b": "pythia", "SmolLM2-360M": "smollm2",
        "Qwen2.5-0.5B": "qwen2.5"}
_mis = {}
_cg29_tasks = {k.split("/")[1] for k in json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_causalgym.post.json"))["cell_classes"]}   # the 29 (other sets' maps share the folder)
for m in TOKZ:
    for f in sorted((figdata.ROOT / "experiments/results/inside").glob(f"inside_{m}_*.json")):
        t = json.load(open(f))["task"]
        if t not in _cg29_tasks:
            continue
        _mis.setdefault(t, {})[TOKZ[m]] = float(1 - np.load(f.with_suffix(".npz"))["aligned"].mean())
assert len(_mis) == 29 and all(len(v) == 4 for v in _mis.values()), "CausalGym's 29 maps in all four tokenizers"
sys.path.insert(0, str(figdata.ROOT / "experiments/lib"))
from huggingface_hub import hf_hub_download  # noqa: E402
from regions import facts  # noqa: E402   (the rule for a region empty in some row)
_cg = {s: json.load(open(hf_hub_download("aryaman/causalgym", f"{s}.json", repo_type="dataset", revision=registry.CAUSALGYM_REV)))
       for s in ("train", "test")}
_empty = {t for t in _mis if facts({s: [r for r in _cg[s] if r["task"] == t] for s in _cg})["degenerate_regions"]}
assert _empty == {t for t, v in _mis.items() if min(v.values()) == 1.0}, "the tasks misaligned on every pair are the empty-region ones"
assert all(x == 0 for t, v in _mis.items() if t.startswith(("npi_", "filler_gap_")) for x in v.values())
_num = [x for t, v in _mis.items() if t.startswith("agr_") and "_num_" in t for x in v.values()]
_gen = [x for t, v in _mis.items() if t == "agr_gender" for x in v.values()]
put("tokNumLo", round(100 * min(_num))); put("tokNumHi", round(100 * max(_num)))
put("tokGenLo", round(100 * min(_gen))); put("tokGenHi", round(100 * max(_gen)))
put("tokEmpty", len(_empty)); put("tokTokenizers", len(set(TOKZ.values())))

# Section 7's sentiment paragraph, Section 3's partial-cut sentence and Remark 4 on partial cuts (the sentiment direction
# re-run with the authors' code on seven models x three starts, experiments/results/score_sentiment_reread.json). The scorer's
# readings: published_value, the published Pythia-1.4b value lies inside our band at the selected layer; above_100, a model significantly above
# 100% (every start); Y3, the network's own interchange at the adjective and verb positions (a partial cut), per layer
SR = json.load(open(figdata.ROOT / "experiments/results/score_sentiment_reread.json"))["models"]
_published_value = SR["pythia-1.4b"]["published_value"]; assert _published_value["consistent_with"], "published_value"
# one interval per object: the band that Table 10 and Figure 9(c) show for this cell (published_value's own interval differs by
# resampling scheme)
_b14 = SR["pythia-1.4b"]["block2"]["mean"]
assert _b14["lower"] <= _published_value["published"] <= _b14["upper"], "the published value lies inside the band as well"
put("srPubOurs", f"{_b14['Y_star']:.1f}"); put("srPubLo", f"{_b14['lower']:.1f}"); put("srPubHi", f"{_b14['upper']:.1f}")
_ab = SR["pythia-410m"]["block2"]                                        # the highest value, DAS's in Pythia-410m
assert all(_ab[k]["against_100"] == "indistinguishable" for k in ("mean", "seed0", "seed1", "seed2")), "the text: its band includes 100%"
put("srPythiaY", f"{_ab['mean']['Y_star']:.1f}"); put("srPythiaLo", f"{_ab['mean']['lower']:.1f}"); put("srPythiaHi", f"{_ab['mean']['upper']:.1f}")
# the reading rests on the band's small-sample factor: the same band without it (plain_band) lies above 100% in every start of
# Pythia-410m and changes no other model's reading; the starts lie so close that the two not seen when the band was chosen
# cannot test that choice
_pl = lambda x: "above" if x[0] > 100 else ("below" if x[1] < 100 else "indistinguishable")  # noqa: E731
assert all(_pl(_ab[k]["plain_band"]) == "above" for k in ("mean", "seed0", "seed1", "seed2")), "the text: without the factor, above 100%"
assert all(_pl(v["block2"][k]["plain_band"]) == v["block2"][k]["against_100"] for m, v in SR.items() if m != "pythia-410m" and "block2" in v
           for k in ("mean", "seed0", "seed1", "seed2")), "the text: the factor changes no other model's reading"
put("srPythiaPlainLo", f"{_ab['mean']['plain_band'][0]:.1f}"); put("srPythiaPlainHi", f"{_ab['mean']['plain_band'][1]:.1f}")
_ys = [_ab[f"seed{k}"]["Y_star"] for k in range(3)]
assert max(_ys) - min(_ys) < 0.5, "the text: the starts lie within half a point"
put("srPythiaYLo", f"{min(_ys):.1f}"); put("srPythiaYHi", f"{max(_ys):.1f}")
_vd = {m: v["block2"]["mean"]["against_100"] for m, v in SR.items() if "block2" in v}
assert "above" not in _vd.values(), "the text: no direction lies above 100% once the layer's choice is counted"
put("srModels", len(_vd)); put("srIndist", sum(x == "indistinguishable" for x in _vd.values())); put("srBelow", sum(x == "below" for x in _vd.values()))
# Section 7.3: the models whose band includes 100% in every start with and without the small-sample factor (every one but
# Pythia-410m); and the Bonferroni reading (Table 10's Bonferroni-t) includes 100% at Pythia-410m only through the same
# correction: with the normal quantile and no sqrt(u/(u-1)) its lower bound lies above 100% in every start
_ks = ("mean", "seed0", "seed1", "seed2")
_ind2 = [m for m, v in SR.items() if "block2" in v and all(v["block2"][k]["against_100"] == "indistinguishable"
                                                          and _pl(v["block2"][k]["plain_band"]) == "indistinguishable" for k in _ks)]
assert "pythia-410m" not in _ind2 and len(_ind2) == macros["srIndist"] - 1, "the text: five models either way, Pythia-410m the exception"
put("srIndistPlain", len(_ind2))
from scipy.stats import norm as _norm, t as _student_t  # noqa: E402
_u410, _L410 = math.ceil(SR["pythia-410m"]["n_test"] / 2), len(SR["pythia-410m"]["layers"])
def _bonf_plain_lo(b):
    lo, hi = b["bonferroni"]; y = b["Y_star"]; assert abs((y - lo) - (hi - y)) < 1e-6, "Bonferroni bounds symmetric about the score"
    sd = (y - lo) / (_student_t.ppf(1 - 0.025 / _L410, _u410 - 1) * math.sqrt(_u410 / (_u410 - 1)))
    return y - _norm.ppf(1 - 0.025 / _L410) * sd
assert all(_pl(_ab[k]["bonferroni"]) == "indistinguishable" and _bonf_plain_lo(_ab[k]) > 100 for k in _ks), \
    "the text: the Bonferroni reading includes 100% at Pythia-410m only through its small-sample correction"
_o = SR["pythia-410m"]["own_E1_per_layer"]; _t2 = sorted(_o.items(), key=lambda kv: -kv[1]["own_E1"])[:2]
assert [m for m, v in SR.items() if any(x["ci"][0] > 100 for x in v.get("own_E1_per_layer", {}).values())] == ["pythia-410m"], "the text: only in Pythia-410m does the network's own interchange lie above 100%"
_L = sorted(int(L) for L, _ in _t2)                                     # a hidden state's index: after that many blocks
put("srOwnMax", f"{_t2[0][1]['own_E1']:.1f}"); put("srOwnSecond", f"{_t2[1][1]['own_E1']:.1f}")
put("srOwnLo", f"{min(v['ci'][0] for _, v in _t2):.0f}"); put("srOwnBlocks", f"{_L[0]} and {_L[1]}")
put("srOwnLayers", f"{_L[0] - 1} and {_L[1] - 1}")                    # the same states as layers (Section 2.2: layer l is the output of block l, from 0)
assert int(_t2[1][0]) < int(_t2[0][0]), "the text names the smaller value at the earlier layer"
# the authors' best-epoch reload returns the last epoch (Appendix E.3): the two rules' directions
# and values, at the selected layer and elsewhere
_cs, _ds, _do = [], [], []
for _m, _v in SR.items():
    _Ls = _v["block2"]["mean"]["L_star"]
    for _s in (0, 1, 2):
        for _L, _x in json.load(open(figdata.ROOT / f"experiments/results/sentiment_reread_{_m}_seed{_s}.json"))["layers"].items():
            if int(_L) == 0:
                continue
            (_cs.append(_x["cos_best_last"]) or _ds.append(abs(_x["das_E2"] - _x["last_das_E2"]))) if int(_L) == _Ls else \
                _do.append(abs(_x["das_E2"] - _x["last_das_E2"]))
put("srBLCosLo", f"{min(_cs):.3f}"); put("srBLDiffSel", f"{max(_ds):.1f}"); put("srBLDiffMax", f"{max(_do):.0f}")

# the readout's run time per model (App. B.5): the runs' recorded seconds summed per model
# (experiments/logs/readout_<model>_m_<group>_*/done.json, runs of experiments/scripts/readout.py only), with up to eight runs sharing the GPU
# and Qwen2.5-32B alone on a second one
_hrs = {}
for d_ in (figdata.ROOT / "experiments/logs").glob("readout_*/done.json"):
    nm_ = d_.parent.name[len("readout_"):]
    if "_m_" in nm_ and any(c_.endswith("experiments/scripts/readout.py") for c_ in json.load(open(d_.parent / "env.json"))["command"]):
        _hrs[nm_.split("_m_")[0]] = _hrs.get(nm_.split("_m_")[0], 0.0) + json.load(open(d_))["seconds"] / 3600
assert set(_hrs) == PAPER, sorted(set(_hrs) ^ PAPER)
_hr = lambda ms: rng([_hrs[m] for m in ms], "{:.0f}")  # noqa: E731
put("ruSmall", _hr(["gpt2", "pythia-160m", "pythia-410m", "SmolLM2-360M", "Qwen2.5-0.5B", "pythia-1b", "Llama-3.2-1B", "Qwen2.5-1.5B"]))
put("ruThree", f"{_hrs['Llama-3.2-3B']:.0f}"); put("ruLarge", _hr(["Llama-3.1-8B", "Qwen2.5-7B", "pythia-6.9b"]))
put("ruGemma", f"{_hrs['gemma-2-9b']:.0f}"); put("ruThirtyTwo", f"{_hrs['Qwen2.5-32B']:.0f}")
# the first-order ranking by source on the full grid (small_all_excess_rank.post.json's per_source: map-level intervals, not the
# construction-level ones; descriptive; Section 6.2): first order beats the per-map best of
# seven on CausalGym, IOI and number-only agreement, is even on Kumon's constructions and falls below it on RAVEL, SST and the
# sentiment templates; on RAVEL the layer alone ranks the excess better than first order
_psrc = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all_excess_rank.post.json"))["per_source"]
assert all(_psrc[k]["minus_best7"]["lo"] > 0 for k in ("causalgym", "ioi", "number_only")) \
       and _psrc["kumon"]["minus_best7"]["lo"] < 0 < _psrc["kumon"]["minus_best7"]["hi"] \
       and all(_psrc[k]["minus_best7"]["hi"] < 0 for k in ("ravel", "sentiment", "sst")), "the text: the ranking by source"
put("gridRavelRho", _f2(_psrc["ravel"]["theory_rho"]["mean"]))
for k_, K_ in (("ioi", "Ioi"), ("number_only", "NumOnly"), ("kumon", "Kumon"), ("ravel", "Ravel"), ("sentiment", "Sent"), ("sst", "Sst")):
    v_ = _psrc[k_]["minus_best7"]; _sg = lambda x: f"${x:+.3f}$"; put(f"gridSrc{K_}", f"{_sg(v_['mean'])} [{_sg(v_['lo'])}, {_sg(v_['hi'])}]")  # noqa: E731
_dep = _psrc["ravel"].get("minus", {}).get("depth")
assert _dep and _dep["hi"] < 0, "the text: on RAVEL the layer alone ranks the excess better than first order"

# word forms of small counts, for prose (macro name + "Word")
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()
put("snReuseWord", WORDS[len(_snt)])
for k in ("roModels", "roModelsFO", "roSources", "inModels", "inExcludedMaps", "gridModelsBelow", "kuUp",
          "oeMaps", "rkModels", "tokEmpty",
          "tokTokenizers", "srModels", "srIndist", "srIndistPlain", "srBelow"):
    if k in macros and isinstance(macros[k], int) and macros[k] < len(WORDS):
        put(k + "Word", WORDS[macros[k]])

lines = ["% Written by paper/scripts/numbers_main.py from experiments/results/score_readout.json and",
         "% experiments/results/readout_decomp/score_readout_decomp.json, experiments/results/score_readout_sentence.json,",
         "% experiments/results/score_adequacy.json and experiments/results/cap_mechanism.json, with the readout's and the inside maps'",
         "% per-pair files (experiments/results/readout/, experiments/results/inside/) and the rank sweep's run files, checked against",
         "% experiments/results/score_readout_rank.json. Do not edit.",
         f"% Models whose instruments held: {', '.join(OK)}."]
_fmt = lambda v: f"{v:,}".replace(",", "{,}") if isinstance(v, int) and not isinstance(v, bool) and v >= 10000 else v  # noqa: E731
lines += [f"\\newcommand{{\\{k}}}{{{_fmt(v)}}}" for k, v in macros.items()]
OUT.write_text("\n".join(lines) + "\n")
for k, v in macros.items():
    print(f"{k:26s} {v}")
print(f"written {OUT}")

"""CausalGym's maps rebuilt under its protocol, its 29 tasks in the six small models at every layer (Section 7.1 and panel (b) of
fig:benchmarks): the paper's numbers as macros (paper/sections/numbers_cgmap.tex) and the per-model table
(paper/sections/tab_causalgym_map.tex). Read from the scorer's output (experiments/results/score_causalgym_map.json,
experiments/score/score_causalgym_map.py) and the symmetric readings beside it (experiments/results/score_causalgym_map.post.json,
experiments/score/causalgym_map_post.py). Pythia-1b's negative-polarity row is read from its maps (the code's odd layers in
experiments/results/inside/, its even layers in experiments/results/inside_even/) with the maps' own resamples; the code's layer L is the
paper's L - 1. The asserts check the sentences the text builds on these numbers.
Usage: python -I paper/scripts/numbers_cgmap.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

S = json.load(open(ROOT / "experiments/results/score_causalgym_map.json"))
P = json.load(open(ROOT / "experiments/results/score_causalgym_map.post.json"))
MODELS = ["gpt2", "pythia-160m", "pythia-410m", "pythia-1b", "SmolLM2-360M", "Qwen2.5-0.5B"]
NAME = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "pythia-1b": "Pythia-1b",
        "SmolLM2-360M": "SmolLM2-360M", "Qwen2.5-0.5B": "Qwen2.5-0.5B"}
assert sorted(S["models"]) == sorted(MODELS) and not S["excluded"] and len(S["maps"]) == 174
mac = {}
f2 = lambda x: f"{x:.2f}"  # noqa: E731

# ---- CausalGym's aggregate on the rebuilt maps: R-bar, DAS over vanilla (the network's own single swap), over every map and
# per model (medians), and the maps whose interval lies above 1
Rb = {k: v["primary"]["Rbar"] for k, v in S["maps"].items()}
allR = np.array([v["pt"] for v in Rb.values()])
mac["cgmMaps"] = str(len(Rb))
mac["cgmRbarMed"], mac["cgmRbarQone"], mac["cgmRbarQthree"] = f2(np.median(allR)), f2(np.quantile(allR, .25)), f2(np.quantile(allR, .75))
mac["cgmRbarAboveOne"] = str(sum(v["ci"][0] is not None and v["ci"][0] > 1 for v in Rb.values()))
medR = {m: float(np.median([Rb[k]["pt"] for k in Rb if k.startswith(m + "/")])) for m in MODELS}
mac["cgmRbarModelLo"], mac["cgmRbarModelHi"] = f"{min(medR.values()):.1f}", f"{max(medR.values()):.1f}"
assert all(r > 1 for r in allR)

# ---- the lead over vanilla (DAS's best minus vanilla's best, summed over layers): its share beyond the network's own
# interchange in its best context G* at DAS's cells (s_N), and its share from off-route cells (s_off: the part of the lead
# from layers where G*'s interval at DAS's cell does not lie above 0; the scorer's holds marks a model whose s_off interval
# lies above 0)
sN = {m: S["models"][m]["s_N"] for m in MODELS}
mac["cgmSNlo"], mac["cgmSNhi"] = f2(min(v[0] for v in sN.values())), f2(max(v[0] for v in sN.values()))
assert all(v[0] > 1 for v in sN.values()), "the whole lead lies beyond the network's own at DAS's cells, in every model"
# G*'s context: the scorer's s_N takes it chosen on the test pairs (which can only raise G*), Section 6 on the training pairs;
# the same share with the training-chosen context, a ratio of sums over each model's maps (sum N / sum X reproduces s_N)
_ntr, _nte, _nsame = {}, {}, {}
for k, v in S["maps"].items():
    m = k.split("/")[0]; p_ = v["primary"]
    if m not in MODELS:
        continue
    a = _ntr.setdefault(m, [0.0, 0.0]); b = _nte.setdefault(m, [0.0, 0.0]); b[1] += p_["X"]["pt"]
    if p_["N"] and p_["N"]["pt"] is not None:
        b[0] += p_["N"]["pt"]
    if p_["s_N_train_context"] and p_["s_N_train_context"]["pt"] is not None:
        a[0] += p_["s_N_train_context"]["pt"] * p_["X"]["pt"]; a[1] += p_["X"]["pt"]
        c_ = _nsame.setdefault(m, [0.0, 0.0]); c_[0] += p_["N"]["pt"]; c_[1] += p_["X"]["pt"]   # the test context on the same maps
assert all(abs(_nte[m][0] / _nte[m][1] - sN[m][0]) < 1e-9 for m in MODELS), "sum N / sum X is the scorer's s_N"
_str = [_ntr[m][0] / _ntr[m][1] for m in MODELS]
assert min(_str) > 1, "the lead lies beyond G* with either context"
mac["cgmSNtrlo"], mac["cgmSNtrhi"] = f2(min(_str)), f2(max(_str))
# the training-context share leaves out the maps whose lead is not established (the scorer gives them no shares); on the same
# maps the share with the test-chosen context is never above the training one's (asserted)
_ssm = [_nsame[m][0] / _nsame[m][1] for m in MODELS]
mac["cgmSNsamelo"], mac["cgmSNsamehi"] = f2(min(_ssm)), f2(max(_ssm))
# Table 8's largest share, Pythia-160m's, rests on the one task whose T is near zero: the share without it, in percent as drawn
assert max(sN, key=lambda m: sN[m][0]) == "pythia-160m"
mac["cgmSNpOneSixty"] = f"{100 * sN['pythia-160m'][0]:.0f}"
mac["cgmSNpOneSixtySame"] = f"{100 * _nsame['pythia-160m'][0] / _nsame['pythia-160m'][1]:.0f}"
assert all(_nsame[m][0] / _nsame[m][1] <= _ntr[m][0] / _ntr[m][1] + 1e-12 for m in MODELS), "a higher G* leaves a smaller share"
# s_off per model: the models where it is small but above zero, the models whose s_off interval lies above 0, and
# Qwen2.5-0.5B's and the pooled share in the scorer's form and in the strict form (only where DAS also lies significantly
# above G*)
so = {m: S["models"][m]["s_off"] for m in MODELS}
small = [m for m in MODELS if 0 < so[m][0] < 0.025]
assert small == ["gpt2", "pythia-410m", "pythia-1b", "SmolLM2-360M"], small
mac["cgmSoffSmallMax"] = f2(max(so[m][0] for m in small))
mac["cgmSoffHolds"] = str(sum(S["models"][m]["s_off_above_0"] for m in MODELS))
assert [m for m in MODELS if not S["models"][m]["s_off_above_0"]] == ["pythia-160m"]
q = so["Qwen2.5-0.5B"]; qs = P["models"]["Qwen2.5-0.5B"]["off_lead_above_gstar"]
mac["cgmSoffQwen"], mac["cgmSoffQwenStrict"] = f2(q[0]), f2(qs[0])
pr, ps = P["pooled"]["off_lead"], P["pooled"]["off_lead_above_gstar"]
assert abs(pr[0] - S["pooled"]["s_off"][0]) < 1e-9
mac["cgmSoffPooled"], mac["cgmSoffPooledLo"], mac["cgmSoffPooledHi"] = f2(pr[0]), f2(pr[1]), f2(pr[2])
mac["cgmSoffPooledStrict"], mac["cgmSoffPooledStrictLo"], mac["cgmSoffPooledStrictHi"] = f2(ps[0]), f2(ps[1]), f2(ps[2])
# the task that dominates Qwen2.5-0.5B's off-route share in the scorer's form and has none in the strict form: its T, and the
# same task's T in Pythia-160m (below zero); then the median share of DAS's excess that freezing attention removes, per model
fg = P["per_map"]["Qwen2.5-0.5B/filler_gap_embed_4"]
assert fg["off_lead"] > 0.9 * sum(v["off_lead"] for k, v in P["per_map"].items() if k.startswith("Qwen2.5-0.5B/")) and fg["off_lead_above_gstar"] == 0
Zq = np.load(ROOT / "experiments/results/inside/inside_Qwen2.5-0.5B_filler_gap_embed_4.npz")
mac["cgmQwenFgT"] = f2(float(Zq["T"].mean()))
Zp = np.load(ROOT / "experiments/results/inside/inside_pythia-160m_filler_gap_embed_4.npz")
mac["cgmPythiaFgT"] = f2(float(Zp["T"].mean()))
assert float(Zp["T"].mean()) < 0
# shares are sums over a model's maps in units of each map's T, so a map with a small T weighs heavily; filler_gap_embed_4 has
# the smallest T in every model. The share at cells where G* is not above zero without that task (Qwen2.5-0.5B's and pooled), the
# part of Qwen2.5-0.5B's share it holds, and the per-map summaries the T-weighting does not touch: the median share beyond G*
# per model and the share where G* is not above zero (median 0 in every model; its mean per model)
PM = P["per_map"]
def _off(keys):
    return sum(PM[k]["off_lead"] for k in keys) / sum(PM[k]["X"] for k in keys)
_qk = [k for k in PM if k.startswith("Qwen2.5-0.5B/")]
assert abs(_off(_qk) - so["Qwen2.5-0.5B"][0]) < 1e-9 and abs(_off(list(PM)) - pr[0]) < 1e-9, "the per-map sums give the scorer's shares"
_nofg = lambda ks: [k for k in ks if not k.endswith("/filler_gap_embed_4")]  # noqa: E731
mac["cgmSoffQwenNoFg"], mac["cgmSoffPooledNoFg"] = f2(_off(_nofg(_qk))), f2(_off(_nofg(list(PM))))
mac["cgmQwenFgPart"] = f2(PM["Qwen2.5-0.5B/filler_gap_embed_4"]["off_lead"] / sum(PM[k]["X"] for k in _qk))
_pmN = {m: np.median([PM[k]["N"] / PM[k]["X"] for k in PM if k.startswith(m + "/") and PM[k]["X"] > 0]) for m in MODELS}
_pmO = {m: [PM[k]["off_lead"] / PM[k]["X"] for k in PM if k.startswith(m + "/") and PM[k]["X"] > 0] for m in MODELS}
assert all(np.median(v) == 0 for v in _pmO.values()), "per map, the share where G* is not above zero has median 0"
mac["cgmMapBeyondLo"], mac["cgmMapBeyondHi"] = f2(min(_pmN.values())), f2(max(_pmN.values()))
mac["cgmMapOffMeanLo"], mac["cgmMapOffMeanHi"] = f"{min(np.mean(v) for v in _pmO.values()):.3f}", f"{max(np.mean(v) for v in _pmO.values()):.3f}"
att = {m: S["models"][m]["through_attention_median"] for m in MODELS}
mac["cgmAttLo"], mac["cgmAttHi"] = f2(min(att.values())), f2(max(att.values()))

# ---- the check against the published values (the scorer's published_figure), on the ten tasks the released code reads correctly in the
# three published sizes we map (Pythia-160m, 410m, 1b): vanilla equal to the published value at its two decimals, the
# published DAS inside the 95% range of our single-start aggregates, and Pythia-1b's negative-polarity map aligned with the
# published figure (Spearman at no shift above both one-layer shifts)
pub = [p for m in MODELS for p in S["models"][m].get("published", [])]
mac["cgmPubN"] = str(len(pub))
mac["cgmPubVanEq"] = str(sum(round(p["vanilla_ours"], 2) == round(p["vanilla"], 2) for p in pub))
assert max(abs(p["vanilla_ours"] - p["vanilla"]) for p in pub) < 0.015                      # the rest within 0.01 of the rounding
mac["cgmPubDasIn"] = str(sum(p["das_inside_single_start_range"] for p in pub))
al = S["published_figure"]["alignment_pythia_1b_npi"]
assert al["aligned"]
mac["cgmPubRho"], mac["cgmPubRhoShift"] = f2(al["no_shift"]), f2(max(al["ours_one_later"], al["ours_one_earlier"]))

# ---- Pythia-1b's negative-polarity map (Section 7.1 and App. E.1, "How the route test reads CausalGym's map", and fig:atlas):
# the aggregate against the published one, and the licensor row, the paper's layers 0 to the last where DAS's best region is
# the licensor (region 1)
npi = S["maps"]["pythia-1b/npi_any_subj-relc"]["primary"]
mac["cgmNpiRbar"], mac["cgmNpiRbarLo"], mac["cgmNpiRbarHi"] = f2(npi["Rbar"]["pt"]), f2(npi["Rbar"]["ci"][0]), f2(npi["Rbar"]["ci"][1])
pubn = next(p for p in S["models"]["pythia-1b"]["published"] if p["task"] == "npi_any_subj-relc")
mac["cgmNpiPub"] = f2(pubn["R_published"])
assert mac["cgmNpiRbar"] == "2.36" and mac["cgmNpiPub"] == "2.38"
cells = sorted(npi["per_layer"].items(), key=lambda kv: int(kv[0]))
lic = [int(L) - 1 for L, v in cells if v["cell"][1] == 1]
mac["cgmNpiLicLast"] = str(max(lic))
assert lic == list(range(max(lic) + 1)), "DAS's best region is the licensor from layer 0 to its last licensor layer"
mac["cgmNpiAtt"] = f2(npi["through_attention"]["pt"])
m, t = "pythia-1b", "npi_any_subj-relc"
J1 = json.load(open(ROOT / f"experiments/results/inside/inside_{m}_{t}.json")); Z1 = np.load(ROOT / f"experiments/results/inside/inside_{m}_{t}.npz")
J2 = json.load(open(ROOT / f"experiments/results/inside_even/inside_{m}_{t}_even.json")); Z2 = np.load(ROOT / f"experiments/results/inside_even/inside_{m}_{t}_even.npz")
Tm = float(Z1["T"].mean())
# the licensor's cells (region 1) by the paper's layer: DAS's effect (three starts), G* and DAS with attention frozen, over T
row = {}
for J, Z in ((J1, Z1), (J2, Z2)):
    for c in J["cells"]:
        if c["region"] == 1 and c.get("live", True) and f"L{c['layer']}r1/gstar_train" in Z.files:
            k = f"L{c['layer']}r1"
            row[c["layer"] - 1] = dict(e=float(np.mean([Z[f"{k}/e/das_ce_s{s}_normal"] for s in range(3)], axis=0).mean() / Tm),
                                       gs=float(Z[f"{k}/gstar_train"].mean() / Tm),   # G* as Section 6 defines it: the context chosen on the training pairs
                                       fr=float(np.mean([Z[f"{k}/e/das_ce_s{s}_frozen"] for s in range(3)], axis=0).mean() / Tm))
first = 6                                                                    # from here the network carries little (asserted)
mid = [row[L] for L in range(first, max(lic) + 1)]
mac["cgmNpiFrom"] = str(first)
mac["cgmNpiDasLo"], mac["cgmNpiDasHi"] = f"{min(r['e'] for r in mid):.1f}", f"{max(r['e'] for r in mid):.1f}"
mac["cgmNpiGsMax"] = f2(max(r["gs"] for r in mid))
mac["cgmNpiFrozenRemoves"] = f"{100 * min(1 - r['fr'] / r['e'] for r in mid):.0f}"
assert max(r["gs"] for r in mid) < 0.1 and all(r["gs"] >= 0.25 for L, r in row.items() if L < first - 1)
# the network's own route on the same map: the region where vanilla (its own single swap) is largest, by the paper's layer
# (regions 1, 3, 7 and 8 are the licensor, "that", "has" and "caught")
G = {}
for J, Z in ((J1, Z1), (J2, Z2)):
    for c in J["cells"]:
        NLm = max(c_["layer"] for c_ in J1["cells"])
        if c["layer"] < NLm and c.get("live", True):
            G[(c["layer"] - 1, c["region"])] = float(Z[f"L{c['layer']}r{c['region']}/e/own_normal"].mean() / Tm)
route = [max(range(1, 9), key=lambda r: G.get((L, r), -9)) for L in range(max(L for L, _ in G) + 1)]
assert route == [1] * 4 + [3] + [7] * 4 + [8] * (len(route) - 9), route          # licensor, that, has, caught
mac["cgmRouteLicLast"], mac["cgmRouteThat"], mac["cgmRouteHasLast"], mac["cgmRouteCaughtFrom"] = "3", "4", "8", "9"
assert all(abs(round(p["vanilla_ours"], 2) - p["vanilla"]) <= 0.0101 for p in pub)          # the rest differ by 0.01
gpt = S["maps"]["gpt2/npi_any_subj-relc"]["primary"]["Rbar"]["pt"]
mac["cgmGptNpiRbar"] = f2(gpt)

# ---- write the macros
out = "% Written by paper/scripts/numbers_cgmap.py from experiments/results/score_causalgym_map.json and its readings; do not edit.\n"
out += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items())
(ROOT / "paper/sections/numbers_cgmap.tex").write_text(out)

# ---- the per-model table (tab:causalgymmap, in Section 7.1)
iv = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"  # noqa: E731
rows = []
for mo in MODELS:
    pm = P["models"][mo]; pubm = S["models"][mo].get("published", [])
    published_match = (f"{sum(round(p['vanilla_ours'], 2) == round(p['vanilla'], 2) for p in pubm)}/{len(pubm)}, "
          f"{sum(p['das_inside_single_start_range'] for p in pubm)}/{len(pubm)}") if pubm else "--"
    offc = (f"{iv(so[mo])} & {iv(pm['off_lead_above_gstar'])}" if S["models"][mo]["s_off_above_0"] else
            r"\multicolumn{2}{c}{not established$^a$}")
    rows.append(f"    {NAME[mo]} & {medR[mo]:.2f} & {iv(sN[mo])} & {offc} & {att[mo]:.2f} & {published_match} \\\\")
tab = r"""\begin{table}[tbp]
  \caption{\textbf{CausalGym's maps rebuilt under its protocol, per model} (29 tasks, every layer). $\bar R$: CausalGym's
  aggregate, DAS divided by vanilla, as a median over tasks. The gap is the aggregate of DAS minus that of vanilla, summed
  over layers. Shares are ratios of sums over the tasks of a model, with intervals that resample constructions (a task
  together with its \texttt{\_mod} twin). Beyond $G^*$: the excess of DAS over $G^*$, the effect of whole-state patching in its best context (chosen on the test pairs), at the best region of DAS in each layer, as a share of the gap. $G^*$ not above 0: the
  share at cells where the interval of $G^*$ does not lie above zero. Off-route: the same share, counted only where DAS also
  lies significantly above $G^*$. Through attention: the median share of the excess of DAS that freezing attention removes.
  Published: of the ten tasks that the released code reads correctly, the tasks where vanilla equals the published value to
  its two decimals, and the tasks where the published DAS lies within the 95\% resampling range of our single-initialisation
  aggregates; -- where CausalGym publishes no values for the model. $^a$One task, \texttt{filler\_gap\_embed\_4}, whose $T$ is $\cgmPythiaFgT$, dominates both sums of
  Pythia-160m. The same task holds \cgmQwenFgPart{} of Qwen2.5-0.5B's \cgmSoffQwen{} (\cgmSoffQwenNoFg{} without it). Shares
  are sums in units of each map's $T$, so a map with a small $T$ weighs heavily. Per map, the median share beyond $G^*$ is
  \cgmMapBeyondLo--\cgmMapBeyondHi, and the share where $G^*$ is not above 0 has median 0 in every model.}
  \label{tab:causalgymmap}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lrrrrrr@{}}
    \toprule
    model & $\bar R$ & beyond $G^*$ & \begin{tabular}[b]{@{}c@{}}$G^*$ not\\above 0\end{tabular} & off-route & \begin{tabular}[b]{@{}c@{}}through\\attention\end{tabular} & published \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
(ROOT / "paper/sections/tab_causalgym_map.tex").write_text(tab)
print("numbers_cgmap: " + ", ".join(f"{k}={v}" for k, v in mac.items()))

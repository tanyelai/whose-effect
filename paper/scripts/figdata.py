"""Shared by the paper's figure scripts: the repository's root and results folder, and Figure 1's run with the rule that picks
it (fig1_check). Every other number a figure draws is read by the figure's own script from the scorers' outputs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT / "experiments/results"

# Figure 1's run, picked by a rule that follows from the figure's purpose: among the one-question tasks of the readout's main
# population whose question any reader can answer without linguistics (RAVEL's attributes, the sentiment templates, SST, IOI;
# not CausalGym's or Kumon's syntax), asked as a plain sentence (no list, JSON or "Conclusion:" prompt) with a minimal pair
# (base and source differ in one span), and defined at the sentence level (experiments/results/score_readout_sentence.json), the model
# and task whose cross-entropy DAS effect lies nearest the median over the one-question tasks of every model whose final logits
# are not capped. SST's sentences are paired at random, so they are not minimal pairs even when a whole review is one span.
# fig1_check() recomputes the rule and stops a script whose FIG1 is not its pick. Panel (a) draws one test pair of two widely
# known cities as the front card of the task's test pairs; no number depends on it (every number is a mean over the test pairs).
# The prompt opens with one example, "People in Cape Town speak Afrikaans, English, Xhosa."
FIG1 = dict(model="Llama-3.2-3B", group="ravel_attributes", task="ravel_city_language", base="Coventry", src="Bogota", split="test")
FIG1_NAME = "Llama-3.2-3B"
_READABLE = ("ravel_attributes", "sentiment_templates", "sst", "ioi_ours", "ioi_ours_label_signed", "ioi_wang", "ioi_wang_label_signed")
_NOT_MINIMAL = ("sst",)                                    # the data appendix: SST's sentences are paired at random


def fig1_check(S=None):
    """Recompute Figure 1's rule on experiments/results/score_readout.json and its sentence-level file: the model and the task are chosen
    together, over every uncapped model of the readout's main population. Return the median, the pick's percentile among the
    one-question tasks of the uncapped models, their number, and the candidates as (distance, model, group, task, ce/T), nearest
    first (ties: the smaller distance at full precision, then the model's label in alphabetical order)."""
    S = S or json.load(open(GPU / "score_readout.json"))
    SS = json.load(open(GPU / "score_readout_sentence.json"))["models"]
    ok = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"
          and not any(r.get("raw_over_cap") for r in S["rows"][m])]
    allce = sorted(r["excess"]["ce_p1"]["pt"] for m in ok for r in S["rows"][m]
                   if r["main"] and r["defined"] and r["one_q"] and r.get("angle", {}).get("P"))
    med = allce[len(allce) // 2] if len(allce) % 2 else 0.5 * (allce[len(allce) // 2 - 1] + allce[len(allce) // 2])
    cands = []
    for m in ok:
        sdef = {(t["group"], t["task"]): t["defined"] for t in SS[m]["tasks"]}
        for r in S["rows"][m]:
            if not (r["main"] and r["defined"] and r["one_q"] and r["group"] in _READABLE and sdef.get((r["group"], r["task"]))):
                continue
            J = json.load(open(GPU / f"readout/readout_{m}_m_{r['group']}.json"))
            f = ROOT / J["data"] / r["task"] / "test.json"
            if not f.is_file():
                continue
            rows = json.load(open(f))
            plain = all(not any(c in "".join(x["base"]) + "".join(x["src"]) for c in ";{\":") for x in rows)
            minimal = r["group"] not in _NOT_MINIMAL and all(sum(a != b for a, b in zip(x["base"], x["src"])) == 1
                                                            and len(x["base"]) == len(x["src"]) for x in rows)
            if plain and minimal:
                cands.append((abs(r["excess"]["ce_p1"]["pt"] - med), m, r["group"], r["task"], r["excess"]["ce_p1"]["pt"]))
    cands.sort(key=lambda c: (c[0], c[1]))
    pick = (FIG1["model"], FIG1["group"], FIG1["task"])
    assert cands and cands[0][1:4] == pick, f"Figure 1's rule picks {cands[:3]}, not {pick}"
    pct = 100.0 * sum(c <= cands[0][4] for c in allce) / len(allce)
    return dict(median=med, pct=pct, n=len(allce), cands=cands)

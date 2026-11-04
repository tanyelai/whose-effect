"""The readout per data source (tab:readout, in App. C.1): Section 5's main population over all fourteen models
(experiments/results/score_readout.json), so the last row is Section 5's counts (size within a family has its own table, tab:scale).
Per source: tasks (model x task), tasks with room (the out-of-sample top's interval above T), cross-entropy DAS above T among
them (its three starts' mean at one pass, interval above T), its median over T over the tasks, and the per-pair squared
error's median distance from T. Writes paper/sections/tab_readout.tex.
Usage: python -I paper/scripts/table_readout.py
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402

S = json.load(open(figdata.ROOT / "experiments/results/score_readout.json"))
OK = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]
SOURCES = ["CausalGym", "CausalGym number-only", "Kumon & Yanaka", "Tigges et al.", "SST", "RAVEL", "IOI (Wang et al.)", "IOI (ours)"]
NAME = {"CausalGym number-only": "CausalGym, number only", "Kumon & Yanaka": r"Kumon \& Yanaka", "Tigges et al.": "sentiment templates", "IOI (Wang et al.)": "IOI",
        "IOI (ours)": "IOI, our frame"}
rows = [r for m in OK for r in S["rows"][m] if r["main"] and r["defined"]]
assert len(OK) == 14, OK


def cells(rs):
    """The five columns for the rows rs (dashes when there are none)."""
    if not rs:
        return ["--"] * 6
    room = [r for r in rs if r["room"]]
    ce = [r["excess"]["ce_p1"]["pt"] for r in rs]; sq = [r["objective"]["p1"]["sq_T"] for r in rs]
    return [str(len({(r["group"], r["task"]) for r in rs})), str(len(rs)), str(len(room)), f"{sum(r['excess']['ce_p1']['above'] for r in room)}", f"{st.median(ce):.2f}",
            f"{100 * st.median(abs(x - 1) for x in sq):.1f}"]


head = "source & tasks & $n$ & margin & ce $>T$ & ce$/T$ & sq\\%"
lines = [r"\begin{table}[tbp]",
         r"  \caption{\textbf{Results at the readout, per data source.} The main population over the fourteen models. "
         r"tasks: distinct tasks; $n$: model--task cases; margin: cases with a margin above $T$, where the interval of the "
         r"out-of-sample maximum lies above $T$; ce $>T$: cases among these where the interval of cross-entropy DAS lies above $T$; "
         r"ce$/T$: the median over the $n$ cases of the effect of cross-entropy DAS divided by $T$; sq\%: the median distance of the "
         r"per-pair squared error from $T$, in percent. The rows of IOI split its tasks by their label pairs. With one label pair, "
         r"every pair contrasts the same two names. With many, the maximum lies below $T$ at rank one (\cref{app:counter}). The "
         r"cases with a margin above $T$ where cross-entropy DAS is not above $T$ are one-directional negative-polarity "
         r"constructions, to whose base label the models give little probability (\cref{sec:shortfall}), and Pythia-160m's "
         r"sentiment classification. In that task, one initialisation lags behind after one pass, and cross-entropy reaches the "
         r"maximum after ten passes (\cref{app:shortfallfull}).}",
         r"  \label{tab:readout}", r"  \centering", r"  \footnotesize",
         r"  \begin{tabular}{@{}lrrrrrr@{}}", r"    \toprule", f"    {head} \\\\"]
lines.append(r"    \midrule")
for src in SOURCES:
    # IOI's two regimes apart: one label pair (the same two names in every pair) and many (the top below T at rank one)
    if src.startswith("IOI"):
        many = max(r["label_pairs"] for r in rows if r["source"] == src)
        parts = [(f"{NAME[src]}, one label pair", lambda r: r["label_pairs"] == 1),
                 (f"{NAME[src]}, {many} label pairs", lambda r: r["label_pairs"] > 1)]
    else:
        parts = [(NAME.get(src, src), lambda r: True)]
    for name, keep in parts:
        lines.append(f"    {name} & " + " & ".join(cells([r for r in rows if r["source"] == src and keep(r)])) + r" \\")
_parts = [l.split(" & ")[1] for l in lines if l.startswith("    ") and " & " in l and l.split(" & ")[1].isdigit()]
assert sum(map(int, _parts)) == len({(r["group"], r["task"]) for r in rows}), "each task in one row"
lines.append(r"    \midrule")
lines.append("    all & " + " & ".join(cells(rows)) + r" \\")
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
out = figdata.ROOT / "paper/sections/tab_readout.tex"
out.write_text("% Written by paper/scripts/table_readout.py from experiments/results/score_readout.json; do not edit.\n" + "\n".join(lines) + "\n")
print("\n".join(lines[-14:]))

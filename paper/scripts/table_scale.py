"""The two tables of size within a family (tab:scale and tab:scalesources, in Appendix C.7), from experiments/results/score_scale.json.

The first table, the tests per family, one row per step: the median over the family's common tasks of the per-task change
from the smallest size to the middle one (s1->s2) and to the largest (s1->s3), and Qwen2.5's step from 7B to 32B, for the room
(rho = out-of-sample top / T), the reported excess (r = cross-entropy DAS / T) and cross-entropy's share of the room
(sigma = r / rho), as percentages, exp(D) - 1, with 95% task-bootstrap intervals. The caption states the outcomes, checked
here against the score: for the room and the excess (the score's room_grows and excess_grows), both increases' intervals above zero and
no fall from the middle size to the largest by its interval; for the share (share_falls), its decrease to the largest size below
zero. The second table (descriptive): the same changes per data source, s1->s3, without intervals.
Usage: python -I paper/scripts/table_scale.py [--out=<path>]
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), ROOT / "paper/sections/tab_scale.tex"))
S = json.load(open(ROOT / "experiments/results/score_scale.json"))["families"]
FAM = (("Pythia", ("410m", "1b", "6.9b")), ("Qwen2.5", ("0.5B", "1.5B", "7B")), ("Llama", ("3.2-1B", "3.2-3B", "3.1-8B")))


def pct(x):
    return 100 * (math.exp(x) - 1)


def num(x):
    return f"{x:+.1f}" if abs(x) >= 0.1 else f"{x:+.2f}"     # a bound near zero keeps its sign and size visible


def cells(d):
    """A change and its interval, side by side in two columns (the change right-aligned, the interval left-aligned)."""
    return f"${num(pct(d['pt']))}$ & $[{num(pct(d['lo']))}, {num(pct(d['hi']))}]$"


# Qwen2.5-32B: one step, from 7B to 32B, over the tasks the two share (the score's "qwen32")
Q = json.load(open(ROOT / "experiments/results/score_scale.json"))["qwen32"]
assert Q["sizes"] == ["Qwen2.5-7B", "Qwen2.5-32B"], Q["sizes"]
# the outcomes, per family and for the step to 32B: the caption states them, so they are checked here
held = {fam: dict(room=S[fam]["room_grows"], excess=S[fam]["excess_grows"], share=S[fam]["share_falls"]) for fam, _ in FAM}
held["Qwen2.5, 7B to 32B"] = dict(room=Q["room_grows"], excess=Q["excess_grows"], share=Q["share_falls"])
failed = [(f, k) for f, v in held.items() for k, ok in v.items() if not ok]
assert failed == [("Llama", "excess")], f"the caption names the outcomes that failed: {failed}"
_L = S["Llama"]; _l1, _l2 = _L["sizes"][:2]
assert _L["D"][f"r:{_l1}->{_l2}"]["lo"] <= 0 < _L["D"][f"r:{_l1}->{_L['sizes'][2]}"]["lo"], "the caption: Llama's first step"
assert not any(S[f]["middle_guard_fired"][k] for f, _ in FAM for k in ("rho", "r")), "the caption: no fall from the middle size"

lines = ["% Written by paper/scripts/table_scale.py from experiments/results/score_scale.json; do not edit.",
         "\\begin{table}[tbp]",
         "  \\caption{\\textbf{Model size within a family (the scale series).} For each family, the table gives the median over"
         " its common tasks of the per-task log ratio $x$, shown as the change in percent that it implies, $100(e^x - 1)$. The"
         " changes run from the smallest of its three sizes to the middle one and to the largest, and in Qwen2.5 from 7B to 32B,"
         " with 95\\% task-bootstrap intervals. The columns give the out-of-sample maximum effect over $T$ (max.), the effect of"
         " cross-entropy DAS over $T$ (DAS) and the ratio of the two, the share of the maximum that cross-entropy reaches (share)."
         " The test asks that the maximum and DAS grow at both steps (both intervals above zero) and that the share fall to the"
         " largest size (its interval below zero). All three hold in every family and in the step to 32B, except for Llama's DAS,"
         " whose interval for the first step includes zero. In no family does the maximum or DAS fall from the middle size to the"
         " largest by its interval.}",
         "  \\label{tab:scale}",
         "  \\centering",
         "  \\footnotesize",
         "  \\begin{tabular}{@{}lrlrlrl@{}}",
         "    \\toprule",
         "    & \\multicolumn{2}{c}{max.} & \\multicolumn{2}{c}{DAS} & \\multicolumn{2}{c}{share} \\\\",
         "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
         "    step & change & interval & change & interval & change & interval \\\\",
         "    \\midrule"]
for k, (fam, sizes) in enumerate(FAM):
    F = S[fam]; s1, s2, s3 = F["sizes"]; D = F["D"]
    assert all(t.endswith("-" + z) for z, t in zip(sizes, (s1, s2, s3))), (sizes, F["sizes"])   # the printed sizes are the scored ones
    tasks = f"{F['common_tasks']} tasks" + (f"; {Q['common_tasks']} for the step to 32B" if fam == "Qwen2.5" else "")
    lines += ([] if k == 0 else ["    \\addlinespace[3pt]"]) + [f"    \\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{fam}}} ({tasks})}} \\\\"]
    for a, b in ((s1, s2), (s1, s3)):
        za, zb = sizes[(s1, s2, s3).index(a)], sizes[(s1, s2, s3).index(b)]
        lines.append(f"    \\quad {za} $\\to$ {zb} & {cells(D[f'rho:{a}->{b}'])} & {cells(D[f'r:{a}->{b}'])} & "
                     f"{cells(D[f'sigma:{a}->{b}'])} \\\\")
    if fam == "Qwen2.5":                                   # the family's step from its largest size to 32B, on the tasks the two share
        qd = lambda k: Q["D"][f"{k}:Qwen2.5-7B->Qwen2.5-32B"]  # noqa: E731
        lines.append(f"    \\quad 7B $\\to$ 32B & {cells(qd('rho'))} & {cells(qd('r'))} & {cells(qd('sigma'))} \\\\")
lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

sources = sorted({so for fam, _ in FAM for so in S[fam].get("per_source", {})})
lines += ["\\begin{table}[tbp]",
          "  \\caption{\\textbf{Where the growth with size comes from} (descriptive, no intervals). For each data source, the"
          " table gives the median per-task change from the smallest to the largest size of each family, in percent, for the"
          " maximum effect (max.) and for cross-entropy DAS. The number of tasks per source is given in parentheses. RAVEL's rows"
          " pass a filter in each model, so its test rows differ across sizes.}",
          "  \\label{tab:scalesources}",
          "  \\centering",
          "  \\footnotesize",
          "  \\setlength{\\tabcolsep}{4pt}",
          "  \\begin{tabular}{@{}l" + "rr" * len(FAM) + "@{}}",
          "    \\toprule",
          "    & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{fam}}}" for fam, _ in FAM) + " \\\\",
          "    source & " + " & ".join("max. & DAS" for _ in FAM) + " \\\\",
          "    \\midrule"]
for so in sources:
    cells = []
    for fam, _ in FAM:
        x = S[fam]["per_source"].get(so)
        cells += ([f"${pct(x['rho13']):+.0f}$ ({x['n']})", f"${pct(x['r13']):+.0f}$"] if x else ["--", "--"])
    lines.append(f"    {so.replace('&', chr(92) + '&')} & " + " & ".join(cells) + " \\\\")
lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
OUT.write_text("\n".join(lines) + "\n")
print("written", OUT)

"""The table of CausalGym's published scores over its own vanilla intervention, at all nine published Pythia sizes
(tab:causalgym, in Section 7.1, beside panel (a) of fig:benchmarks), and the numbers the text quotes from them
(paper/sections/numbers_causalgym.tex).

Reads experiments/causalgym/causalgym_published_tables.json (experiments/causalgym/parse_published_tables.py) and
experiments/causalgym/index_shift_exposure.json (exposure 0: the ten tasks whose base and source positions coincide, which the
released code reads correctly). Every task (the Average row excluded), with no filter. Medians of method / vanilla over the 29
tasks per size; the three smallest sizes publish one probe. Against vanilla, a score is above it, below it or within the
tables' rounding: every published value has two decimals, so each is known to +-0.005 and a difference of two to +-0.01; with
d the difference in hundredths, "above" is d >= 2 and "below" d <= -2 (the reference 1 is the network's own interchange). A
sensitivity keeps only the entries with vanilla >= 0.5. Also written: the selectivity ratio (DAS's selectivity over
vanilla's, Section 7.1) and the mean difference over vanilla on the ten aligned tasks and on the other 19 (Appendix E.2).
Usage: python -I paper/scripts/table_causalgym.py <repo root>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
pub = json.load(open(ROOT / "experiments/causalgym/causalgym_published_tables.json"))
exp = json.load(open(ROOT / "experiments/causalgym/index_shift_exposure.json"))
aligned = {t for t, r in exp.items() if r["exposure"] == 0}
SIZES = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
ROWS = [("DAS", ["das"]), ("probe, first setting", ["probe0"]), ("probe, second setting", ["probe1"]), ("mean difference", ["mean"]), ("PCA", ["pca"]),
        ("$k$-means", ["kmeans"]), ("LDA", ["lda"]), ("random", ["rand"])]
OTHERS = ("probe0", "probe1", "mean", "pca", "kmeans", "lda", "rand")
assert all(abs(round(v * 100) - v * 100) < 1e-6 for k, tab in pub.items() if k.endswith("|main") for r in tab.values()
           for v in r.values() if isinstance(v, (int, float))), "a published value with more than two decimals"


def side(m, v):
    """+1 above vanilla beyond the tables' rounding, -1 below it, 0 within it (differences in hundredths)."""
    d = round(m * 100) - round(v * 100)
    return 1 if d >= 2 else (-1 if d <= -2 else 0)


def count(filt):
    """Over the task-size entries that pass filt(row): DAS's sides and medians per size, the other methods' sides."""
    out = dict(das=[], das_al=[], oth=[], oth_al=[], exceed=[], med={name: [] for name, _ in ROWS}, das_size=[])
    for s in SIZES:
        tab = {t: r for t, r in pub[f"pythia-{s}|main"].items() if t in exp and filt(r)}
        for name, keys in ROWS:
            out["med"][name].append([float(np.median([r[k] / r["vanilla"] for r in tab.values()])) for k in keys
                                     if k in next(iter(tab.values()))])
        sz = [side(r["das"], r["vanilla"]) for r in tab.values()]
        out["das_size"].append((sum(x == 1 for x in sz), len(sz)))
        for t, r in tab.items():
            out["das"].append(side(r["das"], r["vanilla"])); (t in aligned) and out["das_al"].append(out["das"][-1])
            for k in OTHERS:
                if k in r:
                    x = side(r[k], r["vanilla"]); out["oth"].append(x); (t in aligned) and out["oth_al"].append(x)
                    if x == 1:
                        out["exceed"].append(r[k] / r["vanilla"])
    return out


A, F = count(lambda r: True), count(lambda r: r["vanilla"] >= 0.5)
seld, mean_al, mean_ot, mean_al_f, mean_ot_f = [], [], [], [], []
for s in SIZES:
    tab = {t: r for t, r in pub[f"pythia-{s}|main"].items() if t in exp}; st = pub[f"pythia-{s}|sel"]
    seld.append(float(np.median([st[t]["das"] / st[t]["vanilla"] for t in tab if t in st and st[t]["vanilla"] > 0])))
    mean_al.append(np.median([r["mean"] / r["vanilla"] for t, r in tab.items() if t in aligned]))
    mean_ot.append(np.median([r["mean"] / r["vanilla"] for t, r in tab.items() if t not in aligned]))
    mean_al_f.append(np.median([r["mean"] / r["vanilla"] for t, r in tab.items() if t in aligned and r["vanilla"] >= 0.5]))
    mean_ot_f.append(np.median([r["mean"] / r["vanilla"] for t, r in tab.items() if t not in aligned and r["vanilla"] >= 0.5]))
assert [s for s, v in zip(SIZES, seld) if v > 1] == ["14m", "31m"], "appendix.tex: selectivity above 1 only at 14m and 31m"
assert all(x == 1 for x in A["das_al"]), "published.tex: DAS above vanilla on all the ten tasks' entries"
das_med = [m[0] for m in A["med"]["DAS"]]; fdas_med = [m[0] for m in F["med"]["DAS"]]
up = lambda xs: sum(x == 1 for x in xs)  # noqa: E731
# the mean difference over vanilla per size, median over the ten tasks and over the 19 (Appendix E.2): the
# printed ranges hold with or without the entries whose vanilla is below 0.5
r2 = lambda xs: (f"{min(xs):.2f}", f"{max(xs):.2f}")  # noqa: E731
assert r2(mean_al) == r2(mean_al_f) and r2(mean_ot) == r2(mean_ot_f), "appendix.tex: the same range with or without vanilla < 0.5"
mac = dict(cgDasMedLo=f"{min(das_med):.1f}", cgDasMedHi=f"{max(das_med):.1f}", cgDasAbove=up(A["das"]), cgEntries=len(A["das"]),
           cgDasBelow=sum(x == -1 for x in A["das"]), cgDasAboveAligned=up(A["das_al"]), cgEntriesAligned=len(A["das_al"]),
           cgOtherAbove=up(A["oth"]), cgOtherEntries=len(A["oth"]), cgOtherAboveAligned=up(A["oth_al"]),
           cgOtherEntriesAligned=len(A["oth_al"]), cgOtherWithin=sum(x == 0 for x in A["oth"]),
           cgOtherExceedMed=f"{np.median(A['exceed']):.2f}", cgOtherExceedQthree=f"{np.quantile(A['exceed'], 0.75):.2f}",
           cgFiltDasMedLo=f"{min(fdas_med):.1f}", cgFiltDasMedHi=f"{max(fdas_med):.1f}", cgFiltEntries=len(F["das"]),
           cgFiltOtherAbove=up(F["oth"]), cgFiltOtherEntries=len(F["oth"]), cgSelLo=f"{min(seld):.2f}", cgSelHi=f"{max(seld):.2f}",
           cgMeanAlLo=r2(mean_al)[0], cgMeanAlHi=r2(mean_al)[1], cgMeanOtLo=r2(mean_ot)[0], cgMeanOtHi=r2(mean_ot)[1])
tex = [r"\begin{table}[tbp]",
       r"  \caption{\textbf{CausalGym's published scores divided by the score of its vanilla intervention} (whole-state patching scored in",
       r"  the same way, in the same units). The scores come from CausalGym's per-size tables \citep[App.~E.2]{arora2024}.",
       r"  The table gives medians over its 29 tasks at all nine published sizes. Of its seven methods, the probe has two settings,",
       r"  and the three smallest sizes publish only one of them (--). The last row gives the number of tasks per size in which DAS",
       r"  lies above vanilla beyond the rounding of the tables. Every value has two decimals, so a difference is known to",
       r"  $\pm 0.01$. The other six methods lie above vanilla in \cgOtherAbove{} of their \cgOtherEntries{} method--task--size",
       r"  entries, by a median of \cgOtherExceedMed$\times$. On the ten tasks that the released code reads correctly",
       r"  (\cref{app:indexswap}), they do so in \cgOtherAboveAligned{} of \cgOtherEntriesAligned{} entries. Over the entries with",
       r"  vanilla $\ge 0.5$ alone, the medians of DAS are \cgFiltDasMedLo--\cgFiltDasMedHi, and the other methods lie above vanilla",
       r"  in \cgFiltOtherAbove{} of \cgFiltOtherEntries{} entries.}",
       r"  \label{tab:causalgym}",
       r"  \centering",
       r"  \footnotesize",
              r"  \begin{tabular}{@{}l" + "r" * len(SIZES) + "@{}}",
       r"    \toprule",
       "    method / vanilla & " + " & ".join(SIZES) + r" \\",
       r"    \midrule"]
for name, _ in ROWS:
    tex.append(f"    {name} & " + " & ".join(", ".join(f"{m:.2f}" for m in ms) or "--" for ms in A["med"][name]) + r" \\")
tex += [r"    \midrule", r"    DAS above vanilla & " + " & ".join(f"{a}/{n}" for a, n in A["das_size"]) + r" \\", r"    \bottomrule",
        r"  \end{tabular}", r"\end{table}"]
(ROOT / "paper/sections/tab_causalgym.tex").write_text("\n".join(tex) + "\n")
(ROOT / "paper/sections/numbers_causalgym.tex").write_text(
    "% Written by paper/scripts/table_causalgym.py from CausalGym's published tables. Do not edit.\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print("wrote paper/sections/tab_causalgym.tex and numbers_causalgym.tex:", json.dumps(mac))
print("selectivity, DAS over vanilla per size: " + ", ".join(f"{s} {v:.3f}" for s, v in zip(SIZES, seld)) + f" -> {min(seld):.2f}-{max(seld):.2f}")
print("mean / vanilla, ten aligned tasks: " + ", ".join(f"{s} {v:.2f}" for s, v in zip(SIZES, mean_al)) + f" -> {min(mean_al):.2f}-{max(mean_al):.2f}")
print("mean / vanilla, other 19: " + ", ".join(f"{s} {v:.2f}" for s, v in zip(SIZES, mean_ot)) + f" -> {min(mean_ot):.2f}-{max(mean_ot):.2f}")

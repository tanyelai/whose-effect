"""The NPI decomposition (fig:npidecomp, 0.6 text width, 238.5 pt): where the log-odds gain of the top and of cross-entropy DAS
comes from on Kumon & Yanaka's NPI constructions, one-directional and mirrored.

From experiments/results/readout_decomp/score_readout_decomp.json: per model (one row each, the text's names, in Figure 4(c)'s
order), the median over the eight NPI
constructions of the share of a direction's log-odds gain at the readout that comes from lowering the base label's logit,
-mean(dz_b) / mean(dz_s - dz_b), for the out-of-sample top and for cross-entropy DAS (its three starts' mean). Left column: the
one-directional sets (every base label " ever"); right: the same pairs mirrored. The marks are fig:readout (c)'s: an open dark
grey circle is a model's median top, a filled orange disc its median of cross-entropy DAS; a light grey line joins the two in
each row. No key: the first row's marks are named over them. Rows in Figure 4(c)'s order: by family, sizes from small to large. The axis runs from -0.05, so a median just below 0 (one-directional DAS in some models) is drawn where it lies; every
median is asserted to lie at or above -0.02. Subtitles: the top's range and the per-model comparisons, asserted equal to
numbers_main's macros (paper/sections/numbers_main.tex: npiModels, npiTopShare, npiGapMirrored).
Usage: python -I paper/scripts/fig_npi_decomp.py [--out=<dir>]
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, SIXW, WORD, L, T, head, linscale, render, tw  # noqa: E402

sys.path.insert(0, str(figdata.ROOT / "experiments/matrix"))
import registry  # noqa: E402     the model groups, in the registry's order

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
S = json.load(open(figdata.ROOT / "experiments/results/readout_decomp/score_readout_decomp.json"))
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (figdata.ROOT / "paper/sections/numbers_main.tex").read_text()))
NAMES = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "SmolLM2-360M": "SmolLM2-360M",   # the text's
         "Qwen2.5-0.5B": "Qwen2.5-0.5B", "pythia-1b": "Pythia-1b", "Llama-3.1-8B": "Llama-3.1-8B", "gemma-2-9b": "Gemma-2-9B",
         "Qwen2.5-7B": "Qwen2.5-7B", "pythia-6.9b": "Pythia-6.9b", "Qwen2.5-1.5B": "Qwen2.5-1.5B", "Llama-3.2-1B": "Llama-3.2-1B",
         "Llama-3.2-3B": "Llama-3.2-3B", "Qwen2.5-32B": "Qwen2.5-32B"}
# Figure 4(c)'s order: by family, each family's sizes from small to large, the two single models first, a space after each family
FAMILIES = (("gpt2", "SmolLM2-360M"), ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-6.9b"),
            ("Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-7B", "Qwen2.5-32B"), ("Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"), ("gemma-2-9b",))
models = [m for fam in FAMILIES for m in fam]
assert sorted(models) == sorted(m for m, v in S.items() if isinstance(v, dict)), "the rows and the scored models differ"
assert all(Path(h).name in models for h in registry.SMALL + registry.LARGE + registry.SCALE + registry.QWEN32)
assert str(len(models)) == NUM["npiModels"], "the figure's models and the text's (npiModels) differ"
med = {(m, key_, d): st.median(v[key_][d] for v in S[m]["per_task"].values())
       for m in models for key_ in ("one", "mirrored") for d in ("top", "ce")}
rng = lambda xs: f"{min(xs):.2f}–{max(xs):.2f}"  # noqa: E731
top1 = rng([med[m, "one", "top"] for m in models])
gapm = rng([abs(round(med[m, "mirrored", "top"] - med[m, "mirrored", "ce"], 2)) for m in models])
lower = sum(med[m, "one", "ce"] < med[m, "one", "top"] for m in models)
assert top1 == NUM["npiTopShare"].replace("--", "–"), "the top's range and the text's (npiTopShare) differ"
assert gapm == NUM["npiGapMirrored"].replace("--", "–"), "the mirrored gap and the text's (npiGapMirrored) differ"
assert lower == len(models), "the subtitle says DAS's share is lower in every model"

W = SIXW                                                                    # printed at 0.6 \textwidth
NX = 50.0                                                                   # the names' right edge
XLO = -0.05
# the one-directional column carries the finding, so it gets the width; mirrored, every share collapses to about one half
COLS = (("one-directional", "one", 56.0, 164.0), ("mirrored", "mirrored", 188.0, 232.0))
assert all(XLO + 0.03 <= v <= 1.0 for v in med.values()), "a median outside the columns' axis"
PITCH, GAP, RY0 = 7.6, 3.0, 54.0
fam_of = {m: k for k, fam in enumerate(FAMILIES) for m in fam}
ry = {m: RY0 + i * PITCH + GAP * fam_of[m] for i, m in enumerate(models)}
ybot = max(ry.values())
nw = WORD.get(len(models), str(len(models)))
b = head(0.0, "", "Cross-entropy seeks the source label's logit",
         [f"one-directional: maximum's share {top1}, DAS's lower in all {nw} models",
          f"mirrored: maximum and DAS {gapm} apart"])
for name, key_, c0, c1 in COLS:
    X = linscale(XLO, 1.0, c0, c1)
    b += T((X(0) + X(1)) / 2, RY0 - 15.0, name, FS["tick"], C["ink"], anchor="middle", weight=600)
    for v, t_ in ((0, "0"), (0.5, "½"), (1, "1")):
        b += L(X(v), RY0 - 3.0, X(v), ybot + 4.0, C["line"], 0.5)
        b += T(X(v), ybot + 11.6, t_, FS["tick"], C["gray"], anchor="middle")
    for m in models:
        xt, xc, y = X(med[m, key_, "top"]), X(med[m, key_, "ce"]), ry[m]
        if abs(xt - xc) > 3.6:                                              # the row's gap: a light line between the marks' edges
            s = 1 if xt > xc else -1
            b += L(xc + s * 2.0, y, xt - s * 1.9, y, C["line"], 0.6)
        b += f'<circle cx="{xc:.2f}" cy="{y:.2f}" r="1.8" fill="{C["das"]}"/>'
        b += f'<circle cx="{xt:.2f}" cy="{y:.2f}" r="1.6" fill="none" stroke="{C["tgray"]}" stroke-width="0.7"/>'
for m in models:
    b += T(NX, ry[m] + 2.1, NAMES[m], FS["tick"], C["ink"], anchor="end")
# the marks, named once over the first row's, in the one-directional column
X1 = linscale(XLO, 1.0, COLS[0][2], COLS[0][3]); m0 = models[0]
lab = [("cross-entropy DAS", X1(med[m0, "one", "ce"]), C["das"]), ("maximum", X1(med[m0, "one", "top"]), C["tgray"])]
for t_, x_, col in lab:
    b += T(x_, RY0 - 5.0, t_, FS["tick"], col, anchor="middle", weight=600)
assert lab[0][1] + tw(lab[0][0], FS["tick"], 600) / 2 + 4 < lab[1][1] - tw(lab[1][0], FS["tick"], 600) / 2, "the two names apart"
b += T((COLS[0][2] + COLS[1][3]) / 2, ybot + 19.8, "share of the log-odds gain from lowering the base label", FS["axis"], C["muted"],
       anchor="middle")
H = ybot + 23.0
render("fig_npi_decomp", W, H, b, OUT)
print("fig_npi_decomp: medians (one: top, ce; mirrored: top, ce)",
      {NAMES[m]: tuple(round(med[m, k, d], 3) for k in ("one", "mirrored") for d in ("top", "ce")) for m in models})
oth = [m for m in models if m != "gemma-2-9b"]
print(f"fig_npi_decomp: one-directional top {top1}, DAS lower in {lower} of {len(models)}; mirrored gap {gapm}; Gemma-2-9B "
      f"mirrored top {med['gemma-2-9b', 'mirrored', 'top']:.2f}, DAS {med['gemma-2-9b', 'mirrored', 'ce']:.2f}; the other models' "
      f"mirrored medians {rng([med[m, 'mirrored', d] for m in oth for d in ('top', 'ce')])}; height {H:.1f} pt")

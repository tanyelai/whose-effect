"""Appendix figure (fig:benchmarks, full text width, 397 pt; Section 7): what benchmark scores measure. Panel (a) spans the
width; (b) and (c) share the row below it.

(a) CausalGym's published scores over its vanilla intervention (the network's own whole-state interchange at the same site),
    one dot per task-size entry (all 261 of the 29 tasks x 9 Pythia sizes, none filtered on vanilla), per method, in two plots on
    one axis: the ten tasks whose base and source tokenise alike up to the intervened regions, then the 19 where the released
    code reads a state at the other sentence's token index (Appendix E.2). A colour is a method in both plots; the plot says
    which tasks. Bars: each strip's median. The probe column takes the better of the benchmark's two probes; LDA and random are
    not drawn (tab:causalgym). The subtitle's count is over the four other methods drawn (the probe at the better of its two:
    1044 entries) and asks for a difference beyond the tables' rounding (two decimals: known to +-0.01), as
    paper/scripts/table_causalgym.py does. The text's count is over all six (both probes, LDA and random: 1740 entries;
    paper/sections/numbers_causalgym.tex, asserted equal to the count computed here). Triangles: scores beyond the axis (0.15
    to 12), at its end: below it only on the 19, above it DAS alone, where vanilla is near 0 at the smallest sizes. The median
    bar and the triangles are named once, beside DAS's strip of the ten; the vanilla line at its right end. Source:
    experiments/causalgym/causalgym_published_tables.json, experiments/causalgym/index_shift_exposure.json.
(b) CausalGym's aggregate (best region per layer, averaged over layers), rebuilt on our maps of its 29 tasks at every layer in
    the six small models (experiments/results/score_causalgym_map.json), over vanilla's: per model the median over its tasks,
    of DAS (orange) and of G* at DAS's chosen cell in each layer (blue; the scorer's P is the sum over layers of
    that G* minus vanilla's best, so its aggregate is V + P / layers), against vanilla at 1. G* there gives what vanilla's best
    region gives (0.98 to 1.02, asserted), so the gap between DAS and vanilla lies beyond G*. Not a ratio of sums: one task
    whose T is near 0 dominates those (the text's s_N, tab:causalgymmap). DAS's medians are the text's cgmRbarModelLo/Hi
    (paper/sections/numbers_cgmap.tex; asserted equal).
(c) The sentiment direction of Tigges et al. re-run with the authors' code on the GPU in seven models
    (experiments/results/score_sentiment_reread.json; last-epoch rule, every position of the selected layer, a complete cut; the mean
    of three starts, with the band that counts the layer's choice, circular block bootstrap, block 2), and the published 109.8%
    (Pythia-1.4b; diamond) on its own row under our re-run of that model. 100% is the network's own interchange there. The
    subtitle's counts are asserted equal to numbers_main's (paper/sections/numbers_main.tex: srModels, srIndist, srBelow).
Every printed number is computed here; the ones the text quotes are asserted equal to it.
Each panel: a title (the finding), its counts as a subtitle; no key: each mark is named where it is drawn.
Colours: orange DAS, teal representation-only methods, blue the network's own interchange (vanilla, G*), dark grey the
published value.
Usage: python -I paper/scripts/fig_benchmarks.py [--out=<dir>]
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, MK, NIPSW, L, R, T, head, linscale, render, tw, xaxis  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
RES = figdata.ROOT / "experiments/causalgym"
pub = json.load(open(RES / "causalgym_published_tables.json"))
exp = json.load(open(RES / "index_shift_exposure.json"))
SIZES = ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
W = NIPSW                                                                   # printed at \textwidth
mac = lambda name: dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (figdata.ROOT / f"paper/sections/{name}.tex").read_text()))  # noqa: E731
b = ""


def tri_up(x, y, col, h=1.4):
    """A score beyond the axis's upper end: a small triangle pointing up, its tip on the axis."""
    return f'<polygon points="{x - h:.2f},{y + 2.2:.2f} {x + h:.2f},{y + 2.2:.2f} {x:.2f},{y:.2f}" fill="{col}"/>'


def tri(x, y, col, h=1.4):
    """A score beyond the axis's lower end: a small triangle pointing down, its tip on the axis."""
    return f'<polygon points="{x - h:.2f},{y - 2.2:.2f} {x + h:.2f},{y - 2.2:.2f} {x:.2f},{y:.2f}" fill="{col}"/>'


def diamond(x, y, col, h=2.4):
    return f'<polygon points="{x:.2f},{y - h:.2f} {x + h:.2f},{y:.2f} {x:.2f},{y + h:.2f} {x - h:.2f},{y:.2f}" fill="{col}"/>'


# ------------------------------------------------------------------ (a) per method, the ten tasks and the 19 apart
METHODS = [("das", "DAS", C["das"]), ("probe", "probe", C["nat"]), ("mean", "mean diff.", C["nat"]), ("pca", "PCA", C["nat"]),
           ("kmeans", "k-means", C["nat"])]
aligned = sorted(t for t, r in exp.items() if r["exposure"] == 0)
shifted = sorted(t for t, r in exp.items() if r["exposure"] > 0)
assert (len(aligned), len(shifted)) == (10, 19), "the plots and the caption name the ten tasks and the 19"
vals, vals_sh = {m: [] for m, *_ in METHODS}, {m: [] for m, *_ in METHODS}
for s in SIZES:
    tab = pub[f"pythia-{s}|main"]
    for tasks, store in ((aligned, vals), (shifted, vals_sh)):
        for t in tasks:
            r = tab.get(t)
            if not r:
                continue
            for m, *_ in METHODS:
                v = max(r["probe0"], r.get("probe1", r["probe0"])) if m == "probe" else r[m]
                store[m].append(v / r["vanilla"])
n_entries = len(vals["das"]) + len(vals_sh["das"])                           # every task-size entry, none filtered


def side(m, v):
    """+1 above vanilla beyond the tables' rounding (two decimals: a difference is known to +-0.01), -1 below, 0 within."""
    d = round(m * 100) - round(v * 100)
    return 1 if d >= 2 else (-1 if d <= -2 else 0)


over = ncell = 0                                                             # all six other methods (both probes, LDA, random)
for s in SIZES:
    for t, r in pub[f"pythia-{s}|main"].items():
        if t in exp:
            for k in ("probe0", "probe1", "mean", "pca", "kmeans", "lda", "rand"):
                if k in r:
                    ncell += 1; over += side(r[k], r["vanilla"]) == 1
over_d = n_d = 0                                                             # the four drawn: the probe at the better of its two
for s in SIZES:
    for t, r in pub[f"pythia-{s}|main"].items():
        if t in exp:
            for m in ("probe", "mean", "pca", "kmeans"):
                n_d += 1; over_d += side(max(r["probe0"], r.get("probe1", r["probe0"])) if m == "probe" else r[m], r["vanilla"]) == 1
assert n_d == 4 * n_entries
# the text's numbers (Section 7 and tab:causalgym) come from paper/scripts/table_causalgym.py: the figure must agree
MAC = mac("numbers_causalgym")
assert (over, ncell, n_entries) == (int(MAC["cgOtherAbove"]), int(MAC["cgOtherEntries"]), int(MAC["cgEntries"])), (over, ncell, n_entries)
TOP, BOT = 12.0, 0.15                                                        # the axis; a score beyond it is a triangle at its end
assert all(v >= BOT for m in vals for v in vals[m]), "every score below the axis is on the 19"
assert all(v <= TOP for m in vals if m != "das" for v in vals[m] + vals_sh[m]), "only DAS's scores reach above the axis"
n_top = sum(v > TOP for v in vals["das"] + vals_sh["das"])
das_up = sum(side(r["das"], r["vanilla"]) == 1 for s in SIZES for t, r in pub[f"pythia-{s}|main"].items() if t in exp)
assert das_up == int(MAC["cgDasAbove"]), das_up                              # the text's count for DAS
SUBA = ["CausalGym's published scores over vanilla, its whole-state patching: one dot per task and size",
        f"above vanilla beyond rounding: DAS in {das_up} of {n_entries} entries, the four others in {over_d} of their {n_d}"]
assert all(tw(t_, FS["sub"]) < W - 10 for t_ in SUBA), SUBA
b += head(0, "a", "Only DAS consistently exceeds whole-state patching", SUBA)
# two plots on one axis: the ten tasks where the released code's swap changes nothing, then the 19 where it matters; a colour is
# a method in both, the plot says which tasks
AX0, XR, GX, AY1, AY0 = 26.0, W - 25.0, 14.0, 43.0, 133.0                  # the axis's left, the plots' right end, the gap, top, bottom
PW = (XR - AX0 - GX) / 2
lo, hi = math.log(BOT), math.log(TOP)
Y = lambda v: AY0 - (math.log(min(max(v, BOT), TOP)) - lo) / (hi - lo) * (AY0 - AY1)    # noqa: E731
cw = PW / len(METHODS)
SJ, SB = 7.0, 7.5                                                            # a strip's jitter, its median bar's half-width
for v in (0.25, 0.5, 2, 4, 8):
    b += T(AX0 - 3, Y(v) + 2.0, f"{v:g}", FS["tick"], C["gray"], anchor="end")
b += T(AX0 - 3, Y(1) + 2.0, "1", FS["tick"], C["net"], anchor="end", weight=700)
b += T(6.5, (AY0 + AY1) / 2, "score / vanilla", FS["axis"], C["muted"], anchor="middle", rotate=-90)
rng = np.random.default_rng(3)
for h_, (store, title) in enumerate(((vals, "the ten tasks where base and source positions coincide"),
                                     (vals_sh, "the 19 tasks where the released code's swap matters"))):
    x0 = AX0 + h_ * (PW + GX)
    assert tw(title, FS["axis"], 600) < PW, title
    b += T(x0 + PW / 2, AY1 - 6.0, title, FS["axis"], C["ink"], anchor="middle", weight=600)
    b += L(x0, AY0, x0 + PW, AY0, C["line"], 0.6)
    for i, (m, lab, col) in enumerate(METHODS):
        xc = x0 + cw * (i + 0.5)
        for v in store[m]:
            x = xc + rng.uniform(-SJ, SJ)
            b += (tri(x, AY0 - 0.2, col) if v < BOT else tri_up(x, AY1 + 0.2, col) if v > TOP else
                  f'<circle cx="{x:.2f}" cy="{Y(v):.2f}" r="{MK["r"]}" fill="{col}" fill-opacity="0.75"/>')
        b += L(xc - SB, Y(float(np.median(store[m]))), xc + SB, Y(float(np.median(store[m]))), C["ink"], 1.1, cap="butt")
        assert tw(lab, FS["axis"], 600) < cw - 2, lab
        b += T(xc, AY0 + 8.6, lab, FS["axis"], col, anchor="middle", weight=600)
b += L(AX0, Y(1), XR, Y(1), C["net"], 0.9)
b += T(XR + 4.0, Y(1) + 2.1, "vanilla", FS["note"], C["net"], weight=600)
assert XR + 4.0 + tw("vanilla", FS["note"], 600) <= W
# the two marks a reader has not met, named once beside DAS's strip of the ten, in the empty space over the probe's
xd = AX0 + cw / 2 + SB + 3.0
assert max(vals["probe"]) < 2 and float(np.median(vals["das"])) > 2.2, "the labels sit where the probe has no point"
b += T(xd, Y(float(np.median(vals["das"]))) + 2.0, "median", FS["note"], C["muted"])
b += T(xd, AY1 + 4.4, "beyond the axis", FS["note"], C["muted"])

# ------------------------------------------------------------------ (b) on our maps: G* at DAS's own cells against vanilla, and DAS
yb = 158.0
BW = 186.0                                                                   # (b) spans 0..BW, (c) the rest of the row
SB_ = json.load(open(figdata.ROOT / "experiments/results/score_causalgym_map.json"))
BMOD = ["gpt2", "pythia-160m", "pythia-410m", "pythia-1b", "SmolLM2-360M", "Qwen2.5-0.5B"]
BNAME = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "pythia-1b": "Pythia-1b",
         "SmolLM2-360M": "SmolLM2-360M", "Qwen2.5-0.5B": "Qwen2.5-0.5B"}
assert sorted(SB_["models"]) == sorted(BMOD) and len(SB_["maps"]) == 174
recs = {m: [v["primary"] for k, v in SB_["maps"].items() if k.startswith(m + "/") and v["instruments"]] for m in BMOD}
assert all(len(r) == 29 for r in recs.values())
medR = {m: float(np.median([q["Rbar"]["pt"] for q in recs[m]])) for m in BMOD}                    # DAS's aggregate over vanilla's
# G* at DAS's chosen cell in each layer, aggregated as CausalGym aggregates a method (the mean over layers), over vanilla's: the
# scorer's P is the sum over layers of (that G* - vanilla's best), so this aggregate is V + P / layers
medG = {m: float(np.median([(q["V"]["pt"] + q["P"]["pt"] / q["layers"]) / q["V"]["pt"] for q in recs[m]])) for m in BMOD}
MB = mac("numbers_cgmap")
assert (f"{min(medR.values()):.1f}", f"{max(medR.values()):.1f}") == (MB["cgmRbarModelLo"], MB["cgmRbarModelHi"])
assert all(0.98 < g < 1.02 for g in medG.values()), medG                     # G* there gives what vanilla's best region gives
b += head(0, "b", "The DAS\u2013vanilla gap lies beyond G*",
          ["CausalGym's aggregate on our maps of its 29 tasks,", "G* at DAS's best region in each layer; median per model"], y=yb)
BX0 = 54.0
BX = linscale(0.0, 2.4, BX0, BX0 + 2.4 * 40.0)
line_ = lambda x0, y0, x1, y1: L(x0, y0, x1, y1, C["gray"], 0.6)  # noqa: E731
bh, y0b, pb = 2.6, yb + 35.0, 11.0
for k, m in enumerate(BMOD):
    yy = y0b + pb * k
    b += T(BX0 - 4, yy + 2.2, BNAME[m], FS["axis"], C["ink"], anchor="end")
    b += R(BX(0), yy - bh - 0.3, BX(medG[m]) - BX(0), bh, C["net"], r=0.4)
    b += R(BX(0), yy + 0.3, BX(medR[m]) - BX(0), bh, C["das"], r=0.4)
    if k == 0:                                                               # each bar named once, beside it
        b += T(BX(0), yy - bh - 2.4, "G*", FS["note"], C["net"], weight=600)        # over its bar's start, as in Figure 6
        b += T(BX(medR[m]) + 3.0, yy + 3.8, "DAS", FS["note"], C["das"], weight=600)
        # the title's gap, once, in the first row's blue lane, where G* would have to reach: from vanilla to DAS's end
        gx0_, gx1_, gy = BX(1) + 1.2, BX(medR[m]) - 0.3, yy - bh / 2 - 0.3
        b += line_(gx0_, gy, gx1_, gy) + line_(gx1_, gy - 1.4, gx1_, gy + 1.4)
        b += T((gx0_ + gx1_) / 2, gy - 3.0, "gap", FS["note"], C["muted"], anchor="middle")
        assert (gx0_ + gx1_) / 2 - tw("gap", FS["note"]) / 2 > BX(1) + tw("vanilla", FS["note"], 600) / 2 + 3
ya = y0b + pb * (len(BMOD) - 1) + bh + 3.5
b += L(BX(1), y0b - 6.0, BX(1), ya, C["net"], 0.9)                           # vanilla, every model's 1
b += T(BX(1), y0b - 8.4, "vanilla", FS["note"], C["net"], anchor="middle", weight=600)
b += xaxis(BX(0), BX(2.4), ya, BX, [(0, "0\u00d7"), (1, "1\u00d7"), (2, "2\u00d7")], "aggregate / vanilla")

# ------------------------------------------------------------------ (c) the sentiment direction at a complete cut
x0c = 211.0
# the re-run on the GPU (experiments/results/score_sentiment_reread.json, experiments/score/score_sentiment_reread.py): per model the mean of
# three starts at the selected layer, with the band that counts the layer's choice (block bootstrap, block 2); the published
# value is Tigges et al.'s (their Table 1a), and the scorer's published_value records whether our band holds it
SR = json.load(open(figdata.ROOT / "experiments/results/score_sentiment_reread.json"))["models"]
ORDER = ["gpt2", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "SmolLM2-360M", "Qwen2.5-0.5B"]
assert sorted(ORDER) == sorted(SR) and all(SR[m]["built"] and SR[m]["instruments_hold"] for m in ORDER)
mean = {m: SR[m]["block2"]["mean"] for m in ORDER}
published = SR["pythia-1.4b"]["published_value"]["published"]
verd = {m: mean[m]["against_100"] for m in ORDER}
n_in, n_above, n_below = (sum(v == w for v in verd.values()) for w in ("indistinguishable", "above", "below"))
MS = mac("numbers_main")
# the text (Section 7.3): the band includes 100% in six models and lies below it in one; none lies above
# it; the published value is consistent with our re-run
assert (len(ORDER), n_in, n_below, n_above) == (int(MS["srModels"]), int(MS["srIndist"]), int(MS["srBelow"]), 0)
assert published == 109.8 and SR["pythia-1.4b"]["published_value"]["consistent_with"] and f"{mean['pythia-1.4b']['Y_star']:.1f}" == MS["srPubOurs"]
NAME = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "pythia-1b": "Pythia-1b", "pythia-1.4b": "Pythia-1.4b",
        "SmolLM2-360M": "SmolLM2-360M", "Qwen2.5-0.5B": "Qwen2.5-0.5B"}
ROWS = []                                                                    # (name, value, band or None for the published value)
for m in ORDER:
    ROWS.append((NAME[m], mean[m]["Y_star"], (mean[m]["lower"], mean[m]["upper"])))
    if m == "pythia-1.4b":
        ROWS.append(("published, 1.4b", published, None))                    # right under our re-run of the same model
b += head(x0c, "c", f"The published {published:.1f}% is within the spread",
          [f"DAS re-run in {len(ORDER)} models: three runs' mean and 95% band;",
           f"the band holds 100% in {n_in}, lies below it in {n_below}, above in none"], y=yb)
SX0, SX1 = x0c + 56.0, x0c + 160.0
SX = linscale(0.0, 300.0, SX0, SX1)
yy0, pc = yb + 35.0, 9.5
yl = yy0 + (len(ROWS) - 1) * pc
b += L(SX(100), yy0 - 5.0, SX(100), yl + 5.0, C["net"], 0.9)
b += T(SX(100), yy0 - 8.0, "whole-state patching", FS["note"], C["net"], weight=600, anchor="middle")
for k, (name, v, ci) in enumerate(ROWS):
    yy = yy0 + k * pc
    b += T(SX0 - 6, yy + 2.1, name, FS["note"], C["ink"] if ci else C["tgray"], anchor="end")
    if ci:
        assert ci[1] < 300.0
        b += L(SX(ci[0]), yy, SX(ci[1]), yy, C["das"], 1.4) + f'<circle cx="{SX(v):.2f}" cy="{yy:.2f}" r="2.1" fill="{C["das"]}"/>'
    else:
        b += diamond(SX(v), yy, C["tgray"])
    b += T(W, yy + 2.1, f"{v:.1f}%", FS["note"], C["muted"] if ci else C["tgray"], anchor="end")
ya_c = yl + 6.0
b += L(SX0, ya_c, SX1, ya_c, C["line"], 0.6)
for v in (0, 100, 200, 300):
    b += T(SX(v), ya_c + 7.6, f"{v}%", FS["tick"], C["net"] if v == 100 else C["gray"], anchor="middle", weight=700 if v == 100 else 400)
b += T((SX0 + SX1) / 2, ya_c + 16.2, "logit-difference score", FS["axis"], C["muted"], anchor="middle")
H = max(ya + 19.0, ya_c + 19.0)
render("fig_benchmarks", W, H, b, OUT)
print(f"fig_benchmarks: (a) {n_entries} task-size entries ({len(vals['das'])} on the ten, {len(vals_sh['das'])} on the 19), medians "
      f"ten {({m: round(float(np.median(v)), 2) for m, v in vals.items()})}, 19 {({m: round(float(np.median(v)), 2) for m, v in vals_sh.items()})}; "
      f"other methods above vanilla beyond the tables' rounding in {over} of {ncell} entries; DAS above the axis {n_top}; "
      f"below it {sum(v < BOT for m in vals_sh for v in vals_sh[m])} (all on the 19); "
      f"(b) DAS {medR}, G* {medG}; (c) {ROWS}; height {H:.1f}")

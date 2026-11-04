"""The visual guide's map figure (fig:guideb, one sentence through the network), in Qwen2.5-0.5B, in two steps, in
fig:guidea's vocabulary: an effect is a distance on the answer's scale, from 0 (the base run's answer) towards T (the source
run's), blue for whole-state patching, orange for DAS. No box stands for a scale: boxes are states, as in fig:guidea.

1. Patch one cell, two ways: the verb at layer 10, in a small copy of fig:guidea's network, its whole state (G, fig:guidea's
   quantity) or only DAS's direction; the two moves of the answer as two bars on fig:guidea's scale (0, T, 2T, 3T).
2. Every cell of the movie review's map (every second layer and the readout; the words before the adjective are left out,
   since the change of the input never reaches them), each the same two bars, from its column's 0, on one scale with a
   faint line at T in every column (and 2T, 3T in the last word's, which the readout's DAS reaches). The few numbers drawn
   are fig:guidea's (layer 2's 0.50 and 0.49), step 1's cell (outlined) and the readout's DAS.
The reference is G, whole-state patching at the cell, as fig:guidea defines it and as Section 6 opens; the counts are the
paper's: DAS above G by per-cell intervals after false-discovery control over the cells of each model (Benjamini-Hochberg,
q = 0.025, experiments/results/inside_sets/small_all.json's above_own_G), for this map and for all maps (inCeAboveG of inLive,
asserted). Off-route cells are not drawn as a class: in this sentence DAS's effect there is below 0.05 T. Every number is a mean
over the task's 100 test pairs, over T; DAS is the mean of its three trained runs.
Usage: python -I paper/scripts/fig_guide_b.py [--out=<dir>]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
sys.path.insert(0, str(figdata.ROOT / "experiments/lib"))
from guidedata import map_cells  # noqa: E402
from measure import bh  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, R, T, arrows, circ, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, TASK = "Qwen2.5-0.5B", "sentiment_movie"
cells, _, LAYS, _ = map_cells(MODEL, TASK)
RO = max(LAYS)
LAYERS = [l for l in LAYS if l != RO]                                       # the code's layers: the paper's layer is l - 1
REGS = [1, 2, 3, 4]                                                         # adjective, ", I", verb, the last word
G = lambda l, r: cells[(l, r)]["own"]  # noqa: E731   (whole-state patching at the cell, fig:guidea's G)
D = lambda l, r: cells[(l, r)]["das"]  # noqa: E731
assert set(cells) == {(l, r) for l in LAYERS for r in REGS} | {(RO, 4)}, "every live cell drawn"
assert abs(G(RO, 4) - 1) < 1e-3 and abs(G(3, 1) - 0.50) < 0.005 and abs(G(3, 3) - 0.49) < 0.005, "fig:guidea's T, 0.50, 0.49"
assert all(G(l, 4) < 0.1 for l in LAYERS if l - 1 <= 12) and all(G(l, 4) > 0.5 for l in LAYERS if l - 1 >= 14), "last word from 14"
assert max(max(G(l, r), D(l, r)) for (l, r) in cells if r != 4) < 1.0, "the first three words stay under T"
# the counts, by the paper's rule
SI = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.json"))
POST = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json")); POP = set(POST["cell_classes"])
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (figdata.ROOT / "paper/sections/numbers_main.tex").read_text()))
mc = [c for c in SI["cells"][MODEL] if f"{MODEL}/{c['task']}/{c['cell']}" in POP]
sig = bh([c["above_own_G"]["p"] for c in mc], 0.025)
N_MAP = sum(1 for c in mc if c["task"] == TASK); N_ABOVE = sum(s for c, s in zip(mc, sig) if c["task"] == TASK)
assert N_MAP == len(cells), "the map's cells and the population's"
ALL_ABOVE, ALL_LIVE = 0, 0                                                 # every model's cells, false discovery per model
for m_, cs_ in SI["cells"].items():
    cs_ = [c for c in cs_ if f"{m_}/{c['task']}/{c['cell']}" in POP]
    ALL_ABOVE += int(sum(bh([c["above_own_G"]["p"] for c in cs_], 0.025))); ALL_LIVE += len(cs_)
fmt = lambda n: f"{n:,}"  # noqa: E731
assert fmt(ALL_ABOVE).replace(",", "{,}") == NUM["inCeAboveG"] and fmt(ALL_LIVE).replace(",", "{,}") == NUM["inLive"], "the text's counts"
row = json.load(open(figdata.ROOT / f"experiments/data/{MODEL}/sentiment_templates/{TASK}/test.json"))[0]
wb = [w.replace("<|endoftext|>", "").replace("\n", "").strip() for w in row["base"]]
ws = [w.replace("<|endoftext|>", "").replace("\n", "").strip() for w in row["src"]]

W = NIPSW
FN = FS["axis"]
mm = lambda v: f"{v:.2f}"  # noqa: E731
def stepmark(x, y, n, title):
    return circ(x + 6, y - 2.4, 5.0, C["ink"]) + T(x + 6, y, str(n), FN, C["paper"], anchor="middle", weight=700) + \
        T(x + 14.5, y, title, FS["title"], weight=600)
def bars(x0, y, tpt, g, d, hb=3.2):
    """A cell's two moves of the answer, from its 0 at x0, T = tpt points: whole-state patching (blue) over DAS (orange)."""
    o = R(x0, y - hb - 0.4, tpt * max(g, 0), hb, C["net"], r=0.5) if g > 0 else ""
    return o + (R(x0, y + 0.4, tpt * max(d, 0), hb, C["das"], r=0.5) if d > 0 else "")
def dividers(cx, cy, bw, bh, col, seg=5):
    return "".join(L(cx - bw / 2 + k * bw / seg, cy - bh / 2 + 0.9, cx - bw / 2 + k * bw / seg, cy + bh / 2 - 0.9, col, 0.4,
                     cap="butt") for k in range(1, seg))

b = arrows(("ag", C["gray"]), size=4.2)
b += T(0, 10, "Inside the network: where whole-state patching moves the answer, and what DAS reports there", FS["letter"], weight=600)
NLAY = RO                                                                  # the code's readout index: the model's number of blocks
b += T(0, 19.6, f"{MODEL} on the same movie review, every second of its {NLAY} layers; every number is a mean over the task's 100 "
       "test pairs", FS["sub"], C["muted"])

# ---- 1: one cell, patched two ways, on the answer's scale
Y1 = 38.0
TT1 = f"At one cell, the whole state moves the answer {mm(G(11, 3))} T, and DAS's direction {mm(D(11, 3))} T"
assert tw(TT1, FS["title"], 600) < W - 15, TT1
b += stepmark(0, Y1, 1, TT1)
b += T(14.5, Y1 + 8.6, "the verb at layer 10, patched two ways: all of its state, or only its part along DAS's direction",
       FS["sub"], C["muted"])
BW, BHX = 13.0, 5.6                                                         # the four words the map patches, and three layers
WORDS = [wb[1], wb[2].split()[-1], wb[3], wb[4].split()[-1]]                # each region's last word: sweet, I, cherished, is
assert WORDS == ["sweet", "I", "cherished", "is"], WORDS
nxs = [40.0, 64.0, 94.0, 134.0]                                              # "is" set apart: the words before it are not drawn
NY = Y1 + 20
LROWS = [("readout", RO), ("layer 22", 23), ("layer 10", 11), ("layer 0", 1)]
nys = {code: NY + k * 10.5 for k, (_, code) in enumerate(LROWS)}
for i, cx in enumerate(nxs):                                                # dotted: the layers between are not drawn
    b += L(cx, nys[RO] if i == 3 else nys[23], cx, nys[1], C["grayl"], 0.6, dash="0.6 1.6")
HI = (2, 11)                                                                # the verb at layer 10
for lab, code in LROWS:
    b += T(nxs[0] - BW / 2 - 4, nys[code] + 2.1, lab, FS["tick"], C["muted"], anchor="end")
    for i, cx in enumerate(nxs):
        if code == RO and i != 3:
            continue
        on = (i, code) == HI
        b += R(cx - BW / 2, nys[code] - BHX / 2, BW, BHX, C["paper"], r=1.2, stroke=C["ink"] if on else C["grayl"], sw=0.9 if on else 0.45)
        b += dividers(cx, nys[code], BW, BHX, C["gray"] if on else C["grayl"])
yw = nys[1] + 11.5
for i, w in enumerate(WORDS):
    b += T(nxs[i], yw, w, FS["tick"], C["ink"], anchor="middle", weight=600)
b += T((nxs[2] + nxs[3]) / 2 + 4, yw, "\u2026", FS["tick"], C["muted"], anchor="middle")
b += T(nxs[0] - 17, yw, "\u2026", FS["tick"], C["muted"], anchor="middle")
AX0, AX1, GMAX = 172.0, 390.0, 3.25
TP1 = (AX1 - AX0) / GMAX
yb = nys[HI[1]]                                                             # the bars level with the patched cell
# from the patched cell to its bars, arching over the last word's state at the same layer (a straight arrow would run through it)
xs_, xe_ = nxs[HI[0]] + BW / 2 + 3, AX0 - 7
xl_, xr_ = nxs[3] - BW / 2, nxs[3] + BW / 2                                  # the last word's box, which the arrow must clear
t_ = min((xl_ - xs_) / (xe_ - xs_), (xr_ - xs_) / (xe_ - xs_), key=lambda v: v * (1 - v))
hc_ = (BHX / 2 + 2.0) / (2 * t_ * (1 - t_))                                  # the control's height: 2 pt over the box at its edges
b += P(f"M{xs_:.2f},{yb:.2f} Q{(xs_ + xe_) / 2:.2f},{yb - hc_:.2f} {xe_:.2f},{yb:.2f}", C["gray"], 0.7, marker="ag")
assert hc_ / 2 < 10.5 - BHX / 2 - 1.0, "the arch reaches the next row's boxes"                # rows 10.5 pt apart
b += bars(AX0, yb, TP1, G(11, 3), D(11, 3), hb=4.0)
LB1 = f"whole-state patch: G = {mm(G(11, 3))} T"
LO1 = f"DAS's patch: {mm(D(11, 3))} T"
b += T(AX0, yb - 7.4, LB1, FN, C["net"], weight=600)
b += T(AX0 + TP1 * D(11, 3) + 3.5, yb + 4.9, LO1, FN, C["das"], weight=600)
ya = yb + 11.0
b += L(AX0, ya, AX1, ya, C["grayl"], 1.0)
for t, txt in ((0, "0"), (1, "T"), (2, "2T"), (3, "3T")):
    b += L(AX0 + TP1 * t, ya - 2.0, AX0 + TP1 * t, ya + 2.0, C["gray"], 0.6)
    b += T(AX0 + TP1 * t, ya + 8.6, txt, FS["tick"], C["ink"] if t == 1 else C["muted"], anchor="middle", weight=600 if t == 1 else 400)

# ---- 2: every cell: the map is fig:guidea's network, each state box with its two moves beside it
Y2 = max(yw, ya + 8.6) + 20
TT2 = "At almost every cell, DAS moves the answer further than the whole state"
b += stepmark(0, Y2, 2, TT2)
SUB_O = (f"significantly further at {N_ABOVE} of the {N_MAP} cells here, and at {fmt(ALL_ABOVE)} of the {fmt(ALL_LIVE)} "
         "cells of all maps in six models")
assert tw(SUB_O, FS["sub"]) < W - 15, "step 2's line wider than the figure"
b += T(14.5, Y2 + 8.6, SUB_O, FS["sub"], C["das"])
BWm, BHm, GAPB = 9.0, 6.0, 2.0                                               # a state box, as in step 1's network, smaller
TP, GUT, PITCH = 36.0, 22.0, 12.4                                           # T in points, the gutter after T, the row pitch
GX = 36.0
bl = {r: GX + 4 + k * (BWm + GAPB + TP + GUT) for k, r in enumerate(REGS)}   # each column's boxes, left edge
x0s = {r: bl[r] + BWm + GAPB for r in REGS}                                  # each column's 0: the moves start beside the box
ROWS = [RO] + LAYERS[::-1]
GTOP = Y2 + 24
ys = {l: GTOP + k * PITCH for k, l in enumerate(ROWS)}
ytop, ybot = ys[RO] - 6.5, ys[LAYERS[0]] + 6.5
for r in REGS:                                                               # the network's columns, as in step 1
    b += L(bl[r] + BWm / 2, ys[LAYERS[0]], bl[r] + BWm / 2, ys[RO] if r == 4 else ys[LAYERS[-1]], C["grayl"], 0.6, dash="0.6 1.6")
    for t in ((1, 2, 3) if r == 4 else (1,)):                               # each column's T (and 2T, 3T for the last word)
        b += L(x0s[r] + TP * t, ytop if r == 4 else ys[LAYERS[-1]] - 6.5, x0s[r] + TP * t, ybot, C["line"] if t > 1 else C["grayl"], 0.5,
               dash="1.4 1.6")
for l in ROWS:
    b += T(GX, ys[l] + 2.1, "readout" if l == RO else f"layer {l - 1}", FS["tick"], C["muted"], anchor="end")
    for r in REGS:
        if (l, r) in cells:
            on = (l, r) == (11, 3)
            b += R(bl[r], ys[l] - BHm / 2, BWm, BHm, C["paper"], r=1.0, stroke=C["ink"] if on else C["grayl"], sw=0.8 if on else 0.45)
            b += dividers(bl[r] + BWm / 2, ys[l], BWm, BHm, C["gray"] if on else C["grayl"], seg=4)
            b += bars(x0s[r], ys[l], TP, G(l, r), D(l, r))
def num(l, r, which):
    v, col, dy = (G(l, r), C["net"], -1.0) if which == "g" else (D(l, r), C["das"], 5.0)
    other = D(l, r) if which == "g" else G(l, r)
    x = x0s[r] + TP * max(v, 0) + 2.5 if which == "d" else x0s[r] + TP * max(v, other if dy > 0 else v) + 2.5
    for t_ in ((1, 2, 3) if r == 4 else (1,)):                               # never across a column's T (2T, 3T) line
        xl = x0s[r] + TP * t_
        if x - 1.0 < xl < x + tw(mm(v), FS["tick"], 600) + 1.0:
            x = xl + 2.5
    return T(x, ys[l] + dy, mm(v), FS["tick"], col, weight=600), x + tw(mm(v), FS["tick"], 600)
for args in ((3, 1, "g"), (3, 3, "g"), (11, 3, "g"), (11, 3, "d"), (RO, 4, "d")):
    b += num(*args)[0]
_, xr_ = num(11, 3, "d")
ox0, ox1 = bl[3] - 2.5, xr_ + 2.5                                            # step 1's cell, outlined: its box, its moves, its numbers
b += R(ox0, ys[11] - 6.2, ox1 - ox0, 12.4, "none", r=2.0, stroke=C["ink"], sw=0.6)
assert ox1 < bl[4] - 3, "the outline runs into the next column"
def note(x, y, text, col):                                                   # a reading where it is drawn; a white ground over the guides
    return R(x - 1.5, y - 5.0, tw(text, FS["note"]) + 3.0, 6.8, C["paper"], r=1.0) + T(x, y, text, FS["note"], col)
# (layer 2's "half of T on each" has no place of its own: between the two words it reads as the comma's; 0.50 and 0.49 say it)
NB2 = "the last word takes over"
xn2 = x0s[4] + TP * max(G(15, 4), D(15, 4)) + 4.0
xn2 = max(xn2, x0s[4] + TP + 3.0)
assert xn2 + tw(NB2, FS["note"]) < W - 1, "layer 14's reading runs off the page"
assert xn2 + tw(NB2, FS["note"]) + 1.5 < x0s[4] + 3 * TP - 1.0, "layer 14's reading touches the 3T guide"
b += note(xn2, ys[15] + 2.0, NB2, C["net"])
for r in REGS:
    for t, txt in (((1, "T"), (2, "2T"), (3, "3T")) if r == 4 else ((1, "T"),)):
        b += T(x0s[r] + TP * t, ybot + 7.6, txt, FS["tick"], C["ink"] if t == 1 else C["muted"], anchor="middle", weight=600 if t == 1 else 400)
    cx = bl[r] + BWm / 2
    b += T(cx, ybot + 16.5, WORDS[r - 1], FN, weight=600, anchor="middle")
    if r in (1, 3):
        b += T(cx, ybot + 24.0, ws[r], FN, C["net"], weight=600, anchor="middle")
for xe in (bl[1] + BWm / 2 - 20.0, (bl[3] + bl[4]) / 2 + BWm / 2 + 4.0):   # the words not drawn, as in step 1
    b += T(xe, ybot + 16.5, "\u2026", FN, C["muted"], anchor="middle")
assert bl[1] + BWm / 2 - 20.0 - 4 > 0, "the first ellipsis runs off the page"
# the sentence, with the four words the columns patch: the last word of each of its regions
SP = tw(" ", FN)
SEG = [("in the sentence:", C["muted"], 400, SP), (wb[0].replace("<|endoftext|>", ""), C["muted"], 400, SP), ("sweet", C["ink"], 600, 0.0),
       (",", C["muted"], 400, SP), ("I", C["ink"], 600, SP), ("cherished", C["ink"], 600, SP),
       (" ".join(wb[4].split()[:-1]), C["muted"], 400, SP), ("is", C["ink"], 600, 0.0)]
xs_ = 14.5
for t_, col, wt, gap in SEG:
    b += T(xs_, ybot + 34.0, t_, FN, col, weight=wt); xs_ += tw(t_, FN, wt) + gap
assert xs_ < W - 1, "the sentence runs off the page"
H = ybot + 37.0
render("fig_guide_b", W, H, b, OUT)
print(f"fig_guide_b: DAS above G at {N_ABOVE} of {N_MAP} cells (BH per model); all maps {ALL_ABOVE} of {ALL_LIVE}; height {H:.1f}")

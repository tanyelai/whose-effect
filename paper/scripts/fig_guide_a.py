"""The visual guide's first figure (fig:guidea, the test), on one movie review in Qwen2.5-0.5B.

Two runs of one network: the base run reads a positive review and answers "great"; the source run reads the same review with
the two sentiment words changed and answers "terrible". Each gauge measures the answer, the log odds of "terrible" over
"great", from the base run's (0) to the source run's (T, the unit) and past it. A patch takes a state of the source run and
puts it into the base run, which then runs on; how far the base run's answer moves, over T, is the patch's effect. Layer l is
the output of block l, as in the paper (the map's cell layer l + 1). Three steps, in one vocabulary (two runs, a patch arrow,
a gauge; every box a strip of the state's numbers, the state of one word at one layer). The network is drawn at the four
words a cell patches (each region's last word: sweet, I, cherished, is, as in Figure 6); the words between them are not
drawn ("...", no box), and dotted columns elide the layers between the rows:
  1. whole-state patching of one cell, the adjective at layer 2: G = 0.50 T, a partial cut;
  2. two complete cuts, all of layer 2 and the readout alone (the last state at the last word, from which the answer is
     read): G = T exactly, for every pair (asserted against the readout run's T); the layer's cells one at a time below it;
  3. inside the readout: the box opens onto the answer axis, the readout's states at their answers. The whole-state patch
     is the trip from the base run's readout to the source run's (T), a ribbon of all the numbers. DAS's patch, d trained
     by cross-entropy DAS once on the task's training pairs and the same for every test pair, squeezes the change into one
     number, d's score (a weighted mix of all 896), and spreads it back by the same weights: d's part of the change. It lands
     on the base run's readout plus d's part (2.98 T, measured). The rest, what one number cannot carry, takes it on to the
     source run's readout, exactly T, so the second leg is exactly -1.98 T on the means: it takes the excess back. Why the rest pulls back
     is Figure 3's (fig_guide_c.py).
Every number is a mean over the task's 100 test pairs, from the map (experiments/results/inside/inside_Qwen2.5-0.5B_sentiment_movie.json:
whole-state patching, and the mean of DAS's three starts). The visual guide's introduction says why this example: its readout
lies in the middle of the six small models'.
Usage: python -I paper/scripts/fig_guide_a.py [--out=<dir>]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, R, T, arrows, circ, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, GROUP, TASK = "Qwen2.5-0.5B", "sentiment_templates", "sentiment_movie"
MAP = json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.json"))
cell = {(c["layer"], c["region"]): c for c in MAP["cells"]}
RO = max(L_ for L_, _ in cell)                                               # the readout: the last state, at the last word
LAST = max(r for _, r in cell)
own = lambda L_, r: cell[(L_, r)]["mean_T"]["own_normal"]  # noqa: E731
das = lambda L_, r: float(np.mean([cell[(L_, r)]["mean_T"][f"das_ce_s{s}_normal"] for s in range(3)]))  # noqa: E731
TM = float(cell[(RO, LAST)]["T_test"])
COPIES = [(1, own(3, 1), "whole"), (2, own(RO, LAST), "whole"), (3, das(RO, LAST), "dir")]
VERB3 = own(3, 3)
ROW2 = [own(3, 1), own(3, 2), own(3, 3), own(3, 4)]                          # sweet, I, cherished, is
assert MAP["n_test"] == 100 and abs(COPIES[1][1] - 1) < 1e-6, "the readout is a complete cut: the whole copy is T"
ZM = np.load(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.npz")
ZR = np.load(figdata.ROOT / f"experiments/results/readout/readout_{MODEL}_m_{GROUP}.npz")
assert np.abs(ZM[f"L{RO}r{LAST}/e/own_normal"].astype(float) - ZR[f"{TASK}/T"].astype(float)).max() < 1e-3, "not T on every pair"
WIDTH = json.load(open(figdata.ROOT / "paper/scripts/model_configs.json"))[f"Qwen/{MODEL}"]["width"]
assert (round(COPIES[0][1], 2), round(VERB3, 2), round(COPIES[2][1], 2), round(TM, 2)) == (0.50, 0.49, 2.98, 5.06), "a number the guide's text states has changed"
row = json.load(open(figdata.ROOT / f"experiments/data/{MODEL}/{GROUP}/{TASK}/test.json"))[0]
clean = lambda w: w.replace("<|endoftext|>", "").replace("\n", "")  # noqa: E731
wb, ws = [clean(w) for w in row["base"]], [clean(w) for w in row["src"]]
ab, asrc = row["base_label"].strip(), row["src_label"].strip()
DIFF = [i for i, (a, b_) in enumerate(zip(wb, ws)) if a != b_]
assert DIFF == [1, 3] and (ab, asrc) == ("great", "terrible"), (DIFF, ab, asrc)

W = NIPSW
FN = FS["axis"]                                                             # the guides' running text
b = arrows(("ak", C["ink"]), ("acp", C["net"]), ("ad", C["das"]), ("ag", C["gray"]), size=4.2)
b += T(0, 10, "The test: patch one state from a second run, and watch the answer move", FS["letter"], weight=600)
b += T(0, 19.6, f"{MODEL} on a movie review; every number is a mean over the task's 100 test pairs", FS["sub"], C["muted"])
def stepmark(x, y, n, title):
    return circ(x + 6, y - 2.4, 5.0, C["ink"]) + T(x + 6, y, str(n), FN, C["paper"], anchor="middle", weight=700) + \
        T(x + 14.5, y, title, FS["title"], weight=600)

# ---- the answer gauge: the base run's answer at 0, the source run's at T, and past it
GMAX = 3.25
def axis(x0, x1, y, key=False):
    X = lambda t: x0 + (x1 - x0) * t / GMAX  # noqa: E731
    o = L(x0, y, x1, y, C["grayl"], 1.2)
    for t, txt in ((0, "0"), (1, "T"), (2, "2T"), (3, "3T")):
        o += L(X(t), y - 2.2, X(t), y + 2.2, C["gray"], 0.6)
        o += T(X(t), y + 9, txt, FS["tick"], C["ink"] if t == 1 else C["muted"], anchor="middle", weight=600 if t == 1 else 400)
    if key:
        o += T(X(2.55), y - 5.5, "further \u2192", FS["note"], C["muted"], anchor="middle")
    return o, X
def gauge(x0, x1, y, marks, ends=True, key=False):
    o, X = axis(x0, x1, y, key)
    if ends:
        o += T(X(0), y - 5.5, "base run", FN, C["muted"], anchor="middle") + T(X(1), y - 5.5, "source run", FN, C["net"], anchor="middle")
    for k, (v, col, lab) in enumerate(sorted(marks, key=lambda m: -m[0])):
        if k == 0 and v > 0:
            o += R(x0, y - 1.6, X(v) - x0, 3.2, col, r=1.2)
        o += circ(X(v), y, 3.0, col) + (T(X(v), y - 13.5, lab, FN, col, anchor="middle", weight=700) if lab else "")
    return o, X

# ---- the network: four word columns ("...", the adjective, the verb, the last word), four layers and the readout.
#      A patch at (column, row) can change every later box: the same and later words, higher layers (causal attention).
NR = 4
GAP3 = 0.45                                                                  # the extra gap before "is", in column widths
def dividers(cx, cy, bw, bh, col, op=None, seg=5):
    """A box's strip: seg - 1 thin dividers, so that a box reads as a row of numbers (the state's 896), not one thing."""
    return "".join(L(cx - bw / 2 + k * bw / seg, cy - bh / 2 + 0.9, cx - bw / 2 + k * bw / seg, cy + bh / 2 - 0.9, col, 0.4,
                     cap="butt", op=op) for k in range(1, seg))
def network(x0, y0, cw, rp, bw=14.0, bh=6.0, run="base", solid=(), tint=(), words=None, values=None, rowlab=False,
            became=(), band=False):
    xs = [x0 + cw * (i + 0.5) for i in range(3)] + [x0 + cw * 3.5 + GAP3 * cw]   # "is" set apart: the words before it not drawn
    yb = y0 + NR * rp
    ys = [yb - r * rp for r in range(NR + 1)]                              # rows 0..3, then the readout (row 4)
    o = ""
    if band:                                                                # every word of the layer, those not drawn too
        o += R(xs[0] - bw / 2 - 9, ys[LAYER] - bh / 2 - 1.6, xs[3] + bw / 2 + 4 - (xs[0] - bw / 2 - 9), bh + 3.2, C["netl"], r=2.0, op=0.7)
    edge = C["netl"] if run == "source" else C["line"]
    for i, cx in enumerate(xs):
        o += L(cx, yb + bh / 2 + 1, cx, ys[NR] if i == 3 else ys[NR - 1], edge, 0.6, dash="0.6 1.6")   # the layers between: not drawn
    for r in range(NR):
        for i in range(3):
            xa, xb = xs[i] + bw / 2, xs[3] - bw / 2
            dx = (xb - xa) * 0.5
            o += P(f"M{xa:.2f},{ys[r]:.2f} C{xa + dx:.2f},{ys[r]:.2f} {xb - dx:.2f},{ys[r + 1]:.2f} {xb:.2f},{ys[r + 1]:.2f}",
                   edge, 0.4, op=0.7)
    for r in range(NR + 1):
        for i, cx in enumerate(xs):
            if r == NR and i != 3:
                continue
            if run == "source" or (i, r) in solid or (i, r) in became:
                fill, stroke = (C["net"], C["net"]) if (i, r) in solid else (C["netl"], C["net"])            # copied; or the source run's
            elif (i, r) in tint:
                fill, stroke = "#F3F6FC", C["net"]
            else:
                fill, stroke = C["paper"], C["grayl"]
            o += R(cx - bw / 2, ys[r] - bh / 2, bw, bh, fill, r=1.4, stroke=stroke, sw=0.5 if (i, r) not in tint else 0.8,
                   dash="1.6 1.1" if ((i, r) in tint and (i, r) not in solid and run != "source") else None)
            if (i, r) in solid:
                o += dividers(cx, ys[r], bw, bh, C["paper"])
            else:
                o += dividers(cx, ys[r], bw, bh, C["grayl"] if fill == C["paper"] else C["net"], op=None if fill == C["paper"] else 0.45)
    for (i, r), txt in (values or {}).items():
        o += T(xs[i], yb + 19.5, txt, FN, C["net"], anchor="middle", weight=600)
    o += L(xs[3], ys[NR] - bh / 2 - 1.5, xs[3], ys[NR] - 10, C["ink"], 0.6, marker="ak")
    for i, cx in enumerate(xs):
        w = (words or [wb[1].strip(), wb[2].split()[-1], wb[3].strip(), wb[4].split()[-1]])[i]
        diff = i in (0, 2)
        o += T(cx, yb + 11, w, FN, (C["net"] if run == "source" else C["ink"]) if diff else C["muted"], anchor="middle",
               weight=600 if diff else 400)
    o += T(xs[0] - 0.6 * cw, yb + 11, "\u2026", FN, C["muted"], anchor="middle")             # the words not drawn
    o += T((xs[2] + xs[3]) / 2 + 2, yb + 11, "\u2026", FN, C["muted"], anchor="middle")
    if rowlab:
        o += T(xs[0] - bw / 2 - 4, ys[1] + 2, "layer 2", FN, C["muted"], anchor="end")
        o += T(xs[0] - bw / 2 - 4, ys[NR] + 2, "readout", FN, C["muted"], anchor="end")
    return o, xs, ys
def halo(x_end, y, text, weight=400):
    """A white ground behind a label that sits on a network's faint curves, so that the curves do not run through it."""
    w_ = tw(text, FN, weight)
    return R(x_end - w_ - 1.6, y - 5.4, w_ + 3.2, 7.4, C["paper"], r=1.2)
def cone(i0, r0):
    """Every box a patch at (i0, r0) can change: the same and later words, at higher layers, and the readout."""
    return {(i, r) for i in range(i0, 4) for r in range(r0 + 1, NR)} | ({(3, NR)} if i0 <= 3 else set())
LAYER = 1

# ---- 1: the two runs and one patch, drawn large: one step over both halves
Y1 = 34.0
b += stepmark(0, Y1, 1, "Two runs and one patch")
b += T(14.5, Y1 + 8.6, f"the answer: log odds of \u201c{asrc}\u201d over \u201c{ab}\u201d; T: how far switching the input moves it", FS["sub"], C["muted"])
b += T(14.5, Y1 + 16.0, "G: how far a whole-state patch moves it", FS["sub"], C["muted"])
NX1, NX2, NY, CW1, RP1 = 34.0, 236.0, Y1 + 69, 34.0, 18.0
NXS = NX1 - 5.0                                                              # the source run's network, a little left
s_, sx, sy = network(NXS, NY, CW1, RP1, bw=18, bh=7.5, run="source", words=[ws[1].strip(), ws[2].split()[-1], ws[3].strip(), ws[4].split()[-1]], rowlab=True)
b += s_
b += T(NXS + 2 * CW1, sy[0] + 24, "the source run", FN, C["net"], anchor="middle", weight=600)
n_, bx, by = network(NX2, NY, CW1, RP1, bw=18, bh=7.5, run="base", solid={(0, LAYER)}, tint=cone(0, LAYER))
b += n_
b += T(NX2 + 2 * CW1, by[0] + 24, "the base run, with the patch", FN, C["ink"], anchor="middle", weight=600)
# a gauge over each run: the source run's answer is T by definition; the patch moves the base run's to 0.50 T
GY1 = Y1 + 39
b += gauge(NX1 + 6, NX1 + 4 * CW1 + 14, GY1, [(1.0, C["net"], f"T = {TM:.2f} log odds")], key=True)[0]
b += gauge(NX2 + 6, NX2 + 4 * CW1 + 14, GY1, [(COPIES[0][1], C["net"], f"G = {COPIES[0][1]:.2f} T")])[0]
# the patch, routed along the channel just above layer 2: out of the source run's box, across, down into the same place
x0c, y0c = sx[0] + 9, sy[LAYER]                                              # the right edge of the source run's box
yc = sy[LAYER] - RP1 / 2 - 2.5                                               # the channel between layer 2 and the row above, 2 pt
#                                                                              up, so that the drop into the box has room for its head
HL, HW = 4.6, 5.6                                                            # the head, drawn: a sharp tip, not a marker on a round cap
x1c, ytip = bx[0], by[LAYER] - 7.5 / 2 - 0.2                                 # the tip, just over the top of the base run's box
y1c = ytip - HL                                                              # the line ends at the head's base
rr, re_ = 4.0, 2.2                                                           # the bends: the last one tighter
assert y1c - (yc + re_) > 0.4, "no straight drop before the head"
assert yc + 2.0 < sy[LAYER] - 7.5 / 2 and yc - 2.0 > sy[LAYER + 1] + 7.5 / 2, "the channel's white ground touches a box"
route = (f"M{x0c:.2f},{y0c:.2f} H{x0c + 4:.2f} Q{x0c + 4 + rr:.2f},{y0c:.2f} {x0c + 4 + rr:.2f},{y0c - rr:.2f} "
         f"V{yc + rr:.2f} Q{x0c + 4 + rr:.2f},{yc:.2f} {x0c + 4 + 2 * rr:.2f},{yc:.2f} H{x1c - re_:.2f} "
         f"Q{x1c:.2f},{yc:.2f} {x1c:.2f},{yc + re_:.2f} V{y1c + 0.4:.2f}")
b += P(route, C["paper"], 4.0) + P(route, C["net"], 1.3)
b += f'<path d="M{x1c - HW / 2:.2f},{y1c:.2f} L{x1c + HW / 2:.2f},{y1c:.2f} L{x1c:.2f},{ytip:.2f} Z" fill="{C["net"]}"/>'
mid = (sx[3] + 9 + bx[0] - 9) / 2
b += T(mid, yc - 5, "whole-state patch", FN, C["net"], anchor="middle", weight=700)
PL = [f"all {WIDTH} numbers,", "same layer,", "same place"]
for k, t_ in enumerate(PL):
    b += T(mid, yc + 9.5 + 7.4 * k, t_, FN, C["net"], anchor="middle")
assert all(sx[3] + 9 + 6 < mid - tw(t_, FN) / 2 and mid + tw(t_, FN) / 2 < bx[0] - 9 - 6 for t_ in PL), "the patch's words touch a network"
assert yc + 9.5 + 7.4 * 2 < sy[0] + 7.5 / 2 + 11 - 6, "the patch's words reach the words under the networks"
b += halo(bx[3] - 9 - 4, by[NR] + 2.2, "dashed: cells the patch can reach")
b += T(bx[3] - 9 - 4, by[NR] + 2.2, "dashed: cells the patch can reach", FN, C["net"], anchor="end")

# ---- 2: a complete cut, of both kinds: all of layer 2, and the readout alone. Each moves the answer exactly T, because above
#      the cut the base run computes what the source run computes (pale, as in step 1). The halves sit under step 1's halves,
#      their gauges under its gauges, so that T lines up down the page; one cell of the layer alone gives a part.
Y2 = by[0] + 44
CW2, RP2 = 28.0, 12.5
fmt0 = lambda v: f"{v:.2f}" if abs(v) > 0.005 else "0"  # noqa: E731     a box's value as the drawing prints it
b += stepmark(0, Y2, 2, "A complete cut: G = T exactly")
b += T(14.5, Y2 + 8.6, "every path to the answer passes through the patched cells", FS["sub"], C["muted"])
GY2 = Y2 + 34
for gx in (NX1, NX2):
    b += gauge(gx + 6, gx + 4 * CW1 + 14, GY2, [(1.0, C["net"], "G = T")])[0]
nxa = NX1 + 10 + 2 * CW1 - 2 * CW2
na_, xa, ya = network(nxa, Y2 + 58, CW2, RP2, solid={(i, LAYER) for i in range(4)}, band=True,
                      became={(i, r) for i in range(4) for r in range(LAYER + 1, NR)} | {(3, NR)},
                      values={(i, LAYER): fmt0(v) for i, v in enumerate(ROW2)})
b += na_
bkx = xa[0] - 7 - 7                                                          # the bracket, 7 pt clear of the boxes
lx = bkx - 5                                                                 # its label and "all of layer 2" end here
b += T(lx, ya[LAYER] + 2.2, "all of layer 2", FN, C["net"], anchor="end", weight=600)
yt, ybr = ya[NR] - 3.4, ya[LAYER + 1] + 3.4                                  # over the cells that became the source run's
b += P(f"M{bkx + 3:.2f},{yt:.2f} Q{bkx:.2f},{yt:.2f} {bkx:.2f},{yt + 3:.2f} V{ybr - 3:.2f} Q{bkx:.2f},{ybr:.2f} {bkx + 3:.2f},{ybr:.2f}",
       C["net"], 0.7)
ym = (yt + ybr) / 2
b += T(lx, ym - 1.4, "every cell above:", FN, C["net"], anchor="end") + T(lx, ym + 6.0, "the source run's", FN, C["net"], anchor="end", weight=600)
assert lx - max(tw("every cell above:", FN), tw("the source run's", FN, 600)) > 0.5, "the bracket's label runs off the page"
b += T(xa[0] - 11, ya[0] + 19.5, "one cell alone:", FN, C["muted"], anchor="end")
nxb = NX2 + 10 + 2 * CW1 - 2 * CW2
nb_, xb2, yb2 = network(nxb, Y2 + 58, CW2, RP2, solid={(3, NR)})
b += nb_
b += halo(xb2[3] - 7 - 4, yb2[NR] + 2.2, f"the readout alone, all {WIDTH} numbers", 600)
b += T(xb2[3] - 7 - 4, yb2[NR] + 2.2, f"the readout alone, all {WIDTH} numbers", FN, C["net"], anchor="end", weight=600)

# ---- 3: inside the readout, its states on the answer axis at their answers (the zoom of step 2's readout and gauge). The
#      whole-state patch is the trip from the base run's readout to the source run's (T): a ribbon of all the numbers. DAS's
#      patch goes only d's part of the way, through one score of all of them (the bottleneck), and lands on the base run's
#      readout plus d's part (2.98 T, measured). The rest of the trip then lands on the source run's readout, exactly T, so
#      the second leg is exactly -1.98 T on the means: it takes the excess back. Why the rest pulls back is Figure 3's.
Y3 = Y2 + 58 + NR * RP2 + 19.5 + 20
G_, E_ = COPIES[1][1], COPIES[2][1]
b += stepmark(0, Y3, 3, "Inside the readout: through one score, DAS stops partway and goes further")
SUB3 = f"d: a weighted mix of all {WIDTH} numbers that DAS learns, trained so that its patch makes the answer “{asrc}”"
assert tw(SUB3, FS["sub"]) < W - 14.5, SUB3
b += T(14.5, Y3 + 8.6, SUB3, FS["sub"], C["muted"])
AX0, AX1 = 80.0, 386.0
SCL = (AX1 - AX0) / GMAX
Xb = lambda t: AX0 + SCL * t  # noqa: E731
ySt, yA = Y3 + 72, Y3 + 104
NC, CWc, CH = 8, 5.4, 8.0
FX0, FX1, FY0, FY1 = 54.0, 394.0, Y3 + 16, Y3 + 130
# the zoom: the readout box, level with the middle of the frame, opens onto it as a magnifier does, its top corner to the
# frame's top corner and its bottom corner to the frame's bottom corner, the cone between them faintly filled
gx, gy_ = 17.0, (FY0 + FY1) / 2
bxr, byt, byb = gx + 9, gy_ - 3.75, gy_ + 3.75                               # the box's right edge, top and bottom
b += (f'<polygon points="{bxr:.2f},{byt:.2f} {FX0:.2f},{FY0 + 4:.2f} {FX0:.2f},{FY1 - 4:.2f} {bxr:.2f},{byb:.2f}" '
      f'fill="{C["faint"]}"/>')
b += L(bxr, byt, FX0 + 1.2, FY0 + 1.2, C["grayl"], 0.6) + L(bxr, byb, FX0 + 1.2, FY1 - 1.2, C["grayl"], 0.6)
b += R(FX0, FY0, FX1 - FX0, FY1 - FY0, "none", r=4, stroke=C["grayl"], sw=0.6)
b += R(gx - 9, byt, 18, 7.5, C["paper"], r=1.4, stroke=C["gray"], sw=0.5) + dividers(gx, gy_, 18, 7.5, C["gray"])
b += T(gx - 2, byb + 9.5, "readout", FS["note"], C["muted"], anchor="middle")
# the label stays left of the lower line: at the label's right end, the line is still above the label's top
_xr = gx - 2 + tw("readout", FS["note"]) / 2
assert _xr <= bxr or byb + (_xr - bxr) / (FX0 + 1.2 - bxr) * (FY1 - 1.2 - byb) < byb + 9.5 - 4.6, "the zoom's line meets its label"
def strip8(xc, y, fill, stroke):
    x0 = xc - NC * CWc / 2
    return "".join(R(x0 + q * CWc + 0.4, y - CH / 2, CWc - 0.8, CH, fill, r=0.6, stroke=stroke, sw=0.5) for q in range(NC))
# the axis: the answer, as on every gauge, longer
b += L(AX0, yA, AX1, yA, C["grayl"], 1.2)
for t, txt in ((0, "0"), (1, "T"), (2, "2T"), (3, "3T")):
    b += L(Xb(t), yA - 2.2, Xb(t), yA + 2.2, C["gray"], 0.6)
    b += T(Xb(t), yA + 9, txt, FS["tick"], C["ink"] if t == 1 else C["muted"], anchor="middle", weight=600 if t == 1 else 400)
STATES = [(0.0, C["paper"], C["gray"], C["gray"], ("base run", ""), C["muted"]),
          (G_, C["netl"], C["net"], C["net"], ("source run", "whole-state patch"), C["net"]),
          (E_, C["dasl"], C["das"], C["das"], ("base + d's part", "DAS's patch"), C["das"])]
for t, fill, stroke, dot, labs, col in STATES:
    xc = Xb(t)
    b += strip8(xc, ySt, fill, stroke)
    b += T(xc, ySt + CH / 2 + 8, labs[0], FS["note"], col, anchor="middle", weight=600)
    b += T(xc, ySt + CH / 2 + 14.5, labs[1], FS["note"], col, anchor="middle") if labs[1] else ""
    b += L(xc, ySt + CH / 2 + (17.5 if labs[1] else 11), xc, yA - 3.4, C["grayl"], 0.5, dash="1 1.2") + circ(xc, yA, 3.0, dot)
SPREAD = [-12.0, -6.0, 0.0, 6.0, 12.0]                                       # five of the numbers, drawn as a ribbon
ytop = ySt - CH / 2 - 1.5
# the whole trip: every number moves, a low ribbon from the base run's readout to the source run's
for dx in SPREAD:
    xa_, xb_ = Xb(0) + dx, Xb(G_) + dx
    b += P(f"M{xa_:.2f},{ytop:.2f} Q{(xa_ + xb_) / 2:.2f},{ytop - 24:.2f} {xb_:.2f},{ytop:.2f}", C["net"], 1.1 if dx == 0 else 0.5,
           marker="acp" if dx == 0 else None, op=None if dx == 0 else 0.7)

# DAS's leg: the numbers squeeze into one score, d's, which alone carries the way, then spread back over all of them
yb_ = ytop - 34
n1, n2 = Xb(0) + 34, Xb(E_) - 34
for dx in SPREAD:
    xa_, xb_ = Xb(0) + dx, Xb(E_) + dx
    b += P(f"M{xa_:.2f},{ytop:.2f} C{xa_:.2f},{yb_ + 8:.2f} {n1 - 16:.2f},{yb_:.2f} {n1 - 3:.2f},{yb_:.2f}", C["das"], 0.6, op=0.85)
    b += P(f"M{n2 + 3:.2f},{yb_:.2f} C{n2 + 16:.2f},{yb_:.2f} {xb_:.2f},{yb_ + 8:.2f} {xb_:.2f},{ytop:.2f}", C["das"], 1.1 if dx == 0 else 0.6,
           marker="ad" if dx == 0 else None, op=None if dx == 0 else 0.85)
b += circ(n1, yb_, 2.6, C["das"]) + circ(n2, yb_, 2.6, C["das"]) + L(n1, yb_, n2, yb_, C["das"], 2.2)
LAB_D = f"DAS's patch: the change squeezed into one number, d's score, and spread back: d's part, +{E_:.2f} T"
b += T((n1 + n2) / 2, yb_ - 4.5, LAB_D, FS["note"], C["das"], anchor="middle", weight=600)
assert FX0 + 4 < (n1 + n2) / 2 - tw(LAB_D, FS["note"], 600) / 2 and (n1 + n2) / 2 + tw(LAB_D, FS["note"], 600) / 2 < FX1 - 4, LAB_D
# the rest of the trip, just over the axis (clear of the tick labels): on to the source run's readout, exactly T
yg = yA - 8
b += L(Xb(E_), yg, Xb(G_) + 4.5, yg, C["gray"], 1.2, marker="ag")
LAB_R = f"then the rest, what one number cannot carry, takes the excess back: −{E_ - G_:.2f} T, to T"
b += T((Xb(E_) + Xb(G_)) / 2, yA + 19, LAB_R, FS["note"], C["tgray"], anchor="middle")
assert FX0 + 4 < (Xb(E_) + Xb(G_)) / 2 - tw(LAB_R, FS["note"]) / 2 and (Xb(E_) + Xb(G_)) / 2 + tw(LAB_R, FS["note"]) / 2 < FX1 - 4, LAB_R
fy = FY1 + 16
QUESTION = "Is the excess an effect of the network, or of the objective that learned the direction?"   # any objective, not DAS's alone
assert tw(QUESTION, FS["title"], 600) < W
b += T(0, fy, QUESTION, FS["title"], C["das"], weight=600)
H = fy + 5

render("fig_guide_a", W, H, b, OUT)
print(f"fig_guide_a: T {TM:.2f} log odds; step 1 G {COPIES[0][1]:.2f} (verb {VERB3:.2f}), step 2 G {COPIES[1][1]:.2f}, "
      f"step 3 DAS {COPIES[2][1]:.2f}")

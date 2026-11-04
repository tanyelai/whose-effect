"""The paper's first figure (fig:law, text width, 397 pt): a learned part of a model's state can move its answer more than the
whole state does.

The run drawn is figdata.FIG1, Llama-3.2-3B on RAVEL's city-to-language task (experiments/results/score_readout.json; the readout site
is the last block's output at the last word). The rule written at FIG1 picks it, and figdata.fig1_check() recomputes the rule
here; the script prints its cross-entropy DAS's percentile among the one-question tasks of the uncapped models.
One height scale runs through (b) and (c): the direction the answer reads is drawn upward, so what the answer sees of a change
(its first-order effect) is the change's height, in units of T, the move from changing the city itself. The scale is 30 pt per
T unless the whole change, drawn at its angle, would then run past (b)'s 82 pt; then it is 82 pt / tan(theta) per T.
(a) The test, on the task's test pairs (one drawn in front, Coventry and Bogota): one network runs on two sentences and
    the answer is read from its last state (the readout); that state is copied from the source run into the base run, whole
    (blue) or only along one direction (grey). Each run is drawn in Figure 2's language: the last word's states through the
    layers, striped boxes (a state's numbers), the layers between them not drawn (dotted), the source run light blue; the last
    state, outlined in ink, is the readout, the one site the figure concerns.
(b) Why (Proposition 7, with one readout direction): the task's mean change, drawn at its measured angle theta
    to the direction the answer reads (from the training pairs). With one readout direction the first-order effect of copying
    along a direction is linear in the change, so the mean change gives the mean effect. Its part along the bisector (the best
    part) rises to (1 + 1/cos theta)/2; the rest, perpendicular to it, falls back by the difference; the whole rises to 1.
(c) What was measured on the task's 100 test pairs (means, in T): copying the whole state, exactly 1 (the complete-cut
    identity); the average change, an untrained direction; DAS trained with cross-entropy to give the source's answer; the
    same DAS trained with a squared error to each pair's whole-input move. The best part's height (dashed) and T (light) run
    across from (b).
Colours: blue the whole (state, change); orange only the measured DAS bars (open: the squared-error loss); dark grey the best
part, the top of the range (as fig:readout draws the top), which a learned direction approaches but is not; teal the untrained
average change; violet the direction the answer reads; light grey the rest.
The scorer's rows hold these under excess (cross-entropy DAS over T), natural (the untrained directions of a one-question task, here
the average change), angle (the one-angle prediction P of the top) and objective (where the squared error lands).
Usage: python -I paper/scripts/fig_law.py [--out=<dir>]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, L, NIPSW, R, T, arrows, render  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
# The run drawn (experiments/results/score_readout.json): figdata.FIG1, by the rule written there; fig1_check() stops here if the
# rule picks another task. Its cross-entropy DAS's percentile among the one-question tasks is printed at the end.
SCORE = json.load(open(figdata.ROOT / "experiments/results/score_readout.json"))
F1 = figdata.FIG1
F1C = figdata.fig1_check(SCORE)
angle_of = lambda P: math.degrees(math.acos(min(1.0, 1.0 / (2 * P - 1))))  # noqa: E731
row = next(r for r in SCORE["rows"][F1["model"]] if r["group"] == F1["group"] and r["task"] == F1["task"])
assert row["main"] and row["defined"] and row["one_q"], "Figure 1's run is a one-question task of the main population"
RUN = dict(angle=angle_of(row["angle"]["P"]), pred=row["angle"]["P"], ce=row["excess"]["ce_p1"]["pt"], sq=row["objective"]["p1"]["sq_T"],
           own=row["natural"]["exact_u_hat"][0], n=row["n"])
pair = next(x for x in json.load(open(figdata.ROOT / f"experiments/data/{F1['model']}/{F1['group']}/{F1['task']}/{F1['split']}.json"))
            if x["base"][1].strip() == F1["base"] and x["src"][1].strip() == F1["src"])  # RAVEL's prompt: one example, then the city
# App. B.1 quotes the example the pair's prompt opens with (its RAVEL entry): it must be this pair's
_ex = pair["base"][0].rsplit(". ", 1)[0] + "."
_app = figdata.ROOT / "paper/sections/appendix.tex"                       # the paper's LaTeX source, where it is present
assert not _app.is_file() or " ".join(_ex.split()) in " ".join(_app.read_text().replace("\\emph{", "").split()), \
    f"the data appendix must quote Figure 1's opening example: {_ex}"
TH, PRED = RUN["angle"], RUN["pred"]
th = math.radians(TH); tn = math.tan(th)
W, H = NIPSW, 141.2
S, Y0 = min(30.0, 82.0 / tn), 108.0                                    # pt per T (see the docstring); the height 0 (the base answer)
XR = 141.0                                                              # (b)'s readout axis, the base run's state
b = arrows(("ak", C["ink"]), ("an", C["net"]), ("av", C["unt"]), ("ag", C["gray"]), ("at", C["tgray"]), size=4.5)


def unit(a, b_):
    dx, dy = b_[0] - a[0], b_[1] - a[1]; n = math.hypot(dx, dy)
    return dx / n, dy / n


def tick_right(at, d1, d2, q=4.2):
    """A right-angle mark at `at` between unit directions d1 and d2."""
    return (f'<path d="M{at[0] + q * d1[0]:.2f},{at[1] + q * d1[1]:.2f} L{at[0] + q * d1[0] + q * d2[0]:.2f},'
            f'{at[1] + q * d1[1] + q * d2[1]:.2f} L{at[0] + q * d2[0]:.2f},{at[1] + q * d2[1]:.2f}" fill="none" '
            f'stroke="{C["tgray"]}" stroke-width="0.5"/>')


def head(x, letter, title, sub=None, sub_col=None, sub2=None):
    """A panel's letter, title and subtitle; the second subtitle line says which pairs the panel draws: the numbers are means
    over the task's pairs, so the figure itself must not let (a)'s one pair read as their source."""
    o = T(x, 9, letter, 9.0, weight=700) + T(x + 9.5, 9, title, FS["title"], weight=600)
    o += T(x + 9.5, 17.6, sub, FS["sub"], sub_col or C["muted"]) if sub else ""
    return o + (T(x + 9.5, 25.4, sub2, FS["sub"], C["muted"]) if sub2 else "")


# ------------------------------------------------------------------ (a) the test, on the task's test pairs
assert F1["split"] == "test", "Figure 1(a) draws one of the test pairs (c) averages over"
b += head(0.0, "a", "The test", f"{figdata.FIG1_NAME}, two runs", sub2=f"on each of the task's {RUN['n']} test pairs")
# Each run: the sentence, the network and its last state, from which the answer is read (the readout). The sentence is the
# front card of a deck, the task's test pairs: (b) and (c) are means over them, not this pair's. The patch runs between the two
# last states: whole (blue, as (c)'s "patch the whole state") or only along one direction (grey, as (c)'s "patch one direction").
# The network is drawn in Figure 2's language: the last word's states through the layers, striped boxes, the layers between not
# drawn (dotted); the last state, outlined in ink, is the readout (the model is named in the subtitle).
CX, CWD, CH, DK = 0.6, 33.0, 24.0, 1.9                                  # the front card: x, width, height; each card behind is DK up-right
NX, NW = 47.5, 38.5                                                     # the run's states, from the first drawn to the readout
BW_, BH_, NB_ = 10.0, 6.5, 3                                            # a state box, and how many are drawn
BG_ = (NW - NB_ * BW_) / (NB_ - 1)


def dividers(cx, cy, bw, bh, col, op=None, seg=5):
    """Figure 2's strip: seg - 1 thin dividers, so that a box reads as a row of numbers (the state's), not one thing."""
    return "".join(L(cx - bw / 2 + k * bw / seg, cy - bh / 2 + 0.9, cx - bw / 2 + k * bw / seg, cy + bh / 2 - 0.9, col, 0.4,
                     cap="butt", op=op) for k in range(1, seg))


xs_ = [NX + BW_ / 2 + k * (BW_ + BG_) for k in range(NB_)]
XANS = NX + NW + 9.5                                                    # the answers: room for the read-out arrow's shaft
TIP = 0.675                                                             # a marker's tip runs 0.15 x its 4.5 pt past the line's end
YS, YB = 47.0, 98.0                                                     # the source run's row and the base run's
for yy, ans in ((YS, pair["src_label"].strip()), (YB, pair["base_label"].strip())):
    parts = pair["src"] if yy == YS else pair["base"]                   # RAVEL's three spans: the prompt, the city, the verb
    w1, city, verb = parts[0].split(". ")[-1].strip().removesuffix(" in"), parts[1].strip(), parts[2].strip()   # after RAVEL's example
    for k in (2, 1):
        b += R(CX + k * DK, yy - CH / 2 - k * DK, CWD, CH, "#FFFFFF", r=1.5, stroke=C["tgray"], sw=0.4)
    b += R(CX, yy - CH / 2, CWD, CH, "#FFFFFF", r=1.5, stroke=C["tgray"], sw=0.6)
    for i, t_ in enumerate((f"{w1} in", city, verb)):
        b += T(3.0, yy - 5.0 + i * 7.0, t_, FS["note"], C["ink"])
    b += L(CX + CWD + 2 * DK + 1.2, yy, NX - TIP, yy, C["gray"], 0.6, marker="ag")      # the input runs in: its tip on the first state
    src_ = yy == YS                                                     # Figure 2's colours: the source run light blue
    fill, stroke, div = (C["netl"], C["net"], C["net"]) if src_ else (C["paper"], C["grayl"], C["grayl"])
    b += L(xs_[0] + BW_ / 2, yy, xs_[-1] - BW_ / 2, yy, C["netl"] if src_ else C["line"], 0.6, dash="0.6 1.6")   # layers not drawn
    for k, cx_ in enumerate(xs_):
        last = k == NB_ - 1                                             # the readout: the site, outlined in ink
        b += R(cx_ - BW_ / 2, yy - BH_ / 2, BW_, BH_, fill, r=1.4, stroke=C["ink"] if last else stroke, sw=0.8 if last else 0.5)
        b += dividers(cx_, yy, BW_, BH_, div, op=0.45 if src_ else None)
    b += T(NX, yy - BH_ / 2 - 3.0, "source run" if src_ else "base run", FS["note"], C["muted"])   # who gives, who takes
    b += L(NX + NW + 0.8, yy, XANS - 1.2 - TIP, yy, C["gray"], 0.6, marker="ag")     # the answer read out: 1.2 pt clear of it
    b += T(XANS, yy + 2.4, ans, FS["axis"], C["ink"], weight=700)
# one patch, whole or along one direction: alternatives, so one arrow (two arrows read as two things patched together); its words
# carry (c)'s colours, the whole in blue ("patch the whole state"), one direction in grey ("patch one direction")
xw = xs_[-1]                                                            # from the source's readout to the base's
b += L(xw, YS + BH_ / 2 + 1.0, xw, YB - BH_ / 2 - 1.6, C["ink"], 0.9, marker="ak")
for i, (t_, col, wt) in enumerate((("patch the whole state", C["net"], 700), ("or only its part along", C["muted"], 400),
                                    ("one direction", C["muted"], 700))):
    b += T(xw - 3.0, 66.0 + i * 6.6, t_, FS["note"], col, anchor="end", weight=wt)
# the base run's answer is its own, before any copy: the move a copy is measured by runs from it toward the source's, and changing
# the city itself moves it the whole way, T (dark grey: the input's own change)
xa = XANS + 11.6
b += L(xa, YB - 6.2, xa, YS + 4.6, C["tgray"], 0.8, marker="at")
b += T(xa + 2.6, (YS + YB) / 2 + 2.6, "T", FS["note"], C["ink"], weight=700)
b += T(0, Y0 + 10.5, "T", FS["note"], C["ink"], weight=700) + T(3.9, Y0 + 10.5, ": how far changing the city", FS["note"], C["ink"])
b += T(0, Y0 + 17.3, "moves the answer, on average", FS["note"], C["ink"])


# ------------------------------------------------------------------ (b) why: the geometry, the answer reading height
b += head(XR - 12.0, "b", "Why a part can rise higher", "the answer reads only height", C["unt"], sub2="in T, mean over the training pairs")
O = (XR, Y0); Sw = (XR + tn * S, Y0 - S); Pp = (XR + tn / 2 * S, Y0 - PRED * S)   # base state, whole change's tip, best part's tip
b += L(XR, Y0, W - 2.0, Y0, C["line"], 0.6)                            # height 0, T, and the best part's height, carried to (c)
b += L(XR, Sw[1], W - 2.0, Sw[1], C["line"], 0.6)
b += L(Pp[0] + 1.0, Pp[1], W - 2.0, Pp[1], C["tgray"], 0.7, dash="2.2 1.6")
b += L(XR, Y0 + 1.5, XR, Y0 - 2.75 * S, C["unt"], 0.9, marker="av")
for v, t_ in ((0, "0"), (1, "T"), (2, "2T")):
    b += L(XR - 2.2, Y0 - v * S, XR, Y0 - v * S, C["unt"], 0.6)
    b += T(XR - 4.0, Y0 - v * S + 2.1, t_, FS["tick"], C["ink"] if v == 1 else C["muted"], anchor="end", weight=700 if v == 1 else 400)
rA = 17.0                                                               # theta: from the readout axis to the whole change
arc = " ".join(f"{XR + rA * math.sin(a):.2f},{Y0 - rA * math.cos(a):.2f}" for a in np.linspace(0, th, 30))
b += f'<polyline points="{arc}" fill="none" stroke="{C["muted"]}" stroke-width="0.6"/>'
for a_ in (0.0, th):
    b += L(XR + (rA - 2.0) * math.sin(a_), Y0 - (rA - 2.0) * math.cos(a_), XR + (rA + 2.0) * math.sin(a_),
           Y0 - (rA + 2.0) * math.cos(a_), C["muted"], 0.6)
ud = unit(O, Pp)
b += L(Pp[0], Pp[1], Sw[0] - 1.2, Sw[1] - 1.0, C["gray"], 0.9, dash="2 1.4", marker="ag")          # the rest
b += tick_right(Pp, (-ud[0], -ud[1]), unit(Pp, Sw))
b += L(O[0], O[1], Sw[0] - 1.6, Sw[1] + 0.5, C["net"], 1.4, marker="an")                           # the whole change
b += L(O[0], O[1], Pp[0], Pp[1], C["tgray"], 1.4, marker="at")                                     # the best part: the top's colour, not DAS's
b += T(XR + rA + 4.0, Y0 + 7.4, f"θ = {TH:.1f}°", FS["note"], C["muted"])
b += T(Pp[0] + 4.0, Pp[1] - 9.4, "maximum effect", FS["note"], C["tgray"], weight=600)
b += T(Pp[0] + 4.0, Pp[1] - 2.6, "of one direction", FS["note"], C["tgray"])
mr = ((Pp[0] + Sw[0]) / 2, (Pp[1] + Sw[1]) / 2)
b += T(mr[0] + 4.0, mr[1] - 2.0, "rest", FS["note"], C["muted"], weight=600)
_xw = XR + tn * S / 2 + 8.0; _yw = Y0 - (_xw - XR) / tn                 # the arrow's height where the label starts: under it if
b += T(_xw, Y0 - 1.6 if Y0 - _yw >= 8.5 else _yw - 2.5, "whole change", FS["note"], C["net"], weight=600)   # there is room
xA = XR - 14.0                                                          # the heights add; two lines, so the parts keep their names
b += T(xA, Y0 + 17.0, f"top-eigenvector part {PRED:.2f}", FS["note"], C["tgray"], weight=700)
for txt, col, wt in (("+", C["muted"], 400), (f"rest (−{PRED - 1:.2f})", C["muted"], 700),
                     ("=", C["muted"], 400), ("whole change 1", C["net"], 700)):
    b += T(xA, Y0 + 23.8, txt, FS["note"], col, weight=wt)
    xA += len(txt) * 3.05 + 2.6

# ------------------------------------------------------------------ (c) measured: means over the task's test pairs, on the same heights
XC, CW = 254.0, 18.0
# (c)'s subtitle says what each bar scores, where the bars are: how far a patch moves the base run's answer, in units of T
b += head(XC - 9.5, "c", "Measured: DAS above the whole state", "how far each patch moves the base run's answer",
          sub2=f"in T, means over the task's {RUN['n']} test pairs")
xs = (256.0, 295.5, 333.5, 362.0)                                        # the last label clears the right edge
for x, v, col, open_ in ((xs[0], 1.0, C["net"], False), (xs[1], RUN["own"], C["nat"], False), (xs[2], RUN["sq"], C["das"], True),
                         (xs[3], RUN["ce"], C["das"], False)):
    top = Y0 - v * S
    if open_:
        b += R(x + 0.45, top + 0.45, CW - 0.9, Y0 - top - 0.45, "#FFFFFF", r=0.8, stroke=col, sw=0.9)
    else:
        b += R(x, top, CW, Y0 - top, col, r=0.8)
    b += T(x + CW / 2, top + 7.2, f"{v:.2f}", FS["note"], col if open_ else "#FFFFFF", anchor="middle", weight=700)
for x, l1, l2 in ((xs[0], "patch the", "whole state"), (xs[1], "mean difference", "(untrained)")):
    b += T(x + 9, Y0 + 7.6, l1, FS["note"], C["ink"], anchor="middle") + T(x + 9, Y0 + 14.4, l2, FS["note"], C["ink"], anchor="middle")
# the two DAS bars: one header over both, tied to them by a thin rule, then each bar's goal in the same words ("aimed at"), so the
# two read as one method with two targets, and the header and the targets under it do not run together as one phrase
xd0, xd1 = xs[2] + 1.0, xs[3] + CW - 1.0
b += T((xd0 + xd1) / 2, Y0 + 7.0, "DAS", FS["note"], C["das"], anchor="middle", weight=700)
b += L(xd0, Y0 + 8.9, xd1, Y0 + 8.9, C["das"], 0.5)
b += T(xs[2] + CW / 2, Y0 + 15.0, "aimed", FS["note"], C["ink"], anchor="middle")                  # each bar's objective, under it
b += T(xs[2] + CW / 2, Y0 + 21.6, "at T", FS["note"], C["ink"], anchor="middle")
b += T(xs[3] + CW / 2, Y0 + 15.0, "rewarded", FS["note"], C["ink"], anchor="middle")              # cross-entropy, as the abstract says it
b += T(xs[3] + CW / 2, Y0 + 21.6, "for more", FS["note"], C["ink"], anchor="middle")
# each test pair is read toward its own source's answer (each pair and its mirror), so no language is named
yb, x_l, x_r = Y0 + 24.6, xs[1] - 6, xs[3] + 22
b += L(x_l, yb, x_r, yb, C["gray"], 0.5) + L(x_l, yb, x_l, yb - 2.0, C["gray"], 0.5) + L(x_r, yb, x_r, yb - 2.0, C["gray"], 0.5)
b += T((x_l + x_r) / 2, yb + 7.0, "patch one direction", FS["note"], C["muted"], anchor="middle")
b += T(W - 2.0, Pp[1] - 2.6, f"predicted before training: {PRED:.2f}", FS["note"], C["tgray"], anchor="end", weight=600)

render("fig_law", W, H, b, OUT)
print(f"fig_law: {F1['task']}, {F1['base']} <- {F1['src']}; cross-entropy DAS {RUN['ce']:.2f} T, percentile {F1C['pct']:.1f} of {F1C['n']} "
      f"one-question tasks (median {F1C['median']:.2f}); angle {RUN['angle']:.1f} deg, scale {S:.1f} pt per T; sq {RUN['sq']:.2f}, "
      f"mean difference {RUN['own']:.2f}, predicted {RUN['pred']:.2f}; heights: top {PRED:.3f}, rest {1 - PRED:.3f}, whole 1")

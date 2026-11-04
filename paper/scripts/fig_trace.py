"""The trace's error at the readout (fig:trace, 0.6 text width, 238.5 pt): on every mirrored test set, the midpoint trace's
error divided by the base-point trace's (signed), against the size of the base-point trace's error on a log scale. The leading
order of App. A.1 (Proposition 9) makes the midpoint rule's error -1/2 of the trapezoid rule's: half the size and the opposite
sign, so the ratio lies on the line at -1/2 at every scale, and the plot shows both halves of the prediction at once (below 0:
opposite signs; at -1/2: half the size). A log-log plot of the two sizes hid the sign in a colour and the factor in a 0.3-decade
offset from a line it did not draw.

From experiments/results/score_readout.json: every main-population task with a fully mirrored test set, in every model whose
instruments held and whose final logits are not capped; x = |tr M(base point)/T - 1|, y = (tr M(midpoint)/T - 1) /
(tr M(base point)/T - 1), both on the test pairs (the trace test, trace in the scorer's rows). Marks: both errors nonzero by
their intervals (blue discs; the two above 0, of the same sign, are Pythia-160m's, asserted and named on the plot), one of them
not distinguishable from zero (light grey, where the ratio is noise). Subtitles: the counts, the signs and the ratio where both
are nonzero (median and middle half), each asserted equal to numbers_main's macros (paper/sections/numbers_main.tex). No key:
the marks are named where they are drawn. The leading-order line is thin solid dark grey (dark grey dashed is reserved for a
top), under the marks.
Usage: python -I paper/scripts/fig_trace.py [--out=<dir>]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, MK, SIXW, WORD, L, T, head, linscale, logscale, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
S = json.load(open(figdata.ROOT / "experiments/results/score_readout.json"))
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (figdata.ROOT / "paper/sections/numbers_main.tex").read_text()))
OK = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]
# first order's checks count the models whose final logits are not capped (Appendix C.4)
OK = [m for m in OK if not any(r.get("raw_over_cap") for r in S["rows"][m])]
mir = {m: [r for r in S["rows"][m] if r["main"] and r["defined"] and r["mirrored"] == "all"] for m in OK}
pts = [(r["trace"]["dev_base"][0], r["trace"]["dev_mid"][0]) for m in OK for r in mir[m]]
x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
slopes = {}
for m in OK:                                                                # the scorer's estimator: least squares through 0
    xm = np.array([r["trace"]["dev_base"][0] for r in mir[m]]); ym = np.array([r["trace"]["dev_mid"][0] for r in mir[m]])
    slopes[m] = float(xm @ ym / (xm @ xm))
    assert abs(slopes[m] - S["report"]["models"][m]["main"]["trace"]["slope_mid_on_base_mirrored"]) < 1e-9, m
lo_s, hi_s = min(slopes.values()), max(slopes.values())                   # the per-model range, whichever side of -1/2 each lies
assert str(len(pts)) == NUM["roMirrored"], "the subtitle's count and the text's (roMirrored) differ"
assert WORD.get(len(OK), str(len(OK))) == NUM["roModelsFOWord"], "the subtitle's models and the text's differ"
assert f"${hi_s:.2f}$" == NUM["roSlopeHigh"] and f"${lo_s:.2f}$" == NUM["roSlopeLow"], "per-model slopes and the text's differ"

# the marks' classes, as the text counts them: where both errors are nonzero by their intervals, opposite or the same sign
cls = [("opp" if r["trace"]["both_nonzero"] and r["trace"]["opposite"] else "same" if r["trace"]["both_nonzero"] else "zero")
       for m in OK for r in mir[m]]
both = [i for i, c in enumerate(cls) if c != "zero"]
ratio = np.array([y[i] / x[i] for i in both])
assert str(len(both)) == NUM["roOppositeDen"] and str(cls.count("opp")) == NUM["roOppositeNum"], "the signs' counts and the text's"
assert (f"${np.median(ratio):.2f}$", f"${np.quantile(ratio, 0.25):.2f}$", f"${np.quantile(ratio, 0.75):.2f}$") == \
    (NUM["roTraceRatioMed"], NUM["roTraceRatioQone"], NUM["roTraceRatioQthree"]), "the ratio's median and quartiles and the text's"
LO, HI = 1e-5, 1.0                                                          # the base-point error's size: five decades
RLO, RHI = -1.0, 1.6                                                        # the ratio, linear: every case inside (asserted)
ax, rr = np.abs(x), y / x
assert ax.min() > LO and ax.max() < HI and rr.min() > RLO and rr.max() < RHI, "every task inside the axes"
same_m = [m for m in OK for r in mir[m] if r["trace"]["both_nonzero"] and not r["trace"]["opposite"]]
assert same_m == ["pythia-160m"] * 2, f"the two cases of the same sign, named on the plot: {same_m}"

W = SIXW                                                                    # printed at 0.6 \textwidth
X0, X1, Y1 = 32.0, W - 8.0, 40.0
Y0 = Y1 + 120.0
X, Y = logscale(LO, HI, X0, X1), linscale(RLO, RHI, Y0, Y1)
mm = lambda v: f"{v:.2f}".replace("-", "−")  # noqa: E731
subs = [f"{len(pts)} mirrored cases in {WORD.get(len(OK), len(OK))} models; opposite signs in {cls.count('opp')} of the {len(both)}",
        f"with both errors nonzero; ratio: median {mm(np.median(ratio))}, interquartile range {mm(np.quantile(ratio, 0.25))} to "
        f"{mm(np.quantile(ratio, 0.75))}"]
assert all(tw(t_, FS["sub"]) < W - 2 for t_ in subs), "a subtitle line wider than the figure"
b = head(0.0, "", "The trace's error follows the theory", subs)
b += L(X0, Y0, X1, Y0, C["line"], 0.6) + L(X0, Y0, X0, Y1, C["line"], 0.6)  # the axes' baselines
b += L(X0, Y(0), X1, Y(0), C["line"], 0.6)                                  # 0: above it the two errors share a sign
b += L(X0, Y(-0.5), X1, Y(-0.5), C["tgray"], 0.6)                           # the leading order, under the marks
for c_ in ("zero", "opp", "same"):                                          # grey first, the cases with both nonzero over them
    for a_, q_, k in zip(ax, rr, cls):
        if k == c_:
            col, op = (C["grayl"], 0.9) if k == "zero" else (C["net"], MK["op"])
            b += f'<circle cx="{X(a_):.2f}" cy="{Y(q_):.2f}" r="{MK["r"]}" fill="{col}" fill-opacity="{op}"/>'
# every mark named where it is drawn
b += T(X1 - 1.0, Y(-0.5) - 14.0, "the leading order: −0.5,", FS["note"], C["tgray"], anchor="end")   # over the line, right: the band
b += T(X1 - 1.0, Y(-0.5) - 7.0, "half the size, the opposite sign", FS["note"], C["tgray"], anchor="end")  # bends under it there
b += T(X1 - 1.0, Y(-0.88), "both errors nonzero", FS["note"], C["net"], anchor="end")        # under the band's bend
sx = [(a_, q_) for a_, q_, k in zip(ax, rr, cls) if k == "same"]
b += T(X(max(a_ for a_, _ in sx)) + 5.0, Y(sum(q_ for _, q_ in sx) / 2) + 2.0, "the same sign: two tasks of Pythia-160m", FS["note"], C["net"])
b += T(X0 + 2.0, Y(0.62), "one error not distinguishable from 0", FS["note"], C["gray"])
for e in range(-5, 1):                                                      # decades
    lab = "1" if e == 0 else f"{10.0 ** e:.{-e}f}"
    b += T(X(10.0 ** e), Y0 + 7.6, lab, FS["tick"], C["gray"], anchor="middle")
for v, t_ in ((-1, "−1"), (-0.5, "−0.5"), (0, "0"), (1, "1")):
    b += T(X0 - 3.0, Y(v) + 2.0, t_, FS["tick"], C["tgray"] if v == -0.5 else C["gray"], anchor="end", weight=700 if v == -0.5 else 400)
b += T((X0 + X1) / 2, Y0 + 16.2, "base-point trace's error, its size over T", FS["axis"], C["muted"], anchor="middle")
b += T(9.0, (Y0 + Y1) / 2, "midpoint error over base-point error", FS["axis"], C["muted"], anchor="middle", rotate=-90)
H = Y0 + 20.0
render("fig_trace", W, H, b, OUT)
print(f"fig_trace: {len(pts)} tasks in {len(OK)} models; both nonzero {len(both)}, opposite {cls.count('opp')}, same sign "
      f"{cls.count('same')}; ratio median {np.median(ratio):.3f}; per-model slopes {lo_s:.3f} to {hi_s:.3f}; height {H:.1f} pt")

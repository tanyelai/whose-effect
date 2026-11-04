"""The visual guide's target figure (Section 6.3): inside the network too, a learned effect follows the target of its objective.

1. One cell, the map's verb at layer 10 in Qwen2.5-0.5B's movie review (means over its 100 test pairs): the answer's moves under
   whole-state patching (G, blue), and under a patch of one direction found four ways: cross-entropy DAS (rewarded for more;
   three runs), DAS with a per-pair squared error aimed at T (one run), the same aimed at G (three runs), and the untrained mean
   difference (teal, as in fig:guidec). A target is a small triangle on its objective's lane, black for T and blue for G: T for the squared error aimed
   at T, G for the one aimed at G.
2. Every live cell of the six small models (Section 6's population, as numbers_main gathers it; the first bin takes every G
   below 0.02 T, negatives too, and the last every G from T up, so no cell is left out): the learned effect against G, as the
   median in bins of G, with the line where an effect equals G. One loss, two targets: aimed at G, the squared error lies on
   that line (slope inSqGSlope, intercept inSqGIntercept, asserted); aimed at T, it lies above it where G is below T and meets
   it as G nears T. Cross-entropy, another loss, is not drawn: it lies close to the squared error aimed at T while G is well
   below T, but goes on past T as G nears T (asserted; the caption says so). The mean difference is left out here: outside
   IOI it lies on the line too (the text).
Every number is a mean over T; no key: each mark is named where it is drawn.
Usage: python -I paper/scripts/fig_guide_f.py [--out=<dir>]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, T, R, circ, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, TASK, CELL = "Qwen2.5-0.5B", "sentiment_movie", (11, 3)
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (figdata.ROOT / "paper/sections/numbers_main.tex").read_text()))
mt = next(c for c in json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.json"))["cells"]
          if (c["layer"], c["region"]) == CELL)["mean_T"]
V = dict(G=mt["own_normal"], ce=float(np.mean([mt[f"das_ce_s{x}_normal"] for x in range(3)])), sqT=mt["das_sq_T_s0_normal"],
         sqG=float(np.mean([mt[f"das_sq_G_s{x}_normal"] for x in range(3)])), uh=mt["u_hat_normal"])
assert abs(V["G"] - 0.09) < 0.005 and abs(V["ce"] - 0.86) < 0.01 and abs(V["sqG"] - V["G"]) < 0.01 and V["G"] < V["sqT"] < 1, V
# every live cell of the six small models, as numbers_main gathers them
POST = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json")); POP = set(POST["cell_classes"])
SI = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.json"))
scored = {(c["model"], c["task"]) for cs in SI["cells"].values() for c in cs if f"{c['model']}/{c['task']}/{c['cell']}" in POP}
rows = []
for f in sorted((figdata.ROOT / "experiments/results/inside").glob("inside_*.json")):
    mt_ = [(m, t) for m, t in scored if f.name == f"inside_{m}_{t}.json"]
    if not mt_:
        continue
    for c in json.load(open(f))["cells"]:
        if c.get("live") and f"{mt_[0][0]}/{mt_[0][1]}/L{c['layer']}r{c['region']}" in POP:
            m = c["mean_T"]
            rows.append((m["own_normal"], np.mean([m[f"das_ce_s{x}_normal"] for x in range(3)]), m["das_sq_T_s0_normal"],
                         np.mean([m[f"das_sq_G_s{x}_normal"] for x in range(3)])))
A = np.array(rows); Gc, CEc, SQT, SQG = A.T
N_ALL = len(A)
assert f"{N_ALL:,}".replace(",", "{,}") == NUM["inLive"], N_ALL
sl, ic = np.linalg.lstsq(np.vstack([Gc, np.ones_like(Gc)]).T, SQG, rcond=None)[0]
assert f"{sl:.2f}" == NUM["inSqGSlope"] and (f"{abs(ic):.2f}" if abs(ic) < 0.005 else f"{ic:.2f}") == NUM["inSqGIntercept"], (sl, ic)
EDGES = [-np.inf, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, np.inf]           # every live cell in some bin
BINS = []
for lo, hi in zip(EDGES, EDGES[1:]):
    m_ = (Gc >= lo) & (Gc < hi)
    assert m_.sum() >= 200, (lo, hi, m_.sum())                                # every point a median of many cells
    BINS.append((float(np.median(Gc[m_])), float(np.median(CEc[m_])), float(np.median(SQT[m_])), float(np.median(SQG[m_]))))
assert sum(((Gc >= lo) & (Gc < hi)).sum() for lo, hi in zip(EDGES, EDGES[1:])) == N_ALL, "a cell outside the bins"
# the caption's cross-entropy: near the squared error aimed at T while G is well below T, past T as G nears T, where the
# squared error stops at T
assert all(abs(bn[1] - bn[2]) < 0.06 for bn in BINS if bn[0] < 0.75) and all(bn[1] > 1.3 and bn[2] < 1.0 for bn in BINS[-2:])

W = NIPSW
FN = FS["axis"]
mm = lambda v: f"{v:.2f}"  # noqa: E731
def stepmark(x, y, n, title):
    return circ(x + 6, y - 2.4, 5.0, C["ink"]) + T(x + 6, y, str(n), FN, C["paper"], anchor="middle", weight=700) + \
        T(x + 14.5, y, title, FS["title"], weight=600)
def target(x, y, col):
    """An objective's target: a small triangle over its lane, black for T (as the bold T tick), blue for G (blue is G)."""
    return f'<path d="M{x - 2.4:.2f},{y - 4.6:.2f} L{x + 2.4:.2f},{y - 4.6:.2f} L{x:.2f},{y - 1.0:.2f} z" fill="{col}"/>'

TTL = "The excess is the objective's: a learned effect follows its target"
SUB = f"{MODEL}, the movie review: the verb at layer 10, then every live cell of six models; means over the test pairs, in T"
assert tw(TTL, FS["letter"], 600) < W - 2 and tw(SUB, FS["sub"]) < W
b = T(0, 10, TTL, FS["letter"], weight=600) + T(0, 19.6, SUB, FS["sub"], C["muted"])

# ---- 1: one cell, five patches, each on the answer's scale; a target a triangle on its lane
Y1 = 38.0
b += stepmark(0, Y1, 1, "One cell, five patches")
PT = 100.0                                                                   # points per T, in both steps
LX0, LX1 = 80.0, 80.0 + PT                                                   # the lanes' 0 and T
XL = lambda v: LX0 + (LX1 - LX0) * v  # noqa: E731
LANES = [("whole-state patching", C["net"], V["G"], None, "the network's own"),
         ("cross-entropy DAS", C["das"], V["ce"], None, "rewarded for more"),
         ("squared-error DAS", C["das"], V["sqT"], (1.0, C["ink"]), "aimed at T"),
         ("squared-error DAS", C["das"], V["sqG"], (V["G"], C["net"]), "aimed at G"),
         ("the mean difference", C["nat"], V["uh"], None, "untrained")]
LP, Y1L = 22.4, Y1 + 10.6                                                    # the first lane's name on the first line under the title
for q, (name, col, v, tgt, note) in enumerate(LANES):
    y = Y1L + q * LP
    b += T(LX0 - 6, y + 0.4, name, FS["tick"], col, anchor="end", weight=600)
    b += T(LX0 - 6, y + 6.6, note, FS["tick"], C["muted"], anchor="end")
    b += R(LX0, y - 1.9, XL(v) - LX0, 3.8, col, r=0.6)
    b += T(XL(v) + 3.0, y + 2.2, f"{mm(v)} T", FS["tick"], col, weight=600)
    if tgt is not None:
        b += target(XL(tgt[0]), y - 1.9, tgt[1])
        if tgt[0] == 1.0:
            b += T(XL(tgt[0]) + 5.0, y - 1.6, "its target", FS["tick"], C["muted"])
yax1 = Y1 + 119.0                                                            # on step 2's x-axis
b += L(LX0, Y1L - 7.0, LX0, yax1, C["grayl"], 0.6)                           # 0
b += L(XL(1), Y1L + 2 * LP - 6.5, XL(1), yax1, C["grayl"], 0.6, dash="1.4 1.6")   # T, from the lane aimed at it down
b += L(LX0, yax1, LX1 + 6, yax1, C["grayl"], 1.0)
for t, txt in ((0, "0"), (1, "T")):
    b += L(XL(t), yax1 - 2.0, XL(t), yax1 + 2.0, C["gray"], 0.6) + T(XL(t), yax1 + 8.6, txt, FS["tick"], C["ink"] if t else C["muted"],
                                                                       anchor="middle", weight=600 if t else 400)

# ---- 2: every cell: the same squared error aimed at T and at G, against G (medians in bins of G); the line where an effect is G
GMAX = EMAX = 1.1
X0, Yb = 262.0, Y1 + 119.0                                                   # square: PT points per T across and up
X1, Yt = X0 + PT * GMAX, Yb - PT * EMAX
XG = lambda g: X0 + (X1 - X0) * g / GMAX  # noqa: E731
YE = lambda e: Yb - (Yb - Yt) * e / EMAX  # noqa: E731
b += stepmark(214.0, Y1, 2, "Every cell: one loss, two targets")
b += L(X0, Yb, X1, Yb, C["grayl"], 1.0) + L(X0, Yb, X0, Yt, C["grayl"], 1.0)
for t, txt in ((0, "0"), (1, "T")):
    b += L(XG(t), Yb - 2, XG(t), Yb + 2, C["gray"], 0.6) + T(XG(t), Yb + 8.6, txt, FS["tick"], C["ink"] if t else C["muted"], anchor="middle",
                                                               weight=600 if t else 400)
    b += L(X0 - 2, YE(t), X0 + 2, YE(t), C["gray"], 0.6) + T(X0 - 4, YE(t) + 2.1, txt, FS["tick"], C["ink"] if t else C["muted"], anchor="end",
                                                               weight=600 if t else 400)
b += T((X0 + X1) / 2, Yb + 16.6, "whole-state patching, G", FS["tick"], C["net"], anchor="middle", weight=600)
b += L(XG(0), YE(0), XG(GMAX), YE(GMAX), C["net"], 0.8)                     # where an effect equals G: whole-state patching's own
b += T(XG(GMAX) + 2.0, YE(GMAX) + 2.0, "= G", FS["tick"], C["net"], weight=600)
for k_ in (2, 3):
    pts = [(XG(bn[0]), YE(bn[k_])) for bn in BINS]
    b += P("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts), C["das"], 1.0)
    for x, y in pts:
        b += circ(x, y, 1.3, C["das"])
iT = 7                                                                       # the bin where "aimed at T" is named, above its line
b += T(XG(BINS[iT][0]) - 4.0, YE(BINS[iT][2]) - 6.0, "aimed at T", FS["tick"], C["das"], anchor="end", weight=600)
assert XG(BINS[iT][0]) - 4.0 - tw("aimed at T", FS["tick"], 600) > X0 + 3, "the label crosses the axis"
iG = 7                                                                       # and "aimed at G", under the line it lies on
b += T(XG(BINS[iG][0]) + 3.0, YE(BINS[iG][3]) + 9.0, "aimed at G", FS["tick"], C["das"], weight=600)
H = Yb + 22.0
render("fig_guide_f", W, H, b, OUT)
print(f"fig_guide_f: the verb {({k: round(v, 3) for k, v in V.items()})}; {N_ALL} cells, aimed at G slope {sl:.3f} intercept {ic:.3f}; "
      f"bins {[tuple(round(x, 2) for x in bn) for bn in BINS]}")

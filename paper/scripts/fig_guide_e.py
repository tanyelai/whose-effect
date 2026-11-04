"""The visual guide's report figure (fig:guidee): what to report beside a learned effect.

The three steps of Section 8 as a decision flow, as Algorithm 1 has them: measure G* on the test pairs with its interval (the
context chosen on the training pairs); if G*'s interval lies above zero, report e / G*, an excess over G* where its interval
lies above one; if not, e is no evidence that the network uses the cell, and the cell is off-route where e lies significantly
above 0 and G* (otherwise e and G* are reported with their intervals); at every cell, freeze attention. Each outcome shows one
cell's bars, Figure 6's with G* in place of G (no T line: the decision compares e with G*). Then the report, on six cells of
Qwen2.5-0.5B: five of the movie review's map and the last word, at layer 12, of a second template ("I heard the doctor use the
word vaccination and I felt very"), an off-route cell large enough to see. e is cross-entropy DAS, the mean of its three runs.
Every reading names what freezing does by one rule, asserted: "mostly via attention" where it removes 70% of e or more, "little"
where 25% or less, "not" where none. Intervals are the maps' own (95%, resampling a pair with its mirror; paper/scripts/
guidedata.py, and for e / G* the same resamples of the ratio of means). Every number is a mean over the test pairs, over T.
Usage: python -I paper/scripts/fig_guide_e.py [--out=<dir>]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from guidedata import map_cells  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, R, T, arrows, circ, render, tw  # noqa: E402

sys.path.insert(0, str(figdata.ROOT / "experiments/lib"))
from measure import boot_counts, boot_ratio, interval  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, TASK = "Qwen2.5-0.5B", "sentiment_movie"
cells, _, LAYS, _ = map_cells(MODEL, TASK)
RO = max(LAYS)
MED = "sentiment_medical"                                                    # the same model's doctor sentence ("I heard the doctor use the
#                                                                              word vaccination and I felt very"): a clear off-route cell
MK = ("med", 13, 2)                                                          # its last word, "very", at layer 12 (the code's 13)
cells[MK] = map_cells(MODEL, MED)[0][(13, 2)]
J = json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.json"))
Z = np.load(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.npz")
CL = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"])
def ratio(key):
    k = f"L{key[0]}r{key[1]}"
    gs = Z[f"{k}/gstar_train"].astype(float)
    ce = np.mean([Z[f"{k}/e/das_ce_s{s}_normal"] for s in range(3)], axis=0).astype(float)
    r = float(ce.mean() / gs.mean())
    return r, interval(r, boot_ratio(ce, gs, CL))
ROWS = [  # cell, its name, its kind
    ((3, 1), "“sweet”, layer 2", "agree"),
    ((9, 1), "“sweet”, layer 8", "agree"),
    ((11, 3), "“cherished”, layer 10", "excess"),
    ((RO, 4), "the readout", "excess"),
    ((17, 2), "“I”, layer 16", "off"),
    (MK, "“very”, layer 12 (doctor)", "off"),
]
for key, _, kind in ROWS:                                                 # each row is the case its reading names
    c = cells[key]
    if kind == "off":
        assert c["off"], key                                                 # off-route; whether through attention: READ, asserted below
    else:
        assert c["carried"], key
        r, (lo, hi) = ratio(key)
        assert (kind == "excess") == (lo > 1), (key, r, lo, hi)
assert cells[(9, 1)]["diff_ci"][0] <= 0 <= cells[(9, 1)]["diff_ci"][1] and cells[(3, 1)]["diff_ci"][1] < 0
assert cells[(11, 3)]["fr"] < 0.2 * cells[(11, 3)]["das"] and abs(cells[(RO, 4)]["fr"] - cells[(RO, 4)]["das"]) < 0.02
COL = dict(agree=C["net"], excess=C["das"], off=C["off"])
row0 = json.load(open(figdata.ROOT / f"experiments/data/{MODEL}/sentiment_templates/{TASK}/test.json"))[0]
LASTW = {r: row0["base"][r].replace("\n", " ").split()[-1] for r in (1, 2, 3)}       # a cell patches its region's last token
LASTW[MK] = json.load(open(figdata.ROOT / f"experiments/data/{MODEL}/sentiment_templates/{MED}/test.json"))[0]["base"][2].split()[-1]
for key, name, _ in ROWS:                                                    # each row is named by the word whose state it patches
    w_ = LASTW[MK] if key == MK else LASTW.get(key[1], "")
    assert key[0] == RO or name.startswith("\u201c" + w_.strip(",") + "\u201d"), (key, name, LASTW)
assert cells[MK]["das"] > 0.4 and cells[MK]["fr"] > 0.75 * cells[MK]["das"], "the medical cell: large, little through attention"

W = NIPSW
FN = FS["axis"]                                                             # the guides' running text
b = arrows(("ak", C["ink"]), ("ag", C["gray"]), size=4.2)
b += T(0, 10, "What to report beside a learned effect", FS["letter"], weight=600)

# ---- one grid: three equal columns over the full width
G = 12.0
LW = (W - 2 * G) / 3
CX3 = [LW / 2 + k * (LW + G) for k in range(3)]                             # the three column centres
SPAN12 = (0.0, 2 * LW + G)                                                   # columns 1-2
XD = ((SPAN12[0] + SPAN12[1]) / 2 + CX3[2]) / 2                              # the question, midway between its two branches
def wrap(text, width, size=FN, weight=400):
    out, cur = [], ""
    for wd in text.split(" "):
        t = (cur + " " + wd).strip()
        if tw(t, size, weight) > width and cur:
            out.append(cur); cur = wd
        else:
            cur = t
    return out + [cur]
def bars(x0, y, key, tpt, h=2.6):
    """Figure 6's cell: whole-state patching in its best context (G*, blue) over the learned effect e (orange), from x0, tpt points
    per T. No T line: the decision compares e with G*."""
    c = cells[key]
    o = R(x0, y - h - 0.3, tpt * max(c["gs"], 0), h, C["net"], r=0.4) if c["gs"] > 0.004 else ""
    return o + (R(x0, y + 0.3, tpt * max(c["das"], 0), h, C["das"], r=0.4) if c["das"] > 0.004 else "")
def stripes(cx, cy, bw, bh, col, seg=5):
    return "".join(L(cx - bw / 2 + q * bw / seg, cy - bh / 2 + 0.9, cx - bw / 2 + q * bw / seg, cy + bh / 2 - 0.9, col, 0.4,
                     cap="butt") for q in range(1, seg))
def dirbox(x, y):                                                            # Figure 2's patch of d's part: the pale orange state
    return R(x - 7, y - 3, 14, 6, C["dasl"], r=1.2, stroke=C["das"], sw=0.8) + stripes(x, y, 14, 6, C["das"])
LH = 9.6                                                                     # line pitch inside a box
MX = 13.0                                                                    # every mark on one axis, the state glyph 5 pt in from the edge
def box(x0, x1, y, head_, lines, mark=None, head_col=C["ink"], stroke=C["line"], fill=C["faint"], h=None, sw=0.7):
    """A rounded box from x0 to x1: a mark (a number or a glyph) at the left, a bold first line, muted lines under it."""
    tx = x0 + MX + 12                                                       # the text, clear of the widest mark (14 pt)
    body = [ln for t in lines for ln in wrap(t, x1 - tx - 6)]
    h = h or 10 + LH * (1 + len(body))
    o = R(x0 + 0.6, y, x1 - x0 - 1.2, h, fill, r=4, stroke=stroke, sw=sw)
    if isinstance(mark, str):
        o += circ(x0 + MX, y + 10.2, 5.0, C["ink"]) + T(x0 + MX, y + 12.6, mark, FN, C["paper"], anchor="middle", weight=700)
    elif mark:
        o += mark(x0 + MX, y + 10.2)
    o += T(tx, y + 12.6, head_, FN, head_col, weight=700)
    assert tx + tw(head_, FN, 700) < x1 - 4, head_
    for q, ln in enumerate(body):
        o += T(tx, y + 12.6 + LH * (q + 1), ln, FN, C["muted"] if fill == C["faint"] else C["ink"])
    return o, h
down = lambda x, y0, y1: L(x, y0, x, y1 - 1.2, C["ink"], 0.7, marker="ak")  # noqa: E731
GAP = 7.0

y = 18.0
o, h = box(0, W, y, "A learned direction moves the answer by e at a cell", ["the effect a method reports"], mark=dirbox)
b += o; y += h
b += down(XD, y, y + GAP); y += GAP
S1 = "the cell alone or what it adds to one other region of its layer, chosen on training pairs; at a complete cut, G* = T"
assert len(wrap(S1, W - 26)) == 1, S1                                        # one line
o, h = box(0, W, y, "Measure G* on test pairs, whole-state patching of the cell in its best context, with its interval", [S1],
           mark="1")
b += o; y += h
b += down(XD, y, y + GAP); y += GAP
DW, DH = 136.0, 22.0                                                         # the question
b += (f'<path d="M{XD:.2f},{y:.2f} L{XD + DW / 2:.2f},{y + DH / 2:.2f} L{XD:.2f},{y + DH:.2f} L{XD - DW / 2:.2f},{y + DH / 2:.2f} z" '
      f'fill="{C["paper"]}" stroke="{C["ink"]}" stroke-width="0.7"/>')
b += T(XD, y + DH / 2 + 2.3, "Is G*'s interval above zero?", FN, anchor="middle", weight=600)
ym = y + DH / 2
yb = y + DH + GAP + 2
def elbow(x0, y0, x1, y1, r=4.0):
    sgn = 1 if x1 > x0 else -1
    return f"M{x0:.2f},{y0:.2f} H{x1 - sgn * r:.2f} Q{x1:.2f},{y0:.2f} {x1:.2f},{y0 + r:.2f} V{y1 - 1.2:.2f}"
X2 = (SPAN12[0] + SPAN12[1]) / 2
b += P(elbow(XD - DW / 2, ym, X2, yb), C["ink"], 0.7, marker="ak") + P(elbow(XD + DW / 2, ym, CX3[2], yb), C["ink"], 0.7, marker="ak")
b += T((XD - DW / 2 + X2) / 2, ym - 3.5, "yes", FN, anchor="middle", weight=700)
b += T((XD + DW / 2 + CX3[2]) / 2, ym - 3.5, "no", FN, anchor="middle", weight=700)
h2n = 10 + LH * 2
o, _ = box(SPAN12[0], SPAN12[1], yb, "Report e / G* with its interval", ["e as a multiple of G*"], mark="2", h=h2n)
b += o
o, _ = box(CX3[2] - LW / 2, CX3[2] + LW / 2, yb, "Do not read e as use", ["report e and G*, with intervals"], mark="3", h=h2n)
b += o
yl = yb + h2n + GAP
for x in CX3:
    b += down(x, yb + h2n, yl)
LEAVES = [("e / G*'s interval not above one", "no significant excess over G*", C["net"], (9, 1)),
          ("e / G*'s interval above one", "an excess over G*: the objective's choice, not by itself stronger evidence", C["das"], (11, 3)),
          ("off-route", "if e lies significantly above 0 and G*", C["off"], MK)]
TL = 34.0                                                                   # a leaf's glyph: points per T
hl = 0.0
for x, (head_, text, col, key) in zip(CX3, LEAVES):
    x0_, x1_ = x - LW / 2, x + LW / 2
    ls = wrap(text, LW - 12)
    hh = 10 + LH + 12 + LH * len(ls)
    hl = max(hl, hh)
for x, (head_, text, col, key) in zip(CX3, LEAVES):
    x0_, x1_ = x - LW / 2, x + LW / 2
    b += R(x0_ + 0.6, yl, LW - 1.2, hl, C["paper"], r=4, stroke=C["line"], sw=0.7)
    b += T(x0_ + 6, yl + 12.6, head_, FN, col, weight=700)
    b += bars(x0_ + 6, yl + 22.0, key, TL)
    nm = next(n_ for k_, n_, _ in ROWS if k_ == key)
    xn = x0_ + 6 + TL * max(cells[key]["das"], cells[key]["gs"]) + 6                 # the cell's name, just past its bars
    assert xn + tw(nm, FS["tick"]) < x1_ - 4, key
    b += T(xn, yl + 24.2, nm, FS["tick"], C["muted"])
    for q, ln in enumerate(wrap(text, LW - 12)):
        b += T(x0_ + 6, yl + 22.0 + 12.6 + LH * q, ln, FN, C["muted"])
yf = yl + hl + GAP                                                          # freezing, for every cell
for x in CX3:
    b += down(x, yl + hl, yf)
def lockmark(x, y):                                                        # Figure 7's padlock: attention held at the base run's
    return (R(x - 3.6, y - 1.2, 7.2, 5.6, C["tgray"], r=0.9) +
            P(f"M{x - 2.3:.2f},{y - 1.2:.2f} v-2 a2.3,2.3 0 0 1 4.6,0 v2", C["tgray"], 0.9))
o, hf = box(0, W, yf, "Then, at every cell, freeze attention",
            ["hold every attention pattern after the cell at the base run's; the share of e this removes ran through attention"],
            mark=lockmark)
b += o
yflow = yf + hf + 18

# ---- the report, on the same edges
YT = yflow
b += T(0, YT, "The report: each learned effect beside whole-state patching", FS["title"], weight=600)
SUB = f"{MODEL}, the movie review, then a second template; e is cross-entropy DAS, the mean of three runs; means over test pairs, in T"
assert tw(SUB, FS["sub"]) < W - 2, SUB
b += T(0, YT + 9.6, SUB, FS["sub"], C["muted"])
COLS = dict(box=0.0, e=96.0, g=168.0, r=234.0, f=284.0, read=292.0)   # cell names: left edge; numbers: right edges (8 pt or more between)
hy = YT + 42
b += R(COLS["e"] + 6, hy - 16.4, W - COLS["e"] - 6, 17.4 + 12 * len(ROWS) + 2.5, C["netl"], r=3, op=0.35)
def group(x0, x1, y, text, col, italic=True):
    return (L(x0, y, x1, y, col, 0.6) + L(x0, y, x0, y + 2.5, col, 0.6) + L(x1, y, x1, y + 2.5, col, 0.6) +
            T((x0 + x1) / 2, y - 3, text, FN, col, anchor="middle", italic=italic, weight=600))
b += group(COLS["e"] + 10, W - 2, hy - 19.4, "what the check adds", C["net"])
for key, t1, t2, an in (("box", "", "cell", "start"), ("e", "", "e", "end"), ("g", "G*", "[interval]", "end"),
                        ("r", "e / G*", "[interval]", "end"), ("f", "e, frozen", "(removed)", "end"), ("read", "", "reading", "start")):
    b += (T(COLS[key], hy - 7.4, t1, FN, C["muted"], anchor=an, weight=600) if t1 else "") + T(COLS[key], hy, t2, FN, C["muted"], anchor=an, weight=600)
b += L(0, hy + 3.5, W, hy + 3.5, C["gray"], 0.6)
fmt = lambda v, d=2: f"{v:.{d}f}".replace("-0." + "0" * d, "0." + "0" * d)  # noqa: E731
READ = {(3, 1): "below G*, little via attention", (9, 1): "within G*, little via attention",
        (11, 3): "excess, mostly via attention", (RO, 4): "excess, not via attention",
        (17, 2): "off-route, mostly via attention", MK: "off-route, little via attention"}
for key, txt in READ.items():                                               # one rule for what freezing does, checked
    c = cells[key]; share = 1 - c["fr"] / c["das"]
    word = txt.split(", ")[1].split()[0]
    assert {"mostly": share >= 0.70, "little": 0 < share <= 0.25, "not": abs(share) < 0.005}[word], (key, share)
for q, (key, name, kind) in enumerate(ROWS):
    c = cells[key]; yy = hy + 14 + 12 * q
    etxt = fmt(c["das"], 3 if c["das"] < 0.1 else 2)
    assert COLS["box"] + tw(name, FN) + 8 < COLS["e"] - tw(etxt, FN, 700), name
    b += T(COLS["box"], yy, name, FN)
    b += T(COLS["e"], yy, etxt, FN, anchor="end", weight=700)
    dg = 3 if c["gs"] < 0.2 else 2
    gtxt = ("1 (exact)" if key[0] == RO else f"{fmt(c['gs'], dg)} [{fmt(c['g_ci'][0], dg)}, {fmt(c['g_ci'][1], dg)}]")
    if c["off"]:
        assert c["g_ci"][0] <= 0 < c["g_ci"][1] < 0.05, "off-route: G*'s interval includes 0"
        dg_ = 4 if c["g_ci"][1] < 0.001 else 3
        gtxt = f"\u2248 0, at most {math.ceil(c['g_ci'][1] * 10 ** dg_) / 10 ** dg_:.{dg_}f}"
    b += T(COLS["g"], yy, gtxt, FN, C["off"] if c["off"] else C["ink"], anchor="end")
    if kind == "off":
        b += T(COLS["r"], yy, "\u2014", FN, anchor="end")
    else:
        r, (lo, hi) = ratio(key)
        dig = 1 if r >= 5 else 2
        b += T(COLS["r"], yy, f"{r:.{dig}f} [{lo:.{dig}f}, {hi:.{dig}f}]", FN, COL[kind] if kind == "excess" else C["ink"],
               anchor="end", weight=700 if kind == "excess" else 400)
    rm = f"{100 * (1 - c['fr'] / c['das']):.0f}".replace("-0", "0")         # the share of e that freezing removes
    b += T(COLS["f"], yy, f"{fmt(c['fr'], 3 if c['das'] < 0.1 else 2)} ({rm}%)", FN, anchor="end")
    b += T(COLS["read"], yy, READ[key], FN, COL[kind], weight=600)
    assert COLS["read"] + tw(READ[key], FN, 600) < W, READ[key]
    if q < len(ROWS) - 1:
        b += L(0, yy + 4.5, W, yy + 4.5, C["line"], 0.35)
yend = hy + 14 + 12 * (len(ROWS) - 1) + 5
b += L(0, yend, W, yend, C["gray"], 0.6)
H = yend + 3

render("fig_guide_e", W, H, b, OUT)
print("fig_guide_e: " + "; ".join(f"{name} e {cells[k]['das']:.2f} G* {cells[k]['gs']:.2f}" for k, name, _ in ROWS))

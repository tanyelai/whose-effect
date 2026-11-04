"""The visual guide's attention figure (fig:guided): how DAS goes further than the whole state inside the network.

Qwen2.5-0.5B, the movie review, the map's verb at layer 10 (the map's cell layer 11; layer l is the output of block l): the
cell where whole-state patching moves the answer 0.09 T and DAS 0.86 T (means over the 100 test pairs and DAS's three runs).
In fig:guidea's vocabulary: states are striped boxes (only the two the steps use, the verb's and the last word's; the words
between are an ellipsis), the answer is a bar on the scale 0, T, 2T, 3T, blue for whole-state patching and orange for DAS;
a beam from the last word to the verb, as wide as the attention weight of head 3 in block 12 (the second block that reads
layer 10, the head whose attention DAS raises most). Steps 1-4 on the 50 test pairs whose source is negative, with DAS's
first run (the run whose attention was saved): (1) the base run; (2) the verb's whole state from the source run, attention
stays; (3) only its part along DAS's direction, attention rises; (4) every later block, the attention to the verb summed over
heads, whole state against d's part; (5) the verb with every attention pattern after the cell held at the base run's (a
padlock), the free result dashed; (6) the contrast: the last word at layer 20, where holding attention changes nothing, so
the excess there is not attention's but the geometry of a part (fig:guidec). The share over every live cell (freezing
removes inExcessAttnAll of DAS's summed excess over G and 1 - inOwnKeep of whole-state patching's) is in the caption and
the text, asserted against numbers_main's macros. No key: each mark is named where it is drawn.
Usage: python -I paper/scripts/fig_guide_d.py [--out=<dir>]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from guidedata import map_cells  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, R, T, arrows, circ, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, TASK, CELL, LASTW = "Qwen2.5-0.5B", "sentiment_movie", (11, 3), (21, 4)
Z = np.load(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.npz")
rows = json.load(open(figdata.ROOT / f"experiments/data/{MODEL}/sentiment_templates/{TASK}/test.json"))
neg = np.array([r["src_label"].strip() == "terrible" for r in rows])
Tt = Z["T"].astype(float)
def eff(cell, n, m):
    return float(Z[f"L{cell[0]}r{cell[1]}/e/{n}"].astype(float)[m].mean() / Tt[m].mean())
k = f"L{CELL[0]}r{CELL[1]}"
HB, HH = 1, 3                                                                # block 12 (the second after the cell), head 3
B0 = CELL[0]
HW = {n: float(Z[f"{k}/door/{a}"][neg][:, HB, HH].mean()) for n, a in (("base", "w_base"), ("own", "w_own"), ("das", "w_das_ce_s0"))}
ATT = {n: Z[f"{k}/door/{a}"][neg].mean(0).sum(1) for n, a in (("own", "w_own"), ("das", "w_das_ce_s0"), ("base", "w_base"))}
NB = len(ATT["base"]); NH = Z[f"{k}/door/w_base"].shape[2]                      # blocks B0 .. B0 + NB - 1 read the verb after the patch
assert B0 + NB - 1 == 23 and np.abs(ATT["own"] - ATT["base"]).max() < 0.06 and ATT["das"].max() > 5 * ATT["base"].max() * 0.9
HW["mir"] = float(Z[f"{k}/door/w_das_ce_s0"][~neg][:, HB, HH].mean()); HW["mir_base"] = float(Z[f"{k}/door/w_base"][~neg][:, HB, HH].mean())
E = dict(own=eff(CELL, "own_normal", neg), own_fr=eff(CELL, "own_frozen", neg), das=eff(CELL, "das_ce_s0_normal", neg),
         fr=eff(CELL, "das_ce_s0_frozen", neg), mir=eff(CELL, "das_ce_s0_normal", ~neg), all1=eff(CELL, "das_ce_s0_normal", np.ones_like(neg)))
LW = dict(own=eff(LASTW, "own_normal", neg), own_fr=eff(LASTW, "own_frozen", neg), das=eff(LASTW, "das_ce_s0_normal", neg),
          fr=eff(LASTW, "das_ce_s0_frozen", neg))
cells, _, _, _ = map_cells(MODEL, TASK)
assert abs(cells[CELL]["own"] - 0.09) < 0.005 and abs(cells[CELL]["das"] - 0.86) < 0.005, "the map's verb at layer 10"
assert E["fr"] < 0.1 * E["das"] and E["mir"] < 0.05 and HW["das"] > 10 * HW["base"], "steps 3-5"
assert abs(LW["fr"] - LW["das"]) < 0.03 and LW["das"] > 2 * LW["own"], "step 6: holding attention changes nothing at the last word"
STARTS = [(eff(CELL, f"das_ce_s{s}_normal", neg), eff(CELL, f"das_ce_s{s}_frozen", neg)) for s in (1, 2)]
assert all(e_ > 2 * f_ for e_, f_ in STARTS), STARTS                           # the same pattern in the other two runs
row0 = rows[0]
wb = [w.replace("<|endoftext|>", "").replace("\n", "").strip() for w in row0["base"]]
MIR = next(r for r in rows if [w.strip() for w in r["base"][1:4:2]] == [w.strip() for w in row0["src"][1:4:2]])
wm = [w.replace("<|endoftext|>", "").replace("\n", "").strip() for w in MIR["base"]]
assert (wb[1], wb[3], wm[1], wm[3]) == ("sweet", "cherished", "vile", "hated")
# 7: every live cell of the six small models, as numbers_main gathers them: what holding attention removes
NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (figdata.ROOT / "paper/sections/numbers_main.tex").read_text()))
POST = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.post.json")); POP = set(POST["cell_classes"])
SI = json.load(open(figdata.ROOT / "experiments/results/inside_sets/small_all.json"))
scored = {(c["model"], c["task"]) for cs in SI["cells"].values() for c in cs if f"{c['model']}/{c['task']}/{c['cell']}" in POP}
G_, Gf_, E_, Ef_ = [], [], [], []
for f in sorted((figdata.ROOT / "experiments/results/inside").glob("inside_*.json")):
    mt_ = [(m, t) for m, t in scored if f.name == f"inside_{m}_{t}.json"]
    if not mt_:
        continue
    for c in json.load(open(f))["cells"]:
        if c.get("live") and f"{mt_[0][0]}/{mt_[0][1]}/L{c['layer']}r{c['region']}" in POP:
            mt = c["mean_T"]
            G_.append(mt["own_normal"]); Gf_.append(mt["own_frozen"])
            E_.append(np.mean([mt[f"das_ce_s{x}_normal"] for x in (0, 1, 2)])); Ef_.append(np.mean([mt[f"das_ce_s{x}_frozen"] for x in (0, 1, 2)]))
G_, Gf_, E_, Ef_ = map(np.array, (G_, Gf_, E_, Ef_))
RM_X = 1 - (Ef_ - Gf_).sum() / (E_ - G_).sum()                               # removed of DAS's summed excess over G
KEEP_G = Gf_.sum() / G_.sum()                                                # kept of whole-state patching's summed effect
assert f"{RM_X:.2f}" == NUM["inExcessAttnAll"] and f"{KEEP_G:.2f}" == NUM["inOwnKeep"], (RM_X, KEEP_G)
N_ALL = len(G_)
assert f"{N_ALL:,}".replace(",", "{,}") == NUM["inLive"], N_ALL

W = NIPSW
FN = FS["axis"]
mm = lambda v: f"{v:.2f}"  # noqa: E731
assert abs(E["own_fr"] - E["own"]) < 0.02 and abs(LW["own_fr"] - LW["own"]) < 0.02, "holding attention keeps whole-state patching's effect"
def stepmark(x, y, n, title):
    return circ(x + 6, y - 2.4, 5.0, C["ink"]) + T(x + 6, y, str(n), FN, C["paper"], anchor="middle", weight=700) + \
        T(x + 14.5, y, title, FS["title"], weight=600)
def dividers(cx, cy, bw, bh, col, seg=5):
    return "".join(L(cx - bw / 2 + q * bw / seg, cy - bh / 2 + 0.9, cx - bw / 2 + q * bw / seg, cy + bh / 2 - 0.9, col, 0.4,
                     cap="butt") for q in range(1, seg))
def axis(x0, x1, y, ticks):
    X = lambda t: x0 + (x1 - x0) * t / ticks[-1][0] * (ticks[-1][0] / TOP)  # noqa: E731
    o = L(x0, y, x1, y, C["grayl"], 1.0)
    for t, txt in ticks:
        o += L(X(t), y - 2.0, X(t), y + 2.0, C["gray"], 0.6)
        o += T(X(t), y + 8.6, txt, FS["tick"], C["ink"] if txt == "T" else C["muted"], anchor="middle", weight=600 if txt == "T" else 400)
    return o, X
def lane2(X, x0, y, v, col, lab, free, free_lab=False):
    """Attention held: the filled bar to v, its number at its end; attention free: a thin dashed line, 4.5 pt above, to free."""
    o = R(x0, y - 1.7, X(v) - x0, 3.4, col, r=0.6) if v > 0.005 else ""
    ln_, g_ = X(free) - x0, 1.2
    nd = max(1, round((ln_ + g_) / (1.6 + g_))) if ln_ >= 2 * 1.6 + g_ + 1.0 else 1   # too short for two dashes: one
    o += L(x0, y - 4.5, X(free), y - 4.5, col, 0.6, dash=f"{(ln_ - (nd - 1) * g_) / nd:.3f} {g_}", cap="butt")
    if free_lab:
        o += T(X(free) + 2.5, y - 4.5 + 2.1, "free", FS["tick"], col)
    return o + T(max(X(v), x0) + 3.0, y + 2.2, lab, FS["tick"], col, weight=600)
def lane(X, x0, y, v, col, lab, free=None):
    """A filled bar from 0 to v, its number at its end; free: where the same bar ends with attention free, a dashed tick."""
    o = R(x0, y - 1.7, X(v) - x0, 3.4, col, r=0.6) if v > 0.005 else ""
    if free is not None:
        o += L(X(free), y - 3.6, X(free), y + 3.6, col, 0.8, dash="1.2 1.0")
    xl = max(X(v), x0) + (3.0 if v > 0.005 else 0.0)
    if free is not None and xl - 1.0 < X(free) < xl + tw(lab, FS["tick"], 600) + 1.0:
        xl = X(free) + 3.0                                                   # the number never across its own dashed tick
    return o + T(xl, y + 2.2, lab, FS["tick"], col, weight=600)
WIDTH0, WIDTH1 = 0.4, 2.4                                                   # a beam's or arc's width: 0.4 pt + 2.4 pt per unit of attention
BW, BH = 19.0, 7.4                                                          # the mechanism is each panel's main drawing
def beam_path(xa, xb, y, w, col, h=21.0, dash=None):
    """Attention as a route: from the verb (xa) to the last word (xb), as wide as the last word's attention weight on the verb
    (one head); its head at the last word, the way the verb's content travels to the answer."""
    bw_ = WIDTH0 + WIDTH1 * w                                                # one scale with step 4's arcs
    ah = max(2.6, bw_ * 0.95)
    return (f'<path d="M{xa:.2f},{y:.2f} C{xa:.2f},{y - h:.2f} {xb:.2f},{y - h:.2f} {xb:.2f},{y - ah * 0.9:.2f}" '
            f'fill="none" stroke="{col}" stroke-width="{bw_:.2f}" stroke-linecap="butt"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>"
            + ("" if dash else f'<path d="M{xb - ah:.2f},{y - ah:.2f} L{xb + ah:.2f},{y - ah:.2f} L{xb:.2f},{y + 0.4:.2f} z" fill="{col}"/>'))
def lock(x, y):
    """Attention held at the base run's: a padlock on the route it holds."""
    return (R(x - 3.6, y - 1.8, 7.2, 5.6, C["tgray"], r=1.0) + P(f"M{x - 2.2:.2f},{y - 1.8:.2f} v-1.9 a2.2,2.2 0 0 1 4.4,0 v1.9", C["tgray"], 0.9))
def strip(x0, y, base, src=None, verb="plain", last="plain", beam=None, ghost=None, held=False, lab=None, ghost_lab=None, up_lab=False,
          pw=None, held_lab=None, last_note=None):
    """The base run's two states the steps use (striped boxes, as in fig:guidea): a solid blue box, the source run's whole state;
    pale orange, the base run's plus d's part. Under a patched verb, its base word and, in blue, the source word. The last word's
    arrow up: its state goes on to the answer, in every step."""
    span = 64.0                                                              # verb to last word
    xv = x0 + 21.5                                                           # its ellipsis at x0 + 6: the step mark's centre, the 0
    xl = xv + span
    o = T(xv - BW / 2 - 6.0, y + 2.2, "\u2026", FS["tick"], C["muted"], anchor="middle")
    o += T((xv + xl) / 2, y + 2.2, "\u2026", FS["tick"], C["muted"], anchor="middle")
    for x_, kind, w_ in ((xv, verb, base), (xl, last, "is")):
        fill, stroke, dv = {"plain": (C["paper"], C["grayl"], C["grayl"]), "whole": (C["net"], C["net"], C["paper"]),
                            "dir": (C["dasl"], C["das"], C["das"]), "both": (C["paper"], C["ink"], C["gray"])}[kind]
        o += R(x_ - BW / 2, y - BH / 2, BW, BH, fill, r=1.3, stroke=stroke, sw={"plain": 0.5, "both": 0.9}.get(kind, 0.8))
        o += dividers(x_, y, BW, BH, dv)
        o += T(x_, y + BH / 2 + 7.4, w_, FS["tick"], C["ink"], anchor="middle", weight=600)
    if src and verb != "plain":
        o += T(xv, y + BH / 2 + 14.6, src, FS["tick"], C["net"], anchor="middle", weight=600)
    if last_note:
        o += T(xl, y + BH / 2 + 14.6, last_note, FS["tick"], C["muted"], anchor="middle")
    top = y - BH / 2 - 1.0
    o += L(xl + 7.5, top, xl + 7.5, top - 22.0, C["ink"], 0.7, marker="ak")              # on to the answer, under no tick
    if up_lab:
        o += T(xl + 11.0, top - 18.6, "to the", FS["tick"], C["muted"]) + T(xl + 11.0, top - 12.2, "answer", FS["tick"], C["muted"])
    if ghost is not None:
        o += beam_path(xv, xl - 3.0, top, ghost, C["gray"], h=27.0, dash="1.8 1.5")
    if beam is not None:
        o += beam_path(xv, xl - 3.0, top, beam[0], beam[1])
    apex = top - 21.0 * 0.75
    mid = (xv + xl - 3.0) / 2
    if held:
        o += lock(mid, apex - 0.4)
    if held_lab:
        o += T(mid, apex - 8.6, held_lab, FS["tick"], C["tgray"], anchor="middle")          # over the padlock, clear of the beam
    if lab:
        if lab[2] == "inside":                                               # under the apex, inside the arc
            o += T(mid, top - 6.5, lab[0], FS["tick"], lab[1], anchor="middle", weight=600)
        else:
            o += T(mid, apex - (WIDTH0 + WIDTH1 * beam[0]) / 2 - 2.6, lab[0], FS["tick"], lab[1], anchor="middle", weight=600)
    if ghost_lab:
        o += T(mid, top - 27.0 * 0.75 - 3.0, ghost_lab, FS["tick"], C["gray"], anchor="middle")
    return o
def lines(x, y, text, width, nmax=4):
    text = re.sub(r"(\d) T\b", "\\1\u00a0T", text)                           # never break between a number and its T
    out, cur = [], ""
    for wd in text.split(" "):
        t_ = (cur + " " + wd).strip()
        if tw(t_.replace("\u00a0", " "), FN) > width and cur:
            out.append(cur); cur = wd
        else:
            cur = t_
    out.append(cur)
    assert len(out) <= nmax, out
    return "".join(T(x, y + 7.4 * q, ln.replace("\u00a0", " "), FN, C["muted"]) for q, ln in enumerate(out)), len(out)

MG, MGf, ME, MEf = float(G_.mean()), float(Gf_.mean()), float(E_.mean()), float(Ef_.mean())
assert f"{MGf / MG:.2f}" == NUM["inOwnKeep"] and f"{1 - MEf / ME:.2f}" == NUM["inCeAttnAll"], "step 6's means and the text's shares"
b = arrows(("ak", C["ink"]), size=4.0)
TTL = "Inside the network, much of DAS's excess runs through attention"
b += T(0, 10, TTL, FS["letter"], weight=600)
SUB = (f"{MODEL}, the movie review, the verb at layer 10; the {int(neg.sum())} pairs whose source is negative, DAS's first run")
assert tw(TTL, FS["letter"], 600) < W - 2 and tw(SUB, FS["sub"]) < W
b += T(0, 19.6, SUB, FS["sub"], C["muted"])
P3 = W / 3
TOP = 3.25
TT = ((0, "0"), (1, "T"), (2, "2T"), (3, "3T"))
FIRST = 11.0                                                                  # the first line under a title, as in Figures 2 and 3
AX1 = FIRST + 3.8                                                            # row 1: one lane, its number on the first line
AX2 = 28.5                                                                   # row 2: the blue free line 7 pt under the title
BEAMROOM = 8.6 + 1.1 + 5.0 + 31.4 + BH / 2 + 1.0                            # from the scale to the boxes' middle: tick labels, 6 pt, the
#                                                                              tallest beam and its label (30.8 pt over the boxes' top)
STRIP1, STRIP2 = AX1 + BEAMROOM, AX2 + BEAMROOM
WORDS1, WORDS2 = STRIP1 + BH / 2 + 14.6, STRIP2 + BH / 2 + 14.6
ROWH = WORDS1 + 30.0                                                         # 30 pt from the lowest word to the next row's title
# ---- row 1, nothing held: the base run and the two patches (fig:guidea's), the answer in T, attention as the beam's width
ROW1 = [
    ("The base run", [(0.0, C["gray"], "answer: 0")], dict(beam=(HW["base"], C["gray"]), lab=("attention", C["gray"], "above"), up_lab=True)),
    ("Patch the whole state", [(E["own"], C["net"], f"{mm(E['own'])} T")],
     dict(verb="whole", src="hated", beam=(HW["own"], C["net"]), lab=("attention stays", C["net"], "above"))),
    ("Patch d's part", [(E["das"], C["das"], f"{mm(E['das'])} T")],
     dict(verb="dir", src="hated", beam=(HW["das"], C["das"]), lab=("attention rises", C["das"], "above"))),
]
Y1 = 38.0
for n, (title, bars, kw) in enumerate(ROW1):
    x0 = n * P3
    b += stepmark(x0, Y1, n + 1, title)
    ax, X = axis(x0 + 6, x0 + P3 - 10, Y1 + AX1, TT)
    b += ax
    for v, col, lab in bars:
        b += lane(X, x0 + 6, Y1 + AX1 - 6.0, v, col, lab)
    b += strip(x0, Y1 + STRIP1, "cherished", pw=P3 - 4, **kw)
# ---- row 2, attention held (the padlock on the route it holds): blue over orange as in the map, the held value filled with its
#      number, the free value a dashed tick; the verb, the last word, the mean over every cell
Y2 = Y1 + ROWH
ROW2 = [
    (5, "The verb, attention held", (E["own_fr"], E["own"]), (E["fr"], E["das"]),
     dict(verb="both", src="hated", beam=(HW["base"], C["gray"]), held=True, held_lab="held")),
    (6, "The last word, attention held", (LW["own_fr"], LW["own"]), (LW["fr"], LW["das"]),
     dict(last="both", beam=(HW["base"], C["gray"]), held=True, last_note="layer 20")),
]
for col_, (n, title, gpair, dpair, kw) in enumerate(ROW2):
    x0 = (col_ + 1) * P3
    assert tw(title, FS["title"], 600) < P3 - 16, title
    b += stepmark(x0, Y2, n, title)
    ax, X = axis(x0 + 6, x0 + P3 - 10, Y2 + AX2, TT)
    b += ax
    b += lane2(X, x0 + 6, Y2 + AX2 - 17.0, gpair[0], C["net"], f"{mm(gpair[0])} T", gpair[1])
    b += lane2(X, x0 + 6, Y2 + AX2 - 6.0, dpair[0], C["das"], f"{mm(dpair[0])} T", dpair[1], free_lab=(n == 5))
    b += strip(x0, Y2 + STRIP2, "cherished", pw=P3 - 4, **kw)
# ---- 4: every block after the cell: the last word's attention to the verb, summed over each block's heads, for the two patches
#      of steps 2 and 3 (the beam of steps 1-3 is one head of block 12); no scale: attention, not the answer
x0 = 0.0
b += stepmark(x0, Y2, 4, "Every later block")
yt4, yb4 = Y2 + FIRST + 2.0, Y2 + STRIP2 + BH / 2                            # block 23 on the first line, block 11 on the boxes' bottom
rp4 = (yb4 - yt4) / (NB - 1)
for c_, (key, colr, lab) in enumerate((("own", C["net"], "whole state"), ("das", C["das"], "d's part"))):
    xa, xb = x0 + 30.0 + c_ * 52.0, x0 + 64.0 + c_ * 52.0                   # the verb (left) and the last word (right)
    for j in range(NB):
        yy = yb4 - j * rp4
        w_ = WIDTH0 + WIDTH1 * float(ATT[key][j])
        b += P(f"M{xa:.2f},{yy:.2f} C{xa + 8:.2f},{yy - rp4 * 1.15:.2f} {xb - 8:.2f},{yy - rp4 * 1.15:.2f} {xb:.2f},{yy:.2f}", colr, w_)
    b += T((xa + xb) / 2, Y2 + WORDS2 - 7.2, lab, FS["tick"], colr, anchor="middle", weight=600)
    ws_ = [WIDTH0 + WIDTH1 * float(ATT[key][j]) for j in range(NB)]
    assert all(rp4 - (ws_[j] + ws_[j + 1]) / 2 >= 1.0 for j in range(NB - 1)), "two neighbouring arcs touch"
b += T(x0 + 24.0, yt4 + 2.1, "block 23", FS["tick"], C["muted"], anchor="end")
b += T(x0 + 24.0, yb4 - rp4 + 2.1, "12", FS["tick"], C["muted"], anchor="end")
b += T(x0 + 24.0, yb4 + 2.1, "11", FS["tick"], C["muted"], anchor="end")
H = Y2 + WORDS2 + 4.0
render("fig_guide_d", W, H, b, OUT)
print(f"fig_guide_d: verb L10 on the negative half: own {E['own']:.3f} held {E['own_fr']:.3f}, DAS {E['das']:.3f} held {E['fr']:.3f}; "
      f"last word L20 own {LW['own']:.2f} held {LW['own_fr']:.2f} DAS {LW['das']:.2f} held {LW['fr']:.2f}; mean over {N_ALL} cells "
      f"G {MG:.3f} held {MGf:.3f}, DAS {ME:.3f} held {MEf:.3f}; mirrors {E['mir']:.3f}; all pairs run 1 {E['all1']:.3f}; other runs {STARTS}")

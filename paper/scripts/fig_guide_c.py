"""The visual guide's geometry figure (fig:guidec): why a part of the state can move the answer more than the whole state.

Nine steps, one drawing each, in the plane of two arrows at the movie review's readout in Qwen2.5-0.5B (step 1: Figure 2's
scale of the answer, stood upright): the change a copy makes
(the base run's last state to the source run's) and the direction the answer reads, drawn upward, so that how far a change
moves the answer, to first order, is its height. The angle is the task's measured one (the map's readout cell: cos 0.19, 79
degrees, asserted to lie within the paper's middle half over two-label cases). Copying along a direction d brings across the
change's part along d, the foot of the perpendicular from the source onto d; the right angle puts every such part on the
circle with diameter base-source (Thales). Its top lies (1 + 1/cos theta)/2 T high and its bottom (1 - 1/cos theta)/2 T low
(Proposition 7, with one readout direction); the whole change, the readout direction and the mean difference
all land at height T. Step 3's direction is DAS's own (cross-entropy, the same point step 8 marks), so that its part and
the rest are Figure 2's 2.98 T and -1.98 T. Step 8 marks where the two losses trained from the same three starts land, measured (exact effects on
the test pairs), at their heights on the circle; at the readout G = T on every pair, so the squared error aimed at G is aimed
at T. Step 9 draws the circles of the paper's median angle and quartiles (paper/sections/numbers_main.tex), at a smaller
scale.
Usage: python -I paper/scripts/fig_guide_c.py [--out=<dir>]
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, L, NIPSW, P, R, T, arrows, circ, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
MODEL, TASK = "Qwen2.5-0.5B", "sentiment_movie"
MAP = json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{MODEL}_{TASK}.json"))
RO = max(c["layer"] for c in MAP["cells"])
cell = next(c for c in MAP["cells"] if c["layer"] == RO)
COS = cell["cos_u_b"]; TH = math.acos(COS)
top_of = lambda c: 0.5 * (1 + 1 / c)  # noqa: E731
TOP, BOT = top_of(COS), 0.5 * (1 - 1 / COS)                                  # the circle's highest and lowest heights, in T
mt = cell["mean_T"]
CE = sum(mt[f"das_ce_s{s}_normal"] for s in range(3)) / 3
SQ = sum(mt[f"das_sq_G_s{s}_normal"] for s in range(3)) / 3
B_HAT, U_HAT, TOP_M = mt["b_hat_normal"], mt["u_hat_normal"], mt["v1_mid_normal"]
WIDTH = json.load(open(figdata.ROOT / "paper/scripts/model_configs.json"))[f"Qwen/{MODEL}"]["width"]
NUM = {m.group(1): float(m.group(2)) for m in re.finditer(r"\\newcommand\{\\(roTheta\w+)\}\{([0-9.]+)\}",
                                                          (figdata.ROOT / "paper/sections/numbers_main.tex").read_text())}
ANG = [NUM["roThetaQone"], NUM["roThetaMed"], NUM["roThetaQthree"]]
assert abs(TOP - cell["one_angle"]) < 1e-6 and NUM["roThetaQone"] <= math.degrees(TH) <= NUM["roThetaQthree"]   # a typical angle
assert abs(B_HAT - 1) < 0.01 and abs(U_HAT - 1) < 0.05

W = NIPSW
FN = FS["axis"]                                                             # the guides' running text
b = arrows(("an", C["net"]), ("ad", C["das"]), ("au", C["unt"]), ("ag", C["gray"]), ("at", C["nat"]), ("ap", C["muted"]), size=4.2)
PART = C["muted"]   # a part patched along any direction d is grey; orange is kept for a direction DAS trained (steps 3 and 8)
b += T(0, 10, "Why a part of the state can move the answer more than the whole state", FS["letter"], weight=600)
b += T(0, 19.6, f"{MODEL}, the movie review's last state, drawn in the plane of the change a patch makes and the direction "
                f"the answer reads", FS["sub"], C["muted"])

HT = 14.0                                                                    # points per T of height
Ld = HT / COS
dvec = (Ld * math.sin(TH), Ld * COS)
PW = W / 3
ROWS = [(30.0, 1.0), (30.0 + 113.0, 2.0), (30.0 + 113.0 + 136.0, 2.0)]      # top of each row; rows 2 and 3 hold the circle
def frame(n):
    row = n // 3
    x0, y0 = (n % 3) * PW, ROWS[row][0]
    above = (TOP + 0.45) if row else 3.0
    return x0, y0, (x0 + 30, y0 + 15 + above * HT)
at = lambda Bp, v: (Bp[0] + v[0], Bp[1] - v[1])  # noqa: E731
def part(phi):
    u = (math.sin(phi), math.cos(phi)); k = dvec[0] * u[0] + dvec[1] * u[1]
    return (k * u[0], k * u[1]), u
def wrap(text, width, size):
    out, cur = [], ""
    text = re.sub(r"(\d) T\b", "\\1\u00a0T", text)               # a number keeps its unit
    for wd in text.split(" "):
        t = (cur + " " + wd).strip()
        if tw(t, size) > width and cur:
            out.append(cur); cur = wd
        else:
            cur = t
    return out + [cur]
def circle(Bp, stroke, sw=0.8, scale=1.0, dv=None, op=None):
    dv = dv or dvec
    cx, cy, r = Bp[0] + dv[0] / 2, Bp[1] - dv[1] / 2, math.hypot(*dv) / 2
    o = f' opacity="{op}"' if op else ""
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="{stroke}" stroke-width="{sw}"{o}/>'
def arc_above(Bp, h, col, sw=2.6):
    cx, cy, r = Bp[0] + dvec[0] / 2, Bp[1] - dvec[1] / 2, Ld / 2
    y = Bp[1] - h * HT; dx = math.sqrt(max(r * r - (y - cy) ** 2, 0))
    return (f'<path d="M{cx - dx:.2f},{y:.2f} A{r:.2f},{r:.2f} 0 0 1 {cx + dx:.2f},{y:.2f}" fill="none" stroke="{col}" '
            f'stroke-width="{sw}" stroke-linecap="round" opacity="0.9"/>')
def on_circle(Bp, h, side):
    """The point of the circle at height h (in T), on the right (side=1) or left (side=-1) of its top."""
    cx, cy, r = Bp[0] + dvec[0] / 2, Bp[1] - dvec[1] / 2, Ld / 2
    y = Bp[1] - h * HT
    return (cx + side * math.sqrt(max(r * r - (y - cy) ** 2, 0)), y)
def level(x0, Bp, h, lab, col):
    y = Bp[1] - h * HT
    return L(x0 + 10, y, x0 + PW - 8, y, C["line"], 0.5, dash="1.5 1.8") + T(x0 + PW - 8, y - 2.2, lab, FS["tick"], col, anchor="end", weight=600)
def right_angle(Pp, u, toS, s=3.0):
    a = (u[0] * s, -u[1] * s); n_ = math.hypot(*toS); bb = (toS[0] / n_ * s, toS[1] / n_ * s)
    return P(f"M{Pp[0] - a[0]:.2f},{Pp[1] - a[1]:.2f} L{Pp[0] - a[0] + bb[0]:.2f},{Pp[1] - a[1] + bb[1]:.2f} "
             f"L{Pp[0] + bb[0]:.2f},{Pp[1] + bb[1]:.2f}", C["muted"], 0.5)
up = lambda Bp, h=None: L(Bp[0], Bp[1] + 2, Bp[0], Bp[1] - (h or TOP + 0.4) * HT, C["unt"], 0.9, marker="au")  # noqa: E731
base = lambda Bp, lab=True: circ(*Bp, 2.3, C["gray"]) + (T(Bp[0] - 4, Bp[1] + 2.2, "base", FN, C["muted"], anchor="end") if lab else "")  # noqa: E731
def source(S, lab=True, dy=9.5):
    return circ(*S, 2.4, C["net"]) + (T(S[0], S[1] + dy, "source", FN, C["net"], anchor="middle") if lab else "")
def change(Bp, S, col=None, w=1.2):
    return L(*Bp, S[0] - 3, S[1] + 0.6, col or C["net"], w, marker="an")
def part_arrow(Bp, Pp, w=1.4, col=None, gap=3.2):
    """An arrow from the base to a part's point, ending gap short of it along its own direction, and the point."""
    n_ = math.hypot(Pp[0] - Bp[0], Pp[1] - Bp[1]); ux_, uy_ = (Pp[0] - Bp[0]) / n_, (Pp[1] - Bp[1]) / n_
    return L(*Bp, Pp[0] - gap * ux_, Pp[1] - gap * uy_, col or PART, w, marker="ap") + circ(*Pp, 2.3, col or PART)

def phi_at(h, side):
    """The direction whose part lands on the circle at height h (in T), right (side=1) or left of the top: its angle from
    the direction the answer reads, as part() takes it."""
    px, py = on_circle((0.0, 0.0), h, side)
    return math.atan2(px, -py)
PHI3 = phi_at(CE, 1)                                                         # step 3: DAS's own direction, step 8's point
P3 = part(PHI3)[0][1] / HT                                                   # its part, in T
assert abs(P3 - CE) < 1e-9, (P3, CE)
STEPS = [
    ("The answer reads a direction", "The answer's scale, upright: how high a change goes is how far it moves the answer, to first "
                                     "order."),                          # the caption names Figure 2: no number in the artwork
    ("Patch the whole state", f"The change runs almost sideways, at {math.degrees(TH):.1f}\u00b0 to the direction the answer "
                             f"reads: the answer sees only T."),
    ("Patch one direction d", f"Only DAS's part along d comes across, {P3:.2f} T. The rest, at a right angle, lowers the "
                             f"effect: {1 - P3:.2f} T. Together, T.".replace("-", "\u2212")),
    ("Turn d", "Each direction brings its own part, and moves the answer by that part's height."),
    ("Every part ends on one circle", "The right angle at each foot puts every part on the circle whose diameter joins base "
                                       "and source: Thales' theorem."),
    (f"The maximum: {TOP:.1f} T", f"The highest part moves the answer {TOP / 1:.1f} times as far as the whole state; the lowest, "
                              f"{BOT:.1f} T, the wrong way.".replace("-", "\u2212")),
    ("Two directions give T", f"The direction the answer reads and the change itself, the mean difference, both give "
                                 f"T (measured {B_HAT:.2f} and {U_HAT:.2f})."),
    ("The point depends on the loss", f"Measured: cross-entropy DAS near the maximum ({CE:.2f} T); the squared "
                                 f"error, aimed at T, near the whole state ({SQ:.2f} T)."),
    ("The angle sets the circle", f"The paper's angles, interquartile range {ANG[0]:.1f} to {ANG[2]:.1f}\u00b0 (median "
                                  f"{ANG[1]:.1f}\u00b0): maxima of {top_of(math.cos(math.radians(ANG[0]))):.1f} to "
                                  f"{top_of(math.cos(math.radians(ANG[2]))):.1f} T, predicted before training."),
]
for n, (title, cap) in enumerate(STEPS):
    x0, y0, Bp = frame(n)
    S = at(Bp, dvec)
    b += circ(x0 + 6, y0 + 3.2, 5.0, C["ink"]) + T(x0 + 6, y0 + 5.6, str(n + 1), FN, C["paper"], anchor="middle", weight=700)
    b += T(x0 + 14.5, y0 + 5.6, title, FS["title"], weight=600)
    assert 14.5 + tw(title, FS["title"], 600) <= PW - 6, f"step {n + 1}'s title runs into the next panel"
    bottom = Bp[1] + (1.3 if n < 3 else -BOT + 0.65) * HT
    lines = wrap(cap, PW - 14, FN)
    assert len(lines) <= 3, (title, lines)
    for q, ln in enumerate(lines):
        b += T(x0 + 2, bottom + 9 + 7.4 * q, ln, FN, C["muted"])
    if n == 0:
        b += up(Bp, 3.4)
        for h in (-1, 0, 1, 2, 3):
            y = Bp[1] - h * HT
            b += L(Bp[0] - 2.5, y, Bp[0] + 2.5, y, C["unt"], 0.6)
            b += T(Bp[0] - 5, y + 2, {0: "0", 1: "T", -1: "\u2212T"}.get(h, f"{h}T"), FS["tick"], C["unt"], anchor="end")
        b += T(Bp[0] + 6, Bp[1] - 3.15 * HT, "the direction the", FN, C["unt"]) + T(Bp[0] + 6, Bp[1] - 3.15 * HT + 7.4, "answer reads", FN, C["unt"])
        b += T(Bp[0] + 14, Bp[1] - HT + 2.2, "toward \u201cterrible\u201d", FN, C["muted"])
        b += T(Bp[0] + 14, Bp[1] + HT * 0.75 + 2.2, "toward \u201cgreat\u201d", FN, C["muted"])
        b += L(Bp[0] + 9, Bp[1] - 0.3 * HT, Bp[0] + 9, Bp[1] - 0.95 * HT, C["muted"], 0.5, marker="ag")
        b += L(Bp[0] + 9, Bp[1] + 0.2 * HT, Bp[0] + 9, Bp[1] + 0.6 * HT, C["muted"], 0.5, marker="ag")
        b += base(Bp, False) + T(Bp[0] + 14, Bp[1] + 2.2, "the base run's state", FN, C["muted"])
    elif n == 1:
        b += up(Bp, 3.4) + level(x0, Bp, 1, "T", C["net"])
        a1, r_ = -math.pi / 2 + TH, 13.0
        b += P(f"M{Bp[0]:.2f},{Bp[1] - r_:.2f} A{r_},{r_} 0 0 1 {Bp[0] + r_ * math.cos(a1):.2f},{Bp[1] + r_ * math.sin(a1):.2f}", C["muted"], 0.6)
        b += T(Bp[0] + 6, Bp[1] - HT - 3.2, f"{math.degrees(TH):.1f}\u00b0", FN, C["muted"])          # over the T level
        b += change(Bp, S) + base(Bp) + source(S)
    elif n == 2:
        pv, u = part(PHI3); Pp = at(Bp, pv)
        k_ = pv[0] * u[0] + pv[1] * u[1]
        dmx, dmy = (Bp[0] + Pp[0]) / 2, (Bp[1] + Pp[1]) / 2                  # "d" beside the arrow's middle, on its upper left
        # step 3's d is DAS's own direction (step 8's point, Figure 2's step 3): orange, as every direction DAS trained
        b += level(x0, Bp, pv[1] / HT, f"{pv[1] / HT:.2f} T", C["das"]) + level(x0, Bp, 1, "T", C["net"])    # under the points
        b += up(Bp, 3.4) + T(dmx - 6.5 * u[1] - 3, dmy - 6.5 * u[0] + 1, "d", FN, C["das"], anchor="end", weight=600, italic=True)
        n_ = math.hypot(S[0] - Bp[0], S[1] - Bp[1])
        b += L(*Bp, S[0] - 2.8 * (S[0] - Bp[0]) / n_, S[1] - 2.8 * (S[1] - Bp[1]) / n_, C["netl"], 1.0)   # the whole change, faint
        tS = (S[0] - Pp[0], S[1] - Pp[1]); nS = math.hypot(*tS); tS = (tS[0] / nS, tS[1] / nS)
        b += right_angle(Pp, u, (S[0] - Pp[0], S[1] - Pp[1]), s=3.4)
        b += L(Pp[0] + 3.0 * tS[0], Pp[1] + 3.0 * tS[1], S[0] - 3.4 * tS[0], S[1] - 3.4 * tS[1], C["gray"], 1.0, marker="ag")   # the rest
        # the rest's words in two lines right of its line, between the two levels' labels and inside the panel
        lab3 = ("the rest,", f"{1 - pv[1] / HT:.2f} T".replace("-", "\u2212"))
        mx, my = (Pp[0] + S[0]) / 2, (Pp[1] + S[1]) / 2
        lx3 = max(Pp[0] + (q_ - Pp[1]) / (S[1] - Pp[1]) * (S[0] - Pp[0]) for q_ in (my - 9, my + 2)) + 4   # right of the line
        for q, t3 in enumerate(lab3):
            b += T(lx3, my - 4 + 7.4 * q, t3, FN, C["gray"], weight=600)
        assert lx3 + max(tw(t3, FN, 600) for t3 in lab3) < x0 + PW - 2, "the rest's words leave the panel"
        assert Pp[1] + 3 < my - 4 - 4.6 and my - 4 + 7.4 < S[1] - 4.6, "the rest's words meet a level's label"
        ue = (pv[0] / k_, -pv[1] / k_)                                       # u in page coordinates
        b += L(*Bp, Pp[0] - 4.6 * ue[0], Pp[1] - 4.6 * ue[1], C["das"], 1.4, marker="ad") + circ(*Pp, 2.3, C["das"])
        b += base(Bp) + source(S)
    elif n == 3:
        b += up(Bp) + change(Bp, S, C["netl"], 1.0)
        for deg, op in ((-20, 0.55), (22, 0.8), (55, 1.0)):
            pv, u = part(math.radians(deg)); Pp = at(Bp, pv)
            b += L(*S, *Pp, C["grayl"], 0.5, dash="1.2 1.2")
            b += f'<g opacity="{op}">' + part_arrow(Bp, Pp, 1.2) + "</g>"
            yl_ = Pp[1] - 2.5 if pv[1] >= 0 else Pp[1] + 8.5                   # under a part that goes the wrong way
            b += T(Pp[0] + 4, yl_, f"{pv[1] / HT:.1f} T".replace("-", "\u2212"), FS["tick"], PART, weight=600)
        b += base(Bp) + source(S)
    elif n == 4:
        b += up(Bp) + circle(Bp, C["grayl"], 2.0)
        for deg in range(-70, 171, 16):
            pv, u = part(math.radians(deg)); Pp = at(Bp, pv)
            b += L(*Bp, *Pp, PART, 0.45, op=0.45) + circ(*Pp, 1.6, PART)
        pv, u = part(math.radians(22)); Pp = at(Bp, pv)
        b += L(*S, *Pp, C["gray"], 0.5, dash="1.2 1.2") + right_angle(Pp, u, (S[0] - Pp[0], S[1] - Pp[1]))
        b += base(Bp) + source(S, False) + T(S[0] + 5, S[1] + 2.2, "source", FN, C["net"])
        assert S[0] + 5 + tw("source", FN) < x0 + PW - 2, "step 5's source label runs into the next panel"
    elif n == 5:
        b += up(Bp) + circle(Bp, C["grayl"], 0.9) + arc_above(Bp, 1.0, C["dasl"], 2.0)   # where a part beats the whole: DAS's tint
        cx = Bp[0] + dvec[0] / 2
        for h, col in ((TOP, C["das"]), (1, C["net"])):
            b += L(x0 + 10, Bp[1] - h * HT, x0 + PW - 8, Bp[1] - h * HT, C["line"], 0.5, dash="1.5 1.8")
        b += T(cx, Bp[1] - TOP * HT - 5, f"+{TOP:.1f} T, the maximum", FS["tick"], C["tgray"], anchor="middle", weight=600)   # the maximum: geometry, not DAS
        b += T(S[0] + 5, S[1] - 2.6, "T", FS["tick"], C["net"], weight=600)
        b += T(cx + 6, Bp[1] - BOT * HT + 7.4, f"\u2212{-BOT:.1f} T, the least", FS["tick"], C["gray"], weight=600)
        b += circ(cx, Bp[1] - TOP * HT, 3.0, C["tgray"]) + circ(cx, Bp[1] - BOT * HT, 2.6, C["gray"]) + base(Bp, False) + source(S, False)
        b += T(cx, Bp[1] - 1.9 * HT, "exceeds the whole", FN, C["muted"], anchor="middle", italic=True)
    elif n == 6:
        b += up(Bp) + circle(Bp, C["grayl"], 0.9)
        Rw = (Bp[0], Bp[1] - HT)
        b += L(Bp[0], Bp[1] - HT, S[0], S[1], C["line"], 0.5, dash="1.5 1.8")
        b += circ(*Rw, 2.6, C["unt"]) + T(Rw[0] + 5, Rw[1] - 4, "the direction it reads", FN, C["unt"])
        b += circ(*S, 3.3, "none", C["nat"], 1.0) + source(S, False)
        b += T(S[0] - 2, S[1] + 10, "the change itself", FN, C["nat"], anchor="end")
        b += base(Bp)
    elif n == 7:
        b += up(Bp) + circle(Bp, C["grayl"], 0.9) + arc_above(Bp, 1.0, C["dasl"], 2.0)
        pc = on_circle(Bp, CE, 1); ps = on_circle(Bp, SQ, -1)
        # cross-entropy's words close to its point: its name centred over it, between the circle's top and the title; its
        # value just right of it, from the first place where the circle (and its tinted arc) has dropped clear of the words
        cx8, cy8, r8 = Bp[0] + dvec[0] / 2, Bp[1] - dvec[1] / 2, Ld / 2
        l1, l2 = pc[1] - 6.5, pc[1] + 2.2
        box_bot = l2 + 0.18 * FN
        x2 = cx8 + math.sqrt(max(r8 ** 2 - (box_bot - cy8 + 2.5) ** 2, 0)) + 0.5   # the circle is 2.5 below the words from here
        x2 = max(x2, pc[0] + 5.5)
        b += circ(*pc, 3.0, C["das"]) + T(pc[0], l1, "cross-entropy", FN, C["das"], anchor="middle", weight=600)
        b += T(x2, l2, f"{CE:.2f} T", FN, C["das"])
        for t8, (bx0, by0, bx1, by1) in (("cross-entropy", (pc[0] - tw("cross-entropy", FN, 600) / 2, l1 - 0.70 * FN,
                                                             pc[0] + tw("cross-entropy", FN, 600) / 2, l1 + 0.18 * FN)),
                                         (f"{CE:.2f} T", (x2, l2 - 0.70 * FN, x2 + tw(f"{CE:.2f} T", FN), box_bot))):
            nx8 = min(max(cx8, bx0), bx1); ny8 = min(max(cy8, by0), by1)
            assert math.hypot(nx8 - cx8, ny8 - cy8) > r8 + 2.0, ("step 8's words meet the circle", t8)
            assert bx1 < x0 + PW - 1.5 and by0 > y0 + 5.6 + 2.5 and bx0 > Bp[0] + 2.0, ("step 8's words leave their room", t8)
        assert x2 - pc[0] < 12, "step 8's value strays from its point"
        b += R(ps[0] - 2.4, ps[1] - 2.4, 4.8, 4.8, C["paper"], r=0.5, stroke=C["das"], sw=0.9)
        b += T(ps[0] + 5, ps[1] - 3, "squared error", FN, C["das"]) + T(ps[0] + 5, ps[1] + 4.8, f"{SQ:.2f} T", FN, C["das"])
        b += base(Bp, False) + source(S, False)
    else:
        h9 = 7.2                                                             # a smaller scale, so that the widest circle fits
        B9 = (Bp[0] - 6, Bp[1] - 0.75 * HT)
        lx = B9[0] + h9 / math.cos(math.radians(ANG[2])) * 1.0 + 6           # labels to the right of the widest circle
        b += L(B9[0], B9[1] - h9, lx - 4, B9[1] - h9, C["line"], 0.5, dash="1.5 1.8") + T(lx, B9[1] - h9 + 2.2, "T", FS["tick"], C["net"], weight=600)
        for a, col in zip(ANG, (C["gray"], C["tgray"], C["gray"])):           # the median's circle and top in the top's colour
            c_ = math.cos(math.radians(a)); l_ = h9 / c_
            dv = (l_ * math.sin(math.radians(a)), l_ * c_)
            b += circle(B9, col, 1.5 if col == C["tgray"] else 0.7, dv=dv)
            tx, ty = B9[0] + dv[0] / 2, B9[1] - top_of(c_) * h9
            b += circ(tx, ty, 1.9, col) + L(tx + 2.5, ty, lx - 3, ty, col, 0.4, dash="1 1.2")
            b += T(lx, ty + 2.2, f"{a:.1f}\u00b0: {top_of(c_):.1f} T", FS["tick"], col, weight=600)
            b += circ(B9[0] + dv[0], B9[1] - dv[1], 1.5, C["net"])
        b += L(B9[0], B9[1] + 2, B9[0], B9[1] - 5.2 * h9, C["unt"], 0.9, marker="au") + circ(*B9, 2.3, C["gray"])
LAM1, LAMD = cell["lambda1_mid"] / cell["T_test"], cell["lambdaD_mid"] / cell["T_test"]
foot = (f"For one pair the circle is exact (in all {WIDTH} numbers, a sphere with the same highest and lowest points); heights "
        f"are first order. The drawing uses the task's mean change and readout direction, so its top, {TOP:.2f} T, is the maximum "
        f"the angle predicts. One direction serves all 100 pairs, each with its own circle, so its mean effect lies between "
        f"\u2212{-LAMD:.2f} and {LAM1:.2f} T (M's eigenvalues); measured, the top eigenvector gives {TOP_M:.2f} T.")
assert len(wrap(foot, W, FS["note"])) <= 3, wrap(foot, W, FS["note"])
fy = ROWS[2][0] + 142
for q, ln in enumerate(wrap(foot, W, FS["note"])):
    b += T(0, fy + 7.6 * q, ln, FS["note"], C["muted"])
H = fy + 7.6 * len(wrap(foot, W, FS["note"])) - 2

render("fig_guide_c", W, H, b, OUT)
print(f"fig_guide_c: theta {math.degrees(TH):.1f}; top {TOP:.2f}, bottom {BOT:.2f}; CE {CE:.2f}, sq {SQ:.2f}; quartile tops "
      f"{[round(top_of(math.cos(math.radians(a))), 2) for a in ANG]}")

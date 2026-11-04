"""The readout figure of Section 5 (fig:readout, text width, 397 pt; (a) across the top, (b) and (c) under it, (b)'s axis from 0.5
at (a)'s height per decade; no key: every mark is named where it is drawn): at the readout, the loss decides where DAS lands in each
task's range, one angle predicts the top, and larger models leave more room and go further.

Population of (a) and (b): the main population (the rows flagged main in experiments/results/score_readout.json; tasks with at least 20
test clusters whose T's interval lies above 0) in every model whose instruments held, one column per (model, task). Every
effect is over T, on one log scale shared by (a) and (b). A value below the axis is drawn at the floor with a down arrow and
its value; only out-of-sample tops lie there, at most two (asserted: Pythia-160m's SST classification, -0.50 T, and GPT-2's
JSON prompt of the Nobel laureates' gender). Nothing lies above the axis (asserted).
(a) Every task is a column, ordered by its out-of-sample top (the largest mean exact effect found on the training pairs,
    measured on the test pairs; dashed); the tasks without room (the top's interval not above T) come first. Grey: the room from T to the top
    (cross-entropy above T: DAS's interval above T, among the tasks with room). Discs: cross-entropy DAS (three starts, one
    pass); open squares: the same starts trained with a per-pair squared error to each pair's whole-input effect. Ticks under the axis: the tasks with room where
    cross-entropy's interval is not above T, all of them Kumon & Yanaka's NPI constructions in their one-directional form but
    one, Pythia-160m's sentiment classification (asserted).
(b) The one-question tasks of the models whose logits are not capped, ordered by the angle theta between the mean readout
    gradient and the mean input change (training pairs, midpoint gradients); dashed: the top predicted from it before training,
    (1 + 1/cos theta)/2 (Proposition 7), in the same order as theta; pale orange: the middle half of cross-entropy DAS
    around the prediction (the first to third quartile of DAS over the prediction, over these tasks); discs: cross-entropy DAS.
    One task whose cross-entropy effect's interval is not above zero is left out (asserted). The number measures the pattern
    the panel shows: the rank correlation between the prediction and cross-entropy DAS over each model's tasks (angle in
    experiments/results/score_readout.json; numbers_main's roAngleCeSpearman), its range over the models.
(c) Per model, every model scored (experiments/results/score_readout.json): the median over its main tasks of cross-entropy DAS over
    T (r, filled) and of the out-of-sample top over T (rho, open); lines join a family's sizes (Pythia 160m to 6.9b, Qwen2.5
    0.5B to 32B, Llama 3.2-1B to 3.1-8B); GPT-2, SmolLM2 and Gemma-2 stand alone. The lines draw medians and test nothing:
    the test of size within a family is Appendix C.7's (experiments/score/score_scale.py). Row height follows the number of models.
Each panel: a title (the finding) and, as a subtitle, the numbers that measure what the panel shows (where each loss lands in
(a), the prediction's rank correlation with DAS in (b)); no key: each mark is named where it is drawn, and the grey of (a), the
margin between T and the maximum, lies between two named lines (no spot inside it is clear of the discs for a label).
The scorer's rows hold these under excess (cross-entropy DAS over T, its interval against T), angle (the one-angle prediction P
of the top and its rank correlation with DAS) and objective (where the squared error lands).
Usage: python -I paper/scripts/fig_readout.py [--out=<dir>]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, L, MK, NIPSW, P, T, linscale, logscale, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
S = json.load(open(figdata.ROOT / "experiments/results/score_readout.json"))
OK = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]
rows = [r for m in OK for r in S["rows"][m] if r["main"] and r["defined"]]
npi1 = lambda r: r["group"] == "kumon" and r["task"].startswith("NPI")  # noqa: E731
CAPPED = [m for m in OK if any(r.get("raw_over_cap") for r in S["rows"][m])]      # Gemma-2: first order's checks exclude it
one = [r for r in rows if r["one_q"] and r.get("angle", {}).get("P") and r["model"] not in CAPPED]
# the rank correlation (angle) is taken on log x, so a task whose cross-entropy effect's interval is not above zero leaves it;
# it leaves the panel too, so that the panel's number and its columns are one population (one task; the caption says so)
_out = [r for r in one if not (r["angle"]["x"].get("ce_p1") and r["angle"]["x"]["ce_p1"]["lo"] > 0)]
assert len(_out) == 1, "the caption: one two-label task left out"
one = [r for r in one if r not in _out]
ce = lambda r: r["excess"]["ce_p1"]["pt"]  # noqa: E731
sq = lambda r: r["objective"]["p1"]["sq_T"]  # noqa: E731
top = lambda r: r["top_train_T"][0]  # noqa: E731
angle = lambda r: math.degrees(math.acos(min(max(1 / (2 * r["angle"]["P"] - 1), -1), 1)))  # noqa: E731
room = [r for r in rows if r["room"]]
above = [r for r in room if r["excess"]["ce_p1"]["above"]]
miss = [r for r in room if not r["excess"]["ce_p1"]["above"]]
sq_q1, sq_q3 = (float(np.quantile([sq(r) for r in rows], q)) for q in (0.25, 0.75))   # the squared error's middle half (reported below)
sq_med = 100 * float(np.median([abs(sq(r) - 1) for r in rows]))                       # numbers_main's roSqMedianDev
ce_of_top = 100 * float(np.median([ce(r) / top(r) for r in rows]))                    # the pooled median over every case, computed here (the text gives per-model ranges)
ratio_q1, ratio_q3 = (float(np.quantile([ce(r) / r["angle"]["P"] for r in one], q)) for q in (0.25, 0.75))
angle_ce = [S["report"]["models"][m]["main"]["angle"]["ce_p1"] for m in OK if m not in CAPPED]
rho = sorted(v["spearman"][0] for v in angle_ce)                                            # numbers_main's roAngleCeSpearman
assert sum(v["tasks"] for v in angle_ce) == len(one), "(b)'s number and its columns are one population"
# the ticks mark the tasks with room where cross-entropy is not above T; all but one are NPI constructions
assert [(r["model"], r["task"]) for r in miss if not npi1(r)] == [("pythia-160m", "sentiment_classification")], "the exceptions"
W = NIPSW
PH = 96.0                                                               # the plots' height, (a)'s and (b)'s alike: one log scale
PY1, PY0 = 36.0, 36.0 + PH                                              # (a)'s plot, top and bottom
R2 = PY0 + 33.0                                                         # the second row's top
PY1B, PY0B = R2 + PY1, R2 + PY1 + PH                                    # (b)'s and (c)'s plot: as far under their heads as (a)'s
H = 0.0                                                                 # set at the end
LO, HI = 0.1, 30.0                                                    # (30: Qwen2.5-32B reaches 25.9 T)
assert all(v <= HI for r in rows for v in (top(r), ce(r), sq(r))), "a value above the shared axis"
FLOOR = [(r["model"], r["task"], k, v) for r in rows for k, v in (("top", top(r)), ("ce", ce(r)), ("sq", sq(r))) if v < LO]
assert all(k == "top" for _, _, k, _ in FLOOR) and len(FLOOR) <= 2, \
    f"values below the axis other than out-of-sample tops (Pythia-160m's SST classification; GPT-2's JSON prompt of the Nobel laureates' gender): {FLOOR}"
Y_ = logscale(LO, HI, PY0, PY1)
Y = lambda v: Y_(max(v, LO))  # noqa: E731   (a value below the axis sits at the floor, marked)
b = ""


def disc(x, y, r=MK["r"], op=MK["op"]):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{C["das"]}" fill-opacity="{op}"/>'


def box(x, y, h=0.75, op=0.55):
    return (f'<rect x="{x - h:.2f}" y="{y - h:.2f}" width="{2 * h:.2f}" height="{2 * h:.2f}" fill="none" stroke="{C["das"]}" '
            f'stroke-width="{MK["sq_w"]}" stroke-opacity="{op}"/>')


def head(x, letter, title, subs, y=9.0):
    o = T(x, y, letter, 9.0, weight=700) + T(x + 9.5, y, title, FS["title"], weight=600)
    return o + "".join(T(x + 9.5, y + 8.6 + 7.4 * k, t_, FS["sub"], C["muted"]) for k, t_ in enumerate(subs))


def yticks(x, vals):
    return "".join(T(x, Y(v) + 2.0, t_, FS["tick"], C["net"] if v == 1 else C["gray"], anchor="end", weight=700 if v == 1 else 400)
                   for v, t_ in vals)


# ------------------------------------------------------------------ (a) two losses, one range
x0, x1 = 22.0, W - 4.0
b += head(0.0, "a", "Two losses, one range",
          [f"cross-entropy ends up at {ce_of_top:.0f}% of the maximum,", f"squared error ends up {sq_med:.1f}% from T (medians)"])
srt = sorted([r for r in rows if not r["room"]], key=top) + sorted(room, key=top)
n = len(srt)
X = lambda i: x0 + (i + 0.5) / n * (x1 - x0)  # noqa: E731
i0 = next(i for i, r in enumerate(srt) if r["room"])
up = " ".join(f"{X(i):.2f},{Y(top(r)):.2f}" for i, r in enumerate(srt) if r["room"])
b += f'<polygon points="{X(i0):.2f},{Y(1):.2f} {up} {X(n - 1):.2f},{Y(1):.2f}" fill="{C["soft"]}"/>'    # the margin above T: grey, and only it
b += L(x0, Y(1), x1, Y(1), C["net"], 0.6)                                # T, under the marks, so that the squares near it show
for i, r in enumerate(srt):
    b += box(X(i), Y(sq(r)))
for i, r in enumerate(srt):
    b += disc(X(i), Y(ce(r)))
b += P("M" + " L".join(f"{X(i):.2f},{Y(top(r)):.2f}" for i, r in enumerate(srt)), C["tgray"], 0.8, dash="2.2 1.6")
_fl = [(i, r) for i, r in enumerate(srt) if top(r) < LO]                  # tops below the axis: at the floor, their values beside
for i, r in _fl:
    b += (f'<path d="M{X(i) - 1.3:.2f},{Y(LO) - 1.8:.2f} L{X(i) + 1.3:.2f},{Y(LO) - 1.8:.2f} L{X(i):.2f},{Y(LO) + 0.4:.2f} Z" '
          f'fill="{C["tgray"]}"/>')
if _fl:
    b += T(X(_fl[-1][0]) + 6.0, Y(LO) - 1.2, ("maxima " if len(_fl) > 1 else "maximum ") + ", ".join(f"{top(r):.2f}" for _, r in _fl).replace("-", "−"),
           FS["tick"], C["gray"])
b += yticks(x0 - 3, ((0.1, "0.1"), (0.3, "0.3"), (1, "T"), (3, "3"), (10, "10"))) + L(x0, PY0, x1, PY0, C["line"], 0.6)
def clear_run(cases, Xf, width, ok, start):
    """The first run of consecutive cases from `start` on, at least `width` pt wide, where ok holds for every case."""
    run = None
    for j in range(start, len(cases)):
        if not ok(cases[j]):
            run = None
            continue
        run = j if run is None else run
        if Xf(j) - Xf(run) >= width:
            return run, j
    raise AssertionError("no clear run for a label")
def label_at(cases, Xf, lines, col, v_text, ok, start, v_to):
    """A label of one or two lines at value v_text over a run of cases its words leave clear, and a leader from it to v_to
    (a value, or a function of the run's middle case) at the run's middle. Returns the label and the run's last case."""
    ws_ = [tw(t_, FS["axis"], 600 if k == 0 else 400) for k, t_ in enumerate(lines)]
    j0, j1 = clear_run(cases, Xf, max(ws_) + 2.0, ok, start)
    jm = (j0 + j1) // 2; xm = Xf(jm)
    vt = v_to(cases[jm]) if callable(v_to) else v_to
    o = "".join(T(Xf(j0) + 1.0, Y(v_text) + 7.4 * k, t_, FS["axis"], col, weight=600 if k == 0 else 400) for k, t_ in enumerate(lines))
    y_from = Y(v_text) + 7.4 * (len(lines) - 1) + 1.6 if vt < v_text else Y(v_text) - 5.6
    return o + L(xm, y_from, xm, Y(vt), col, 0.5), j1
mx_ = lambda r: max(top(r), ce(r), sq(r))  # noqa: E731
mn_ = lambda r: min(top(r), ce(r), sq(r))  # noqa: E731
lab_, jt = label_at(srt, X, ["maximum, out of sample"], C["tgray"], 5.2, lambda r: mx_(r) < 2.9, i0 + 25, top)    # the dashed curve
b += lab_
lab_, _ = label_at(srt, X, ["cross-entropy DAS"], C["das"], 5.2, lambda r: mx_(r) < 2.9, jt + 12, 1.75)          # the discs
b += lab_
lab_, _ = label_at(srt, X, ["squared error, aimed at T"], C["das"], 0.21,
                   lambda r: mn_(r) > 0.34, int(0.40 * n), 0.86)                                                    # the squares, under T
b += lab_
# the exceptions: the cases with a margin where cross-entropy is not above T, as ticks under the axis
key_ = lambda r: (r["model"], r["group"], r["task"])  # noqa: E731
miss_ids = [i for i, r in enumerate(srt) if key_(r) in {key_(m) for m in miss}]
n_npi = sum(npi1(srt[i]) for i in miss_ids)
assert len(miss_ids) == len(miss) and n_npi == len(miss) - 1, "the exceptions: all but one negative polarity"
for i in miss_ids:
    b += L(X(i), PY0 + 1.0, X(i), PY0 + 4.8, C["tgray"], 0.6)
b += T((x0 + x1) / 2, PY0 + 12.5, f"{len(rows)} model–task cases, ordered by their maximum, those without a margin above T first",
       FS["axis"], C["muted"], anchor="middle")                                                                     # the order the plot uses
b += T((x0 + x1) / 2, PY0 + 19.9, f"ticks: the {len(miss)} cases with a margin where cross-entropy is not above T, {n_npi} of them "
       "negative-polarity constructions", FS["axis"], C["muted"], anchor="middle")
b += T(6.5, (PY0 + PY1) / 2, "how far the answer moves", FS["axis"], C["muted"], anchor="middle", rotate=-90)

# ------------------------------------------------------------------ (b) one angle predicts the top, in (a)'s grammar
LOB = 0.5                                                               # (b)'s floor: its lowest point is 0.84 T (asserted)
assert all(ce(r) > LOB and ratio_q1 * r["angle"]["P"] > LOB for r in one), "a point or band edge of (b) under its axis"
PY0Bb = PY1B + PH * math.log10(HI / LOB) / math.log10(HI / LO)          # (b)'s bottom: (a)'s height per decade
Y_ = logscale(LOB, HI, PY0Bb, PY1B)
PY1, PY0 = PY1B, PY0Bb
bx0, bx1 = 22.0, 186.0
# the subtitle's rank correlation is angle's per-model one, given as its range over the models (the caption says so)
b += head(0.0, "b", "θ sets the maximum",
          ["DAS follows the prediction:", f"rank correlation {rho[0]:.2f}–{rho[-1]:.2f}"], y=R2 + 9.0)
so = sorted(one, key=lambda r: r["angle"]["P"])
no = len(so)
XB = lambda i: bx0 + (i + 0.5) / no * (bx1 - bx0)  # noqa: E731
up = " ".join(f"{XB(i):.2f},{Y(min(ratio_q3 * r['angle']['P'], HI)):.2f}" for i, r in enumerate(so))
dn = " ".join(f"{XB(i):.2f},{Y(max(ratio_q1 * r['angle']['P'], LO)):.2f}" for i, r in reversed(list(enumerate(so))))
b += f'<polygon points="{up} {dn}" fill="{C["dasl"]}"/>'                  # DAS's middle half around the prediction: DAS's pale orange
b += L(bx0, Y(1), bx1, Y(1), C["net"], 0.9)
for i, r in enumerate(so):
    b += disc(XB(i), Y(ce(r)))
b += P("M" + " L".join(f"{XB(i):.2f},{Y(min(ratio_q3 * r['angle']['P'], HI)):.2f}" for i, r in enumerate(so)), C["das"], 0.4, op=0.6)
b += P("M" + " L".join(f"{XB(i):.2f},{Y(ratio_q1 * r['angle']['P']):.2f}" for i, r in enumerate(so)), C["das"], 0.4, op=0.6)   # the band's edges, over the discs
b += P("M" + " L".join(f"{XB(i):.2f},{Y(r['angle']['P']):.2f}" for i, r in enumerate(so)), C["tgray"], 0.8, dash="2.2 1.6")
mxb = lambda r: max(ce(r), ratio_q3 * r["angle"]["P"], r["angle"]["P"])  # noqa: E731
lab_, _ = label_at(so, XB, ["predicted from θ"], C["tgray"], 9.0, lambda r: mxb(r) < 5.0, int(0.15 * no), lambda r: r["angle"]["P"])
b += lab_
lab_, _ = label_at(so, XB, ["cross-entropy DAS"], C["das"], 1.25,                  # under the band, over T: the discs there are under T
                   lambda r: (ce(r) > 1.85 or ce(r) < 1.05) and ratio_q1 * r["angle"]["P"] > 1.8, int(0.38 * no),
                   lambda r: ratio_q1 * r["angle"]["P"])
b += lab_
b += L(bx0, PY0, bx1, PY0, C["line"], 0.6)
for a in (60, 75, 85):                                                   # where the ordered angles pass 60, 75, 85 degrees
    k = next(i for i, r in enumerate(so) if angle(r) >= a)
    b += L(XB(k), PY0, XB(k), PY0 + 2.0, C["gray"], 0.5) + T(XB(k), PY0 + 7.4, f"{a}°", FS["tick"], C["gray"], anchor="middle")
b += T((bx0 + bx1) / 2, PY0 + 15.0, f"{no} cases, ordered by θ", FS["axis"], C["muted"], anchor="middle")
b += yticks(bx0 - 3, ((0.5, "0.5"), (1, "T"), (3, "3"), (10, "10")))
b += T(6.5, (PY0 + PY1) / 2, "how far the answer moves", FS["axis"], C["muted"], anchor="middle", rotate=-90)

# ------------------------------------------------------------------ (c) larger models leave more room and go further
erows = {m: [r for r in S["rows"][m] if r["main"] and r["defined"]] for m in OK}   # main rows: Gemma-2 with its cap
med_r = {m: float(np.median([ce(r) for r in rs])) for m, rs in erows.items()}
med_rho = {m: float(np.median([top(r) for r in rs])) for m, rs in erows.items()}
# a family's sizes are joined, Pythia-160m too: the line draws per-model medians and tests nothing. The test of size within a
# family (Appendix C.7, experiments/score/score_scale.py) compares three sizes per family, and Pythia-160m lies below them
GROUPS = (("GPT-2", (("gpt2", "124M"),)), ("SmolLM2", (("SmolLM2-360M", "360M"),)),
          ("Pythia", (("pythia-160m", "160M"), ("pythia-410m", "410M"), ("pythia-1b", "1B"), ("pythia-6.9b", "6.9B"))),
          ("Qwen2.5", (("Qwen2.5-0.5B", "0.5B"), ("Qwen2.5-1.5B", "1.5B"), ("Qwen2.5-7B", "7B"), ("Qwen2.5-32B", "32B"))),
          ("Llama", (("Llama-3.2-1B", "1B"), ("Llama-3.2-3B", "3B"), ("Llama-3.1-8B", "8B"))), ("Gemma-2", (("gemma-2-9b", "9B"),)))
assert sorted(m for _, ms in GROUPS for m, _ in ms) == sorted(OK), "(c)'s rows and the scored models differ"
PAPER_NAME = {"gpt2": "GPT-2 small", "SmolLM2-360M": "SmolLM2-360M", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m",
              "pythia-1b": "Pythia-1b", "pythia-6.9b": "Pythia-6.9b", "Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen2.5-1.5B": "Qwen2.5-1.5B",
              "Qwen2.5-7B": "Qwen2.5-7B", "Qwen2.5-32B": "Qwen2.5-32B", "Llama-3.2-1B": "Llama-3.2-1B", "Llama-3.2-3B": "Llama-3.2-3B",
              "Llama-3.1-8B": "Llama-3.1-8B", "gemma-2-9b": "Gemma-2-9B"}
assert set(PAPER_NAME) == set(OK), "(c): a model without its paper name"
assert max(max(med_r.values()), max(med_rho.values())) < 4.45, "(c)'s scale: a median beyond its right edge"
PY0 = PY0B                                                              # (c): the row's full height
cx0 = 212.0
b += head(cx0, "c", "Within a family", ["the largest has a larger margin", "above T and goes further"], y=R2 + 9.0)
LXc, SX0, SX1 = cx0 + 53.0, cx0 + 58.0, W - 4.0
SX = linscale(1.0, 4.5, SX0, SX1)                                     # (4.5: Qwen2.5-32B's median top is 4.31)
GAP = lambda fam, ms: 1.5 if len(ms) > 1 or fam == "SmolLM2" else 0.0  # noqa: E731   (a space after each family, and the singles)
YC0, YC1 = PY1 - 3.0, PY0 - 3.5                                          # the first and the last row
nrow = sum(len(ms) for _, ms in GROUPS)
rh = (YC1 - YC0 - sum(GAP(f, ms) for f, ms in GROUPS[:-1])) / (nrow - 1)   # the row height from the number of models
ys, y = {}, YC0
for fam, ms in GROUPS:
    for m, size in ms:
        ys[m] = y; y += rh
        b += T(LXc, ys[m] + 2.1, PAPER_NAME[m], FS["tick"], C["ink"], anchor="end")   # the name the text and Table 3 use
    y += GAP(fam, ms)
assert abs(max(ys.values()) - YC1) < 1e-6 and rh >= 5.0, f"(c)'s rows do not fit: row height {rh:.2f}"
ytop, ybot = min(ys.values()) - rh / 2, max(ys.values()) + rh / 2
for v in (2, 3, 4):
    b += L(SX(v), ytop, SX(v), ybot, C["line"], 0.4)
b += L(SX(1), ytop, SX(1), ybot, C["net"], 0.9)
for fam, ms in GROUPS:
    if len(ms) > 1:                                                     # a family's sizes: tops dashed, DAS solid
        b += P("M" + " L".join(f"{SX(med_rho[m]):.2f},{ys[m]:.2f}" for m, _ in ms), C["tgray"], 0.6, dash="1.6 1.2")
        b += P("M" + " L".join(f"{SX(med_r[m]):.2f},{ys[m]:.2f}" for m, _ in ms), C["das"], 0.9)
for m, y_ in ys.items():
    b += f'<circle cx="{SX(med_rho[m]):.2f}" cy="{y_:.2f}" r="1.6" fill="#FFFFFF" stroke="{C["tgray"]}" stroke-width="0.7"/>'
    b += f'<circle cx="{SX(med_r[m]):.2f}" cy="{y_:.2f}" r="1.8" fill="{C["das"]}"/>'
b += L(SX0, PY0, SX1, PY0, C["line"], 0.6)
for v, t_ in ((1, "T"), (2, "2"), (3, "3"), (4, "4")):
    b += T(SX(v), PY0 + 7.4, t_, FS["tick"], C["net"] if v == 1 else C["gray"], anchor="middle", weight=700 if v == 1 else 400)
parts_ = (("median per model:", None), ("cross-entropy DAS", "disc"), ("maximum", "ring"))
wtot = sum(tw(t_, FS["axis"]) for t_, _ in parts_) + 2 * (4.0 + 3.6 + 3.0)
xq = min(max(cx0, (SX0 + SX1) / 2 - wtot / 2), W - 1.0 - wtot)            # under the scale, as far as the page allows
for t_, mk in parts_:
    if mk == "disc":
        xq += 4.0; b += f'<circle cx="{xq + 1.8:.2f}" cy="{PY0 + 13.0:.2f}" r="1.8" fill="{C["das"]}"/>'; xq += 3.6 + 3.0
    elif mk == "ring":
        xq += 4.0; b += f'<circle cx="{xq + 1.8:.2f}" cy="{PY0 + 13.0:.2f}" r="1.6" fill="#FFFFFF" stroke="{C["tgray"]}" stroke-width="0.7"/>'; xq += 3.6 + 3.0
    b += T(xq, PY0 + 15.0, t_, FS["axis"], C["muted"]); xq += tw(t_, FS["axis"])
assert xq < W - 0.9 and xq - wtot >= cx0 - 0.1, "(c)'s axis title runs off its panel"

# ------------------------------------------------------------------ no key: every mark is named where it is drawn
H = PY0B + 20.0
render("fig_readout", W, H, b, OUT)
print(f"fig_readout: {len(rows)} tasks in {len(OK)} models; ce above T in {len(above)}/{len(room)} with room, others {len(miss)} "
      f"(one-directional NPI but Pythia-160m's sentiment classification); squared error median {sq_med:.1f}% from T, middle half {sq_q1:.3f}-{sq_q3:.3f} T; one-question "
      f"{len(one)}, rank correlation with DAS {rho[0]:.3f}-{rho[-1]:.3f} per model, DAS over the prediction's middle half {ratio_q1:.3f}-{ratio_q3:.3f}")
print("fig_readout: (c) per-model medians r, rho", {m: (round(med_r[m], 3), round(med_rho[m], 3)) for m in ys})

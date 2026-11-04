"""Appendix figure (fig:atlas, text width, 397 pt): CausalGym's negative-polarity map, rebuilt in Pythia-1b under the
benchmark's protocol at every layer (the code's odd layers and the readout cell from experiments/results/inside/, its even layers from
experiments/results/inside_even/), read twice. Drawn as a heat map, as CausalGym's own Figure 5(b): words down, layers across (the paper's layer l = the output of
block l, the code's L - 1; out: the readout cell), one scale from 0 to T. Left, the network's own interchange in its best
context, G* (the context chosen on the training pairs, measured on the test pairs; at the readout cell, its own swap); right,
cross-entropy DAS, the mean of its three starts. Each map carries the other's path, outlined: on the left map DAS's best
region in each layer (orange), on the right map whole-state patching's largest region in each layer (blue; the text's route,
by vanilla, the network's own single swap; G*'s largest is asserted to be the same cell in every layer). The two paths part
at layers 4-8, where DAS stays on the licensor: the title's claim. No key: each path is named over its first cells (the
bracket's note on the left, which also gives DAS's and G*'s numbers there; a blue note on the right), and under each map its
shade from 0 to T is drawn as a scale. DAS's picks are asserted equal to
experiments/results/score_causalgym_map.json's. Brackets: the licensor at the paper's layers cgmNpiFrom to cgmNpiLicLast, where DAS still
picks it, with G*'s largest value and DAS's range there (the text's numbers, paper/sections/numbers_cgmap.tex, asserted equal).
Every number is a mean over the task's test pairs, over T.
Usage: python -I paper/scripts/fig_cgmap.py [--out=<dir>]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
from svgfig import C, FS, NIPSW, R, T, render, tw
from svgfig import L as line  # L is a layer here  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
M, TASK = "pythia-1b", "npi_any_subj-relc"
J1 = json.load(open(figdata.ROOT / f"experiments/results/inside/inside_{M}_{TASK}.json"))
Z1 = np.load(figdata.ROOT / f"experiments/results/inside/inside_{M}_{TASK}.npz")
J2 = json.load(open(figdata.ROOT / f"experiments/results/inside_even/inside_{M}_{TASK}_even.json"))
Z2 = np.load(figdata.ROOT / f"experiments/results/inside_even/inside_{M}_{TASK}_even.npz")
Tm = float(Z1["T"].mean()); NL = max(c["layer"] for c in J1["cells"])
NAMES = ["No/The", "farmer", "that", "contacted", "no", "guard", "has", "caught"]
val = {}
for J, Z in ((J1, Z1), (J2, Z2)):
    for c in J["cells"]:
        if (J is J2 and c["layer"] == NL) or not c.get("live", True):
            continue
        k = f"L{c['layer']}r{c['region']}"
        e = float(np.mean([Z[f"{k}/e/das_ce_s{s}_normal"] for s in range(3)], axis=0).mean() / Tm)
        g = float(Z[f"{k}/e/own_normal"].mean() / Tm)
        gs = float(Z[f"{k}/gstar_train"].mean() / Tm) if c["layer"] < NL else g     # the context chosen on the training pairs
        val[(c["layer"], c["region"])] = dict(e=e, g=g, gs=gs)
layers = sorted({L for L, _ in val})
das_best = {L: max((k for k in val if k[0] == L), key=lambda k: val[k]["e"]) for L in layers}
van_best = {L: max((k for k in val if k[0] == L), key=lambda k: val[k]["g"]) for L in layers}
gs_best = {L: max((k for k in val if k[0] == L), key=lambda k: val[k]["gs"]) for L in layers}
assert gs_best == van_best, "the blue path is both vanilla's largest and G*'s, layer by layer"
S = json.load(open(figdata.ROOT / "experiments/results/score_causalgym_map.json"))["maps"][f"{M}/{TASK}"]["primary"]["per_layer"]
assert all(tuple(S[str(L)]["cell"]) == das_best[L] for L in layers), "DAS's picks are the scorer's"


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]; z = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, z))


ramp = lambda col, v: mix("#F4F4F4", col, min(max(v, 0.0), 1.0) ** 0.7)  # noqa: E731
W = NIPSW
MB = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (figdata.ROOT / "paper/sections/numbers_cgmap.tex").read_text()))
# the licensor's cells after the network has moved on (the paper's layers cgmNpiFrom..cgmNpiLicLast, the code's one more): what
# G* and DAS give there, the text's numbers (asserted equal)
LIC = [L for L in layers if int(MB["cgmNpiFrom"]) <= L - 1 <= int(MB["cgmNpiLicLast"])]
assert all(das_best[L][1] == 1 for L in LIC)
gs_max = max(val[(L, 1)]["gs"] for L in LIC); das_lo, das_hi = min(val[(L, 1)]["e"] for L in LIC), max(val[(L, 1)]["e"] for L in LIC)
assert (f"{gs_max:.2f}", f"{das_lo:.1f}", f"{das_hi:.1f}") == (MB["cgmNpiGsMax"], MB["cgmNpiDasLo"], MB["cgmNpiDasHi"])
cw, ch = 10.2, 8.0
gx1, y0 = 44.0, 49.0
gx2 = gx1 + len(layers) * cw + 2.0 + 22.0
b = T(0, 9, "Rebuilt at every layer, DAS stays on the licensor after the network has moved on", FS["title"], weight=600)
b += T(0, 17.6, "Pythia-1b, CausalGym's negative-polarity task under its protocol; means over the test pairs, in T", FS["sub"], C["muted"])
b += T(gx1 + len(layers) * cw / 2, 31.0, "whole-state patching in its best context, G*", FS["axis"], C["net"], anchor="middle", weight=600)
b += T(gx2 + len(layers) * cw / 2, 31.0, "cross-entropy DAS, mean of three trained runs", FS["axis"], C["das"], anchor="middle", weight=600)
for k, nm in enumerate(NAMES):
    b += T(gx1 - 3.0, y0 + k * ch + ch / 2 + 2.1, nm, FS["axis"], C["ink"] if k == 0 else C["muted"], anchor="end")
NOTE = f"DAS still picks the licensor, {das_lo:.1f}\u2013{das_hi:.1f} T; G* at most {gs_max:.2f} T"
PATH = "whole-state patching's largest region in each layer"
for gx0, col, kv, mark, mark_col, note, top in ((gx1, C["net"], "gs", das_best, C["das"], NOTE, "T"),
                                                 (gx2, C["das"], "e", van_best, C["net"], None, "T or more")):
    cx = lambda jx, g=gx0: g + jx * cw + (2.0 if jx == len(layers) - 1 else 0.0)  # noqa: E731
    for (L, r), v in val.items():
        x, y = cx(layers.index(L)), y0 + (r - 1) * ch
        b += R(x + 0.3, y + 0.3, cw - 0.6, ch - 0.6, ramp(col, v[kv]), r=0.4)
    for L in layers:
        _, r = mark[L]; x, y = cx(layers.index(L)), y0 + (r - 1) * ch
        b += R(x + 0.5, y + 0.5, cw - 1.0, ch - 1.0, "none", r=0.5, stroke=mark_col, sw=0.9)
    if note:                                                                 # left: a bracket over the licensor's late cells
        bx0, bx1 = cx(layers.index(LIC[0])) + 0.8, cx(layers.index(LIC[-1])) + cw - 0.8
        b += line(bx0, y0 - 2.0, bx1, y0 - 2.0, C["ink"], 0.6) + line(bx0, y0 - 2.0, bx0, y0 - 0.2, C["ink"], 0.6) + line(bx1, y0 - 2.0, bx1, y0 - 0.2, C["ink"], 0.6)
        b += T((bx0 + bx1) / 2, y0 - 5.0, note, FS["note"], C["ink"], anchor="middle")
        assert (bx0 + bx1) / 2 + tw(note, FS["note"]) / 2 < gx2 - 4 and (bx0 + bx1) / 2 - tw(note, FS["note"]) / 2 > 0, note
    else:                                                                    # right: the blue path named over its first cells
        assert all(van_best[L][1] == 1 for L in layers[:4]), "the path starts on the licensor's row"
        b += T(cx(0), y0 - 5.0, PATH, FS["note"], C["net"])                  # as the left note: a note, not a heading
        assert cx(0) + tw(PATH, FS["note"]) < W
    yl = y0 + len(NAMES) * ch
    for jx, L in enumerate(layers):
        if L == NL or (L - 1) % 2 == 0:
            b += T(cx(jx) + cw / 2, yl + 6.2, "out" if L == NL else str(L - 1), FS["tick"], C["muted"], anchor="middle")
    b += T(gx0 + len(layers) * cw / 2, yl + 13.5, "layer", FS["tick"], C["muted"], anchor="middle")
    # the shade's scale, under the map's first columns: 0, five steps, its top
    sx = gx0 + tw("0", FS["tick"]) + 2.0
    b += T(gx0, yl + 13.5, "0", FS["tick"], C["muted"])
    b += "".join(R(sx + i * 4.0, yl + 9.4, 4.0, 4.0, ramp(col, (i + 0.5) / 5), r=0) for i in range(5))
    b += T(sx + 22.0, yl + 13.5, top, FS["tick"], C["muted"])
    assert sx + 22.0 + tw(top, FS["tick"]) + 4 < gx0 + len(layers) * cw / 2 - tw("layer", FS["tick"]) / 2
assert gx1 - 3.0 - tw("contacted", FS["axis"]) >= 0
render("fig_cgmap", W, yl + 17.0, b, OUT)
print("fig_cgmap: DAS picks " + " ".join(f"{L - 1}:{NAMES[das_best[L][1] - 1]}" for L in layers[:-1])
      + f"; licensor, paper layers {LIC[0] - 1}-{LIC[-1] - 1}: G* at most {gs_max:.3f} T, DAS {das_lo:.2f}-{das_hi:.2f} T")

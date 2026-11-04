"""Appendix figure (fig:indexswap, 288 pt wide, the width of its sentences, printed at that size and centred; the two pairs one
above the other): what CausalGym's released call reads when base and source tokenise to different lengths, and the one-line
fix, on two training pairs.

Both pairs are drawn by CausalGym's own sampler (seed 42, batch 4, as in their train.py: cleft batch 1 pair 1, agreement
batch 5 pair 3) and are rows of their released train.json; the script reads them there and rebuilds each region's last-token
index in Pythia's tokenizer, which equals their compute_pos('last'). The weight-free unit test
(experiments/causalgym/released_call_unit_test.py --all) checks every shifted (pair, region) of those batches, these two
included: the released call reads exactly what is drawn in grey, the fixed call what is outlined in ink.

Their intervention config has two slots: slot 0 collects the base run's states, which fit the representation-only methods
(mean difference, probes, PCA, k-means); slot 1 is the interchange. compute_pos returns pos = [source positions, base
positions], and the call (eval.py:31, train.py:62) passes
  released  {"sources->base": ([None, pos[1]], pos)}                    slot 1 reads the source at the base's index;
                                                                       slot 0 collects the base at the source's index
  fixed     {"sources->base": ([None, pos[0]], [pos[1], pos[1]])}
(a) cleft, the source longer: at "was" the released swap takes the source's "down", an earlier word, and the released collect
    reads the base past its last word, a padding position.
(b) subject-verb agreement across a relative clause, the base longer: at the subject the released swap takes the source's
    "that", a later word, and the released collect reads the base's "taxi".
Each panel's title names what the released call reads, computed from the pair; one-word labels name the two arrows (the swap);
what each slot does is defined in the caption, so the figure holds no paragraphs.
Colours: ink the fixed call (blue is whole-state patching in every other figure, so it is not used here); light grey the
released call (fill `grayl`; outline and arrow `gray`, solid, since dark grey dashed is reserved for a top).
Usage: python -I paper/scripts/fig_indexswap.py [--out=<dir>]
"""
import html
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402
sys.path.insert(0, str(figdata.ROOT / "experiments/lib"))
import pins  # noqa: E402
from svgfig import C, FS, L, R, T, arrows, head, render, tw  # noqa: E402

OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), str(figdata.ROOT / "paper/figures"))
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision=pins.MODELS["EleutherAI/pythia-160m"])
ROWS = json.load(open(hf_hub_download("aryaman/causalgym", "train.json", repo_type="dataset", revision=pins.CAUSALGYM[1])))
plain = lambda spans: "".join(spans).replace("<|endoftext|>", "").strip()      # noqa: E731
PAD = "⟨pad⟩"
REL, REL_FILL = C["gray"], C["grayl"]                                    # the released call: light grey, solid


def MONO(x, y, s, size, fill):
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" font-family="Menlo, SF Mono, monospace">'
            f'{html.escape(s)}</text>')


def pair(task, base, src):
    return next(r for r in ROWS if r["task"] == task and plain(r["base"]) == base and plain(r["src"]) == src)


def tokens(spans):
    """The token strings of a sentence built span by span, and the index of each span's last token (CausalGym's 'last')."""
    ids, ends = [], []
    for sp in spans:
        ids += tok(sp.replace("<|endoftext|>", tok.eos_token), add_special_tokens=False)["input_ids"]
        ends.append(len(ids) - 1)
    words = [("⟨eot⟩" if t == tok.eos_token_id else tok.decode([t]).strip()) for t in ids]
    return words, ends


def quote(w):
    return "the padding" if w == PAD else f"“{w}”"


def panel(x0, x1, y0, letter, r, region):
    """One training pair, columns aligned by token index. Ink: the fixed call (base and source at their own region index);
    light grey: the released call (the source read at the base's index, the base collected at the source's index)."""
    base, be = tokens(r["base"]); src, se = tokens(r["src"])
    pb, ps = be[region], se[region]
    assert pb != ps, "the region must sit at different indices in the two sentences"
    if ps >= len(base):                                                      # the released collect reads past the base's end
        assert ps == len(base); base = base + [PAD]
    longer = "Source" if ps > pb else "Base"
    s = head(x0, letter, f"{longer} longer: the released call reads {quote(src[pb])} and {quote(base[ps])}", (), y=y0 + 9.0)
    n = max(len(base), len(src))
    width = lambda w: max(15.5, 3.35 * len(w) + 6.5)                       # noqa: E731     a box fits its word
    cw = [max(width(base[i]) if i < len(base) else 0, width(src[i]) if i < len(src) else 0) for i in range(n)]
    avail = x1 - (x0 + 28.0) - 22.0 - 1.6 * (n - 1)                         # room for the boxes, the gaps and "→ label"
    assert sum(cw) <= avail, "the canvas fits every box at its word's width"
    xs = [x0 + 28.0 + sum(cw[:i]) + 1.6 * i for i in range(n)]
    yb, ys = y0 + 26.0, y0 + 70.0
    s += T(xs[0] - 4, yb + 6.6, "base", FS["note"], C["ink"], anchor="end", weight=600)
    s += T(xs[0] - 4, ys + 6.6, "source", FS["note"], C["ink"], anchor="end", weight=600)
    for row, y, blue, grey, num_y in ((base, yb, pb, ps, yb - 2.6), (src, ys, ps, pb, ys + 16.6)):
        for i, w in enumerate(row):
            if i == blue:
                s += R(xs[i], y, cw[i], 9.4, C["paper"], r=1.2, stroke=C["ink"], sw=0.9)
            elif i == grey:
                s += R(xs[i], y, cw[i], 9.4, REL_FILL, r=1.2, stroke=REL, sw=0.9)
            else:
                s += R(xs[i], y, cw[i], 9.4, C["faint"], r=1.2)
            s += T(xs[i] + cw[i] / 2, y + 6.6, w, FS["note"], C["ink"], anchor="middle")
            s += T(xs[i] + cw[i] / 2, num_y, str(i), FS["tick"], C["gray"], anchor="middle")
    lb = len(base) - 1 - (base[-1] == PAD)
    s += T(xs[n - 1] + cw[n - 1] + 3.0 if base[-1] == PAD or len(base) == n else xs[lb] + cw[lb] + 3.0, yb + 6.6,
           "→ " + r["base_label"].strip(), FS["note"], C["muted"])
    s += T(xs[len(src) - 1] + cw[len(src) - 1] + 3.0, ys + 6.6, "→ " + r["src_label"].strip(), FS["note"], C["muted"])
    cx = lambda i: xs[i] + cw[i] / 2                                        # noqa: E731
    lean = 4.0 if ps > pb else -4.0                                          # the fixed arrow lands beside the grey one
    y_from, y_to = ys - 1.6, yb + 11.4
    s += L(cx(ps), y_from, cx(pb) + lean, y_to, C["ink"], 0.9, marker="ab")
    s += L(cx(pb), y_from, cx(pb) - lean / 4, y_to, REL, 0.9, marker="ar")
    # one word per arrow, at its middle, on the side away from the other arrow, 4 pt clear of the line
    my = (y_from + y_to) / 2 + 2.1
    xb = (cx(ps) + cx(pb) + lean) / 2; xg = cx(pb) - lean / 8
    if ps > pb:                                                              # the fixed on the right, grey on the left
        s += T(xb + 4.5, my, "fixed", FS["note"], C["ink"], weight=600) + T(xg - 4.5, my, "released", FS["note"], C["muted"], anchor="end", weight=600)
    else:
        s += T(xb - 4.5, my, "fixed", FS["note"], C["ink"], anchor="end", weight=600) + T(xg + 4.5, my, "released", FS["note"], C["muted"], weight=600)
    return s, base, src, pb, ps, ys + 16.6


W = 288.0                                                                   # the sentences' width; centred on the page
b = arrows(("ab", C["ink"]), ("ar", REL), size=4.5)
cleft = pair("cleft", "What the old woman did was", "What the old woman walked down was")
agr = pair("agr_sv_num_subj-relc", "The taxi drivers that hurt the customer", "The officer that hurt the customer")
pa, base_a, src_a, pb_a, ps_a, end_a = panel(0.0, W, 0.0, "a", cleft, region=3)
pb_, base_b, src_b, pb_b, ps_b, end_b = panel(0.0, W, end_a + 10.0, "b", agr, region=1)
assert (src_a[ps_a], src_a[pb_a], base_a[ps_a], base_a[pb_a]) == ("was", "down", PAD, "was")
assert (src_b[ps_b], src_b[pb_b], base_b[ps_b], base_b[pb_b]) == ("officer", "that", "taxi", "drivers")
b += pa + pb_
# the call and its fix, in their code (eval.py:31 and train.py:62)
cy = end_b + 18.0
b += L(0.0, cy - 9.0, W, cy - 9.0, C["line"], 0.6)
b += T(0.0, cy, "released", FS["note"], C["muted"], weight=600)
b += MONO(50.0, cy, '{"sources->base": ([None, pos[1]], pos)}', FS["tick"], C["muted"])
b += T(0.0, cy + 9.5, "fixed", FS["note"], C["ink"], weight=600)
b += MONO(50.0, cy + 9.5, '{"sources->base": ([None, pos[0]], [pos[1], pos[1]])}', FS["tick"], C["ink"])
b += T(50.0, cy + 19.0, "in eval.py, line 31, and train.py, line 62, where pos is pos_interv", FS["note"], C["muted"])
assert 50.0 + tw("in eval.py, line 31, and train.py, line 62, where pos is pos_interv", FS["note"]) < W
H = cy + 22.0
render("fig_indexswap", W, H, b, OUT)
print(f"fig_indexswap: (a) base {pb_a}, source {ps_a}; (b) base {pb_b}, source {ps_b}; height {H:.1f}")

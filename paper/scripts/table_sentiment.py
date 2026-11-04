"""The sentiment direction of Tigges et al. (hollinsworth2024 in paper/refs.bib) re-run with the authors' code on seven models
from three starts (tab:sentiment, in Appendix E.3; experiments/results/score_sentiment_reread.json). Per model: its layers, the layer the
published rule selects (the largest score over layers, all positions patched: a complete cut), the score there (the three
starts' mean, % on the authors' logit-difference metric), the interval with that choice counted (the simultaneous band over
layers, blocks of two neighbouring prompts) read against 100%, the same reading in every start (asserted), the Bonferroni-t
reading beside it, and the network's own interchange at the adjective and verb positions (a partial cut): its largest value
over the layers after the first block. Writes paper/sections/tab_sentiment.tex.
Usage: python -I paper/scripts/table_sentiment.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402

S = json.load(open(figdata.ROOT / "experiments/results/score_sentiment_reread.json"))["models"]
ORDER = ["gpt2", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "SmolLM2-360M", "Qwen2.5-0.5B"]
NAME = {"gpt2": "GPT-2 small", "pythia-160m": "Pythia-160m", "pythia-410m": "Pythia-410m", "pythia-1b": "Pythia-1b",
        "pythia-1.4b": "Pythia-1.4b", "SmolLM2-360M": "SmolLM2-360M", "Qwen2.5-0.5B": "Qwen2.5-0.5B"}
READ = {"above": "above", "indistinguishable": "includes", "below": "below"}
# the band was chosen after the first start (s = 0) of Pythia-160m and of Pythia-410m had been seen (Section 7.3); the table
# marks them
RETRO = {"pythia-160m", "pythia-410m"}
DAG = r"$^\dagger$"


def bonf(b):   # the conservative reading beside the band's: a t interval at the selected layer, Bonferroni over the layers
    lo, hi = b["bonferroni"]
    return f"[{lo:.1f}, {hi:.1f}] " + ("above" if lo > 100 else "below" if hi < 100 else "includes")
assert sorted(S) == sorted(ORDER), "the table's models and the scored ones differ"
lines = [r"\begin{table}[tbp]",
         r"  \caption{\textbf{The sentiment direction re-run on seven models.} We use the authors' code with three initialisations "
         r"per model. The score is the effect of patching the direction at every position of a layer, a complete cut where "
         r"whole-state patching gives exactly 100\%. It is given in \% on the authors' logit-difference metric (the share of the gap "
         r"from the flipped run to the clean one), at the layer that the published rule selects (the largest over layers), averaged "
         r"over the initialisations. Layers are numbered as in the rest of the paper, the output of block $\ell$ as layer $\ell$. Interval: a band that is simultaneous over "
         r"the layers and so counts the choice of layer (blocks of two neighbouring prompts, 10{,}000 resamples), compared with "
         r"100\%. Each initialisation on its own gives the same result. Bonferroni-t: a conservative $t$ interval at "
         r"the selected layer, corrected over the layers as if they were independent (neighbouring layers are strongly "
         r"correlated). $^\dagger$The band was chosen after the first initialisation of this model had been seen, but before its "
         r"other two initialisations were seen. Whole-state patching, partial cut: the effect of whole-state patching at the adjective and verb "
         r"positions, as its largest value over the layers after the first block. At the embeddings, where only these two positions "
         r"differ, this site is a complete cut, and the effect is exactly 100\%. Published, Pythia-1.4b: 109.8\%.}",
         r"  \label{tab:sentiment}", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{5pt}",
         r"  \begin{tabular}{@{}lrrrllr@{}}", r"    \toprule",
         r"    model & layers & selected & \begin{tabular}[b]{@{}c@{}}score\\(\%)\end{tabular} & \begin{tabular}[b]{@{}c@{}}band,\\choice counted\end{tabular} & Bonferroni-t & \begin{tabular}[b]{@{}c@{}}whole-state\\patching,\\partial cut (\%)\end{tabular} \\", r"    \midrule"]
for m in ORDER:
    v = S[m]; b = v["block2"]["mean"]
    assert all(v["block2"][f"seed{s}"]["against_100"] == b["against_100"] for s in (0, 1, 2)), f"{m}: the starts disagree"
    own = max(x["own_E1"] for L, x in v["own_E1_per_layer"].items() if int(L) > 0)       # a complete cut at the embeddings: exactly 100
    lines.append(f"    {NAME[m]}{DAG if m in RETRO else ''} & {len(v['own_E1_per_layer']) - 1} & {b['L_star'] - 1} & {b['Y_star']:.1f} & "
                 f"[{b['lower']:.1f}, {b['upper']:.1f}] {READ[b['against_100']]} & {bonf(b)} & {own:.1f} \\\\")
# the text (Section 7.3 and Appendix E.3): no model lies above 100% under the band or under the conservative reading
assert not any(S[m]["block2"]["mean"]["against_100"] == "above" or bonf(S[m]["block2"]["mean"]).endswith("above") for m in S if "block2" in S[m])
lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
out = figdata.ROOT / "paper/sections/tab_sentiment.tex"
out.write_text("% Written by paper/scripts/table_sentiment.py from experiments/results/score_sentiment_reread.json; do not edit.\n"
               + "\n".join(lines) + "\n")
print("\n".join(lines))

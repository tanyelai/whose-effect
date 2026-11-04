"""The models table (tab:models, in App. B.2): every model any part of the paper uses, its architecture from its config at the
pinned revision (paper/scripts/model_configs.json, fetched once from the Hub), and where it is used, read from the result
files themselves:
  readout   Section 5 and App. C.1: the readout's outputs exist for the model (experiments/results/readout/readout_<tag>_m_*.json)
  maps      Section 6: the model is in a scored set of maps (experiments/results/inside_sets/<set>.post.json)
  benchmarks Section 7: c, CausalGym's released code (experiments/results/score_causalgym_rerun.json); s, the sentiment authors' code
            (experiments/results/score_sentiment_reread.json); p, CausalGym's map protocol (the maps Section 7.1 reads, PROTOCOL below)
Usage: python -I paper/scripts/table_models.py     writes paper/sections/tab_models.tex
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.load(open(ROOT / "paper/scripts/model_configs.json"))
tag = lambda m: m.split("/")[-1]  # noqa: E731
# CausalGym's map protocol: the models of CausalGym's 29 maps rebuilt at every layer (experiments/results/score_causalgym_map.json)
PROTOCOL = set(json.load(open(ROOT / "experiments/results/score_causalgym_map.json"))["models"])
FAMILIES = [  # display order: family, its citation key (on the family's first row), members
    ("Pythia", "biderman2023", ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
                                "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-6.9b"]),
    ("GPT-2", "radford2019", ["gpt2"]),
    ("SmolLM2", "allal2025", ["HuggingFaceTB/SmolLM2-360M"]),
    ("Qwen2.5", "qwen2024", ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-32B"]),
    ("Llama", "dubey2025", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.1-8B"]),
    ("Gemma-2", "team2024", ["google/gemma-2-9b"]),
]
LICENCE = {"mit": "MIT", "apache-2.0": "Apache-2.0", "llama3.1": "Llama 3.1 Community", "llama3.2": "Llama 3.2 Community",
           "gemma": "Gemma Terms of Use"}
NAME = {"gpt2": "GPT-2 small", "google/gemma-2-9b": "Gemma-2-9B"}

readout = {Path(f).name[len("readout_"):].split("_m_")[0] for f in glob.glob(str(ROOT / "experiments/results/readout/readout_*_m_*.json"))}
maps = set()
for f in glob.glob(str(ROOT / "experiments/results/inside_sets/*.post.json")):
    if Path(f).name.endswith(("_excess_rank.post.json", "_forms.post.json", "_offroute.post.json")):
        continue                                                   # the readings beside a set, not a set
    maps |= set(json.load(open(f)).get("models", {}))
rerun = set(json.load(open(ROOT / "experiments/results/score_causalgym_rerun.json")))
sent = set(json.load(open(ROOT / "experiments/results/score_sentiment_reread.json"))["models"])
used = {m for _, _, ms in FAMILIES for m in ms}
assert used == set(CFG), f"configs and table differ: {sorted(used ^ set(CFG))}"
for name, s in (("readout", readout), ("maps", maps), ("CausalGym's code", rerun), ("sentiment code", sent)):
    assert s <= {tag(m) for m in used}, f"{name}: a model the table lacks: {sorted(s - {tag(m) for m in used})}"

ck = r"$\checkmark$"
rows = []
for _, _, ms in FAMILIES:
    for m in ms:
        c, t = CFG[m], tag(m)
        heads = f"{c['heads']}" + (f" ({c['kv_heads']})" if c["kv_heads"] and c["kv_heads"] < c["heads"] else "")
        reruns = ", ".join(x for x, s in (("c", rerun), ("s", sent), ("p", PROTOCOL)) if t in s)
        if not (t in readout or t in maps or reruns):
            continue                                               # a model nothing uses is not listed
        nm = NAME.get(m, t.replace("pythia", "Pythia"))
        rows.append(f"    {nm} & {c['layers']} & {c['width']} & {heads} & {LICENCE[c['licence']]} & "
                    f"{ck if t in readout else ''} & {ck if t in maps else ''} & {reruns} \\\\")
out = r"""\begin{table}[tbp]
  \caption{\textbf{Models.} Width is the width of the residual stream, and heads counts the query heads, with the key--value
  heads in parentheses where there are fewer. The last three columns show where each model is used. Readout refers to
  \cref{sec:inflation} and \cref{app:readout} (every readout analysis), and maps refers to \cref{sec:offroute}. Benchmarks
  refers to \cref{sec:published}, where c marks CausalGym's released code, s the code of the sentiment authors, and p our
  maps under CausalGym's protocol. All models are base checkpoints (not instruction-tuned) under the Hugging Face
  organisations of their authors (EleutherAI, \texttt{gpt2}, HuggingFaceTB, Qwen, meta-llama, google). Our runs read them at
  pinned revisions, and CausalGym's code reads them at their current address. Pythia is the non-deduplicated suite; at 14m
  and 31m, the weights at that address were replaced in February 2026 (\cref{app:indexswap}). Licences are as released by
  their authors. Sources: """ + "; ".join(
    f"{fam} \\citep{{{cite}}}" for fam, cite, _ in FAMILIES) + r""".}
  \label{tab:models}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}lrrrlccl@{}}
    \toprule
    model & layers & width & heads & licence & readout & maps & benchmarks \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
(ROOT / "paper/sections/tab_models.tex").write_text(out)
print(f"tab_models: {len(rows)} models; readout {len(readout)}, maps {len(maps)}, CausalGym's code {len(rerun)}, sentiment {len(sent)}")

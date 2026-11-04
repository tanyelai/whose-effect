"""One test pair from each data source of Section 5's main population (tab:tasks, in App. B.1): the first pair of one
task's test file, split into its regions, with its labels, the number of label pairs among the task's test pairs, and the
task's train/test counts. Regions where base and source differ are written base/source, as in the map figures; the
start-of-text token is left out. The files are the ones the readout read (experiments/matrix/registry.py's groups): CausalGym from
the Hugging Face snapshot the runs pinned (registry.CAUSALGYM_REV), the other sources from their data folders, and a source
built per model (Kumon & Yanaka, Tigges et al., RAVEL) as built for GPT-2. Each source's task is named in the table; every
one is in GPT-2's main population, and the table has one row per source of the population (asserted against
experiments/results/score_readout.json).
Writes paper/sections/tab_tasks.tex. Usage: python -I paper/scripts/table_tasks.py
"""
import json
import os
import sys
from pathlib import Path

# the Hugging Face cache, read offline (the pinned CausalGym snapshot must be in it: python -I experiments/run/download.py)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figdata  # noqa: E402

ROOT = figdata.ROOT
sys.path.insert(0, str(ROOT / "experiments/matrix"))
import registry  # noqa: E402


def causalgym(task):
    from huggingface_hub import hf_hub_download
    load = lambda split: json.load(open(hf_hub_download(registry.CAUSALGYM_HF, f"{split}.json", repo_type="dataset",  # noqa: E731
                                                        revision=registry.CAUSALGYM_REV)))
    return [r for r in load("train") if r["task"] == task], [r for r in load("test") if r["task"] == task]


def local(folder, task):
    return json.load(open(ROOT / folder / task / "train.json")), json.load(open(ROOT / folder / task / "test.json"))


# (source as the paper names it, what the task is, the readout's group, the task, its rows); groups and folders as
# registry.groups
SOURCES = [
    ("CausalGym", "negative polarity", "causalgym", "npi_any_subj-relc", lambda: causalgym("npi_any_subj-relc")),
    ("CausalGym, number only", "agreement", "number_only", "agr_sv_num_subj-relc__num",
     lambda: local("experiments/tasks/agr_num", "agr_sv_num_subj-relc__num")),
    ("Kumon \\& Yanaka", "negative polarity", "kumon", "NPIConditionals", lambda: local("experiments/data/gpt2/kumon", "NPIConditionals")),
    ("Tigges et al.", "movie template", "sentiment_templates", "sentiment_movie",
     lambda: local("experiments/data/gpt2/sentiment_templates", "sentiment_movie")),
    ("SST", "continuation", "sst", "sst_continuation", lambda: local("experiments/tasks/sst", "sst_continuation")),
    ("RAVEL", "city to continent", "ravel_attributes", "ravel_city_continent",
     lambda: local("experiments/data/gpt2/ravel_attributes", "ravel_city_continent")),
    ("IOI, Wang et al.", "twenty names", "ioi_wang", "ioi_wang_20", lambda: local("experiments/tasks/ioi_wang", "ioi_wang_20")),
    ("IOI, ours", "one name pair", "ioi_ours_label_signed", "ioi_fixed", lambda: local("experiments/tasks/ioi", "ioi_fixed")),
]
S = json.load(open(ROOT / "experiments/results/score_readout.json"))
MAIN = {(r["group"], r["task"]) for r in S["rows"]["gpt2"] if r["main"] and r["defined"]}
assert {(g, t) for _, _, g, t, _ in SOURCES} <= MAIN, "a task shown is not in GPT-2's main population"
assert len({s for s, *_ in SOURCES}) == len(SOURCES) == len({r["source"] for m in S["rows"] for r in S["rows"][m] if r["main"]})


def esc(s):
    s = s.replace("\\", "\\textbackslash{}")
    for c in "&%$#_{}":
        s = s.replace(c, "\\" + c)
    return s.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}").replace("\n", " ")


def tt(name):
    """A task identifier in typewriter type, breakable after its underscores and hyphens."""
    return "\\texttt{" + esc(name).replace("\\_", "\\_\\allowbreak{}").replace("-", "-\\allowbreak{}") + "}"


def clean(w):
    return " ".join(w.replace("<|endoftext|>", "").split())


def pair_cell(r):
    out = []
    for b, s in zip(r["base"], r["src"]):
        b, s = clean(b), clean(s)
        if not b and not s:
            continue
        out.append(esc(b) if b == s else f"\\textbf{{{esc(b)}/{esc(s)}}}")
    return " \\textperiodcentered\\ ".join(out)


def labels(r):
    return f"{esc(clean(r['base_label']) or r['base_label'].strip() or '.')} / {esc(clean(r['src_label']) or '.')}"


rows = []
for source, what, _, task, load in SOURCES:
    tr, te = load()
    r = te[0]
    npairs = len({tuple(sorted((x["base_label"], x["src_label"]))) for x in te})
    rows.append(f"    {source}: {what}\\newline {tt(task)} & {pair_cell(r)} & {labels(r)} & {npairs} & {len(tr)}/{len(te)} \\\\")
    print(f"{source:24s} {task:28s} {npairs:3d} {len(tr)}/{len(te)}  {clean(' '.join(r['base']))!r} -> {r['base_label']!r} | {r['src_label']!r}")
tex = r"""\begin{table}[tbp]
  \caption{\textbf{One test pair from each data source.} Each row shows the first pair of one task's test file, split into
  its regions. Regions are separated by dots, a region where base and source differ is written base/source, as in the map
  figures, and the start-of-text token is left out. Labels: the next token of the base input, then that of the source
  input. Label pairs: the distinct pairs among the task's test pairs. Train/test: the pairs in the task's files; the readout
  keeps those whose labels are single tokens. Kumon \& Yanaka's, Tigges et al.'s and RAVEL's tasks are built per model and
  shown as built for GPT-2. \Cref{tab:readout} counts the tasks of each source. The maps of \cref{sec:offroute} cover
  \inTasks{} tasks of these sources in the six small models (\cref{app:fullgrid}), and in the four large models they cover the
  first task shown and two more of CausalGym's tasks.}
  \label{tab:tasks}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \hyphenpenalty=10000 \exhyphenpenalty=10000
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.4\textwidth}>{\raggedright\arraybackslash}p{0.12\textwidth}cc@{}}
    \toprule
    source: task & pair & labels & \shortstack{label\\pairs} & \shortstack{train/\\test} \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
(ROOT / "paper/sections/tab_tasks.tex").write_text(tex)
print("wrote paper/sections/tab_tasks.tex")

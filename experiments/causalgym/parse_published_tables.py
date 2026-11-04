"""CausalGym's per-size score tables (Arora et al. 2024, their Appendix E.2, Tables 3-20) at all nine published Pythia sizes.

Section 7 (benchmark scores) and Section 7.1 quote these published values
(paper/scripts/table_causalgym.py, paper/scripts/fig_benchmarks.py). The three smallest sizes (14m, 31m, 70m) have one probe
column, nine numbers per row; the larger sizes have two, ten numbers per row. Both layouts are read:
  14m, 31m, 70m:  Task Acc., DAS, Probe, Mean, PCA, k-means, LDA, Rand., Vanilla
  160m to 6.9b:   Task Acc., DAS, Probe 0, Probe 1, Mean, PCA, k-means, LDA, Rand., Vanilla
for both the main and the selectivity table of each size (in the flattened text a table's rows precede its caption). Rows:
the 29 tasks, named as in experiments/causalgym/index_shift_exposure.json, and the "Average" row; every table must have all 30.
Checks: the captions are Tables 3 to 20 in order; every table has its 30 rows; one spot value (pythia-1b,
npi_any_subj-relc: DAS 10.51, vanilla 4.41).
Input: the text of the arXiv HTML version of their paper, saved as plain text; the parser expects each table cell on a line
of its own.
Output: {"<size>|main" or "<size>|sel": {task: {column: value}}}; the release keeps it as
experiments/causalgym/causalgym_published_tables.json.
Usage: python experiments/causalgym/parse_published_tables.py <their paper as text> <repo root> <out.json>
"""
import json
import re
import sys
from pathlib import Path

t = open(sys.argv[1]).read().split("\n"); ROOT = Path(sys.argv[2])
TASKS = set(json.load(open(ROOT / "experiments/causalgym/index_shift_exposure.json"))) | {"Average"}
SMALL = ("pythia-14m", "pythia-31m", "pythia-70m")
caps = {}
for i, ln in enumerate(t):
    m = re.search(r"Table (\d+): (pythia-[0-9.]+[mb])( \(selectivity\))?", ln)
    if m:
        caps[i] = (int(m.group(1)), m.group(2), bool(m.group(3)))
idx = sorted(caps)
assert [caps[i][0] for i in idx] == list(range(3, 21)), [caps[i][0] for i in idx]
out = {}
for k, ci in enumerate(idx):
    _, model, sel = caps[ci]
    nc = 9 if model in SMALL else 10
    start = idx[k - 1] if k else ci - 600
    block = [x.strip() for x in t[start:ci] if x.strip()]
    rows, cur, nums = {}, None, []
    for x in block:
        if x in TASKS:
            if cur and len(nums) >= nc:
                rows[cur] = nums[:nc]
            cur, nums = x, []
        elif cur and re.fullmatch(r"-?\d+\.\d+", x):
            nums.append(float(x))
    if cur and len(nums) >= nc:
        rows[cur] = nums[:nc]
    keys = (["acc", "das", "probe0", "mean", "pca", "kmeans", "lda", "rand", "vanilla"] if nc == 9 else
            ["acc", "das", "probe0", "probe1", "mean", "pca", "kmeans", "lda", "rand", "vanilla"])
    out[f"{model}|{'sel' if sel else 'main'}"] = {task: dict(zip(keys, v)) for task, v in rows.items()}
for key, tab in out.items():
    assert len(tab) == 30, (key, len(tab))                                  # 29 tasks and the Average row
assert abs(out["pythia-1b|main"]["npi_any_subj-relc"]["das"] - 10.51) < 1e-9 and abs(out["pythia-1b|main"]["npi_any_subj-relc"]["vanilla"] - 4.41) < 1e-9
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(f"wrote {sys.argv[3]}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))

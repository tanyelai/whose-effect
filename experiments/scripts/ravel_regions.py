"""One region layout for the RAVEL tasks whose prompts come in two forms. A RAVEL prompt is either a few-shot prefix, the
entity and the query (three regions) or the entity and the query alone (two regions: the prompt starts with the entity), and
some tasks mix the two. experiments/scripts/inside.py takes a map's regions from its first training row and never swaps region 0, the
start of the prompt, so in such a task every two-region row gets an empty region 0 in front, in the base and in the source.
An empty span has no tokens, so the token ids and T stay as they are, and region 1 is the entity in every row. Every other
field and the order of the rows stay as they are.
The tasks: those that experiments/matrix/inside_maps.tsv maps on experiments/data/<model>/ravel_attributes_regions. Input: the same task's
rows in experiments/data/<model>/ravel_attributes/<task>/{train,test}.json (experiments/scripts/filter_known_attributes.py, --start=0).
Output: experiments/data/<model>/ravel_attributes_regions/<task>/{train,test}.json, which experiments/run/check_data.py compares with the data
of the paper's runs. Strings only: no tokenizer, no model.
Usage: python -I experiments/scripts/ravel_regions.py [<model> ...]   (default: every model the list names)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = "ravel_attributes_regions"


def layout(r):
    """The row in the three-region layout, and whether it changed."""
    if len(r["base"]) != len(r["src"]):
        raise ValueError(f"base and source have different region counts: {r}")
    if len(r["base"]) == 3:
        if r["prompt"].startswith("%s") or r["base"][0] != r["src"][0]:
            raise ValueError(f"a three-region row must open with the same prefix in base and source: {r}")
        return r, False
    if len(r["base"]) != 2 or not r["prompt"].startswith("%s"):
        raise ValueError(f"a two-region row must start with the entity: {r}")
    return dict(r, base=[""] + r["base"], src=[""] + r["src"]), True


maps = set()
for line in (REPO / "experiments/matrix/inside_maps.tsv").read_text().splitlines():
    model, _, task, data = line.split("\t")[:4]
    if data == f"experiments/data/{model}/{OUT}" and (len(sys.argv) == 1 or model in sys.argv[1:]):
        maps.add((model, task))
for model, task in sorted(maps):
    src, out = REPO / f"experiments/data/{model}/ravel_attributes/{task}", REPO / f"experiments/data/{model}/{OUT}/{task}"
    rows = {s: json.load(open(src / f"{s}.json")) for s in ("train", "test")}
    laid_out = {s: [layout(r) for r in rows[s]] for s in rows}
    changed = sum(c for s in laid_out for _, c in laid_out[s])
    if not changed:
        raise ValueError(f"{model}/{task}: every row already has three regions, so the map should read {src}")
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "test"):
        json.dump([r for r, _ in laid_out[s]], open(out / f"{s}.json", "w"))
        back = json.load(open(out / f"{s}.json"))
        same = all({k: v for k, v in b.items() if k not in ("base", "src")} == {k: v for k, v in a.items() if k not in ("base", "src")}
                   and "".join(b["base"]) == "".join(a["base"]) and "".join(b["src"]) == "".join(a["src"]) and len(b["base"]) == 3
                   for b, a in zip(back, rows[s]))
        if len(back) != len(rows[s]) or not same:
            raise ValueError(f"{model}/{task}/{s}: the rewritten rows differ from the input beyond region 0")
    print(f"{model}/{task}: {changed} of {len(rows['train']) + len(rows['test'])} rows given an empty region 0")
print(f"{len(maps)} tasks in one region layout")

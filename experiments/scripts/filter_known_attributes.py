"""Build each RAVEL attribute task of experiments/tasks/ravel/attributes/candidates.json from what a model knows, before any run.

An (entity, prompt) combination is known when the model's most likely next token after the prompt with the entity is the
entity's label (RAVEL's own criterion, in one-token form: RAVEL keeps an example when the model's continuation is the
attribute's value). A prompt is usable in a split when it has at least two known entities for each value there.
Pairs, seed 94, without replacement: a prompt drawn uniformly among the usable prompts that still have an unused pair, then an
unused pair of known entities under that prompt, one per value. Base and source use the same prompt (a minimal pair). Up to
200 training and 50 test pairs, each with its mirror (CausalGym's sizes and format). Drawing without replacement keeps rows
from repeating where few entities are known. A task's pairs depend only on its own known sets and seed.
A task is written when each split has a usable prompt; otherwise it is reported as not known enough, and an earlier folder of
it is removed. The number of distinct known entities per value is reported, not required.
Spans: [the text before the entity (without its last space), (" " if that text ended in a space) + entity, the rest].
  --start=0   RAVEL's convention: the raw prompt, no start-of-text token (default); writes ravel_attributes/
  --start=1   CausalGym's convention, a start-of-text token before the prompt; writes ravel_attributes_sot/
  --ood=1     also the held-out-prompt tasks <name>_ood (training rows from RAVEL's train and val prompts, test rows from its
              test prompts), from the same known sets (every (entity, prompt) is scored once)
Writes experiments/data/<model tag>/<folder>/<task>/{train,test}.json and a summary of the known counts, <folder>/known.json.
Usage: python -I experiments/scripts/filter_known_attributes.py --model=<hf id> [--revision=<sha>] [--start=0|1] [--ood=1] [--tasks=a,b]
       [--device=cuda]
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402

_, args, model, tok = common.start("filter_known_attributes", sys.argv[1:])
DEV, FIX = common.RUN["dev"], families.start_of_text(tok)
OOD, START = args.get("ood") == "1", args.get("start", "0") == "1"
C = json.load(open(common.REPO / "experiments/tasks/ravel/attributes/candidates.json"))["tasks"]
TASKS = args["tasks"].split(",") if "tasks" in args else sorted(C)
OUT = common.REPO / "experiments/data" / common.RUN["tag"] / ("ravel_attributes_sot" if START else "ravel_attributes")


def spans(prompt, ent):
    pre, post = prompt.split("%s")
    head = families.START if START else ""
    if pre.endswith(" "):
        sp = [head + pre[:-1], " " + ent, post]
    else:
        sp = [head + pre, ent, post] if pre else [head, ent, post]
    return [s for s in sp if s != ""] if not START else sp


def known_set(task, split, pool):
    """{prompt: {value: [known entities]}} for one split, batched by token length (no padding, so no mask needed)."""
    t = C[task]; lab = [tok(x, add_special_tokens=False)["input_ids"] for x in t["labels"]]
    assert all(len(x) == 1 for x in lab), lab
    lab = [x[0] for x in lab]
    jobs = [(p, vi, e) for p in pool for vi, v in enumerate(t["values"]) for e in t["entities"][split][v]]
    ids = [families.encode(tok, spans(p, e), FIX)[0] for p, _, e in jobs]
    groups, keep = {}, {p: {v: [] for v in t["values"]} for p in pool}
    for i, q in enumerate(ids):
        groups.setdefault(len(q), []).append(i)
    with torch.no_grad():
        for idx in groups.values():
            for k in range(0, len(idx), 64):
                chunk = idx[k:k + 64]
                z = model(torch.tensor([ids[i] for i in chunk], device=DEV)).logits[:, -1]
                top = z.argmax(-1)
                for j, i in enumerate(chunk):
                    p, vi, e = jobs[i]
                    if int(top[j]) == lab[vi]:
                        keep[p][t["values"][vi]].append(e)
    return keep


outs, report = [], {}
for task in TASKS:
    t = C[task]; vals = t["values"]; allp = sorted(t["prompts"])
    known_all = {s: known_set(task, s, allp) for s in ("train", "test")}    # every (entity, prompt) scored once
    prompt_pools = {task: {"train": allp, "test": allp}}
    if OOD:
        prompt_pools[task + "_ood"] = {"train": [p for p in allp if t["prompts"][p] in ("train", "val")],
                                   "test": [p for p in allp if t["prompts"][p] == "test"]}
    for name, pools in prompt_pools.items():
        known = {s: {p: known_all[s][p] for p in pools[s]} for s in ("train", "test")}
        distinct = {s: {v: len({e for p in known[s] for e in known[s][p][v]}) for v in vals} for s in known}
        usable = {s: [p for p in pools[s] if all(len(known[s][p][v]) >= 2 for v in vals)] for s in known}
        ok = all(usable[s] for s in usable)                             # a usable prompt in each split is the only condition
        report[name] = dict(distinct_known=distinct, usable_prompts={s: len(u) for s, u in usable.items()},
                            prompts={s: len(pools[s]) for s in pools}, written=ok)
        print(json.dumps({name: report[name]}), flush=True)
        if not ok:
            if (OUT / name).is_dir():                                 # no stale task from an earlier run
                shutil.rmtree(OUT / name)
            continue
        rng = random.Random(94); d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        lb = dict(zip(vals, t["labels"]))
        for split, n in (("train", 200), ("test", 50)):
            # pairs without replacement: a prompt drawn uniformly among those with an unused pair left, then an unused pair
            left = {p: [(a, b) for a in known[split][p][vals[0]] for b in known[split][p][vals[1]]] for p in usable[split]}
            rows, i = [], 0
            while i < n and any(left.values()):
                p = rng.choice(sorted(q for q, v in left.items() if v))
                a, b = left[p].pop(rng.randrange(len(left[p])))
                for base, src, bt, st in ((a, b, vals[0], vals[1]), (b, a, vals[1], vals[0])):
                    rows.append(dict(base=spans(p, base), src=spans(p, src), base_label=lb[bt], src_label=lb[st],
                                     base_type=bt, src_type=st, task=name, pair=i, prompt=p))
                i += 1
            json.dump(rows, open(d / f"{split}.json", "w"), indent=0); outs.append(d / f"{split}.json")
json.dump(dict(criterion="argmax is the label", start=START, ood=OOD, report=report), open(OUT / "known.json", "w"), indent=1)
common.finish(outs + [OUT / "known.json"], summary=dict(written=sum(r["written"] for r in report.values()),
                                                                          tasks=len(report)))

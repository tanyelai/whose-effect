"""How often CausalGym's released code reads a base or source state at the other sentence's token index, per task.

The paper's Appendix E.2 (token positions in the released code). CausalGym's Batch.compute_pos returns [source positions, base
positions] (its own comment: "dim 0 -> src, base"), and train.py and eval.py, which das.py calls, pass unit_locations
{"sources->base": ([None, pos[1]], pos)}: the collect intervention (index 0) reads the BASE run at the SOURCE's token index,
and the interchange (index 1) reads the SOURCE run at the BASE's token index. Where the two sentences tokenise to different
lengths before or inside an intervened span, both are off.
Counted per task: the share of pairs whose base and source indices differ at each intervened span
(range(first_var_pos, length), strategy "last"), with the code's own samplers (train: seed 42, 100 batches of 4; eval:
seed 1, 25 batches, training bases discarded) and the Pythia tokenizer (the same for every Pythia size). A task's exposure is
the largest of these shares over its spans in the training sample.
It imports and runs CausalGym's own data code (data.py), so it needs a clone of CausalGym's repository at commit 0f3129f
(github.com/aryamanarora/causalgym) and an environment with their dependencies, pyvene among them
(experiments/causalgym/rerun/requirements_frozen.txt); it changes the working directory to the clone.
Output: one JSON object per task (spans, first_var_pos, length, the train and eval shares, exposure); the release keeps it as
experiments/causalgym/index_shift_exposure.json, which paper/scripts/table_causalgym.py and fig_benchmarks.py read, and
parse_published_tables.py for the task names.
Usage: <Python of that environment> experiments/causalgym/index_shift_exposure.py <causalgym clone> <out.json>
"""
import json
import os
import sys

CG = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, CG); os.chdir(CG)
from transformers import AutoTokenizer  # noqa: E402
from data import Dataset, list_datasets  # noqa: E402

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"); tok.pad_token = tok.eos_token
out = {}
for name in [d for d in list_datasets() if d.startswith("syntaxgym/")]:
    ds = Dataset.load_from(name)
    train = ds.sample_batches(tok, 4, 100, "cpu", seed=42)
    discard = {"".join(p.base) for b in train for p in b.pairs}
    evalset = ds.sample_batches(tok, 4, 25, "cpu", seed=1, discard=discard)
    row = dict(spans=ds.span_names, first_var_pos=ds.first_var_pos, length=ds.length)
    for split, batches in (("train", train), ("eval", evalset)):
        n, mis = 0, {}
        for b in batches:
            src, base = b.compute_pos("last")
            for i in range(len(b.pairs)):
                n += 1
                for s in range(ds.first_var_pos, ds.length):
                    if base[i][s] != src[i][s]:
                        mis[s] = mis.get(s, 0) + 1
        row[split] = dict(pairs=n, shifted={ds.span_names[s]: mis.get(s, 0) / n for s in range(ds.first_var_pos, ds.length)})
    row["exposure"] = max(row["train"]["shifted"].values()) if row["train"]["shifted"] else 0.0
    out[name.split("/")[-1]] = row
    print(name, f"exposure {row['exposure']:.3f}", {k: round(v, 3) for k, v in row["train"]["shifted"].items()}, flush=True)
json.dump(out, open(OUT, "w"), indent=1)

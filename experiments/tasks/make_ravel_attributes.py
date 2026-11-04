"""Two-label factual tasks from every RAVEL entity type and attribute that passes a fixed rule (no model).

RAVEL (Huang et al., 2024; MIT licence; github.com/explanare/ravel, commit e421fc9e132b3613b1d0020f856d539217fb30ca, data.tgz,
sha256 b148f0ded1979cb0e17e301cfda41cb120517a109d3c758cc768e376946d96bc). For each entity type and attribute:
  prompts    RAVEL's prompts for the attribute with one "%s", grouped by the form of the word that follows them: "spaced" when
             the text after "%s" ends in a letter, ":" or "," (the next word starts with a space), "json" when it ends in a
             quote (the next word follows without a space); each group is a task of its own (<name>, <name>_json), since a
             task asks one question with one label pair; prompts of any other ending are listed as left out. Each prompt keeps
             RAVEL's prompt split.
  values     per task, the two most frequent values of the attribute (ties by name) whose label is one token for every
             tokenizer given. The label is the value in the task's form (" Asia" or "Asia"), except for the Gender prompts
             that end in "for" ("%s: for his contributions"): their labels are " his" / " her", the word those prompts elicit
             (RAVEL labels each example by the model's own continuation, prompt_to_output); the other spaced Gender prompts
             (those that end in "gender:") form their own task (<name>_value) with the value as label.
  entities   RAVEL's entity split (training rows: its train and val entities; test rows: its test entities); per value and
             split, a sample of up to 60 drawn with seed 94 from the sorted list (RAVEL samples entities at random).
  rule       at least one prompt and at least one entity per value and split (a split without an entity of a value has no
             pair), else the task is reported as dropped. What can be built for a model is then decided by the per-model
             filter's own rule (two known entities per value under a usable prompt).
experiments/scripts/filter_known_attributes.py then keeps, per model, the (entity, prompt) combinations the model knows and builds the
pairs.

Output: experiments/tasks/ravel/attributes/candidates.json: the tasks with their values, labels, prompts (with RAVEL's prompt
split) and entities, the dropped tasks with the reason, the prompts left out, and a pointer to this docstring for the
rules.
Usage: python -I experiments/tasks/make_ravel_attributes.py <RAVEL's extracted data directory> --tokenizers=<dir>,<dir>,...
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

SRC = Path(sys.argv[1]); OUT = Path(__file__).resolve().parent / "ravel/attributes"
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:] if "=" in a)
TOKS = [AutoTokenizer.from_pretrained(p) for p in args["tokenizers"].split(",")]
one = lambda s: all(len(t(s, add_special_tokens=False)["input_ids"]) == 1 for t in TOKS)  # noqa: E731
CAP, MIN = 60, 1
ANSWER = {"Gender": {"male": "his", "female": "her"}}                  # the word a prompt ending in "for" elicits


def form(p):
    tail = p.split("%s")[1]
    if tail[-1:].isalpha() or tail[-1:] in (":", ","):
        return "spaced"
    if tail[-1:] == '"':
        return "json"
    return None


tasks, dropped, left_out = {}, {}, {}
for et in ("city", "nobel_prize_winner", "occupation", "physical_object", "verb"):
    A = json.load(open(SRC / f"ravel_{et}_entity_attributes.json")); S = json.load(open(SRC / f"ravel_{et}_entity_to_split.json"))
    P = json.load(open(SRC / f"ravel_{et}_attribute_to_prompts.json")); PS = json.load(open(SRC / f"ravel_{et}_prompt_to_split.json"))
    for attr, prompts in P.items():
        base = f"ravel_{et}_{attr.lower().replace(' ', '_')}"
        groups = {"spaced": [], "json": [], "spaced_for": []}
        for p in sorted(prompts):
            f = form(p) if p.count("%s") == 1 else None
            if f == "spaced" and attr in ANSWER and p.rstrip().endswith("for"):
                f = "spaced_for"                                       # "...%s: for his contributions"
            if f is None:
                left_out.setdefault(base, []).append(p)
            else:
                groups[f].append(p)
        freq = Counter(a[attr] for a in A.values() if a.get(attr))
        for f, ps in groups.items():
            if f == "spaced_for" and not ps:
                continue
            name = {"spaced": base if attr not in ANSWER else f"{base}_value", "json": f"{base}_json", "spaced_for": base}[f]
            lab = {"spaced": lambda v: " " + v, "json": lambda v: v, "spaced_for": lambda v: " " + ANSWER[attr][v]}[f]
            vals = [v for v, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0])) if one(lab(v))][:2]
            why = [] if ps else ["no prompt of this form"]
            if len(vals) < 2:
                why.append("fewer than two one-token values")
            ent = {}
            if len(vals) == 2 and ps:
                for split, which in (("train", ("train", "val")), ("test", ("test",))):
                    ent[split] = {}
                    for v in vals:
                        pool = sorted(e for e, a in A.items() if a.get(attr) == v and S.get(e) in which)
                        ent[split][v] = sorted(random.Random(94).sample(pool, min(CAP, len(pool))))
                        if len(pool) < MIN:
                            why.append(f"{split} {v}: {len(pool)} entities")
            if why:
                dropped[name] = why; continue
            tasks[name] = dict(entity_type=et, attribute=attr, form=f, values=vals, labels=[lab(v) for v in vals],
                               prompts={p: PS.get(p) for p in ps}, entities=ent)
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(source="RAVEL e421fc9e", rule="the construction rules: experiments/tasks/make_ravel_attributes.py", tasks=tasks, dropped=dropped, left_out_prompts=left_out),
          open(OUT / "candidates.json", "w"), indent=0)
for k, v in tasks.items():
    print(k, v["values"], v["labels"], "prompts", len(v["prompts"]), {s: {x: len(y) for x, y in d.items()} for s, d in v["entities"].items()})
print("dropped:", json.dumps(dropped))
print("prompts left out:", {k: len(v) for k, v in left_out.items()})

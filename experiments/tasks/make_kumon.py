"""Kumon and Yanaka's constructions, built by their own loader's rules, per model (labels are read by their first token).

Source: kumoryo9/shared-mech (Hugging Face, MIT licence) snapshot d57aeab; their code github.com/ynklab/shared_syntactic_mechanism
at 0a12355, data/load_data.py and activation.py (read, not run). Their rules, followed here:
  mirroring   collect_dataset stores every row and the same row with base and source exchanged (labels, types and
              alignments swap with them);
  test files  for test_id.json (in distribution) and test.json (out of distribution) only the first half of the rows is
              taken (`if half and i >= len(ds_dict) // 2: break`), then mirrored; training rows are all taken, then mirrored;
  labels      a label of several tokens is read by its first token (`tokenizer(label, ...)['input_ids'][0]`); here each label
              is replaced by the string of its first token for the given tokenizer, and a row is dropped (and counted) only
              if that string does not encode back to the same single token, or if base and source labels then coincide;
  NPI         their NPI runs keep only pairs whose base type is in LABEL1 ("ever") and whose source type is in LABEL2
              ("never") (`--src_base_pattern label2`, activation.py); filler-gap runs keep both directions ("default");
  OOD labels  their DAS is evaluated on test.json with the original labels and with the *_label_ood labels
              (`--label_methods None label-ood`); both are written.
Excluded by rule: constructions without a training split (CtrlCapital) and with fewer than 20 test rows (FGNaturalEmbed).
Built beside theirs and labelled as ours: <construction>_mirrored for NPI, both directions (the method's mirrored set);
<construction>_disjoint, the in-distribution test rows whose base and source strings both occur in no training row (their
split shares inputs between training and test in some NPI constructions, e.g. 38 of NPISimple's 50 test rows).

Output per model, in CausalGym's format (train.json, test.json): experiments/data/<model tag>/kumon/<construction>/ (in-distribution
test), <construction>_ood/ (out-of-distribution test, original labels), <construction>_ood_label/ (its *_label_ood labels),
<construction>_disjoint/, and for NPI <construction>_mirrored/; a summary kumon/build.json with the counts, the snapshot and
a pointer to this docstring for the rules. experiments/run/data.sh runs it for every model.
Usage: python -I experiments/tasks/make_kumon.py <snapshot dir> --models=<hf id>,... [--tokenizer_dirs=<dir>,...]
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SNAP = Path(sys.argv[1]); REPO = Path(__file__).resolve().parents[2]
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:] if "=" in a)
MODELS = args["models"].split(",")
DIRS = args["tokenizer_dirs"].split(",") if "tokenizer_dirs" in args else MODELS
LABEL1, LABEL2 = ("that", "ever"), ("wh", "never")          # as in their activation.py


def mirror(r):
    m = dict(r)
    for a, b in (("base", "src"), ("base_label", "src_label"), ("base_label_ood", "src_label_ood"), ("base_type", "src_type")):
        m[a], m[b] = r[b], r[a]
    return m


def collect(rows, half):
    out = []
    for i, r in enumerate(rows):
        if half and i >= len(rows) // 2:
            break
        out += [r, mirror(r)]
    return out


def first_token(tok, s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    if not t:
        return None
    w = tok.decode([t[0]])
    return w if tok(w, add_special_tokens=False)["input_ids"] == [t[0]] else None


for name, tdir in zip(MODELS, DIRS):
    tok = AutoTokenizer.from_pretrained(tdir)
    out = REPO / "experiments/data" / name.rstrip("/").split("/")[-1] / "kumon"
    summary = {}
    for c in sorted(p.name for p in SNAP.iterdir() if p.is_dir()):
        files = {f: SNAP / c / f"{f}.json" for f in ("train", "test_id", "test")}
        if not files["train"].is_file():
            summary[c] = "excluded: no training split"; continue
        raw = {f: json.load(open(p)) for f, p in files.items()}
        if len(raw["test_id"]) < 20:
            summary[c] = f"excluded: {len(raw['test_id'])} test rows"; continue
        npi = c.startswith("NPI")
        sets = {"train": collect(raw["train"], False), "id": collect(raw["test_id"], True), "ood": collect(raw["test"], True)}

        def labelled(rows, ood_labels=False, pattern=npi):
            keep, drop = [], 0
            for r in rows:
                if pattern and not (r["base_type"] in LABEL1 and r["src_type"] in LABEL2):
                    continue
                bl, sl = (r["base_label_ood"], r["src_label_ood"]) if ood_labels else (r["base_label"], r["src_label"])
                b, s = first_token(tok, bl), first_token(tok, sl)
                if b is None or s is None or b == s:
                    drop += 1; continue
                keep.append(dict(base=r["base"], src=r["src"], base_label=b, src_label=s, base_type=r["base_type"],
                                 src_type=r["src_type"], task=c, mapping=r.get("mapping")))
            return keep, drop

        tr, d0 = labelled(sets["train"])
        folders = {c: (tr, labelled(sets["id"])), f"{c}_ood": (tr, labelled(sets["ood"])),
                    f"{c}_ood_label": (tr, labelled(sets["ood"], ood_labels=True))}
        seen = {"".join(r[k]) for r in tr for k in ("base", "src")}
        te, d1 = labelled(sets["id"])
        folders[f"{c}_disjoint"] = (tr, ([r for r in te if "".join(r["base"]) not in seen and "".join(r["src"]) not in seen], d1))
        if npi:
            trm, _ = labelled(sets["train"], pattern=False)
            folders[f"{c}_mirrored"] = (trm, labelled(sets["id"], pattern=False))
        for v, (train_rows, (test_rows, d1)) in folders.items():
            d = out / v; d.mkdir(parents=True, exist_ok=True)
            json.dump([dict(r, task=v) for r in train_rows], open(d / "train.json", "w"), indent=0)
            json.dump([dict(r, task=v) for r in test_rows], open(d / "test.json", "w"), indent=0)
            summary[v] = dict(train=len(train_rows), test=len(test_rows), dropped_train=d0, dropped_test=d1)
    json.dump(dict(snapshot="d57aeab6a73778023e65a9cfa7cf1eb77709ca62", model=name, rules="the construction rules: experiments/tasks/make_kumon.py",
                   tasks=summary), open(out / "build.json", "w"), indent=1)
    print(name, {k: v for k, v in summary.items() if not k.endswith(("_ood", "_ood_label"))})

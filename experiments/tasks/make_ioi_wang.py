"""IOI pairs from the templates of Wang et al. (2023), in CausalGym's format (the paper's "IOI, Wang et al.").

Source: redwoodresearch/Easy-Transformer (MIT licence), commit ea15315dd24481e9e2ac5c3ef335d82907a1dc34,
easy_transformer/ioi_dataset.py (sha256 94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231), read as text,
not imported: NAMES, PLACES, OBJECTS and the 15 BABA_TEMPLATES. The 15 ABBA templates are derived by their rule (in the first
clause, the first [B] becomes [A] and the first [A] becomes [B]). In every template the subject S is [B] and the indirect
object IO is [A], and the sentence ends with "... to [A]"; the prompt stops before that last [A].
Base: the template with S repeated (label IO); source: the same frame with the repeat replaced by IO (label S); every pair is
also stored with base and source exchanged (a mirrored set), as experiments/tasks/make_ioi_pairs.py does.
Spans: [text before the first name, " N1", text between, " N2", text up to the repeat, " repeat", tail].
  ioi_wang_1    IO and S in {John, Mary} (ioi_fixed's names; both are in Wang et al.'s list): one label pair
  ioi_wang_20   IO and S from the first 20 of their NAMES that are one token (with a leading space) for every tokenizer given
Labels must be single tokens, hence the one-token names. Frames are template x place x object; 200 training and 50 test pairs
with seed 94, each with its mirror; no test base sentence is a training base sentence. With two names (ioi_wang_1) a test pair
can be a training frame with the names swapped.
--ood=1 writes ioi_wang_{1,20}_ood: training frames from BABA templates 1-10 and their ABBA forms, test frames from
templates 11-15 and theirs (held-out templates).
Start of text: Wang et al. feed the prompt without one (ioi_dataset.py: "no end of texts, GPT-2 small wasn't trained this way");
that is the default here. --start=1 writes the same tasks with CausalGym's start-of-text token as <task>_sot (ours).
Usage: python -I experiments/tasks/make_ioi_wang.py <ioi_dataset.py> <out dir> --tokenizers=<dir>,<dir>,... [--ood=1] [--start=1]
(the release's files: <out dir> = experiments/tasks/ioi_wang)
"""
import ast
import json
import random
import re
import sys
from pathlib import Path

src = open(sys.argv[1]).read(); OUT = Path(sys.argv[2])
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[3:] if "=" in a)
START = args.get("start") == "1"


def lit(name):
    return ast.literal_eval(re.search(rf"^{name} = (\[.*?^\])", src, re.S | re.M).group(1))      # up to the "]" that opens a line


NAMES, PLACES, OBJECTS, BABA = lit("NAMES"), lit("PLACES"), lit("OBJECTS"), lit("BABA_TEMPLATES")
assert len(BABA) == 15, len(BABA)


def abba(t):                                                          # their rule, on the first clause only
    i, j = t.index("[B]"), t.index("[A]")
    return t[:i] + "[A]" + t[i + 3:j] + "[B]" + t[j + 3:]


TEMPL = [(k, t) for k, t in enumerate(BABA)] + [(k, abba(t)) for k, t in enumerate(BABA)]
for _, t in TEMPL:
    assert t.endswith(" to [A]") and t.count("[A]") == 2 and t.count("[B]") == 2, t


def spans(t, io, s, repeat, place, obj):
    t = t[: -len("[A]")].rstrip(" ")                                    # stop before the answer
    t = t.replace("[PLACE]", place).replace("[OBJECT]", obj)
    parts = re.split(r"(\[A\]|\[B\])", t)                             # text, name, text, name, text, name, tail
    names = [p for p in parts if p in ("[A]", "[B]")]
    assert len(names) == 3, t
    out, k = [], 0
    for p in parts:
        if p in ("[A]", "[B]"):
            nm = {"[A]": io, "[B]": s}[p] if k < 2 else repeat
            out[-1] = out[-1].rstrip(" "); out.append(" " + nm); k += 1
        else:
            out.append(p)
    if START:
        out[0] = "<|endoftext|>" + out[0]
    return [x for x in out if x != ""]


def frames(templ):
    seen, fr = set(), []
    for _, t in templ:
        for p in PLACES:
            for o in OBJECTS:
                key = t.replace("[PLACE]", p).replace("[OBJECT]", o)
                if key not in seen:
                    seen.add(key); fr.append((t, p, o))
    return fr


def make(task, names, fr_train, fr_test):
    rng = random.Random(94)

    def draw(fr, n, exclude):
        rows, seen = [], set()
        while len(rows) < 2 * n:
            t, p, o = rng.choice(fr); io, s = rng.sample(names, 2)
            b = spans(t, io, s, s, p, o); x = spans(t, io, s, io, p, o)
            key = "".join(b)
            if key in exclude or key in seen:
                continue
            seen.add(key)
            rows.append(dict(base=b, src=x, base_type="ABB", src_type="ABA", base_label=" " + io, src_label=" " + s, task=task))
            rows.append(dict(base=x, src=b, base_type="ABA", src_type="ABB", base_label=" " + s, src_label=" " + io, task=task))
        return rows
    train = draw(fr_train, 200, frozenset())
    test = draw(fr_test, 50, frozenset("".join(r["base"]) for r in train))
    d = OUT / task; d.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(d / "train.json", "w"), indent=0); json.dump(test, open(d / "test.json", "w"), indent=0)
    print(task, "frames", len(fr_train), len(fr_test), "names", len(names), "rows", len(train), len(test))


from transformers import AutoTokenizer  # noqa: E402

TOKS = [AutoTokenizer.from_pretrained(p) for p in args["tokenizers"].split(",")]
one = [n for n in NAMES if all(len(t(" " + n, add_special_tokens=False)["input_ids"]) == 1 for t in TOKS)]
assert "John" in one and "Mary" in one
SETS = {"ioi_wang_1": ["John", "Mary"], "ioi_wang_20": one[:20]}
SUFFIX = "_sot" if START else ""
if args.get("ood") == "1":
    tr = [(k, t) for k, t in TEMPL if k < 10]; te = [(k, t) for k, t in TEMPL if k >= 10]
    for task, names in SETS.items():
        make(task + "_ood" + SUFFIX, names, frames(tr), frames(te))
else:
    for task, names in SETS.items():
        fr = frames(TEMPL); make(task + SUFFIX, names, fr, fr)
print("names of ioi_wang_20:", SETS["ioi_wang_20"])

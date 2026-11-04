"""IOI minimal pairs in our own sentence frame, in CausalGym's format (the paper's "IOI, ours").

The frame and the lists of names, places and objects are ours, not Wang et al.'s (make_ioi_wang.py builds theirs).
Base and source differ only in which name is repeated: base "When A and B went to the store, B gave a drink to" (label A),
source "... , A gave a drink to" (label B). Every pair is also stored with base and source exchanged (a mirrored set).
Two datasets:
  ioi_fixed   A, B in {Mary, John} only: one label pair, so the readout gradients share one direction
  ioi_varied  A, B drawn from 20 names: 380 ordered label pairs, readout directions spread over many names
The names are not checked against a tokenizer here; the readout drops any pair whose labels are not single tokens.
Spans: ["<|endoftext|>" + opener, " A", " and", " B", middle, " C", tail]; 400 frames (4 openers x 10 places x
10 objects). 200 training pairs (seed 6801) and 50 test pairs (seed 6802), each with its mirror; no test base sentence is a
training base sentence.
Usage: python -I experiments/tasks/make_ioi_pairs.py <out dir>      (the release's files: <out dir> = experiments/tasks/ioi)
"""
import json
import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OPENERS = ["When", "After", "While", "Then, when"]
PLACES = ["store", "park", "market", "office", "library", "cafe", "school", "beach", "garden", "museum"]
OBJECTS = ["drink", "ball", "bag", "pen", "book", "cup", "gift", "ticket", "key", "phone"]
TEMPLATES = [(o, f" went to the {p},", f" gave a {x} to") for o in OPENERS for p in PLACES for x in OBJECTS]  # 400
NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Lisa", "Paul", "Kevin", "Laura", "David", "Sarah",
         "Jack", "Anna", "Mark", "Emma", "Peter", "Kate", "Alex"]


def pair(t, a, b):
    opener, middle, tail = t
    spans = lambda c: ["<|endoftext|>" + opener, " " + a, " and", " " + b, middle, " " + c, tail]  # noqa: E731
    base = dict(base=spans(b), src=spans(a), base_type="ABB", src_type="ABA", base_label=" " + a, src_label=" " + b)
    mirror = dict(base=base["src"], src=base["base"], base_type="ABA", src_type="ABB", base_label=" " + b, src_label=" " + a)
    return [base, mirror]


def make(task, names, n_pairs, seed, exclude=frozenset()):
    rng = random.Random(seed); rows, seen = [], set()
    while len(rows) < 2 * n_pairs:
        t = rng.choice(TEMPLATES); a, b = rng.sample(names, 2)
        rs = pair(t, a, b)
        key = "".join(rs[0]["base"])
        if key in exclude or key in seen:
            continue
        seen.add(key)
        rows += [dict(r, task=task) for r in rs]
    return rows


for task, names in (("ioi_fixed", ["Mary", "John"]), ("ioi_varied", NAMES)):
    d = OUT / task; d.mkdir(parents=True, exist_ok=True)
    train = make(task, names, 200, 6801)
    test = make(task, names, 50, 6802, exclude=frozenset("".join(r["base"]) for r in train))
    json.dump(train, open(d / "train.json", "w"), indent=0)
    json.dump(test, open(d / "test.json", "w"), indent=0)
    print(task, "written")

"""Build RAVEL's city-to-continent task from the cities a model knows, before any map or readout (RAVEL also keeps only the
entities a model knows).

From experiments/tasks/ravel/ravel_continent/cities.json (European and Asian cities, split by RAVEL's entity split), a city is
kept when the model, on "<start>CITY is in the continent of", prefers the city's continent to the other label by at least
--margin log odds (default 1.0; with two labels a bare preference would pass half the cities by chance). Then, with a fixed
seed (82), 200 training pairs are drawn with replacement from the known training cities and 50 test pairs from the known test
cities. Each pair is one European and one Asian city and comes with its mirror (400 and 100 rows, CausalGym's sizes and
format). The regions are the start-of-text marker, the city, " is", " in", " the", " continent" and " of". The task needs at
least 10 known cities per continent and split; otherwise the model is reported as not knowing enough cities and no file is
written.
Writes experiments/data/<model tag>/ravel_continent/{train,test}.json.
Usage: python -I experiments/scripts/filter_known.py --model=<hf id> [--revision=<sha>] [--margin=1.0] [--device=cuda]
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402

_, args, model, tok = common.start("filter_known", sys.argv[1:])
DEV, FIX = common.RUN["dev"], families.start_of_text(tok)
MARGIN = float(args.get("margin", "1.0"))
SRC = json.load(open(common.REPO / "experiments/tasks/ravel/ravel_continent/cities.json"))
OUT = common.REPO / "experiments/data" / common.RUN["tag"] / "ravel_continent"
LAB = {c: tok(" " + c, add_special_tokens=False)["input_ids"] for c in ("Europe", "Asia")}
assert all(len(v) == 1 for v in LAB.values()), LAB


def spans(city):
    return ["<|endoftext|>", city, " is", " in", " the", " continent", " of"]


known, report = {}, {}
with torch.no_grad():
    for split, by in SRC["cities"].items():
        known[split] = {}
        for cont, cities in by.items():
            other = "Asia" if cont == "Europe" else "Europe"
            keep = []
            for c in cities:
                z = model(torch.tensor([families.encode(tok, spans(c), FIX)[0]], device=DEV)).logits[0, -1]
                if float(z[LAB[cont][0]] - z[LAB[other][0]]) >= MARGIN:
                    keep.append(c)
            known[split][cont] = keep
            report[f"{split} {cont}"] = f"{len(keep)} of {len(cities)}"
print(json.dumps(report), flush=True)
enough = all(len(v) >= 10 for d in known.values() for v in d.values())
outs = []
if enough:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(82)
    for split, n in (("train", 200), ("test", 50)):
        rows = []
        for i in range(n):
            e, a = rng.choice(known[split]["Europe"]), rng.choice(known[split]["Asia"])
            for base, src, bt in ((e, a, "Europe"), (a, e, "Asia")):
                rows.append(dict(base=spans(base), src=spans(src), base_label=" " + bt,
                                 src_label=" " + ("Asia" if bt == "Europe" else "Europe"), base_type=bt, task="ravel_continent", pair=i))
        json.dump(rows, open(OUT / f"{split}.json", "w"), indent=0); outs.append(OUT / f"{split}.json")
else:
    print("not enough known cities (need 10 per continent and split); no task written", flush=True)
common.finish(outs, summary=dict(known=report, margin=MARGIN, written=enough))

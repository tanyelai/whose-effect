"""The continent task (ravel_continent) from RAVEL's city entities (Huang et al., 2024; MIT licence).

Source: github.com/explanare/ravel, commit e421fc9e132b3613b1d0020f856d539217fb30ca, data.tgz, sha256
b148f0ded1979cb0e17e301cfda41cb120517a109d3c758cc768e376946d96bc.

One label pair, so every pair asks the same two-label question: cities in Europe against cities in Asia, with RAVEL's prompt
"%s is in the continent of" and the labels " Europe" / " Asia". Only the candidate cities are written here, split by
RAVEL's entity split (training: RAVEL's train and val; test: its test), so training and test cities never meet.
experiments/scripts/filter_known.py keeps, per model, the cities the model places on their continent and builds the pairs from them
(RAVEL also filters entities by what the model knows).
Output: experiments/tasks/ravel/ravel_continent/cities.json.
Usage: python -I experiments/tasks/make_ravel_continent.py <RAVEL's extracted data directory>
"""
import json
import sys
from pathlib import Path

SRC = Path(sys.argv[1]); OUT = Path(__file__).resolve().parent / "ravel/ravel_continent"
attrs = json.load(open(SRC / "ravel_city_entity_attributes.json"))
split = json.load(open(SRC / "ravel_city_entity_to_split.json"))
assert "%s is in the continent of" in json.load(open(SRC / "ravel_city_attribute_to_prompts.json"))["Continent"]
cities = {s: {c: sorted(e for e, a in attrs.items() if a.get("Continent") == c and split.get(e) in which)
              for c in ("Europe", "Asia")} for s, which in (("train", ("train", "val")), ("test", ("test",)))}
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(prompt="%s is in the continent of", source="RAVEL e421fc9e", cities=cities), open(OUT / "cities.json", "w"), indent=0)
print({s: {c: len(v) for c, v in d.items()} for s, d in cities.items()})

"""Compare each model's built task data (experiments/data/<model>/) with the data the paper's runs read (experiments/matrix/data.sha256): per
model, how many files are the same, differ or are missing. Data that differ are still usable (the same tasks under the same
rules); the numbers then come out close to the paper's rather than equal.
Usage: python -I experiments/run/check_data.py [<model> ...]      (default: every model in the list)"""
import hashlib
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
want = {}
for line in (REPO / "experiments/matrix/data.sha256").read_text().splitlines():
    h, p = line.split("  ", 1)
    want[p] = h
tags = [m.rstrip("/").split("/")[-1] for m in sys.argv[1:]] or sorted({p.split("/")[2] for p in want})
for t in tags:
    c, diff = Counter(), []
    for p, h in want.items():
        if p.split("/")[2] != t:
            continue
        f = REPO / p
        k = "missing" if not f.is_file() else "same" if hashlib.sha256(f.read_bytes()).hexdigest() == h else "differ"
        c[k] += 1
        if k == "differ":
            diff.append(p)
    print(f"{t:14s} {c['same']:4d} same  {c['differ']:4d} differ  {c['missing']:4d} missing" + (f"  e.g. {diff[0]}" if diff else ""))

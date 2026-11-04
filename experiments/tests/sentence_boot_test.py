"""Tests experiments/score/score_readout_sentence.py's sentence clusters on synthetic rows. Synthetic data only (no model, no outputs of
the runs); exits on the first failed assert.
The scorer reads the readout's counts with a sentence-level cluster bootstrap, where every two test rows that share a base or a
source input are one cluster (App. C.1, "Intervals when sentences repeat").
1. Every sentence in one row and its mirror only: the sentence clusters are the mirror clusters (same partition), so the
   resamples and every interval equal the mirror bootstrap's.
2. One sentence in every row: one cluster, the task is not defined.
3. A chain (row i shares its source with row i+1's base): one cluster; two disjoint chains: two.
Usage: python -I experiments/tests/sentence_boot_test.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/score"))
from measure import boot_counts, clusters  # noqa: E402
import score_readout_sentence as sb  # noqa: E402


def same_partition(a, b):
    return all(len(set(b[a == c])) == 1 for c in np.unique(a)) and all(len(set(a[b == c])) == 1 for c in np.unique(b))


def fake(rows, rng):
    n = len(rows); T = rng.normal(3.0, 0.5, n)
    Z = {"t/T": T, "t/cluster": clusters([(json.dumps(r["base"]), json.dumps(r["src"])) for r in rows]),
         "t/e/top_train": 2.0 * T + rng.normal(0, 0.2, n), "t/e/sq_sur": T + rng.normal(0, 0.05, n),
         "t/e/b_hat_mid": T + rng.normal(0, 0.05, n), "t/e/u_hat": T + rng.normal(0, 0.1, n)}
    for obj, f in (("ce", 1.8), ("eff", 2.0), ("sq", 1.0)):
        for p in (1, 10):
            for s in (0, 1, 2):
                Z[f"t/e/das_{obj}_p{p}_s{s}"] = f * T + rng.normal(0, 0.1, n)
    return Z


rng = np.random.default_rng(0)
# 1. unique sentences, each pair with its mirror
rows = []
for i in range(30):
    a, b = [f"base sentence {i}"], [f"source sentence {i}"]
    rows += [dict(base=a, src=b), dict(base=b, src=a)]
keys = [(json.dumps(r["base"]), json.dumps(r["src"])) for r in rows]
sc, mc = sb.sentence_clusters(keys), clusters(keys)
assert same_partition(sc, mc), "1: partitions differ"
J = dict(data="d", boot=dict(B=2000, seed=95))
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "d" / "t"; d.mkdir(parents=True); json.dump(rows, open(d / "test.json", "w"))
    Z = fake(rows, rng)
    s = sb.task_stats(J, Z, "t", Path(tmp))
    assert s["defined"] and s["clusters"] == 30, s
    # the same resamples as the mirror bootstrap: both partitions number their clusters in the same order, so the counts agree
    Cm = boot_counts(mc, 2000, 95); Cs = boot_counts(sc, 2000, 95)
    relabel = np.array_equal(np.unique(mc, return_inverse=True)[1], np.unique(sc, return_inverse=True)[1])
    assert relabel and np.array_equal(Cm, Cs), "1: resamples differ"
    print("1 ok: unique sentences give the mirror clusters and the same resamples")
    # 2. one sentence everywhere
    rows2 = [dict(base=["shared"], src=[f"s{i}"]) for i in range(40)]
    d2 = Path(tmp) / "d" / "u"; d2.mkdir(parents=True); json.dump(rows2, open(d2 / "test.json", "w"))
    Z2 = {k.replace("t/", "u/"): v for k, v in fake(rows2, rng).items()}
    s2 = sb.task_stats(J, Z2, "u", Path(tmp))
    assert s2["clusters"] == 1 and not s2["defined"], s2
    print("2 ok: one shared sentence gives one cluster, not defined")
# 3. chains
ch = [(f"x{i}", f"x{i + 1}") for i in range(10)] + [(f"y{i}", f"y{i + 1}") for i in range(10)]
assert len(np.unique(sb.sentence_clusters(ch))) == 2
print("3 ok: two chains give two clusters")
print("all sentence-bootstrap tests passed")

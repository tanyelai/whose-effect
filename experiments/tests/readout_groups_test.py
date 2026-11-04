"""Tests the readout's construction rule (experiments/lib/readout_groups.py) and RAVEL's two-way interval (experiments/lib/ravel_twoway.py), as
experiments/score/score_readout.py and experiments/score/score_readout_rank.py apply them. Synthetic data only (no model, no outputs of the
runs); exits on the first failed assert.
- Constructions: RAVEL's plain, JSON and value forms of an attribute are one construction; distinct attributes stay distinct.
- The two-way interval: RAVEL's pairs share entities across mirror clusters, so its room and its excess are read with a two-way
  cluster interval by the entity of each value (Appendix B.4). On a synthetic RAVEL task whose effect varies by entity, the mirror
  bootstrap's "above" must not survive the two-way reading; the two-way bounds must equal an independent call of ratio_twoway
  on the same data (a scorer that kept the mirror bounds, or used one-way clusters, fails); the mirror bootstrap's readings stay
  beside (mirror_readings); a task whose T is not above zero under the two-way interval leaves the counts; rows that do not
  match the run stop the scorer.
- The same in the rank sweep: at every rank, the room, the excess (excess_k, cross-entropy DAS over T) and the comparison with the
  squared error (objective_k, cross-entropy minus the squared error) equal independent two-way calls; the mirror values stay beside; an
  entity-driven excess does not count; mismatched rows stop the scorer.
Usage: python -I experiments/tests/readout_groups_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/score"))
from ravel_twoway import apply_twoway, apply_twoway_rank  # noqa: E402
from readout_groups import construction  # noqa: E402
from score_readout_tinterval import ratio_twoway, value_entities  # noqa: E402

# constructions
g = "ravel_attributes"
assert construction(g, "ravel_nobel_prize_winner_gender") == construction("ravel_attributes_model_specific", "ravel_nobel_prize_winner_gender_value")
assert construction(g, "ravel_nobel_prize_winner_gender_json") == construction(g, "ravel_nobel_prize_winner_gender")
assert construction(g, "ravel_city_country") != construction(g, "ravel_city_language")
assert construction("causalgym", "cleft") == "causalgym:cleft" and construction("sst", "sst_classification") == "sst"


# the two-way interval: a synthetic RAVEL task, 3 entities per value, every pair with its mirror (60 rows, 30 mirror clusters)
def task(rng, t_mean, t_ent, e_mean, e_ent):
    """Rows and npz entries of one task "x": in both rows of a pair, T = t_mean plus the shift t_ent of the pair's entity of the
    first value, and every DAS start's effect is T times (e_mean plus that entity's shift e_ent); the top at 1.5 T, the squared
    error at half of DAS."""
    pre, post = '"name": "', '", "gender": "'
    A, B = ["a0", "a1", "a2"], ["b0", "b1", "b2"]
    ea_eff = dict(zip(A, e_ent)); ta_eff = dict(zip(A, t_ent))
    rows, T, E = [], [], []
    for i in range(30):
        a, b = A[i % 3], B[(i // 3) % 3]
        for base, src, bt, st in ((a, b, "female", "male"), (b, a, "male", "female")):
            rows.append(dict(base=[pre, base, post], src=[pre, src, post], prompt=pre + "%s" + post, base_type=bt, src_type=st))
            t = t_mean + ta_eff[a] + rng.normal(0, 0.01); T.append(t)
            E.append(t * (e_mean + ea_eff[a] + rng.normal(0, 0.01)))
    T, E = np.array(T), np.array(E); n = len(T)
    Z = {"x/T": T, "x/cluster": np.repeat(np.arange(n // 2), 2), "x/e/top_train": 1.5 * T}
    for obj, scale in (("ce", 1.0), ("eff", 1.0), ("sq", 0.5)):
        for p in (1, 10):
            for s in (0, 1, 2):
                Z[f"x/e/das_{obj}_p{p}_s{s}"] = E * scale
    return rows, Z


def mirror_row(above):
    """A main task's row as score_readout.py scores it under the mirror bootstrap, with the excess (key excess) above 1 or not."""
    v = dict(pt=1.1, lo=1.05 if above else 0.9, hi=1.2, above=above, below=False, p=0.01, seeds_above=3)
    return dict(defined=True, room=True, no_room=False, excess={k: dict(v) for k in ("ce_p1", "ce_p10", "eff_p1", "eff_p10")},
                objective={"p1": dict(ce_above_sq=True, sq_T=1.0), "p10": dict(ce_above_sq=True, sq_T=1.0)})


rng = np.random.default_rng(7)
rows, Z = task(rng, t_mean=2.0, t_ent=[0.0, 0.0, 0.0], e_mean=1.15, e_ent=[0.9, -0.6, -0.3])   # entity-driven effect
s = apply_twoway(mirror_row(above=True), Z, "x", rows)
ea, eb = value_entities(rows)
avg = np.mean([Z[f"x/e/das_ce_p1_s{k}"] for k in (0, 1, 2)], axis=0)
_, lo, hi, _, _, _ = ratio_twoway(avg, Z["x/T"], ea, eb)
assert s["interval"] == "two-way" and s["defined"]
assert abs(s["excess"]["ce_p1"]["lo"] - lo) < 1e-12 and abs(s["excess"]["ce_p1"]["hi"] - hi) < 1e-12, "two-way bounds"
assert lo < 1 < hi and not s["excess"]["ce_p1"]["above"], "an entity-driven excess must not count under the two-way reading"
assert s["mirror_readings"]["excess"]["ce_p1"]["above"] is True, "the mirror reading stays beside"
assert s["excess"]["ce_p1"]["seeds_above"] == 3, "per-start results use the mirror bootstrap"
assert s["twoway"]["G"] == 3

rows2, Z2 = task(rng, t_mean=0.2, t_ent=[2.0, -1.5, -0.5], e_mean=1.1, e_ent=[0.0, 0.0, 0.0])      # T driven by entities
s2 = apply_twoway(mirror_row(above=True), Z2, "x", rows2)
assert s2["defined"] is False and "left out" in s2["interval"], "a task whose two-way T is not above zero leaves the counts"

rows3, Z3 = task(rng, t_mean=2.0, t_ent=[0.0, 0.0, 0.0], e_mean=1.6, e_ent=[0.01, -0.01, 0.0])     # a clear, even excess
s3 = apply_twoway(mirror_row(above=False), Z3, "x", rows3)
assert s3["excess"]["ce_p1"]["above"] and s3["room"], "an even excess counts under the two-way reading"

try:
    apply_twoway(mirror_row(above=True), Z, "x", rows[:-2])
    raise AssertionError("mismatched rows must stop the scorer")
except SystemExit:
    pass


# the two-way interval in the rank sweep: the rank npz of the entity-driven task above, at ranks 1 and 16
def rank_npz(Z):
    Zr = {"x/T": Z["x/T"], "x/cluster": Z["x/cluster"]}
    for k, f in (("1", 1.0), ("16", 1.1)):
        Zr[f"x/k{k}/top_train/e"] = Z["x/e/top_train"] * f
        for obj in ("ce", "sq", "eff"):
            for p in (1, 10):
                for x in (0, 1, 2):
                    Zr[f"x/k{k}/das_{obj}_p{p}_s{x}/e"] = Z[f"x/e/das_{obj}_p{p}_s{x}"] * f
    return Zr


def rank_row():
    m = dict(pt=1.1, lo=1.05, hi=1.2)
    return {"k": {k: dict(room=dict(m, room=True, no_room=False), excess_k={p: dict(m) for p in ("p1", "p10")},
                          objective_k={p: dict(pt=0.3, lo=0.1, hi=0.5) for p in ("p1", "p10")}) for k in ("1", "16")}}


Zr = rank_npz(Z)
r = apply_twoway_rank(rank_row(), Zr, "x", rows)
for k in ("1", "16"):
    T_ = Zr["x/T"]; ce = np.mean([Zr[f"x/k{k}/das_ce_p1_s{x}/e"] for x in (0, 1, 2)], axis=0)
    sq = np.mean([Zr[f"x/k{k}/das_sq_p1_s{x}/e"] for x in (0, 1, 2)], axis=0)
    _, lo2, hi2, _, _, _ = ratio_twoway(ce, T_, ea, eb); _, lo3, _, _, _, _ = ratio_twoway(ce - sq, T_, ea, eb)
    _, lot, _, _, _, _ = ratio_twoway(Zr[f"x/k{k}/top_train/e"], T_, ea, eb)
    rec = r["k"][k]
    assert abs(rec["excess_k"]["p1"]["lo"] - lo2) < 1e-12 and abs(rec["excess_k"]["p1"]["hi"] - hi2) < 1e-12, "rank: two-way excess bounds"
    assert abs(rec["objective_k"]["p1"]["lo"] - lo3) < 1e-12 and abs(rec["room"]["lo"] - lot) < 1e-12, "rank: two-way objective and room bounds"
    assert rec["room"]["room"] == (lot > 1) and rec["mirror"]["excess_k"]["p1"]["lo"] == 1.05, "rank: reading and mirror beside"
assert not r["k"]["1"]["excess_k"]["p1"]["lo"] > 1, "rank: an entity-driven excess must not count"
try:
    apply_twoway_rank(rank_row(), Zr, "x", rows[:-2])
    raise AssertionError("rank: mismatched rows must stop the scorer")
except SystemExit:
    pass
print("readout grouping tests: OK")

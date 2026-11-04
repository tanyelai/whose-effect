"""RAVEL's interval at the readout (Appendix B.4): the two-way cluster interval by the entity of each value (Cameron, Gelbach and
Miller 2011), as experiments/score/score_readout_tinterval.py computes it, for the readings the paper takes from it: T's interval in the
population rule, the room, the excess (key excess) and the comparison with the squared error (objective). The mirror bootstrap's values stay
beside the row, under "mirror_readings". Every count's numerator and denominator are read with the same clustering, so a RAVEL
main task whose T is not above zero under the two-way interval leaves the counts (defined becomes False, with the reason). The
other tests (trace, ceiling, natural, angle) and the per-start results use the mirror bootstrap. A task whose test rows cannot be rebuilt stops
the scorer, so the two-way reading is never skipped silently.
apply_twoway_rank does the same for the rank sweep (experiments/score/score_readout_rank.py): at every rank, the room, excess_k (excess at rank k)
and objective_k (objective at rank k) come from the two-way interval, with the mirror values beside them under "mirror"; ceiling_k (ceiling at rank k), one_question_ratios
and gains use the mirror bootstrap.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/score"))
import numpy as np  # noqa: E402
from score_readout_tinterval import BUDGETS, SEEDS, classify, ratio_twoway, value_entities  # noqa: E402

RAVEL_MAIN = ("ravel_attributes", "ravel_attributes_model_specific")


def apply_twoway(s, Z, task, trows):
    """s: one main task row of score_readout that the population rule keeps (defined); Z: its npz; trows: its test rows as
    the run read them. Changes s in place and returns it."""
    excess = {k: {kk: v[kk] for kk in ("pt", "lo", "hi", "above", "below", "p") if kk in v} for k, v in s.get("excess", {}).items()}
    s["mirror_readings"] = dict(defined=s["defined"], room=s.get("room"), no_room=s.get("no_room"), excess=excess,
                                objective={p: dict(ce_above_sq=v["ce_above_sq"]) for p, v in s.get("objective", {}).items()})
    n = len(Z[f"{task}/cluster"])
    if trows is None or len(trows) != n:
        raise SystemExit(f"{task}: the two-way reading needs the {n} test rows the run read, got {None if trows is None else len(trows)}")
    tw = classify(Z, task, ent=value_entities(trows))
    s["twoway"] = dict(G=tw["G"], G_eff=tw["G_eff"], T=tw["T"], T_lo=tw["T_lo"])
    if not tw["defined"]:
        s["defined"], s["interval"] = False, "two-way: T not above zero, left out of the counts"
        return s
    s["interval"] = "two-way"
    s["room"], s["no_room"] = tw["room"], tw["no_room"]
    for k, v in tw["excess"].items():
        if k in s["excess"]:
            s["excess"][k].update(lo=v["lo"], hi=v["hi"], above=v["above"], below=bool(v["hi"] < 1), p=v["p"])
    for p, v in tw["objective"].items():
        if p in s["objective"]:
            s["objective"][p]["ce_above_sq"] = v["ce_above_sq"]
    return s


def apply_twoway_rank(s, Zr, task, trows):
    """s: one task row of score_readout_rank (task_stats); Zr: its rank npz; trows: its test rows as the run read them."""
    n = len(Zr[f"{task}/cluster"])
    if trows is None or len(trows) != n:
        raise SystemExit(f"{task}: the two-way reading needs the {n} test rows the run read, got {None if trows is None else len(trows)}")
    ea, eb = value_entities(trows)
    T = Zr[f"{task}/T"].astype(float)
    _, Tlo, _, _, G, Gs = ratio_twoway(T, np.ones(n), ea, eb)
    if not (G >= 2 and Tlo is not None and Tlo > 0):
        raise SystemExit(f"{task}: its two-way T is not above zero, yet the readout scorer counted it")
    s["twoway"] = dict(G=G, G_eff=Gs)
    for k, rec in s["k"].items():
        e = lambda name: Zr[f"{task}/k{k}/{name}/e"].astype(float)  # noqa: E731
        avg = lambda obj, p: np.mean([e(f"das_{obj}_p{p}_s{x}") for x in SEEDS], axis=0)  # noqa: E731
        rec["mirror"] = dict(room=dict(rec["room"]), excess_k={p: dict(v) for p, v in rec["excess_k"].items()}, objective_k={p: dict(v) for p, v in rec["objective_k"].items()})
        Rh, lo, hi, _, _, _ = ratio_twoway(e("top_train"), T, ea, eb)
        rec["room"].update(pt=Rh, lo=lo, hi=hi, room=bool(lo > 1), no_room=bool(hi < 1))
        for p in BUDGETS:
            Rh, lo, hi, _, _, _ = ratio_twoway(avg("ce", p), T, ea, eb)
            rec["excess_k"][f"p{p}"].update(pt=Rh, lo=lo, hi=hi)
            Rh, lo, hi, _, _, _ = ratio_twoway(avg("ce", p) - avg("sq", p), T, ea, eb)
            rec["objective_k"][f"p{p}"].update(pt=Rh, lo=lo, hi=hi)
    s["interval"] = "two-way"
    return s

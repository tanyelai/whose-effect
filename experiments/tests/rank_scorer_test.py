"""Tests experiments/score/score_readout_rank.py: the rank sweep's tests, its twin check against the readout, and the checks that stop
the scorer. Synthetic data only (rank sweeps and readout outputs with known answers; no model, no outputs of the runs); exits 1
on a failure.
The rank sweep trains DAS at the readout at ranks 1, 2, 4, 16, 64, q (the number of labels less one) and D-1 (Section 5.4; the
theory is App. A.3). The scorer's tests, with their output keys: a cross-entropy start above the first-order rank-k top K_k,
the ceiling (ceiling_k); cross-entropy DAS above T in every task with room at rank k, the excess (excess_k); cross-entropy above the squared
error there (objective_k); rank-k over rank-one ratios in two-label tasks (one_question_ratios); the gains from rank one in tasks with more than one label
pair (gains); medians per rank (medians). Each violation also carries the readout's trace deviation (trace), and the twin compares the
sweep's rank-one directions with the readout's own (experiments/results/readout/ in a real run).
D = 768, 100 test pairs in 50 mirrored clusters, per-pair effects = value x T_i + noise, nine trained directions (ce, sq, eff
from three starts). Planted tasks: two two-label tasks with room where K_k stays at lambda_1 = 1.3 T, the out-of-sample top at
1.25 T and DAS at 1.2 T at every rank (sq at T); three tasks with six labels (q = 5) whose top grows with the rank at three
speeds (no room at rank one, room at rank four and above; DAS 0.02 T under the out-of-sample top, which is 0.02 T under K_k);
one task outside the main population. Models: small mA, mB; large mC, mD, mE.
Checks:
- A planted ceiling violation (one start, one budget, rank four) is counted and named with its landing, adequacy and trace
  deviation, and fails the ceiling test in its model only.
- The excess and the comparison with the squared error hold where DAS is above T.
- The growing tasks are counted apart (no room at rank one) at rank four, but never at D-1.
- In two-label tasks the top's gain is 0, and the top's, ce's and eff's rank-k over rank-one ratios are 1.
- The medians per rank give e/K_k, its landing and e_sq/T.
- Where the top grows, DAS's gain is counted and ranks with the top's.
- A violation at rank one only leaves the strong form beyond rank one standing.
- A task whose excess holds at rank one and fails at rank 16 is counted across ranks.
- A model whose rank-one directions differ from the readout's by 0.1 T fails the twin, and passes when the same tasks trained
  alone (--stack=0) agree.
- The large tier is not pooled while one of its models fails its instruments (the readout scorer's rule).
- The task outside the population is not scored, and the ranks are named by value, q and D-1.
A second tree: DAS at T where there is room takes the excess away; a population task without output makes the scoring
incomplete; a resample checksum that is not the run's stops the scorer.
Usage: python -I experiments/tests/rank_scorer_test.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from owneffect.stats import boot_counts  # noqa: E402

B, SEED, D, N = 400, 95, 768, 100
OBJ_STARTS = tuple((o, s) for o in ("ce", "sq", "eff") for s in (0, 1, 2))
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def values(kind, k, f):
    """(K_k, top, DAS ce, sq) in units of T for a task kind at rank k."""
    if kind == "oq":
        return 1.3, 1.25, 1.2, 1.0
    if kind == "oq_null":
        return 1.3, 1.25, 1.0, 1.0
    step = {1: 0.0, 2: 0.25, 4: 0.75}.get(k, 1.0)                     # the top grows until q = 5, then stays
    K = 0.8 + f * 0.8 * step
    return K, K - 0.02, K - 0.04, 1.0


def write(d, model, tasks, violate=None, bad_twin=False, bad_sha=False, alone=False, at_T=None, stem_tag=""):
    """One group's rank-sweep output and the readout's own output for a model; tasks: name -> (kind, factor, labels, label
    pairs). With stem_tag only the rank-sweep output is written, under that suffix."""
    J = dict(boot=dict(B=B, seed=SEED), tasks={}); Z = {}
    Jv = dict(tasks={}); Zv = {}
    for task, (kind, f, labels, lp) in tasks.items():
        r_ = np.random.default_rng(abs(hash((task, model))) % 2 ** 32)   # the same pairs each time this task is written
        T = np.repeat(r_.normal(3.0, 0.3, N // 2), 2); cl = np.repeat(np.arange(N // 2), 2)
        C = boot_counts(cl, B, SEED).astype(float); w = C.sum(1); Tb = C @ T / w
        q = labels - 1 if 1 < labels - 1 < D - 1 else None
        ranks = sorted({1, 2, 4, 16, 64, D - 1} | ({q} if q else set()))
        KB, ro = [], {}
        for k in ranks:
            K, top, ce, sq = values(kind, k, f)
            if at_T and (task, k) == at_T:
                ce = 1.0                                                # DAS at T where there is room, at this rank only
            KB.append(K * Tb * (1 + r_.normal(0, 0.002, B)))
            sub, losses = {}, {}
            for o, s in OBJ_STARTS:
                for p in (1, 10):
                    v = {"ce": ce, "sq": sq, "eff": top}[o]
                    if violate and (task, k, o, s, p) == violate:
                        v = 1.5 * K
                    x = v * T + r_.normal(0, 0.01, N)
                    if k == 1:                                          # the readout's direction: 0.1 T off the sweep's if bad_twin
                        Zv[f"{task}/e/das_{o}_p{p}_s{s}"] = x.astype(float) + (0.1 * T if bad_twin else 0)
                        if alone:                                       # trained alone, the direction agrees with the readout's
                            x = Zv[f"{task}/e/das_{o}_p{p}_s{s}"]
                    Z[f"{task}/k{k}/das_{o}_p{p}_s{s}/e"] = x.astype(np.float32); Z[f"{task}/k{k}/das_{o}_p{p}_s{s}/fm"] = (0.99 * x).astype(np.float32)
                    sub[f"das_{o}_p{p}_s{s}"] = dict(overlap_kyfan=0.8, share_v1_train=0.9)
                    losses[f"das_{o}_p{p}_s{s}"] = [2.0 + s, 1.0]
            for name, v in (("kyfan", K), ("kyfan_oos", K - 0.01), ("random", 0.01), ("top_train", top)):
                x = v * T + r_.normal(0, 0.01, N)
                Z[f"{task}/k{k}/{name}/e"] = x.astype(np.float32); Z[f"{task}/k{k}/{name}/fm"] = (0.99 * x).astype(np.float32)
                sub[name] = dict(overlap_kyfan=1.0, share_v1_train=0.9)
                if k == 1:
                    Zv[f"{task}/e/" + {"kyfan": "v1_test_mid", "kyfan_oos": "v1_mid", "top_train": "top_train", "random": "random"}[name]] = x.astype(float)
            ro[str(k)] = dict(K_mid_T=K, subspaces=sub, losses=losses,
                              instruments=dict(kyfan_first_order=1e-9, above_kyfan=0.0, top_start=0.01))
        KB = np.stack(KB, 1)
        Z[f"{task}/T"], Z[f"{task}/g0"], Z[f"{task}/cluster"], Z[f"{task}/kyfan_mid_boot"] = T, np.zeros(N), cl, KB
        Zv[f"{task}/T"], Zv[f"{task}/cluster"], Zv[f"{task}/lam_mid_boot"] = T, cl, KB[:, 0]
        sha = hashlib.sha256(np.ascontiguousarray(C.astype(np.int64)).tobytes()).hexdigest()[:16]
        inst = dict(readout_vs_logits=1e-5, T_vs_forward=1e-5)
        J["tasks"][task] = dict(T=float(T.mean()), D=D, ranks=ranks, rank_q=q, labels=labels, label_pairs=lp, n_test=N, scale=50.0,
                                ranks_out=ro, instruments=dict(train=inst, test=inst, boot_check=1e-9, lam_check=1e-9),
                                boot_counts_sha="0" * 16 if bad_sha else sha)
        Jv["tasks"][task] = dict(T=float(T.mean()), losses={f"das_{o}_p1_s{s}": [2.0 + s, 1.0] for o, s in OBJ_STARTS})
    (d / "readout").mkdir(exist_ok=True)
    targets = [(J, Z, d / f"readout_rank_{model}_m_g1{stem_tag}")]
    if not stem_tag:
        targets.append((Jv, Zv, d / "readout" / f"readout_{model}_m_g1"))
    for JJ, ZZ, stem in targets:
        np.savez_compressed(f"{stem}.npz", **ZZ)
        JJ["npz_sha256"] = hashlib.sha256(Path(f"{stem}.npz").read_bytes()).hexdigest()
        json.dump(JJ, open(f"{stem}.json", "w"))


def rows_for(model, tasks, extra=()):
    out = [dict(model=model, group="g1", task=t, main=(t != "side"), defined=True, logit_cap="capped", one_q=kind.startswith("oq"),
                construction=f"g1:{t}", source="S", trace=dict(dev_mid=[0.004, 0.003, 0.005])) for t, (kind, _, _, _) in tasks.items()]
    return out + [dict(model=model, group="g1", task=t, main=True, defined=True, logit_cap="capped", one_q=True, construction=f"g1:{t}",
                       source="S", trace=dict(dev_mid=[0.0, 0.0, 0.0])) for t in extra]


def run(d):
    p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout_rank.py"), f"--fake={d}"], capture_output=True, text=True)
    f = d / "score_readout_rank.json"
    return p, (json.load(open(f)) if f.is_file() and p.returncode == 0 else None)


TASKS = {"oq_a": ("oq", 0, 2, 1), "oq_b": ("oq", 0, 2, 1), "mq_1": ("mq", 0.5, 6, 10), "mq_2": ("mq", 1.0, 6, 10),
         "mq_3": ("mq", 1.5, 6, 10), "side": ("oq", 0, 2, 1)}
d = Path(tempfile.mkdtemp())
json.dump([["mA", "mB"], ["mC", "mD", "mE"]], open(d / "models.json", "w"))
write(d, "mA", TASKS, violate=("oq_b", 4, "ce", 1, 10))
write(d, "mB", TASKS)
write(d, "mC", TASKS, violate=("oq_a", 1, "ce", 2, 1), at_T=("oq_b", 16))
write(d, "mD", TASKS, bad_twin=True)
write(d, "mE", TASKS, bad_twin=True)
write(d, "mE", TASKS, bad_twin=True, alone=True, stem_tag="_stack0")      # trained alone, these agree with the readout's
json.dump(dict(rows={m: rows_for(m, TASKS) for m in ("mA", "mB", "mC", "mD", "mE")}), open(d / "score_readout.json", "w"))
p, out = run(d)
check("scorer runs and is complete", out is not None and out["complete"], (p.stdout + p.stderr)[-800:])
if out:
    M = out["report"]["models"]
    a, b, c = M["mA"]["main"], M["mB"]["main"], M["mC"]["main"]
    v = a["ceiling_k_violations"]
    check("the planted ceiling violation is counted and named with landing, adequacy and the trace deviation, and takes the ceiling test from its model only",
          len(v) == 1 and v[0]["task"] == "oq_b" and v[0]["rank"] == "4" and v[0]["fit"] == "s1_p10" and abs(v[0]["adequacy"] - 1 / 0.99) < 1e-3
          and abs(v[0]["landing"] - 0.99 * 1.5) < 0.01 and v[0]["trace_dev_mid"] == 0.004 and not a["holds"]["ceiling_k"] and b["holds"]["ceiling_k"], json.dumps(v))
    check("the excess test and the objective test hold where DAS is above T", a["holds"]["excess_k"] and a["holds"]["objective_k"] and b["holds"]["excess_k"] and b["holds"]["objective_k"],
          json.dumps({x: a[x].get("4") for x in ("excess_k", "objective_k")}))
    r4 = b["excess_k"]["4"]["p1"]
    check("room at rank four: two one-question tasks and three growing ones, these counted apart; never at D-1",
          r4["room"] == 5 and r4["above"] == 5 and r4["no_room_at_one"] == 3 and r4["no_room_at_one_above"] == 3
          and b["excess_k"]["1"]["p1"]["room"] == 2 and "no_room_at_one" not in b["excess_k"]["D-1"]["p1"], json.dumps({k: b["excess_k"][k]["p1"] for k in ("1", "4", "D-1")}))
    o = b["one_question_ratios"]["16"]
    check("one-question tasks: the top's gain 0; the top's, ce's and eff's rank-k over rank-one ratios 1",
          abs(o["K_gain_T"]["median"]) < 0.01 and all(abs(o[q]["median"] - 1) < 0.01 for q in ("top_ratio", "ratio_ce_p1", "ratio_eff_p10")),
          json.dumps({q: o[q]["median"] for q in o if q != "K_gain_T"}))
    r6 = b["medians"]["16"]
    check("the medians reports e/K_k (between the planted 1.2/1.3 and the growing tasks' (K - 0.04)/K), its landing (0.99 of it) and e_sq/T",
          1.2 / 1.3 - 0.01 < r6["e_ce_over_K_p1"] < 0.99 and abs(r6["landing_ce"] / r6["e_ce_over_K_p1"] - 0.99) < 0.005
          and abs(r6["e_sq_T_p1"] - 1.0) < 0.01, json.dumps(r6))
    t = out["report"]["tiers"]["small"]["main"]["gains"]["q"]
    check("where the top grows, DAS's gain is counted and ranks with the top's (rank q = labels - 1)", t["top_up"] == 6 and t["p1"]["das_up"] == 6
          and t["p1"]["both_up"] == 6 and t["p1"]["spearman"] > 0.8, json.dumps(t))
    check("a violation at rank one only leaves the strong form beyond rank one standing", not c["holds"]["ceiling_k"] and c["holds_beyond_one"]["ceiling_k"],
          json.dumps({k: c[k] for k in ("holds", "holds_beyond_one")}))
    ac = c["across"]["16"]["p1"]
    check("a task whose the excess test holds at rank one and fails at rank 16 is counted across ranks", ac["excess_k_holds_at_one_fails_at_k"] == 1
          and not c["holds_beyond_one"]["excess_k"], json.dumps(ac))
    check("rank-one directions off by 0.1 T fail the twin; the same tasks trained alone and agreeing restore it",
          M["mD"]["instruments"] == "FAILED" and M["mE"]["instruments"] == "OK" and M["mE"]["worst"]["fits_alone"], json.dumps({k: M[k]["instruments"] for k in M}))
    check("the large tier is not pooled while one of its models is not scored", isinstance(out["report"]["tiers"]["large"], str)
          and isinstance(out["report"]["tiers"]["small"], dict), str(out["report"]["tiers"]["large"])[:80])
    check("the task outside the population is not scored", all(r["task"] != "side" for r in out["rows"]["mA"]) and a["tasks"] == 5)
    check("the ranks are named by value, q and D-1", list(a["ceiling_k"]) == ["1", "2", "4", "16", "64", "q", "D-1"], str(list(a["ceiling_k"])))
shutil.rmtree(d)
d = Path(tempfile.mkdtemp())
json.dump([["nA", "nB"], []], open(d / "models.json", "w"))
NULL = {"oq_a": ("oq_null", 0, 2, 1), "oq_b": ("oq_null", 0, 2, 1)}
for m in ("nA", "nB"):
    write(d, m, NULL)
json.dump(dict(rows={"nA": rows_for("nA", NULL, extra=("ghost",)), "nB": rows_for("nB", NULL)}), open(d / "score_readout.json", "w"))
p, out = run(d)
check("DAS at T where there is room takes the excess test away", out is not None and not out["report"]["models"]["nB"]["main"]["holds"]["excess_k"],
      (p.stdout + p.stderr)[-400:])
check("a population task without output makes the scoring incomplete", out is not None and not out["complete"] and out["missing"] == ["nA/g1/ghost"],
      str(out and out["missing"]))
write(d, "nB", NULL, bad_sha=True)
p, out = run(d)
check("a resample checksum that is not the run's stops the scorer", p.returncode != 0 and "checksum" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-300:])
shutil.rmtree(d)
print("\nRANK SCORER OK" if not fails else "RANK SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

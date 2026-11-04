"""The readout's counts without a minimum number of clusters: a sensitivity analysis beside score_readout.py's percentile
cluster bootstrap and its rule that a task needs at least 20 clusters.

Why: the percentile bootstrap's coverage falls smoothly with the number of clusters G (for a ratio of means, about 93-94% at
G = 50 and 91-93% at G = 20 in a normal-theory approximation), with no step at 20: the 20-cluster rule removes the worst cases
(it binds on a few small sentiment tasks, and at the sentence level on RAVEL tasks with 3-17 clusters) but does not make the
others exact. Those sentence-level clusters are very unequal (the largest holds 20-56% of the rows), where the linearized
variance with Student's t on G - 1 degrees of freedom covers only 76-93% at the observed sizes (experiments/tests/tinterval_scorer_test.py
prints its coverage beside this interval's). The interval used here: the ratio estimator's linearized residuals (Cochran,
Sampling Techniques, 1977, ch. 6) with the bias-reduced cluster variance CR2 (Bell and McCaffrey, Survey Methodology, 2002: each
cluster's squared residual over 1 - h_g, h_g its share of the denominator, its leverage in the ratio's weighted fit) and
Student's t on G* - 1 degrees of freedom, G* = 1 / sum h_g^2 the effective number of clusters (Carter, Schnepel and Steigerwald,
Rev. Econ. Stat., 2017); on the observed unequal sizes and on balanced ones it covers close to 95%
(experiments/tests/tinterval_scorer_test.py simulates both). It exists for every G >= 2; a task with one cluster has no variance
estimate and stays undefined (a fact of the data).

Population: per model of experiments/results/score_readout.json whose instruments held, every row of its main population (main, arm
capped), whether or not the 20-cluster rule defined it. Three clusterings: the mirror clusters saved by the run; the sentence
clusters of the sentence-level reading (experiments/score/score_readout_sentence.py: rows sharing a base or a source input, i.e. one
entity under one prompt in RAVEL); and, for RAVEL's main groups (RAVEL_MAIN), a two-way reading by the entity of each value
(ratio_twoway). The sentence clusters do not join the same entity under another prompt, 40% of RAVEL's entity occurrences, and
a union-find over entities would chain the entities of the two values into one or two effective clusters, hence the two-way
reading. value_entities reads each row's two entities from its texts, after a leading start-of-text marker that the prompt
does not carry (the rows of the ravel_attributes_sot groups, which experiments/lib/ravel_twoway.py reads through it).
The interval, for the ratio R = sum(y) / sum(x) over rows, with cluster sums Y_g, X_g and h_g = |X_g| / sum |X_g|:
se^2 = sum_g (Y_g - R X_g)^2 / (1 - h_g) / (sum_g X_g)^2 and the interval R -/+ t_{G*-1, 0.975} se; the one-sided p of
"above c" is 1 - F_{t, G*-1}((R - c)/se). A task whose rows cannot be rebuilt is counted apart, never as undefined.
Defined: G >= 2 and T's interval (y = T, x = 1) above 0. Room: the out-of-sample top over T above 1 (none: below 1). The excess
(key excess): each objective and budget's mean of the three starts over T, above 1 among the tasks with room, raw and with
Benjamini-Hochberg at q = 0.025 over the model's tasks with room. The objective test (key objective): cross-entropy minus squared error
over T above 0 among the tasks with room. Every count is set beside score_readout.py's (key "registered": score_readout.json's
cluster bootstrap with its 20-cluster rule, for RAVEL with the two-way interval) and, under sentence clusters, beside the
sentence-level bootstrap's (key "bootstrap_sentence", score_readout_sentence.json); every task whose classification differs
is named.
Output: experiments/results/score_readout_tinterval.json. Usage: python -I experiments/score/score_readout_tinterval.py [--fake=<dir>]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/score"))
from measure import bh  # noqa: E402
from score_readout_sentence import sentence_clusters, test_rows  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = Path(A["fake"]) if A.get("fake") else None
RES = FAKE or ROOT / "experiments/results/readout"
SCORE = (FAKE or ROOT / "experiments/results") / "score_readout.json"
SENT = (FAKE or ROOT / "experiments/results") / "score_readout_sentence.json"
DEST = (FAKE or ROOT / "experiments/results") / "score_readout_tinterval.json"
SEEDS, BUDGETS, Q, LEVEL = (0, 1, 2), (1, 10), 0.025, 0.975


def ratio(y, x, cl):
    """The ratio of sums over rows with its CR2 linearized cluster variance and the effective number of clusters:
    (R, lo, hi, se, G, G*); lo = hi = se = None when G < 2 (one cluster: no variance estimate)."""
    ids, inv = np.unique(cl, return_inverse=True); G = len(ids)
    Y, X = np.bincount(inv, weights=y, minlength=G), np.bincount(inv, weights=x, minlength=G)
    Rh = float(Y.sum() / X.sum())
    if G < 2:
        return Rh, None, None, None, G, 1.0
    h = np.abs(X) / np.abs(X).sum()                           # each cluster's share of the denominator: its leverage
    se = float(np.sqrt(np.sum((Y - Rh * X) ** 2 / (1 - h))) / abs(X.sum()))
    Gs = float(1 / np.sum(h ** 2))
    q = float(stats.t.ppf(LEVEL, Gs - 1))
    return Rh, Rh - q * se, Rh + q * se, se, G, Gs


def p_above(Rh, se, Gs, c):
    return float(stats.t.sf((Rh - c) / se, Gs - 1)) if se and se > 0 else (0.0 if Rh > c else 1.0)


RAVEL_MAIN = ("ravel_attributes", "ravel_attributes_model_specific")       # RAVEL's main groups (raw prompts: the entity fills the prompt's %s)


def value_entities(rows):
    """Per row, the entity carrying each of the task's two values, ordered by value: every RAVEL row has one of each (base and
    source carry different values; a row and its mirror share both). The entity is the text between the prompt's parts before
    and after %s; a leading start-of-text marker that the prompt does not carry is skipped first."""
    def ent(r, side):
        text = "".join(r[side]); pre, post = r["prompt"].split("%s")
        if text.startswith("<|endoftext|>") and not pre.startswith("<|endoftext|>"):
            text = text[len("<|endoftext|>"):]              # the start-of-text rows open with the marker; the prompt does not
        assert text.startswith(pre) and text.endswith(post), (text, r["prompt"])
        return text[len(pre):len(text) - len(post)]
    out = []
    for r in rows:
        (va, ea), (vb, eb) = sorted([(r["base_type"], ent(r, "base")), (r["src_type"], ent(r, "src"))])
        assert va != vb, r
        out.append((ea, eb))
    return [a for a, _ in out], [b for _, b in out]


def ratio_twoway(y, x, ea, eb):
    """The ratio of sums with a two-way cluster variance by the entity of each value (Cameron, Gelbach and Miller, J. Bus. Econ.
    Stat., 2011): V = V_A + V_B - V_AB, each term CR2 (as ratio()), never below max(V_A, V_B) (their fix for a negative two-way
    variance, conservative); Student t on min(G*_A, G*_B) - 1 degrees of freedom. Returns ratio()'s tuple with G = min(G_A, G_B)
    and G* = min(G*_A, G*_B)."""
    Rh = float(y.sum() / x.sum()); res = y - Rh * x

    def term(keys):
        ids, inv = np.unique(keys, return_inverse=True)
        E, X = np.bincount(inv, weights=res, minlength=len(ids)), np.bincount(inv, weights=x, minlength=len(ids))
        h = np.abs(X) / np.abs(X).sum()
        return (float(np.sum(E ** 2 / (1 - h))) if len(ids) > 1 else None), len(ids), float(1 / np.sum(h ** 2))
    (vA, gA, sA), (vB, gB, sB) = term(np.array(ea)), term(np.array(eb))
    vAB = term(np.array([f"{a}\x00{b}" for a, b in zip(ea, eb)]))[0]
    G, Gs = min(gA, gB), min(sA, sB)
    if vA is None or vB is None:
        return Rh, None, None, None, G, Gs
    se = float(np.sqrt(max(vA + vB - (vAB or 0.0), vA, vB)) / abs(x.sum()))
    q = float(stats.t.ppf(LEVEL, Gs - 1))
    return Rh, Rh - q * se, Rh + q * se, se, G, Gs


def classify(Z, task, cl=None, ent=None):
    """The readings of one task under one clustering: cluster ids cl, or (ent) the two value entities of a RAVEL task's rows."""
    rf = (lambda y, x: ratio(y, x, cl)) if ent is None else (lambda y, x: ratio_twoway(y, x, *ent))  # noqa: E731
    k = lambda name: Z[f"{task}/{name}"]  # noqa: E731
    T = k("T").astype(float); n = len(T); ones = np.ones(n)
    avg = lambda obj, p: np.mean([k(f"e/das_{obj}_p{p}_s{s}") for s in SEEDS], axis=0).astype(float)  # noqa: E731
    Tm, Tlo, _, _, G, Gs = rf(T, ones)
    s = dict(G=G, G_eff=Gs, T=Tm, T_lo=Tlo, defined=bool(G >= 2 and Tlo is not None and Tlo > 0))
    if not s["defined"]:
        return s
    top = rf(k("e/top_train").astype(float), T)
    s["room"], s["no_room"] = bool(top[1] > 1), bool(top[2] < 1)
    s["excess"] = {}
    for obj in ("ce", "eff"):
        for p in BUDGETS:
            Rh, lo, hi, se, _, gs = rf(avg(obj, p), T)
            s["excess"][f"{obj}_p{p}"] = dict(pt=Rh, lo=lo, hi=hi, above=bool(lo > 1), p=p_above(Rh, se, gs, 1.0))
    s["objective"] = {}
    for p in BUDGETS:
        Rh, lo, hi, _, _, _ = rf(avg("ce", p) - avg("sq", p), T)
        s["objective"][f"p{p}"] = dict(pt=Rh, lo=lo, ce_above_sq=bool(lo > 0))
    return s


def main():
    S = json.load(open(SCORE)); SS = json.load(open(SENT))["models"] if SENT.is_file() else {}
    data_root = FAKE or ROOT
    out = dict(scheme="ratio estimator, CR2 linearized cluster variance, Student t on G* - 1 degrees of freedom (G* = 1 / sum h_g^2); "
                      "every task with G >= 2; mirror and sentence clusters; for RAVEL, two-way by each value's entity", models={})
    for m in [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]:
        rows = [r for r in S["rows"][m] if r.get("main") and r.get("logit_cap", "capped") == "capped"]
        sent_of = {(t["group"], t["task"]): t for t in SS.get(m, {}).get("tasks", [])}
        cache, per = {}, []
        for r in rows:
            stem = f"readout_{m}_m_{r['group']}"
            if stem not in cache:
                J = json.load(open(RES / f"{stem}.json")); fz = RES / f"{stem}.npz"
                if J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
                    raise SystemExit(f"{fz.name}: not the npz its json names")
                cache[stem] = (J, np.load(fz))
            J, Z = cache[stem]
            if f"{r['task']}/T" not in Z.files:
                continue                                              # a task the run skipped (fewer than 20 rows): no data
            mirror = Z[f"{r['task']}/cluster"]
            rec = dict(group=r["group"], task=r["task"], source=r.get("source"), bootstrap=dict(
                defined=r["defined"], room=r.get("room"), excess={k_: r["excess"][k_]["above"] for k_ in r.get("excess", {})},
                objective={p: r["objective"][p]["ce_above_sq"] for p in r.get("objective", {})}))
            rec["mirror"] = classify(Z, r["task"], mirror)
            try:
                trows = test_rows(J["data"], r["task"], data_root)
            except Exception as e:                                    # noqa: BLE001
                trows = None; rec["sentence_skipped"] = f"rows not rebuilt: {e}"
            if trows is not None and len(trows) == len(mirror):
                keys = [(json.dumps(x["base"]), json.dumps(x["src"])) for x in trows]
                rec["sentence"] = classify(Z, r["task"], sentence_clusters(keys))
                if r["group"] in RAVEL_MAIN:                          # RAVEL: the two-way reading by each value's entity
                    rec["twoway"] = classify(Z, r["task"], ent=value_entities(trows))
            elif trows is not None:
                rec["sentence_skipped"] = f"the run kept {len(mirror)} of {len(trows)} test rows"
            sr = sent_of.get((r["group"], r["task"]))
            rec["bootstrap_sentence"] = None if sr is None else dict(
                defined=sr["defined"], room=sr.get("room"), excess={k_: v_["above"] for k_, v_ in sr.get("excess", {}).items()},
                objective={p: v_["ce_above_sq"] for p, v_ in sr.get("objective", {}).items()})
            per.append(rec)
        summ = {}
        for scheme in ("mirror", "sentence", "twoway"):
            # the population of each reading: every task (mirror); every task whose rows were rebuilt (sentence; the others are
            # counted apart, never as undefined); RAVEL's main tasks with rebuilt rows (twoway). The sentence reading is set
            # against score_readout_sentence.json, the other two against score_readout.json
            pop = {"mirror": per, "sentence": [x for x in per if "sentence_skipped" not in x],
                   "twoway": [x for x in per if "twoway" in x]}[scheme]
            D = [x for x in pop if x[scheme]["defined"]]
            room = [x for x in D if x[scheme]["room"]]
            boot = (lambda x: x["bootstrap_sentence"]) if scheme == "sentence" else (lambda x: x["bootstrap"])
            sm = dict(tasks=len(pop), defined=len(D), room=len(room), one_cluster=sum(x[scheme]["G"] == 1 for x in pop),
                      skipped=[(x["group"], x["task"], x["sentence_skipped"]) for x in per if "sentence_skipped" in x] if scheme != "mirror" else [],
                      median_G_eff=float(np.median([x[scheme]["G_eff"] for x in pop])) if pop else None,
                      bootstrap_defined=sum(bool(boot(x) and boot(x)["defined"]) for x in pop), excess={}, objective={})
            per_s = pop
            for key in ("ce_p1", "ce_p10", "eff_p1", "eff_p10"):
                ab = [x for x in room if x[scheme]["excess"][key]["above"]]
                sm["excess"][key] = dict(above_in_room=len(ab), room=len(room),
                                     above_bh=int(bh([x[scheme]["excess"][key]["p"] for x in room], Q).sum()) if room else 0,
                                     bootstrap_above=sum(bool(boot(x) and boot(x)["defined"] and boot(x)["room"] and boot(x)["excess"].get(key)) for x in per_s),
                                     above_here_only=[(x["group"], x["task"]) for x in ab if not (boot(x) and boot(x)["defined"] and boot(x)["room"] and boot(x)["excess"].get(key))],
                                     above_in_bootstrap_only=[(x["group"], x["task"]) for x in per_s if boot(x) and boot(x)["defined"] and boot(x)["room"]
                                                               and boot(x)["excess"].get(key) and x not in ab])
            for p in ("p1", "p10"):
                ab = [x for x in room if x[scheme]["objective"][p]["ce_above_sq"]]
                sm["objective"][p] = dict(ce_above_sq=len(ab), room=len(room),
                                   bootstrap=sum(bool(boot(x) and boot(x)["defined"] and boot(x)["room"] and boot(x)["objective"].get(p)) for x in per_s),
                                   changed=sorted({(x["group"], x["task"]) for x in ab} ^ {(x["group"], x["task"]) for x in per_s if boot(x) and boot(x)["defined"]
                                                                                        and boot(x)["room"] and boot(x)["objective"].get(p)}))
            summ[scheme] = sm
        out["models"][m] = dict(summary=summ, tasks=per)
        for scheme in ("mirror", "sentence", "twoway"):
            v = summ[scheme]["excess"]["ce_p1"]
            print(f"{m} [{scheme}]: defined {summ[scheme]['defined']}/{summ[scheme]['tasks']} (bootstrap {summ[scheme]['bootstrap_defined']}, one cluster "
                  f"{summ[scheme]['one_cluster']}); ce above T {v['above_in_room']}/{v['room']} (bootstrap {v['bootstrap_above']}); now above, "
                  f"above here only: {len(v['above_here_only'])}; above in the bootstrap only: {len(v['above_in_bootstrap_only'])}; "
                  f"objective changed {len(summ[scheme]['objective']['p1']['changed'])}")
    json.dump(out, open(DEST, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"written {DEST}")


if __name__ == "__main__":
    main()

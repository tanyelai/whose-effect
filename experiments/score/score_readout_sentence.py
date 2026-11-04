"""The readout under a sentence-level cluster bootstrap, a sensitivity analysis beside score_readout.py's intervals (Appendix C.1,
"Intervals when sentences repeat").

Per task of the main population that score_readout.py defined: the test rows are rebuilt from the run's data folder, their
mirror clusters recomputed (measure.clusters) and checked against the saved ones; sentence clusters join every two rows that share
a base or a source input (whole text, either role). With the same resampling (B, seed, measure.boot_counts) over sentence
clusters: T's interval, the room, the excess (ce and eff, both budgets; interval and BH per model over tasks with room), the
objective test (ce - sq; and where squared error lands against its first-order optimum) and the exact intervals of the natural
directions b_hat_mid and u_hat. Every result is set beside score_readout.py's from experiments/results/score_readout.json (key "mirror":
the mirror-cluster bootstrap, and for RAVEL the two-way interval by each value's entity). Over the tasks defined at the sentence
level, the changes in the excess and in cross-entropy against the squared error are listed both ways: "changed" (either way),
"lost" (above only under score_readout.py's interval) and "gained" (above only under sentence clusters); for where squared error
lands, "sur_changed".
Not recomputed: the ceiling and the angle's lambda_1 target (the run resampled lambda_1 over mirror clusters), the trace, the
angle's other targets, the share of the top and accuracy.
Output: experiments/results/score_readout_sentence.json.
Usage: python -I experiments/score/score_readout_sentence.py [--fake=<dir with readout_*.json/npz and a score_readout.json>]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
from measure import bh, boot_counts, boot_ratio, clusters, interval, p_above  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = Path(A["fake"]) if A.get("fake") else None
RES = FAKE or ROOT / "experiments/results/readout"
SCORE = (FAKE or ROOT / "experiments/results") / "score_readout.json"
DEST = (FAKE or ROOT / "experiments/results") / "score_readout_sentence.json"
SEEDS, BUDGETS, Q, MIN_CLUSTERS = (0, 1, 2), (1, 10), 0.025, 20


def sentence_clusters(keys):
    """Union-find over rows: two rows are joined when they share a base or a source input (either role). Returns int ids (n,)."""
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    first = {}
    for i, (b, s) in enumerate(keys):
        for sent in (b, s):
            if sent in first:
                ri, rj = find(i), find(first[sent])
                if ri != rj:
                    parent[ri] = rj
            else:
                first[sent] = i
    return np.array([find(i) for i in range(len(keys))])


def test_rows(source, task, data_root):
    """The test rows as the run read them (experiments/lib/common.rows): CausalGym's tasks and Kumon & Yanaka's constructions (source fg)
    from the Hugging Face snapshots pinned in experiments/lib/pins.py (registry.CAUSALGYM_REV, common.FG_REV), every other source from
    its folder."""
    if source in ("causalgym", "fg"):
        from huggingface_hub import hf_hub_download
        if source == "causalgym":
            f = hf_hub_download("aryaman/causalgym", "test.json", repo_type="dataset", revision="95349c3a5e53e2506e8b212482ea6dd784978156")
            return [r for r in json.load(open(f)) if r["task"] == task]
        return json.load(open(hf_hub_download("kumoryo9/shared-mech", f"{task}/test_id.json", repo_type="dataset",
                                              revision="d57aeab6a73778023e65a9cfa7cf1eb77709ca62")))
    return json.load(open(data_root / source / task / "test.json"))


def R(num, den, C):
    r = boot_ratio(num, den, C)
    pt = float(np.mean(num) / np.mean(den))
    lo, hi = interval(pt, r)
    return dict(pt=pt, lo=lo, hi=hi, r=r, p=p_above(pt, r))


def task_stats(J, Z, task, data_root):
    """The sentence-level readings of one task, or a dict with 'skipped' and the reason."""
    k = lambda name: Z[f"{task}/{name}"]  # noqa: E731
    T = k("T"); n = len(T)
    rows = test_rows(J["data"], task, data_root)
    if len(rows) != n:
        return dict(skipped=f"the run kept {n} of {len(rows)} test rows (dropped rows need the tokenizer)")
    keys = [(json.dumps(r["base"]), json.dumps(r["src"])) for r in rows]
    if not np.array_equal(clusters(keys), k("cluster")):
        return dict(skipped="rebuilt mirror clusters differ from the saved ones")
    sc = sentence_clusters(keys)
    mirror = k("cluster")
    if any(len(set(sc[mirror == c])) != 1 for c in np.unique(mirror)):
        raise SystemExit(f"{task}: a mirror cluster is split by the sentence clusters")
    C = boot_counts(sc, J["boot"]["B"], J["boot"]["seed"])
    e = lambda d: k(f"e/{d}")  # noqa: E731
    avg = lambda obj, p: np.mean([e(f"das_{obj}_p{p}_s{s}") for s in SEEDS], axis=0)  # noqa: E731
    ones = np.ones(n)
    Tr = R(T, ones, C)
    s = dict(n=n, clusters=int(len(np.unique(sc))), mirror_clusters=int(len(np.unique(mirror))), T_lo=Tr["lo"])
    s["defined"] = s["clusters"] >= MIN_CLUSTERS and Tr["lo"] > 0
    if not s["defined"]:
        return s
    top = R(e("top_train"), T, C)
    s["room"], s["no_room"] = top["lo"] > 1, top["hi"] < 1
    s["excess"] = {}
    for obj in ("ce", "eff"):
        for p in BUDGETS:
            r = R(avg(obj, p), T, C)
            s["excess"][f"{obj}_p{p}"] = dict(pt=r["pt"], lo=r["lo"], hi=r["hi"], above=r["lo"] > 1, below=r["hi"] < 1, p=r["p"])
    s["objective"] = {}
    for p in BUDGETS:
        dce, dsq = avg("ce", p), avg("sq", p)
        r = R(dce - dsq, T, C)
        xs = boot_ratio(dsq, T, C); xu = boot_ratio(e("sq_sur"), T, C)
        dd = np.abs(xs - xu) - np.abs(xs - 1)
        sur = R(e("sq_sur"), T, C)
        s["objective"][f"p{p}"] = dict(ce_above_sq=r["lo"] > 0, sur_differs=sur["lo"] > 1 or sur["hi"] < 1,
                                sur_better=bool(np.quantile(dd, 0.975) < 0), sur_worse=bool(np.quantile(dd, 0.025) > 0))
    s["natural"] = {d: [R(e(d), T, C)[q] for q in ("pt", "lo", "hi")] for d in ("b_hat_mid", "u_hat")}
    return s


def main():
    S = json.load(open(SCORE))
    models = S.get("models_scored") or list(S["rows"])
    data_root = FAKE or ROOT
    out = dict(scheme="sentence clusters (rows sharing a base or a source input), else as score_readout.py", models={})
    for m in models:
        mr = [r for r in S["rows"].get(m, []) if r.get("main") and r.get("defined")]
        cache, per, skipped = {}, [], []
        for r in mr:
            stem = f"readout_{m}_m_{r['group']}" + ("__uncapped" if r.get("logit_cap") == "uncapped" else "")
            if stem not in cache:
                J = json.load(open(RES / f"{stem}.json")); fz = RES / f"{stem}.npz"
                if J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
                    raise SystemExit(f"{fz.name}: not the npz its json names")
                cache[stem] = (J, np.load(fz))
            J, Z = cache[stem]
            s = task_stats(J, Z, r["task"], data_root)
            if "skipped" in s:
                skipped.append((r["group"], r["task"], s["skipped"])); continue
            s.update(group=r["group"], task=r["task"], source=r.get("source"), mirror=dict(
                room=r["room"], excess={key: r["excess"][key]["above"] for key in r["excess"]}, objective={p: r["objective"][p]["ce_above_sq"] for p in r["objective"]},
                objective_landing={p: (r["objective"][p]["sur_better"], r["objective"][p]["sur_worse"]) for p in r["objective"]}))
            per.append(s)
        D = [s for s in per if s["defined"]]
        room = [s for s in D if s["room"]]
        summ = dict(tasks=len(mr), rebuilt=len(per), skipped=skipped, defined=len(D),
                    not_defined=[(s["group"], s["task"], s["clusters"]) for s in per if not s["defined"]],
                    median_clusters=float(np.median([s["clusters"] for s in per])) if per else None,
                    room=len(room), room_mirror=sum(s["mirror"]["room"] for s in per), excess={}, objective={})
        for key in ("ce_p1", "ce_p10", "eff_p1", "eff_p10"):
            both = [s for s in D if s["room"] and s["mirror"]["room"]]
            summ["excess"][key] = dict(above_in_room=sum(s["excess"][key]["above"] for s in room), room=len(room),
                                   above_bh=int(bh([s["excess"][key]["p"] for s in room], Q).sum()) if room else 0,
                                   above_both_clusterings=sum(s["excess"][key]["above"] and s["mirror"]["excess"][key] for s in both),
                                   room_both=len(both),
                                   changed=[(s["group"], s["task"]) for s in D if bool(s["mirror"]["room"] and s["mirror"]["excess"][key])
                                            != bool(s["room"] and s["excess"][key]["above"])],          # either way
                                   lost=[(s["group"], s["task"]) for s in D if s["mirror"]["room"] and s["mirror"]["excess"][key]
                                         and not (s["room"] and s["excess"][key]["above"])],           # above only under score_readout.py's interval
                                   gained=[(s["group"], s["task"]) for s in D if not (s["mirror"]["room"] and s["mirror"]["excess"][key])
                                           and s["room"] and s["excess"][key]["above"]])               # above only under sentence clusters
        for p in ("p1", "p10"):
            summ["objective"][p] = dict(ce_above_sq=sum(s["objective"][p]["ce_above_sq"] for s in room),
                                 changed=[(s["group"], s["task"]) for s in D if bool(s["mirror"]["room"] and s["mirror"]["objective"][p])
                                          != bool(s["room"] and s["objective"][p]["ce_above_sq"])],      # either way
                                 lost=[(s["group"], s["task"]) for s in D if s["mirror"]["room"] and s["mirror"]["objective"][p]
                                       and not (s["room"] and s["objective"][p]["ce_above_sq"])],
                                 gained=[(s["group"], s["task"]) for s in D if not (s["mirror"]["room"] and s["mirror"]["objective"][p])
                                         and s["room"] and s["objective"][p]["ce_above_sq"]],
                                 sur_better=sum(s["objective"][p]["sur_better"] for s in D), sur_worse=sum(s["objective"][p]["sur_worse"] for s in D),
                                 sur_changed=[(s["group"], s["task"]) for s in D if (s["objective"][p]["sur_better"], s["objective"][p]["sur_worse"])
                                              != tuple(s["mirror"]["objective_landing"][p])])
        by_source = {}
        for s in per:
            b = by_source.setdefault(s["source"], [0, 0]); b[0] += 1; b[1] += not s["defined"]
        summ["sources_mostly_undefined"] = [src for src, (tot, nd) in by_source.items() if nd > tot / 2]
        out["models"][m] = dict(summary=summ, tasks=per)
        v = summ["excess"]["ce_p1"]
        print(f"{m}: {len(per)}/{len(mr)} tasks rebuilt, {len(D)} defined at sentence level (median {summ['median_clusters']} clusters); "
              f"room {summ['room']} (mirror {summ['room_mirror']}); ce above T {v['above_in_room']}/{v['room']}, in both clusterings "
              f"{v['above_both_clusterings']}/{v['room_both']}, changed {len(v['changed'])}; objective (ce > sq) changed {len(summ['objective']['p1']['changed'])}; "
              f"squared-error landing changed {len(summ['objective']['p1']['sur_changed'])}; mostly undefined sources {summ['sources_mostly_undefined']}")
    json.dump(out, open(DEST, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"written {DEST}")


if __name__ == "__main__":
    main()

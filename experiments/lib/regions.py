"""String-level facts of an inside map's rows (no tokenizer, no model): what experiments/scripts/inside.py will see in their regions.
Used by experiments/score/inside_set_post.py, experiments/score/score_causalgym_map.py and paper/scripts/numbers_main.py, so that all three
read the same degenerate regions.
"""
from collections import Counter


def facts(rows):
    """rows: dict(train=[...], test=[...]) in CausalGym's format. inside.py takes the map's regions from the first training
    row, drops a row whose base and source counts differ, and encodes each span on its own (owneffect/families.py), giving an
    empty span the previous span's last token. Returns the region count of the first training row, the rows with fewer ("crash")
    or more ("partial") regions, per region the share of kept rows whose base or source span is empty, the degenerate regions
    (regions >= 1 empty in some row: their cell swaps another region's token), the regions where base and source differ (per
    row) and the share of rows where more than one differs."""
    n0 = len(rows["train"][0]["base"])
    every = rows["train"] + rows["test"]
    counts = Counter((len(r["base"]), len(r["src"])) for r in every)
    kept = [r for r in every if len(r["base"]) == len(r["src"])]
    empty = {}
    for g in range(n0):
        eb = sum(len(r["base"]) > g and r["base"][g] == "" for r in kept)
        es = sum(len(r["src"]) > g and r["src"][g] == "" for r in kept)
        if eb or es:
            empty[g] = dict(base=round(eb / len(kept), 3), src=round(es / len(kept), 3))
    differ = Counter(tuple(g for g in range(min(len(r["base"]), n0)) if r["base"][g] != r["src"][g]) for r in kept)
    return dict(n_train=len(rows["train"]), n_test=len(rows["test"]), regions_first_row=n0,
                region_counts={f"{a}/{b}": c for (a, b), c in sorted(counts.items())},
                unequal_dropped=len(every) - len(kept),
                crash_rows=sum(len(r["base"]) < n0 for r in kept), partial_rows=sum(len(r["base"]) > n0 for r in kept),
                empty=empty, degenerate_regions=sorted(g for g in empty if g > 0),
                differing={",".join(map(str, k)) or "none": c for k, c in differ.most_common()},
                multi_region=sum(c for k, c in differ.items() if len(k) > 1) / max(1, len(kept)))

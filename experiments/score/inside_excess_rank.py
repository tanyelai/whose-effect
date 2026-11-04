"""excess_rank's parts, map by map: the first-order excess ranked against each naive predictor before training (Section 6.2).
excess_rank (experiments/score/inside_set_post.py, excess_rank_compare) keeps, per map, only the first-order excess's rank
correlation minus the best of the naive predictors on that map. This keeps every predictor's own correlation, with the same
arithmetic (copied; checked against the post scorer's excess_rank_per_map, which must agree exactly), on the cells the post scorer
scored, below the readout layer, and the median share freezing removes (s_das, over the cells where DAS's interval excludes 0)
at door and other cells. Summaries, each mean over maps with a map bootstrap (B = 4000, seed 0): first order against each
single predictor and against the per-map best; per source; at door cells first order's own correlation there and at all cells,
against the best of nine (excess_rank_compare's door form), the best of the seven, ||Delta|| and ||Delta|| over its layer; Spearman
across maps of the door cells' attention share with first order's door correlation and with the door form's difference.
Writes experiments/results/inside_sets/<set>_excess_rank.post.json. Usage: python -I experiments/score/inside_excess_rank.py --set=<set>
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "experiments/lib"))
from measure import boot_counts  # noqa: E402

A_ = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
SET = A_["set"]
S = json.load(open(R / f"experiments/results/inside_sets/{SET}.json"))
post = json.load(open(R / f"experiments/results/inside_sets/{SET}.post.json"))
keep = set(post["cell_classes"])
NAMES7 = ("G", "-G", "|g|", "|g|-G", "G*-G", "u-G", "depth")


def _row_spearman(A, B_):
    ra, rb = _row_avg_ranks(A), _row_avg_ranks(B_)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def _avg_ranks(v):
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _row_avg_ranks(X):
    X = np.asarray(X, float); n = X.shape[1]; o = np.argsort(X, axis=1, kind="mergesort"); xs = np.take_along_axis(X, o, 1)
    starts = np.ones(xs.shape, bool); starts[:, 1:] = xs[:, 1:] != xs[:, :-1]
    run = np.cumsum(starts, axis=1) - 1; rows = np.arange(X.shape[0])[:, None]
    sums = np.zeros(X.shape); cnts = np.zeros(X.shape)
    np.add.at(sums, (rows, run), np.broadcast_to(np.arange(n, dtype=float), X.shape)); np.add.at(cnts, (rows, run), 1.0)
    r = np.empty(X.shape); np.put_along_axis(r, o, sums[rows, run] / cnts[rows, run], 1)
    return r


def _tied_spearman(r, B_):
    rb = _row_avg_ranks(B_); rb -= rb.mean(1, keepdims=True); ra = r - r.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum(1))
    return np.where(den > 0, (rb @ ra) / np.where(den > 0, den, 1), np.nan)


def readout_layer(J):
    extra_ = {c["layer"] for c in J["cells"]} - set(J.get("layers", []))
    return max(extra_) if J.get("layers") and extra_ else None


def d3(J, Z, cs, door):
    if door:
        cs = [c for c in cs if c.get("door") and f"{c['cell']}/door/delta_norm" in Z.files]
    if len(cs) < 3:
        return None
    C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"]).astype(float); oob = (C == 0).astype(float)
    ok = oob.sum(1) > 0; C, oob = C[ok], oob[ok]; w, wo = C.sum(1)[:, None], oob.sum(1)[:, None]
    ks = [c["cell"] for c in cs]
    e = np.stack([np.mean([Z[f"{k}/e/das_ce_s{s_}_normal"] for s_ in (0, 1, 2)], axis=0) for k in ks], 1)
    get = lambda n: np.stack([Z[f"{k}/{n}"] for k in ks], 1)
    g, gs, tr, uh = get("e/own_normal"), get("gstar_test"), get("tr_mid"), get("e/u_hat_normal")
    lam = np.stack([Z[f"{k}/lam_mid_boot"][ok] for k in ks], 1)
    gb, ab = C @ g / w, C @ np.abs(g) / w
    naive = {"G": gb, "-G": -gb, "|g|": ab, "|g|-G": ab - gb, "G*-G": C @ gs / w - gb, "u-G": C @ uh / w - gb}
    depth = _avg_ranks([c["layer"] for c in cs])
    if door:
        dn = C @ get("door/delta_norm") / w
        lay = np.array([c["layer"] for c in cs]); rel = np.empty_like(dn)
        for L in np.unique(lay):
            j = lay == L; rel[:, j] = dn[:, j] / dn[:, j].mean(1, keepdims=True)
        naive["|D|"] = dn
    xo = oob @ (e - g) / wo
    if np.nanmax(np.abs(xo - xo.mean(1, keepdims=True))) < 1e-12:
        return None
    out = dict(n=len(cs), theory=float(np.nanmean(_row_spearman(lam - C @ tr / w, xo))))
    per = {n: _row_spearman(x, xo) for n, x in naive.items()}; per["depth"] = _tied_spearman(depth, xo)
    if door:
        per["|D|/layer"] = _row_spearman(rel, xo)
    for n, v in per.items():
        out[n] = float(np.nanmean(v))
    out["s_das_median"] = float(np.median([c["attention_share"]["s_das"] for c in cs if c["attention_share"].get("s_das") is not None and c.get("ce_nonzero")] or [np.nan]))
    out["G_T_median"] = float(np.median([c["G_T"] for c in cs]))
    return out


res = {}
for m, cs_all in S["cells"].items():
    for t in sorted({c["task"] for c in cs_all}):
        cs = [c for c in cs_all if c["task"] == t and f"{m}/{t}/{c['cell']}" in keep]
        if not cs:
            continue
        f = R / f"experiments/results/inside/inside_{m}_{t}.json"; J = json.load(open(f)); Z = np.load(f.with_suffix(".npz"))
        rl = readout_layer(J); inner = [c for c in cs if c["layer"] != rl]
        nd = [c for c in inner if not c.get("door")]
        res[f"{m}/{t}"] = dict(all=d3(J, Z, inner, False), door=d3(J, Z, inner, True),
                               nondoor_s_das=float(np.median([c["attention_share"]["s_das"] for c in nd if c["attention_share"].get("s_das") is not None and c.get("ce_nonzero")] or [np.nan])),
                               n_nondoor=len(nd))
N9 = NAMES7 + ("|D|", "|D|/layer")
rng = np.random.default_rng(0)


def boot(v, B=4000):
    v = np.array([x for x in v if x is not None and not np.isnan(x)])
    m = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.quantile(m, .025)), hi=float(np.quantile(m, .975)), n=int(len(v)))


def src(t):
    if t.endswith("__num"): return "number_only"
    if t.startswith(("NPI", "FG")): return "kumon"
    if t.startswith(("npi_", "filler_gap_", "agr_", "cleft", "garden", "gss")): return "causalgym"
    if t.startswith("ravel_"): return "ravel"
    if t.startswith("sentiment_"): return "sentiment"
    if t.startswith("sst_"): return "sst"
    if t.startswith("ioi_"): return "ioi"
    return "other"


def best(x, names):
    vals = {n: x[n] for n in names if x.get(n) is not None and not np.isnan(x[n])}
    return max(vals.values())


dif = [abs((v["all"]["theory"] - best(v["all"], NAMES7)) - post["excess_rank_per_map"][k]["ce"]) for k, v in res.items()
       if v["all"] and post["excess_rank_per_map"].get(k) and post["excess_rank_per_map"][k]["ce"] is not None]
assert dif and max(dif) < 1e-9, f"not excess_rank_compare's arithmetic: max |diff| {max(dif) if dif else None}"


def summary(maps, door_maps):
    out = dict(maps=len(maps), theory_rho=boot([v["all"]["theory"] for v in maps.values()]),
               rho={n: float(np.nanmean([v["all"][n] for v in maps.values()])) for n in NAMES7},
               minus={n: boot([v["all"]["theory"] - v["all"][n] for v in maps.values()]) for n in NAMES7},
               minus_best7=boot([v["all"]["theory"] - best(v["all"], NAMES7) for v in maps.values()]),
               best7_rho=float(np.mean([best(v["all"], NAMES7) for v in maps.values()])))
    if door_maps:
        dm = door_maps
        x = np.array([v["door"]["s_das_median"] for v in dm.values()])
        y = np.array([v["door"]["theory"] - best(v["door"], N9) for v in dm.values()]); yt = np.array([v["door"]["theory"] for v in dm.values()])
        ok = ~np.isnan(x) & ~np.isnan(y)
        out["door"] = dict(maps=len(dm), theory_rho_door=boot([v["door"]["theory"] for v in dm.values()]),
                           theory_rho_all=boot([v["all"]["theory"] for v in dm.values() if v["all"]]),
                           minus_best9=boot(list(y)), minus_best7=boot([v["door"]["theory"] - best(v["door"], NAMES7) for v in dm.values()]),
                           minus_D=boot([v["door"]["theory"] - v["door"]["|D|"] for v in dm.values()]),
                           minus_D_layer=boot([v["door"]["theory"] - v["door"]["|D|/layer"] for v in dm.values() if v["door"].get("|D|/layer") is not None]),
                           cells_door_median=float(np.median([v["door"]["n"] for v in dm.values()])),
                           cells_other_median=float(np.median([v["n_nondoor"] for v in dm.values()])),
                           s_das_door_median=float(np.nanmedian(x)), s_das_other_median=float(np.nanmedian([v["nondoor_s_das"] for v in dm.values()])),
                           spearman_attn_theory=list(map(float, spearmanr(x[ok], yt[ok]))), spearman_attn_diff=list(map(float, spearmanr(x[ok], y[ok]))))
    return out


maps = {k: v for k, v in res.items() if v["all"]}; dm = {k: v for k, v in res.items() if v["door"]}
report = dict(set_name=SET, check_max_abs_diff=float(max(dif)), pooled=summary(maps, dm),
              per_source={s: summary({k: v for k, v in maps.items() if src(k.split("/")[1]) == s},
                                     {k: v for k, v in dm.items() if src(k.split("/")[1]) == s})
                          for s in sorted({src(k.split("/")[1]) for k in maps})},
              per_map=res)
dest = R / f"experiments/results/inside_sets/{SET}_excess_rank.post.json"
json.dump(report, open(dest, "w"), indent=0)
p = report["pooled"]
print(f"{SET}: {len(maps)} maps (check {max(dif):.1e}); theory {p['theory_rho']['mean']:+.3f}; minus best7 {p['minus_best7']['mean']:+.3f}"
      f" [{p['minus_best7']['lo']:+.3f}, {p['minus_best7']['hi']:+.3f}]; written {dest.relative_to(R)}")

"""Readings of one named set of inside maps (Section 6; App. D.1), from the inside-map scorer's per-cell records
(experiments/score/score_inside.py, run by experiments/score/score_inside_set.py) and the maps' own files; the scorer's tests trace-families are
described there. experiments/tests/inside_set_post_test.py tests it on synthetic data.
- Checks after the run, per map of the list of maps (experiments/matrix/inside_maps.tsv): the run's --data is the list's (a map whose
  data differ from the list's sha256 is kept and reported in data_not_the_papers); the test pairs are the readout's for the
  same task (as many, the same mirror clusters, and T within float32's rounding bound for an effect at the model's logit
  scale, experiments/lib/instrument_tol.py; the largest deviation, in units of max(|T|, 1), is reported per map). For a large model's
  CausalGym map the same pair check against the readout's CausalGym file. A map that fails, or whose instruments failed in
  the scorer, enters no test and is listed.
- Degenerate cells: a cell below the readout cell whose region, or the region of its best context chosen on the test pairs, is
  empty in the base or the source in some row of its map (experiments/lib/regions.py) leaves every count, share and correlation and
  is reported apart.
- Families across sources: NPI (CausalGym npi_*, Kumon NPI*), filler-gap (filler_gap_*, Kumon FG*), agreement (agr_*,
  number-only agr_*__num), attribute (RAVEL), sentiment (templates, SST), IOI. Medians of s_DAS per model, family and source,
  with a map-level bootstrap interval, and the "more on NPI and filler-gap than agreement" reading per source, both over the
  cells where DAS's interval excludes 0 (the share of an effect indistinguishable from 0 says nothing).
- IOI's four maps leave attention_share's natural directions, squared_error(a)'s u-hat and b-hat and door_readings(b)-(c) (their signs follow the base type, which
  does not fix IOI's answer) and are reported apart.
- Benjamini-Hochberg (q = 0.025) recounted for ceiling and above_own over each model's non-degenerate live cells.
- excess_rank (excess_rank_compare): across a map's cells, does the first-order excess lambda_1 - tr M rank where DAS exceeds the network
  (e - g) better than the best of seven naive predictors available before training (G, -G, |g|, |g| - G, G* - G, the
  untrained mean difference's own excess, the depth)? On every resample the predictors come from the drawn pairs and DAS's
  excess from the pairs not drawn; the difference of the rank correlations is averaged over resamples. For ce (the mean of the
  three starts) and eff (s = 0). The same on the door cells alone (the cells before the readout position, where the run stores
  each pair's input change at the site) with two more comparators, the input change's norm ||Delta|| and that norm over its
  layer's mean; excess_rank holds only where both forms hold. A reading without
  the maps whose test sentences recur across mirror clusters is reported. The third form (lambda_1 against DAS's whole
  effect), the scorer's rank-residual form and the spline form are reported only; door_readings(b)'s strong form is its comparison with
  the best naive predictor.
- Strong forms of excess_rank, door_readings(b), squared_error(b) and D7b' (per-map median of the route difference; and cells where sqm_G's effect excludes
  0): per model with the map-level bootstrap; pooled with the map-level and the construction-level bootstrap (a construction:
  the task, RAVEL's name without _json, SST's two frames, IOI's ioi_varied + ioi_fixed and ioi_wang_20 + ioi_wang_1, a
  number-only task with its CausalGym parent, a _mod task with its twin; a resample takes every map of each drawn
  construction, in every model), a pooled strong form holding only if both do; with and without SST.
- Upper bounds: on tasks where base and source differ in more than one region the not-produced share is an upper bound; the
  share of cells in such maps is reported beside it.
Inputs: experiments/results/inside_sets/<set>.json and .meta.json (experiments/score/score_inside_set.py), the maps in experiments/results/inside/,
the list, the readout's per-pair files (experiments/results/readout/), the run records in experiments/logs/.
- A descriptive set (--descriptive=1, as score_all.sh runs large_npi_fg_agr: three maps per model, where a bootstrap's 2.5%
  quantile is the smallest value and every strong form becomes a sign rule): the same numbers with no "holds". The soft-capped
  model (Gemma-2; --capped=<tags>, default gemma-2-9b) is reported per model and per map, marked, and left out of every pooled
  number.
- A comparator that does not vary over a map's cells (||Delta|| over its layer's mean where every layer has one door cell) has
  no correlation; it is skipped and named per map ("undefined").
Output: experiments/results/inside_sets/<set>.post.json and a summary on stdout.
Usage: python -I experiments/score/inside_set_post.py --set=<name> [--set_scope=list] [--descriptive=1] [--root=<dir>]
       (--set_scope=list: every map of the list must have an output, else it is excluded as missing; --root: a synthetic
       tree, for the test)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
ROOT = Path(A.get("root", Path(__file__).resolve().parents[2]))
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "experiments/lib"))
from partial import beyond, rank_residual  # noqa: E402
from regions import facts  # noqa: E402
sys.path.insert(0, str(CODE))
from owneffect.stats import bh, boot_counts, boot_ratio, clusters, interval  # noqa: E402   (the same functions as experiments/lib/measure.py; experiments/tests/owneffect_stats_identity_test.py checks they agree)
import instrument_tol  # noqa: E402
from instrument_tol import tol  # noqa: E402

Q, NB = 0.025, 2000
FAKE = "root" in A
instrument_tol.scales(ROOT / "experiments/results/logit_scale.json")          # the data root's scales (a synthetic tree brings its own)
DESCRIPTIVE = A.get("descriptive") == "1"
CAPPED = tuple(A.get("capped", "gemma-2-9b").split(","))               # the soft-capped family (Gemma-2): first order loosens


def family(task):
    if task.startswith("npi_") or task.startswith("NPI"):
        return "npi"
    if task.startswith("filler_gap_") or task.startswith("FG"):
        return "fg"
    if task.startswith("agr_"):
        return "agreement"
    if task.startswith("ravel_"):
        return "attribute"
    if task.startswith(("sentiment_", "sst_")):
        return "sentiment"
    if task.startswith("ioi_"):
        return "ioi"
    return "other"


def source(task):
    if task.endswith("__num"):
        return "number_only"
    if task.startswith(("NPI", "FG")):
        return "kumon"
    if task.startswith(("npi_", "filler_gap_", "agr_")):
        return "causalgym"
    return {"ravel": "ravel", "senti": "sentiment_templates", "sst_c": "sst", "ioi_w": "ioi_wang", "ioi_v": "ioi_ours",
            "ioi_f": "ioi_ours"}.get(task[:5], "other")


def construction(task):
    if task.endswith("__num"):
        return task[:-len("__num")]                                    # the CausalGym parent's sentences, row for row
    if task.endswith("_mod"):
        return task[:-len("_mod")]                                     # CausalGym's twin of a task: its templates (as score_causalgym_map.py counts it)
    if task.startswith("ravel_"):
        return task.replace("_json", "")
    if task.startswith("sst_"):
        return "sst"
    if task in ("ioi_varied", "ioi_fixed"):
        return "ioi_ours"
    if task in ("ioi_wang_20", "ioi_wang_1"):
        return "ioi_wang"
    return task


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---- inputs
NAME = A["set"]
S = json.load(open(ROOT / f"experiments/results/inside_sets/{NAME}.json"))
LIST, DATA_DIFFERS = {}, set()
lp = ROOT / "experiments/matrix/inside_maps.tsv"
if lp.is_file():
    for l in lp.read_text().strip().split("\n"):
        m, g, t, d, _, s_tr, s_te = l.split("\t")
        LIST[(m, t)] = dict(group=g, data=d, sha=[s_tr, s_te])


def rows_of(m, t):
    """The map's rows, as the run read them: the list's directory for a map of the list, CausalGym's snapshot otherwise."""
    if (m, t) in LIST:
        d = ROOT / LIST[(m, t)]["data"] / t
        return {s: json.load(open(d / f"{s}.json")) for s in ("train", "test")}
    if FAKE:
        return json.load(open(ROOT / f"fake_rows/{t}.json"))
    sys.path.insert(0, str(CODE / "experiments/matrix")); import registry  # noqa: E402
    from huggingface_hub import hf_hub_download
    rev = registry.CAUSALGYM_REV
    return {s: [r for r in json.load(open(hf_hub_download("aryaman/causalgym", f"{s}.json", repo_type="dataset", revision=rev)))
                if r["task"] == t] for s in ("train", "test")}


LARGE_TAGS = ("Llama-3.1-8B", "gemma-2-9b", "Qwen2.5-7B", "pythia-6.9b")
T_DEV = {}                                                             # (model, task): the largest |T - T_readout| / max(|T_readout|, 1)


def same_pairs(m, t, Z, R):
    """The map's test pairs are the readout's: as many, the same mirror clusters, and T within float32's bound for an effect at
    the model's logit scale (experiments/lib/instrument_tol.py). The largest deviation is kept (in units of max(|T|, 1)). None when they
    are, else why not."""
    a, b = np.asarray(Z["T"], float), np.asarray(R[f"{t}/T"], float)
    if a.shape != b.shape:
        return f"{len(a)} test pairs inside, {len(b)} at the readout"
    if f"{t}/cluster" in R.files and not np.array_equal(np.asarray(Z["cluster"]), np.asarray(R[f"{t}/cluster"])):
        return "the test pairs' mirror clusters are not the readout's"
    T_DEV[(m, t)] = float((np.abs(a - b) / np.maximum(np.abs(b), 1)).max())
    d = float(np.abs(a - b).max())
    if d > tol(m, "effect", fake=FAKE):
        return f"T differs from the readout's by up to {d:.2e}, beyond float32's scale {tol(m, 'effect', fake=FAKE):.2e}"
    return None


def post_run(m, t, J, Z):
    """Checks after the run: for a map of the list, its data and its test pairs; for a large model's CausalGym map, its test
    pairs against the readout's; None when they hold, else why not."""
    if (m, t) not in LIST:
        if m not in LARGE_TAGS:
            return None
        ro = ROOT / f"experiments/results/readout/readout_{m}_m_causalgym.npz"
        if not ro.is_file() or f"{t}/T" not in np.load(ro).files:
            return f"no readout T for {t}"
        return same_pairs(m, t, Z, np.load(ro))
    L = LIST[(m, t)]
    env = ROOT / "experiments/logs" / J["run"] / "env.json"
    if not env.is_file():
        return f"no run record {env.relative_to(ROOT)}"
    E = json.load(open(env)); data = [a.split("=", 1)[1] for a in E["command"] if a.startswith("--data=")]
    if data != [L["data"]]:
        return f"the run read --data={data} and the list says {L['data']}"
    if not FAKE and any(sha((ROOT / L["data"] / t / f"{s}.json").read_bytes()) != h for s, h in zip(("train", "test"), L["sha"])):
        DATA_DIFFERS.add(f"{m}/{t}")              # data built other than the paper's (experiments/matrix/inside_maps.tsv): reported, kept
    ro = ROOT / f"experiments/results/readout/readout_{m}_m_{L['group']}.npz"
    if not ro.is_file():
        return f"no readout file {ro.name}"
    R = np.load(ro)
    if f"{t}/T" not in R.files:
        return f"the readout has no T for {t}"
    return same_pairs(m, t, Z, R)


# ---- the maps, their checks and degenerate regions
cells_all = [c for m in S["cells"] for c in S["cells"][m]]
maps = sorted({(c["model"], c["task"]) for c in cells_all})
excluded, degenerate, multi, files = {}, {}, {}, {}
for k, v in S["report"]["maps"].items():
    if v["instruments"] != "OK" or not v["enters"]:
        excluded[k] = f"scorer: instruments {v['instruments']}, enters {v['enters']}"
res = ROOT / "experiments/results/inside"
for m, t in maps:
    J = json.load(open(res / f"inside_{m}_{t}.json")); Z = np.load(res / f"inside_{m}_{t}.npz")
    files[(m, t)] = (J, Z)
    if why := post_run(m, t, J, Z):
        excluded[f"{m}/{t}"] = why
    f = facts(rows_of(m, t)); degenerate[(m, t)] = set(f["degenerate_regions"]); multi[(m, t)] = f["multi_region"] > 0
keep = [c for c in cells_all if f"{c['model']}/{c['task']}" not in excluded]


def readout_layer(J):
    """The readout cell's layer: the one cell layer that is not among the map's every-second layers (inside.py appends it)."""
    extra_ = {c["layer"] for c in J["cells"]} - set(J.get("layers", []))
    return max(extra_) if J.get("layers") and extra_ else None


def context_of(c):
    """The best context (chosen on the test pairs, as G*_test) of a scorer cell, from its map's own record."""
    J = files[(c["model"], c["task"])][0]
    return next((x.get("context_test") for x in J["cells"] if x["layer"] == c["layer"] and x["region"] == c["region"]), None)


deg_cells = [c for c in keep if c["layer"] != readout_layer(files[(c["model"], c["task"])][0])
             and (c["region"] in degenerate[(c["model"], c["task"])] or context_of(c) in degenerate[(c["model"], c["task"])])]
if LIST and A.get("set_scope") == "list":                             # a map of the list without an output is excluded, by name
    META = json.load(open(ROOT / f"experiments/results/inside_sets/{NAME}.meta.json"))    # the set's own tasks and models
    for (m, t) in LIST:
        if t in META["tasks"] and m in META["models"] and f"{m}/{t}" not in S["report"]["maps"]:
            excluded[f"{m}/{t}"] = "missing: no output"
cells = [c for c in keep if c not in deg_cells]


def extra(c):
    """Per-cell quantities the scorer does not keep: sqm_G s = 0's effect interval, from the map's own pairs."""
    J, Z = files[(c["model"], c["task"])]
    C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"])
    x = Z[f"{c['cell']}/e/das_sqm_G_s0_normal"]
    lo, hi = interval(float(x.mean()), boot_ratio(x, np.ones_like(x), C))
    return dict(sqm_nonzero=bool(lo > 0 or hi < 0))


for c in cells:
    c.update(extra(c))


# ---- statistics
def rank(v):
    """Ranks with ties averaged, as Spearman's correlation needs."""
    return _avg_ranks(v)


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def _row_spearman(A, B_):
    """Spearman correlation of A[b] and B_[b] for every row b (rows: resamples; columns: cells), ties averaged in both
    (|g| - G and G* - G are exactly 0 at many cells)."""
    ra, rb = _row_avg_ranks(A), _row_avg_ranks(B_)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def _avg_ranks(v):
    """Ranks with ties averaged (for a predictor that is constant over a layer's cells, such as the depth)."""
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v), dtype=float)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _tied_spearman(r, B_):
    """Spearman correlation of a fixed rank vector r (ties averaged) with each row of B_ (resamples x cells)."""
    rb = _row_avg_ranks(B_); rb -= rb.mean(1, keepdims=True); ra = r - r.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum(1))
    return np.where(den > 0, (rb @ ra) / np.where(den > 0, den, 1), np.nan)


NAMES7 = ("G", "-G", "|g|", "|g|-G", "G*-G", "u-G", "depth")
NAMES9 = NAMES7 + ("|D|", "|D|/layer")


def _row_avg_ranks(X):
    """Ranks along the last axis with ties averaged, every row at once (equal to _avg_ranks row by row)."""
    X = np.asarray(X, float); n = X.shape[1]; o = np.argsort(X, axis=1, kind="mergesort"); xs = np.take_along_axis(X, o, 1)
    starts = np.ones(xs.shape, bool); starts[:, 1:] = xs[:, 1:] != xs[:, :-1]
    run = np.cumsum(starts, axis=1) - 1; rows = np.arange(X.shape[0])[:, None]
    sums = np.zeros(X.shape); cnts = np.zeros(X.shape)
    np.add.at(sums, (rows, run), np.broadcast_to(np.arange(n, dtype=float), X.shape)); np.add.at(cnts, (rows, run), 1.0)
    r = np.empty(X.shape); np.put_along_axis(r, o, sums[rows, run] / cnts[rows, run], 1)
    return r


def excess_rank_compare(key, cs, objective, door=False):
    """excess_rank over a map's cells cs (live, non-degenerate, below the readout).
    On every resample b of the map's own clusters: the Spearman correlation across cells of the first-order excess lambda_1 -
    tr M from the drawn pairs (the run's lam_mid_boot, the pairs' midpoint traces) with DAS's excess over the network's own
    interchange, e - g, on the pairs not drawn; and the same for seven naive predictors of that excess from the drawn pairs,
    all available before training: G, -G, |g|, |g| - G, G* - G, e(u-hat) - G (the untrained mean difference's own excess) and the
    layer's depth. Each is averaged over resamples; "diff" is the excess's correlation minus the best naive one on this map.
    door=True: the door cells of cs only (the run stores each test pair's input change at the site there), with two more
    comparators from the drawn pairs: ||Delta|| (door/delta_norm) and ||Delta|| over the mean of its layer's door cells.
    "total" is the third form, reported only: lambda_1 against DAS's whole effect, minus the best of G, G* and |g|.
    arm: "ce" (the mean of the three starts' per-pair effects) or "eff" (s = 0). NaN with fewer than three cells or a target that
    does not vary."""
    J, Z = files[key]
    if door:
        cs = [c for c in cs if c.get("door") and f"{c['cell']}/door/delta_norm" in Z.files]
    out = dict(n=len(cs))
    names = NAMES9 if door else NAMES7
    if len(cs) < 3:
        return dict(out, diff=np.nan, theory=np.nan, best=None, total=np.nan, diff_within_map=None, **{n: np.nan for n in names})
    C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"]).astype(float); oob = (C == 0).astype(float)
    ok = oob.sum(1) > 0; C, oob = C[ok], oob[ok]; w, wo = C.sum(1)[:, None], oob.sum(1)[:, None]
    ks = [c["cell"] for c in cs]
    if objective == "ce":
        e = np.stack([np.mean([Z[f"{k}/e/das_ce_s{s_}_normal"] for s_ in (0, 1, 2)], axis=0) for k in ks], 1)
    else:
        e = np.stack([Z[f"{k}/e/das_eff_s0_normal"] for k in ks], 1)
    get = lambda n: np.stack([Z[f"{k}/{n}"] for k in ks], 1)  # noqa: E731
    g, gs, tr, uh = get("e/own_normal"), get("gstar_test"), get("tr_mid"), get("e/u_hat_normal")
    lam = np.stack([Z[f"{k}/lam_mid_boot"][ok] for k in ks], 1)          # (resamples, cells)
    gb, ab = C @ g / w, C @ np.abs(g) / w
    naive = {"G": gb, "-G": -gb, "|g|": ab, "|g|-G": ab - gb, "G*-G": C @ gs / w - gb, "u-G": C @ uh / w - gb}
    depth = _avg_ranks([c["layer"] for c in cs])                        # the same on every resample; ties kept
    rel = None
    if door:
        dn = C @ get("door/delta_norm") / w                             # ||Delta|| from the drawn pairs (resamples, cells)
        lay = np.array([c["layer"] for c in cs]); rel = np.empty_like(dn)
        for L in np.unique(lay):
            j = lay == L; rel[:, j] = dn[:, j] / dn[:, j].mean(1, keepdims=True)
        naive["|D|"] = dn
    xo = oob @ (e - g) / wo
    if np.nanmax(np.abs(xo - xo.mean(1, keepdims=True))) < 1e-12:
        out.update(diff=np.nan, theory=np.nan, best=None, diff_within_map=None, **{n: np.nan for n in names})
    else:
        th = _row_spearman(lam - C @ tr / w, xo); out["theory"] = float(np.nanmean(th))
        per = {n: _row_spearman(x, xo) for n, x in naive.items()}; per["depth"] = _tied_spearman(depth, xo)
        if door:
            per["|D|/layer"] = _row_spearman(rel, xo)                  # ties where a layer has one door cell (the ratio is 1)
        for n in names:
            out[n] = float(np.nanmean(per[n]))
        out["undefined"] = [n for n in names if np.isnan(out[n])]     # a comparator constant over the cells (||Delta|| over its
        defined = [n for n in names if not np.isnan(out[n])]          # layer's mean where every layer has one door cell): skipped
        out["best"] = max(defined, key=lambda n: out[n]) if defined else None
        out["diff"] = out["theory"] - out[out["best"]] if defined else np.nan
        if defined:
            d_b = th - per[out["best"]]                                 # within the map: over its own resamples (reported)
            out["diff_within_map"] = [float(np.nanquantile(d_b, 0.025)), float(np.nanquantile(d_b, 0.975))]
        else:
            out["diff_within_map"] = None
    eo = oob @ e / wo                                                   # the third form, reported only
    th3 = float(np.nanmean(_row_spearman(lam, eo)))
    out["total"] = th3 - max(float(np.nanmean(_row_spearman(x, eo))) for x in (gb, C @ gs / w, ab))
    return out


def sentence_reuse(m, t):
    """True when two test rows in different mirror clusters share a base or a source text: their errors are then shared across
    a resample's drawn and undrawn pairs (the sentence-level rule of experiments/score/score_readout_sentence.py)."""
    rows = rows_of(m, t)["test"]
    keys = [("".join(r["base"]), "".join(r["src"])) for r in rows]
    parent = list(range(len(keys))); first = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, (b_, s_) in enumerate(keys):
        for sent in (b_, s_):
            if sent in first:
                ri, rj = find(i), find(first[sent])
                if ri != rj:
                    parent[ri] = rj
            else:
                first[sent] = i
    return len({find(i) for i in range(len(keys))}) < len(set(clusters(keys).tolist()))


def per_map(bym):
    """Per-map statistics of the strong forms."""
    out = {}
    for key, v in bym.items():
        d6 = [c for c in v if c.get("door_predictor") and c["door_predictor"]["pred_open"] is not None and c["ce_nonzero"] and family(key[1]) != "ioi"]
        P, Y = [c["door_predictor"]["pred_open"] for c in d6], [c["attention_share"]["s_das"] for c in d6]
        naive = [sp([c["door_predictor"]["naive_delta"] for c in d6], Y), sp([c["door_predictor"]["naive_alpha"] for c in d6], Y), sp([-c["G_T"] for c in d6], Y)]
        rd = [c["squared_error"]["route_diff"][0] for c in v if c["squared_error"].get("route_diff")]
        rd_nz = [c["squared_error"]["route_diff"][0] for c in v if c["squared_error"].get("route_diff") and c["sqm_nonzero"]]
        inner = [c for c in v if c["layer"] != readout_layer(files[key][0])]
        d3, d3e = excess_rank_compare(key, inner, "ce"), excess_rank_compare(key, inner, "eff")
        dd, dde = excess_rank_compare(key, inner, "ce", door=True), excess_rank_compare(key, inner, "eff", door=True)
        parts, parts9 = ("theory",) + NAMES7 + ("n",), ("theory",) + NAMES9 + ("n",)
        out[key] = dict(
            excess_rank_ce=d3["diff"], excess_rank_eff=d3e["diff"], excess_rank_total_ce=d3["total"], excess_rank_total_eff=d3e["total"],
            excess_rank_parts=dict(ce={k: d3[k] for k in parts}, eff={k: d3e[k] for k in parts}), excess_rank_best=dict(ce=d3["best"], eff=d3e["best"]),
            excess_rank_within_map=dict(ce=d3["diff_within_map"], eff=d3e["diff_within_map"]),
            excess_rank_door_ce=dd["diff"], excess_rank_door_eff=dde["diff"],
            excess_rank_door_parts=dict(ce={k: dd[k] for k in parts9}, eff={k: dde[k] for k in parts9}), excess_rank_door_best=dict(ce=dd["best"], eff=dde["best"]),
            excess_rank_door_within_map=dict(ce=dd["diff_within_map"], eff=dde["diff_within_map"]),
            excess_rank_undefined=dict(ce=d3.get("undefined", []), eff=d3e.get("undefined", [])),
            excess_rank_door_undefined=dict(ce=dd.get("undefined", []), eff=dde.get("undefined", [])),
            excess_rank_rank_residual=rank_residual([c["pred"] for c in v], [c["meas_ce"] for c in v], [c["G_T"] for c in v]),
            excess_rank_spline_form=beyond([c["pred"] for c in v], [c["meas_ce"] for c in v], [c["G_T"] for c in v]),
            door_predictor_diff=(sp(P, Y) - max(naive)) if family(key[1]) != "ioi" and not any(np.isnan(naive + [sp(P, Y)])) else np.nan,
            squared_error_route=float(np.mean(rd)) if rd else np.nan,
            squared_error_route_median=float(np.median(rd)) if rd else np.nan,
            squared_error_route_sqm_nonzero=float(np.mean(rd_nz)) if rd_nz else np.nan)
    return out


def boot_maps(vals, B=NB, seed=11):
    v = np.array([x for x in vals if not np.isnan(x)])
    if len(v) == 0:
        return [None, None, None, 0]
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(B, len(v)))].mean(1)
    return [float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), int(len(v))]


def boot_constructions(stat, B=NB, seed=12):
    """stat: {(model, task): value}. Resample constructions; each draw takes every map of the construction."""
    groups = {}
    for (m, t), x in stat.items():
        if not np.isnan(x):
            groups.setdefault(construction(t), []).append(x)
    if not groups:
        return [None, None, None, 0]
    keys = sorted(groups); sums = np.array([sum(groups[k]) for k in keys]); ns = np.array([len(groups[k]) for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(B, len(keys)))
    m = sums[idx].sum(1) / ns[idx].sum(1); pt = sums.sum() / ns.sum()
    return [float(pt), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), int(len(keys))]


SIGN = {"excess_rank_ce": 1, "excess_rank_eff": 1, "excess_rank_door_ce": 1, "excess_rank_door_eff": 1, "door_predictor_diff": 1, "squared_error_route": -1, "squared_error_route_median": -1,
        "squared_error_route_sqm_nonzero": -1, "excess_rank_total_ce": 1, "excess_rank_total_eff": 1, "excess_rank_rank_residual": 1, "excess_rank_spline_form": 1}
# reported, not tested: the third form (lambda_1 against DAS's whole effect) and the rank-residual and spline forms (both
# biased under the no-information null)
REPORT_ONLY = {"excess_rank_total_ce", "excess_rank_total_eff", "excess_rank_rank_residual", "excess_rank_spline_form"}


def holds(iv, sign):
    if iv[1] is None:
        return False
    return iv[1] > 0 if sign > 0 else iv[2] < 0


def strong(pm, pooled):
    out = {}
    for k, sg in SIGN.items():
        vals = {key: v[k] for key, v in pm.items()}
        mb = boot_maps(list(vals.values()))
        row = dict(map_level=mb, holds_map=holds(mb, sg))
        if pooled:
            cb = boot_constructions(vals); row.update(construction_level=cb, holds_construction=holds(cb, sg),
                                                    holds=holds(mb, sg) and holds(cb, sg))
        else:
            row["holds"] = holds(mb, sg)
        if k in REPORT_ONLY or DESCRIPTIVE:
            row = {kk: vv for kk, vv in row.items() if not kk.startswith("holds")}
            row["report_only" if k in REPORT_ONLY else "descriptive"] = True
        out[k] = row
    if DESCRIPTIVE:                                                    # three maps per model: a sign rule, not a test
        return out
    out["door_predictor"] = dict(holds=out["door_predictor_diff"]["holds"])
    out["excess_rank"] = dict(holds=out["excess_rank_ce"]["holds"] and out["excess_rank_door_ce"]["holds"],      # every cell, and the door cells with ||Delta||
                     eff=out["excess_rank_eff"]["holds"] and out["excess_rank_door_eff"]["holds"])
    out["squared_error_route_median_and_sqm"] = dict(holds=out["squared_error_route_median"]["holds"] and out["squared_error_route_sqm_nonzero"]["holds"])
    return out


def door_block(cs):
    """door_readings(c) over the halves that have rows (a single-type task has one), without IOI's maps: the share of door cells whose
    largest rise of the site's attention weight under DAS (ce s = 0) is in block L, and the median share of the rise there."""
    hits, shares = 0, []
    for c in cs:
        if not c.get("door") or family(c["task"]) == "ioi":
            continue
        _, Z = files[(c["model"], c["task"])]; k = c["cell"]
        if f"{k}/door/w_base" not in Z.files:
            continue
        half = Z["half"].astype(bool); wb, wd = Z[f"{k}/door/w_base"], Z[f"{k}/door/w_das_ce_s0"]
        rises = np.stack([(wd[h] - wb[h]).mean(0) for h in (half, ~half) if h.sum() > 0])      # halves x blocks x heads
        flat = np.unravel_index(np.argmax(rises), rises.shape); hits += int(flat[1] == 0)
        if rises.max() > 0:
            shares.append(float(rises[:, 0].max() / rises.max()))
    n = sum(1 for c in cs if c.get("door") and family(c["task"]) != "ioi")
    return dict(door_cells=n, argmax_in_block_L=hits, block_L_share_median=float(np.median(shares)) if shares else None)


def g_class(c):
    """The cell's class from G's own interval (score_inside.py's own_defined and G's sign): carried (above 0), zero
    (indistinguishable from 0) or negative (below 0). No threshold of ours."""
    return ("carried" if c["G_T"] > 0 else "negative") if c["attention_share"]["own_defined"] else "zero"


def counts(cs):
    """The scorer's per-model counts, recomputed over the given cells (BH over these cells only)."""
    return dict(
        live_cells=len(cs),
        classes={k: sum(g_class(c) == k for c in cs) for k in ("carried", "zero", "negative")},
        ceiling=dict(above=sum(any(v["above"] for v in c["ceiling"].values()) for c in cs), above_bh=int(bh([c["ceiling_p"] for c in cs], Q).sum()) if cs else 0),
        above_own={n: dict(above=sum(c[f"above_own_{n}"]["above"] for c in cs), below=sum(c[f"above_own_{n}"]["below"] for c in cs),
                    above_bh=int(bh([c[f"above_own_{n}"]["p"] for c in cs], Q).sum()) if cs else 0,
                    above_bh_G_indistinct=int(sum(r and not c["attention_share"]["own_defined"] for c, r in zip(cs, bh([c[f"above_own_{n}"]["p"] for c in cs], Q))))
                    if cs else 0)
            for n in ("G", "Gstar_train", "Gstar_test")},
        # the shares not produced by the network are against G*_test
        attention_share=dict(not_produced_share_gstar_test=float(sum(max(c["ce_T"] - c["Gstar_test_T"], 0) for c in cs) / max(sum(c["ce_T"] for c in cs), 1e-12)),
                not_produced_through_attention=float(sum(max(c["ce_T"] - c["Gstar_test_T"], 0) * min(max(c["attention_share"]["s_das"], 0), 1)
                                                         for c in cs if c["attention_share"]["s_das"] is not None)   # undefined: DAS does nothing there
                                                     / max(sum(c["ce_T"] for c in cs), 1e-12)),
                s_das_undefined=[[c["model"], c["task"], c["layer"], c["region"]] for c in cs if c["attention_share"]["s_das"] is None],
                upper_bound_share_of_cells=float(np.mean([multi[(c["model"], c["task"])] for c in cs])) if cs else None,
                natural_median_share={n: (float(np.median(x)) if (x := [c["attention_share"]["natural"][n] for c in cs if family(c["task"]) != "ioi"
                                                                        and c["attention_share"]["natural"][n] is not None]) else None)
                                      for n in ("u_hat", "b_hat", "bisector", "v1_mid")}),
        squared_error_landing=dict(population=len(p7 := [c for c in cs if c["above_own_G"]["above"] and c["squared_error"]["in_range"]]),
                 land_contains0={n: sum(c["squared_error"]["land"][n]["contains0"] for c in p7 if not (n in ("u_hat", "b_hat") and family(c["task"]) == "ioi"))
                                 for n in ("sqm", "sqG", "u_hat", "b_hat", "ce")}))


def families_table(cs):
    """Median s_DAS per family and source, with a map-level bootstrap of the per-map medians' mean, over the cells where
    s_DAS is defined and DAS's interval excludes zero (elsewhere the share of a near-zero effect says nothing), as for the
    paper's medians."""
    out = {}
    cs = [c for c in cs if c["attention_share"]["s_das"] is not None and c.get("ce_nonzero")]   # as the paper's medians: DAS's interval off 0
    for src in sorted({source(c["task"]) for c in cs}):
        for fam in sorted({family(c["task"]) for c in cs if source(c["task"]) == src}):
            sel = [c for c in cs if source(c["task"]) == src and family(c["task"]) == fam]
            bym = {}
            for c in sel:
                bym.setdefault(c["task"], []).append(c["attention_share"]["s_das"])
            out[f"{src}/{fam}"] = dict(cells=len(sel), maps=len(bym), s_das_median=float(np.median([c["attention_share"]["s_das"] for c in sel])),
                                       per_map_median=boot_maps([float(np.median(v)) for v in bym.values()]))
    return out


def npi_fg_reading(cs_by_model, B=NB, seed=13):
    """Per source pairing (CausalGym; Kumon against number-only) and model: the mean over NPI and filler-gap maps of each map's
    median s_DAS (over the cells where DAS's interval excludes 0), minus the same over agreement maps, with a bootstrap of maps
    within each group; the reading holds in a model when the interval lies above 0, and for a source when it holds in every
    model with two maps in each group (None, not computable, when no model has them, as in the large models' three maps)."""
    rng = np.random.default_rng(seed); out = {}
    for name, nf, ag in (("causalgym", {("causalgym", "npi"), ("causalgym", "fg")}, {("causalgym", "agreement")}),
                         ("kumon_number_only", {("kumon", "npi"), ("kumon", "fg")}, {("number_only", "agreement")})):
        per = {}
        for m, cs in cs_by_model.items():
            med = {}
            for c in cs:
                if c["attention_share"]["s_das"] is not None and c.get("ce_nonzero"):   # defined, and DAS's interval excludes 0, as the medians
                    med.setdefault((source(c["task"]), family(c["task"]), c["task"]), []).append(c["attention_share"]["s_das"])
            a = np.array([np.median(v) for (s_, f_, _), v in med.items() if (s_, f_) in nf])
            b = np.array([np.median(v) for (s_, f_, _), v in med.items() if (s_, f_) in ag])
            if len(a) < 2 or len(b) < 2:
                per[m] = None; continue
            d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]
            lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            per[m] = dict(diff=float(a.mean() - b.mean()), ci=[lo, hi], holds=lo > 0, maps=[len(a), len(b)])
            if DESCRIPTIVE:
                del per[m]["holds"]
        judged = [v for v in per.values() if v is not None]
        out[name] = dict(per_model=per, models_not_judged=sorted(m for m, v in per.items() if v is None))
        if not DESCRIPTIVE:
            out[name]["holds_every_model"] = all(v["holds"] for v in judged) if judged else None
    return out


# per-family medians of the cells' own s_DAS, over the live non-degenerate cells where DAS's interval excludes 0 (families_table)
models = sorted({c["model"] for c in cells})
report = dict(set_name=NAME, data_not_the_papers=sorted(DATA_DIFFERS), excluded_maps=excluded, degenerate_cells=len(deg_cells),
              degenerate_by_map={f"{m}/{t}": n for (m, t) in maps if (n := sum(1 for c in deg_cells if (c["model"], c["task"]) == (m, t)))},
              models={}, pooled={}, ioi={})
tables = {}
for m in models:
    cs = [c for c in cells if c["model"] == m]
    bym = {}
    for c in cs:
        bym.setdefault((m, c["task"]), []).append(c)
    pm = per_map(bym); tables[m] = families_table(cs)
    report["models"][m] = dict(maps=len(bym), counts=counts(cs), door_block=door_block(cs), families=tables[m],
                               strong=strong(pm, pooled=False))
report["npi_fg_reading"] = npi_fg_reading({m: [c for c in cells if c["model"] == m] for m in models})
reuse = {(m, t): sentence_reuse(m, t) for (m, t) in maps}
pool = [c for c in cells if c["model"] not in CAPPED]                  # the soft-capped model stays out of every pooled number
for label, sel in (("with_sst", pool), ("without_sst", [c for c in pool if not c["task"].startswith("sst_")])):
    bym = {}
    for c in sel:
        bym.setdefault((c["model"], c["task"]), []).append(c)
    pm = per_map(bym)
    report["pooled"][label] = dict(maps=len(bym), constructions=len({construction(t) for _, t in bym}), strong=strong(pm, pooled=True),
                                   **{f"{form}_parts": {objective: {q: boot_maps([v[f"{form}_parts"][objective][q] for v in pm.values()])
                                                              for q in ("theory",) + names} for objective in ("ce", "eff")}
                                      for form, names in (("excess_rank", NAMES7), ("excess_rank_door", NAMES9))},
                                   **{f"{form}_best": {objective: {n: sum(v[f"{form}_best"][objective] == n for v in pm.values()) for n in names}
                                                       for objective in ("ce", "eff")}
                                      for form, names in (("excess_rank", NAMES7), ("excess_rank_door", NAMES9))})
    if label == "with_sst":                                            # a reading: maps whose sentences do not recur across clusters
        nr = {k: v for k, v in pm.items() if not reuse[k]}
        report["excess_rank_without_sentence_reuse"] = dict(maps=len(nr), maps_with_reuse=sorted(f"{m}/{t}" for (m, t) in pm if reuse[(m, t)]),
                                                             **{f"{form}_{objective}": dict(map_level=boot_maps([v[f"{form}_{objective}"] for v in nr.values()]),
                                                                                      construction_level=boot_constructions({k: v[f"{form}_{objective}"] for k, v in nr.items()}))
                                                                for form in ("excess_rank", "excess_rank_door") for objective in ("ce", "eff")})
report["cell_classes"] = {f"{c['model']}/{c['task']}/{c['cell']}": g_class(c) for c in cells}
_pm_all = per_map({k: v for k, v in ((key, [c for c in cells if (c["model"], c["task"]) == key]) for key in sorted({(c["model"], c["task"]) for c in cells})) if v})
report["excess_rank_per_map"] = {f"{m}/{t}": dict(ce=v["excess_rank_ce"], ce_within_map=v["excess_rank_within_map"]["ce"], eff=v["excess_rank_eff"], eff_within_map=v["excess_rank_within_map"]["eff"],
                                         best=v["excess_rank_best"]["ce"], cells=v["excess_rank_parts"]["ce"]["n"],
                                         door_ce=v["excess_rank_door_ce"], door_ce_within_map=v["excess_rank_door_within_map"]["ce"], door_eff=v["excess_rank_door_eff"],
                                         door_eff_within_map=v["excess_rank_door_within_map"]["eff"], door_best=v["excess_rank_door_best"]["ce"],
                                         door_cells=v["excess_rank_door_parts"]["ce"]["n"], undefined=v["excess_rank_undefined"]["ce"],
                                         door_undefined=v["excess_rank_door_undefined"]["ce"], capped=m in CAPPED) for (m, t), v in _pm_all.items()}
report["descriptive"] = DESCRIPTIVE
report["capped_models"] = sorted(m for m in models if m in CAPPED)   # reported per model and per map, never pooled
report["T_vs_readout"] = {f"{m}/{t}": d for (m, t), d in sorted(T_DEV.items())}
ioi = [c for c in cells if family(c["task"]) == "ioi"]
report["ioi"] = dict(cells=len(ioi), natural_median_share={n: (float(np.median(x)) if (x := [c["attention_share"]["natural"][n] for c in ioi if c["attention_share"]["natural"][n] is not None]) else None)
                                                           for n in ("u_hat", "b_hat", "bisector", "v1_mid")})


def strip(x):
    if isinstance(x, dict):
        return {str(k): strip(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    if isinstance(x, (np.bool_, np.integer)):
        return x.item()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


dest = ROOT / f"experiments/results/inside_sets/{NAME}.post.json"
json.dump(strip(report), open(dest, "w"), indent=1, default=float)
print(f"{NAME}: {len(maps)} maps, {len(excluded)} excluded, {len(deg_cells)} degenerate cells left out, {len(cells)} cells scored")
for k, v in excluded.items():
    print(f"  excluded {k}: {v}")
for label, p in report["pooled"].items():
    if DESCRIPTIVE:
        print(f"  pooled {label} ({p['maps']} maps, {p['constructions']} constructions): descriptive set, not tested"); continue
    print(f"  pooled {label} ({p['maps']} maps, {p['constructions']} constructions): "
          + ", ".join(f"{k} {'holds' if v['holds'] else 'does not hold'}" for k, v in p["strong"].items() if "holds" in v))
print(f"  NPI/FG reading: {json.dumps(strip(report['npi_fg_reading']))}")
print(f"written {dest.relative_to(ROOT) if not FAKE else dest}")

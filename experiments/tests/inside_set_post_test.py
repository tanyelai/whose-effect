"""Tests experiments/score/inside_set_post.py: the readings of one named set of maps inside the network, made after
experiments/score/score_inside.py has scored the set (Section 6; App. D.1). Synthetic data only (a tree with known answers; no model, no
outputs of the runs); exits 1 on a failure.
Here score_inside.py (--fake) scores the synthetic maps first, as it does a real set. The readings, with their output keys: the
ranking, in which the first-order excess (lambda_1 - tr M) ranks DAS's excess over the network's own interchange G against
naive predictors (excess_rank, excess_rank_ce, excess_rank_eff, excess_rank_door_ce, excess_rank_door_eff); the share of an effect that freezing attention removes (attention_share); whether
attention to the site rises most in the first block that reads it (door_block); the route of the squared error aimed at G
(sqm_G) against cross-entropy's (squared_error_route, squared_error_route_median, squared_error_route_sqm_nonzero).

Two models, sixteen maps each: five CausalGym-like tasks (rows from fake_rows/; agr_c_mod is agr_c's _mod twin) and eleven tasks
of the other sources from a synthetic map list, experiments/matrix/inside_maps.tsv (Kumon NPI and FG, two number-only agreement tasks,
RAVEL plain and JSON, a sentiment template, SST's two frames, two IOI tasks). Planted:
- FGMatrixQ's region 2 empty in half the base rows: its three region-2 cells per model are degenerate and must leave every count;
- one map whose run read another --data, one whose test pairs' T is off the readout's by 0.01 and one whose mirror clusters are
  not the readout's: all three excluded, no others; one whose T is off by 2e-3, within the float32 tolerance for an effect at
  the synthetic models' logit scale (S = 15, D = 768: 5.5e-3; experiments/lib/instrument_tol.py), is kept, its deviation reported;
- attention shares 0.6 on NPI and filler-gap cells, 0.05 on agreement: the "more on NPI and filler-gap" reading holds on both
  sources, with Kumon's tasks in their families (score_inside.py files them under "other");
- the share of u_hat's effect that freezing attention removes is 0.9 on IOI maps only, 0.05 elsewhere: IOI leaves the models'
  natural-direction median and is reported apart;
- a cell where DAS's effect is exactly 0: its share through attention (s_DAS) is undefined, skipped and counted;
- DAS follows the first-order top: where the network carries something its effect is G + 0.9 of the first-order excess, where
  it carries nothing 0.9 lambda_1. The ranking (the first-order excess against the best of seven naive predictors of DAS's
  excess) then holds pooled under both bootstraps (map and construction) and per model, for ce and eff, on every cell and on
  the door cells, where the input change's norm ||Delta|| is among the comparators;
- with the top and DAS constant, the ranking does not hold;
- with DAS equal to the network's own interchange pair by pair, the ranking is undefined (no excess to rank), and the third
  form, which is reported only, is negative there;
- with DAS's excess following ||Delta||, the door form fails while the seven-comparator form still holds, which is why the door
  form is needed;
- sqm_G's route lies below ce's, by the mean and by the median over each map's cells, and over the cells where sqm_G's effect
  excludes 0;
- score_inside.py's rank-residual form of the ranking stays above 0 under the constant null (it is biased there), so it is
  reported and decides nothing;
- RAVEL's two prompts, SST's two frames and IOI's two tasks are one construction each, a number-only task is its CausalGym
  parent's and a CausalGym task's _mod twin is the task's: twelve constructions with SST, eleven without;
- a sentiment template whose test rows share base texts across mirror clusters is named as sentence reuse and left out of the
  ranking's reading without it;
- Section 6's cell classes come from G's interval;
- the block where attention to the site rises most is recomputed over non-empty halves without IOI;
- a descriptive reading (--descriptive=1, as experiments/score/score_all.sh runs it for large_npi_fg_agr) gives the same numbers and carries
  no "holds" field;
- a comparator that is constant over a map's cells (G* - G here, since the synthetic G* equals G) is named undefined and
  skipped: Spearman's ranks average ties, so it has no correlation, where ordinal ranks would give it one by cell order.
Usage: python -I experiments/tests/inside_set_post_test.py
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
sys.path.insert(0, str(ROOT / "experiments/lib"))
from measure import boot_counts, clusters  # noqa: E402

B, SEED, H, NB = 400, 95, 4, 2
DIRECTIONS = ["das_ce_s0", "das_ce_s1", "das_ce_s2", "das_sq_T_s0", "das_sq_G_s0", "das_sq_G_s1", "das_sq_G_s2", "das_sqm_G_s0",
        "das_jsd_s0", "das_eff_s0", "u_hat", "b_hat", "bisector", "v1_mid"]
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def make_map(d, model, task, rng, share=0.05, uhat_share=0.05, null=False, follow_g=False, follow_delta=False, zero_cell=None):
    """One synthetic map (built as in experiments/tests/inside_scorer_test.py), with its run id and layers; returns T and the mirror
    clusters. share: the share of cross-entropy's effect that freezing attention removes; uhat_share: the same for u_hat;
    null: lambda_1 and DAS constant; follow_g: DAS equal to G pair by pair; zero_cell: one (layer, region) where DAS does nothing.
    follow_delta: each door cell's input change has the norm 1 + DAS's excess over T (the excess follows ||Delta||)."""
    n = 100; T = np.repeat(rng.normal(3.0, 0.3, n // 2), 2)
    cl = clusters([(f"b{i}", f"s{i}") if k == 0 else (f"s{i}", f"b{i}") for i in range(n // 2) for k in range(2)])
    C = boot_counts(cl, B, SEED); half = np.tile([True, False], n // 2)
    Z = {"T": T, "cluster": cl, "half": half, "aligned": np.ones(n, bool)}; cells = []
    for L in (1, 3, 5):
        for region in (1, 2, 3, 4):
            if region == 1 and L in (1, 3):
                cells.append(dict(layer=L, region=region, live=False, door=False)); continue
            c = f"L{L}r{region}"; j = len(cells)
            g = rng.uniform(0.2, 0.6); pred = 0.1 + 0.08 * ((j * 7) % 10)
            lam = g + pred; G = T * g; ce = T * (g + 0.9 * pred + rng.normal(0, 0.005))
            if null:
                lam = 1.0 + rng.normal(0, 0.01); ce = T * (0.95 + rng.normal(0, 0.01))
            if region in (3, 4):                                        # cells where the network carries nothing (G ~ 0)
                g = 0.0; lam = (1.0 + rng.normal(0, 0.01)) if null else rng.uniform(0.5, 2.0)
                G = rng.normal(0, 0.02, n); ce = T * (0.9 if null else 0.9 * lam) + rng.normal(0, 0.05, n)
            if follow_g:
                ce = G.copy()                                           # DAS reports the network's own interchange, pair by pair
            if (L, region) == zero_cell:
                ce = np.zeros(n)                                        # DAS does nothing here: its share through attention is undefined
            Z[f"{c}/e/own_normal"] = G; Z[f"{c}/e/own_frozen"] = G * 0.95; Z[f"{c}/e/own_row"] = G * 0.95
            for a in DIRECTIONS:
                v = ce if a.startswith(("das_ce", "das_eff")) else (G if a.startswith(("das_sqm_G", "das_sq_G")) else T)
                f_ = (1 - share) if a.startswith("das_ce") else (1 - 0.5 * share if a.startswith("das_sqm") else
                                                                (1 - uhat_share if a == "u_hat" else 0.95))
                Z[f"{c}/e/{a}_normal"] = v; Z[f"{c}/e/{a}_frozen"] = v * f_; Z[f"{c}/e/{a}_row"] = v * f_
                Z[f"{c}/fm/{a}"] = v
            Z[f"{c}/gstar_train"] = G; Z[f"{c}/gstar_test"] = G; Z[f"{c}/tr_mid"] = G; Z[f"{c}/tr_base"] = G * 1.02
            Z[f"{c}/lam_mid_boot"] = np.full(B, lam * T.mean()) * (1 + rng.normal(0, 0.01, B))
            door = L in (1, 3) and region >= 2
            if door:
                opening = 3.0 * (j % 7); sh = 0.1 + 0.1 * (j % 7)
                Z[f"{c}/door/alpha"] = np.full((n, H), 0.05); Z[f"{c}/door/a_dot_delta"] = np.zeros((n, H))
                Z[f"{c}/door/a_norm_delta_norm"] = np.full((n, H), 2.0)
                Z[f"{c}/door/delta_norm"] = (1.0 + (ce - G) / T) if follow_delta else np.full(n, 1.0 + rng.uniform())
                for nm in ("half0", "half1"):
                    Z[f"{c}/door/range_lam_{nm}"] = np.array([opening, 0.0, 0.0, 0.0]); Z[f"{c}/door/range_tr_{nm}"] = np.zeros(H)
                wb = np.full((n, NB, H), 0.05); wd = wb.copy(); wd[:, 0, 0] += 0.01 * opening
                Z[f"{c}/door/w_base"] = wb; Z[f"{c}/door/w_own"] = wb
                for k in ("das_ce_s0", "das_sq_G_s0", "das_sqm_G_s0", "das_eff_s0"):
                    Z[f"{c}/door/w_{k}"] = wd; Z[f"{c}/door/dscore_{k}"] = np.full((n, H), 0.1); Z[f"{c}/door/first_{k}"] = np.full((n, H), 0.1)
                Z[f"{c}/door/dscore_own"] = np.zeros((n, H))
                for s_ in (0, 1, 2):
                    Z[f"{c}/e/das_ce_s{s_}_frozen"] = ce * (1 - sh)
            cells.append(dict(layer=L, region=region, live=True, door=door, lambda1_mid=lam * T.mean(), lambdaD_mid=-1.0,
                              trM_mid=g * T.mean(), lambda1=lam * T.mean(), trM=g * T.mean() * 1.02))
    fz = d / f"inside_{model}_{task}.npz"; np.savez(fz, **Z)
    sha = hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16]
    J = dict(task=task, model=model, run=f"inside_{model}_{task}_fake", layers=[1, 3, 5], n_train=400, n_test=n,
             n_clusters_test=n // 2, boot=dict(B=B, seed=SEED), boot_counts_sha=sha, cells=cells,
             npz_sha256=hashlib.sha256(fz.read_bytes()).hexdigest(), full_grid=True, stack=True, frozen_no_patch_max_abs=0.0)
    json.dump(J, open(d / f"inside_{model}_{task}.json", "w"))
    return T, cl


def rows(n=50, empty2=False, reuse=False):
    """Five-region rows; region 1 differs; with empty2, region 2 is empty in half the base rows (a degenerate site); with reuse,
    rows 2k and 2k + 1 share their base text (different mirror clusters, one sentence cluster)."""
    out = []
    for i in range(n):
        b = ["<s>", f" a{i - i % 2 if reuse else i}", "" if (empty2 and i % 2 == 0) else " x", " y", " z"]
        out.append(dict(base=b, src=["<s>", f" c{i}", " x", " y", " z"], base_label=" p", src_label=" q", base_type="A"))
    return dict(train=out[:40], test=out[40:])


LIST = [("kumon", "NPIOnly", 0.6), ("kumon", "FGMatrixQ", 0.6), ("number_only", "agr_sv_num_pp__num", 0.05),
        ("number_only", "agr_refl_num_pp__num", 0.05),
        ("ravel_attributes", "ravel_city_country", 0.05), ("ravel_attributes", "ravel_city_country_json", 0.05),
        ("sentiment_templates", "sentiment_adverb", 0.05), ("sst", "sst_classification", 0.05), ("sst", "sst_continuation", 0.05),
        ("ioi_ours_label_signed", "ioi_fixed", 0.05), ("ioi_ours", "ioi_varied", 0.05)]
CG = [("npi_any_a", 0.6), ("filler_gap_b", 0.6), ("agr_c", 0.05), ("agr_c_mod", 0.05), ("agr_d", 0.05)]


def tree(null=False, follow_g=False, follow_delta=False):
    """Builds the synthetic tree (maps, rows, data, run records, readout files, the map list and logit scales), scores it with
    score_inside.py into experiments/results/inside_sets/fake.json and runs inside_set_post.py on it. Returns the tree, the
    post-processing run, its output (None if it failed) and the scorer's output."""
    r = Path(tempfile.mkdtemp()); res = r / "experiments/results/inside"; res.mkdir(parents=True)
    (r / "fake_rows").mkdir(); (r / "experiments/results/readout").mkdir(parents=True); (r / "experiments/results/inside_sets").mkdir(parents=True)
    rng = np.random.default_rng(5); lines = []; ro = {}
    for m in ("m1", "m2"):
        for t, sh in CG:
            make_map(res, m, t, rng, share=sh, null=null, follow_g=follow_g, follow_delta=follow_delta,
                     zero_cell=(5, 2) if (m, t) == ("m1", CG[0][0]) and not (null or follow_g or follow_delta) else None)
            json.dump(rows(), open(r / "fake_rows" / f"{t}.json", "w"))
            (r / "experiments/logs" / f"inside_{m}_{t}_fake").mkdir(parents=True)
            json.dump(dict(command=["python", "inside.py", t], git=dict(commit="fake")), open(r / "experiments/logs" / f"inside_{m}_{t}_fake/env.json", "w"))
        for g, t, sh in LIST:
            T, cl = make_map(res, m, t, rng, share=sh, uhat_share=0.9 if t.startswith("ioi_") else 0.05, null=null, follow_g=follow_g,
                             follow_delta=follow_delta)
            data = f"data/{m}/{g}"; dd = r / data / t; dd.mkdir(parents=True)
            R = rows(empty2=(t == "FGMatrixQ"), reuse=(t == "sentiment_adverb"))
            for s in ("train", "test"):
                json.dump(R[s], open(dd / f"{s}.json", "w"))
            sh_ = [hashlib.sha256((dd / f"{s}.json").read_bytes()).hexdigest() for s in ("train", "test")]
            lines.append("\t".join([m, g, t, data, "defined=True"] + sh_))
            ran = "data/wrong" if (m, t) == ("m2", "sentiment_adverb") else data
            (r / "experiments/logs" / f"inside_{m}_{t}_fake").mkdir(parents=True)
            json.dump(dict(command=["python", "inside.py", t, f"--data={ran}"], git=dict(commit="fake")),
                      open(r / "experiments/logs" / f"inside_{m}_{t}_fake/env.json", "w"))
            ro.setdefault((m, g), {})[f"{t}/T"] = T + {("m1", "ravel_city_country_json"): 0.01, ("m2", "ravel_city_country_json"): 2e-3}.get((m, t), 0.0)
            ro[(m, g)][f"{t}/cluster"] = cl[::-1].copy() if (m, t) == ("m1", "sst_continuation") else cl
    for (m, g), arr in ro.items():
        np.savez(r / f"experiments/results/readout/readout_{m}_m_{g}.npz", **arr)
    (r / "experiments/matrix").mkdir(parents=True)
    (r / "experiments/matrix/inside_maps.tsv").write_text("\n".join(lines) + "\n")
    u = 2.0 ** -24; g = 768 * u / (1 - 768 * u)                        # the synthetic models' logit scale: 8 gamma_D S = 5.5e-3
    json.dump(dict(models={m: dict(D=768, max_abs_label_logit=15.0, gamma_D=g, bound=g * 15.0) for m in ("m1", "m2")}),
              open(r / "experiments/results/logit_scale.json", "w"))
    s = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_inside.py"), f"--fake={res}"], capture_output=True, text=True)
    assert s.returncode == 0, s.stdout[-800:] + s.stderr[-800:]
    shutil.move(str(res / "score_inside.json"), r / "experiments/results/inside_sets/fake.json")
    p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/inside_set_post.py"), "--set=fake", f"--root={r}"],
                       capture_output=True, text=True)
    out = json.load(open(r / "experiments/results/inside_sets/fake.post.json")) if p.returncode == 0 else None
    scored = json.load(open(r / "experiments/results/inside_sets/fake.json"))
    return r, p, out, scored


r, p, out, scored = tree()
check("post-processing runs", out is not None, (p.stdout + p.stderr)[-1200:])
if out is not None:
    und = out["models"]["m1"]["counts"]["attention_share"].get("s_das_undefined")
    check("a cell where DAS's effect is exactly 0 leaves s_DAS undefined: skipped by the medians and counted",
          und == [["m1", CG[0][0], 5, 2]], str(und))
if out:
    ex = set(out["excluded_maps"])
    check("the three planted post-run failures are excluded, and nothing else",
          ex == {"m2/sentiment_adverb", "m1/ravel_city_country_json", "m1/sst_continuation"}, str(out["excluded_maps"]))
    tv = out["T_vs_readout"]
    check("T's float32-sized deviation (2e-3, within the bound 5.5e-3) is kept and reported", "m2/ravel_city_country_json" not in ex
          and 3e-4 < tv["m2/ravel_city_country_json"] < 1e-3 and tv.get("m2/ravel_city_country", 1) < 1e-12, json.dumps(tv)[:300])
    check("FGMatrixQ's region-2 cells are degenerate (three live per model)", out["degenerate_cells"] == 6
          and out["degenerate_by_map"] == {"m1/FGMatrixQ": 3, "m2/FGMatrixQ": 3}, str(out["degenerate_by_map"]))
    n_scorer = {m: len(v) for m, v in scored["cells"].items()}
    n_ex = {m: sum(k.startswith(m + "/") for k in ex) for m in ("m1", "m2")}
    check("the counts drop exactly the excluded maps' and the degenerate cells", all(
        out["models"][m]["counts"]["live_cells"] == n_scorer[m] - 10 * n_ex[m] - 3 for m in ("m1", "m2")),
        str({m: (out["models"][m]["counts"]["live_cells"], n_scorer[m]) for m in ("m1", "m2")}))
    fam = out["models"]["m1"]["families"]
    check("Kumon files under NPI and filler-gap, number-only under agreement", {"kumon/npi", "kumon/fg", "number_only/agreement"} <= set(fam)
          and min(fam["kumon/npi"]["s_das_median"], fam["kumon/fg"]["s_das_median"]) > fam["number_only/agreement"]["s_das_median"],
          json.dumps({k: v["s_das_median"] for k, v in fam.items()}))
    check("score_inside.py files Kumon under 'other' (why the mapping is needed)",
          scored["report"]["models"]["m1"]["families"]["other"]["cells"] > 0, str(scored["report"]["models"]["m1"]["families"]["other"]))
    rd = out["npi_fg_reading"]
    check("the 'more on NPI and filler-gap' reading holds on both sources", rd["causalgym"]["holds_every_model"]
          and rd["kumon_number_only"]["holds_every_model"], json.dumps(rd))
    nat = out["models"]["m1"]["counts"]["attention_share"]["natural_median_share"]["u_hat"]
    check("IOI leaves the natural-direction median and is reported apart", abs(nat - 0.05) < 1e-6
          and abs(out["ioi"]["natural_median_share"]["u_hat"] - 0.9) < 1e-6, f"{nat} {out['ioi']['natural_median_share']}")
    pw, po = out["pooled"]["with_sst"], out["pooled"]["without_sst"]
    check("RAVEL's prompts, SST's frames, IOI's tasks, number-only parents and _mod twins form the constructions",
          pw["constructions"] == 12 and po["constructions"] == 11, f"{pw['constructions']} {po['constructions']}")
    for k in ("excess_rank_ce", "excess_rank_eff", "excess_rank_door_ce", "excess_rank_door_eff", "squared_error_route", "squared_error_route_median", "squared_error_route_sqm_nonzero"):
        check(f"{k} holds pooled under both bootstraps", pw["strong"][k]["holds"], json.dumps(pw["strong"][k]))
    check("the ranking (both forms) holds pooled, for ce and eff", pw["strong"]["excess_rank"]["holds"] and pw["strong"]["excess_rank"]["eff"], json.dumps(pw["strong"]["excess_rank"]))
    check("the ranking per model too, for ce and eff", all(out["models"][m]["strong"]["excess_rank"]["holds"] and out["models"][m]["strong"]["excess_rank"]["eff"]
                                                  for m in ("m1", "m2")))
    pd = pw["excess_rank_door_parts"]["ce"]
    check("on the door cells the first-order excess beats the nine naive predictors, ||Delta|| among them",
          pd["theory"][0] > max(pd[n][0] for n in ("G", "-G", "|g|", "|g|-G", "G*-G", "u-G", "depth", "|D|", "|D|/layer")
                                if pd[n][0] is not None),
          json.dumps({k: v[0] for k, v in pd.items()}))
    pr = pw["excess_rank_parts"]["ce"]
    check("the first-order excess ranks DAS's excess better than each naive predictor",
          pr["theory"][0] > max(pr[n][0] for n in ("G", "-G", "|g|", "|g|-G", "G*-G", "u-G", "depth") if pr[n][0] is not None),
          json.dumps({k: v[0] for k, v in pr.items()}))
    check("a comparator constant over a map's cells (here G* - G, since G* = G) is named undefined, not ranked by cell order",
          all("G*-G" in v["undefined"] and "G*-G" in v["door_undefined"] for v in out["excess_rank_per_map"].values()) and pr["G*-G"][0] is None,
          json.dumps({k: (v["undefined"], v["door_undefined"]) for k, v in list(out["excess_rank_per_map"].items())[:2]}))
    check("the best naive predictor is tallied per map", sum(pw["excess_rank_best"]["ce"].values()) == pw["maps"], json.dumps(pw["excess_rank_best"]["ce"]))
    nr = out["excess_rank_without_sentence_reuse"]
    check("the template with shared base texts is named as sentence reuse and left out of the reading without it",
          nr["maps_with_reuse"] == ["m1/sentiment_adverb"] and nr["maps"] == pw["maps"] - 1, json.dumps({k: nr[k] for k in ("maps", "maps_with_reuse")}))
    cl = out["models"]["m1"]["counts"]["classes"]; cc = {k: v for k, v in out["cell_classes"].items() if k.startswith("m1/")}
    carried_reg = [v for k, v in cc.items() if k[-2:] in ("r1", "r2")]; zero_reg = [v for k, v in cc.items() if k[-2:] in ("r3", "r4")]
    check("Section 6's classes come from G's interval: every planted carried cell (regions 1-2) is carried, the zero cells (3-4) "
          "are indistinguishable from 0 but for the interval's own rate", all(v == "carried" for v in carried_reg)
          and sum(v == "zero" for v in zero_reg) >= 0.9 * len(zero_reg) and sum(cl.values()) == len(cc), json.dumps(cl))
    dm = out["excess_rank_per_map"]
    check("the ranking is reported per map with its within-map interval (above 0 on every planted map here)",
          len(dm) == pw["maps"] and all(v["ce_within_map"][0] <= v["ce"] <= v["ce_within_map"][1] + 1e-9 and v["ce_within_map"][0] > 0 for v in dm.values()),
          json.dumps(list(dm.items())[:2]))
    db = out["models"]["m1"]["door_block"]
    check("the attention block recomputed over non-empty halves, IOI left out", db["door_cells"] > 0 and db["argmax_in_block_L"] == db["door_cells"], str(db))
    check("the rank-residual form is reported and decides nothing", "holds" not in pw["strong"]["excess_rank_rank_residual"]
          and pw["strong"]["excess_rank_rank_residual"]["report_only"])
    pdsc = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/inside_set_post.py"), "--set=fake", "--descriptive=1", f"--root={r}"],
                          capture_output=True, text=True)
    od = json.load(open(r / "experiments/results/inside_sets/fake.post.json")) if pdsc.returncode == 0 else None

    def decisions(x):
        if isinstance(x, dict):
            return [k for k in x if k.startswith("holds")] + [v for y in x.values() for v in decisions(y)]
        return [v for y in x for v in decisions(y)] if isinstance(x, list) else []
    check("a descriptive reading carries the same numbers and decides nothing", od is not None and od["descriptive"]
          and not decisions(od) and od["pooled"]["with_sst"]["strong"]["excess_rank_ce"]["map_level"] == pw["strong"]["excess_rank_ce"]["map_level"]
          and od["excess_rank_per_map"] == out["excess_rank_per_map"], (pdsc.stdout + pdsc.stderr)[-600:] if od is None else str(decisions(od))[:300])
rn, pn, outn, _ = tree(null=True)
check("the ranking does not hold under the null (the top constant, DAS constant)",
      outn is not None and not outn["pooled"]["with_sst"]["strong"]["excess_rank_ce"]["holds"],
      json.dumps(outn["pooled"]["with_sst"]["strong"]["excess_rank_ce"]) if outn else (pn.stdout + pn.stderr)[-600:])
if outn:
    rf = outn["pooled"]["with_sst"]["strong"]["excess_rank_rank_residual"]["map_level"]
    check("while the rank-residual form is above 0 under the same null (why it only reports)", rf[1] is not None and rf[1] > 0, str(rf))
rg, pg, outg, _ = tree(follow_g=True)
check("the ranking is undefined when DAS reports the network's own interchange pair by pair, and the third form is negative there",
      outg is not None and not outg["pooled"]["with_sst"]["strong"]["excess_rank_ce"]["holds"] and outg["pooled"]["with_sst"]["strong"]["excess_rank_ce"]["map_level"][3] == 0
      and outg["pooled"]["with_sst"]["strong"]["excess_rank_total_ce"]["map_level"][0] < 0,
      json.dumps({k: outg["pooled"]["with_sst"]["strong"][k] for k in ("excess_rank_ce", "excess_rank_total_ce")}) if outg else (pg.stdout + pg.stderr)[-600:])
rd_, pd_, outd, _ = tree(follow_delta=True)
if outd:
    sd = outd["pooled"]["with_sst"]["strong"]
check("the ranking does not hold when DAS's excess follows ||Delta|| (the door form fails), though the seven-comparator form holds",
      outd is not None and sd["excess_rank_ce"]["holds"] and not sd["excess_rank_door_ce"]["holds"] and not sd["excess_rank"]["holds"],
      json.dumps({k: sd[k] for k in ("excess_rank_ce", "excess_rank_door_ce", "excess_rank")}) if outd else (pd_.stdout + pd_.stderr)[-600:])
for x in (r, rn, rg, rd_):
    shutil.rmtree(x)
print("\nINSIDE SET POST-PROCESSING OK" if not fails else "INSIDE SET POST-PROCESSING FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

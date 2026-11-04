"""The cross-entropy law's scorer (Appendix C.2, "Cross-entropy ends near its own optimum"; checked on synthetic outputs by
experiments/tests/ce_law_scorer_test.py). Every prediction is on v_f, the top eigenvector of M_f built from training pairs before any
training; the exact f-top is a method check only.

Population: the readout's main population in every model whose readout instruments held (experiments/results/score_readout.json). shortfall
and alignment: tasks with room, not one-directional NPI, whose cross-entropy shortfall a(ce) has a within-task cluster-bootstrap
interval (the readout's clusters and resample counts) that excludes 0. npi_contrast: Kumon & Yanaka's eight NPI constructions,
one-directional (kumon) and mirrored (kumon_npi_mirrored). A task whose own instruments in the law's run failed is left out and
listed by name (mirrored ones included). Those instruments are experiments/scripts/readout_ce_law.py's: T within an absolute 1e-3 of the
readout's, and v_z, the top eigenvector of the log odds' midpoint M, at |cos| > 0.999 from the readout's v1_mid. The readout npz
and direction file each output read are checked by sha256.
Per task, on test pairs, ratios of means; a(x) = log(e_top / e_x) (effects at or below 0.01 T floored there); ce = the three
starts' mean per row at ten passes (one pass reported).
- shortfall: R = 1 - sum|a(v_f) - a(ce)| / sum a(ce), interval above 1/2; against the constant-fraction baseline (e_ce predicted as the
  leave-one-out median ce/top times e_top), sum|a(base) - a(ce)| - sum|a(v_f) - a(ce)| with interval above 0; Spearman (average
  ranks) of a(v_f) with a(ce), interval above 0. All three: "held". Beside it, "held_without_bar": the same without R's bar of
  1/2 (the baseline comparison and the Spearman interval decide); R is reported in both.
- alignment: a_t = mean over starts |cos(off-top ce, off-top v_f)|, null n_t = mean |cos(off-top ce, off-top eff)| from the same start
  (off-top: the part orthogonal to v1_mid); tasks where the off-top norm of v_f, or of any start of ce or eff, is below 0.05 are
  left out (counted). Median (a - n), interval above 0: "held", None (untestable) when the floor leaves no task, as it does in
  every model of the paper. "without_floor": the same over the whole population without the floor, with the share of
  tasks whose alignment exceeds the null.
- npi_contrast (models of a tier together; bootstrap over the eight constructions as clusters): per (construction, model) the contrast
  Da(x) = a_one(x) - a_mirrored(x); R_c = 1 - sum|Da(v_f) - Da(ce)| / sum Da(ce), interval above 1/2 ("held"). Beside it,
  "beats_naive_without_bar": R_c(v_f) minus the best of the naive directions (the untrained mean difference u_hat, the training
  M's top eigenvector v1_mid, a no-effect direction, a constant ratio with R_c = 0) on the same resample, interval above 0.
- Tiers: the six small and the four large models, each pooled apart (the other sizes are scored per model only); shortfall, alignment and npi_contrast
  hold in a tier when they hold in every model of the tier with at least five tasks in the population (npi_contrast pooled within the
  tier); the forms without a bar or floor are summarised the same way.
- sensitivity (report): e(v_f)/e_top beside ce/e_top; shortfall with one-pass ce and with v_f^base; the f-top from each start (training CE
  against ce's, effect, |cos| with ce); tasks removed by the noise floor; shortfall's R without the noise floor and leaving one source
  out at a time; shortfall's Spearman beside u_hat's.
Bootstrap: B = 2000, numpy PCG64 seed 7, percentile intervals (tasks resampled; npi_contrast constructions).
Usage: python -I experiments/score/score_readout_ce_law.py [--fake=<dir>]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib")); sys.path.insert(0, str(ROOT / "experiments/matrix"))
import registry  # noqa: E402
from measure import boot_counts  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = Path(A["fake"]) if A.get("fake") else None
RES = FAKE or ROOT / "experiments/results/readout_ce_law"
READOUT = FAKE or ROOT / "experiments/results/readout"
SCORE = (FAKE or ROOT / "experiments/results") / "score_readout.json"
DEST = RES / "score_readout_ce_law.json"
B, SEED, FLOOR, SIN_FLOOR = 2000, 7, 0.01, 0.05


def ranks(x):
    """Average ranks (ties share the mean of their positions)."""
    x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


def boot(f, n, clusters=None):
    """f(indices) on all items and on B resamples (of items, or of clusters with all their items); percentile interval."""
    rng = np.random.default_rng(SEED)
    if clusters is None:
        idx = [rng.integers(0, n, size=n) for _ in range(B)]
    else:
        cl = np.asarray(clusters); u = np.unique(cl); members = {c: np.nonzero(cl == c)[0] for c in u}
        idx = [np.concatenate([members[c] for c in rng.choice(u, size=len(u))]) for _ in range(B)]
    r = np.array([f(i) for i in idx])
    return [float(f(np.arange(n))), float(np.nanquantile(r, 0.025)), float(np.nanquantile(r, 0.975))]


def short(top, x):
    return float(np.log(max(top, FLOOR) / max(x, FLOOR)))


def R_of(rows, key, ce="ce10", clusters=None):
    p, c = np.array([r["a"][key] for r in rows]), np.array([r["a"][ce] for r in rows])
    return boot(lambda i: 1 - np.abs(p[i] - c[i]).sum() / c[i].sum() if c[i].sum() > 0 else np.nan, len(rows), clusters)


def load(stem, folder):
    J = json.load(open(folder / f"{stem}.json")); fz = folder / f"{stem}.npz"
    if J.get("npz_sha256") != hashlib.sha256(fz.read_bytes()).hexdigest():
        raise SystemExit(f"{fz.name}: not the npz its json names")
    return J, np.load(fz)


def main():
    S = json.load(open(SCORE))
    ok_models = [m for m, v in S["report"]["models"].items() if v.get("instruments") == "OK"]
    report, cache, npi = {}, {}, []
    for m in ok_models:
        per, bad = [], []
        for r in S["rows"][m]:
            npi1 = r["group"] == "kumon" and r["task"].startswith("NPI")
            mir = r["group"] == "kumon_npi_mirrored"
            if not r["defined"] or r.get("logit_cap") == "uncapped" or not (r["main"] or mir or npi1):
                continue
            key = (m, r["group"])
            if key not in cache:
                try:
                    J, Z = load(f"readout_ce_law_{m}_m_{r['group']}", RES)
                    readout_npz, readout_dirs = READOUT / f"readout_{m}_m_{r['group']}.npz", READOUT / "dirs" / f"readout_{m}_m_{r['group']}.npz"
                    if (J["readout_sha256"] != hashlib.sha256(readout_npz.read_bytes()).hexdigest()
                            or J["readout_dirs_sha256"] != hashlib.sha256(readout_dirs.read_bytes()).hexdigest()):
                        raise SystemExit(f"{m} {r['group']}: the readout's files are not the ones the law's run read")
                    JR = json.load(open(readout_npz.with_suffix(".json")))
                    cache[key] = ((J, Z), np.load(readout_npz), JR)
                except FileNotFoundError as e:
                    cache[key] = str(e)
            if isinstance(cache[key], str):
                bad.append((r["group"], r["task"], "missing output", r["main"])); continue
            (J, Z), ZR, JR = cache[key]
            t = r["task"]; row = J["tasks"].get(t, {})
            if not row or "skipped" in row or not all(row["instruments"].values()):
                bad.append((r["group"], t, row.get("skipped") or f"instruments {row.get('instruments')}", r["main"])); continue
            Tr = ZR[f"{t}/T"]; T = Tr.mean()
            ce_rows = {p: np.mean([ZR[f"{t}/e/das_ce_p{p}_s{s}"] for s in (0, 1, 2)], axis=0) for p in (1, 10)}
            top_rows = ZR[f"{t}/e/top_train"]
            e = {f"ce{p}": float(ce_rows[p].mean() / T) for p in (1, 10)}
            e.update(top=float(top_rows.mean() / T), v_f=float(Z[f"{t}/e/v_f"].mean() / T), v_f_base=float(Z[f"{t}/e/v_f_base"].mean() / T),
                     u_hat=float(ZR[f"{t}/e/u_hat"].mean() / T), v1_mid=float(ZR[f"{t}/e/v1_mid"].mean() / T))
            C = boot_counts(ZR[f"{t}/cluster"], JR["boot"]["B"], JR["boot"]["seed"])            # the readout's own resamples
            a_ce_r = np.log(np.maximum((C @ top_rows), 1e-12) / np.maximum((C @ ce_rows[10]), 1e-12))
            noise_ok = bool(np.quantile(a_ce_r, 0.025) > 0 or np.quantile(a_ce_r, 0.975) < 0)
            al = row["align"]
            sins = [al[f"s{s}"][k] for s in (0, 1, 2) for k in ("sin_ce", "sin_eff")] + [row["sin_v_f"]]
            ft = row["f_tops"]
            per.append(dict(group=r["group"], task=t, construction=t.replace("_mirrored", ""), main=r["main"], room=r["room"], npi1=npi1,
                            mirrored=mir, source=r.get("source"), e=e, a={k: short(e["top"], e[k]) for k in ("ce1", "ce10", "v_f", "v_f_base", "u_hat", "v1_mid")},
                            noise_ok=noise_ok, alignment_ok=min(sins) >= SIN_FLOOR,
                            align=float(np.mean([al[f"s{s}"]["ce_v_f"] for s in (0, 1, 2)])),
                            null=float(np.mean([al[f"s{s}"]["ce_eff"] for s in (0, 1, 2)])),
                            f_top={k: dict(e_T=v["e_T"], train_ce=v["train_ce"], cos_ce10=float(np.mean(v["cos_ce10"])),
                                           ce_train_min=min(row["train_ce"][f"das_ce_p10_s{s}"] for s in (0, 1, 2))) for k, v in ft.items()},
                            p_b=row["median_p_b_mid"], outside=row["median_outside_mid"]))
        main_ = [x for x in per if x["main"]]
        # left-out tasks do not exclude a model: they are listed by name
        pop = [x for x in main_ if x["room"] and not x["npi1"] and x["noise_ok"]]
        out = dict(tasks=len(main_), room=sum(x["room"] for x in main_), population=len(pop),
                   removed_by_noise_floor=sum(x["room"] and not x["npi1"] and not x["noise_ok"] for x in main_),
                   left_out=bad)
        if len(pop) >= 5:
            ratio = np.array([x["e"]["ce10"] / x["e"]["top"] for x in pop])
            base_a = [short(x["e"]["top"], float(np.median(np.delete(ratio, i))) * x["e"]["top"]) for i, x in enumerate(pop)]
            pv, pc, pb = np.array([x["a"]["v_f"] for x in pop]), np.array([x["a"]["ce10"] for x in pop]), np.array(base_a)
            c1 = dict(R=R_of(pop, "v_f"),
                      beats_baseline=boot(lambda i: np.abs(pb[i] - pc[i]).sum() - np.abs(pv[i] - pc[i]).sum(), len(pop)),
                      spearman=boot(lambda i: spearman(pv[i], pc[i]), len(pop)))
            # "held" with R's bar of 1/2 and "held_without_bar" without it (where the baseline test fails, it alone decides both)
            c1["held"] = c1["R"][1] > 0.5 and c1["beats_baseline"][1] > 0 and c1["spearman"][1] > 0
            c1["held_without_bar"] = c1["beats_baseline"][1] > 0 and c1["spearman"][1] > 0
            p2 = [x for x in pop if x["alignment_ok"]]
            dd = np.array([x["align"] - x["null"] for x in p2])
            # with the 0.05 floor on the off-top norm, and (without_floor) without it over the whole population; the
            # null n_t (ce against eff from the same starts) separates alignment from noise
            dd_all = np.array([x["align"] - x["null"] for x in pop])
            alignment_no_floor = dict(tasks=len(pop), median_diff=boot(lambda i: np.median(dd_all[i]), len(dd_all)) if len(dd_all) else None,
                               share_align_above_null=float(np.mean(dd_all > 0)) if len(dd_all) else None)
            alignment_no_floor["held"] = bool(len(dd_all) and alignment_no_floor["median_diff"][1] > 0)
            c2 = dict(tasks=len(p2), removed_by_sin_floor=len(pop) - len(p2), without_floor=alignment_no_floor,
                      median_diff=boot(lambda i: np.median(dd[i]), len(dd)) if len(dd) else None,
                      median_align=float(np.median([x["align"] for x in p2])) if p2 else None,
                      median_null=float(np.median([x["null"] for x in p2])) if p2 else None)
            # with no task left by the floor the test is untestable (None), not failed
            c2["held"] = None if not len(dd) else bool(c2["median_diff"][1] > 0)
            c2["untestable"] = not len(dd)
            c4 = dict(e_over_top={k: [float(np.quantile([x["e"][k] / x["e"]["top"] for x in pop], q)) for q in (0.25, 0.5, 0.75)]
                                  for k in ("ce1", "ce10", "v_f", "v_f_base")},
                      R_ce1=R_of(pop, "v_f", "ce1"), R_v_f_base=R_of(pop, "v_f_base"),
                      R_no_noise_floor=R_of([x for x in main_ if x["room"] and not x["npi1"]], "v_f"),
                      R_leave_one_source_out={src: R_of([x for x in pop if x["source"] != src], "v_f")[0] for src in sorted({x["source"] for x in pop})},
                      spearman_u_hat=boot(lambda i: spearman(np.array([x["a"]["u_hat"] for x in pop])[i], pc[i]), len(pop)),
                      f_top={k: dict(median_e_over_ce10=float(np.median([x["f_top"][k]["e_T"] / x["e"]["ce10"] for x in pop])),
                                     median_cos_ce10=float(np.median([x["f_top"][k]["cos_ce10"] for x in pop])),
                                     train_ce_at_or_below_ce=f"{sum(x['f_top'][k]['train_ce'] <= x['f_top'][k]['ce_train_min'] + 1e-4 for x in pop)}/{len(pop)}",
                                     # the tail (reported): tasks where the optimum lies off cross-entropy DAS's direction,
                                     # and how much more effect it carries
                                     cos_below_095=int(sum(x['f_top'][k]['cos_ce10'] < 0.95 for x in pop)),
                                     p90_e_over_ce10=float(np.quantile([x['f_top'][k]['e_T'] / x['e']['ce10'] for x in pop], 0.9)),
                                     max_e_over_ce10=float(max(x['f_top'][k]['e_T'] / x['e']['ce10'] for x in pop)))
                             for k in ("from_v_f", "from_v1_mid")})
            out.update(shortfall=c1, alignment=c2, sensitivity=c4)
            print(f"{m}: population {len(pop)} (noise floor removed {out['removed_by_noise_floor']}); shortfall R {c1['R'][0]:.2f} "
                  f"[{c1['R'][1]:.2f}, {c1['R'][2]:.2f}], beats baseline {c1['beats_baseline'][0]:+.3f} [{c1['beats_baseline'][1]:+.3f}], "
                  f"Spearman {c1['spearman'][0]:.2f} [{c1['spearman'][1]:.2f}] held {c1['held']}; alignment align {c2['median_align']} vs null "
                  f"{c2['median_null']} diff {c2['median_diff']} held {c2['held']}; ce10/top {c4['e_over_top']['ce10'][1]:.3f}, v_f/top "
                  f"{c4['e_over_top']['v_f'][1]:.3f}")
        n1 = {x["construction"]: x for x in per if x["npi1"]}; nm = {x["construction"]: x for x in per if x["mirrored"]}
        npi += [(m, k, n1[k], nm[k]) for k in n1 if k in nm]
        out["per_task"] = per
        report[m] = out
    tier_of = {Path(h).name: ("small" if h in registry.SMALL else "large") for h in registry.SMALL + registry.LARGE}
    report["tiers"] = {}
    for tier in ("small", "large"):
        ms = [m for m in report if m != "tiers" and tier_of.get(m) == tier]
        if not ms:
            continue
        pairs = [x for x in npi if x[0] in ms]
        c3 = None
        if pairs:
            cl = [k for _, k, _, _ in pairs]
            dv = np.array([a_["a"]["v_f"] - b_["a"]["v_f"] for _, _, a_, b_ in pairs])
            dc = np.array([a_["a"]["ce10"] - b_["a"]["ce10"] for _, _, a_, b_ in pairs])
            du = np.array([a_["a"]["u_hat"] - b_["a"]["u_hat"] for _, _, a_, b_ in pairs])
            d1 = np.array([a_["a"]["v1_mid"] - b_["a"]["v1_mid"] for _, _, a_, b_ in pairs])
            dz = np.array([short(a_["e"]["top"], 0.0) - short(b_["e"]["top"], 0.0) for _, _, a_, b_ in pairs])   # a no-effect direction
            rcf = lambda d, i: 1 - np.abs(d[i] - dc[i]).sum() / dc[i].sum() if dc[i].sum() > 0 else np.nan  # noqa: E731
            rc = boot(lambda i: rcf(dv, i), len(pairs), clusters=cl)
            # beside "held" (R_c's bar of 1/2), the comparison without a bar: v_f against the best of the naive directions (the
            # untrained mean difference u_hat, the training M's top eigenvector v1_mid, a no-effect direction, a constant ratio,
            # whose R_c is 0) on the same resample
            rc_vs = boot(lambda i: rcf(dv, i) - max(rcf(du, i), rcf(d1, i), rcf(dz, i), 0.0), len(pairs), clusters=cl)
            c3 = dict(pairs=len(pairs), constructions=len(set(cl)), R_c=rc, R_c_u_hat=boot(lambda i: rcf(du, i), len(pairs), clusters=cl),
                      R_c_v1_mid=boot(lambda i: rcf(d1, i), len(pairs), clusters=cl),
                      R_c_no_effect=boot(lambda i: rcf(dz, i), len(pairs), clusters=cl), R_c_minus_best_naive=rc_vs,
                      held=rc[1] > 0.5, beats_naive_without_bar=rc_vs[1] > 0,
                      median_one=dict(ce10_over_top=float(np.median([a_["e"]["ce10"] / a_["e"]["top"] for _, _, a_, _ in pairs])),
                                      v_f_over_top=float(np.median([a_["e"]["v_f"] / a_["e"]["top"] for _, _, a_, _ in pairs]))))
            print(f"NPI contrast {tier} ({len(pairs)} pairs, {len(set(cl))} constructions): R_c {rc[0]:.2f} [{rc[1]:.2f}, {rc[2]:.2f}] held {c3['held']}")
        scored = [m for m in ms if "shortfall" in report[m]]          # the models with at least five tasks in the population
        report["tiers"][tier] = dict(models=ms, scored=scored,
                                     shortfall=bool(scored) and all(report[m]["shortfall"]["held"] for m in scored),
                                     alignment=(None if all(report[m]["alignment"]["held"] is None for m in scored) else
                                         bool(scored) and all(report[m]["alignment"]["held"] for m in scored
                                                                    if report[m]["alignment"]["held"] is not None)),
                                     untestable_alignment=[m for m in scored if report[m]["alignment"]["held"] is None],
                                     npi_contrast=c3, failed_shortfall=[m for m in scored if not report[m]["shortfall"]["held"]],
                                     failed_alignment=[m for m in scored if report[m]["alignment"]["held"] is False],
                                     shortfall_without_bar=bool(scored) and all(report[m]["shortfall"]["held_without_bar"] for m in scored),
                                     alignment_without_floor=bool(scored)
                                     and all(report[m]["alignment"]["without_floor"]["held"] for m in scored))
        print(f"tier {tier}: C1 {report['tiers'][tier]['shortfall']}, C2 {report['tiers'][tier]['alignment']}, NPI contrast {c3 and c3['held']} "
              f"(failed shortfall: {report['tiers'][tier]['failed_shortfall']}, C2: {report['tiers'][tier]['failed_alignment']})")
    json.dump(report, open(DEST, "w"), indent=1)
    print(f"written {DEST}")


if __name__ == "__main__":
    main()

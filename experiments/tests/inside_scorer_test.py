"""Tests experiments/score/score_inside.py: its tests of the maps inside the network (Section 6; App. D.1) and the checks that stop it.
Synthetic data only (maps with known answers; no model, no outputs of the runs); exits 1 on a failure.
The scorer's tests used here, with their output keys: a cross-entropy start above the first-order top lambda_1, the ceiling
(ceiling); the first-order excess (lambda_1 - tr M) ranking DAS's excess over the network's own interchange G (excess_rank); the share of an
effect that freezing attention removes (attention_share); the door, a prediction before training of where attention to the site opens
(door_readings); the squared error aimed at G (sqm_G) landing on G, and its route against cross-entropy's (squared_error); the counts by task
family (families).

One model, two NPI maps, one filler-gap map and one agreement map, mirrored test sets (clusters of two), two class halves. In
every map:
- two dead cells (Delta = 0: no effects stored), which must enter no test;
- live cells whose DAS excess over the network's own interchange is 0.9 x the predicted excess (lambda_1 - tr M) plus noise,
  with G unrelated to it (the ranking's strong form, the partial Spearman given G, must hold);
- one planted ceiling violation (ce start 1 at 1.5 lambda_1) in the first map only;
- an equal-share cell (ce = 2 G, both lose 50% when attention is frozen): not "more through attention";
- an attention-routed cell (ce loses 90%, G 5%): "more through attention", and sqm_G's share smaller there;
- sqm_G landing on G in every cell, ce inflated;
- door cells whose predicted opening (the score range's lambda_1 minus the median log(1/alpha)) ranks their attention share,
  with the norm of Delta and alpha unrelated (the door's strong form).
Also: under a null with lambda_1 and DAS constant and G varying, the ranking's strong form fails; a resample checksum that
does not match stops the scorer, and so do two outputs for one model and task; a map without the frozen-attention check fails
its instruments.
Usage: python -I experiments/tests/inside_scorer_test.py
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


def make_map(d, model, task, rng, plant=False, bad_sha=False, null=False, name=None, no_frozen_check=False):
    """One synthetic map in experiments/scripts/inside.py's format (layers 1, 3, 5 x regions 1-4); the flags plant the cases above
    (name: a second output for the same model and task)."""
    n = 100; T = np.repeat(rng.normal(3.0, 0.3, n // 2), 2)
    cl = clusters([(f"b{i}", f"s{i}") if k == 0 else (f"s{i}", f"b{i}") for i in range(n // 2) for k in range(2)])
    C = boot_counts(cl, B, SEED); half = np.tile([True, False], n // 2)
    Z = {"T": T, "cluster": cl, "half": half, "aligned": np.ones(n, bool)}; cells = []
    for L in (1, 3, 5):
        for region in (1, 2, 3, 4):
            dead = region == 1 and L in (1, 3)
            if dead:
                cells.append(dict(layer=L, region=region, live=False, door=False)); continue
            c = f"L{L}r{region}"; j = len(cells)
            g = rng.uniform(0.2, 0.6)                                   # the own interchange, unrelated to the excess
            pred = 0.1 + 0.08 * ((j * 7) % 10)                            # predicted excess, units of T
            lam = g + pred; G = T * g; ce = T * (g + 0.9 * pred + rng.normal(0, 0.005))
            if null:                                                      # lambda_1 and DAS constant, G varying: no information
                lam = 1.0 + rng.normal(0, 0.01); ce = T * (0.95 + rng.normal(0, 0.01))
            eq = (L, region) == (5, 2); routed = (L, region) == (5, 3)
            if eq:
                ce = 2 * G; lam = max(lam, 2.5 * g)                     # the top above 2 G: no second ceiling violation
            fr = {"ce": 0.1 if routed else (0.5 if eq else 0.95), "own": 0.5 if eq else 0.95}
            Z[f"{c}/e/own_normal"] = G; Z[f"{c}/e/own_frozen"] = G * fr["own"]; Z[f"{c}/e/own_row"] = G * fr["own"]
            for a in DIRECTIONS:
                v = ce if a.startswith(("das_ce", "das_eff")) else (G if a.startswith(("das_sqm_G", "das_sq_G")) else T)
                if plant and c == "L3r2" and a == "das_ce_s1":
                    v = T * lam * 1.5
                f_ = fr["ce"] if a.startswith("das_ce") else (0.97 if a.startswith("das_sqm") else 0.95)
                Z[f"{c}/e/{a}_normal"] = v; Z[f"{c}/e/{a}_frozen"] = v * f_; Z[f"{c}/e/{a}_row"] = v * f_
                Z[f"{c}/fm/{a}"] = v
            Z[f"{c}/gstar_train"] = G; Z[f"{c}/gstar_test"] = G; Z[f"{c}/tr_mid"] = G; Z[f"{c}/tr_base"] = G * 1.02
            Z[f"{c}/lam_mid_boot"] = np.full(B, lam * T.mean())
            door = L in (1, 3) and region >= 2
            if door:
                share = 0.1 + 0.1 * (j % 7)                               # the attention share, ranked by the opening
                opening = 3.0 * (j % 7)
                Z[f"{c}/door/alpha"] = np.full((n, H), 0.05); Z[f"{c}/door/a_dot_delta"] = np.zeros((n, H))
                Z[f"{c}/door/a_norm_delta_norm"] = np.full((n, H), 2.0); Z[f"{c}/door/delta_norm"] = np.full(n, 1.0 + rng.uniform())
                for nm in ("half0", "half1"):
                    Z[f"{c}/door/range_lam_{nm}"] = np.array([opening, 0.0, 0.0, 0.0]); Z[f"{c}/door/range_tr_{nm}"] = np.zeros(H)
                wb = np.full((n, NB, H), 0.05); wd = wb.copy(); wd[:, 0, 0] += 0.01 * opening
                Z[f"{c}/door/w_base"] = wb; Z[f"{c}/door/w_own"] = wb
                for k in ("das_ce_s0", "das_sq_G_s0", "das_sqm_G_s0", "das_eff_s0"):
                    Z[f"{c}/door/w_{k}"] = wd; Z[f"{c}/door/dscore_{k}"] = np.full((n, H), 0.1); Z[f"{c}/door/first_{k}"] = np.full((n, H), 0.1)
                Z[f"{c}/door/dscore_own"] = np.zeros((n, H))
                Z[f"{c}/e/das_ce_s0_frozen"] = ce * (1 - share); Z[f"{c}/e/das_ce_s1_frozen"] = ce * (1 - share); Z[f"{c}/e/das_ce_s2_frozen"] = ce * (1 - share)
            cells.append(dict(layer=L, region=region, live=True, door=door, lambda1_mid=lam * T.mean(), lambdaD_mid=-1.0,
                              trM_mid=g * T.mean(), lambda1=lam * T.mean(), trM=g * T.mean() * 1.02))
    fz = d / f"inside_{model}_{task}{'_' + name if name else ''}.npz"; np.savez(fz, **Z)
    sha = hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16]
    J = dict(task=task, n_train=400, n_test=n, n_clusters_test=n // 2, boot=dict(B=B, seed=SEED), boot_counts_sha="0" * 16 if bad_sha else sha,
             cells=cells, npz_sha256=hashlib.sha256(fz.read_bytes()).hexdigest(), full_grid=True, stack=True)
    if not no_frozen_check:
        J["frozen_no_patch_max_abs"] = 0.0
    json.dump(J, open(d / f"inside_{model}_{task}{'_' + name if name else ''}.json", "w"))


def run(d):
    r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_inside.py"), f"--fake={d}"], capture_output=True, text=True)
    return r, (json.load(open(d / "score_inside.json")) if (d / "score_inside.json").is_file() else None)


rng = np.random.default_rng(5)
d = Path(tempfile.mkdtemp())
for i, t in enumerate(("npi_any_a", "npi_any_b", "filler_gap_c", "agr_d")):
    make_map(d, "m1", t, rng, plant=(i == 0))
r, out = run(d)
check("scorer runs", r.returncode == 0 and out is not None, (r.stdout + r.stderr)[-600:])
if out:
    M = out["report"]["models"]["m1"]; P = out["report"]["pooled"]; cs = out["cells"]["m1"]
    check("dead cells enter no test", M["live_cells"] == 4 * 10 and all(c["cell"] not in ("L1r1", "L3r1") for c in cs), str(M["live_cells"]))
    check("the planted ceiling violation is the only one", M["ceiling"]["cells_above"] == 1, str(M["ceiling"]))
    check("the ranking's strong form (partial Spearman given G) holds when the prediction carries information", P["excess_rank_meas_ce"]["strong_form"],
          json.dumps(P["excess_rank_meas_ce"]))
    eqc = [c for c in cs if c["cell"] == "L5r2"]; rtc = [c for c in cs if c["cell"] == "L5r3"]
    check("an equal-share cell is not 'more through attention'", all(c["attention_share"]["das_more_through_attention"] is False for c in eqc),
          str([c["attention_share"].get("diff") for c in eqc]))
    check("an attention-routed cell is", all(c["attention_share"]["das_more_through_attention"] for c in rtc), str([c["attention_share"].get("diff") for c in rtc]))
    check("sqm_G lands on G in its whole population, ce never", M["squared_error"]["land_contains0"]["sqm"] == M["squared_error"]["population"] > 0
          and M["squared_error"]["land_contains0"]["ce"] == 0, str(M["squared_error"]))
    check("sqm_G's attention share is below ce's (same start), over every cell", P["squared_error_route"][4], str(P["squared_error_route"]))
    check("the door's training-free predictor ranks the attention share, given G and against the best naive one", P["door_predictor"]["strong_form"],
          json.dumps(P["door_predictor"]))
    check("the counts split the families", M["families"]["npi/fg"]["cells"] == 30 and M["families"]["agreement"]["cells"] == 10, str(M["families"]))
dn = Path(tempfile.mkdtemp())
for t in ("npi_any_a", "npi_any_b", "filler_gap_c", "agr_d"):
    make_map(dn, "m1", t, rng, null=True)
rn, outn = run(dn)
check("the ranking's strong form fails under a null with no information (lambda_1 and DAS constant, G varying)",
      outn is not None and not outn["report"]["pooled"]["excess_rank_meas_ce"]["strong_form"], json.dumps(outn["report"]["pooled"]["excess_rank_meas_ce"]) if outn else rn.stderr[-300:])
d2 = Path(tempfile.mkdtemp()); make_map(d2, "m1", "npi_any_a", rng, bad_sha=True)
r2, _ = run(d2)
check("a resample checksum that does not match stops the scorer", r2.returncode != 0 and "checksum" in (r2.stdout + r2.stderr))
d3 = Path(tempfile.mkdtemp()); make_map(d3, "m1", "npi_any_a", rng); make_map(d3, "m1", "npi_any_a", rng, name="second")
r3, _ = run(d3)
check("two outputs for one model and task stop the scorer", r3.returncode == 2 and "two outputs" in r3.stdout, r3.stdout[-200:])
d4 = Path(tempfile.mkdtemp()); make_map(d4, "m1", "npi_any_a", rng, no_frozen_check=True)
_, out4 = run(d4)
check("a missing frozen-attention check fails the map's instruments", out4 is not None and out4["report"]["maps"]["m1/npi_any_a"]["instruments"] == "FAILED")
for x in (d, dn, d2, d3, d4):
    shutil.rmtree(x)
print("\nINSIDE SCORER OK" if not fails else "INSIDE SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

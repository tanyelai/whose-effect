"""Tests experiments/lib/measure.py on cases with known answers. Synthetic data only (no model, no outputs of the runs); exits 1 on a
failure.
measure.py holds the arithmetic the readout and its scorers share: mirror clusters, cluster resampling, ratios of means,
intervals, p-values, false-discovery control, lambda_1 of M on each resample and the optimiser on the unit sphere (Section 4.4;
App. C.1).

- clusters pairs each row with its mirror (duplicates with duplicates) and leaves unmirrored rows alone;
- boot_counts resamples clusters: mirror partners always carry the same count, each resample draws as many clusters as
  exist, and on equal partners the interval is wider than the row bootstrap's (by about the square root of 2);
- lam_boot against a direct eigendecomposition, differences against the spectrum's scale (full row space, a smaller row
  space with the complement's zero, mirrored and so rank-deficient rows, a negative semidefinite M whose top is the
  complement's 0);
- sphere_opt finds a quadratic form's top, stacked starts each solve their own problem, and the best iterate is kept;
- boot_ratio, interval (percentile and basic), p_above and bh on cases with known answers.
Usage: python -I experiments/tests/measure_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from measure import bh, boot_counts, boot_ratio, clusters, interval, lam_boot, m_sym, p_above, sphere_opt  # noqa: E402

torch.manual_seed(0); fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


keys = [("a", "b"), ("c", "d"), ("b", "a"), ("e", "f"), ("d", "c"), ("a", "b"), ("b", "a"), ("g", "h")]
cid = clusters(keys)
check("clusters pairs mirrors, duplicates with duplicates", list(cid) == [0, 1, 0, 3, 1, 5, 5, 7], str(list(cid)))
C = boot_counts(cid, 500, 3)
check("boot_counts: partners carry the same count", np.all(C[:, 0] == C[:, 2]) and np.all(C[:, 1] == C[:, 4]) and np.all(C[:, 5] == C[:, 6]))
first = [list(cid).index(c) for c in sorted(set(cid))]
check("boot_counts: each resample draws as many clusters as exist", np.all(C[:, first].sum(1) == len(first)))
rng = np.random.default_rng(0); base = rng.normal(1.0, 1.0, 50); x = np.repeat(base, 2)          # 50 mirrored pairs, equal partners
cm = clusters([(f"b{i}", f"s{i}") if k == 0 else (f"s{i}", f"b{i}") for i in range(50) for k in range(2)])
r_cl = boot_ratio(x, np.ones(100), boot_counts(cm, 2000, 1)); r_row = boot_ratio(x, np.ones(100), boot_counts(np.arange(100), 2000, 1))
ratio = np.std(r_cl) / np.std(r_row)
check("cluster bootstrap is wider than the row bootstrap on equal partners (about sqrt 2)", 1.25 < ratio < 1.6, f"({ratio:.2f})")


def direct(W, D, c):
    M = 0.5 * (W.T @ (c[:, None] * D) + D.T @ (c[:, None] * W)) / c.sum()
    return float(torch.linalg.eigvalsh(M)[-1])


for name, n, dim, make in (("full row space", 40, 30, None), ("smaller row space", 10, 50, None),
                           ("mirrored rows", 12, 60, "mirror"), ("negative semidefinite", 8, 40, "neg")):
    W = torch.randn(n, dim, dtype=torch.float64); D = torch.randn(n, dim, dtype=torch.float64)
    if make == "mirror":
        D = torch.cat([D[: n // 2], -D[: n // 2]]); W = torch.cat([W[: n // 2], W[: n // 2]])
    if make == "neg":
        W = -D
    part = np.array([i if i % 3 else max(i - 1, 0) for i in range(n)])   # clusters of unequal size: the row total varies
    Cn = torch.tensor(boot_counts(part, 50, 7), dtype=torch.float64)
    got = lam_boot(W, D, torch.cat([torch.ones(1, n, dtype=torch.float64), Cn]))
    want = [direct(W, D, torch.ones(n, dtype=torch.float64))] + [direct(W, D, c) for c in Cn]
    scale = float((W.norm(dim=1) * D.norm(dim=1)).mean())          # the spectrum's scale: a true top of 0 is compared to it
    err = max(abs(float(a) - b) for a, b in zip(got, want)) / scale
    check(f"lam_boot, {name}", err < 1e-10, f"(max difference / scale {err:.1e})")
    if make is None:
        check(f"m_sym, {name}", abs(float(torch.linalg.eigvalsh(m_sym(W, D))[-1]) - want[0]) < 1e-9)

A = torch.randn(30, 30, dtype=torch.float64); M = 0.5 * (A + A.T); top = float(torch.linalg.eigvalsh(M)[-1])
_, f, _ = sphere_opt(lambda D: -((D @ M) * D).sum(1), [("random", torch.randn(30, dtype=torch.float64))], 3000, 1e-2)
check("sphere_opt finds the top of a quadratic form", abs(-f - top) < 1e-4 * abs(top), f"({-f:.6f} vs {top:.6f})")
A2 = torch.randn(30, 30, dtype=torch.float64); M2 = 0.5 * (A2 + A2.T); top2 = float(torch.linalg.eigvalsh(M2)[-1])
starts = [("a", torch.randn(30, dtype=torch.float64)), ("b", torch.randn(30, dtype=torch.float64))]   # two problems at once
_, _, recs = sphere_opt(lambda D: -torch.stack([D[0] @ M @ D[0], D[1] @ M2 @ D[1]]), starts, 3000, 1e-2)
check("sphere_opt: stacked starts each solve their own problem",
      abs(-recs[0]["loss"] - top) < 1e-4 * abs(top) and abs(-recs[1]["loss"] - top2) < 1e-4 * abs(top2),
      f"({-recs[0]['loss']:.5f} vs {top:.5f}; {-recs[1]['loss']:.5f} vs {top2:.5f})")
v1 = torch.linalg.eigh(M)[1][:, -1]
_, f, _ = sphere_opt(lambda D: -((D @ M) * D).sum(1), [("at the top", v1)], 200, 5e-1)   # a step size that wanders off
check("sphere_opt keeps the best iterate (a start at the optimum is never lost)", -f >= top - 1e-12, f"({-f:.6f} vs {top:.6f})")

rng = np.random.default_rng(1); num = rng.normal(2.0, 0.1, 400); den = rng.normal(1.0, 0.05, 400)
Cr = boot_counts(np.arange(400), 2000, 95); r = boot_ratio(num, den, Cr); pt = num.mean() / den.mean()
lo, hi = interval(pt, r)
check("percentile interval covers the ratio of means", lo < pt < hi and hi - lo < 0.1, f"[{lo:.3f}, {hi:.3f}]")
blo, bhi = interval(pt, r, basic=True)
check("basic interval reflects the percentile one about the point", abs((blo + hi) - 2 * pt) < 1e-12 and abs((bhi + lo) - 2 * pt) < 1e-12)
check("p_above is small far above the bound", p_above(pt, r, 1.0) < 1e-6)
check("p_above is large below the bound", p_above(pt, r, 3.0) > 1 - 1e-6)
r2 = boot_ratio(num, None, Cr, den_boot=Cr @ den / Cr.sum(1))
check("boot_ratio with a resampled denominator equals the plain one", np.allclose(r2, r, rtol=0, atol=1e-12))
check("bh rejects the expected set", list(bh([0.001, 0.02, 0.03, 0.5], 0.05)) == [True, True, True, False])
check("bh rejects nothing when nothing is small", not bh([0.2, 0.5, 0.9]).any())
check("boot_counts is reproducible", np.array_equal(boot_counts(cid, 100, 9), boot_counts(cid, 100, 9)))
print("\nMEASURE OK" if not fails else "MEASURE FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

"""Tests experiments/lib/rankmath.py on cases with known answers. Synthetic data only (no model, no outputs of the runs); exits 1 on a
failure.
rankmath.py is the arithmetic of the rank sweep at the readout (experiments/scripts/readout_rank.py; App. A.3): Ky Fan sums K_k of M on
each resample, rank-k bases and patched states, first-order values and subspace overlaps. Checks:
- the resampled Ky Fan sums equal a brute-force eigendecomposition of each resampled M (when [W; Delta] spans less than the
  whole space and when it spans all of it), at every rank including D - 1 and D;
- the Ky Fan principle: random subspaces stay at or below K_k, the top-k eigenvectors attain it (at D - 1, the complement of
  the bottom eigenvector);
- the patched states match their formulas at k = 1, 5 and D - 1, and the whole space gives h_s;
- with a linear readout the exact effect of every rank-k interchange equals its first-order value;
- bases are orthonormal;
- under one shared direction (w_i = c_i w-bar) every K_k with k < D equals lambda_1 and K_D equals tr M (App. A.3: a larger
  rank does not raise the top);
- the overlap of a subspace with itself is 1 and with an orthogonal subspace 0.
Usage: python -I experiments/tests/rankmath_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
from rankmath import basis, first_order, ksum, kyfan_boot, overlap, patched  # noqa: E402
from measure import boot_counts  # noqa: E402

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def msym(W, Dl, c=None):
    c = torch.ones(W.shape[0], dtype=torch.float64) if c is None else c
    K = (W.double().T * c) @ Dl.double() / c.sum()
    return 0.5 * (K + K.T)


for D, n in ((40, 12), (16, 30)):                                     # [W; Delta] spans 24 < D dimensions; then all D = 16
    W, Dl = torch.randn(n, D, dtype=torch.float64), torch.randn(n, D, dtype=torch.float64)
    cl = np.repeat(np.arange(n // 2), 2)
    C = torch.tensor(boot_counts(cl, 50, 3), dtype=torch.float64)
    ranks = [1, 2, 4, D - 1, D]
    kb = kyfan_boot(W, Dl, C, ranks, D)
    brute = torch.stack([torch.stack([torch.linalg.eigvalsh(msym(W, Dl, c)).flip(0)[:k].sum() for k in ranks]) for c in C])
    check(f"resampled Ky Fan sums equal brute force (D={D}, n={n})", float((kb - brute).abs().max()) < 1e-10,
          f"{float((kb - brute).abs().max()):.1e}")
    M = msym(W, Dl); ev, U = torch.linalg.eigh(M)
    one = kyfan_boot(W, Dl, torch.ones(1, n, dtype=torch.float64), ranks, D)[0]
    check(f"at unit counts they equal the direct sums (D={D})", max(abs(float(one[j]) - (ksum(ev, k, D) if k < D else float(ev.sum())))
                                                               for j, k in enumerate(ranks)) < 1e-10)
    ok = True
    for k in (1, 2, 4, D - 2):
        top = float(torch.trace(U[:, -k:].T @ M @ U[:, -k:]))
        rnd = max(float(torch.trace(Q.T @ M @ Q)) for Q in (torch.linalg.qr(torch.randn(D, k, dtype=torch.float64))[0] for _ in range(200)))
        ok &= abs(top - ksum(ev, k, D)) < 1e-10 and rnd <= ksum(ev, k, D) + 1e-10
    u = U[:, 0]; topD1 = float(torch.trace(M) - u @ M @ u)
    rndD1 = max(float(torch.trace(M) - v @ M @ v) for v in (x / x.norm() for x in torch.randn(200, D, dtype=torch.float64)))
    ok &= abs(topD1 - ksum(ev, D - 1, D)) < 1e-10 and rndD1 <= ksum(ev, D - 1, D) + 1e-10
    check(f"Ky Fan: random subspaces stay at or below K_k, the eigenvectors attain it (D={D})", ok)

D, n, Kn = 24, 10, 3
hb, hs = torch.randn(Kn, n, D, dtype=torch.float64), torch.randn(Kn, n, D, dtype=torch.float64)
V = torch.randn(Kn, D, 5, dtype=torch.float64); P = basis(V, 5, D)
check("QR bases are orthonormal", float((P.transpose(1, 2) @ P - torch.eye(5, dtype=torch.float64)).abs().max()) < 1e-12)
P1 = basis(torch.randn(Kn, D, 1, dtype=torch.float64), 1, D); u = basis(torch.randn(Kn, D, dtype=torch.float64), D - 1, D)
check("rank-one and D - 1 bases are unit vectors", float((P1.norm(dim=1) - 1).abs().max()) < 1e-12 and float((u.norm(dim=1) - 1).abs().max()) < 1e-12)
Dl = hs - hb
f5 = hb + torch.einsum("kbd,kdj,kej->kbe", Dl, P, P)
f1 = hb + (Dl @ P1)[..., :1] * P1.transpose(1, 2)
fD1 = hb + Dl - (Dl * u[:, None, :]).sum(-1, keepdim=True) * u[:, None, :]
Pfull = torch.linalg.qr(torch.randn(Kn, D, D, dtype=torch.float64))[0]
check("patched states match their formulas (k = 1, 5, D - 1) and the whole space gives h_s",
      max(float((patched(hb, hs, P, 5, D) - f5).abs().max()), float((patched(hb, hs, P1, 1, D) - f1).abs().max()),
          float((patched(hb, hs, u, D - 1, D) - fD1).abs().max()), float((patched(hb, hs, Pfull, D, D) - hs).abs().max())) < 1e-10)
A = torch.randn(7, D, dtype=torch.float64); yb, ys = torch.randint(0, 7, (n,)), torch.randint(0, 7, (n,))
g = lambda H: ((A[ys] - A[yb]) * H).sum(-1)  # noqa: E731
w = (A[ys] - A[yb]); ok = True
for k, B_ in ((1, P1), (5, P), (D - 1, u)):
    for j in range(Kn):
        exact = g(patched(hb[j:j + 1], hs[j:j + 1], B_[j:j + 1], k, D)[0]) - g(hb[j])
        ok &= float((exact - first_order(w, Dl[j], B_[j], k, D)).abs().max()) < 1e-10
check("with a linear readout the exact effect equals the first-order value at k = 1, 5, D - 1", ok)

D, n = 30, 40
wbar = torch.randn(D, dtype=torch.float64); c = torch.rand(n, dtype=torch.float64) + 0.5
Wsh, Dsh = c[:, None] * wbar, torch.randn(n, D, dtype=torch.float64) + 2 * wbar
evs = torch.linalg.eigvalsh(msym(Wsh, Dsh))
m = (c[:, None] * Dsh).mean(0); lam1 = 0.5 * (float(wbar @ m) + float(wbar.norm() * m.norm()))
ks = [1, 2, 4, 16, D - 1]
kbs = kyfan_boot(Wsh, Dsh, torch.ones(1, n, dtype=torch.float64), ks + [D], D)[0]
check("one shared direction: K_k = lambda_1 for every k < D and K_D = tr M = w-bar . m",
      max(abs(ksum(evs, k, D) - lam1) for k in ks) < 1e-9 and max(abs(float(kbs[j]) - lam1) for j in range(len(ks))) < 1e-9
      and abs(float(kbs[-1]) - float(wbar @ m)) < 1e-9, f"lambda_1 {lam1:.4f}, K {[round(ksum(evs, k, D), 4) for k in ks]}")
Uq = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64))[0]
check("overlap: a subspace with itself 1, with its orthogonal complement 0",
      abs(overlap(Uq[:, :4], Uq[:, :4], 4, D) - 1) < 1e-12 and abs(overlap(Uq[:, 4:8], Uq[:, :4], 4, D)) < 1e-12
      and abs(overlap(Uq[:, 0], Uq[:, 0], D - 1, D) - 1) < 1e-12)
print("\nRANK MATH OK" if not fails else "RANK MATH FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

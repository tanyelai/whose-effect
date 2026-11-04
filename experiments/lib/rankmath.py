"""The arithmetic of the rank sweep at the readout (experiments/scripts/readout_rank.py), checked on synthetic data by
experiments/tests/rankmath_test.py (no model).

A rank-k interchange replaces h_b by h_b + P P^T (h_s - h_b), P with k orthonormal columns; at k = D - 1 it is
h_b + (I - u u^T)(h_s - h_b), u a unit vector. Functions take stacked bases (K of them) where the script trains or measures
several at once. torch throughout.
"""
import torch


def basis(V, k, D):
    """Orthonormal bases of stacked parameters V: (K, D, k) -> (K, D, k), the QR basis; k = 1 normalises the column
    (readout.py's arithmetic for rank one); k = D - 1: V (K, D) -> unit vectors u."""
    if k == D - 1 or k == 1:
        return V / V.norm(dim=1, keepdim=True)
    return torch.linalg.qr(V)[0]


def patched(hb, hs, P, k, D):
    """The patched states for stacked bases: hb, hs (K, b, D), P (K, D, k) (k = D - 1: (K, D)). Returns (K, b, D)."""
    Dl = hs - hb
    if k == D - 1:
        return hb + Dl - (Dl * P[:, None, :]).sum(-1, keepdim=True) * P[:, None, :]
    if k == 1:
        return hb + (Dl * P[:, None, :, 0]).sum(-1, keepdim=True) * P[:, None, :, 0]
    return hb + torch.bmm(torch.bmm(Dl, P), P.transpose(1, 2))


def first_order(W, Dl, P, k, D):
    """w_i^T P P^T Delta_i for every pair i, for one basis P (D, k) (k = D - 1: a unit vector u), in float64."""
    W, Dl, P = W.double(), Dl.double(), P.double()
    if k == D - 1:
        return (W * Dl).sum(1) - (W @ P) * (Dl @ P)
    return ((W @ P) * (Dl @ P)).sum(1)


def ksum(ev, k, D):
    """The Ky Fan sum: the k largest of the D eigenvalues ev (ascending, all D of them) summed; the first-order rank-k top."""
    return float(ev.sum() - ev[0]) if k == D - 1 else float(ev[-k:].sum())


def kyfan_boot(W, Dl, counts, ranks, D):
    """K_k of sym(sum_i c_i w_i Delta_i^T) / sum_i c_i for each row of counts (float64, resamples x n) and each k in ranks
    (resamples x ranks). Exact: the matrix vanishes off the row space of [W; Delta], so its D eigenvalues are those of the
    compressed matrix Q^T M Q (Q an orthonormal basis of that space, r columns) and D - r zeros."""
    X = torch.cat([W, Dl]).double(); Q = torch.linalg.qr(X.T)[0]; r = Q.shape[1]
    A, Bm = W.double() @ Q, Dl.double() @ Q
    idx = torch.tensor([k - 1 for k in ranks], device=W.device); out = []
    for c in torch.split(counts, 100):
        K = torch.einsum("bi,ij,ik->bjk", c, A, Bm) / c.sum(1)[:, None, None]
        ev = torch.linalg.eigvalsh(0.5 * (K + K.transpose(1, 2)))
        allev = torch.cat([ev, ev.new_zeros(ev.shape[0], D - r)], 1).sort(dim=1, descending=True).values
        out.append(allev.cumsum(1)[:, idx])
    return torch.cat(out)


def overlap(P, U, k, D):
    """How much of the subspace U_k (D, k) a basis P (D, k) spans, |P^T U_k|_F^2 / k (1: the same subspace); k = D - 1:
    (u^T u_D)^2 for the removed directions."""
    if k == D - 1:
        return float((P @ U) ** 2)
    return float((P.T @ U).pow(2).sum()) / k

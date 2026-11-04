"""Resampling, intervals and spectra for OwnEffect and the paper's scorers (experiments/score). Its clusters, resamples, ratios,
intervals, p-values and false-discovery counts are those of experiments/lib/measure.py, the arithmetic of the readout and inside-map
measurements; experiments/tests/owneffect_stats_identity_test.py checks that they agree bit for bit.

Intervals resample clusters: a pair and its mirror (base and source exchanged) form one cluster, every other pair its own.
A ratio of means is computed on the same resample. Intervals are percentile intervals, except for lambda_1, which takes the
basic (reverse percentile) interval: lambda_1 is convex in M, so resampling pushes it up. spectrum() gives M's trace and
extreme eigenvalues exactly, in the span of the gradients and differences, where M's nonzero spectrum lies.
readings() turns a cell's intervals into the paper's readings. For the scorers: p_above() and bh() test many cells with false-discovery control; block_counts() resamples rows in circular
blocks, for rows whose neighbours share an input; selected_bounds() and selected_band() report the best of several layers or
cells with the choice counted.
"""
import numpy as np


def clusters(keys):
    """Cluster ids for a list of (base, source) keys: a row and its mirror, the row whose key is (source, base), form one
    cluster, every other row its own. Rows pair greedily in order, so each duplicate of a row pairs with a duplicate of its
    mirror while one is left. Returns an int array (n,): the index of the cluster's first row."""
    cid, free = [-1] * len(keys), {}
    for i, (b, s) in enumerate(keys):
        j = free.get((s, b), [])
        if j:
            cid[i] = cid[j.pop(0)]
        else:
            cid[i] = i; free.setdefault((b, s), []).append(i)
    return np.array(cid)


def boot_counts(cluster, B=2000, seed=95):
    """Row counts (B, n) of a cluster bootstrap with numpy's default_rng(seed): each resample draws as many clusters as there
    are, with replacement, and a row counts as often as its cluster was drawn. B must be at least 1."""
    u, inv = np.unique(cluster, return_inverse=True)
    draw = np.random.default_rng(seed).integers(0, len(u), size=(B, len(u)))
    per = np.stack([np.bincount(r, minlength=len(u)) for r in draw])
    return per[:, inv]


def boot_ratio(num, den, C, num_boot=None, den_boot=None):
    """Resampled values of mean(num) / mean(den) under row counts C (B, n), each mean weighted by the counts of the same
    resample; either side may be passed already resampled (num_boot, den_boot, shape (B,))."""
    w = C.sum(1)
    nb = C @ num / w if num_boot is None else num_boot
    db = C @ den / w if den_boot is None else den_boot
    return nb / db


def interval(point, r, level=0.95, basic=False):
    """The two-sided interval at `level` from the resampled values r: the percentile interval (the quantiles of r), or with
    basic=True the basic interval (2 * point - the upper quantile, 2 * point - the lower one). Returns (low, high)."""
    a = (1 - level) / 2; lo, hi = float(np.quantile(r, a)), float(np.quantile(r, 1 - a))
    return (2 * point - hi, 2 * point - lo) if basic else (lo, hi)


def readings(own, effect, excess, ratio=None):
    """The readings of a cell inside the network from intervals (low, high): own, G*'s; effect, e's; excess, that of
    e - G*; ratio, that of e / G*, given where G*'s interval lies above zero. Returns a dict:
      carried: G*'s interval lies above zero (the network's own interchange carries the cell's effect);
      excess: the interval of e - G* lies above zero (the count of Section 6: an effect the network's own interchange does
        not produce);
      not_evidence_of_use: G*'s interval does not lie above zero and e's does (the conclusion's third step: e is then no
        evidence that the network uses the site);
      off_route: not_evidence_of_use and excess both (Algorithm 1's off-route cell);
      reading: the flow of the paper's figure on what to report: where carried, "below", "within" or "excess" as the
        interval of e / G* lies below one, holds one or lies above it; else "off-route" or "no evidence of use"; None where
        neither G*'s interval nor e's lies above zero."""
    carried = own[0] > 0
    out = dict(carried=carried, excess=excess[0] > 0, not_evidence_of_use=not carried and effect[0] > 0)
    out["off_route"] = out["not_evidence_of_use"] and out["excess"]
    if carried:
        out["reading"] = "excess" if ratio[0] > 1 else "below" if ratio[1] < 1 else "within"
    else:
        out["reading"] = "off-route" if out["off_route"] else "no evidence of use" if out["not_evidence_of_use"] else None
    return out


def p_above(point, r, bound=1.0):
    """One-sided p-value for "the value lies above bound", from a normal approximation: 1 - Phi((point - bound) / se), with
    se the standard deviation of the resampled values r (0 or 1 when se is zero)."""
    from math import erf, sqrt
    se = float(np.std(r, ddof=1))
    if se == 0:
        return 0.0 if point > bound else 1.0
    z = (point - bound) / se
    return 0.5 * (1 - erf(z / sqrt(2)))


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: a boolean array, which of the p-values are rejected at false-discovery rate q."""
    p = np.asarray(pvals, dtype=float); m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p); ok = p[order] <= q * (np.arange(1, m + 1) / m)
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool); rej[order[:k]] = True
    return rej


def block_counts(n, block, B=10000, seed=0):
    """Row counts (B, n) of a circular block bootstrap: each resample draws ceil(n / block) starting rows and takes `block`
    consecutive rows from each, wrapping around, keeping the first n rows drawn (block 1 is the row bootstrap). For rows
    whose neighbours share an input."""
    rng = np.random.default_rng(seed); k = int(np.ceil(n / block))
    idx = (rng.integers(0, n, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n] % n
    return np.stack([np.bincount(r, minlength=n) for r in idx])


def selected_bounds(Y, Yb, units, level=0.95):
    """Lower and upper bounds at every candidate (layer, cell) at once, a two-sided interval at `level` for each, for
    reporting the best one with its choice counted: Bonferroni over the L candidates, each bound at the Student-t quantile
    1 - (1 - level) / (2 L) with units - 1 degrees of freedom, times the resampling's standard error scaled by
    sqrt(units / (units - 1)). units is the number of independent units resampled (clusters, or blocks). Y (L,) holds the
    point values, Yb (B, L) the resampled ones. Returns (lower, upper), each (L,)."""
    from scipy.stats import t as student_t
    Y, Yb = np.asarray(Y, float), np.asarray(Yb, float)
    q = student_t.ppf(1 - (1 - level) / (2 * len(Y)), units - 1); sd = Yb.std(0, ddof=1) * np.sqrt(units / (units - 1))
    return Y - q * sd, Y + q * sd


def selected_band(Y, Yb, units, level=0.95):
    """A simultaneous band at `level` over several candidates (layers, cells), for reporting the best one with its choice
    counted: the max-type studentised band. Each side takes the quantile q = (1 + level) / 2 (0.975 for a 95% band) of the
    resampled maximum over the candidates, of (Yb - Y) / sigma for the lower bounds and of (Y - Yb) / sigma for the upper
    ones, with a small-sample factor: sigma, the resampling's standard deviation, times sqrt(units / (units - 1)), and the
    critical value times t_{q, units-1} / z_q. It adapts to how correlated the candidates are (close to one candidate's
    interval when they move together); Bonferroni (selected_bounds) is the conservative alternative. Y (L,), Yb (B, L),
    units as in selected_bounds. Returns (lower, upper), each (L,)."""
    from scipy.stats import norm, t as student_t
    Y, Yb = np.asarray(Y, float), np.asarray(Yb, float)
    sd = Yb.std(0, ddof=1) * np.sqrt(units / (units - 1)); sd = np.where(sd > 0, sd, np.inf)
    q = (1 + level) / 2; f = student_t.ppf(q, units - 1) / norm.ppf(q)
    cl = np.quantile(((Yb - Y) / sd).max(1), q) * f; cu = np.quantile(((Y - Yb) / sd).max(1), q) * f
    return Y - cl * sd, Y + cu * sd


def spectrum(W, D):
    """M = sym(E[w Delta^T]) from per-pair gradients W and differences D (torch, n x dim), computed in float64: a dict with
    tr, its trace (the mean of w . Delta); lam1 and lamD, its largest and smallest eigenvalues; and v1, the top eigenvector
    (a unit numpy vector), all exact. M x = 0 for every x orthogonal to the rows of W and D, so M's nonzero spectrum is that
    of Q^T M Q for an orthonormal basis Q of their span. When Q has fewer columns than the dimension, the complement adds
    the eigenvalue 0: lam1 is then at least 0 (v1 None when that 0 is the top) and lamD at most 0."""
    import torch
    n, dim = W.shape
    Wd, Dd = W.double(), D.double()
    Q = torch.linalg.qr(torch.cat([Wd, Dd]).T)[0]
    K = (Wd @ Q).T @ (Dd @ Q) / n
    ev, V = torch.linalg.eigh(0.5 * (K + K.T))
    lam1, lamD, v1 = float(ev[-1]), float(ev[0]), (Q @ V[:, -1]).float().cpu().numpy()
    if Q.shape[1] < dim:
        if lam1 < 0:
            lam1, v1 = 0.0, None
        lamD = min(lamD, 0.0)
    return dict(tr=float((Wd * Dd).sum(1).mean()), lam1=lam1, lamD=lamD, v1=v1)


def lam_boot(W, D, counts):
    """lambda_1 of sym(sum_i c_i w_i Delta_i^T) / sum_i c_i for each row c of counts (a torch float64 tensor, resamples x n,
    on W's device): the resample's own mean, as boot_ratio's numerators use, since a cluster bootstrap with clusters of
    unequal size draws a varying number of rows. Exact, in the span of the rows of W and D, as in spectrum(). Returns a
    torch tensor (resamples,)."""
    import torch
    Q = torch.linalg.qr(torch.cat([W, D]).double().T)[0]
    A, Bm = W.double() @ Q, D.double() @ Q
    out = []
    for c in torch.split(counts, 100):
        K = torch.einsum("bi,ij,ik->bjk", c, A, Bm) / c.sum(1)[:, None, None]
        lam = torch.linalg.eigvalsh(0.5 * (K + K.transpose(1, 2)))[:, -1]
        out.append(lam.clamp(min=0) if Q.shape[1] < W.shape[1] else lam)
    return torch.cat(out)

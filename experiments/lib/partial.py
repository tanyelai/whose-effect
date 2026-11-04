"""Two rank correlations of quantities x and y beyond a third quantity z, reported beside the inside maps' comparison of
predictors (experiments/score/inside_set_post.py) and used for no test (Section 6.2).
rank_residual residualises the ranks of x and y linearly on z's rank, as experiments/score/score_inside.py's partial_sp does. It is
biased under the null of no information beyond z: when x and y both follow z plus independent noise, cells close in z swap
places in both rankings and leave identical residual patterns, so the statistic is positive. beyond fits each quantity by least
squares on a cubic regression spline in z, with knots at equally spaced quantiles (one knot per eight cells beyond the first
eight, at most eight), and rank-correlates the two residuals: residuals are continuous, so no swap pattern is shared.
"""
import numpy as np


def _rank(v):
    return np.argsort(np.argsort(v, kind="mergesort"), kind="mergesort").astype(float)


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0


def spline_basis(z, per=8, kmax=8):
    """Cubic regression spline in z (standardised): 1, z, z^2, z^3 and (z - q)_+^3 at k = min(kmax, n // per - 1) interior
    quantiles."""
    z = np.asarray(z, float); z = (z - z.mean()) / (z.std() + 1e-12)
    cols = [np.ones_like(z), z, z ** 2, z ** 3]
    k = int(min(kmax, max(0, len(z) // per - 1)))
    if k:
        cols += [np.clip(z - q, 0, None) ** 3 for q in np.unique(np.quantile(z, np.linspace(0, 1, k + 2)[1:-1]))]
    return np.column_stack(cols)


def residual(v, X):
    beta, *_ = np.linalg.lstsq(X, np.asarray(v, float), rcond=None)
    return np.asarray(v, float) - X @ beta


def beyond(x, y, z):
    """Spearman correlation of the parts of x and y that a cubic spline in z does not explain; NaN with fewer than six cells."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if len(x) < 6:
        return np.nan
    X = spline_basis(z)
    return _corr(_rank(residual(x, X)), _rank(residual(y, X)))


def rank_residual(x, y, z):
    """The correlation of the ranks of x and y, each residualised linearly on z's rank, as score_inside.py's partial_sp
    computes it; NaN with fewer than four cells."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    if len(x) < 4:
        return np.nan
    rk = lambda v: np.argsort(np.argsort(v))                         # noqa: E731  (the scorer's own rank, ties included)
    rz = rk(z).astype(float)
    if np.std(rz) == 0:
        return _corr(rk(x).astype(float), rk(y).astype(float)) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)  # noqa: E731
    return _corr(res(rk(x).astype(float)), res(rk(y).astype(float)))

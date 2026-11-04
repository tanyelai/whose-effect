"""Tests that OwnEffect's statistics (owneffect/stats.py) equal experiments/lib/measure.py, the arithmetic the readout and inside-map
scorers use: the same clusters, resamples, ratios, intervals, p-values and false-discovery counts, bit for bit, on 50 random
trials, so that a scorer built on the library computes what a scorer built on measure.py computes (Section 8 and Appendix F).
Also a text check, not an import check: the source of the sentiment re-read's scorer (experiments/score/score_sentiment_reread.py)
contains an import from owneffect.stats and names block_counts and selected_bounds. Synthetic data only (no model, no outputs
of the runs); exits 1 on a failure.
Usage: python -I experiments/tests/owneffect_stats_identity_test.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/lib"))
import measure  # noqa: E402
from owneffect import stats  # noqa: E402

fails = []
rng = np.random.default_rng(7)
for trial in range(50):
    n = int(rng.integers(6, 200))
    keys = [(f"b{int(i)}", f"s{int(j)}") for i, j in rng.integers(0, max(2, n // 3), size=(n, 2))]
    a, b = measure.clusters(keys), stats.clusters(keys)
    if not np.array_equal(a, b):
        fails.append(f"clusters {trial}")
    B, seed = int(rng.integers(10, 500)), int(rng.integers(0, 10 ** 6))
    Ca, Cb = measure.boot_counts(a, B, seed), stats.boot_counts(b, B, seed)
    if not np.array_equal(Ca, Cb):
        fails.append(f"boot_counts {trial}")
    x, y = rng.normal(1, 1, n), rng.normal(2, 0.5, n)
    ra, rb = measure.boot_ratio(x, y, Ca), stats.boot_ratio(x, y, Cb)
    if not np.array_equal(ra, rb):
        fails.append(f"boot_ratio {trial}")
    for basic in (False, True):
        if measure.interval(float(x.mean() / y.mean()), ra, basic=basic) != stats.interval(float(x.mean() / y.mean()), rb, basic=basic):
            fails.append(f"interval {trial} {basic}")
    if measure.p_above(0.3, ra, 0.2) != stats.p_above(0.3, rb, 0.2):
        fails.append(f"p_above {trial}")
    p = rng.uniform(0, 1, n) ** 3
    if not np.array_equal(measure.bh(p, 0.025), stats.bh(p, 0.025)):
        fails.append(f"bh {trial}")
src = (ROOT / "experiments/score/score_sentiment_reread.py").read_text()
ok = "from owneffect.stats import" in src and "block_counts" in src and "selected_bounds" in src
print("ok  " if not fails else "FAIL", f"owneffect.stats equals measure on 50 random trials ({len(fails)} differences)", fails[:5])
print("ok  " if ok else "FAIL", "the sentiment re-read's scorer takes its block counts and bounds from OwnEffect")
sys.exit(1 if fails or not ok else 0)

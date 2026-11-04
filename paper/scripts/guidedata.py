"""The visual guide's map data: one map of experiments/scripts/inside.py (experiments/results/inside/inside_<model>_<task>.json and .npz), cell
by cell, with Section 6's classes computed as experiments/score/inside_offroute_direct.py computes them. G* is the network's own copy in
its best context (the context chosen on the training pairs, measured on the test pairs: the npz's gstar_train). DAS is
cross-entropy DAS averaged over its three starts. Every interval resamples the map's clusters with the map's own bootstrap
(B and seed from its JSON).
  carried    G*'s interval lies above zero
  off-route  G*'s interval does not; DAS's interval and DAS - G*'s both do
  above G*   DAS - G*'s interval lies above zero (any cell)
The classes are Section 6's, read from the intervals alone.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/lib"))
from measure import boot_counts, boot_ratio, interval  # noqa: E402


def map_cells(model, task):
    """{(layer, region): dict(gs, das, fr, own, g_ci, d_ci, diff_ci, carried, off, above)} over T for every live cell with a
    G*, plus the map's mean T, its layers and its regions. gs is G*, das cross-entropy DAS, fr the same with attention frozen,
    own the network's own single swap; layers are the code's (the input to block L, the paper's layer L - 1)."""
    J = json.load(open(ROOT / f"experiments/results/inside/inside_{model}_{task}.json"))
    Z = np.load(ROOT / f"experiments/results/inside/inside_{model}_{task}.npz")
    T = Z["T"].astype(float); Tm = float(T.mean()); C = boot_counts(Z["cluster"], J["boot"]["B"], J["boot"]["seed"])
    out = {}
    for c in J["cells"]:
        k = f"L{c['layer']}r{c['region']}"
        if not c.get("live") or f"{k}/gstar_train" not in Z.files:
            continue
        gs = Z[f"{k}/gstar_train"].astype(float)
        ce = np.mean([Z[f"{k}/e/das_ce_s{s}_normal"] for s in range(3)], axis=0).astype(float)
        fr = np.mean([Z[f"{k}/e/das_ce_s{s}_frozen"] for s in range(3)], axis=0).astype(float)
        g, d, dd = gs.mean() / Tm, ce.mean() / Tm, (ce - gs).mean() / Tm
        gci, dci = interval(g, boot_ratio(gs, T, C)), interval(d, boot_ratio(ce, T, C))
        ddci = interval(dd, boot_ratio(ce - gs, T, C))
        out[(c["layer"], c["region"])] = dict(gs=float(g), das=float(d), fr=float(fr.mean() / Tm),
                                              own=float(Z[f"{k}/e/own_normal"].mean() / Tm), g_ci=gci, d_ci=dci, diff_ci=ddci,
                                              carried=gci[0] > 0, off=gci[0] <= 0 and dci[0] > 0 and ddci[0] > 0, above=ddci[0] > 0)
    return out, Tm, sorted({c["layer"] for c in J["cells"]}), J["regions"]

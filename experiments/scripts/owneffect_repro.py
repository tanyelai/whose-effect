"""OwnEffect against the paper's measurement scripts on a real model (Section 8 and Appendix F): from the same pairs and the saved
directions, the library must give the saved results of experiments/scripts/readout.py and experiments/scripts/inside.py.

At the readout, per task of the readout's output for the model and task group: T per test pair, the exact effect of every
saved direction on every test pair, the test M's trace (midpoint and base point) and lambda_1 (midpoint), and, on a
two-label task (one label pair), the readout direction's share on the test pairs and cos theta on the training pairs.
readout.py signs a pair by its class by default and OwnEffect by its base label: the same on a two-label task, up to one
sign.
Inside, per map named: per live cell below the readout, the own swap; G* with its context chosen on the test pairs (as
OwnEffect chooses it on the pairs it is given); the paper's G*, whose context OwnEffect chooses on the training pairs (it must
be the saved one) and measures on the test pairs through context=; five saved directions free and with every attention pattern
frozen; and w . Delta per test pair and lambda_1 (midpoint). Regions empty in some pair are neither cells nor contexts.
Layer numbers: inside.py's layer L is the input to block L; OwnEffect's layer, like the paper's, is L - 1.
Outputs: experiments/results/owneffect_repro/owneffect_repro_<model>[_<out_tag>].json: per quantity the largest absolute difference
(log odds) and the number of values compared, and per task or map the same. The saved results and directions are read from
--results (default experiments/results; another results folder when the maps live there): readout/ (directions in readout/dirs/) and
inside/ (or inside_even/ with --inside_tag=even, CausalGym's maps at inside.py's layers 2, 4, ...).
--list=experiments/matrix/inside_maps.tsv --one_per_source=1 adds, for this model, the first map of every task group in the list,
each with the list's own data directory, so each map is recomputed on the rows it was measured on.
Usage: python -I experiments/scripts/owneffect_repro.py --model=gpt2 [--readout=causalgym,...] [--inside=agr_sv_num_pp,...]
       [--data=causalgym] [--list=experiments/matrix/inside_maps.tsv --one_per_source=1] [--inside_tag=even] [--results=<dir>]
       [--out_tag=<name>] [--device=cuda]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402

from owneffect import OwnEffect, stats  # noqa: E402

_, args, model, tok = common.start("owneffect_repro", sys.argv[1:])
REPO, TAG = common.REPO, common.RUN["tag"]
RES = Path(args["results"]) if Path(args.get("results", "experiments/results")).is_absolute() else REPO / args.get("results", "experiments/results")
OUT = dict(model=common.RUN["model"], results=str(args.get("results", "experiments/results")), total={}, readout={}, inside={})


def note(where, name, a, b):
    """Record the largest absolute difference between a and b under the total and under one task or map."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    assert a.shape == b.shape, (where, name, a.shape, b.shape)
    d = float(np.abs(a - b).max()) if a.size else 0.0
    for book in (OUT["total"], where):
        m = book.setdefault(name, [0.0, 0]); m[0] = max(m[0], d); m[1] += int(a.size)
    return d


t0 = time.time()
for group in [g for g in args.get("readout", "causalgym").split(",") if g]:
    stem = f"readout_{TAG}_m_{group}"
    J = json.load(open(RES / f"readout/{stem}.json"))
    Z, Dz = np.load(RES / f"readout/{stem}.npz"), np.load(RES / f"readout/dirs/{stem}.npz")
    for task, row in J["tasks"].items():
        if row.get("skipped"):
            continue
        R, here = common.rows(J["data"], task), OUT["readout"].setdefault(f"{group}/{task}", {})
        oe = OwnEffect(model, tok); oe.prepare(R["test"])
        note(here, "readout: T per test pair", oe.Tpairs, Z[f"{task}/T"])
        ro = oe.readout(B=10, per_pair=True)
        note(here, "readout: T from the readout states, per pair", ro["per_pair"]["T"], Z[f"{task}/T"])
        Wd = ro["range"]
        note(here, "readout: lambda_1 of the test M (midpoint)", Wd["lambda1"] * oe.T, row["lambda1_mid"])
        note(here, "readout: tr M (midpoint)", Wd["trM_mid"] * oe.T, row["trM_mid"])
        note(here, "readout: tr M (base point)", Wd["trM_base"] * oe.T, row["trM"])
        for k in sorted(f.split("/", 1)[1] for f in Dz.files if f.startswith(task + "/")):
            if f"{task}/e/{k}" in Z.files:
                e = oe.readout(direction=Dz[f"{task}/{k}"], B=10, per_pair=True)["per_pair"]["effect"]
                note(here, "readout: exact effect of a saved direction, per pair", e, Z[f"{task}/e/{k}"])
        if row.get("label_pairs") == 1 and ro["angle"] is not None:
            note(here, "readout: readout direction's share (test pairs)", ro["angle"]["readout_share"], row["grad_share_mid"])
            tr = OwnEffect(model, tok); tr.prepare(R["train"])
            a = tr.readout(B=10)["angle"]
            note(here, "readout: cos theta (training pairs)", np.cos(np.radians(a["theta_deg"])), row["cos_u_b_mid"])
        print(f"readout {group}/{task}: " + ", ".join(f"{n.split(': ')[1]} {v[0]:.1e}" for n, v in here.items()), flush=True)

jobs = [(t, args.get("data", "causalgym")) for t in args.get("inside", "").split(",") if t]
if args.get("list") and args.get("one_per_source") == "1":
    seen = set()
    for line in (REPO / args["list"]).read_text().strip().split("\n"):
        m_, group, t, data = line.split("\t")[:4]
        if m_ == TAG and group not in seen:
            seen.add(group); jobs.append((t, data))
IDIR = RES / ("inside_even" if args.get("inside_tag") == "even" else "inside")
for task, data in jobs:
    stem = f"inside_{TAG}_{task}" + (f"_{args['inside_tag']}" if args.get("inside_tag") else "")
    J = json.load(open(IDIR / f"{stem}.json"))
    Z, Dz = np.load(IDIR / f"{stem}.npz"), np.load(IDIR / f"dirs/{stem}.npz")
    R, here = common.rows(data, task), OUT["inside"].setdefault(task, {})
    here["data"] = data
    oe = OwnEffect(model, tok); oe.prepare(R["test"])
    tr = OwnEffect(model, tok); tr.prepare(R["train"])                   # G*'s context, chosen on the training pairs
    note(here, "inside: T per test pair", oe.Tpairs, Z["T"])
    NL = len(oe.blocks)
    empty = set(oe.empty) | set(tr.empty)                                 # regions empty in some pair: no cell, no context
    partners = [g for g in J["regions"] if g not in empty]
    for c in J["cells"]:
        L, rg = c["layer"], c["region"]
        if L == NL or not c["live"] or rg in empty:                        # empty regions leave every count, as in the paper
            continue
        layer, cell = L - 1, f"L{L}r{rg}"
        note(here, "inside: own swap, per pair", oe.effects(layer, rg)[0], Z[f"{cell}/e/own_normal"])
        if c["context_test"] not in empty:
            note(here, "inside: G* on the test pairs, per pair", oe.own(layer, rg, partners=partners, per_pair=True)[0],
                 Z[f"{cell}/gstar_test"])
        if c["context_train"] not in empty:                               # the paper's G*: context from the training pairs
            chosen = tr.own(layer, rg, partners=partners)[1]
            saved = "alone" if c["context_train"] is None else f"with region {c['context_train']}"
            note(here, "inside: G*'s context chosen on the training pairs, 0 where it is the saved one", float(chosen != saved), 0.0)
            note(here, "inside: G* in the training pairs' context (context=), per pair",
                 oe.own(layer, rg, context=saved, per_pair=True)[0], Z[f"{cell}/gstar_train"])
        for k in ("das_ce_s0", "das_sq_G_s0", "das_eff_s0", "u_hat", "v1_mid"):
            if f"{cell}/{k}" not in Dz.files:
                continue
            d = Dz[f"{cell}/{k}"]
            note(here, "inside: a saved direction's effect, per pair", oe.effects(layer, rg, d)[0], Z[f"{cell}/e/{k}_normal"])
            note(here, "inside: the same with attention frozen, per pair", oe.effects(layer, rg, d, frozen=True)[0],
                 Z[f"{cell}/e/{k}_frozen"])
        W, D = oe.grads(layer, rg, "mid")
        note(here, "inside: w . Delta per pair (midpoint)", (W * D).sum(1).double().cpu().numpy(), Z[f"{cell}/tr_mid"])
        note(here, "inside: lambda_1 (midpoint)", stats.spectrum(W, D)["lam1"], c["lambda1_mid"])
    print(f"inside {task}: " + ", ".join(f"{n.split(': ')[1]} {v[0]:.1e}" for n, v in here.items() if n != "data"), flush=True)

OUT["seconds"] = round(time.time() - t0)
res = REPO / "experiments/results/owneffect_repro"; res.mkdir(parents=True, exist_ok=True)
f = res / f"owneffect_repro_{TAG}{'_' + args['out_tag'] if args.get('out_tag') else ''}.json"
json.dump(OUT, open(f, "w"), indent=1)
print(json.dumps(OUT["total"], indent=1))
common.finish([f], summary={k: v[0] for k, v in OUT["total"].items()})

"""Tests experiments/score/score_causalgym_map.py: CausalGym's aggregate on the rebuilt maps, the off-route share of DAS's lead over
vanilla, and the checks that stop the scorer. Synthetic data only (odd- and even-layer maps with known answers; no model, no
outputs of the runs); exits 1 on a failure.
The scorer splits DAS's lead over vanilla by where it lies (Appendix E.1, "Where the gap between DAS and vanilla comes from"). It compares
the maps with the published values and figure (output key published_figure), and it asks whether the off-route share s_off, the part of the
lead from layers where DAS's chosen cell has G*'s interval not above 0 (G* is the network's own interchange in its best
context), lies above 0 in each model.
Two models (pythia-1b, gpt2), five CausalGym-like tasks each, six layers (odd 1, 3, 5 and the readout cell 6 in one run; even
2, 4 in the other), regions 1-3. Planted, per layer: vanilla's best is region 1 (G = 0.4 T; its best context G* = 0.5 T) and
DAS's best is the same cell at 0.9 T, except at layers 2 and 4, where DAS picks region 3, a cell the network does not carry
(G = G* = 0) at 1.2 T; at the readout cell G = G* = T and DAS 1.2 T. So the lead X = 3 x 0.5 + 2 x 0.8 + 0.2 = 3.3 T, the
off-route part 1.6 T and s_off = 0.485. In flat_w DAS reports the network's own interchange, so it has no lead. Also planted:
a dead cell (layer 1, region 2 of every odd run), which enters with every effect 0.
Checks:
- s_off is recovered at the point, and it lies above 0 in each model.
- The off-route picks are counted per construction, with G*'s upper bound there, and the off-route lead lies all at region 3.
- A map whose lead is not established has no shares of its own.
- Under a null with no off-route choice, s_off's interval holds 0 and the off-route test fails.
- One task has region 2 empty in half its rows: those cells are degenerate and leave the primary choice.
- A map with mean T near 0 keeps its choices across resamples (they are made on log odds).
- One map whose readout fails the instrument is excluded by name, and its model is still judged.
- The published vanilla (set to ours) differs by less than 0.005, the published DAS (set to our single-start mean) lies in our
  range, and DAS's map aligned with a darkness grid equal to it is "aligned".
- The scorer stops when the even run's readout cell disagrees with the odd run's, when a trained start there moves by more than
  its starts' range, or when the even run's layers are wrong; a move within the starts' range passes.
Usage: python -I experiments/tests/causalgym_map_scorer_test.py
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
sys.path.insert(0, str(ROOT))
from owneffect.stats import boot_counts, clusters  # noqa: E402

B, SEED, NL, REG = 400, 95, 6, (1, 2, 3)
TASKS = ("npi_any_subj-relc", "agr_x", "gss_y", "garden_z", "flat_w")
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def cell_values(L, r, null, task=""):
    """(G, G*, ce) in units of the scale for one cell."""
    if task == "flat_w":                                               # DAS reports the network's own interchange: no lead
        return (1.0, 1.0, 1.0) if L == NL else (0.4, 0.4, 0.4)
    if L == NL:
        return 1.0, 1.0, 1.2
    if r == 3 and L in (2, 4) and not null:
        return 0.0, 0.0, 1.2                                           # DAS's choice, a cell the network does not carry
    return {1: (0.4, 0.5, 0.9), 2: (0.2, 0.3, 0.4), 3: (0.0, 0.0, 0.2)}[r]


def write_run(d, model, task, layers, rng, T, scale, cl, null=False, even=False, bad_readout=False, failing=False, spread=0.0, shift=0.0):
    """One run's json and npz in experiments/scripts/inside.py's format: per cell the network's own interchange, its best context and
    three cross-entropy starts (with attention free and frozen); spread separates the starts at the readout cell."""
    Z = {"T": T, "cluster": cl, "half": np.tile([True, False], len(T) // 2), "aligned": np.ones(len(T), bool)}; cells = []
    C = boot_counts(cl, B, SEED)
    for L in layers + [NL]:
        for r in (REG if L < NL else (REG[-1],)):
            c = f"L{L}r{r}"
            if (L, r) == (1, 2) and not even:                           # a dead cell
                cells.append(dict(layer=L, region=r, live=False, door=False)); continue
            G, Gs, ce = cell_values(L, r, null, task)
            sd = 0.0 if (G == 0.0 and Gs == 0.0) else 0.05                 # a cell the network does not carry: G* exactly 0
            noise = lambda: rng.normal(0, sd, len(T))  # noqa: E731
            g = (T if L == NL else scale * G + noise())
            if bad_readout and L == NL:
                g = g + 1.0             # a third of T: far above the effect tolerance (0.0545 in GPT-2, 1e-3 if synthetic)
            Z[f"{c}/e/own_normal"] = g
            Z[f"{c}/gstar_test"] = g if L == NL else scale * Gs + noise(); Z[f"{c}/gstar_train"] = Z[f"{c}/gstar_test"]
            for s in (0, 1, 2):
                v = (T * (ce + spread * s) if L == NL else scale * ce + rng.normal(0, 0.05, len(T)))
                if L == NL and s == 0:
                    v = v + shift * T                                   # the even run's start 0 moved by shift T at the readout cell
                Z[f"{c}/e/das_ce_s{s}_normal"] = v; Z[f"{c}/e/das_ce_s{s}_frozen"] = 0.3 * v
            Z[f"{c}/e/u_hat_normal"] = g
            rec = dict(layer=L, region=r, live=True, door=False, context_test=None)
            if L == NL:
                rec["own_vs_T_max_abs"] = 1.0 if failing else 0.0          # a patching error: an effect off by a third of T
            cells.append(rec)
    name = f"inside_{model}_{task}{'_even' if even else ''}"
    fz = d / f"{name}.npz"; np.savez(fz, **Z)
    J = dict(task=task, model=model, run=name, layers=layers, regions=list(REG), cells=cells, boot=dict(B=B, seed=SEED),
             boot_counts_sha=hashlib.sha256(np.ascontiguousarray(C, dtype=np.int64).tobytes()).hexdigest()[:16],
             npz_sha256=hashlib.sha256(fz.read_bytes()).hexdigest(), full_grid=not even, stack=True, long=False, cells_arg=None,
             frozen_no_patch_max_abs=0.0, n_train=400, n_test=len(T), n_clusters_test=len(T) // 2)
    json.dump(J, open(d / f"{name}.json", "w"))


def rows(task):
    """Four-region rows; in gss_y region 2 is empty in half the rows (a degenerate site)."""
    out = []
    for i in range(50):
        b = ["<s>", f" a{i}", "" if (task == "gss_y" and i % 2 == 0) else " x", " y"]
        out.append(dict(base=b, src=["<s>", f" c{i}", b[2], " y"], base_label=" p", src_label=" q", base_type="A"))
    return dict(train=out[:40], test=out[40:])


def tree(null=False, bad_readout=False, bad_layers=False, spread=0.0, shift=0.0):
    """A synthetic results folder: inside/ (odd layers and the readout cell), inside_even/ and fake_rows/, for both models and
    every task; gpt2's agr_x fails its instrument, and gpt2's gss_y carries the even run's planted faults."""
    r = Path(tempfile.mkdtemp()); (r / "inside").mkdir(); (r / "inside_even").mkdir(); (r / "fake_rows").mkdir()
    rng = np.random.default_rng(3)
    for t in TASKS:
        json.dump(rows(t), open(r / "fake_rows" / f"{t}.json", "w"))
    for m in ("pythia-1b", "gpt2"):
        for t in TASKS:
            n = 100; T = np.repeat(rng.normal(0.1 if t == "garden_z" else 3.0, 0.5 if t == "garden_z" else 0.3, n // 2), 2)
            scale = 3.0                                                # effects on a fixed log-odds scale, so T near 0 cannot flip them
            cl = clusters([(f"b{i}", f"s{i}") if k == 0 else (f"s{i}", f"b{i}") for i in range(n // 2) for k in range(2)])
            failing = (m, t) == ("gpt2", "agr_x")
            write_run(r / "inside", m, t, [1, 3, 5], rng, T, scale, cl, null=null, failing=failing, spread=spread)
            write_run(r / "inside_even", m, t, [2] if bad_layers and (m, t) == ("gpt2", "gss_y") else [2, 4], rng, T, scale, cl,
                      null=null, even=True, bad_readout=bad_readout and (m, t) == ("gpt2", "gss_y"), failing=failing, spread=spread,
                      shift=shift if (m, t) == ("gpt2", "gss_y") else 0.0)
    return r


def run(r):
    p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_causalgym_map.py"), f"--fake={r}"], capture_output=True, text=True)
    out = json.load(open(r / "score_causalgym_map.json")) if (r / "score_causalgym_map.json").is_file() else None
    return p, out


r = tree()
p, out = run(r)
check("scorer runs", p.returncode == 0 and out is not None, (p.stdout + p.stderr)[-800:])
if out:
    mp = out["maps"]["pythia-1b/agr_x"]["primary"]
    check("s_off recovered at the point (1.6 of 3.3)", abs(mp["s_off"]["pt"] - 1.6 / 3.3) < 0.03, f"{mp['s_off']['pt']:.3f}")
    check("the off-route share is above 0 in each model on the planted maps", out["models"]["pythia-1b"]["s_off_above_0"] and out["models"]["gpt2"]["s_off_above_0"],
          json.dumps({m: v.get("s_off") for m, v in out["models"].items()}))
    check("the failing map is excluded by name and its model still judged", out["excluded"] == {"gpt2/agr_x": "instruments"}
          and out["models"]["gpt2"]["tasks"] == 4, f"{out['excluded']} {out['models']['gpt2'].get('tasks')}")
    op = out["models"]["pythia-1b"]["off_picks"]
    check("off-route picks are counted by construction, with G*'s upper bound there", op["constructions_with_off_pick"] == 4
          and op["constructions"] == 5 and op["Gstar_hi_T"] is not None and op["Gstar_hi_T"]["max"] < 0.05, json.dumps(op))
    ob = out["maps"]["pythia-1b/agr_x"]["primary"]["off_lead_by_region"]
    check("the off-route lead is located: all of it at region 3, where DAS's off-route picks are, none at its other picks",
          abs(ob["3"]["pt"] - 1) < 1e-9 and all(abs(v["pt"]) < 1e-9 for r, v in ob.items() if r != "3"), json.dumps({r: v["pt"] for r, v in ob.items()}))
    fw = out["maps"]["pythia-1b/flat_w"]["primary"]
    check("a map whose lead over vanilla is not established has no shares of its own", not fw["lead_established"] and fw["s_off"]["pt"] is None,
          json.dumps({k: fw[k] for k in ("X", "lead_established", "s_off")}))
    g = out["maps"]["pythia-1b/gss_y"]
    check("region 2 of gss_y is degenerate and leaves the primary choice", g["degenerate_regions"] == [2] and g["degenerate_cells"] > 0,
          f"{g['degenerate_regions']} {g['degenerate_cells']}")
    z = out["maps"]["pythia-1b/garden_z"]["primary"]["per_layer"]
    check("a map with mean T near 0 keeps its choices across resamples", all(v["stable"] >= 0.9 for v in z.values()),
          str({L: v["stable"] for L, v in z.items()}))
    pub = {"pythia-1b|main": {"npi_any_subj-relc": {"vanilla": out["maps"]["pythia-1b/npi_any_subj-relc"]["primary"]["raw"]["V"],
                                                     "das": float(np.mean(out["maps"]["pythia-1b/npi_any_subj-relc"]["primary"]["raw"]["Ds"]))}}}
    json.dump(pub, open(r / "published.json", "w"))
    dark = np.zeros((4, NL))
    J1 = json.load(open(r / "inside/inside_pythia-1b_npi_any_subj-relc.json")); Z1 = np.load(r / "inside/inside_pythia-1b_npi_any_subj-relc.npz")
    J2 = json.load(open(r / "inside_even/inside_pythia-1b_npi_any_subj-relc_even.json")); Z2 = np.load(r / "inside_even/inside_pythia-1b_npi_any_subj-relc_even.npz")
    for J, Z in ((J1, Z1), (J2, Z2)):
        for c in J["cells"]:
            if c["layer"] < NL and c.get("live", True):
                dark[c["region"], c["layer"] - 1] = np.mean([Z[f"L{c['layer']}r{c['region']}/e/das_ce_s{s}_normal"].mean() for s in (0, 1, 2)])
    json.dump({"das": {"143000": dark.tolist()}}, open(r / "darkness.json", "w"))
    _, out2 = run(r)
    pm = out2["maps"]["pythia-1b/npi_any_subj-relc"].get("published", {}) if out2 else {}
    check("the published vanilla differs by less than its rounding and the published DAS lies in our single-start range",
          pm and abs(pm["vanilla_difference"]) < 0.005 and pm["das_inside_single_start_range"], str(pm))
    al = out2["published_figure"].get("alignment_pythia_1b_npi", {}) if out2 else {}
    check("DAS's map aligned with an equal darkness grid is 'aligned'", al.get("aligned") is True, str(al))
shutil.rmtree(r)
rn = tree(null=True); pn, outn = run(rn)
check("under a null with no off-route choice s_off's interval holds 0 and the off-route test fails",
      outn is not None and all(v["s_off"][1] <= 0 <= v["s_off"][2] + 1e-12 and not v["s_off_above_0"] for v in outn["models"].values() if v.get("tasks")),
      json.dumps({m: v.get("s_off") for m, v in outn["models"].items()}) if outn else (pn.stdout + pn.stderr)[-500:])
shutil.rmtree(rn)
rb = tree(bad_readout=True); pb, _ = run(rb)
check("a readout cell that disagrees between the runs stops the scorer", pb.returncode == 2 and "readout cell" in pb.stdout, pb.stdout[-300:])
shutil.rmtree(rb)
rs = tree(shift=0.5); ps, _ = run(rs)
check("a trained start that moves between the runs by more than its starts' range (here 0: the starts agree) stops the scorer",
      ps.returncode == 2 and "starts' range" in ps.stdout, ps.stdout[-300:])
shutil.rmtree(rs)
rs = tree(spread=0.2, shift=0.1); ps, _ = run(rs)
check("a move within the starts' own range (0.1 T against 0.4 T) passes", ps.returncode == 0, (ps.stdout + ps.stderr)[-300:])
shutil.rmtree(rs)
rl = tree(bad_layers=True); pl, _ = run(rl)
check("an even run with the wrong layers stops the scorer", pl.returncode == 2 and "layers" in pl.stdout, pl.stdout[-300:])
shutil.rmtree(rl)
print("\nCAUSALGYM MAP SCORER OK" if not fails else "CAUSALGYM MAP SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

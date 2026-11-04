"""Tests experiments/score/score_readout_ce_law.py end to end: whether first order applied to cross-entropy's own measure, log p(y_s),
predicts where cross-entropy DAS lands (Appendix C.2, "Cross-entropy ends near its own optimum"). Needs outputs of the runs: the
readout's outputs for GPT-2 and Pythia-410m (experiments/results/score_readout.json, experiments/results/readout/ and experiments/results/readout/dirs/;
run experiments/run/readout.sh and experiments/score/score_readout.py first). The law's outputs are built on them synthetically; no model is
loaded. Exits on the first failed assert.
v_f is the top eigenvector of that measure's M_f. The scorer's tests, with their output keys: v_f's shortfall from the top
against cross-entropy's, per model (shortfall); the alignment of cross-entropy's off-top part with v_f's against a null (alignment); the
shortfall comparison on the contrast between one-directional and mirrored negative-polarity constructions, over a tier's
models (npi_contrast).
Stand-ins for v_f's effect, each scored end to end:
1. "ce": v_f's effect = cross-entropy's own (ten passes): the shortfall test holds (R = 1, beats the constant-fraction
   baseline, Spearman 1), and so does the contrast (R_c = 1).
2. Negative controls, which must fail both: "zero" (no effect), "near_top" (0.995 of the top: near the top, not near
   cross-entropy), "u_hat" (the mean difference, unrelated to the theory), "v1_mid" (the log odds' own top direction),
   "constfrac" (0.9 of the top everywhere).
The synthetic alignments (0.8 against a null of 0.2, off-top norm 0.3) must pass the alignment test, and the sha checks must
pass. Controls of the alignment test: an alignment equal to its null, or below it, fails with the floor on the off-top norm
and without it. An off-top norm of 0.01, under the floor of 0.05, leaves no task to test with the floor (held None, for the
tier too), while the form without the floor (without_floor) holds. The contrast's comparison without a bar, against
the best of the naive directions (u_hat, v1_mid, no effect, a constant ratio), holds for "ce" and fails for "u_hat", "v1_mid"
and "zero"; so does the tier's shortfall test without R's bar (shortfall_without_bar).
Usage: python -I experiments/tests/ce_law_scorer_test.py
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
if not (ROOT / "experiments/results/score_readout.json").is_file():
    print("SKIPPED: this check builds on the readout's outputs (experiments/results/score_readout.json and experiments/results/readout/); run bash experiments/run/readout.sh and bash experiments/score/score_all.sh first")
    sys.exit(0)
MODELS = ["gpt2", "pythia-410m"]


def build(F, mode, c2=(0.8, 0.2, 0.3)):
    """A fake folder F for one stand-in (mode) of v_f's effect: the two models' readout outputs and directions copied in, and
    per group a readout_ce_law json and npz with v_f's effect, the top's effect standing in for v_f^base's (v_f built at the
    base point), and the alignment test's inputs c2 = (alignment, null, off-top norm)."""
    (F / "dirs").mkdir()
    S = json.load(open(ROOT / "experiments/results/score_readout.json"))
    S["report"]["models"] = {m: S["report"]["models"][m] for m in MODELS}; S["rows"] = {m: S["rows"][m] for m in MODELS}
    json.dump(S, open(F / "score_readout.json", "w"))
    for m, g in {(m, r["group"]) for m in MODELS for r in S["rows"][m]}:
        stem = f"readout_{m}_m_{g}"
        if not (ROOT / f"experiments/results/readout/{stem}.npz").is_file():
            continue
        for suf in (".json", ".npz"):
            shutil.copy(ROOT / f"experiments/results/readout/{stem}{suf}", F / f"{stem}{suf}")
        shutil.copy(ROOT / f"experiments/results/readout/dirs/{stem}.npz", F / "dirs" / f"{stem}.npz")
        ZR = np.load(F / f"{stem}.npz"); JR = json.load(open(F / f"{stem}.json"))
        arrays, tasks = {}, {}
        for t in JR["tasks"]:
            if f"{t}/T" not in ZR.files:
                continue
            ce10 = np.mean([ZR[f"{t}/e/das_ce_p10_s{s}"] for s in (0, 1, 2)], axis=0); top = ZR[f"{t}/e/top_train"]
            arrays[f"{t}/e/v_f"] = {"ce": ce10, "zero": 0 * ce10, "near_top": 0.995 * top, "u_hat": ZR[f"{t}/e/u_hat"],
                                    "v1_mid": ZR[f"{t}/e/v1_mid"], "constfrac": 0.9 * top}[mode]
            arrays[f"{t}/e/v_f_base"] = top; arrays[f"{t}/T"] = ZR[f"{t}/T"]
            tasks[t] = dict(instruments=dict(T_matches_readout=True, v_z_is_v1_mid=True), sin_v_f=c2[2],
                            align={f"s{s}": dict(ce_v_f=c2[0], ce_eff=c2[1], sin_ce=c2[2], sin_eff=c2[2]) for s in (0, 1, 2)},
                            f_tops={k: dict(e_T=float(ce10.mean() / ZR[f"{t}/T"].mean()), train_ce=0.1, cos_ce10=[0.99] * 3)
                                    for k in ("from_v_f", "from_v1_mid")},
                            train_ce={f"das_ce_p10_s{s}": 0.1 for s in (0, 1, 2)}, median_p_b_mid=0.1, median_outside_mid=0.1)
        lstem = f"readout_ce_law_{m}_m_{g}"
        np.savez_compressed(F / f"{lstem}.npz", **arrays)
        json.dump(dict(tasks=tasks, readout_sha256=hashlib.sha256((F / f"{stem}.npz").read_bytes()).hexdigest(),
                       readout_dirs_sha256=hashlib.sha256((F / "dirs" / f"{stem}.npz").read_bytes()).hexdigest(),
                       npz_sha256=hashlib.sha256((F / f"{lstem}.npz").read_bytes()).hexdigest()), open(F / f"{lstem}.json", "w"))


for mode, shortfall_should in (("ce", True), ("zero", False), ("near_top", False), ("u_hat", False), ("v1_mid", False), ("constfrac", False)):
    with tempfile.TemporaryDirectory() as tmp:
        F = Path(tmp); build(F, mode)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout_ce_law.py"), f"--fake={F}"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        R = json.load(open(F / "score_readout_ce_law.json"))
        for m in MODELS:
            assert R[m]["shortfall"]["held"] == shortfall_should, (mode, m, R[m]["shortfall"])
            assert R[m]["alignment"]["held"], (mode, m, R[m]["alignment"])
        assert R["tiers"]["small"]["npi_contrast"]["held"] == shortfall_should, (mode, R["tiers"]["small"]["npi_contrast"])
        assert R["tiers"]["small"]["shortfall"] == shortfall_should
        print(f"{mode}: shortfall test held = {shortfall_should} as expected;", " | ".join(f"{m} R {R[m]['shortfall']['R'][0]:.2f}, pop {R[m]['population']}" for m in MODELS),
              f"| NPI contrast R_c {R['tiers']['small']['npi_contrast']['R_c'][0]:.2f}, held {R['tiers']['small']['npi_contrast']['held']}")
for mode, should in (("ce", True), ("u_hat", False), ("v1_mid", False), ("zero", False)):
    with tempfile.TemporaryDirectory() as tmp:
        F = Path(tmp); build(F, mode)
        subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout_ce_law.py"), f"--fake={F}"], capture_output=True, text=True)
        R = json.load(open(F / "score_readout_ce_law.json"))
        assert R["tiers"]["small"]["npi_contrast"]["beats_naive_without_bar"] == should, (mode, R["tiers"]["small"]["npi_contrast"])
        assert R["tiers"]["small"]["shortfall_without_bar"] == should, (mode, R["tiers"]["small"])
        print(f"C3 and the tier's C1 without a bar, {mode}: {should} as expected")
for c2, with_floor, without_floor in (((0.5, 0.5, 0.3), False, False), ((0.2, 0.8, 0.3), False, False), ((0.8, 0.2, 0.01), None, True)):
    with tempfile.TemporaryDirectory() as tmp:
        F = Path(tmp); build(F, "ce", c2)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_readout_ce_law.py"), f"--fake={F}"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        R = json.load(open(F / "score_readout_ce_law.json"))
        for m in MODELS:
            assert R[m]["alignment"]["held"] is with_floor and R[m]["alignment"]["without_floor"]["held"] == without_floor, (c2, m, R[m]["alignment"])
        assert R["tiers"]["small"]["alignment"] is with_floor, R["tiers"]["small"]                    # the tier follows its models
        print(f"alignment control: align {c2[0]}, null {c2[1]}, off-top norm {c2[2]}: with the floor {with_floor}, without it {without_floor} as expected")
print("all cross-entropy-law scorer tests passed")

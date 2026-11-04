"""The sentiment direction of Tigges et al. re-read with the authors' code and fitting protocol, scored with intervals (Section 7;
Section 7.3 and Appendix E.3, Table 10). experiments/tests/sentiment_reread_scorer_test.py tests it on synthetic re-reads.
Inputs: experiments/results/sentiment_reread_<model>_seed<s>.json and _rows.json (experiments/scripts/sentiment_reread.py, --seed=0,1,2,
--save_rows=1), seven models, and each run's record in experiments/logs/ (env.json, done.json).
Provenance: each output names its seed, objective, epochs, learning rate and the authors' prompts.yaml by sha256, and these
must be the protocol's (seed s, objective "lin", 64 epochs, learning rate 1e-3, the sha256 experiments/lib/pins.py pins); its run
record must show the model's pinned revision, and every file the run executed must be byte-identical in this tree
(experiments/lib/provenance.py). A run that fails is listed and not scored; a model with any seed missing is "not built", with the
reason.
Per model and layer, on the test rows: Y(L) = 100 (mean P - mean k) / (mean c - mean k), with P the patched logit difference
(the last epoch's direction, every position: E2; the seed mean per row, and each seed alone), c and k the clean and corrupted
runs' (k_r = -c_{r+1}). The table's rule picks L* = argmax Y(L).
Intervals: the circular block bootstrap (block 2, primary; block 4 and the row bootstrap as sensitivity), B = 10 000, seed 0, a
ratio of means on the same resample (owneffect.stats.block_counts). The layer's choice is counted by the simultaneous band of
owneffect.stats.selected_band (the max-type studentised band with a small-sample factor, u = the number of blocks), a 95%
band (each bound at 2.5%, as the paper's two-sided 95% intervals), read at L*; Bonferroni-t (selected_bounds, the
conservative alternative) and the plain band, at the same level, are reported beside it.
Instruments (instruments_hold), each within the float32 tolerance at the model's logit scale (experiments/lib/instrument_tol.py), a
difference dY in percent counting as |dY| |mean c - mean k| / 100 in logits: own E2 equal to 100 at every layer, own_vs_logits,
own E1 the same in every seed; the clean logit difference above 0 and the corrupted one below. A model that fails them is not
read against 100 (its reading is "not read (instruments)").
Readings: the published value (key published_value): for Pythia-1.4b, seed 0, does Tigges et al.'s 109.8 (their Table 1a) lie inside the
band at L* ("consistent with")? Above 100 (key above_100): the band's lower bound at L* above 100, per seed and for the seed mean;
"significantly above" needs every seed. Own E1 per layer (key own_E1_per_layer), exact, with a descriptive interval; DAS's E1 minus
own E1 per layer (key das_E1_minus_own_E1), with its interval. Reports: how far training turned the direction (bounds on
cos(initial, last) from the recorded cos(initial, kept) and cos(kept, last)), and, where a second seed-0 run of Pythia-160m or
410m is present (TWIN), seed 0's per-layer difference from it.
Output: experiments/results/score_sentiment_reread.json. Usage: python -I experiments/score/score_sentiment_reread.py [--fake=<dir>] [--b=10000]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from owneffect.stats import block_counts, selected_band, selected_bounds  # noqa: E402   (the library's)
sys.path.insert(0, str(ROOT / "experiments/lib"))
from instrument_tol import tol  # noqa: E402
import pins  # noqa: E402
from provenance import not_this_code  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
FAKE = bool(A.get("fake"))
RES = Path(A["fake"]) if FAKE else ROOT / "experiments/results"
NB = int(A.get("b", 10000))
MODELS = {"gpt2": "gpt2", "pythia-160m": "EleutherAI/pythia-160m", "pythia-410m": "EleutherAI/pythia-410m",
          "pythia-1b": "EleutherAI/pythia-1b", "pythia-1.4b": "EleutherAI/pythia-1.4b", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
          "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B"}
PIN_14B = pins.MODELS["EleutherAI/pythia-1.4b"]
YAML_SHA = pins.TIGGES_PROMPTS_SHA256
STRATUM = {m: ("pythia" if m.startswith("pythia") else "own word lists") for m in MODELS}
PUBLISHED = ("pythia-1.4b", 109.8)
TWIN = {"pythia-160m": "sentiment_reread_pythia-160m_rm_reread_cuda_seed0.json",
        "pythia-410m": "sentiment_reread_pythia-410m_rm_reread_cuda_seed0.json"}


def pin(m):
    if m == "pythia-1.4b":
        return PIN_14B
    sys.path.insert(0, str(ROOT / "experiments/matrix")); import registry  # noqa: E402
    return registry.REVISION[MODELS[m]]


def provenance(m, s, J, stem):
    """None when the run has the protocol's settings, the pinned revision and this tree's code; else the reason."""
    want = dict(seed=s, objective="lin", epochs=64, lr=1e-3, yaml_sha256=YAML_SHA)
    for k, v in want.items():
        if J.get(k) != v:
            return f"{k} is {J.get(k)!r}, not {v!r}"
    if FAKE:
        return None
    recs = [d for d in (ROOT / "experiments/logs").glob("sentiment_reread_*") if (d / "done.json").is_file()
            and any(o["path"] == f"experiments/results/{stem}.json" for o in json.load(open(d / "done.json"))["outputs"])]
    if len(recs) != 1:
        return f"{len(recs)} run records name this output"
    E = json.load(open(recs[0] / "env.json"))
    # the record's resolved revision, else the --revision its command passed to from_pretrained (common._revision returns None
    # for a snapshot fetched by commit offline, where the cache has no refs/main, as for Pythia-1.4b; offline the command can
    # load no other snapshot)
    rev = E.get("model_revision") or next((a.split("=", 1)[1] for a in E.get("command", []) if a.startswith("--revision=")), None)
    if rev != pin(m):
        return f"run record: revision {rev}, pinned {pin(m)}"
    return not_this_code(json.load(open(recs[0] / "done.json")).get("code_sha256"))


def pct(W, P, c, k):
    """100 (mean P - mean k) / (mean c - mean k) on each row of weights W (ratio of means on the same resample)."""
    w = W.sum(1); mp, mc, mk = W @ P / w, W @ c / w, W @ k / w
    return 100 * (mp - mk) / (mc - mk)


def plain_band(Y, Yb):
    sd = Yb.std(0, ddof=1); sd = np.where(sd > 0, sd, np.inf)
    cl = np.quantile(((Yb - Y) / sd).max(1), 0.975); cu = np.quantile(((Y - Yb) / sd).max(1), 0.975)
    return Y - cl * sd, Y + cu * sd


def judge(lo, hi):
    return "above" if lo > 100 else ("below" if hi < 100 else "indistinguishable")


def cos_bounds(a, b):
    """cos(initial, last) given cos(initial, kept) = a and cos(kept, last) = b: the angle lies between |t_a - t_b| and t_a + t_b."""
    ta, tb = np.arccos(np.clip(a, -1, 1)), np.arccos(np.clip(b, -1, 1))
    return [float(np.cos(min(ta + tb, np.pi))), float(np.cos(abs(ta - tb)))]


def score_model(m, failed):
    runs = {}
    for s in (0, 1, 2):
        stem = f"sentiment_reread_{m}_seed{s}"; f, fr = RES / f"{stem}.json", RES / f"{stem}_rows.json"
        if not f.is_file() or not fr.is_file():
            return dict(built=False, reason=f"seed {s}: no output")
        J = json.load(open(f))
        if why := provenance(m, s, J, stem):
            failed[f"{m}/seed{s}"] = why; return dict(built=False, reason=f"seed {s}: {why}")
        runs[s] = (J, json.load(open(fr)))
    J0, R0 = runs[0]
    layers = sorted(int(L) for L in R0["layers"])
    c = np.array(R0["base"]["test"]["clean"], float); k = np.array(R0["base"]["test"]["corrupt"], float); n = len(c)
    rows = {s: {L: np.array(runs[s][1]["layers"][str(L)]["last_das_E2"], float) for L in layers} for s in runs}
    mean_rows = {L: np.mean([rows[s][L] for s in runs], axis=0) for L in layers}
    own_e1 = {L: np.array(R0["layers"][str(L)]["own_E1"], float) for L in layers}
    e1 = {L: np.mean([np.array(runs[s][1]["layers"][str(L)]["last_das_E1"], float) for s in runs], axis=0) for L in layers}
    one = np.ones((1, n))
    # every identity within the instruments' float32 tolerance at the model's logit scale; a deviation of Y in percent is
    # |Y - Y'| |mean c - mean k| / 100 in logit units, a difference of logit differences (an effect)
    to_logits = abs(c.mean() - k.mean()) / 100; te = tol(m, "effect", fake=FAKE)
    inst = dict(own_E2_within=all(abs(runs[s][0]["layers"][str(L)]["own_E2"] - 100) * to_logits <= te for s in runs for L in layers),
                own_vs_logits=max(runs[s][0]["layers"][str(L)]["own_vs_logits"] for s in runs for L in layers) <= tol(m, "logit", fake=FAKE),
                clean_above_corrupt=bool(c.mean() > 0 > k.mean()),
                own_E1_same_seeds=all(abs(runs[s][0]["layers"][str(L)]["own_E1"] - J0["layers"][str(L)]["own_E1"]) * to_logits <= te
                                      for s in runs for L in layers))
    out = dict(built=True, n_test=n, layers=layers, stratum=STRATUM.get(m), instruments=inst, instruments_hold=all(inst.values()))
    for name, block in (("block2", 2), ("block4", 4), ("rows", 1)):
        W = block_counts(n, block, NB, 0); u = int(np.ceil(n / block)); res = {}
        for who, P in [("mean", mean_rows)] + [(f"seed{s}", rows[s]) for s in runs]:
            Y = np.array([pct(one, P[L], c, k)[0] for L in layers]); Yb = np.stack([pct(W, P[L], c, k) for L in layers], 1)
            lo, hi = selected_band(Y, Yb, u); blo, bhi = selected_bounds(Y, Yb, u); plo, phi = plain_band(Y, Yb); j = int(np.argmax(Y))
            res[who] = dict(L_star=layers[j], Y_star=float(Y[j]), lower=float(lo[j]), upper=float(hi[j]),
                            against_100=judge(lo[j], hi[j]) if out["instruments_hold"] else "not read (instruments)",
                            bonferroni=[float(blo[j]), float(bhi[j])], plain_band=[float(plo[j]), float(phi[j])],
                            max_bootstrap=[float(np.quantile(Yb.max(1), 0.025)), float(np.quantile(Yb.max(1), 0.975))],
                            fixed_layer=[float(np.quantile(Yb[:, j], 0.025)), float(np.quantile(Yb[:, j], 0.975))])
        if name == "block2":
            out["own_E1_per_layer"] = {L: dict(own_E1=float(pct(one, own_e1[L], c, k)[0]),
                                        ci=[float(x) for x in np.quantile(pct(W, own_e1[L], c, k), [0.025, 0.975])]) for L in layers}
            out["das_E1_minus_own_E1"] = {L: dict(diff=float(pct(one, e1[L], c, k)[0] - pct(one, own_e1[L], c, k)[0]),
                                              ci=[float(x) for x in np.quantile(pct(W, e1[L], c, k) - pct(W, own_e1[L], c, k), [0.025, 0.975])])
                                      for L in layers}
        out[name] = res
    b2 = out["block2"]
    out["above_100"] = dict(seed_mean=b2["mean"]["against_100"], per_seed={s: b2[s]["against_100"] for s in b2 if s.startswith("seed")},
                     significantly_above=out["instruments_hold"] and all(b2[s]["against_100"] == "above" for s in b2 if s.startswith("seed")),
                     block4_agrees=out["block4"]["mean"]["against_100"] == b2["mean"]["against_100"])
    if m == PUBLISHED[0]:
        s0 = b2["seed0"]; out["published_value"] = dict(published=PUBLISHED[1], ours=s0["Y_star"], interval=[s0["lower"], s0["upper"]],
                                          consistent_with=bool(s0["lower"] <= PUBLISHED[1] <= s0["upper"]))
    out["turning"] = {f"seed{s}": {L: cos_bounds(runs[s][0]["layers"][str(L)]["cos_init_kept"], runs[s][0]["layers"][str(L)]["cos_best_last"])
                                   for L in layers if "cos_init_kept" in runs[s][0]["layers"][str(L)]} for s in runs}
    if m in TWIN and (RES / TWIN[m]).is_file():
        T0 = json.load(open(RES / TWIN[m]))
        out["twin_seed0_difference"] = {L: float(J0["layers"][str(L)]["last_das_E2"] - T0["layers"][str(L)]["last_das_E2"])
                                        for L in layers if str(L) in T0["layers"]}
    return out


failed = {}
report = dict(models={m: score_model(m, failed) for m in MODELS}, provenance_failures=failed)
dest = RES / "score_sentiment_reread.json"
json.dump(report, open(dest, "w"), indent=1, default=float)
for m, r in report["models"].items():
    if not r.get("built"):
        print(f"{m:14s} not built: {r['reason']}"); continue
    b = r["block2"]["mean"]
    print(f"{m:14s} n={r['n_test']:3d} instruments {'held' if r['instruments_hold'] else 'FAILED'}  L*={b['L_star']:2d}  "
          f"Y*={b['Y_star']:6.1f}  band [{b['lower']:6.1f}, {b['upper']:6.1f}] -> {b['against_100']}; per seed "
          f"{r['above_100']['per_seed']}" + (f"; published value {r['published_value']}" if "published_value" in r else ""))
print(f"written {dest}")

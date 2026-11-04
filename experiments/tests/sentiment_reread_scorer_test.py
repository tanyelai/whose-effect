"""Tests experiments/score/score_sentiment_reread.py on synthetic re-reads with known answers. Synthetic data only (no model, no outputs
of the runs); exits 1 on a failure.
The scorer re-reads the published sentiment direction: per layer, the share Y(L) of the clean-minus-corrupted logit difference
that the direction restores, with a circular block bootstrap and a band over the layers that counts the choice of layer (Section 7.3). It tests whether the published 109.8 for Pythia-1.4b lies inside the band at
the chosen layer ("consistent with"; output key published_value) and whether the band's lower bound lies above 100 (above_100).
22 test rows with k_r = -c_{r+1}, 17 layers, three seeds. Planted: one model whose DAS restores 150% at one layer (80% at the
others) must be "above" in every seed; Pythia-1.4b's planted 110% must be consistent with 109.8; a broken own E2 (the network's
own interchange, which must restore 100%) must leave the model unjudged; a run trained for 32 epochs must not be scored
(provenance: every output must name the scorer's fixed training settings); models without outputs are reported as not built; a
circular block of two draws neighbouring rows together. Null (100% at every layer), 120 replicates each, with neighbouring
layers correlated 0.95 and with independent layers: the band says "above" at most at the 5% level (at most 10 of 120, allowing
the binomial spread), and the fixed-layer interval at the selected layer says it more often (the choice the band counts).
Usage: python -I experiments/tests/sentiment_reread_scorer_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from owneffect.stats import block_counts  # noqa: E402

# the sha256 of Tigges et al.'s prompts.yaml (experiments/lib/pins.py), which every output must name
YAML_SHA = "592b699a2cfa0d3a5f489bd6167e58a0d27239cd77e7252eb4bfc2f297bd2d7a"
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        fails.append(name)


def write(d, m, profile, rng, n=22, layers=17, noise=8.0, rho=0.0, broken_own=False, epochs=64):
    """Three seeds' outputs and rows for model m: at layer L the patched logit differences restore profile(L) percent plus
    noise (rho: the noise's correlation between neighbouring layers); broken_own moves own E2 off 100 at layer 3; epochs is
    seed 1's training length."""
    c = rng.normal(3.0, 0.6, n); k = -np.roll(c, -1)                     # the corrupted input is the next row's clean one
    for s in (0, 1, 2):
        lay, rows = {}, {}
        z = rng.normal(size=n)
        for L in range(layers):
            z = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=n)        # per-row noise, correlated across layers
            y = profile(L) + noise * z
            P = k + y / 100 * (c - k)
            own1 = k + 0.6 * (c - k); e1 = k + 0.7 * (c - k)
            rows[str(L)] = dict(das_E2=P.tolist(), last_das_E2=P.tolist(), das_E1=e1.tolist(), last_das_E1=e1.tolist(), own_E1=own1.tolist())
            lay[str(L)] = dict(own_E2=100.0 + (5.0 if broken_own and L == 3 else 0.0), own_vs_logits=0.0, own_E1=60.0,
                               last_das_E2=float(100 * (P.mean() - k.mean()) / (c.mean() - k.mean())), cos_init_kept=0.8, cos_best_last=0.99)
        J = dict(layers=lay, seed=s, objective="lin", epochs=epochs if s == 1 else 64, lr=1e-3, yaml_sha256=YAML_SHA)
        json.dump(J, open(d / f"sentiment_reread_{m}_seed{s}.json", "w"))
        json.dump(dict(base=dict(test=dict(clean=c.tolist(), corrupt=k.tolist())), layers=rows),
                  open(d / f"sentiment_reread_{m}_seed{s}_rows.json", "w"))


def run(d, b=2000):
    p = subprocess.run([sys.executable, "-I", str(ROOT / "experiments/score/score_sentiment_reread.py"), f"--fake={d}", f"--b={b}"],
                       capture_output=True, text=True)
    return p, (json.load(open(d / "score_sentiment_reread.json")) if p.returncode == 0 else None)


rng = np.random.default_rng(1)
d = Path(tempfile.mkdtemp())
write(d, "pythia-410m", lambda L: 150.0 if L == 9 else 80.0, rng)
write(d, "pythia-1.4b", lambda L: 110.0 if L == 12 else 70.0, rng)
write(d, "gpt2", lambda L: 80.0, rng, broken_own=True)
write(d, "pythia-1b", lambda L: 80.0, rng, epochs=32)
p, out = run(d)
check("scorer runs", out is not None, (p.stdout + p.stderr)[-600:])
if out:
    M = out["models"]; r = M["pythia-410m"]
    check("a planted 150% layer is 'above' in every seed", r["above_100"]["significantly_above"] and r["block2"]["mean"]["L_star"] == 9, json.dumps(r["above_100"]))
    check("Pythia-1.4b's planted 110% is consistent with 109.8", M["pythia-1.4b"]["published_value"]["consistent_with"], json.dumps(M["pythia-1.4b"]["published_value"]))
    check("a broken own E2 takes the reading away", not M["gpt2"]["instruments_hold"] and M["gpt2"]["above_100"]["seed_mean"].startswith("not read"),
          json.dumps(M["gpt2"]["instruments"]))
    check("a run trained for 32 epochs is not scored", not M["pythia-1b"]["built"] and "epochs" in M["pythia-1b"]["reason"], str(M["pythia-1b"]))
    check("models without outputs are reported as not built", not M["SmolLM2-360M"]["built"], str(M["SmolLM2-360M"]))
    W = block_counts(22, 2, 500, 0)
    starts = np.random.default_rng(0).integers(0, 22, size=(500, 11))
    adj = all(np.array_equal(np.sort(np.concatenate([[s, (s + 1) % 22] for s in starts[b]])), np.sort(np.repeat(np.arange(22), W[b]))) for b in range(500))
    check("a circular block of two draws neighbouring rows together", adj)
shutil.rmtree(d)
for label, rho in (("neighbouring layers correlated 0.95", 0.95), ("independent layers", 0.0)):
    band = fixed = 0; R = 120
    for rep in range(R):
        dn = Path(tempfile.mkdtemp()); write(dn, "pythia-160m", lambda L: 100.0, np.random.default_rng(100 + rep), rho=rho)
        _, on = run(dn, b=2000)
        if on and on["models"]["pythia-160m"]["built"]:
            b = on["models"]["pythia-160m"]["block2"]["seed0"]
            band += b["against_100"] == "above"; fixed += b["fixed_layer"][0] > 100
        shutil.rmtree(dn)
    check(f"null, {label}: the band says 'above' at most at the 5% level (<= 10 of {R})", band <= 10, f"{band}/{R}")
    check(f"null, {label}: the fixed-layer interval says it more often", fixed > band, f"fixed {fixed}/{R}")
print("\nSENTIMENT RE-READ SCORER OK" if not fails else "SENTIMENT RE-READ SCORER FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)

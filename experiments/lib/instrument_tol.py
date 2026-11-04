"""The instruments' tolerance (App. C.1, "Numerical checks"): an instrument compares two computations of quantities that are equal
in exact arithmetic (the readout's logits and the model's; T from the readout and from the forward pass; the network's own
interchange at the readout cell and T; the logits of an unpatched run with attention frozen and of the base run). In float32,
the paper's precision, the tolerance is a scale of the rounding, not a rigorous bound: gamma_D * S, with
gamma_D = D u / (1 - D u), u = 2^-24, D the hidden size (the a priori factor of one dot product of length D; Higham, Accuracy
and Stability of Numerical Algorithms, 2002, Section 3.1) and S the model's largest |label logit| on a sample of test inputs
(experiments/scripts/logit_scale.py, experiments/results/logit_scale.json). The rigorous bound has sum |w_i h_i| >= |o| in place of S, and some
instruments take the largest difference over the whole vocabulary, so the scale errs on the strict side. Two computations of one
logit may differ by twice the scale, of an effect (a difference of two log odds, four logits on each side) by eight times:
tol(kind) = k * gamma_D * S, k = 2 for a logit and 8 for an effect. An absolute tolerance such as 1e-3 does not scale with the
logits: in Pythia-160m, whose logits lie near 840, rounding alone gives deviations of 19-26 steps of 2^-14 at the readout,
more than 1e-3. A gross patching error, an effect off by a large fraction of T, still fails in every model; in Pythia-160m,
the model with the widest tolerance, an effect error below 8 gamma_D S = 0.31 log odds would pass.
A model absent from the file stops a scorer of real outputs; with fake=True (the scorers' synthetic tests) the tolerance is an
absolute 1e-3.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = {"logit": 2, "effect": 8}
_SCALES = None


def scales(path=None):
    global _SCALES
    if _SCALES is None or path is not None:
        p = Path(path) if path else ROOT / "experiments/results/logit_scale.json"
        _SCALES = json.load(open(p))["models"] if p.is_file() else {}
    return _SCALES


def tol(model, kind, fake=False):
    """The tolerance for an instrument of the given kind ("logit" or "effect") in the model (its tag, e.g. "pythia-160m", or
    its Hugging Face id). fake: a synthetic model of a scorer's test, whose tolerance is an absolute 1e-3."""
    tag = str(model).rstrip("/").split("/")[-1]
    s = scales().get(tag)
    if s is None:
        if fake:
            return 1e-3
        raise SystemExit(f"instrument_tol: no logit scale for {tag!r} in experiments/results/logit_scale.json (run experiments/scripts/logit_scale.py)")
    return K[kind] * s["gamma_D"] * s["max_abs_label_logit"]

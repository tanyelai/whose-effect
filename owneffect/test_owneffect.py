"""Checks of OwnEffect and its family layer on tiny random models of every supported family: GPT-2, GPT-NeoX, Llama (with
grouped-query attention), Qwen2 (grouped-query) and Gemma-2 (grouped-query, a sliding window and soft-capping). No weights are
downloaded; the models use GPT-2's tokenizer. Three pairs and their mirrors, in the paper's format: region 0 is the
start-of-text token, the same in base and source, and each of the regions 1-4 is one token.

1. Swapping regions 1-4 of a layer, every position where base and source differ, a complete cut, gives T: the network's own
   interchange equals T at every layer below the readout.
2. A rank-one swap along a pair's own difference equals the whole-state swap at that position (one pair).
3. Frozen attention without a patch reproduces the unpatched logits.
4. The family's readout map, applied to the last block's raw output, reproduces the model's logits.
5. Subspaces: a one-column matrix gives the rank-one effect, and a basis whose span holds the pair's own difference gives
   the whole-state swap.
6. readout(): the instruments are near zero; on the mirrored two-label set the trace test and the angle are given, the
   angle's top exactly where the angle is below 90 degrees; along a pair's own readout difference the effect is T (one pair).
7. The Rayleigh range theorem's part (iv) for one pair, exactly: lambda_1 = (w . Delta + |w| |Delta|) / 2, at a cell and at
   the readout; and the top eigenvector's first-order value is lambda_1.
8. cut(): the complete-cut identity at every layer below the readout (every pair is aligned).
9. report() and range() return their fields, with ordered intervals, and a direction's first-order value lies between
   lambda_D and lambda_1.
10. own() and report() with a context chosen elsewhere measure G* in that context without choosing.
11. stats.readings() gives the readings of the paper's figure on what to report, and report() returns those of its intervals.
12. With a patch, frozen attention changes the effect at some cell (freezing is not a no-op).
13. The refusals: region 0, a layer below -1, B = 0, a bad context, and a region that is empty in some pair, which
    prepare() lists in .empty and regions() leaves out.
14. close() removes the hooks the object added.
Usage: python -I owneffect/test_owneffect.py [cuda]   (the first run downloads GPT-2's tokenizer, about 1 MB). Raises
AssertionError on a failure; prints "ok" when every check holds.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import (AutoModelForCausalLM, AutoTokenizer, Gemma2Config, GPT2Config, GPTNeoXConfig,  # noqa: E402
                          LlamaConfig, Qwen2Config)

from owneffect import OwnEffect, families, stats  # noqa: E402

S0 = "<|endoftext|>"
PAIRS = [dict(base=[S0, "The", " cat", " sat", " on"], src=[S0, "A", " dog", " sat", " on"], base_label=" the", src_label=" a"),
         dict(base=[S0, "A", " dog", " ran", " to"], src=[S0, "The", " cat", " ran", " to"], base_label=" a", src_label=" the"),
         dict(base=[S0, "The", " man", " went", " to"], src=[S0, "A", " boy", " went", " to"], base_label=" the", src_label=" a")]
PAIRS += [dict(base=p["src"], src=p["base"], base_label=p["src_label"], src_label=p["base_label"]) for p in PAIRS]   # mirrors
warnings.filterwarnings("ignore", message="the pairs miss the paper's population rule")     # a handful of pairs, by design
warnings.filterwarnings("ignore", message="the mean whole-input effect T")                 # random models
DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
tok = AutoTokenizer.from_pretrained("gpt2")
V = len(tok)
CONFIGS = {
    "gpt2": GPT2Config(vocab_size=V, n_layer=3, n_embd=64, n_head=4),
    "gpt_neox": GPTNeoXConfig(vocab_size=V, num_hidden_layers=3, hidden_size=64, num_attention_heads=4, intermediate_size=128),
    "llama": LlamaConfig(vocab_size=V, num_hidden_layers=3, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128, rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                              "high_freq_factor": 4.0, "original_max_position_embeddings": 64}),
    "qwen2": Qwen2Config(vocab_size=V, num_hidden_layers=3, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=128),
    "gemma2": Gemma2Config(vocab_size=V, num_hidden_layers=3, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, intermediate_size=128, sliding_window=2, query_pre_attn_scalar=16,
                           attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
}


def tiny(cfg):
    """A random model of the config, float32, eager attention, eval mode, without gradients on its weights."""
    torch.manual_seed(0); cfg._attn_implementation = "eager"
    m = AutoModelForCausalLM.from_config(cfg).float().eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# 11. the readings on synthetic intervals (own, effect, excess, ratio)
R = stats.readings
assert R((0.2, 0.4), (0.5, 0.9), (0.1, 0.6), (1.2, 2.5))["reading"] == "excess"
assert R((0.2, 0.4), (0.3, 0.5), (-0.1, 0.2), (0.8, 1.3))["reading"] == "within"
assert R((0.2, 0.4), (0.0, 0.2), (-0.3, -0.1), (0.2, 0.7))["reading"] == "below"
r = R((-0.1, 0.1), (0.3, 0.6), (0.2, 0.6))
assert r["not_evidence_of_use"] and r["off_route"] and r["reading"] == "off-route" and not r["carried"]
r = R((-0.3, -0.1), (0.2, 0.4), (0.0, 0.4))                       # G* wholly below zero is "not above zero" too
assert r["not_evidence_of_use"] and not r["off_route"] and r["reading"] == "no evidence of use"
r = R((-0.1, 0.1), (-0.4, -0.2), (-0.4, -0.1))                    # an e below zero is no reading
assert not r["not_evidence_of_use"] and r["reading"] is None


for fam, cfg in CONFIGS.items():
    model = tiny(cfg)
    assert families.name(model) == fam
    rt = OwnEffect(model, tok); dropped = rt.prepare(PAIRS)
    assert dropped == 0, dropped
    L = len(rt.blocks)
    # 1. every position of the layer: a complete cut
    for layer in range(L - 1):
        whole = rt.effect(layer, [1, 2, 3, 4])[0]
        assert abs(whole - rt.T) < 1e-4 * max(1.0, abs(rt.T)), (fam, layer, whole, rt.T)
    # 2. rank one along the pair's own difference at layer 0, region 1
    one = OwnEffect(model, tok); one.prepare(PAIRS[:1])
    r = one.rows[0]; blk = 1                                                # layer 0 = block 1's input; full attention in Gemma-2
    with torch.no_grad():
        hb = model(torch.tensor([r["b"]], device=DEV), output_hidden_states=True).hidden_states[blk][0, r["be"][1]]
    d = (r["hs"][blk, r["se"][1]] - hb).cpu().numpy()
    e_d, e_full = one.effect(0, 1, d)[0], one.effect(0, 1)[0]
    assert abs(e_d - e_full) < 1e-4 and abs(e_full) > 1e-4, (fam, e_d, e_full)
    # 3. frozen attention without a patch; 4. the readout map against the logits
    ids = torch.tensor([rt.rows[0]["b"], rt.rows[1]["b"]], device=DEV)
    att = families.attention(model)
    with torch.no_grad():
        att.update(mode="record", store={}); z0 = model(ids, use_cache=False).logits
        att["mode"] = "frozen"; z1 = model(ids, use_cache=False).logits; att["mode"] = "normal"
        raw = {}
        hk = rt.blocks[-1].register_forward_hook(lambda m, a, o: raw.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        z2 = model(ids, use_cache=False).logits; hk.remove()
        z3 = families.readout(model)(raw["h"])
    assert float((z0 - z1).abs().max()) < 1e-5, (fam, "frozen attention without a patch changed the logits")
    assert float((z2 - z3).abs().max()) < 1e-4, (fam, "readout map differs from the model's logits")
    # 5. subspaces
    e_col = one.effect(0, 1, d[:, None])[0]
    e_k = one.effect(0, 1, np.stack([d, np.random.default_rng(0).standard_normal(d.shape)], 1))[0]
    assert abs(e_col - e_d) < 1e-5 and abs(e_k - e_full) < 1e-4, (fam, e_col, e_d, e_k, e_full)
    # 6. the readout
    ro = rt.readout(B=100)
    assert ro["instruments"]["readout_vs_logits"] < 1e-4 and ro["instruments"]["T_vs_forward"] < 1e-4, (fam, ro["instruments"])
    tt, an, rr = ro["trace_test"], ro["angle"], ro["range"]
    assert tt is not None and abs(tt["simpson"] - (rr["trM_base"] + 2 * rr["trM_mid"]) / 3) < 1e-9, (fam, tt)
    assert an is not None and 0 <= an["theta_deg"] <= 180 and (an["top"] is None) == (an["theta_deg"] >= 90), (fam, an)
    r0 = one.rows[0]
    ro1 = one.readout(direction=(r0["hs_out"] - r0["hb_out"]).cpu().numpy(), B=10)
    assert abs(ro1["direction"]["effect"]["value"] - 1.0) < 1e-4, (fam, ro1["direction"])
    # 7. the Rayleigh range theorem's part (iv), one pair: at a cell (grads()) and at the readout (the gradient of z through
    #    the readout map at the swap's midpoint, with the pair's difference)
    for W, Dl in (one.grads(0, 1, "mid"),
                  (torch.autograd.functional.jacobian(lambda H: (lambda o: o[0, r0["ys"]] - o[0, r0["yb"]])(one.read(H)),
                                                      (r0["hb_out"] + 0.5 * (r0["hs_out"] - r0["hb_out"]))[None]).reshape(1, -1),
                   (r0["hs_out"] - r0["hb_out"])[None])):
        sp = stats.spectrum(W, Dl); w, dl = W[0].double(), Dl[0].double()
        want = float(0.5 * (w @ dl + w.norm() * dl.norm()))
        assert abs(sp["lam1"] - want) < 1e-6 * max(1.0, abs(want)), (fam, sp["lam1"], want)
        v = torch.as_tensor(sp["v1"], dtype=W.dtype, device=W.device)
        assert abs(float((W @ v) * (Dl @ v)) - want) < 1e-5 * max(1.0, abs(want)), (fam, "v1's first-order value")
    # 8. the complete-cut proposition through cut()
    for layer in range(L - 1):
        c = rt.cut(layer)
        assert c["pairs"] == len(PAIRS) and c["max_diff"] < 1e-4 * max(1.0, abs(rt.T)), (fam, layer, c)
    # 9. report() and range(); the direction's first-order value within [lambda_D, lambda_1]
    rep = rt.report(0, 1, d, partners=[2, 3, 4], B=100)
    for k in ("own", "effect", "excess", "frozen"):
        lo, hi = rep[k]["interval"]; assert lo <= hi, (fam, k, rep[k])
    # 11 (in report()): the readings are those of the returned intervals
    assert rep["readings"] == stats.readings(rep["own"]["interval"], rep["effect"]["interval"], rep["excess"]["interval"],
                                             rep["ratio"] and rep["ratio"]["interval"]), (fam, rep["readings"])
    assert (rep["ratio"] is not None) == (rep["own"]["interval"][0] > 0), (fam, "e / G* only where G* is above zero")
    # 10. a context chosen elsewhere: passed back, own()'s context gives the same G*; "alone" is the cell's own swap; a given
    #     partner y is the swap with y minus y alone; report() keeps a passed context
    g_best, ctx = rt.own(0, 1, partners=[2, 3, 4])
    assert rt.own(0, 1, context=ctx) == (g_best, ctx), (fam, ctx)
    assert abs(rt.own(0, 1, context="alone")[0] - rt.effect(0, 1)[0]) < 1e-9, fam
    g2 = rt.own(0, 1, context=2, per_pair=True)[0]
    assert np.allclose(g2, rt.effects(0, [1, 2])[0] - rt.effects(0, 2)[0]), fam
    assert rt.report(0, 1, d, B=100, context=2)["cell"]["context"] == "with region 2"
    rg = rt.range(0, 1, direction=d, B=100)
    lamD_, fo_, lam1_ = (x * rt.T for x in (rg["lambdaD"], rg["direction"]["first_order"], rg["lambda1"]))   # log odds
    assert lamD_ - 1e-9 <= fo_ <= lam1_ + 1e-9, (fam, lamD_, fo_, lam1_)
    assert rg["lambda1_interval"][0] <= rg["lambda1_interval"][1]
    # 12. with a patch, freezing attention changes the effect somewhere
    assert any(abs(rt.effect(lr, g)[0] - rt.effect(lr, g, frozen=True)[0]) > 1e-6 for lr in range(L - 1) for g in (1, 2, 3)), fam
    # 13. the refusals
    for call in (lambda: rt.effect(0, 0), lambda: rt.effect(-2, 1), lambda: rt.report(0, 1, d, B=0),
                 lambda: rt.own(0, 1, context=1), lambda: rt.own(0, 1, context="with")):
        try:
            call(); raise AssertionError((fam, "accepted a call it should refuse"))
        except ValueError:
            pass
    ex = OwnEffect(model, tok)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ex.prepare([dict(base=[S0, "The", "", " sat", " on"], src=[S0, "A", "", " sat", " on"], base_label=" the", src_label=" a")])
    assert ex.empty == [2] and ex.regions() == [1, 3, 4] and any("empty" in str(w.message) for w in caught), (fam, ex.empty)
    try:
        ex.effect(0, 2); raise AssertionError((fam, "swapped an empty region"))
    except ValueError:
        pass
    # 14. close() removes the object's hooks
    n_hooks = lambda: sum(len(b._forward_pre_hooks) for b in rt.blocks)                    # noqa: E731
    before = n_hooks(); ex.close()
    assert n_hooks() == before - L, (fam, before, n_hooks())
    print(f"{fam:8s} complete cut = T at all {L - 1} inner layers (T {rt.T:.5f}); rank-one along the difference {e_d:.5f} = whole "
          f"swap {e_full:.5f}; frozen attention without a patch exact; readout map = logits; subspaces, "
          f"readout (e = T along the pair's own difference, trace test, angle), the range theorem's part (iv), cut(), report() "
          f"and its readings, range(), freezing with a patch, the refusals and close() hold")
print("ok")

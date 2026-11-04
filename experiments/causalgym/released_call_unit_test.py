"""CausalGym's released unit-locations call against the fixed one, on the same pairs: which token each call reads.

The paper's Appendix E.2 (token positions in the released code). A weight-free unit test on a tiny random GPT-NeoX (2 layers,
width 64, 4 heads, MLP width 128; Pythia-160m's configuration and tokenizer otherwise, torch seed 0), with CausalGym's own
Dataset, compute_pos and intervention_config and pyvene's interventions. It needs a clone of CausalGym's repository at
commit 0f3129f (github.com/aryamanarora/causalgym) and an environment with pyvene 0.1.6, as their requirements pin
(experiments/causalgym/rerun/requirements_frozen.txt); it changes the working directory to the clone. Tasks: cleft and
agr_sv_num_subj-relc.

For pairs whose base and source tokenise to different lengths before a region, run their two interventions (a collect
intervention and a VanillaIntervention at block 0's output) with
  released:  {"sources->base": ([None, pos[1]], pos)}                  (eval.py:31, train.py:62 at commit 0f3129f)
  corrected: {"sources->base": ([None, pos[0]], [pos[1], pos[1]])}     (experiments/causalgym/rerun/position_fix.patch)
where compute_pos returns pos = [source positions, base positions]. For each call, report the largest difference between
  (1) the collected activation and the base run's state at the base's index / at the source's index;
  (2) the patched run's logits and a manual patch writing, at the base's index, the source run's state at the source's
      index / at the base's index.
A difference of 0 says which token the call read. By default it stops after 12 checks per task (shifted pair x region); with
--all, every shifted (pair, region) of the sampled training batches (seed 42, as in the authors' training) is checked.
Usage: <Python of that environment> experiments/causalgym/released_call_unit_test.py <causalgym clone> [--all]
"""
import os
import sys

CG = os.path.abspath(sys.argv[1]); sys.path.insert(0, CG); os.chdir(CG)
import torch  # noqa: E402
import pyvene as pv  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, GPTNeoXForCausalLM  # noqa: E402
from data import Dataset  # noqa: E402
from interventions import intervention_config  # noqa: E402

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"); tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.intermediate_size = 2, 64, 4, 128
gpt = GPTNeoXForCausalLM(cfg).eval()
CALLS = {"released": lambda p: ([None, p[1]], p), "fixed": lambda p: ([None, p[0]], [p[1], p[1]])}
ALL = "--all" in sys.argv                                    # every shifted (pair, region) of the sampled training batches
for task in ("syntaxgym/cleft", "syntaxgym/agr_sv_num_subj-relc"):
    ds = Dataset.load_from(task)
    batches = ds.sample_batches(tok, 4, 100, "cpu", seed=42)
    found, worst, pairs = 0, {}, set()
    for b in batches:
        src_pos, base_pos = b.compute_pos("last")
        for pos_i in range(ds.first_var_pos, ds.length):
            mis = [i for i in range(len(b.pairs)) if base_pos[i][pos_i] != src_pos[i][pos_i]]
            if not mis:
                continue
            pos_interv = [[x[pos_i] for x in y] for y in b.compute_pos("last")]
            with torch.no_grad():
                hb = gpt(**b.base, output_hidden_states=True).hidden_states[1]
                hs = gpt(**b.src, output_hidden_states=True).hidden_states[1]
            outs = {}
            for cname, call in CALLS.items():
                conf = intervention_config("block_output", pv.VanillaIntervention, 0, 0)
                iv = pv.IntervenableModel(conf, gpt); iv.disable_model_gradients()
                with torch.no_grad():
                    outs[cname] = iv(b.base, [None, b.src], {"sources->base": call(pos_interv)}, output_original_output=True)
            for i in mis:
                pb, ps = base_pos[i][pos_i][0], src_pos[i][pos_i][0]
                manual = {}
                for name, s_idx in (("src@src", ps), ("src@base", pb)):
                    def hook(m, a, o, s_idx=s_idx, i=i, pb=pb):
                        h = (o[0] if isinstance(o, tuple) else o).clone(); h[i, pb] = hs[i, s_idx]
                        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                    hd = gpt.gpt_neox.layers[0].register_forward_hook(hook)
                    with torch.no_grad():
                        manual[name] = gpt(**b.base).logits[i]
                    hd.remove()
                for cname, (base_out, cf_out) in outs.items():
                    coll = base_out[-1][i][0]
                    row = {"collected=base@base": float((coll - hb[i, pb]).abs().max()),
                           "collected=base@src": float((coll - hb[i, ps]).abs().max()),
                           "patch=src@src": float((cf_out.logits[i] - manual["src@src"]).abs().max()),
                           "patch=src@base": float((cf_out.logits[i] - manual["src@base"]).abs().max())}
                    for k, v in row.items():
                        worst.setdefault(cname, {}).setdefault(k, []).append(v)
                    if found < 3:
                        print(f"{task.split('/')[-1]} span {ds.span_names[pos_i]}, pair {i}: base index {pb}, source index {ps}; "
                              f"{cname}: " + ", ".join(f"{k} {v:.1e}" for k, v in row.items()))
                pairs.add(str(b.pairs[i])); found += 1
            if found >= 12 and not ALL:
                break
        if found >= 12 and not ALL:
            break
    print(f"== {task}: {found} checks (shifted pair x region) on {len(pairs)} distinct shifted pairs")
    for cname, d in worst.items():
        print(f"   {cname:9s}: " + ", ".join(f"{k} max {max(v):.1e} / min {min(v):.1e}" for k, v in d.items()))

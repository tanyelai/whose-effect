"""The logits' scale per model, which sets the instruments' float32 tolerance (experiments/lib/instrument_tol.py; App. C.1,
"Instruments").

Inference only, in the paper's precision (float32, TF32 off, eager attention), on test inputs of the readout's data: CausalGym's
29 tasks from the pinned snapshot and every per-model task directory under experiments/data/<model>/, up to --rows=10 rows per task,
base and source inputs. Per model: the largest and the median |logit| of the two label tokens at the last position, the hidden
size D, and the float32 scale gamma_D * S with u = 2^-24, gamma_D = D u / (1 - D u) and S the largest |label logit|. This is
the a priori factor of one float32 dot product of length D (Higham, Accuracy and Stability of Numerical Algorithms, 2002,
Section 3.1) at that magnitude: a scale of a computed logit's rounding, not a rigorous bound (that has sum |w_i h_i| >= |o| in
place of S). The output field that holds gamma_D * S is named "bound".
Usage (from the repository root): python -I experiments/scripts/logit_scale.py --out=<json> [--models=<id>,...] [--rows=10]
       [--revision=<commit>]  (for a model that experiments/lib/pins.py does not pin)
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

TREE = Path.cwd()
sys.path.insert(0, str(TREE / "experiments/matrix"))
import registry  # noqa: E402

A = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
ROWS = int(A.get("rows", "10"))
SCALE_SERIES = {"Qwen/Qwen2.5-1.5B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"}
MODELS = A["models"].split(",") if "models" in A else list(registry.SMALL) + list(registry.LARGE) + sorted(SCALE_SERIES)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
U = 2.0 ** -24


def inputs(tag):
    """(base text, source text, base label, source label) for up to ROWS test rows of each CausalGym task and of each test.json
    under experiments/data/<model tag>/."""
    out = []
    cg = json.load(open(hf_hub_download("aryaman/causalgym", "test.json", repo_type="dataset", revision=registry.CAUSALGYM_REV)))
    for t in sorted({r["task"] for r in cg}):
        out += [r for r in cg if r["task"] == t][:ROWS]
    d = TREE / "experiments/data" / tag
    for f in sorted(d.glob("**/test.json")) if d.is_dir() else []:
        out += json.load(open(f))[:ROWS]
    return [("".join(r["base"]), "".join(r["src"]), r["base_label"], r["src_label"]) for r in out]


def first_token(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"][0]


res = {}
dev = "cuda"
for m in MODELS:
    tag = m.split("/")[-1]; rev = registry.REVISION.get(m) or A.get("revision")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(m, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(m, revision=rev, dtype=torch.float32, attn_implementation="eager").eval().to(dev)
    D = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    vals = []
    groups = {}                                   # inputs of equal token length run as one batch: no padding, no mask
    for b, s, yb, ys in inputs(tag):
        ib, is_ = first_token(tok, yb), first_token(tok, ys)
        for text in (b, s):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            groups.setdefault(len(ids), []).append((ids, ib, is_))
    with torch.no_grad():
        for items in groups.values():
            for k in range(0, len(items), 32):
                chunk = items[k:k + 32]
                o = model(torch.tensor([x[0] for x in chunk], device=dev)).logits[:, -1].double()
                for j, (_, ib, is_) in enumerate(chunk):
                    vals += [abs(float(o[j, ib])), abs(float(o[j, is_]))]
    vals.sort()
    S = vals[-1]; gamma = D * U / (1 - D * U)
    res[tag] = dict(model=m, revision=rev, D=int(D), inputs=len(vals) // 4, max_abs_label_logit=S,
                    median_abs_label_logit=vals[len(vals) // 2], gamma_D=gamma, bound=gamma * S, seconds=round(time.time() - t0, 1))
    print(json.dumps({tag: res[tag]}), flush=True)
    del model; torch.cuda.empty_cache()
meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=__import__("transformers").__version__,
            python=platform.python_version(), host=platform.node(), rows_per_task=ROWS, tf32=False, dtype="float32",
            finished=time.strftime("%Y-%m-%d %H:%M:%S"))
store = json.load(open(A["out"])) if os.path.isfile(A["out"]) else dict(models={}, runs=[])   # the models measured here replace theirs
store["models"].update(res); store.setdefault("runs", []).append(dict(meta, models=sorted(res)))
json.dump(store, open(A["out"], "w"), indent=1)
print("written", A["out"], "with", ", ".join(sorted(res)))

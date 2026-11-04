# OwnEffect

A learned direction, whether from DAS, a probe or a steering vector, is usually judged by its effect: swap that part of a
state from a source input into a base input, and see how far the answer moves. OwnEffect adds the number that effect should be
read against, the network's **own interchange**: the *whole* state swapped at the same place, in its best context. The paper
calls it $G^\ast$.

For any direction at any site, it reports the learned effect $e$, how far it lies beyond $G^\ast$, and how much of it runs
through attention, each with an interval.

## Install

From the repository's root:

```bash
pip install -e ".[stats]"     # torch, numpy, transformers >= 5.18; scipy for the bands
```

## A first report

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from owneffect import OwnEffect

name = "EleutherAI/pythia-160m"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", torch_dtype=torch.float32).to("cuda")

pair = {"base": ["<|endoftext|>", "No", " farmer", " that", " contacted", " no", " guard", " has", " caught"],
        "src":  ["<|endoftext|>", "The", " farmer", " that", " contacted", " no", " guard", " has", " caught"],
        "base_label": " any", "src_label": " some"}

with OwnEffect(model, tok) as oe:
    oe.prepare(pairs)                                   # many pairs like the one above, each with its mirror
    rep = oe.report(layer=9, region=1, direction=d)     # d: your direction at the output of block 9
    print(rep["readings"]["reading"], rep["effect"], rep["own"])
```

Each input is a list of spans, one per **region**: here region 0 is the start of text, region 1 the licensor (*No* or *The*),
region 2 *farmer*, and so on. A site is a **cell** `(layer, region)`: the output of block `layer` (counting from 0; `-1` is the
embeddings) at the region's last token. The direction is a vector, or a matrix whose columns span a subspace, in that state's
space.

## What goes into `prepare()`

- **Spans** are tokenised one by one without special tokens, so leading spaces matter. The first span is the start-of-text
  marker, written as the tokenizer's own start-of-text token where `<|endoftext|>` is not one token (Llama-3, Gemma-2).
  Region 0 is never swapped and never used as a context.
- **Every pair has the same regions.** A region that is empty in some pair is listed in `oe.empty` and refused.
- **Labels are single tokens**, written as the model sees them after the prompt. Pairs that break this are dropped with a
  warning, and `prepare()` returns how many.
- **Include each pair's mirror**, base and source exchanged along with their labels. A pair and its mirror are resampled
  together for the intervals, and the trace test needs them.

## Train on one set, measure on another

As in the paper: train the direction on training pairs, choose $G^\ast$'s context on training pairs too, and measure on test
pairs.

```python
with OwnEffect(model, tok) as tr:
    tr.prepare(train_pairs)
    _, context = tr.own(layer=9, region=2)          # "alone", or "with region y"

oe = OwnEffect(model, tok)
oe.prepare(test_pairs)
rep = oe.report(layer=9, region=2, direction=d, context=context)
```

Without `context=`, the context is chosen on the pairs being measured, which flatters $G^\ast$ when they are few.

## Reading the report

Everything is in units of $T$, the mean effect of switching the whole input, with 95% intervals from 2000 resamples of
pairs and their mirrors (`B`, `seed` and `level` change them).

| field | meaning |
|---|---|
| `own` | $G^\ast$, the network's own interchange at the cell |
| `effect` | $e$, your direction's effect |
| `excess` | $e - G^\ast$ |
| `ratio` | $e / G^\ast$, when $G^\ast$'s interval lies above zero |
| `frozen` | $e$ with every attention pattern held at the base run's |
| `frozen_removed` | the share of $e$ that freezing attention removes, when $e$'s interval excludes zero |
| `iia` | the share of pairs whose patched answer prefers the source label |
| `population` | whether the pairs meet the paper's rule, at least 20 pairs with their mirrors and $T$'s interval above zero; a warning says when they do not |

`rep["readings"]["reading"]` sums it up:

- **`"below"`, `"within"`, `"excess"`**: $G^\ast$ is clearly above zero, and $e / G^\ast$ lies below one, holds one, or lies above it.
  Above one, the objective found a direction beyond what the network's own interchange carries.
- **`"no evidence of use"`**: $G^\ast$ is not clearly above zero but $e$ is. Then $e$ says nothing about the network using the
  site; `frozen_removed` says how much of it runs through attention.
- **`"off-route"`**: as above, and $e$ also lies clearly beyond $G^\ast$.
- **`None`**: neither is clearly above zero.

## At the readout

`oe.readout(direction=d)` works on the last block's output at the last position, before the final norm. Swapping the whole
state there is the same as switching the input, so $e / T$ is the reading. With two answers shared by every pair, it also
returns the angle $\theta$ between the readout direction and the input's change, and the top it predicts before any
training, $\lambda_1 / \mathrm{tr}\,M = (1 + 1/\cos\theta)/2$ to first order. The trace test and two instruments that
should be near zero come with it.

## Further tools

- `oe.range(layer, region, direction=d)`: the first-order range of a cell, from $\lambda_D$ to $\lambda_1$, and where `d`
  lies in it beside its exact effect.
- `oe.cut(layer)`: swapping every position of a layer must give $T$; a quick check on a new model.
- `oe.effects(...)`, `oe.own(...)`, `oe.grads(...)`: the per-pair swaps, $G^\ast$ with its context, and the gradients behind the
  range, in log odds.

## Requirements and cost

GPT-2, GPT-NeoX (Pythia), Llama, Qwen2 and Gemma-2 are supported. The model needs eager attention, because freezing attention
replaces that kernel, and must sit on one device; the paper's tolerances are for float32. OwnEffect hooks every block; `close()`,
or a `with` block, removes the hooks and the cached states. The patched attention kernel stays for the process, and computes
the unchanged kernel unless a swap freezes attention.

A swap costs two forward passes per pair, in batches of 25; choosing $G^\ast$'s context adds four per candidate region.
`range()` takes one backward pass per pair. `prepare()` caches the source run's state at every block and position, so memory
grows with pairs × blocks × tokens × width.

## Check an install

```bash
python -I owneffect/test_owneffect.py cuda      # or cpu
```

It builds tiny random models of every family (downloading GPT-2's tokenizer, about 1 MB, the first time) and checks the
identities the paper relies on, the readings and the refusals. It prints `ok`. Against the paper's saved results on GPT-2,
`experiments/scripts/owneffect_repro.py` checks the library on a real model.

Please cite the paper ([`CITATION.cff`](../CITATION.cff)). Apache-2.0.

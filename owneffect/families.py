"""Model families: where the decoder blocks are, the readout map from the last block's output to the logits, how the start
of text is written, and how attention is frozen. OwnEffect and the experiment scripts (experiments/scripts) take these from here.

Supported: GPT-2, GPT-NeoX (Pythia), Llama, Qwen2 and Gemma-2 (Hugging Face model_type gpt2, gpt_neox, llama, qwen2 and
gemma2), loaded with attn_implementation="eager". Layer l, in OwnEffect and in the paper, is the output of block l, the input
to block l + 1 (hidden_states[l + 1]).
"""
import importlib

import torch

# model_type -> (the attribute path of the decoder blocks, that of the final norm)
_PATHS = {"gpt2": ("transformer.h", "transformer.ln_f"),
          "gpt_neox": ("gpt_neox.layers", "gpt_neox.final_layer_norm"),
          "llama": ("model.layers", "model.norm"),
          "qwen2": ("model.layers", "model.norm"),
          "gemma2": ("model.layers", "model.norm")}
START = "<|endoftext|>"                     # CausalGym's start-of-text marker, written into the first span


def _attr(obj, path):
    """The attribute at a dotted path, such as "transformer.h"."""
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def name(model):
    """The model's family, its config.model_type; raises ValueError for a family that is not supported."""
    fam = model.config.model_type
    if fam not in _PATHS:
        raise ValueError(f"model family {fam!r} is not supported (supported: {', '.join(_PATHS)})")
    return fam


def blocks(model):
    """The decoder blocks, in order. The input to block b is Hugging Face hidden_states[b]."""
    return _attr(model, _PATHS[name(model)][0])


def readout(model, private_copy=False):
    """The readout map, from the last block's raw output to the logits: the final norm, the unembedding and, where the config
    sets final_logit_softcapping (Gemma-2), the soft cap cap * tanh(logits / cap). private_copy=True reads out through a
    private, contiguous copy of the unembedding weight and bias (the same values, no gradient) instead of the model's own
    module. OwnEffect uses the copy on CPU: there a backward pass through GPT-2's unembedding weight, which is tied to the
    input embedding, ended in a bus error, and the copy avoids it."""
    norm, unembed = _attr(model, _PATHS[name(model)][1]), model.get_output_embeddings()
    if private_copy:
        lin = torch.nn.Linear(unembed.in_features, unembed.out_features, bias=unembed.bias is not None).to(unembed.weight.device)
        lin.weight = torch.nn.Parameter(unembed.weight.detach().clone().contiguous(), requires_grad=False)
        if unembed.bias is not None:
            lin.bias = torch.nn.Parameter(unembed.bias.detach().clone(), requires_grad=False)
        unembed = lin
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:
        return lambda h: cap * torch.tanh(unembed(norm(h)) / cap)
    return lambda h: unembed(norm(h))


def start_of_text(tok):
    """The function applied to every span before tokenising. A tokenizer that encodes CausalGym's start-of-text marker
    "<|endoftext|>" as one token (GPT-2's, Pythia's, Qwen2's) keeps it; for the others (Llama-3's, Gemma-2's) the marker is
    replaced by the tokenizer's own start-of-text token, tokenizer.bos_token. Raises ValueError when the marker is not one
    token and the tokenizer has no start-of-text token."""
    if len(tok(START, add_special_tokens=False)["input_ids"]) == 1:
        return lambda s: s
    if tok.bos_token is None:
        raise ValueError(f"{tok.name_or_path}: no start-of-text token, and {START!r} is not one token")
    bos = tok.bos_token
    return lambda s: s.replace(START, bos)


def encode(tok, spans, fix=None):
    """Token ids of the concatenated spans, each tokenised on its own (after fix, if given) with no special tokens added,
    and the index of each span's last token."""
    ids, ends = [], []
    for sp in spans:
        ids += tok(fix(sp) if fix else sp, add_special_tokens=False)["input_ids"]; ends.append(len(ids) - 1)
    return ids, ends


def attention(model):
    """The switch of the family's eager attention kernel, a dict `state`. The first call replaces the family's
    eager_attention_forward in transformers, so one switch serves every model of the family in this process. Modes, set in
    state["mode"]:
      "normal": the kernel unchanged;
      "record": also store every layer's attention pattern of the run in state["store"] (by layer index);
      "frozen": use the stored patterns in place of the computed ones, in every layer;
      "frozen_row": use the stored pattern for the query row state["row"] only (one position per batch element), in the
        layers from state["first"] on (default 0); with the last position as the row, the readout position's own attention.
    A frozen run takes the patterns of the last recorded run, so it needs the same batch and sequence shape. Raises
    ValueError unless the model was loaded with attn_implementation="eager", and for a GPT-2 configuration with
    reorder_and_upcast_attn, whose attention bypasses that kernel."""
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        raise ValueError("load the model with attn_implementation='eager' (frozen attention patches the eager kernel)")
    if getattr(model.config, "reorder_and_upcast_attn", False):
        raise ValueError("reorder_and_upcast_attn=True computes GPT-2's attention outside the eager kernel, so it cannot be "
                         "frozen; load the model with reorder_and_upcast_attn=False")
    mod = importlib.import_module(f"transformers.models.{name(model)}.modeling_{name(model)}")
    if not hasattr(mod, "_owneffect_state"):
        orig, state = mod.eager_attention_forward, {"mode": "normal", "store": {}}

        def eager(module, query, key, value, attention_mask, *a, **kw):
            out, w = orig(module, query, key, value, attention_mask, *a, **kw)
            if state["mode"] == "record":
                state["store"][module.layer_idx] = w.detach()
            elif state["mode"] == "frozen" or (state["mode"] == "frozen_row" and module.layer_idx >= state.get("first", 0)):
                if state["mode"] == "frozen":
                    w = state["store"][module.layer_idx]
                else:
                    ar, q = torch.arange(w.shape[0], device=w.device), state["row"]
                    w = w.clone(); w[ar, :, q, :] = state["store"][module.layer_idx][ar, :, q, :]
                # grouped-query attention: repeat the value heads to the number of query heads
                v = value if value.shape[1] == w.shape[1] else value.repeat_interleave(w.shape[1] // value.shape[1], dim=1)
                out = torch.matmul(w.to(v.dtype), v).transpose(1, 2).contiguous()
            return out, w

        mod.eager_attention_forward = eager
        mod._owneffect_state = state
    return mod._owneffect_state

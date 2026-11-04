"""The published sentiment direction re-read (Section 7.3), for any supported family, on CPU or GPU.

Tigges et al.'s own procedure, rebuilt from their repository (curt-tigges/eliciting-latent-sentiment, 037cbc9):
  data      prompts.yaml word lists, " " prepended, kept only if one token for the model, deduplicated, cycled; simple_train
            n = min(len(positive_core_adjectives), len(negative_core_adjectives)) with the train adjectives, simple_test n =
            min of the test lists; template "I thought this movie was{ADJ}, I{VRB} it. \\nConclusion: This movie is"; a
            start-of-text token prepended; rows alternate positive, negative; corrupted = the next row; answers: the
            positive and negative answer tokens, paired in order, correct first for the clean row; logit difference = mean
            over the pairs.
  DAS       one direction per layer, trained at the ADJ position on simple_train on (patched - clean) / (corrupted - clean)
            (their loss; or --objective=jsd|sq), Adam lr 1e-3, 64 epochs, their batch sizes (128 / 64 / 32 for Pythia-160m /
            410m / 1.4b, else 32), no shuffling; a d x d rotation under orthogonal(use_trivialization=False), initialised by
            orthogonal_ after torch.manual_seed(seed) (0, theirs); the direction is W[:, 0]; gradient norm clipped at 1; kept
            = the epoch with the lowest summed training loss; the last epoch's direction (what their code returns) also read.
  patching  at one layer, corrupt + ((clean - corrupt) . d) d at (E1) the ADJ and VRB positions and (E2) every position;
            100 * (patched - corrupted) / (clean - corrupted) on simple_test.
  reference the network's own interchange at the same positions (E2: exactly 100%, the positive control).
Layer L is the input to block L (Hugging Face hidden_states[L]); L = NL is the last block's raw output. Table 10 numbers this
layer L (the blocks before the state); elsewhere the paper numbers it L - 1.
How this script runs it:
  - the family layer (owneffect.families): the blocks and the readout (final norm, unembedding, and Gemma-2's soft-capping)
    of every supported family. The blocks from L on run inside the model's own forward pass: a forward pre-hook replaces
    block L's whole input with the patched states, so the mask and the position embeddings are the model's own. The forward
    pass also runs the blocks before L on dummy token ids, and their output is discarded;
  - the start-of-text token by the family layer's rule: "<|endoftext|>" where it is one token, else the tokenizer's own;
  - the device. The rotation is initialised on the CPU generator and then moved, so every device starts from the same
    initial direction. --clone_unembed=1 reads out through a private copy of the unembedding (as in experiments/scripts/readout.py);
  - one instrument per layer: own_vs_logits, the maximum absolute difference between the clean run's last-position logits
    and the logits of the run from block L on the clean run's own states (test rows);
  - the output records the family, the seed, the objective, the start-of-text id, and the YAML's path and sha256.
Usage: python -I experiments/scripts/sentiment_reread.py --model=<hf id> [--revision=<sha>] --yaml=<their prompts.yaml>
       [--save_rows=1] [--seed=k] [--objective=lin|jsd|sq] [--epochs=64] [--lr=1e-3] [--clone_unembed=1] [--device=cuda]
       [--tag=<name>]
Outputs: experiments/results/sentiment_reread_<model tag>[_<tag>]<suffix>.json, where the suffix is _seed<k> when --seed is given,
then _<objective> unless lin. With --save_rows=1 the run also writes <same stem>_rows.json (per-row test logit differences,
and the clean and corrupted runs' rows) and <same stem>_dirs.npy (the kept and the last direction per layer).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from owneffect import families  # noqa: E402

_, args, model, tok = common.start("sentiment_reread", sys.argv[1:])
DEV, NAME, TAG = common.RUN["dev"], common.RUN["model"], common.RUN["tag"]
BATCH = {"pythia-160m": 128, "pythia-410m": 64, "pythia-1.4b": 32}.get(TAG, 32)   # their fit_directions.py
EPOCHS, LR = int(args.get("epochs", 64)), float(args.get("lr", 1e-3))
SEED = int(args.get("seed", 0))                       # their seed is 0
SFX = f"_seed{SEED}" if "seed" in args else ""     # the output name's suffix: _seed<k>, then _<objective> unless lin
OBJ = args.get("objective", "lin")                     # lin (their loss) | jsd (to the clean run) | sq (per row)
assert OBJ in ("lin", "jsd", "sq"), OBJ
SFX += "" if OBJ == "lin" else "_" + OBJ
YAML = args["yaml"]
Y = yaml.safe_load(open(YAML))
t0 = time.time()

BLOCKS, BASE = families.blocks(model), model.base_model
READ = families.readout(model, private_copy=args.get("clone_unembed") == "1")
NL, D = len(BLOCKS), model.config.hidden_size
_sot = families.encode(tok, [families.START], families.start_of_text(tok))[0]
assert len(_sot) == 1, f"start of text is {len(_sot)} tokens"
BOS = _sot[0]
TEMPLATE = "I thought this movie was{ADJ}, I{VRB} it. \nConclusion: This movie is"


def words(key):
    out = []
    for w in Y[key]:
        w = " " + str(w).strip()
        if len(tok(w, add_special_tokens=False)["input_ids"]) == 1 and w not in out:
            out.append(w)
    return out


W = {k: words(k) for k in ["positive_core_adjectives", "negative_core_adjectives", "positive_adjectives_train",
                           "negative_adjectives_train", "positive_adjectives_test", "negative_adjectives_test",
                           "positive_verbs", "negative_verbs", "positive_answer_tokens", "negative_answer_tokens"]}


def build(split):
    if split == "train":
        n = min(len(W["positive_core_adjectives"]), len(W["negative_core_adjectives"]))
        P, N = W["positive_adjectives_train"], W["negative_adjectives_train"]
    else:
        P, N = W["positive_adjectives_test"], W["negative_adjectives_test"]
        n = min(len(P), len(N))
    pv, nv = W["positive_verbs"], W["negative_verbs"]
    prompts = []
    for i in range(n):
        prompts.append(TEMPLATE.format(ADJ=P[i % len(P)], VRB=pv[i % len(pv)]))
        prompts.append(TEMPLATE.format(ADJ=N[i % len(N)], VRB=nv[i % len(nv)]))
    pa = [tok(w, add_special_tokens=False)["input_ids"][0] for w in W["positive_answer_tokens"]]
    na = [tok(w, add_special_tokens=False)["input_ids"][0] for w in W["negative_answer_tokens"]]
    k = min(len(pa), len(na))
    ans = torch.tensor([[[pa[j], na[j]] for j in range(k)] if r % 2 == 0 else [[na[j], pa[j]] for j in range(k)]
                        for r in range(len(prompts))], device=DEV)
    ids = [[BOS] + tok(s, add_special_tokens=False)["input_ids"] for s in prompts]
    assert len({len(x) for x in ids}) == 1, "prompts differ in length"
    clean = torch.tensor(ids, device=DEV)
    corrupt = torch.cat([clean[1:], clean[:1]])
    pre_adj = [BOS] + tok(TEMPLATE.split("{ADJ}")[0], add_special_tokens=False)["input_ids"]
    pre_vrb = [BOS] + tok(TEMPLATE.split("{VRB}")[0].format(ADJ=P[0]), add_special_tokens=False)["input_ids"]
    adj, vrb = len(pre_adj), len(pre_vrb)
    assert clean[0, adj] == tok(P[0], add_special_tokens=False)["input_ids"][0]
    assert clean[0, vrb] == tok(pv[0], add_special_tokens=False)["input_ids"][0]
    return dict(prompts=prompts, clean=clean, corrupt=corrupt, ans=ans, adj=adj, vrb=vrb, n=n)


SITE, RAW = {"L": None, "H": None}, {}


def pre_hook(mod, a, kw):
    """Block L's whole input replaced by the states SITE["H"]."""
    if "hidden_states" in kw:
        return a, dict(kw, hidden_states=SITE["H"])
    return (SITE["H"],) + tuple(a[1:]), kw


for i, blk in enumerate(BLOCKS):
    blk.register_forward_pre_hook(lambda m, a, kw, i=i: pre_hook(m, a, kw) if SITE["L"] == i else None, with_kwargs=True)
BLOCKS[-1].register_forward_hook(lambda m, i, o: RAW.__setitem__("h", o[0] if isinstance(o, tuple) else o))


def states(ids):
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack(list(hs[:NL]) + [RAW["h"]])                   # (NL+1, B, T, D): input to blocks 0..NL-1, then raw last


def run_from(L, H):
    """The last position's logits when the blocks from L on run on the states H (B, T, D): block L's input, or for L = NL
    the last block's raw output. The blocks run inside the model's own forward pass; the token ids only feed the blocks
    before L, whose output is replaced."""
    if L < NL:
        SITE.update(L=L, H=H)
        try:
            BASE(torch.zeros(H.shape[:2], dtype=torch.long, device=H.device), use_cache=False)
        finally:
            SITE.update(L=None, H=None)
        H = RAW["h"]
    return READ(H[:, -1])


def ld_rows(z, ans):
    ar = torch.arange(len(z), device=z.device)[:, None]
    return (z[ar, ans[:, :, 0]] - z[ar, ans[:, :, 1]]).mean(1)


def patch_dir(Hc, Hk, d, positions):
    """corrupt + ((clean - corrupt) . d) d at the given positions (None = all)."""
    H = Hk.clone()
    idx = slice(None) if positions is None else positions
    H[:, idx] = Hk[:, idx] + ((Hc[:, idx] - Hk[:, idx]) @ d)[..., None] * d
    return H


def own_E1_states(d_, L):
    """The network's own interchange at the ADJ and VRB positions: the clean run's whole state there."""
    at = torch.isin(torch.arange(d_["Sk"].shape[2], device=DEV), torch.tensor([d_["adj"], d_["vrb"]], device=DEV))
    return torch.where(at[None, :, None], d_["Sc"][L], d_["Sk"][L])


out = dict(model=NAME, family=families.name(model), epochs=EPOCHS, lr=LR, batch=BATCH, seed=SEED, objective=OBJ,
           start_of_text=BOS, yaml=YAML, yaml_sha256=hashlib.sha256(Path(YAML).read_bytes()).hexdigest(),
           words={k: v for k, v in W.items()}, layers={})
SAVE_ROWS = args.get("save_rows", "0") == "1"           # per-row logit differences and the kept and last directions
per_row, dirs = {}, []
data = {s: build(s) for s in ("train", "test")}
for s, d_ in data.items():
    d_["Sc"], d_["Sk"] = states(d_["clean"]), states(d_["corrupt"])
    with torch.no_grad():
        zc = model(d_["clean"]).logits[:, -1]; zk = model(d_["corrupt"]).logits[:, -1]
    d_["clean_ld"] = float(ld_rows(zc, d_["ans"]).mean()); d_["corr_ld"] = float(ld_rows(zk, d_["ans"]).mean())
    d_["clean_logp"] = torch.log_softmax(zc, -1); d_["clean_ld_rows"] = ld_rows(zc, d_["ans"]); d_["zc"] = zc
    out[s] = dict(n_pairs=d_["n"], rows=len(d_["prompts"]), adj=d_["adj"], vrb=d_["vrb"],
                  clean_ld=d_["clean_ld"], corr_ld=d_["corr_ld"])
print(json.dumps({k: out[k] for k in ("train", "test")}), flush=True)


def jsd(logp, logq):
    p, q = logp.exp(), logq.exp()
    logm = torch.log(0.5 * (p + q) + 1e-30)
    return 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))


def pct(z, d_):
    return 100 * (float(ld_rows(z, d_["ans"]).mean()) - d_["corr_ld"]) / (d_["clean_ld"] - d_["corr_ld"])


class Rot(torch.nn.Module):                                           # their RotateLayer under their parametrisation
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)                                          # their fit_rotation: torch.manual_seed(config.seed), 0
        w = torch.empty(D, D); torch.nn.init.orthogonal_(w)             # on the CPU generator: the same start on every device
        self.weight = torch.nn.Parameter(w.to(DEV))


for L in range(NL + 1):
    tr, te = data["train"], data["test"]
    rot = torch.nn.utils.parametrizations.orthogonal(Rot(), use_trivialization=False)
    opt = torch.optim.Adam(rot.parameters(), lr=LR)
    d0 = rot.weight[:, 0].detach().clone()
    rows, losses, kept = len(tr["prompts"]), [], []
    for _ in range(EPOCHS):
        ep_loss = 0.0
        for i in range(0, rows, BATCH):                                  # their DataLoader: no shuffling
            j = torch.arange(i, min(i + BATCH, rows), device=DEV)
            d = rot.weight[:, 0]
            H = patch_dir(tr["Sc"][L][j], tr["Sk"][L][j], d, [tr["adj"]])
            zp = run_from(L, H)
            if OBJ == "lin":
                pl = ld_rows(zp, tr["ans"][j]).mean()
                loss = (pl - tr["clean_ld"]) / (tr["corr_ld"] - tr["clean_ld"])  # their loss: 0 at clean, 1 at corrupted
            elif OBJ == "jsd":
                loss = jsd(torch.log_softmax(zp, -1), tr["clean_logp"][j]).mean()
            else:
                loss = ((ld_rows(zp, tr["ans"][j]) - tr["clean_ld_rows"][j]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rot.parameters(), 1.0); opt.step()
            ep_loss += float(loss.detach())
        losses.append(ep_loss)
        with torch.no_grad():
            kept.append(rot.weight[:, 0].detach().clone())               # their models.append(state_dict()) after the epoch
    best = int(np.argmin(losses))
    d = kept[best] / kept[best].norm()
    with torch.no_grad():
        z_own2 = run_from(L, te["Sc"][L].clone())
        row = dict(
            das_train_adj=pct(run_from(L, patch_dir(tr["Sc"][L], tr["Sk"][L], d, [tr["adj"]])), tr),
            das_E1=pct(run_from(L, patch_dir(te["Sc"][L], te["Sk"][L], d, [te["adj"], te["vrb"]])), te),
            das_E2=pct(run_from(L, patch_dir(te["Sc"][L], te["Sk"][L], d, None)), te),
            own_E1=pct(run_from(L, own_E1_states(te, L)), te),
            own_E2=pct(z_own2, te),
        )
    row.update(best_epoch=best, loss_first=losses[0], loss_best=losses[best], cos_init_kept=float(d0 @ d))
    dl = kept[-1] / kept[-1].norm()          # what their code returns: its saved state_dicts alias the live weights
    with torch.no_grad():
        row.update(last_das_train_adj=pct(run_from(L, patch_dir(tr["Sc"][L], tr["Sk"][L], dl, [tr["adj"]])), tr),
                   last_das_E1=pct(run_from(L, patch_dir(te["Sc"][L], te["Sk"][L], dl, [te["adj"], te["vrb"]])), te),
                   last_das_E2=pct(run_from(L, patch_dir(te["Sc"][L], te["Sk"][L], dl, None)), te),
                   loss_last=losses[-1], cos_best_last=float(d @ dl))
    row["own_vs_logits"] = float((z_own2 - te["zc"]).abs().max())       # instrument: the run from block L reproduces the clean run
    if SAVE_ROWS:
        with torch.no_grad():
            per_row[L] = {k: ld_rows(run_from(L, H), te["ans"]).tolist() for k, H in (
                ("das_E1", patch_dir(te["Sc"][L], te["Sk"][L], d, [te["adj"], te["vrb"]])),
                ("das_E2", patch_dir(te["Sc"][L], te["Sk"][L], d, None)), ("own_E1", own_E1_states(te, L)),
                ("last_das_E1", patch_dir(te["Sc"][L], te["Sk"][L], dl, [te["adj"], te["vrb"]])),
                ("last_das_E2", patch_dir(te["Sc"][L], te["Sk"][L], dl, None)))}
        dirs.append(np.stack([d.cpu().numpy(), dl.cpu().numpy()]))
    out["layers"][L] = row
    print(json.dumps({"L": L, **{k: round(x, 3) for k, x in row.items()}}), f"{time.time() - t0:.0f}s", flush=True)

res = common.REPO / "experiments/results"; res.mkdir(parents=True, exist_ok=True)
stem = str(res / f"sentiment_reread_{TAG}{'_' + args['tag'] if args.get('tag') else ''}{SFX}")
outs = [Path(stem + ".json")]
json.dump(out, open(outs[0], "w"), indent=1)
if SAVE_ROWS:
    with torch.no_grad():
        base = {s_: dict(clean=ld_rows(model(d_["clean"]).logits[:, -1], d_["ans"]).tolist(),
                         corrupt=ld_rows(model(d_["corrupt"]).logits[:, -1], d_["ans"]).tolist()) for s_, d_ in data.items()}
    json.dump(dict(base=base, layers=per_row), open(stem + "_rows.json", "w"))
    np.save(stem + "_dirs.npy", np.stack(dirs))
    outs += [Path(stem + "_rows.json"), Path(stem + "_dirs.npy")]
common.finish(outs, summary=dict(layers=len(out["layers"]),
                                 own_vs_logits_max=max(r["own_vs_logits"] for r in out["layers"].values())))

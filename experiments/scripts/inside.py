"""The maps inside the network (Section 6 and App. D.1; CausalGym's maps of Section 7.1 come from the same script):
for one task and model, the route test at every cell with every test pair's values kept, the first-order range at each cell,
and the quantities of the door proposition (Proposition 8).

A cell is (layer L, region): the input to block L (hidden_states[L]; the paper's layer L - 1) at the region's last token, or,
at L = NL, the readout cell (the last block's raw output at the last token). Rank-one DAS is trained for one pass over the
training rows, batch 4, Adam 5e-3, seed 100 L + region + 100000 s (CPU generator, normalised on the device); cross-entropy
DAS follows CausalGym's protocol (Section 7.1 lists the differences). Effects are log odds ratios
(source label over base label) on the test rows, in batches of 25 with right padding. Rows whose labels are not single tokens,
or whose base and source have different region counts, are dropped; each row records whether its two sequences are aligned
(equal length, equal region ends).
Per cell:
- the own swap G (the cell alone) and, for every other region Y, the pair swap LO({X, Y}) and LO({Y}); the best context Y is
  chosen on the training rows (G* = LO({X, Y}) - LO({Y}) on the test rows, or G when no Y adds more), as in Section 6, and
  also on the test rows, as in Section 7.1;
- DAS directions trained together (one Adam over the stacked directions, each moved by its own loss): ce s = 0, 1, 2; sq_T
  (per-pair squared error to T) s = 0; sq_G (per-pair squared error to the cell's own swap) s = 0, 1, 2; sqm_G ((batch-mean
  effect - batch-mean own swap)^2, the target-aimed objective in pure form, estimated without bias on each batch as the product
  of two half-batch mean residuals) s = 0; jsd s = 0; eff (-mean effect) s = 0. A per-pair squared error depends on the
  direction through more than the mean effect, so Corollary 6(ii) does not cover sq_G, and where it lands is measured, not
  predicted;
- a dead cell (in every row the tokens up to the site are the same, at the same place, so Delta = 0 by causal attention) is
  recorded as such and not measured;
- every direction (the trained directions and u-hat, b-hat, their bisector, the training midpoint M's v1) and the own swap:
  free, with every attention pattern frozen to the base run's, and with only the readout row frozen in the blocks from L on;
  every intervention of one batch in one forward pass. b-hat here comes from midpoint gradients (and is turned toward u-hat),
  so the cell's b_hat, bisector, cos_u_b, one_angle and grad_share (training rows) are midpoint quantities, the ones
  experiments/scripts/readout.py names with _mid;
- DAS (ce s = 0, 1, 2 and eff) in the best context chosen on training rows: LO({X along d, Y}) - LO({Y});
- first order at the cell: gradients of z in the cell's state (the rest of the run recomputed) at the midpoint h_b + Delta/2
  on training and test rows, at h_b on test rows, and at the midpoint with the best context swapped on test rows; M,
  lambda_1, lambda_D and tr M; lambda_1 on 2000 cluster resamples of the test pairs (a row and its mirror one cluster; seed
  95); every direction's first-order effect per test pair;
- the door proposition, at cells before the readout position and below the readout cell. In every head of block L, at the
  readout query and the site's key, per test row: the base weight alpha; a, the gradient of logit(alpha) in the site's state
  at h_b (no other score of block L depends on the site); a.Delta, |a||Delta| and |Delta|. Per head, for each class half
  (base type) and for all rows: lambda_1 and the trace of sym(E[a Delta^T]), the score's own Rayleigh range (the largest mean
  score change one direction can make). The weight on the site in every head of blocks L to NL - 1, in the base run, with
  the own swap and with four DAS directions (ce, sq_G, sqm_G and eff, each s = 0); for the own swap and the four directions,
  the exact score change in block L (the logit of the patched weight minus that of the base weight, exact since the other
  scores of block L do not change); for the four directions, the first-order score change (a.d)(d.Delta) per row and head.
Instruments: an unpatched run with attention frozen against the base run, and at the readout cell the own swap against T.
Outputs: experiments/results/inside/inside_<model>_<task>[_<tag>].json (per cell, point values), .npz (per test pair; keys
"L<l>r<region>/<name>"), and the directions in experiments/results/inside/dirs/ (git-ignored).
Usage: python -I experiments/scripts/inside.py <task> --model=<hf id> [--revision=<sha>] [--data=causalgym|fg|<dir>]
       [--layers=every2|all|1,3] [--regions=all|1,2] [--cells=L1r2,...] [--stack=0] [--chunk=16] [--device=cuda]
       [--tag=<name>]
--layers=every2 (the default) maps layers 1, 3, 5, ... (the paper's 0, 2, 4, ...) and the readout cell. --regions=all (the
default) maps every region but region 0, the start-of-text region in CausalGym's format, which is not a context either.
--stack=0 trains each (objective, start) alone (one direction per training), to compare with the stacked training.
"""
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402
from measure import boot_counts, clusters, lam_boot, m_sym, unit  # noqa: E402

pos_args, args, model, tok = common.start("inside", sys.argv[1:])
task = pos_args[0]; DEV = common.RUN["dev"]
BLOCKS, FIX, ATT = families.blocks(model), families.start_of_text(tok), families.attention(model)
NL, D = len(BLOCKS), model.config.hidden_size
rows = common.rows(args.get("data", "causalgym"), task)
LR, BATCH, EVB, B, BOOT_SEED, CHUNK = 5e-3, 4, 25, 2000, 95, int(args.get("chunk", "16"))
OBJECTIVES = [("ce", 0), ("ce", 1), ("ce", 2), ("sq_T", 0), ("sq_G", 0), ("sq_G", 1), ("sq_G", 2), ("sqm_G", 0), ("jsd", 0), ("eff", 0)]
DOOR_DIRS = ("das_ce_s0", "das_sq_G_s0", "das_sqm_G_s0", "das_eff_s0")
CTX_DIRS = ("das_ce_s0", "das_ce_s1", "das_ce_s2", "das_eff_s0")
STACK = args.get("stack", "1") == "1"

# The attention weights with their graph at one layer (the door's gradient): a wrapper around the eager kernel that
# owneffect.families.attention installed above, so its record, frozen and frozen_row modes still apply.
_amod = importlib.import_module(f"transformers.models.{families.name(model)}.modeling_{families.name(model)}")
GRADW, _inner = {"layer": None, "w": None}, _amod.eager_attention_forward


def _eager(module, *a, **kw):
    out, w = _inner(module, *a, **kw)
    if GRADW["layer"] == module.layer_idx:
        GRADW["w"] = w
    return out, w


_amod.eager_attention_forward = _eager


class _Stop(Exception):
    pass


STOP = {"after": None}                                                 # stop the forward pass after this block (the door)


def _stop_hook(i):
    def h(m, a, o):
        if STOP["after"] == i:
            raise _Stop()
    return h


for i, blk in enumerate(BLOCKS):
    blk.register_forward_hook(_stop_hook(i))

SITE = {"L": None, "patches": []}                                      # patches: [(positions (rows,), fn(states) -> states)]


def _apply(h):
    h = h.clone(); ar = torch.arange(h.shape[0], device=h.device)
    for p_, fn in SITE["patches"]:
        h[ar, p_] = fn(h[ar, p_])
    return h


def pre_hook(mod, a, kw):
    h = kw["hidden_states"] if "hidden_states" in kw else a[0]
    h = _apply(h)
    if "hidden_states" in kw:
        return a, dict(kw, hidden_states=h)
    return (h,) + tuple(a[1:]), kw


def post_hook(mod, a, out):
    if SITE["L"] != NL or not SITE["patches"]:
        return None
    h = out[0] if isinstance(out, tuple) else out
    h = _apply(h)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


for i, blk in enumerate(BLOCKS):
    blk.register_forward_pre_hook(lambda m, a, kw, i=i: pre_hook(m, a, kw) if SITE["L"] == i and SITE["patches"] else None,
                                  with_kwargs=True)
BLOCKS[-1].register_forward_hook(post_hook)


def label(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0], len(t)


def states(ids):
    """Every layer's input (index L < NL) and the last block's raw output (index NL) for one sequence, and its last logits."""
    raw = {}
    hk = BLOCKS[-1].register_forward_hook(lambda m, a, o: raw.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
    o = model(torch.tensor([ids], device=DEV), output_hidden_states=True); hk.remove()
    return torch.cat([torch.stack(list(o.hidden_states[:NL]))[:, 0], raw["h"][None]]), o.logits[0, -1]


data, dropped = {}, 0
with torch.no_grad():
    for split, rs in rows.items():
        out = []
        for r in rs:
            bi, be = families.encode(tok, r["base"], FIX); si, se = families.encode(tok, r["src"], FIX)
            (yb, nb), (ys, ns) = label(r["base_label"]), label(r["src_label"])
            if nb != 1 or ns != 1 or len(be) != len(se):
                dropped += 1; continue
            hb, zb = states(bi); hs, zs = states(si)
            out.append(dict(b=bi, s=si, be=be, se=se, yb=yb, ys=ys, hb=hb, hs=hs, ls=torch.log_softmax(zs, -1),
                            T=float((zb[yb] - zb[ys]) - (zs[yb] - zs[ys])), g0=float(zb[ys] - zb[yb]), btype=r.get("base_type"),
                            aligned=len(bi) == len(si) and be == se, key=(json.dumps(r["base"]), json.dumps(r["src"]))))
        data[split] = out
TR, TE = data["train"], data["test"]
n_regions = len(rows["train"][0]["base"])
lay = args.get("layers", "every2")
LAYERS = (list(range(1, NL)) if lay == "all" else list(range(1, NL, 2)) if lay == "every2" else [int(x) for x in lay.split(",")])
REGIONS = list(range(1, n_regions)) if args.get("regions", "all") == "all" else [int(x) for x in args["regions"].split(",")]
CELLS = [(l, r) for l in LAYERS if l != NL for r in REGIONS] + [(NL, n_regions - 1)]
if "cells" in args:
    CELLS = [tuple(int(x) for x in c.lstrip("L").split("r")) for c in args["cells"].split(",")]
types = sorted({r["btype"] for r in TR})
CLUSTER = clusters([r["key"] for r in TE])


def batch_ids(rs):
    T_ = max(len(r["b"]) for r in rs)
    ids = torch.zeros(len(rs), T_, dtype=torch.long); am = torch.zeros(len(rs), T_, dtype=torch.long)
    for i, r in enumerate(rs):
        ids[i, :len(r["b"])] = torch.tensor(r["b"]); am[i, :len(r["b"])] = 1
    return ids.to(DEV), am.to(DEV)


def site(rs, L, region):
    """Positions in the base, the base's state there, the source's state at its own position (the region's last token; at the
    readout cell the last token of each sequence), and Delta."""
    pb = [r["be"][region] if L < NL else len(r["b"]) - 1 for r in rs]
    ps = [r["se"][region] if L < NL else -1 for r in rs]
    hb = torch.stack([r["hb"][L, i] for r, i in zip(rs, pb)]); hs = torch.stack([r["hs"][L, i] for r, i in zip(rs, ps)])
    return torch.tensor(pb, device=DEV), hb, hs, hs - hb


def run(rs, L, patches):
    """Last-position logits of the base runs with the patches at layer L (patches: [(positions, fn)])."""
    ids, am = batch_ids(rs)
    SITE.update(L=L, patches=patches)
    try:
        z = model(ids, attention_mask=am).logits
    finally:
        SITE.update(L=None, patches=[])
    return z[torch.arange(len(rs), device=DEV), torch.tensor([len(r["b"]) - 1 for r in rs], device=DEV)]


def lo(z, rs):
    ar = torch.arange(len(rs), device=DEV)
    return z[ar, torch.tensor([r["ys"] for r in rs], device=DEV)] - z[ar, torch.tensor([r["yb"] for r in rs], device=DEV)]


def intervention_patch(rs, L, spec):
    """The patch of one intervention on rows rs: ("own", X) swaps X's state; ("dir", X, d) the rank-one interchange along d at
    X; an intervention may carry a second, own swap at a context region ("ctx", Y)."""
    out = []
    for part in spec:
        kind, X = part[0], part[1]
        p_, _, hs, _ = site(rs, L, X)
        if kind in ("own", "ctx"):
            out.append((p_, lambda h, s=hs: s))
        else:
            d = part[2]
            out.append((p_, (lambda h, s=hs, d=d: h + ((s - h) @ d)[:, None] * d)))
    return out


BASE = {}                                                              # the base runs' log odds per (split, batch)


def evaluate(split, L, specs, mode="normal"):
    """Per-row effects of every intervention (a list of patch specs) on a split, the interventions of one batch in one forward
    pass (rows repeated; CHUNK at a time). mode: normal | frozen (every pattern the base run's) | row (the readout row frozen
    in the blocks from L on). Returns {index: (effects, accuracy)}."""
    S = data[split]; res = {k: ([], []) for k in range(len(specs))}
    with torch.no_grad():
        for i in range(0, len(S), EVB):
            rs = S[i:i + EVB]; n = len(rs)
            if (split, i) not in BASE:
                BASE[(split, i)] = lo(run(rs, None, []), rs)
            for c0 in range(0, len(specs), CHUNK):
                ks = list(range(c0, min(c0 + CHUNK, len(specs)))); rep = rs * len(ks)
                n_parts = max(len(specs[k]) for k in ks)
                patches = []
                for j in range(n_parts):
                    pos_, fns = [], []
                    for k in ks:
                        part = specs[k][j] if j < len(specs[k]) else specs[k][0]   # a shorter spec repeats its first part
                        (p_, fn), = intervention_patch(rs, L, [part]); pos_.append(p_); fns.append(fn)
                    cat_pos = torch.cat(pos_)
                    patches.append((cat_pos, (lambda h, fns=fns, n=n: torch.cat([f(h[q * n:(q + 1) * n]) for q, f in enumerate(fns)]))))
                if mode == "normal":
                    l0 = BASE[(split, i)].repeat(len(ks)); z1 = run(rep, L, patches)
                else:
                    ATT["mode"] = "record"; ATT["store"] = {}
                    l0 = lo(run(rep, None, []), rep)
                    if mode == "frozen":
                        ATT["mode"] = "frozen"
                    else:
                        ATT.update(mode="frozen_row", first=L, row=torch.tensor([len(r["b"]) - 1 for r in rep], device=DEV))
                    try:
                        z1 = run(rep, L, patches)
                    finally:
                        ATT["mode"] = "normal"
                l1 = lo(z1, rep)
                for q, k in enumerate(ks):
                    res[k][0].append((l1 - l0)[q * n:(q + 1) * n]); res[k][1].append((l1[q * n:(q + 1) * n] > 0).float())
    return {k: (torch.cat(e), torch.cat(a)) for k, (e, a) in res.items()}


def jsd(logp, logq):
    p, q = logp.exp(), logq.exp(); logm = torch.log(0.5 * (p + q) + 1e-30)
    return 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))


def train_das(L, region, G_tr, specs):
    """DAS for every (objective, start) of specs at once (one Adam over the stacked directions, each moved by its own loss),
    for one pass over the training rows; returns {spec: (direction, (first tenth, last tenth of the loss))}. The base log
    odds come from each row's own base run."""
    gens = [torch.Generator().manual_seed(L * 100 + region + 100000 * s) for _, s in specs]
    V0, orders = [], []
    for g in gens:
        v = torch.randn(D, generator=g).to(DEV); V0.append(v / v.norm())
        orders.append(torch.randperm(len(TR), generator=g).tolist())
    V = torch.stack(V0).requires_grad_(True); opt = torch.optim.Adam([V], lr=LR)
    G_tr = torch.tensor(G_tr, device=DEV); T_tr = torch.tensor([r["T"] for r in TR], device=DEV)
    g0 = torch.tensor([r["g0"] for r in TR], device=DEV); hist = []
    for i in range(0, len(TR), BATCH):
        idx = [o[i:i + BATCH] for o in orders]; nb = len(idx[0])
        rs = [TR[j] for o in idx for j in o]; flat = torch.tensor([j for o in idx for j in o], device=DEV)
        A = V / V.norm(dim=1, keepdim=True)
        p_pos, _, hs, _ = site(rs, L, region)
        if len(specs) == 1:                                           # one direction (--stack=0): one-direction arithmetic
            fn = lambda h, s=hs, a=A[0]: h + ((s - h) @ a)[:, None] * a  # noqa: E731
        else:
            a = A.repeat_interleave(nb, 0); fn = lambda h, s=hs, a=a: h + ((s - h) * a).sum(-1, keepdim=True) * a  # noqa: E731
        z = run(rs, L, [(p_pos, fn)])
        e = lo(z, rs) - g0[flat]
        ys = torch.tensor([r["ys"] for r in rs], device=DEV); Ls = []
        for k, (obj, _) in enumerate(specs):
            sl = slice(k * nb, (k + 1) * nb)
            if obj == "ce":
                Ls.append(torch.nn.functional.cross_entropy(z[sl], ys[sl]))
            elif obj == "jsd":
                Ls.append(jsd(torch.log_softmax(z[sl], -1), torch.stack([rs[j]["ls"] for j in range(sl.start, sl.stop)])).mean())
            elif obj == "eff":
                Ls.append(-e[sl].mean())
            elif obj == "sqm_G":                                      # unbiased for (mean effect - mean G)^2: two independent halves
                r_ = e[sl] - G_tr[flat[sl]]; h_ = len(r_) // 2
                Ls.append(r_[:h_].mean() * r_[h_:].mean() if h_ >= 1 else r_.mean() ** 2)
            else:
                tgt = T_tr[flat[sl]] if obj == "sq_T" else G_tr[flat[sl]]
                Ls.append(((e[sl] - tgt) ** 2).mean())
        Ls = torch.stack(Ls)
        opt.zero_grad(); Ls.sum().backward(); opt.step(); hist.append(Ls.detach())
    H = torch.stack(hist).cpu(); k = max(1, len(H) // 10)
    A = (V / V.norm(dim=1, keepdim=True)).detach()
    return {sp: (A[j].clone(), (float(H[:k, j].mean()), float(H[-k:, j].mean()))) for j, sp in enumerate(specs)}


def grads(split, L, region, where, ctx=None):
    """Per-row gradient of z in the cell's state, at the midpoint (where="mid") or at h_b ("base"), with the best context's own
    swap at ctx when given."""
    S = data[split]; out = []
    for i in range(0, len(S), EVB):
        rs = S[i:i + EVB]
        p_, hb, _, dl = site(rs, L, region)
        X = (hb + 0.5 * dl if where == "mid" else hb).clone().requires_grad_(True)
        patches = [(p_, lambda h, X=X: X)]
        if ctx is not None:
            patches.append(intervention_patch(rs, L, [("ctx", ctx)])[0])
        z = run(rs, L, patches)
        g, = torch.autograd.grad(lo(z, rs).sum(), X)
        out.append(g.detach())
    return torch.cat(out)


def head_range(A_, dl, idx):
    """Per head: lambda_1 and the trace of sym(mean a Delta^T) over rows idx (the score's own Rayleigh range: the Rayleigh range
    theorem, Theorem 5, applied to the score), exactly in the row space of the rows' a and Delta."""
    lam, tr = [], []
    for h in range(A_.shape[1]):
        W, Dm = A_[idx, h], dl[idx]
        lam.append(float(lam_boot(W, Dm, torch.ones(1, len(idx), dtype=torch.float64, device=DEV))[0]))
        tr.append(float((W * Dm).sum(-1).mean()))
    return lam, tr


def door(L, region, dirs):
    """The door proposition's quantities per test row and head of block L, and the measured weights on the site in every head
    of blocks L to NL - 1."""
    out = {k: [] for k in ("alpha", "a_dot_delta", "a_norm_delta_norm", "delta_norm", "w_base", "w_own", "dscore_own")}
    for k in DOOR_DIRS:
        out[f"w_{k}"], out[f"dscore_{k}"], out[f"first_{k}"] = [], [], []
    A_all, D_all = [], []
    for i in range(0, len(TE), EVB):
        rs = TE[i:i + EVB]
        p_, hb, hs, dl = site(rs, L, region)
        n_ = torch.tensor([len(r["b"]) - 1 for r in rs], device=DEV); ar = torch.arange(len(rs), device=DEV)
        X = hb.clone().requires_grad_(True)
        GRADW["layer"], STOP["after"] = L, L
        try:
            run(rs, L, [(p_, lambda h, X=X: X)])
        except _Stop:
            pass
        finally:
            GRADW["layer"], STOP["after"] = None, None
        w = GRADW["w"]; GRADW["w"] = None
        al = w[ar, :, n_, p_].clamp(1e-12, 1 - 1e-12)                 # rows x heads
        lg = torch.log(al) - torch.log1p(-al)
        A_ = torch.stack([torch.autograd.grad(lg[:, h].sum(), X, retain_graph=h < lg.shape[1] - 1)[0] for h in range(lg.shape[1])], 1)
        A_all.append(A_.detach()); D_all.append(dl)
        out["alpha"].append(al.detach()); out["a_dot_delta"].append((A_ * dl[:, None, :]).sum(-1))
        out["a_norm_delta_norm"].append(A_.norm(dim=-1) * dl.norm(dim=-1)[:, None]); out["delta_norm"].append(dl.norm(dim=-1))
        with torch.no_grad():
            weights = {}
            for key, patches in [("base", []), ("own", [(p_, lambda h, s=hs: s)])] + \
                                [(k, [(p_, lambda h, s=hs, d=dirs[k]: h + ((s - h) @ d)[:, None] * d)]) for k in DOOR_DIRS]:
                ATT["mode"] = "record"; ATT["store"] = {}
                try:
                    run(rs, L, patches)
                finally:
                    ATT["mode"] = "normal"
                weights[key] = torch.stack([ATT["store"][l][ar, :, n_, p_] for l in range(L, NL)], 1)   # rows x blocks x heads
            lgb = torch.logit(weights["base"][:, 0].clamp(1e-12, 1 - 1e-12))
            out["w_base"].append(weights["base"]); out["w_own"].append(weights["own"])
            out["dscore_own"].append(torch.logit(weights["own"][:, 0].clamp(1e-12, 1 - 1e-12)) - lgb)
            for k in DOOR_DIRS:
                d = dirs[k]
                out[f"w_{k}"].append(weights[k])
                out[f"dscore_{k}"].append(torch.logit(weights[k][:, 0].clamp(1e-12, 1 - 1e-12)) - lgb)
                out[f"first_{k}"].append((A_ @ d) * (dl @ d)[:, None])
    res = {k: torch.cat(v).double().cpu().numpy() for k, v in out.items()}
    A_all, D_all = torch.cat(A_all).double(), torch.cat(D_all).double()
    half = np.array([r["btype"] == types[0] for r in TE])
    for name, sel in (("half0", np.nonzero(half)[0]), ("half1", np.nonzero(~half)[0]), ("all", np.arange(len(TE)))):
        if len(sel) >= 2:
            lam, tr = head_range(A_all, D_all, torch.tensor(sel, device=DEV))
            res[f"range_lam_{name}"], res[f"range_tr_{name}"] = np.array(lam), np.array(tr)
    return res


# The best context, per layer: every region alone and with every other region, on training and test rows; Y chosen on the
# training rows (Section 6's choice) and on the test rows (Section 7.1's choice).
t0 = time.time(); BEST = {}; SINGLE = {}
for L in [l for l in LAYERS if l != NL]:
    pairs = [(X, Y) for X in REGIONS for Y in REGIONS if X < Y]
    for split in ("train", "test"):
        out = evaluate(split, L, [[("own", X)] for X in REGIONS] + [[("own", X), ("own", Y)] for X, Y in pairs])
        single = {X: out[j][0] for j, X in enumerate(REGIONS)}
        pair = {}
        for j, (X, Y) in enumerate(pairs):
            pair[(X, Y)] = pair[(Y, X)] = out[len(REGIONS) + j][0]
        SINGLE[(L, split)] = (single, pair)
    for X in REGIONS:
        for which in ("train", "test"):
            single, pair = SINGLE[(L, which)]
            best, by = float(single[X].mean()), None
            for Y in REGIONS:
                if Y != X and float((pair[(X, Y)] - single[Y]).mean()) > best:
                    best, by = float((pair[(X, Y)] - single[Y]).mean()), Y
            s_te, p_te = SINGLE[(L, "test")]
            BEST[(L, X, which)] = (by, s_te[X] if by is None else p_te[(X, by)] - s_te[by])
t_ctx = time.time() - t0

res, arrays, dirs_store = [], {}, {}
with torch.no_grad():                                                 # instrument: frozen attention without a patch = the base run
    rs = TE[:EVB]; ATT["mode"] = "record"; ATT["store"] = {}; z0 = run(rs, None, []); ATT["mode"] = "frozen"
    try:
        FROZEN_CHECK = float((run(rs, None, []) - z0).abs().max())
    finally:
        ATT["mode"] = "normal"
cnt = boot_counts(CLUSTER, B, BOOT_SEED); counts = torch.tensor(cnt, dtype=torch.float64, device=DEV)
T_te = torch.tensor([r["T"] for r in TE], device=DEV, dtype=torch.float64)
for L, region in CELLS:
    tk = {"start": time.time()}
    _, _, _, D_tr = site(TR, L, region); _, _, _, D_te = site(TE, L, region)

    def same_prefix(r):                                               # the tokens up to the site are the same, at the same place
        if L == NL:
            return False
        pb, ps = r["be"][region], r["se"][region]
        return pb == ps and r["b"][:pb + 1] == r["s"][:ps + 1]
    if all(same_prefix(r) for r in TR + TE):                          # a dead cell: Delta = 0 by causal attention (exactly,
        res.append(dict(layer=L, region=region, live=False, T_test=float(T_te.mean()), door=False,      # up to the GPU's rounding)
                        max_abs_delta=float(max(D_tr.abs().max(), D_te.abs().max())))); continue
    G_tr = evaluate("train", L, [[("own", region)]])[0][0]
    trained = train_das(L, region, G_tr.tolist(), OBJECTIVES) if STACK else {sp: train_das(L, region, G_tr.tolist(), [sp])[sp] for sp in OBJECTIVES}
    tk["train"] = time.time()
    Wm_tr, Wm_te, Wb_te = grads("train", L, region, "mid"), grads("test", L, region, "mid"), grads("test", L, region, "base")
    ctx_tr = BEST.get((L, region, "train"), (None, None))[0]
    Wc_te = grads("test", L, region, "mid", ctx=ctx_tr) if ctx_tr is not None else None
    tk["grads"] = time.time()
    sig = torch.tensor([1.0 if r["btype"] == types[0] else -1.0 for r in TR], device=DEV)[:, None]
    uh = unit((sig * D_tr).mean(0)); bh = unit((sig * Wm_tr).mean(0))
    if float(uh @ bh) < 0:
        bh = -bh
    v1 = torch.linalg.eigh(m_sym(Wm_tr, D_tr))[1][:, -1].float()
    evm = torch.linalg.eigvalsh(m_sym(Wm_te, D_te)); evb = torch.linalg.eigvalsh(m_sym(Wb_te, D_te))
    evc = torch.linalg.eigvalsh(m_sym(Wc_te, D_te)) if Wc_te is not None else None
    lam_m = lam_boot(Wm_te, D_te, counts)
    dirs = {f"das_{o}_s{s}": d for (o, s), (d, _) in trained.items()}
    dirs.update(u_hat=uh, b_hat=bh, bisector=unit(uh + bh), v1_mid=v1)
    tk["range"] = time.time()
    names = ["own"] + list(dirs)
    specs = [[("own", region)]] + [[("dir", region, dirs[k])] for k in dirs]
    E = {mode: evaluate("test", L, specs, mode) for mode in ("normal", "frozen", "row")}
    if ctx_tr is not None:
        cspecs = [[("own", region), ("ctx", ctx_tr)]] + [[("dir", region, dirs[k]), ("ctx", ctx_tr)] for k in CTX_DIRS] + [[("ctx", ctx_tr)]]
        Ec = evaluate("test", L, cspecs)
        ref = Ec[len(cspecs) - 1][0]                                  # LO({Y}) alone
        for j, k in enumerate(["own"] + list(CTX_DIRS)):
            arrays[f"L{L}r{region}/ctx/{k}"] = (Ec[j][0] - ref).double().cpu().numpy()
    tk["eval"] = time.time()
    cell = f"L{L}r{region}"; T = float(T_te.mean())
    for mode, Em in E.items():
        for j, k in enumerate(names):
            arrays[f"{cell}/e/{k}_{mode}"] = Em[j][0].double().cpu().numpy(); arrays[f"{cell}/acc/{k}_{mode}"] = Em[j][1].double().cpu().numpy()
    for k, d in dirs.items():
        arrays[f"{cell}/fm/{k}"] = ((Wm_te @ d) * (D_te @ d)).double().cpu().numpy()
    arrays[f"{cell}/tr_mid"] = (Wm_te * D_te).sum(1).double().cpu().numpy(); arrays[f"{cell}/tr_base"] = (Wb_te * D_te).sum(1).double().cpu().numpy()
    if Wc_te is not None:
        arrays[f"{cell}/tr_ctx"] = (Wc_te * D_te).sum(1).double().cpu().numpy()
    arrays[f"{cell}/lam_mid_boot"] = lam_m.cpu().numpy()
    for which in ("train", "test"):
        _, gstar = BEST.get((L, region, which), (None, E["normal"][0][0]))
        arrays[f"{cell}/gstar_{which}"] = gstar.double().cpu().numpy()
    door_ok = L < NL and all(r["be"][region] < len(r["b"]) - 1 for r in TE)
    if door_ok:
        for k, v in door(L, region, dirs).items():
            arrays[f"{cell}/door/{k}"] = v
    tk["door"] = time.time()
    c = float(uh @ bh)
    row = dict(layer=L, region=region, live=True, T_test=T, door=door_ok,
               context_train=BEST.get((L, region, "train"), (None,))[0], context_test=BEST.get((L, region, "test"), (None,))[0],
               mean_T={f"{k}_{mode}": float(Em[j][0].mean()) / T for mode, Em in E.items() for j, k in enumerate(names)},
               lambda1_mid=float(evm[-1]), lambdaD_mid=float(evm[0]), trM_mid=float(evm.sum()), lambda1=float(evb[-1]), trM=float(evb.sum()),
               lambda1_ctx=float(evc[-1]) if evc is not None else None, trM_ctx=float(evc.sum()) if evc is not None else None,
               cos_u_b=c, one_angle=0.5 * (1 + 1 / c) if c > 0 else None,
               grad_share=float((sig * Wm_tr).mean(0).norm() / Wm_tr.norm(dim=1).mean()),
               losses={f"das_{o}_s{s}": lf for (o, s), (_, lf) in trained.items()},
               cos_v1={k: float(abs(d @ v1)) for k, d in dirs.items()},
               seconds={k: round(tk[k] - tk[p], 2) for p, k in zip(list(tk)[:-1], list(tk)[1:])})
    if L == NL:                                                       # instrument: at the readout cell the own swap is T
        row["own_vs_T_max_abs"] = float((E["normal"][0][0].double() - T_te).abs().max())
    res.append(row)
    dirs_store[cell] = {k: d.float().cpu().numpy() for k, d in dirs.items()}
    print(json.dumps({cell: {"own": round(row["mean_T"]["own_normal"], 2), "ce": round(row["mean_T"]["das_ce_s0_normal"], 2),
                             "sqmG": round(row["mean_T"]["das_sqm_G_s0_normal"], 2), "l1": round(row["lambda1_mid"] / T, 2),
                             "trm": round(row["trM_mid"] / T, 2)}}), row["seconds"], f"{time.time() - t0:.0f}s", flush=True)
res_dir = common.REPO / "experiments/results/inside"; (res_dir / "dirs").mkdir(parents=True, exist_ok=True)
stem = f"inside_{common.RUN['tag']}_{task}" + (f"_{args['tag']}" if "tag" in args else "")
f, fz, fd = res_dir / f"{stem}.json", res_dir / f"{stem}.npz", res_dir / "dirs" / f"{stem}.npz"
arrays["T"] = T_te.cpu().numpy(); arrays["cluster"] = np.asarray(CLUSTER)
arrays["half"] = np.array([r["btype"] == types[0] for r in TE]); arrays["aligned"] = np.array([r["aligned"] for r in TE])
np.savez_compressed(fz, **arrays)
np.savez_compressed(fd, **{f"{c}/{k}": v for c, ds in dirs_store.items() for k, v in ds.items()})
CODE_FILES = ("experiments/scripts/inside.py", "experiments/lib/measure.py", "owneffect/families.py", "owneffect/core.py", "experiments/lib/common.py")
code_sha = {f: hashlib.sha256((common.REPO / f).read_bytes()).hexdigest() for f in CODE_FILES}
out = dict(task=task, model=common.RUN["model"], run=common.RUN["id"], code_sha256=code_sha,
           full_grid="cells" not in args and STACK and args.get("layers", "every2") == "every2" and args.get("regions", "all") == "all",
           cells_arg=args.get("cells"), family=families.name(model), dropped_rows=dropped, n_train=len(TR), n_test=len(TE),
           layers=LAYERS, regions=REGIONS, objectives=OBJECTIVES, stack=STACK,
           context_seconds=round(t_ctx, 1),
           frozen_no_patch_max_abs=FROZEN_CHECK,
           boot=dict(B=B, seed=BOOT_SEED, scheme="clusters"),
           boot_counts_sha=hashlib.sha256(np.ascontiguousarray(cnt, dtype=np.int64).tobytes()).hexdigest()[:16],
           n_clusters_test=int(len(set(np.asarray(CLUSTER).tolist()))), cells=res,
           npz_sha256=hashlib.sha256(fz.read_bytes()).hexdigest(), dirs_sha256=hashlib.sha256(fd.read_bytes()).hexdigest())
json.dump(out, open(f, "w"), indent=1)
common.finish([f, fz, fd], summary=dict(cells=len(res), dropped=dropped))

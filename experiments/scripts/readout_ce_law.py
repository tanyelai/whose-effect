"""Cross-entropy's own range at the readout (Appendix C.2, "Cross-entropy ends near its own optimum"). Minimising the source label's
cross-entropy maximises the mean of f = log p(y_s), so to first order it seeks the top of f's range, v_f = v1(M_f), and its
effect on the log odds is that direction's. Nothing about the model is trained.

Per task, at the readout cell (the last block's raw output at the last token), on the rows of experiments/scripts/readout.py (labels
that are single tokens), with that script's saved output for the same model and group:
- training pairs: f's gradient at the midpoint and at the base point, M_f = m_sym(W_f, Delta) and its top eigenvector (v_f and
  v_f_base); the log odds' midpoint M_z and its top eigenvector v_z, checked against readout.py's saved v1_mid (instrument:
  |cos| > 0.999); p_s, p_b and the mass outside the two labels at the midpoint (medians);
- test pairs: the exact log-odds effect of the rank-one patch h_b + ((h_s - h_b).d) d along v_f and v_f_base, per row; T per
  row, checked against readout.py's (instrument: within an absolute 1e-3);
- the exact f-top (a check of the method only, since cross-entropy's exact optimum is the f-top by definition): Adam on an
  unnormalised v with d = v/|v|, minimising the mean cross-entropy on training pairs (measure.sphere_opt, 300 steps, lr 1e-3
  falling on a cosine to 1e-5), from v_f and from v1_mid separately; each start's direction, training cross-entropy, test
  effect, last-50-step change and |cos| with cross-entropy's ten-pass directions;
- |cos| of v_f, v_f_base and v_z with readout.py's saved directions (cross-entropy's three starts at one and ten passes, eff's
  first start at ten passes, v1_mid, top_train), and the mean training cross-entropy along v_f, v_f_base, v_z and
  cross-entropy's six saved directions;
- the parts orthogonal to v1_mid of v_f and of cross-entropy's and eff's ten-pass directions: their norms (the sine of the
  angle to v1_mid) and, per start, |cos| of cross-entropy's off-top part with v_f's and with eff's from the same start (the
  scorer's test C2 asks whether the first exceeds the second, which serves as its null).
The run stops when readout.py's npz or direction file is not the one its json names (sha256).
Output: experiments/results/readout_ce_law/readout_ce_law_<model>_m_<group>.{json,npz}; directions in readout_ce_law/dirs/.
Usage: python -I experiments/scripts/readout_ce_law.py --model=<hf id> [--revision=<sha>] --data=causalgym|fg|<dir> [--tasks=a,b]
       [--sign=...] [--device=cuda] --tag=m_<group>
(--sign changes nothing here; experiments/matrix/run.py passes a group's options to every kind of run.)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402
from measure import m_sym, sphere_opt, unit  # noqa: E402

_, args, model, tok = common.start("readout_ce_law", sys.argv[1:])
DEV = common.RUN["dev"]
BLOCKS, FIX = families.blocks(model), families.start_of_text(tok)
READ = families.readout(model)
DATA = args.get("data", "causalgym")
TASKS = (common.causalgym_tasks() if DATA == "causalgym" else []) if "tasks" not in args else args["tasks"].split(",")
GROUP = args["tag"][2:] if args.get("tag", "").startswith("m_") else args.get("tag", "")
RAW = {}
BLOCKS[-1].register_forward_hook(lambda m, a, o: RAW.__setitem__("h", o[0] if isinstance(o, tuple) else o))
READOUT_NPZ = common.REPO / "experiments/results/readout" / f"readout_{common.RUN['tag']}_m_{GROUP}.npz"
READOUT_DIRS = common.REPO / "experiments/results/readout/dirs" / f"readout_{common.RUN['tag']}_m_{GROUP}.npz"
ZV, ZD = np.load(READOUT_NPZ), np.load(READOUT_DIRS)
READOUT_JSON = json.load(open(READOUT_NPZ.with_suffix(".json")))
if READOUT_JSON.get("dirs_sha256") != hashlib.sha256(READOUT_DIRS.read_bytes()).hexdigest() or READOUT_JSON.get("npz_sha256") != hashlib.sha256(READOUT_NPZ.read_bytes()).hexdigest():
    sys.exit(f"{READOUT_NPZ.name}: the readout's npz or direction file is not the one its json names")
COMPARE = [f"das_ce_p{p}_s{s}" for p in (1, 10) for s in (0, 1, 2)] + ["das_eff_p10_s0", "v1_mid", "top_train"]
CHUNK = 64
FSTEPS, FLR = 300, 1e-3


def one(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None


def last_states(seqs):
    out, groups = [None] * len(seqs), {}
    for i, q in enumerate(seqs):
        groups.setdefault(len(q), []).append(i)
    with torch.no_grad():
        for idx in groups.values():
            for k in range(0, len(idx), CHUNK):
                chunk = idx[k:k + CHUNK]
                model(torch.tensor([seqs[i] for i in chunk], device=DEV))
                for j, i in enumerate(chunk):
                    out[i] = RAW["h"][j, -1].clone()
    return torch.stack(out)


def side(rows):
    keep = [r for r in rows if one(r["base_label"]) is not None and one(r["src_label"]) is not None]
    hb = last_states([families.encode(tok, r["base"], FIX)[0] for r in keep])
    hs = last_states([families.encode(tok, r["src"], FIX)[0] for r in keep])
    return dict(hb=hb, hs=hs, D=hs - hb, n=len(keep), yb=torch.tensor([one(r["base_label"]) for r in keep], device=DEV),
                ys=torch.tensor([one(r["src_label"]) for r in keep], device=DEV))


def z_of(H, yb, ys):
    z = READ(H); ar = torch.arange(len(yb), device=H.device)
    return z[ar, ys] - z[ar, yb]


def grads(H0, yb, ys):
    """Per row: the gradient of f = log p(y_s) and of the log odds at H0, and p_s, p_b (chunked: the vocabulary can be 256k)."""
    Wf, Wz, ps, pb = [], [], [], []
    for k in range(0, len(H0), CHUNK):
        H = H0[k:k + CHUNK].clone().requires_grad_(True); b, s = yb[k:k + CHUNK], ys[k:k + CHUNK]
        ar = torch.arange(len(b), device=H.device)
        lg = READ(H); lp = torch.log_softmax(lg, -1)
        gf, = torch.autograd.grad(lp[ar, s].sum(), H, retain_graph=True)
        gz, = torch.autograd.grad((lg[ar, s] - lg[ar, b]).sum(), H)
        Wf.append(gf); Wz.append(gz)
        with torch.no_grad():
            p = lp.exp(); ps.append(p[ar, s]); pb.append(p[ar, b])
    return torch.cat(Wf), torch.cat(Wz), torch.cat(ps), torch.cat(pb)


def top(M):
    return torch.linalg.eigh(M)[1][:, -1].float()


def effect(te, d):
    with torch.no_grad():
        return z_of(te["hb"] + (te["D"] @ d)[:, None] * d, te["yb"], te["ys"]) - z_of(te["hb"], te["yb"], te["ys"])


def ce_of(S, Dn):
    """Mean cross-entropy of the source label over S's rows for K unit directions at once: (K,). Rows go in chunks, each
    checkpointed, so only one chunk's logits (K x CHUNK x vocabulary) are held, in the forward and in the backward pass."""
    def chunk(hb, D, ys, Dn):
        H = hb[None] + ((D[None] * Dn[:, None, :]).sum(-1, keepdim=True)) * Dn[:, None, :]
        lp = torch.log_softmax(READ(H.reshape(-1, H.shape[-1])), -1)
        yy = ys.repeat(Dn.shape[0])
        return -lp[torch.arange(len(yy), device=DEV), yy].reshape(Dn.shape[0], -1).sum(1)
    tot = 0
    for k in range(0, S["n"], CHUNK):
        sl = slice(k, k + CHUNK)
        args_ = (S["hb"][sl], S["D"][sl], S["ys"][sl], Dn)
        tot = tot + (checkpoint(chunk, *args_, use_reentrant=False) if Dn.requires_grad else chunk(*args_))
    return tot / S["n"]


def off(x, v):
    """x's part orthogonal to the unit v: (unit direction, its norm = sin of the angle for a unit x)."""
    x = unit(x); r = x - (x @ v) * v
    return r / r.norm(), float(r.norm())


out, arrays, dirs_out = dict(model=common.RUN["model"], group=GROUP, data=DATA, readout=str(READOUT_NPZ.relative_to(common.REPO)),
                   readout_sha256=hashlib.sha256(READOUT_NPZ.read_bytes()).hexdigest(), readout_dirs_sha256=hashlib.sha256(READOUT_DIRS.read_bytes()).hexdigest(),
                   tasks={}), {}, {}
for task in TASKS:
    if f"{task}/T" not in ZV.files:
        out["tasks"][task] = dict(skipped="not in the readout's output"); continue
    R = common.rows(DATA, task)
    tr, te = side(R["train"]), side(R["test"])
    Wf_m, Wz_m, ps_m, pb_m = grads(tr["hb"] + 0.5 * tr["D"], tr["yb"], tr["ys"])
    Wf_b, _, _, _ = grads(tr["hb"], tr["yb"], tr["ys"])
    v_f, v_fb, v_z = top(m_sym(Wf_m, tr["D"])), top(m_sym(Wf_b, tr["D"])), top(m_sym(Wz_m, tr["D"]))
    with torch.no_grad():
        T = z_of(te["hs"], te["yb"], te["ys"]) - z_of(te["hb"], te["yb"], te["ys"])
    T_readout = ZV[f"{task}/T"]
    ok_T = len(T_readout) == te["n"] and float(np.max(np.abs(T.double().cpu().numpy() - T_readout))) < 1e-3
    vmid = torch.tensor(ZD[f"{task}/v1_mid"], device=DEV, dtype=v_z.dtype)
    cos_vz = float(abs(v_z @ vmid) / (v_z.norm() * vmid.norm()))
    e_f, e_fb = effect(te, v_f), effect(te, v_fb)
    arrays[f"{task}/e/v_f"] = e_f.double().cpu().numpy(); arrays[f"{task}/e/v_f_base"] = e_fb.double().cpu().numpy()
    arrays[f"{task}/T"] = T.double().cpu().numpy()
    cos = {}
    for k in COMPARE:
        if f"{task}/{k}" in ZD.files:
            d = torch.tensor(ZD[f"{task}/{k}"], device=DEV, dtype=v_f.dtype)
            cos[k] = dict(v_f=float(abs(v_f @ d) / d.norm()), v_f_base=float(abs(v_fb @ d) / d.norm()), v_z=float(abs(v_z @ d) / d.norm()))
    outside = 1 - ps_m - pb_m
    vec = lambda k: torch.tensor(ZD[f"{task}/{k}"], device=DEV, dtype=v_f.dtype)  # noqa: E731
    v1 = unit(vmid)
    f_tops = {}
    for name, start in (("from_v_f", v_f), ("from_v1_mid", v1)):          # each start alone: its own best direction
        d_, loss_, rec_ = sphere_opt(lambda Dn: ce_of(tr, Dn), [(name, start)], FSTEPS, FLR)
        e_ = effect(te, d_)
        arrays[f"{task}/e/f_top_{name}"] = e_.double().cpu().numpy(); dirs_out[f"{task}/f_top_{name}"] = d_.float().cpu().numpy()
        f_tops[name] = dict(train_ce=float(loss_), e_T=float(e_.mean() / T.mean()), last50_change=rec_[0]["last50_change"],
                            best_step=rec_[0]["best_step"], cos_ce10=[float(abs(d_ @ unit(vec(f"das_ce_p10_s{s_}")))) for s_ in (0, 1, 2)])
    with torch.no_grad():
        names = ["v_f", "v_f_base", "v_z"] + [f"das_ce_p{p_}_s{s_}" for p_ in (1, 10) for s_ in (0, 1, 2)]
        Dn = torch.stack([v_f, v_fb, v_z] + [unit(vec(f"das_ce_p{p_}_s{s_}")) for p_ in (1, 10) for s_ in (0, 1, 2)])
        train_ce = dict(zip(names, ce_of(tr, Dn).tolist()))
        u_vf, sin_vf = off(v_f, v1)
        align = {}
        for s_ in (0, 1, 2):
            u_ce, sin_ce = off(vec(f"das_ce_p10_s{s_}"), v1); u_eff, sin_eff = off(vec(f"das_eff_p10_s{s_}"), v1)
            align[f"s{s_}"] = dict(ce_v_f=float(abs(u_ce @ u_vf)), ce_eff=float(abs(u_ce @ u_eff)), sin_ce=sin_ce, sin_eff=sin_eff)
    for k, v in (("v_f", v_f), ("v_f_base", v_fb), ("v_z", v_z)):
        dirs_out[f"{task}/{k}"] = v.float().cpu().numpy()
    out["tasks"][task] = dict(n_train=tr["n"], n_test=te["n"], T=float(T.mean()), e_v_f_T=float(e_f.mean() / T.mean()),
                              f_tops=f_tops, train_ce=train_ce, align=align, sin_v_f=sin_vf,
                              e_v_f_base_T=float(e_fb.mean() / T.mean()), cos=cos, cos_v_z_v1_mid=cos_vz,
                              instruments=dict(T_matches_readout=ok_T, v_z_is_v1_mid=cos_vz > 0.999),
                              median_p_s_mid=float(ps_m.median()), median_p_b_mid=float(pb_m.median()),
                              median_outside_mid=float(outside.median()))
    print(json.dumps({task: dict(e_v_f_T=round(out["tasks"][task]["e_v_f_T"], 3), f_top_T=round(f_tops["from_v_f"]["e_T"], 3), cos_ce=round(cos.get("das_ce_p1_s0", {}).get("v_f", float("nan")), 3),
                                 T_ok=ok_T, vz_ok=cos_vz > 0.999)}), flush=True)
res = common.REPO / "experiments/results/readout_ce_law"; res.mkdir(parents=True, exist_ok=True)
stem = f"readout_ce_law_{common.RUN['tag']}_m_{GROUP}"
np.savez_compressed(res / f"{stem}.npz", **arrays)
(res / "dirs").mkdir(exist_ok=True); np.savez_compressed(res / "dirs" / f"{stem}.npz", **dirs_out)
out["npz_sha256"] = hashlib.sha256((res / f"{stem}.npz").read_bytes()).hexdigest()
json.dump(out, open(res / f"{stem}.json", "w"), indent=1)
common.finish([res / f"{stem}.json", res / f"{stem}.npz", res / "dirs" / f"{stem}.npz"], summary=dict(tasks=len(out["tasks"])))

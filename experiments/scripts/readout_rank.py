"""The rank sweep at the readout (Section 5.4, "Higher ranks"; the theory is App. A.3), measured as experiments/scripts/readout.py
measures rank one, with every test pair's values kept so that the scorer (experiments/score/score_readout_rank.py) can give intervals.

readout.py's rows, states, gradients, clusters, resamples and seeds. The interchange is h_b + P P^T (h_s - h_b), P with k
orthonormal columns, at k = 1, 2, 4, 16, 64, at q = the task's number of distinct labels less one (when it lies strictly
between 1 and D - 1: with a linear readout M has at most q positive eigenvalues, so the top stops growing there) and at
k = D - 1 (I - u u^T, the complement of one direction u). At k = D the interchange is the whole state, whose effect is T and
whose first-order value is tr M: recorded, not trained. Per rank, trained on the training pairs and measured on the test pairs:
- DAS under three objectives, each from three starts, trained together (one Adam over the stacked parameters, each moved by
  its own loss: separate trainings; --stack=0 trains each alone): ce, sq and eff (-mean effect, the effect-seeking objective
  in pure form).
  Seeds 2000 + the task's index + 1000 s, as in readout.py: the first column of a start is readout.py's initial direction and
  the batch order is its order (both drawn on the CPU); further columns come from the seed + 7919, scaled by 1/sqrt(D). P is
  the QR basis of the parameter (k = 1: the parameter normalised, as in readout.py; k = D - 1: u = V/|V|). Ten passes (Adam
  5e-3, batch 4; each later pass in a new order from the same generator), kept after the first pass (CausalGym's budget) and
  the tenth. At k = 1 these repeat readout.py's das_{ce,sq,eff}_p{1,10}_s{0,1,2}, and the scorer compares the two.
- controls, not trained: the test M^mid's top-k eigenvectors (Ky Fan, in sample), the training M^mid's (out of sample) and a
  random subspace (seed 94); at k = D - 1 the complements of the bottom eigenvectors and of a random direction;
- the out-of-sample rank-k top (top_train): the subspace with the largest mean exact effect on the training pairs, by Adam on
  the parameter (QR basis; learning rate 1e-3 falling on a cosine to 1e-5, 100 steps, each start's best step on the training
  pairs) from the eff start with the largest training effect after ten passes and from the out-of-sample Ky Fan subspace,
  measured on the test pairs (readout.py's top_train at rank k; at k = 1 the same starts).
Per test pair, for every subspace: the exact effect and its first-order value with midpoint gradients, w^T P P^T Delta.
Per rank: the Ky Fan sums K_k (the k largest eigenvalues of the test M^mid, and of M^base, summed: the first-order rank-k
top, Theorem 5(iii)), and K_k of M^mid on readout.py's 2000 cluster resamples of the test pairs (numpy PCG64, seed 95;
their checksum is in the output), computed exactly in the row space of [W; Delta] with the zero eigenvalues of its
complement. For each subspace, its overlap with the in-sample Ky Fan subspace (|P^T U_k|_F^2 / k) and its
share of the training top direction (|P^T v1|^2).
Instruments, recorded here and judged by the scorer: readout.py's two (the readout map against the model's logits, and T
against the forward pass); every K_k at unit counts against the direct one (boot_check) and K_1 there against lambda_1 from
measure.lam_boot (lam_check), in units of the spectrum's scale; per rank, from first-order values alone: the Ky Fan subspace's
first-order value against K_k, and how far the largest first-order value of a stored subspace exceeds K_k (Ky Fan's principle
says it cannot), both in units of the scale; and the top's training value minus the largest start value (log odds; at least
0, since each start's best step includes the start). The run prints rank one's values and the times only.
Outputs: experiments/results/readout_rank/readout_rank_<model>_<tag>.json (per task, the point values) and .npz next to it (per
test pair, keys "<task>/k<k>/<subspace>/{e,fm}" in float32, "<task>/{T,g0}" and "<task>/kyfan_mid_boot" (B x ranks) in
float64, and "<task>/cluster" (integer cluster ids)).
Usage: python -I experiments/scripts/readout_rank.py --model=<hf id> [--revision=<sha>] --data=causalgym|fg|<dir> [--tasks=a,b]
       [--sign=class|label] [--seedfrom=causalgym] [--clone_unembed=1] [--stack=0] [--device=cuda] --tag=<name>
(--sign changes nothing here; experiments/matrix/run.py passes a group's options to every kind of run, and the value is recorded.)
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402
from rankmath import basis, first_order, ksum, kyfan_boot, overlap, patched  # noqa: E402
from measure import boot_counts, clusters, lam_boot, m_sym  # noqa: E402

_, args, model, tok = common.start("readout_rank", sys.argv[1:])
DEV = common.RUN["dev"]
BLOCKS, FIX = families.blocks(model), families.start_of_text(tok)
READ = families.readout(model, private_copy=args.get("clone_unembed") == "1")
DATA = args.get("data", "causalgym")
TASKS = (common.causalgym_tasks() if DATA == "causalgym" else []) if "tasks" not in args else args["tasks"].split(",")
RAW = {}
BLOCKS[-1].register_forward_hook(lambda m, a, o: RAW.__setitem__("h", o[0] if isinstance(o, tuple) else o))
LR, BATCH, PASSES, SNAP = 5e-3, 4, 10, (1, 10)
B, BOOT_SEED, RANDOM_SEED, TOP_STEPS, TOP_LR = 2000, 95, 94, 100, 1e-3
OBJS = ("ce", "sq", "eff")
OBJ_STARTS = tuple((o, s) for o in OBJS for s in (0, 1, 2))
STACK = args.get("stack", "1") == "1"
CG_ALL = common.causalgym_tasks() if args.get("seedfrom") == "causalgym" else None
SEEDIDX = (lambda ti, task: CG_ALL.index(task.replace("__num", "").replace("__lex", ""))) if CG_ALL else (lambda ti, task: ti)


def one(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None


def z_of(H, yb, ys):
    z = READ(H); ar = torch.arange(len(yb), device=H.device)
    return z, z[ar, ys] - z[ar, yb]                                   # logits and the log odds of y_s over y_b


def grad_at(H0, yb, ys):
    H = H0.clone().requires_grad_(True)
    W, = torch.autograd.grad(z_of(H, yb, ys)[1].sum(), H)
    return W


def side(rows):
    """readout.py's side(): the rows whose labels are single tokens, their last-position states, gradients and T."""
    keep = [r for r in rows if one(r["base_label"]) is not None and one(r["src_label"]) is not None]

    def last_states(seqs):                                            # batched by length: no padding, so no mask needed
        out, lg, groups = [None] * len(seqs), [None] * len(seqs), {}
        for i, q in enumerate(seqs):
            groups.setdefault(len(q), []).append(i)
        with torch.no_grad():
            for idx in groups.values():
                for k in range(0, len(idx), 64):
                    chunk = idx[k:k + 64]
                    z = model(torch.tensor([seqs[i] for i in chunk], device=DEV)).logits[:, -1]
                    for j, i in enumerate(chunk):
                        out[i] = RAW["h"][j, -1].clone(); lg[i] = z[j].clone()
        return torch.stack(out), torch.stack(lg)
    (hb, zb), (hs, zs) = (last_states([families.encode(tok, r[k], FIX)[0] for r in keep]) for k in ("base", "src"))
    S = dict(hb=hb, hs=hs, yb=torch.tensor([one(r["base_label"]) for r in keep], device=DEV),
             ys=torch.tensor([one(r["src_label"]) for r in keep], device=DEV),
             pairs=len({frozenset((r["base_label"], r["src_label"])) for r in keep}), n=len(keep), dropped=len(rows) - len(keep),
             labels=len({r["base_label"] for r in keep} | {r["src_label"] for r in keep}))
    S["cluster"] = clusters([(json.dumps(r["base"]), json.dumps(r["src"])) for r in keep])   # a row and its mirror: one cluster
    S["W"], S["D"] = grad_at(S["hb"], S["yb"], S["ys"]), S["hs"] - S["hb"]
    S["Wm"] = grad_at(S["hb"] + 0.5 * S["D"], S["yb"], S["ys"])       # the Rayleigh range's midpoint gradient
    with torch.no_grad():
        zb_, gb = z_of(S["hb"], S["yb"], S["ys"]); zs_, gs = z_of(S["hs"], S["yb"], S["ys"])
        S["g0"], S["T"] = gb, gs - gb
        ar = torch.arange(S["n"], device=DEV)
        Tf = (zs[ar, S["ys"]] - zs[ar, S["yb"]]) - (zb[ar, S["ys"]] - zb[ar, S["yb"]])
        S["inst"] = dict(readout_vs_logits=float(max((zb_ - zb).abs().max(), (zs_ - zs).abs().max())),
                         T_vs_forward=float((S["T"] - Tf).abs().max()))
    return S


def patch(S, P, J, k, Dm):
    """Exact effects of K rank-k interchanges at once: P (K, D, k) orthonormal bases (k = D - 1: (K, D) unit vectors u),
    J (K, b) each one's rows. Returns the logits (K*b rows) and the effects (K, b)."""
    H = patched(S["hb"][J], S["hs"][J], P, k, Dm)
    z, g = z_of(H.reshape(-1, Dm), S["yb"][J].reshape(-1), S["ys"][J].reshape(-1))
    return z, (g - S["g0"][J].reshape(-1)).reshape(J.shape)


def full(S, P, k, Dm):
    """The exact effect of one basis on every pair of S."""
    with torch.no_grad():
        return patch(S, P[None], torch.arange(S["n"], device=DEV)[None], k, Dm)[1][0]


def start_of(seed, k, Dm):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(Dm, generator=g).to(DEV); v = v / v.norm()          # readout.py's initial direction, normalised on the device
    if k == Dm - 1:
        V = v
    elif k == 1:
        V = v[:, None]
    else:
        V = torch.cat([v[:, None], (torch.randn(Dm, k - 1, generator=torch.Generator().manual_seed(seed + 7919)) / Dm ** 0.5).to(DEV)], 1)
    return g, V


def train(S, k, Dm, specs):
    """Every (objective, seed) of specs at rank k, for PASSES passes; returns {spec: {pass: (basis, (first, last tenth of
    the loss))}}. Each start keeps its own generator, start and batch orders, drawn in readout.py's sequence."""
    gens, V0 = [], []
    for _, sd in specs:
        g, V = start_of(sd, k, Dm); gens.append(g); V0.append(V)
    orders = [torch.randperm(S["n"], generator=g) for g in gens]
    V = torch.stack(V0).requires_grad_(True); opt = torch.optim.Adam([V], lr=LR)
    kind, ks = torch.tensor([OBJS.index(o) for o, _ in specs], device=DEV), torch.arange(len(specs), device=DEV)
    hist, out = [], {sp: {} for sp in specs}
    for p_ in range(PASSES):
        if p_ > 0:
            orders = [torch.randperm(S["n"], generator=g) for g in gens]
        for i in range(0, S["n"], BATCH):
            J = torch.stack([o[i:i + BATCH] for o in orders]).to(DEV)
            z, e = patch(S, basis(V, k, Dm), J, k, Dm)
            rows = torch.stack([torch.nn.functional.cross_entropy(z, S["ys"][J].reshape(-1), reduction="none").reshape(J.shape),
                                (e - S["T"][J]) ** 2, -e])           # in OBJS's order: ce, sq, eff
            L = rows[kind, ks].mean(1)
            opt.zero_grad(); L.sum().backward(); opt.step(); hist.append(L.detach())
        if p_ + 1 in SNAP:
            H = torch.stack(hist).cpu(); m = max(1, len(H) // 10)
            P = basis(V, k, Dm).detach()
            for j, sp in enumerate(specs):
                out[sp][p_ + 1] = (P[j].clone(), (float(H[:m, j].mean()), float(H[-m:, j].mean())))
    return out


def top_train(S, k, Dm, starts):
    """The largest mean exact effect on the pairs of S over rank-k bases, by Adam on the parameter from every start at once
    (each moved by its own loss), the learning rate falling from TOP_LR to TOP_LR/100 on a cosine; each start's best step is
    kept (the start included). Returns the best basis, its value and per start (name, best value, start value, best step)."""
    V = torch.stack([P.clone() for _, P in starts]).requires_grad_(True); opt = torch.optim.Adam([V], lr=TOP_LR)
    J = torch.arange(S["n"], device=DEV).expand(len(starts), -1)
    best = torch.full((len(starts),), -float("inf"), device=DEV); bestP = basis(V, k, Dm).detach().clone()
    step = torch.zeros(len(starts), dtype=torch.long, device=DEV); first = None
    for t in range(TOP_STEPS + 1):
        P = basis(V, k, Dm); val = patch(S, P, J, k, Dm)[1].mean(1)
        with torch.no_grad():
            first = val.detach().clone() if first is None else first
            up = val.detach() > best
            best = torch.where(up, val.detach(), best); step = torch.where(up, torch.full_like(step, t), step)
            bestP = torch.where(up.view(-1, *([1] * (P.dim() - 1))), P.detach(), bestP)
        if t == TOP_STEPS:
            break
        for g in opt.param_groups:
            g["lr"] = TOP_LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t / TOP_STEPS)))
        opt.zero_grad(); (-val.sum()).backward(); opt.step()
    b = int(torch.argmax(best))
    return bestP[b].clone(), float(best[b]), [dict(start=n, value=float(best[j]), start_value=float(first[j]), best_step=int(step[j]))
                                              for j, (n, _) in enumerate(starts)]


out = dict(model=common.RUN["model"], family=families.name(model), data=DATA, sign=args.get("sign", "class"), passes=PASSES, lr=LR,
           objectives=[list(a) for a in OBJ_STARTS], boot=dict(B=B, seed=BOOT_SEED, scheme="clusters (a row and its mirror), measure.boot_counts"),
           tasks={})
arrays, t0 = {}, time.time()
for ti, task in enumerate(TASKS):
    R = common.rows(DATA, task)
    tk = {"start": time.time()}
    tr, te = side(R["train"]), side(R["test"]); tk["states"] = time.time()
    if te["n"] < 20 or tr["n"] < 20:
        out["tasks"][task] = dict(skipped=True, n_test=te["n"], n_train=tr["n"], dropped=tr["dropped"] + te["dropped"]); continue
    Dm = te["hb"].shape[1]; T = float(te["T"].mean())
    q = te["labels"] - 1 if 1 < te["labels"] - 1 < Dm - 1 else None             # the label differences' dimension
    ranks = sorted({k for k in (1, 2, 4, 16, 64) if k < Dm - 1} | {Dm - 1} | ({q} if q else set()))
    Mte, Mte_b, Mtr = m_sym(te["Wm"], te["D"]), m_sym(te["W"], te["D"]), m_sym(tr["Wm"], tr["D"])
    ev, U = torch.linalg.eigh(Mte); evb = torch.linalg.eigvalsh(Mte_b); _, Utr = torch.linalg.eigh(Mtr)
    v1tr = Utr[:, -1].float()
    cnt = boot_counts(te["cluster"].tolist(), B, BOOT_SEED)
    counts = torch.tensor(cnt, dtype=torch.float64, device=DEV); ones = torch.ones(1, te["n"], dtype=torch.float64, device=DEV)
    kb = kyfan_boot(te["Wm"], te["D"], counts, ranks, Dm); kb1 = kyfan_boot(te["Wm"], te["D"], ones, ranks, Dm)[0]
    scale = float((te["W"].norm(dim=1) * te["D"].norm(dim=1)).mean())
    boot_check = max(abs(float(kb1[j]) - ksum(ev, k, Dm)) for j, k in enumerate(ranks)) / scale
    lam_check = abs(float(lam_boot(te["Wm"], te["D"], ones)[0]) - float(kb1[0])) / scale
    tk["spectrum"] = time.time()
    row = dict(T=T, D=Dm, ranks=ranks, rank_q=q, labels=te["labels"], label_pairs=te["pairs"], n_test=te["n"], n_train=tr["n"],
               dropped=tr["dropped"] + te["dropped"], stack=STACK,
               seed_index=SEEDIDX(ti, task), steps_per_pass=-(-tr["n"] // BATCH),
               K_mid_T={str(k): ksum(ev, k, Dm) / T for k in ranks}, K_base_T={str(k): ksum(evb, k, Dm) / T for k in ranks},
               trM_mid_T=float(ev.sum()) / T, trM_base_T=float(evb.sum()) / T, lambda1_mid_T=float(ev[-1]) / T,
               instruments=dict(train=tr["inst"], test=te["inst"], boot_check=boot_check, lam_check=lam_check), scale=scale,
               ranks_out={}, n_clusters_test=int(len(set(te["cluster"].tolist()))),
               boot_counts_sha=hashlib.sha256(np.ascontiguousarray(cnt, dtype=np.int64).tobytes()).hexdigest()[:16])
    arrays[f"{task}/T"] = te["T"].double().cpu().numpy(); arrays[f"{task}/g0"] = te["g0"].double().cpu().numpy()
    arrays[f"{task}/cluster"] = np.asarray(te["cluster"]); arrays[f"{task}/kyfan_mid_boot"] = kb.cpu().numpy()
    specs = [(o, 2000 + SEEDIDX(ti, task) + 1000 * s) for o, s in OBJ_STARTS]
    for k in ranks:
        tk[f"k{k}_start"] = time.time()
        trained = train(tr, k, Dm, specs) if STACK else {sp: train(tr, k, Dm, [sp])[sp] for sp in specs}
        subs, losses = {}, {}
        for (o, sd), snaps in trained.items():
            s = (sd - 2000 - SEEDIDX(ti, task)) // 1000
            for p_, (P, lf) in snaps.items():
                subs[f"das_{o}_p{p_}_s{s}"], losses[f"das_{o}_p{p_}_s{s}"] = P, lf
        if k == Dm - 1:
            subs["kyfan"], subs["kyfan_oos"] = U[:, 0].float(), Utr[:, 0].float()
            g = torch.Generator().manual_seed(RANDOM_SEED); u = torch.randn(Dm, generator=g); subs["random"] = (u / u.norm()).to(DEV)
        else:
            subs["kyfan"], subs["kyfan_oos"] = U[:, -k:].float(), Utr[:, -k:].float()
            subs["random"] = torch.linalg.qr(torch.randn(Dm, k, generator=torch.Generator().manual_seed(RANDOM_SEED)))[0].to(DEV)
        tr_eff = {f"das_eff_p10_s{s_}": float(full(tr, subs[f"das_eff_p10_s{s_}"], k, Dm).mean()) for s_ in (0, 1, 2)}
        eff_best = max(tr_eff, key=tr_eff.get)                       # chosen on the training pairs, as readout.py's top_train
        starts = [(eff_best, subs[eff_best]), ("kyfan_oos", subs["kyfan_oos"])]
        subs["top_train"], top_val, top_rec = top_train(tr, k, Dm, starts)
        Kk = ksum(ev, k, Dm)
        rec = dict(K_mid_T=Kk / T, losses=losses, top_train=dict(train_value_T=top_val / T, starts=top_rec, eff_best=eff_best),
                   subspaces={}, instruments={})
        for name, P in subs.items():
            e = full(te, P, k, Dm); fm = first_order(te["Wm"], te["D"], P, k, Dm); fb = first_order(te["W"], te["D"], P, k, Dm)
            ov = overlap(P, U[:, 0].float() if k == Dm - 1 else U[:, -k:].float(), k, Dm)
            share = None if k == Dm - 1 else float((P.T @ v1tr).pow(2).sum())
            rec["subspaces"][name] = dict(e_T=float(e.mean()) / T, fm_T=float(fm.mean()) / T, fb_T=float(fb.mean()) / T,
                                          overlap_kyfan=ov, share_v1_train=share)
            arrays[f"{task}/k{k}/{name}/e"] = e.float().cpu().numpy(); arrays[f"{task}/k{k}/{name}/fm"] = fm.float().cpu().numpy()
        fo = {n: v["fm_T"] * T for n, v in rec["subspaces"].items()}   # first-order means, log odds
        rec["instruments"] = dict(kyfan_first_order=abs(fo["kyfan"] - Kk) / scale,              # tr(U_k^T M U_k) = K_k
                                  above_kyfan=max(fo.values()) - Kk if max(fo.values()) > Kk else 0.0,
                                  top_start=top_val - max(x["start_value"] for x in top_rec))   # >= 0
        rec["instruments"]["above_kyfan"] /= scale
        row["ranks_out"][str(k)] = rec
        tk[f"k{k}"] = time.time()
    row["whole_state"] = dict(k=Dm, e_T=1.0, fm_T=float(ev.sum()) / T)          # P = I: the network's own interchange
    row["seconds"] = {k: round(tk[k] - tk[p], 2) for p, k in zip(list(tk)[:-1], list(tk)[1:])}
    out["tasks"][task] = row
    r1 = row["ranks_out"]["1"]                                         # the printout shows rank one only
    print(json.dumps({task: {"lambda1/T": round(r1["K_mid_T"], 2), "ce_p1": round(r1["subspaces"]["das_ce_p1_s0"]["e_T"], 2),
                             "ce_p10": round(r1["subspaces"]["das_ce_p10_s0"]["e_T"], 2), "ranks": len(ranks),
                             "seconds": round(sum(row["seconds"].values()))}}), f"{time.time() - t0:.0f}s", flush=True)
res = common.REPO / "experiments/results/readout_rank"; res.mkdir(parents=True, exist_ok=True)
stem = f"readout_rank_{common.RUN['tag']}_{args.get('tag', DATA)}"
f = res / f"{stem}.json"; fz = res / f"{stem}.npz"
np.savez_compressed(fz, **arrays)
out["npz_sha256"] = hashlib.sha256(fz.read_bytes()).hexdigest()
json.dump(out, open(f, "w"), indent=1)
common.finish([f, fz], summary=dict(tasks=len(out["tasks"])))

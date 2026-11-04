"""DAS and the network's own interchange at the readout (Section 5 and App. C.1), for one model and one task group, with every
test pair's values kept so that the scorer (experiments/score/score_readout.py) can give intervals.

The cell is the readout: the last block's raw output at the last position, before the final norm, where the network's own
interchange is T exactly (the complete-cut identity, Proposition 3). Per task, on the training and test rows whose labels
are single tokens:
- T, the change in the log odds of the source label over the base label from the base to the source state, read through the
  readout map; two instruments compare the readout map's logits with the model's and T with T from the forward pass;
- M = sym(mean w Delta^T) with midpoint gradients w_i = grad z(h_b + Delta/2) (the definition of Theorem 5) and
  with base-point gradients, on training and test rows; lambda_1, lambda_D, tr M and the top three eigenvalues of the test M;
- untrained directions from the training rows: u-hat (the mean difference, each row signed by its class, or by its base label
  with --sign=label), b-hat (the mean readout gradient signed the same way, turned toward u-hat), their bisector and v1 (the
  top eigenvector of M); b-hat, the bisector and v1 with base-point and with midpoint gradients;
- DAS under four objectives from three starts: ce (the source label's cross-entropy), jsd (the Jensen-Shannon divergence to the
  source state's next-token distribution), sq (the per-pair squared error to T; inside.py's sq_T) and eff (-mean effect,
  the effect-seeking objective in pure form). All twelve train together: one Adam over the stacked directions, each moved by
  its own loss, and Adam works per coordinate, so these are twelve separate trainings. Seeds 2000 + the task's index + 1000 s,
  s = 0, 1, 2; Adam 5e-3, batch 4, ten passes, each later pass in a new order from the same generator. Each direction is kept
  after the first pass (CausalGym's budget: 100 steps of batch 4 on its 400 training rows; other sources have other sizes, and
  the steps per pass are recorded) and after the tenth;
- sq_sur: the first-order optimum of the per-pair squared error on training rows, mean_i ((d.w_i)(d.Delta_i) - T_i)^2 over unit
  d (midpoint gradients), by Adam on an unnormalised v with d = v/|v| (measure.sphere_opt: 1000 steps, lr 1e-3 falling on a
  cosine to 1e-5, each start's best step kept) from u-hat and the midpoint b-hat, bisector and v1; nothing about the model is
  trained;
- top_train, the out-of-sample top: the unit d with the largest mean exact effect on the training pairs, by the same Adam
  (100 steps) from the eff start with the largest training effect after ten passes and from the midpoint v1, measured on the
  test pairs;
- top_exact, the exact top on test rows: the unit d with the largest mean exact effect on the test pairs, by the same Adam
  (300 steps) from the test M's top eigenvectors (midpoint and base point) and the direction with the largest test effect so
  far. Any other direction, trained or not, with a larger test effect replaces the optimiser's result. top_exact is the
  exact counterpart of lambda_1, which is also a maximum over the test pairs;
- per test pair, for every direction: the exact effect, and its first-order values with midpoint and base-point gradients;
- lambda_1 of the test M (midpoint and base point) on 2000 cluster resamples of the test pairs, a row and its mirror one
  cluster (numpy PCG64, seed 95, which the scorer regenerates; their checksum is in the output). It is computed exactly in the
  row space of [W; Delta], where M's nonzero spectrum lies, with the zero eigenvalue of the complement when that space is
  smaller than D (measure.lam_boot); boot_check compares it at unit counts with the direct eigenvalue;
- the two label logits before any soft cap at the base, source and midpoint states (raw_logits_b, _s, _m), and the compressed
  factors W Q and Delta Q (Q an orthonormal basis of the rows of [W; Delta]), from which M can be resampled without a GPU.
A task is skipped when fewer than 20 training or test rows have single-token labels.
--uncapped=1 (the diagnostic of Appendix C.4, Gemma-2's soft cap) removes the final-logit soft cap from the model
(config.final_logit_softcapping = None) before anything runs: the readout map and the model's own logits are then both
uncapped, and the instruments compare like with like.
--seedfrom=causalgym takes a task's index from CausalGym's sorted task list; a re-paired agreement task (number-only __num or
lemma-only __lex) takes the index of the CausalGym task it re-pairs. --clone_unembed=1 reads out through a private copy of
the unembedding (owneffect.families.readout). --stack=0 trains each (objective, start) alone, with the one-direction
arithmetic, to compare with the stacked training.
Outputs: experiments/results/readout/readout_<model>_<tag>.json (per task, the point values), .npz next to it (per test pair, keys
"<task>/<name>"), and every direction in experiments/results/readout/dirs/ (git-ignored).
Usage: python -I experiments/scripts/readout.py --model=<hf id> [--revision=<sha>] --data=causalgym|fg|<dir> [--tasks=a,b]
       [--sign=class|label] [--seedfrom=causalgym] [--clone_unembed=1] [--uncapped=1] [--stack=0] [--device=cuda] --tag=<name>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402
from measure import boot_counts, clusters, lam_boot, m_sym, sphere_opt, unit  # noqa: E402

_, args, model, tok = common.start("readout", sys.argv[1:])
DEV = common.RUN["dev"]
UNCAPPED, STACK = args.get("uncapped") == "1", args.get("stack", "1") == "1"
CAP = getattr(model.config, "final_logit_softcapping", None)        # recorded before --uncapped=1 removes it
if UNCAPPED:
    if not getattr(model.config, "final_logit_softcapping", None):
        sys.exit("--uncapped=1 is for a model with a final-logit soft-cap")
    model.config.final_logit_softcapping = None                       # the model's forward reads it at every call
BLOCKS, FIX = families.blocks(model), families.start_of_text(tok)
READ = families.readout(model, private_copy=args.get("clone_unembed") == "1")
DATA = args.get("data", "causalgym")
TASKS = (common.causalgym_tasks() if DATA == "causalgym" else []) if "tasks" not in args else args["tasks"].split(",")
RAW = {}
BLOCKS[-1].register_forward_hook(lambda m, a, o: RAW.__setitem__("h", o[0] if isinstance(o, tuple) else o))
LR, BATCH, PASSES, SNAP = 5e-3, 4, 10, (1, 10)
SEEDS, B, BOOT_SEED = (0, 1, 2), 2000, 95
OBJS = ("ce", "jsd", "sq", "eff")
CG_ALL = common.causalgym_tasks() if args.get("seedfrom") == "causalgym" else None
SEEDIDX = (lambda ti, task: CG_ALL.index(task.replace("__num", "").replace("__lex", ""))) if CG_ALL else (lambda ti, task: ti)
SIGN = "base_label" if args.get("sign") == "label" else "base_type"


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


def side(rows, types):
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
             sig=torch.tensor([1.0 if r[SIGN] == types[0] else -1.0 for r in keep], device=DEV),
             pairs=len({frozenset((r["base_label"], r["src_label"])) for r in keep}), n=len(keep), dropped=len(rows) - len(keep))
    S["cluster"] = clusters([(json.dumps(r["base"]), json.dumps(r["src"])) for r in keep])   # a row and its mirror: one cluster
    S["W"], S["D"] = grad_at(S["hb"], S["yb"], S["ys"]), S["hs"] - S["hb"]
    S["Wm"] = grad_at(S["hb"] + 0.5 * S["D"], S["yb"], S["ys"])       # the Rayleigh range's midpoint gradient
    with torch.no_grad():
        zb_, gb = z_of(S["hb"], S["yb"], S["ys"]); zs_, gs = z_of(S["hs"], S["yb"], S["ys"])
        S["g0"], S["T"], S["logq"] = gb, gs - gb, torch.log_softmax(zs_, -1)
        ar = torch.arange(S["n"], device=DEV)
        Tf = (zs[ar, S["ys"]] - zs[ar, S["yb"]]) - (zb[ar, S["ys"]] - zb[ar, S["yb"]])
        S["inst"] = dict(readout_vs_logits=float(max((zb_ - zb).abs().max(), (zs_ - zs).abs().max())),
                         T_vs_forward=float((S["T"] - Tf).abs().max()))
    return S


_NORM = families._attr(model, families._PATHS[families.name(model)][1]); _UNEMB = model.get_output_embeddings()


def label_logits(H, yb, ys):
    """The two label logits before any soft cap: the final norm and the unembedding's rows of y_s and y_b, the raw values that
    the analysis of Gemma-2's soft cap (Appendix C.4: experiments/score/score_readout.py, cap_mechanism.py, score_adequacy.py) and
    experiments/score/logit_scale_from_saved.py read."""
    with torch.no_grad():
        n = _NORM(H); Wu = _UNEMB.weight; b = _UNEMB.bias
        o = torch.stack([(n * Wu[ys]).sum(-1), (n * Wu[yb]).sum(-1)], 1)
        return o + torch.stack([b[ys], b[yb]], 1) if b is not None else o


def effect(S, d, idx=None):
    hb, hs = (S["hb"], S["hs"]) if idx is None else (S["hb"][idx], S["hs"][idx])
    yb, ys, g0 = (S["yb"], S["ys"], S["g0"]) if idx is None else (S["yb"][idx], S["ys"][idx], S["g0"][idx])
    z, g = z_of(hb + ((hs - hb) @ d)[:, None] * d, yb, ys)
    return z, g - g0


def jsd(logp, logq):
    p, q = logp.exp(), logq.exp(); logm = torch.log(0.5 * (p + q) + 1e-30)
    return 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))


def patched(S, Dn, J=None):
    """Exact effects of K rank-one interchanges at once: Dn (K, D) unit directions, J (K, b) each direction's rows (None: every
    row under every direction). Returns the logits (K*b rows), the effects (K, b) and J."""
    if J is None:
        J = torch.arange(S["n"], device=DEV).expand(Dn.shape[0], -1)
    hb, hs = S["hb"][J], S["hs"][J]
    if Dn.shape[0] == 1:                                              # one direction (--stack=0 or alone): one-direction arithmetic
        H = (hb[0] + ((hs[0] - hb[0]) @ Dn[0])[:, None] * Dn[0])[None]
    else:
        H = hb + ((hs - hb) * Dn[:, None, :]).sum(-1, keepdim=True) * Dn[:, None, :]
    z, g = z_of(H.reshape(-1, H.shape[-1]), S["yb"][J].reshape(-1), S["ys"][J].reshape(-1))
    return z, (g - S["g0"][J].reshape(-1)).reshape(J.shape), J


def train_all(S, specs):
    """DAS for every (objective, seed) of specs at once, for PASSES passes. Each start keeps its own generator (on the CPU),
    initial direction and batch orders, drawn in this sequence: direction, then order, and a new order for each later pass.
    One Adam over the stacked directions updates each from its own loss: Adam works per coordinate, so this is K separate
    trainings. Returns {(objective, seed): {pass: (direction, (first tenth, last tenth of the loss so far))}} for the passes in
    SNAP (1 and 10)."""
    gens = [torch.Generator().manual_seed(sd) for _, sd in specs]
    V0 = []
    for g in gens:
        v = torch.randn(S["hb"].shape[1], generator=g).to(DEV); V0.append(v / v.norm())   # normalised on the device
    orders = [torch.randperm(S["n"], generator=g) for g in gens]
    V = torch.stack(V0).requires_grad_(True); opt = torch.optim.Adam([V], lr=LR)
    kind, ks = torch.tensor([OBJS.index(o) for o, _ in specs], device=DEV), torch.arange(len(specs), device=DEV)
    hist, out = [], {sp: {} for sp in specs}
    for p_ in range(PASSES):
        if p_ > 0:
            orders = [torch.randperm(S["n"], generator=g) for g in gens]
        for i in range(0, S["n"], BATCH):
            J = torch.stack([o[i:i + BATCH] for o in orders]).to(DEV)
            z, e, _ = patched(S, V / V.norm(dim=1, keepdim=True), J)
            rows = torch.stack([torch.nn.functional.cross_entropy(z, S["ys"][J].reshape(-1), reduction="none").reshape(J.shape),
                                jsd(torch.log_softmax(z, -1), S["logq"][J].reshape(z.shape)).reshape(J.shape),
                                (e - S["T"][J]) ** 2, -e])           # in OBJS's order: ce, jsd, sq, eff
            L = rows[kind, ks].mean(1)
            opt.zero_grad(); L.sum().backward(); opt.step(); hist.append(L.detach())
        if p_ + 1 in SNAP:
            H = torch.stack(hist).cpu(); k = max(1, len(H) // 10)
            Dn = (V / V.norm(dim=1, keepdim=True)).detach()
            for j, sp in enumerate(specs):
                out[sp][p_ + 1] = (Dn[j].clone(), (float(H[:k, j].mean()), float(H[-k:, j].mean())))
    return out


out = dict(model=common.RUN["model"], family=families.name(model), data=DATA, sign=SIGN, uncapped=UNCAPPED, cap=CAP, stack=STACK,
           passes=PASSES,
           lr=LR, seeds=list(SEEDS), boot=dict(B=B, seed=BOOT_SEED, scheme="clusters (a row and its mirror), measure.boot_counts"),
           tasks={})
arrays, dir_store, t0 = {}, {}, time.time()
for ti, task in enumerate(TASKS):
    R = common.rows(DATA, task)
    types = sorted({r[SIGN] for r in R["train"]})
    tk = {"start": time.time()}
    tr, te = side(R["train"], types), side(R["test"], types); tk["states"] = time.time()
    if te["n"] < 20 or tr["n"] < 20:
        out["tasks"][task] = dict(skipped=True, n_test=te["n"], n_train=tr["n"], dropped=tr["dropped"] + te["dropped"]); continue
    s = tr["sig"][:, None]
    uh = unit((s * tr["D"]).mean(0)); bh = unit((s * tr["W"]).mean(0)); bhm = unit((s * tr["Wm"]).mean(0))
    # Both b-hats are turned toward u-hat before the angle is taken: cos theta = u-hat . b-hat is then never negative, so
    # theta is at most 90 degrees, and a b-hat at an angle phi above 90 degrees from u-hat is reported at 180 - phi. The turn
    # is not recorded.
    if float(uh @ bh) < 0:
        bh = -bh
    if float(uh @ bhm) < 0:
        bhm = -bhm
    Mtr, Mtr_m = m_sym(tr["W"], tr["D"]), m_sym(tr["Wm"], tr["D"])
    dirs = dict(u_hat=uh, b_hat=bh, bisector=unit(uh + bh), v1=torch.linalg.eigh(Mtr)[1][:, -1].float(),
                b_hat_mid=bhm, bisector_mid=unit(uh + bhm), v1_mid=torch.linalg.eigh(Mtr_m)[1][:, -1].float())
    losses = {}
    specs = [(obj, 2000 + SEEDIDX(ti, task) + 1000 * sd) for obj in OBJS for sd in SEEDS]
    trained = train_all(tr, specs) if STACK else {sp: train_all(tr, [sp])[sp] for sp in specs}
    tk["das"] = time.time()
    for (obj, seed), snaps in trained.items():
        sd = (seed - 2000 - SEEDIDX(ti, task)) // 1000
        for p_, (d, lf) in snaps.items():
            dirs[f"das_{obj}_p{p_}_s{sd}"], losses[f"das_{obj}_p{p_}_s{sd}"] = d, lf
    tk["dirs"] = time.time()
    sur_loss = lambda Dn: ((((tr["Wm"] @ Dn.T) * (tr["D"] @ Dn.T)) - tr["T"][:, None]) ** 2).mean(0)          # noqa: E731
    dirs["sq_sur"], sur_val, sur_rec = sphere_opt(sur_loss, [("u_hat", uh), ("b_hat_mid", bhm), ("bisector_mid", unit(uh + bhm)),
                                                             ("v1_mid", dirs["v1_mid"])], 1000, 1e-3)   # no trained start
    tk["sq_sur"] = time.time()
    with torch.no_grad():                                             # the out-of-sample top: fitted on training pairs only
        tr_eff = {k: float(patched(tr, dirs[k][None])[1].mean()) for k in ("das_eff_p10_s0", "das_eff_p10_s1", "das_eff_p10_s2")}
    eff_best = max(tr_eff, key=tr_eff.get)
    dirs["top_train"], toptr_val, toptr_rec = sphere_opt(lambda Dn: -patched(tr, Dn)[1].mean(1),
                                                         [(eff_best, dirs[eff_best]), ("v1_mid", dirs["v1_mid"])], 100, 1e-3)
    tk["top_train"] = time.time()
    Mte, Mte_m = m_sym(te["W"], te["D"]), m_sym(te["Wm"], te["D"])
    evb, Vb = torch.linalg.eigh(Mte); evm, Vm = torch.linalg.eigh(Mte_m)
    dirs["v1_test"], dirs["v1_test_mid"] = Vb[:, -1].float(), Vm[:, -1].float()
    with torch.no_grad():
        pre = {k: float(effect(te, d)[1].mean()) for k, d in dirs.items()}
    best_start = max(pre, key=pre.get)                                # the largest test effect among the directions so far
    starts = [(k, dirs[k]) for k in dict.fromkeys(("v1_test_mid", "v1_test", best_start))]   # up to three starts, no repeat
    dirs["top_exact"], _, top_rec = sphere_opt(lambda Dn: -patched(te, Dn)[1].mean(1), starts, 300, 1e-3)
    tk["top_exact"] = time.time()
    T = float(te["T"].mean())
    with torch.no_grad():
        E = {k: effect(te, d)[1] for k, d in dirs.items()}
        best_trained = max(float(E[k].mean()) for k in dirs if k != "top_exact")
        if best_trained > float(E["top_exact"].mean()):                # another direction, trained or not, beats the optimiser
            kbest = max((k for k in dirs if k != "top_exact"), key=lambda k: float(E[k].mean()))
            dirs["top_exact"], E["top_exact"] = dirs[kbest], E[kbest]
        else:
            kbest = None
        A = {k: dict(e=E[k], fm=(te["Wm"] @ d) * (te["D"] @ d), fb=(te["W"] @ d) * (te["D"] @ d)) for k, d in dirs.items()}
    cnt = boot_counts(te["cluster"].tolist(), B, BOOT_SEED)
    counts = torch.tensor(cnt, dtype=torch.float64, device=DEV)
    ones = torch.ones(1, te["n"], dtype=torch.float64, device=DEV)
    lam_m, lam_b = lam_boot(te["Wm"], te["D"], counts), lam_boot(te["W"], te["D"], counts)
    scale = float((te["W"].norm(dim=1) * te["D"].norm(dim=1)).mean())   # the spectrum's scale, the unit of boot_check
    boot_check = max(abs(float(lam_boot(te["Wm"], te["D"], ones)[0]) - float(evm[-1])),
                     abs(float(lam_boot(te["W"], te["D"], ones)[0]) - float(evb[-1]))) / scale
    tk["boot"] = time.time()
    c = float(uh @ bh); cm = float(uh @ bhm)
    share = float((te["sig"][:, None] * te["W"]).mean(0).norm() / te["W"].norm(dim=1).mean())
    share_m = float((te["sig"][:, None] * te["Wm"]).mean(0).norm() / te["Wm"].norm(dim=1).mean())
    mean = {k: {q: float(v.mean()) / T for q, v in a.items()} for k, a in A.items()}
    row = dict(T=T, lambda1=float(evb[-1]), trM=float(evb.sum()), label_pairs=te["pairs"], n_test=te["n"], n_train=tr["n"],
               dropped=tr["dropped"] + te["dropped"], cos_u_b=c, grad_share=share, one_angle=0.5 * (1 + 1 / c) if c > 0 else None,
               instruments=dict(train=tr["inst"], test=te["inst"]),
               # midpoint quantities, the per-direction means, the optimised directions and the resampling
               lambda1_mid=float(evm[-1]), trM_mid=float(evm.sum()), lambdaD=float(evb[0]), lambdaD_mid=float(evm[0]),
               top3=[float(x) for x in evb[-3:].flip(0)], top3_mid=[float(x) for x in evm[-3:].flip(0)],
               cos_u_b_mid=cm, one_angle_mid=0.5 * (1 + 1 / cm) if cm > 0 else None, grad_share_mid=share_m,
               mean_T=mean, losses=losses, cos_v1_mid={k: float(abs(d @ dirs["v1_mid"])) for k, d in dirs.items()},
               sq_sur=dict(loss=sur_val, starts=sur_rec), top_exact=dict(value_T=mean["top_exact"]["e"], starts=top_rec,
                                                                          replaced_by=kbest, best_start=best_start,
                                                                          gain_over_best_start_T=mean["top_exact"]["e"] - pre[best_start] / T),
               top_train=dict(value_T=mean["top_train"]["e"], train_value=-toptr_val, starts=toptr_rec),
               n_clusters_test=int(len(set(te["cluster"].tolist()))), n_clusters_train=int(len(set(tr["cluster"].tolist()))),
               steps_per_pass=-(-tr["n"] // BATCH), boot_check=boot_check,
               seconds={k: round(tk[k] - tk[p], 2) for p, k in zip(list(tk)[:-1], list(tk)[1:])},
               boot_counts_sha=hashlib.sha256(np.ascontiguousarray(cnt, dtype=np.int64).tobytes()).hexdigest()[:16])
    out["tasks"][task] = row
    for k in ("T", "g0", "sig"):
        arrays[f"{task}/{k}"] = te[k].double().cpu().numpy()
    arrays[f"{task}/cluster"] = np.asarray(te["cluster"])
    for where, H in (("b", te["hb"]), ("s", te["hs"]), ("m", te["hb"] + 0.5 * te["D"])):
        arrays[f"{task}/raw_logits_{where}"] = label_logits(H, te["yb"], te["ys"]).double().cpu().numpy()   # (y_s, y_b), pre-cap
    arrays[f"{task}/tr_mid"] = (te["Wm"] * te["D"]).sum(1).double().cpu().numpy()
    arrays[f"{task}/tr_base"] = (te["W"] * te["D"]).sum(1).double().cpu().numpy()
    arrays[f"{task}/lam_mid_boot"], arrays[f"{task}/lam_base_boot"] = lam_m.cpu().numpy(), lam_b.cpu().numpy()
    for k, a in A.items():
        for q, v in a.items():
            arrays[f"{task}/{q}/{k}"] = v.double().cpu().numpy()
    dir_store[task] = {k: d.float().cpu().numpy() for k, d in dirs.items()}
    for name, W in (("mid", te["Wm"]), ("base", te["W"])):          # compressed factors: any resampling of M without a GPU
        Q = torch.linalg.qr(torch.cat([W, te["D"]]).double().T)[0]
        dir_store[task][f"A_{name}"] = (W.double() @ Q).float().cpu().numpy(); dir_store[task][f"B_{name}"] = (te["D"].double() @ Q).float().cpu().numpy()
    print(json.dumps({task: {"T": round(T, 2), "l1/T": round(row["lambda1"] / T, 2), "l1m/T": round(row["lambda1_mid"] / T, 2),
                             "trm/T": round(row["trM_mid"] / T, 2), "top/T": round(mean["top_exact"]["e"], 2),
                             "ce1": round(mean["das_ce_p1_s0"]["e"], 2), "ce10": round(mean["das_ce_p10_s0"]["e"], 2),
                             "sq1": round(mean["das_sq_p1_s0"]["e"], 2), "sur": round(mean["sq_sur"]["e"], 2)}}),
          f"{time.time() - t0:.0f}s", flush=True)
res = common.REPO / "experiments/results/readout"; (res / "dirs").mkdir(parents=True, exist_ok=True)
stem = f"readout_{common.RUN['tag']}_{args.get('tag', DATA)}"
f = res / f"{stem}.json"; fz = res / f"{stem}.npz"; fd = res / "dirs" / f"{stem}.npz"
np.savez_compressed(fz, **arrays)
np.savez_compressed(fd, **{f"{t}/{k}": v for t, ds in dir_store.items() for k, v in ds.items()})
out["npz_sha256"] = hashlib.sha256(fz.read_bytes()).hexdigest()       # the json's sha256: in the unit's done record
out["dirs_sha256"] = hashlib.sha256(fd.read_bytes()).hexdigest()
json.dump(out, open(f, "w"), indent=1)
common.finish([f, fz, fd], summary=dict(tasks=len(out["tasks"])))

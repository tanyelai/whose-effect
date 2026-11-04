"""OwnEffect: how much of a learned intervention's effect is the network's own?

A subspace interchange swaps part of a state, along a direction or a subspace, from a source run into a base run. OwnEffect
measures its effect beside the network's own interchange at the same site (the swap of the whole state there), with
intervals, for a direction or subspace from any method (DAS, a probe, a difference of means, a steering vector). It also
computes the first-order quantities of the paper's theory (Section 3, the whole is a trace). The paper's conclusion gives the
three reporting steps that report() runs; the appendix on using the check describes this library.

  prepare(pairs)            tokenise the pairs and cache what the other methods need; returns the number of pairs dropped
  report(layer, region, d)  the three reporting steps at a cell inside the network: G*, the network's own interchange in its
                            best context; the direction's effect e; the excess e - G*; e / G*; and the share of e that
                            freezing attention removes; each with its interval, and the readings of the paper's figure on
                            what to report; context= measures G* in a context chosen on other pairs (the paper chooses it
                            on training pairs)
  readout(d)                at the readout, a complete cut where the network's own interchange is T exactly (the
                            complete-cut identity): T and two instrument checks, e / T with its interval, the Rayleigh
                            range, the trace test on a mirrored set and the one-angle top on a two-label set
  range(layer, region, d)   the Rayleigh range at a cell (the Rayleigh range theorem): tr M (the cell's own swap, to first
                            order), lambda_1 (the first-order top) and lambda_D, from midpoint gradients, and where d lies
  cut(layer)                the complete-cut identity on the model at hand: swapping every position of a layer gives T
  close()                   remove the hooks and the cached states (or use the object in a with statement)
  effect(), effects(), own(), grads()   the values behind these: a swap's mean effect and IIA, its per-pair effects, G* with
                            its context, and the per-pair gradients w and differences Delta at a cell

Units. z is the log odds of the source label over the base label at the last position (the difference of their logits).
An effect is the change in z that a swap makes in the base run, a log odds ratio; T, the whole-input effect, is z on the
source input minus z on the base input. report(), range() and readout() give effects in units of T, a mean over the
prepared pairs divided by the mean T (their docstrings mark the values in other units); effect(), effects(), own(), cut()
and readout()'s per-pair values give log odds.

Sites. A pair's spans are its regions. A cell is one region at one layer and swaps the state at the region's last token.
Layer l is the output of block l, counting from 0, which is Hugging Face hidden_states[l + 1] (the input to block l + 1). The
readout, the last block's raw output at the last position before the final norm, is a site of readout() only. Directions
saved by experiments/scripts/inside.py at its layer L (the input to block L) belong to layer L - 1 here.

Directions. A direction is a vector (rank one, normalised here) or a matrix whose columns span a subspace (rank k; an
orthonormal basis of its column space is taken here), in the space of the state at the site, as a numpy array or a torch
tensor. It is taken in the dtype and on the device of the model's parameters; the paper's runs and tolerances are float32.

Intervals resample clusters of pairs (a pair and its mirror, base and source exchanged, form one), B times; they are
percentile intervals, and basic intervals for lambda_1 (owneffect/stats.py). The swaps take forward passes only. The range
and the angle take one gradient per pair, at the swap's midpoint, and readout()'s trace test a second, at the base point.

Models: GPT-2, GPT-NeoX (Pythia), Llama, Qwen2 and Gemma-2, loaded with attn_implementation="eager", since freezing
attention patches the eager kernel (owneffect/families.py).

    from owneffect import OwnEffect
    oe = OwnEffect(model, tokenizer)                  # eager attention; OwnEffect puts the model in eval mode
    oe.prepare(pairs)                                 # [{"base": [spans], "src": [spans], "base_label": str, "src_label": str}]
    oe.report(layer=6, region=1, direction=d)         # inside: G*, e, e - G*, e / G*, frozen share, with intervals
    oe.readout(direction=d_out)                       # at the readout: e / T, the range, the trace test, the one-angle top
    oe.range(layer=6, region=1, direction=d)          # tr M, lambda_1, lambda_D, and where d lies

This is the reference implementation of the check of the paper "Whose Effect Is It? How Objectives Inflate Learned
Subspace Interventions". The paper's numbers come from the experiment scripts (experiments/scripts); experiments/scripts/owneffect_repro.py
checks this module against their saved results.
"""
import json
import warnings

import numpy as np
import torch

from . import families, stats


class OwnEffect:
    """The check on one model and its tokenizer (see the module docstring). batch is the number of pairs per forward pass.
    The constructor puts the model in eval mode and adds a forward pre-hook to every block, which patches only while a swap
    runs; close() removes the hooks and the cached states (or use the object in a with statement). The model must sit on
    one device."""

    def __init__(self, model, tokenizer, batch=25):
        if len({p.device for p in model.parameters()}) > 1:
            raise ValueError("the model's parameters are on several devices; OwnEffect needs the model on one device")
        self.model, self.tok, self.batch = model.eval(), tokenizer, batch
        self.blocks = families.blocks(model)
        self.att = families.attention(model)               # the family's shared switch: "normal", "record" or "frozen"
        self.dev, self.dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.fix = families.start_of_text(tokenizer)
        self.read = families.readout(model, private_copy=self.dev.type == "cpu")   # on CPU, a private copy (families.readout)
        self.site, self.rows, self.empty = None, [], []     # site, while a swap runs: (block index, [(positions, patch)])
        # Each block's pre-hook patches that block's input when the current swap's site is this block.
        self.hooks = [blk.register_forward_pre_hook(
            lambda m, a, kw, i=i: self._patch(a, kw) if self.site and self.site[0] == i else None, with_kwargs=True)
            for i, blk in enumerate(self.blocks)]

    def close(self):
        """Removes the blocks' hooks and drops the cached states. The family's attention kernel stays patched: it is shared by
        every model of the family in the process and computes the unchanged kernel unless a swap freezes it."""
        for h in self.hooks:
            h.remove()
        self.hooks, self.rows = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ data
    def _encode(self, spans):
        """Token ids of a list of spans and the index of each span's last token (families.encode)."""
        return families.encode(self.tok, spans, self.fix)

    def prepare(self, pairs):
        """Tokenise the pairs and run each base and source input once. Caches per pair: the source run's state at the input
        to every block, at every position; both runs' readout states (the last block's raw output at the last position); and
        T. Also records the clusters (a pair and its mirror, base and source exchanged, form one; matched by their spans) and
        an instrument, readout_vs_logits: the largest absolute difference, over the vocabulary and both runs of every pair,
        between the readout map applied to the readout state and the model's own logits.

        pairs: a list of {"base": [spans], "src": [spans], "base_label": str, "src_label": str}. The spans are the regions,
        each tokenised on its own with no special tokens added; the first span holds CausalGym's start-of-text marker
        "<|endoftext|>", which families.start_of_text writes as the tokenizer's own start-of-text token where needed (a
        warning says when it does not). Each label must be one token, and every pair has the regions of the first.
        Drops every pair whose labels are not single tokens or whose base or source has another number of spans than the
        first pair's base, warns with the count for each reason, and returns the number dropped; raises ValueError when no
        pair is left. A region that is empty in some pair, whose swap would take another region's token, is listed in
        self.empty with a warning: effects() refuses it and regions() leaves it out. Warns when the mean T is not positive.
        Each call replaces the pairs prepared before."""
        self.rows, keys, inst, raw = [], [], 0.0, {}
        dropped, n_regions = dict(labels=0, regions=0), None
        sot = self._encode([families.START])[0]            # the start-of-text token, as the first span should encode
        # records the last block's raw output on every forward pass while prepare() runs
        hook = self.blocks[-1].register_forward_hook(lambda m, a, o: raw.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            with torch.no_grad():
                for p in pairs:
                    bi, be = self._encode(p["base"]); si, se = self._encode(p["src"])
                    yb = self.tok(p["base_label"], add_special_tokens=False)["input_ids"]
                    ys = self.tok(p["src_label"], add_special_tokens=False)["input_ids"]
                    if len(yb) != 1 or len(ys) != 1:
                        dropped["labels"] += 1; continue
                    n_regions = len(be) if n_regions is None else n_regions
                    if len(be) != n_regions or len(se) != n_regions:
                        dropped["regions"] += 1; continue
                    ob = self.model(torch.tensor([bi], device=self.dev), output_hidden_states=True); hb_out = raw["h"][0, -1].clone()
                    os_ = self.model(torch.tensor([si], device=self.dev), output_hidden_states=True); hs_out = raw["h"][0, -1].clone()
                    zb, zs = ob.logits[0, -1], os_.logits[0, -1]
                    inst = max(inst, float((self.read(hb_out[None])[0] - zb).abs().max()),
                               float((self.read(hs_out[None])[0] - zs).abs().max()))
                    T = float((zb[yb[0]] - zb[ys[0]]) - (zs[yb[0]] - zs[ys[0]]))         # z(source) - z(base)
                    hs = torch.stack(list(os_.hidden_states[:len(self.blocks)]))[:, 0]   # hs[b]: the input to block b
                    self.rows.append(dict(b=bi, be=be, se=se, yb=yb[0], ys=ys[0], hs=hs, T=T, hb_out=hb_out, hs_out=hs_out,
                                          aligned=len(bi) == len(si) and be == se))
                    keys.append((json.dumps(p["base"]), json.dumps(p["src"])))
        finally:
            hook.remove()
        if dropped["labels"] or dropped["regions"]:
            warnings.warn(f"prepare() dropped {dropped['labels']} pairs whose labels are not single tokens and "
                          f"{dropped['regions']} whose span counts differ from the first pair's ({n_regions})")
        if not self.rows:
            raise ValueError("no pair is left to prepare")
        if any(r["b"][:r["be"][0] + 1] != sot or r["se"][0] != r["be"][0] for r in self.rows):
            warnings.warn("the first span is not the start-of-text token in every pair: region 0 is never swapped and never a "
                          "context, as in the paper, so a region of the sentence there is left out")
        # a region whose span is empty ends where the one before it ends (region 0: before the first token)
        self.empty = sorted({g for r in self.rows for ends in (r["be"], r["se"]) for g in range(n_regions)
                             if ends[g] == (ends[g - 1] if g else -1)})
        if self.empty:
            warnings.warn(f"regions {self.empty} are empty in some pair, where a swap would take another region's token: "
                          "effects() refuses them and regions() leaves them out")
        self.Tpairs = np.array([r["T"] for r in self.rows])
        self.T = float(self.Tpairs.mean())
        self.cluster = stats.clusters(keys)
        self.instrument = dict(readout_vs_logits=inst)
        if self.T <= 0:
            warnings.warn(f"the mean whole-input effect T = {self.T:.4g} is not positive: values in units of T change sign (the "
                          "paper's population requires T's interval above zero)")
        return dropped["labels"] + dropped["regions"]

    # ------------------------------------------------------------------ patching
    def _patch(self, a, kw):
        """The pre-hook's patch: at each of the site's positions (one per batch row), the block's input state is replaced by
        the patch function of it. The block receives its input as the keyword hidden_states or as its first argument."""
        h = kw["hidden_states"] if "hidden_states" in kw else a[0]
        h = h.clone(); ar = torch.arange(h.shape[0], device=h.device)
        for pos, fn in self.site[1]:
            h[ar, pos] = fn(h[ar, pos])
        if "hidden_states" in kw:
            return a, dict(kw, hidden_states=h)
        return (h,) + tuple(a[1:]), kw

    def _logits(self, rs, block=None, patches=()):
        """Logits at each pair's last token for a batch of base inputs, right-padded with an attention mask; with `block`,
        the patches apply to that block's input."""
        T_ = max(len(r["b"]) for r in rs)
        ids = torch.zeros(len(rs), T_, dtype=torch.long); am = torch.zeros(len(rs), T_, dtype=torch.long)
        for i, r in enumerate(rs):
            ids[i, :len(r["b"])] = torch.tensor(r["b"]); am[i, :len(r["b"])] = 1
        self.site = (block, patches) if block is not None else None
        try:
            z = self.model(ids.to(self.dev), attention_mask=am.to(self.dev)).logits     # use_cache at its default, as inside.py
        finally:
            self.site = None
        return z[torch.arange(len(rs), device=self.dev), torch.tensor([len(r["b"]) - 1 for r in rs], device=self.dev)]

    def _basis(self, direction):
        """The direction in the model's dtype on its device: a vector scaled to unit length, or an orthonormal basis of a
        matrix's column space (from its singular vectors; a warning says when the columns are linearly dependent, and the
        basis then spans their column space); and whether it is a vector. Accepts numpy arrays and torch tensors, and
        raises ValueError for a zero direction."""
        if torch.is_tensor(direction):
            direction = direction.detach().cpu().numpy()
        P = torch.as_tensor(np.asarray(direction, dtype=np.float64))
        if P.dim() == 1:
            if P.norm() == 0:
                raise ValueError("the direction is the zero vector")
            return (P / P.norm()).to(self.dev, self.dtype), True
        U, S, _ = torch.linalg.svd(P, full_matrices=False)
        keep = S > S.max() * max(P.shape) * torch.finfo(torch.float32).eps
        if not keep.any():
            raise ValueError("the direction matrix is zero")
        if not keep.all():
            warnings.warn(f"the direction matrix has rank {int(keep.sum())}, below its {P.shape[1]} columns: the swap uses "
                          "their column space")
        return U[:, keep].to(self.dev, self.dtype), False

    @staticmethod
    def _swap(src, P=None, one=True):
        """The patch function of a swap: the source state itself (P None), or the base state h plus the projection of
        src - h on P (h + d d^T (src - h) for a unit vector d, h + P P^T (src - h) for a basis P)."""
        if P is None:
            return lambda h: src
        if one:
            return lambda h: h + ((src - h) @ P)[:, None] * P
        return lambda h: h + ((src - h) @ P) @ P.T

    def _block(self, layer):
        """The block whose input is layer `layer` (the output of block `layer`; layer -1 is the embeddings, block 0's input):
        block layer + 1. Raises ValueError below -1 and for the last block's output, the readout, which readout() takes."""
        block = layer + 1
        if block < 0:
            raise ValueError(f"layer {layer}: layers run from -1 (the embeddings) to {len(self.blocks) - 2}")
        if block >= len(self.blocks):
            raise ValueError("the last block's output is the readout, a complete cut: there the network's own interchange is T "
                             "exactly (the complete-cut identity); use readout()")
        return block

    def _regions(self, regions):
        """`regions` (an int or a list) as a list of ints, after the checks: no region 0 (the start-of-text span, which the
        paper never swaps), no region outside the pairs' and none that is empty in some pair (self.empty)."""
        regions = [int(regions)] if isinstance(regions, (int, np.integer)) else [int(g) for g in regions]
        n = len(self.rows[0]["be"])
        for g in regions:
            if not 1 <= g < n:
                raise ValueError(f"region {g}: the regions run from 1 to {n - 1} (region 0, the start of text, is never swapped)")
            if g in self.empty:
                raise ValueError(f"region {g} is empty in some pair, where its swap would take another region's token")
        return regions

    def effects(self, layer, regions, direction=None, frozen=False):
        """Per pair, as numpy arrays: the swap's effect, the change in z in the base run (log odds), and whether the patched
        output prefers the source label to the base label (1.0 or 0.0). The swap takes the source run's state at the last
        token of each region in `regions` (an int or a list) at layer `layer`: the whole state if `direction` is None, else
        its component along `direction`, the same direction at each region. frozen=True holds every attention pattern at the
        base run's during the swap. Raises ValueError for a region _regions() refuses."""
        regions = self._regions(regions)
        block = self._block(layer)
        P, one = self._basis(direction) if direction is not None else (None, True)
        out, hit = [], []
        with torch.no_grad():
            for i in range(0, len(self.rows), self.batch):
                rs = self.rows[i:i + self.batch]
                patches = []
                for g in regions:
                    pos = torch.tensor([r["be"][g] for r in rs], device=self.dev); src = torch.stack([r["hs"][block, r["se"][g]] for r in rs])
                    patches.append((pos, self._swap(src, P, one)))
                # the unpatched base run records its attention patterns; the swapped run replays them if frozen
                self.att.update(mode="record", store={})
                try:
                    z0 = self._logits(rs)
                    if frozen and len(self.att["store"]) != len(self.blocks):
                        raise RuntimeError(f"the eager attention kernel recorded {len(self.att['store'])} of {len(self.blocks)} "
                                           "blocks' patterns, so freezing attention would leave the others free")
                    self.att["mode"] = "frozen" if frozen else "normal"
                    z1 = self._logits(rs, block, patches)
                finally:
                    self.att["mode"] = "normal"
                for j, r in enumerate(rs):
                    out.append(float((z0[j, r["yb"]] - z0[j, r["ys"]]) - (z1[j, r["yb"]] - z1[j, r["ys"]])))
                    hit.append(float(z1[j, r["ys"]] > z1[j, r["yb"]]))
        return np.array(out), np.array(hit)

    def effect(self, layer, regions, direction=None, frozen=False):
        """The means of effects() over the prepared pairs: (effect, IIA). The effect is a mean log odds ratio (not in units
        of T); IIA is the share of pairs whose patched output prefers the source label to the base label. The arguments are
        those of effects()."""
        e, hit = self.effects(layer, regions, direction, frozen)
        return float(e.mean()), float(hit.mean())

    def regions(self):
        """The regions a cell can be at, and own()'s partners by default: every region of the prepared pairs but region 0
        (the start-of-text span) and those empty in some pair (self.empty)."""
        return [g for g in range(1, len(self.rows[0]["be"])) if g not in self.empty]

    def own(self, layer, region, partners=None, per_pair=False, context=None):
        """G*, the network's own interchange at the cell (layer, region) in its best context: the whole-state swap of the
        cell alone, or the effect it adds when swapped together with one partner region y of the same layer (the swap of
        both minus the swap of y alone), whichever has the larger mean over the prepared pairs. partners defaults to
        regions(), every region but region 0, as in the paper; pass a list to restrict them (the cell's own region is
        skipped). Returns (G*, context): G* in log odds, its mean or, with per_pair, a numpy array per pair; context is
        "alone" or "with region y".
        context: None chooses the context on the prepared pairs, the pairs G* is then measured on, which pushes G* up
        where the pairs are few. Pass a context chosen elsewhere ("alone", a partner region y, or the context own()
        returned) to measure G* in it without choosing. The paper's protocol: prepare the training pairs and take own()'s context there, then prepare
        the test pairs and pass that context."""
        if isinstance(context, str) and context.startswith("with region "):   # own()'s returned context, passed back
            context = int(context[len("with region "):])
        if context is not None:
            if context == "alone":
                g, ctx = self.effects(layer, region)[0], "alone"
            else:
                if not isinstance(context, (int, np.integer)) or isinstance(context, bool) or context == region:
                    raise ValueError(f"context must be None, 'alone' or a partner region other than {region}, not {context!r}")
                context = int(context)
                g, ctx = self.effects(layer, [region, context])[0] - self.effects(layer, context)[0], f"with region {context}"
            return (g if per_pair else float(g.mean())), ctx
        partners = self.regions() if partners is None else partners
        best, ctx = self.effects(layer, region)[0], "alone"
        for y in partners:
            if y == region:
                continue
            region_gain = self.effects(layer, [region, y])[0] - self.effects(layer, y)[0]
            if region_gain.mean() > best.mean():
                best, ctx = region_gain, f"with region {y}"
        return (best if per_pair else float(best.mean())), ctx

    # ------------------------------------------------------------------ the paper's reporting steps, with intervals
    def report(self, layer, region, direction, partners=None, B=2000, seed=95, level=0.95, context=None):
        """The paper's three reporting steps at the cell (layer, region) inside the network, for `direction` (a vector or a
        matrix, as in effects()). Intervals: B >= 1 cluster resamples drawn with `seed`, percentile intervals at `level`,
        each ratio of means taken on the same resample. Returns a dict:
          cell: the layer, the region and G*'s context (own()); pairs, clusters: their numbers;
          population: whether the prepared pairs meet the paper's rule, at least 20 clusters (clusters_ok) and T's interval
            above zero (T_above_zero); a warning says when they do not, since values in units of T then mean little;
          T: the mean T (log odds) with its interval;
          own (step 1): G*; effect: the direction's effect e; excess: e - G*; frozen: e with every attention pattern held at
            the base run's; these four in units of T, with intervals;
          iia: the share of pairs whose patched output prefers the source label (along the direction);
          ratio (step 2): e / G* with its interval, where G*'s interval lies above zero, else None;
          frozen_removed (step 3): 1 - e_frozen / e, the share of e that freezing attention removes, with its interval,
            where e's interval excludes zero, else None;
          readings: stats.readings() of these intervals: carried, excess (the count of the paper's Section 6),
            not_evidence_of_use (the conclusion's third step), off_route (Algorithm 1) and reading, the label of the
            paper's figure on what to report.
        partners and context are own()'s. Without a context, it is chosen once, on all prepared pairs, and kept in every
        resample; pass one chosen on other pairs (own()) to follow the paper's protocol."""
        if B < 1:
            raise ValueError("B, the number of cluster resamples, must be at least 1")
        G, ctx = self.own(layer, region, partners, per_pair=True, context=context)
        e, hit = self.effects(layer, region, direction)
        ef = self.effects(layer, region, direction, frozen=True)[0]
        C, T = stats.boot_counts(self.cluster, B, seed), self.Tpairs

        def in_T(x):                        # mean(x) / mean(T), with its interval from the same resamples
            pt = float(x.mean() / T.mean())
            return dict(value=pt, interval=list(stats.interval(pt, stats.boot_ratio(x, T, C), level)))
        out = dict(cell=dict(layer=layer, region=region, context=ctx), pairs=len(T), clusters=int(len(np.unique(self.cluster))),
                   T=dict(value=self.T, interval=list(stats.interval(self.T, C @ T / C.sum(1), level))),
                   own=in_T(G), effect=in_T(e), excess=in_T(e - G), frozen=in_T(ef), iia=float(hit.mean()), ratio=None,
                   frozen_removed=None)
        out["population"] = self._population(out["clusters"], out["T"]["interval"])
        g_int, e_int = out["own"]["interval"], out["effect"]["interval"]
        if g_int[0] > 0:
            pt = float(e.mean() / G.mean())
            out["ratio"] = dict(value=pt, interval=list(stats.interval(pt, stats.boot_ratio(e, G, C), level)))
        if e_int[0] > 0 or e_int[1] < 0:
            pt = float(1 - ef.mean() / e.mean())
            out["frozen_removed"] = dict(value=pt, interval=list(stats.interval(pt, 1 - stats.boot_ratio(ef, e, C), level)))
        out["readings"] = stats.readings(g_int, e_int, out["excess"]["interval"], out["ratio"] and out["ratio"]["interval"])
        return out

    @staticmethod
    def _population(clusters, T_interval):
        """The paper's population rule for the prepared pairs: at least 20 clusters, and T's interval above zero."""
        pop = dict(clusters_ok=clusters >= 20, T_above_zero=T_interval[0] > 0)
        if not all(pop.values()):
            warnings.warn(f"the pairs miss the paper's population rule ({clusters} clusters, at least 20 needed; T's interval "
                          f"{'lies' if pop['T_above_zero'] else 'does not lie'} above zero)")
        return pop

    # ------------------------------------------------------------------ the theory's quantities
    def grads(self, layer, region, point="mid"):
        """Per pair, as torch tensors (pairs x dim) on the model's device: w, the gradient of z in the state of the cell
        (layer, region), with the rest of the base run recomputed downstream, at the midpoint of the cell's whole-state swap
        (point="mid", the w of the Rayleigh range theorem) or at the base state (point="base"); and Delta = h_s - h_b, the
        cell's source-minus-base difference. stats.spectrum(w, Delta) gives the spectrum of M = sym(E[w Delta^T])."""
        region, = self._regions(region)
        block = self._block(layer)
        W, D = [], []
        for i in range(0, len(self.rows), self.batch):
            rs = self.rows[i:i + self.batch]
            pos = torch.tensor([r["be"][region] for r in rs], device=self.dev); src = torch.stack([r["hs"][block, r["se"][region]] for r in rs])
            box = {}

            def fn(h, src=src, box=box):            # moves the state to the point where w is taken, kept for autograd
                d = (src - h).detach()
                x = (h.detach() + 0.5 * d if point == "mid" else h.detach()).requires_grad_(True)
                box["x"], box["d"] = x, d
                return x
            with torch.enable_grad():
                z = self._logits(rs, block, [(pos, fn)])
                ar = torch.arange(len(rs), device=self.dev)
                zz = z[ar, torch.tensor([r["ys"] for r in rs], device=self.dev)] - z[ar, torch.tensor([r["yb"] for r in rs], device=self.dev)]
                g, = torch.autograd.grad(zz.sum(), box["x"])
            W.append(g.detach()); D.append(box["d"])
        return torch.cat(W), torch.cat(D)

    def _lam_interval(self, W, D, lam1, B, seed, level):
        """The basic interval of lambda_1 / T: on each of B cluster resamples, lambda_1 of the resample's M over the
        resample's mean T (the denominator on the same resample, as for every ratio here)."""
        C = stats.boot_counts(self.cluster, B, seed)
        lam = stats.lam_boot(W, D, torch.tensor(C, dtype=torch.float64, device=W.device)).cpu().numpy()
        return list(stats.interval(lam1 / self.T, lam / (C @ self.Tpairs / C.sum(1)), level, basic=True))

    @staticmethod
    def _first_order(W, D, P, one):
        """Per pair, a swap's first-order effect along P: (w . d)(Delta . d) for a unit vector d, summed over the columns of
        a basis P, i.e. w^T P P^T Delta. Its mean is d^T M d, or tr P^T M P."""
        P2 = P[:, None] if one else P
        return ((W @ P2) * (D @ P2)).sum(1)

    def range(self, layer, region, direction=None, B=0, seed=95, level=0.95):
        """The Rayleigh range theorem at the cell (layer, region) inside the network: M = sym(E[w Delta^T]) with w the
        gradient of z at the midpoint of the cell's swap (grads()). Returns a dict, in units of T except v1:
          cell: the layer and the region;
          trM: tr M, the cell's own swap to first order; own_exact: that swap's exact effect;
          lambda1: M's largest eigenvalue, the first-order top (no rank-one swap goes higher, to first order); lambdaD: the
            smallest; v1: the top eigenvector, a unit vector in the state's space (None when the top is the eigenvalue 0 of
            the directions the pairs' w and Delta do not span);
          with B > 0, lambda1_interval: the basic interval of lambda_1 / T over B cluster resamples, both recomputed on each;
          with a direction, direction: its first-order effect (d^T M d, or tr P^T M P for a subspace) and its exact effect.
        B defaults to 0 here (no interval), where report() and readout() default to 2000."""
        W, D = self.grads(layer, region, "mid")
        sp, Tm = stats.spectrum(W, D), self.T
        out = dict(cell=dict(layer=layer, region=region), trM=sp["tr"] / Tm, lambda1=sp["lam1"] / Tm, lambdaD=sp["lamD"] / Tm,
                   own_exact=float(self.effects(layer, region)[0].mean()) / Tm, v1=sp["v1"])
        if B:
            out["lambda1_interval"] = self._lam_interval(W, D, sp["lam1"], B, seed, level)
        if direction is not None:
            P, one = self._basis(direction)
            out["direction"] = dict(first_order=float(self._first_order(W, D, P, one).mean()) / Tm,
                                    exact=float(self.effects(layer, region, direction)[0].mean()) / Tm)
        return out

    def readout(self, direction=None, B=2000, seed=95, level=0.95, per_pair=False):
        """The check at the readout, the last block's raw output at the last position (before the final norm): a complete
        cut, where the network's own interchange equals T exactly (the complete-cut identity), so e / T is the reading.
        Works on the readout states prepare() cached: swaps and gradients run through the readout map (families.readout),
        not through the whole model. `direction` lives in the space of that raw output. Intervals as in report(); B must
        be at least 1. Returns a dict:
          pairs, clusters: their numbers; population: as in report(); T: the mean T (log odds) with its interval;
          instruments, both near zero when the readout map reproduces the model: readout_vs_logits (from prepare(), in
            logits) and T_vs_forward, the largest per-pair difference between T recomputed from the readout states and T
            from the forward runs (log odds);
          range, in units of T: trM_mid and trM_base, tr M with midpoint and with base-point gradients; lambda1 and the
            basic interval of lambda_1 / T (both recomputed on each resample); lambdaD; v1, the top eigenvector (as in
            range());
          trace_test, the trace's error on a mirrored set (the appendix on the error of the midpoint expansion), when every
            cluster is a pair with its mirror and tr M_base differs from T, else None: ratio, (tr M_mid - T) / (tr M_base -
            T), near -1/2 where the expansion's leading (third-order) error dominates both traces; simpson,
            (tr M_base + 2 tr M_mid) / 3 in units of T, Simpson's rule, which gives T to fifth order;
          angle (the proposition on one shared direction), when every pair contrasts the same two labels, else None:
            theta_deg, the angle between the mean change u and the mean midpoint gradient b, each pair signed by which of the
            two labels is its base label (between 0 and 180 degrees; the sign convention flips u and b together, so it does
            not change the angle); top, the predicted lambda_1 / tr M = (1 + 1 / cos theta) / 2, None at 90 degrees or more,
            where the prediction is undefined; readout_share, |b| over the mean |w|, 1 when every pair's signed gradient
            points the same way. The angle needs no direction: on prepared training pairs it predicts the top before any
            direction is trained;
          direction (with a direction): effect, the exact e / T with its interval; first_order, its first-order value in
            units of T;
          per_pair (if per_pair; numpy, log odds): T, recomputed from the readout states; tr_mid and tr_base, w . Delta
            with midpoint and with base-point gradients; effect, the direction's exact effect (None without a direction)."""
        if B < 1:
            raise ValueError("B, the number of cluster resamples, must be at least 1")
        rows = self.rows
        Hb, Hs = torch.stack([r["hb_out"] for r in rows]), torch.stack([r["hs_out"] for r in rows])
        yb = torch.tensor([r["yb"] for r in rows], device=self.dev); ys = torch.tensor([r["ys"] for r in rows], device=self.dev)
        ar = torch.arange(len(rows), device=self.dev)

        def z(H):                           # z per pair from readout states, through the readout map
            o = self.read(H)
            return o[ar, ys] - o[ar, yb]

        def grad(H):                        # w per pair: the gradient of z at the readout states H
            H = H.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                g, = torch.autograd.grad(z(H).sum(), H)
            return g
        with torch.no_grad():
            zb = z(Hb); Tr = (z(Hs) - zb).cpu().numpy()
        D = Hs - Hb; Wb, Wm = grad(Hb), grad(Hb + 0.5 * D)                 # Delta; w at the base point and at the midpoint
        T, Tm = self.Tpairs, self.T
        C = stats.boot_counts(self.cluster, B, seed)
        out = dict(pairs=len(T), clusters=int(len(np.unique(self.cluster))),
                   T=dict(value=Tm, interval=list(stats.interval(Tm, C @ T / C.sum(1), level))),
                   instruments=dict(readout_vs_logits=self.instrument["readout_vs_logits"], T_vs_forward=float(np.abs(Tr - T).max())))
        out["population"] = self._population(out["clusters"], out["T"]["interval"])
        spm, spb = stats.spectrum(Wm, D), stats.spectrum(Wb, D)
        out["range"] = dict(trM_mid=spm["tr"] / Tm, trM_base=spb["tr"] / Tm, lambda1=spm["lam1"] / Tm,
                            lambda1_interval=self._lam_interval(Wm, D, spm["lam1"], B, seed, level),
                            lambdaD=spm["lamD"] / Tm, v1=spm["v1"])
        sizes = np.unique(self.cluster, return_counts=True)[1]            # mirrored set: every cluster has two pairs
        out["trace_test"] = None
        if len(sizes) and (sizes == 2).all() and spb["tr"] != Tm:
            out["trace_test"] = dict(ratio=(spm["tr"] - Tm) / (spb["tr"] - Tm), simpson=(spb["tr"] + 2 * spm["tr"]) / 3 / Tm)
        labels = {frozenset((r["yb"], r["ys"])) for r in rows}             # two-label set: one label pair for every pair
        out["angle"] = None
        if len(labels) == 1:
            a = min(next(iter(labels)))                                     # sign +1 where the base label is token a
            # u: the signed mean change Delta; b: the signed mean midpoint gradient
            c = torch.tensor([1.0 if r["yb"] == a else -1.0 for r in rows], device=self.dev)[:, None]
            u, b = (c * D).mean(0), (c * Wm).mean(0)
            cos = float(u @ b / (u.norm() * b.norm()))
            out["angle"] = dict(theta_deg=float(np.degrees(np.arccos(min(1.0, max(-1.0, cos))))),
                                top=0.5 * (1 + 1 / cos) if cos > 0 else None,
                                readout_share=float(b.norm() / Wm.norm(dim=1).mean()))
        if direction is not None:
            P, one = self._basis(direction)
            with torch.no_grad():                                           # the exact swap along P at the readout
                e = (z(Hb + ((D @ P)[:, None] * P if one else (D @ P) @ P.T)) - zb).cpu().numpy()
            pt = float(e.mean() / T.mean())
            out["direction"] = dict(effect=dict(value=pt, interval=list(stats.interval(pt, stats.boot_ratio(e, T, C), level))),
                                    first_order=float(self._first_order(Wm, D, P, one).mean()) / Tm)
        if per_pair:
            out["per_pair"] = dict(T=Tr, tr_mid=(Wm * D).sum(1).cpu().numpy(), tr_base=(Wb * D).sum(1).cpu().numpy(),
                                   effect=e if direction is not None else None)
        return out

    def cut(self, layer):
        """The complete-cut identity on the model at hand: swapping the whole state at every position of layer `layer`
        (a complete cut of the paper's kind (a)) gives the whole-input effect T. Runs on the pairs whose base and source are
        aligned (the same number of tokens and the same span ends) and leaves out the others. Returns a dict in log odds:
        pairs, the number of aligned pairs; own, the swap's mean effect on them; T, their mean T; max_diff, the largest
        per-pair |own - T|. With no aligned pair, only pairs=0."""
        block = self._block(layer)
        idx = [i for i, r in enumerate(self.rows) if r["aligned"]]
        if not idx:
            return dict(pairs=0)
        G, by_len = {}, {}
        for i in idx:                       # batches of one length: every position is swapped, with no padding
            by_len.setdefault(len(self.rows[i]["b"]), []).append(i)
        with torch.no_grad():
            for n_tok, ii in by_len.items():
                for k in range(0, len(ii), self.batch):
                    chunk = ii[k:k + self.batch]; rs = [self.rows[i] for i in chunk]
                    patches = [(torch.full((len(rs),), j, device=self.dev), self._swap(torch.stack([r["hs"][block, j] for r in rs])))
                               for j in range(n_tok)]
                    z0, z1 = self._logits(rs), self._logits(rs, block, patches)
                    for j, (i, r) in enumerate(zip(chunk, rs)):
                        G[i] = float((z0[j, r["yb"]] - z0[j, r["ys"]]) - (z1[j, r["yb"]] - z1[j, r["ys"]]))
        Gi, Ti = np.array([G[i] for i in idx]), self.Tpairs[idx]
        return dict(pairs=len(idx), own=float(Gi.mean()), T=float(Ti.mean()), max_diff=float(np.abs(Gi - Ti).max()))

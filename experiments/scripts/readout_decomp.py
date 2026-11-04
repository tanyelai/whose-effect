"""Where a direction's log-odds effect at the readout comes from (Section 5.4 and Appendix C.6, and Figure 11 on the
negative-polarity constructions): with the readout's saved directions, how much of each direction's effect comes from raising
the source label's logit and how much from lowering the base label's.

Per task and test pair, at the readout cell (the last block's raw output at the last token, as in experiments/scripts/readout.py), for
the directions das_ce_p1_s0..2, das_eff_p10_s0 and top_train of the readout's output for the same model and group
(experiments/results/readout/dirs/readout_<model>_m_<group>.npz): the patched state h_b + ((h_s - h_b).d) d, its label logits against
the base's, dz_s = z_s(patched) - z_s(base) and dz_b likewise, and the base state's probabilities p_s and p_b. The share that
comes from lowering the base label is -mean dz_b / mean (dz_s - dz_b). Rows whose labels are not single tokens are left out.
No training.
Output: experiments/results/readout_decomp/readout_decomp_<model>_<group>.{json,npz}.
Usage: python -I experiments/scripts/readout_decomp.py --model=<hf id> [--revision=<sha>] --data=<dir> --group=<group> --tasks=a,b
       [--device=cuda]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import common  # noqa: E402
import torch  # noqa: E402

from owneffect import families  # noqa: E402

_, args, model, tok = common.start("readout_decomp", sys.argv[1:])
DEV = common.RUN["dev"]
BLOCKS, FIX = families.blocks(model), families.start_of_text(tok)
READ = families.readout(model)
RAW = {}
BLOCKS[-1].register_forward_hook(lambda m, a, o: RAW.__setitem__("h", o[0] if isinstance(o, tuple) else o))
DIRS = ("das_ce_p1_s0", "das_ce_p1_s1", "das_ce_p1_s2", "das_eff_p10_s0", "top_train")
group, data_dir = args["group"], args["data"]
dirs_file = common.REPO / "experiments/results/readout/dirs" / f"readout_{common.RUN['tag']}_m_{group}.npz"
DZ = np.load(dirs_file)


def one(s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None


def last_states(seqs):
    out = [None] * len(seqs); groups = {}
    for i, q in enumerate(seqs):
        groups.setdefault(len(q), []).append(i)
    with torch.no_grad():
        for idx in groups.values():
            for k in range(0, len(idx), 64):
                chunk = idx[k:k + 64]
                model(torch.tensor([seqs[i] for i in chunk], device=DEV))
                for j, i in enumerate(chunk):
                    out[i] = RAW["h"][j, -1].clone()
    return torch.stack(out)


out, arrays = dict(model=common.RUN["model"], group=group, dirs_file=str(dirs_file.relative_to(common.REPO)), tasks={}), {}
for task in args["tasks"].split(","):
    rows = [r for r in common.rows(data_dir, task)["test"] if one(r["base_label"]) is not None and one(r["src_label"]) is not None]
    hb = last_states([families.encode(tok, r["base"], FIX)[0] for r in rows])
    hs = last_states([families.encode(tok, r["src"], FIX)[0] for r in rows])
    yb = torch.tensor([one(r["base_label"]) for r in rows], device=DEV); ys = torch.tensor([one(r["src_label"]) for r in rows], device=DEV)
    ar = torch.arange(len(rows), device=DEV)
    with torch.no_grad():
        zb = READ(hb); pb_ = torch.softmax(zb, -1)
        arrays[f"{task}/p_s"] = pb_[ar, ys].double().cpu().numpy(); arrays[f"{task}/p_b"] = pb_[ar, yb].double().cpu().numpy()
        row = dict(n=len(rows), base_labels=sorted({r["base_label"] for r in rows}), src_labels=sorted({r["src_label"] for r in rows}),
                   median_p_b=float(pb_[ar, yb].median()), median_p_s=float(pb_[ar, ys].median()), dirs={})
        for k in DIRS:
            key = f"{task}/{k}"
            if key not in DZ.files:
                continue
            d = torch.tensor(DZ[key], device=DEV, dtype=hb.dtype)
            z1 = READ(hb + ((hs - hb) @ d)[:, None] * d)
            dzs, dzb = z1[ar, ys] - zb[ar, ys], z1[ar, yb] - zb[ar, yb]
            arrays[f"{task}/dz_s/{k}"] = dzs.double().cpu().numpy(); arrays[f"{task}/dz_b/{k}"] = dzb.double().cpu().numpy()
            e = float((dzs - dzb).mean())
            row["dirs"][k] = dict(effect=e, mean_dz_s=float(dzs.mean()), mean_dz_b=float(dzb.mean()),
                                  share_lowering_b=float(-dzb.mean() / (dzs - dzb).mean()) if abs(e) > 1e-9 else None)
    out["tasks"][task] = row
    print(json.dumps({task: {k: round(v["share_lowering_b"], 3) if v["share_lowering_b"] is not None else None for k, v in row["dirs"].items()}}),
          f"p_b {row['median_p_b']:.3g}", flush=True)
res = common.REPO / "experiments/results/readout_decomp"; res.mkdir(parents=True, exist_ok=True)
stem = f"readout_decomp_{common.RUN['tag']}_{group}"
np.savez_compressed(res / f"{stem}.npz", **arrays); json.dump(out, open(res / f"{stem}.json", "w"), indent=1)
common.finish([res / f"{stem}.json", res / f"{stem}.npz"], summary=dict(tasks=len(out["tasks"])))

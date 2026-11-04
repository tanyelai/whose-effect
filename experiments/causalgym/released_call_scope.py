"""The scope of CausalGym's position fix, without a model (the paper's Appendix E.2): on every CausalGym task, with the
authors' own sampler and compute_pos('last') in Pythia's tokenizer (the training sample, seed 42, 100 batches of 4; the two
evaluation samples, seeds 1 and 420, 25 batches of 4, as in their data.py and das.py), for every (pair, region) from the first
variable region on, the index each call reads. compute_pos returns pos = [source positions, base positions]; the fixed call
reads the source at pos[0] and writes and collects the base at pos[1], the released call reads the source at pos[1] and
collects the base at pos[0] (released_call_unit_test.py checks this on a small model with random weights). Counts:
  (1) the fixed call's reads are always inside the real tokens (attention mask 1) of their own sentence, at the region's own
      last token, so it is the region-aligned interchange on every pair;
  (2) where pos[0] == pos[1] the two calls are identical (the fix changes nothing there);
  (3) how often the released call reads another token, and how often that token is padding;
  (4) null alignments (-1) appear on both sides together.
The paper's counts are on the line that starts with ALL: shifted cells, the released swap's and collect's reads past the end,
and pairs.
Usage: <the re-run environment's python> experiments/causalgym/released_call_scope.py <a CausalGym clone at its pinned commit>
(experiments/run/causalgym_rerun.sh builds both).
"""
import json
import os
import sys

CG = os.path.abspath(sys.argv[1]); sys.path.insert(0, CG); os.chdir(CG)
from transformers import AutoTokenizer
from data import Dataset

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m"); tok.pad_token = tok.eos_token
tasks = sorted(json.load(open("data/templates/syntaxgym.json")).keys())
tot = dict(cells=0, shifted=0, fixed_bad=0, rel_src_pad=0, rel_base_pad=0, null_mismatch=0, pairs=0, shifted_pairs=0)
rows = []
for t in tasks:
    ds = Dataset.load_from(f"syntaxgym/{t}")
    sets = {"train": ds.sample_batches(tok, 4, 100, "cpu", seed=42)}
    discard = {"".join(p.base) for b in sets["train"] for p in b.pairs}
    sets["eval1"] = ds.sample_batches(tok, 4, 25, "cpu", seed=1, discard=discard)
    sets["eval420"] = ds.sample_batches(tok, 4, 25, "cpu", seed=420, discard=discard)
    c = dict(cells=0, shifted=0, fixed_bad=0, rel_src_pad=0, rel_base_pad=0, null_mismatch=0, pairs=0, shifted_pairs=0)
    for batches in sets.values():
        for b in batches:
            src_pos, base_pos = b.compute_pos("last")
            mb, ms = b.base["attention_mask"], b.src["attention_mask"]
            for i in range(len(b.pairs)):
                c["pairs"] += 1; any_shift = False
                nb, ns = int(mb[i].sum()), int(ms[i].sum())          # real tokens (right padding)
                for j in range(ds.first_var_pos, ds.length):
                    ps, pb = src_pos[i][j][0], base_pos[i][j][0]
                    if (ps == -1) != (pb == -1):
                        c["null_mismatch"] += 1
                    if ps == -1:
                        continue
                    c["cells"] += 1
                    # the fixed call: source at ps, base at pb; both inside their own real tokens
                    if not (0 <= ps < ns and 0 <= pb < nb):
                        c["fixed_bad"] += 1
                    if ps != pb:
                        c["shifted"] += 1; any_shift = True
                        if pb >= ns: c["rel_src_pad"] += 1      # released swap reads the source past its end
                        if ps >= nb: c["rel_base_pad"] += 1     # released collect reads the base past its end
                c["shifted_pairs"] += any_shift
    rows.append((t, c)); [tot.__setitem__(k, tot[k] + v) for k, v in c.items()]
    print(f"{t:34s} pairs {c['pairs']:4d}  shifted pairs {c['shifted_pairs']:4d}  cells {c['cells']:5d}  shifted {c['shifted']:5d}  "
          f"fixed off-sentence {c['fixed_bad']}  released reads padding: source {c['rel_src_pad']}, base {c['rel_base_pad']}  "
          f"null mismatch {c['null_mismatch']}")
print("ALL", tot)
print("tasks with no shifted cell:", [t for t, c in rows if c["shifted"] == 0])

"""Number-only pairs for CausalGym's six number-agreement tasks, with lemma-only pairs as a control (the paper's Appendix B.1).

Source: CausalGym (Arora et al., 2024; MIT licence), the Hugging Face dataset aryaman/causalgym, read from the local
Hugging Face cache (HF_HUB_OFFLINE is 1 unless set otherwise). CausalGym's agreement pairs change the subject noun's
number and its lemma at once ("The senators that ..." -> "The minister that ..."); every other region is identical. Number
alone is the variable to isolate, so each row (train and test) gives two further pairs with the same base:
  num   the source keeps the base's lemma in the source's number ("The senators ..." -> "The senator ..."): number only;
  lex   the source keeps the base's number with the source's lemma ("The senators ..." -> "The ministers ..."): lemma only,
        a control.
Labels are the row's own pair (base_label, src_label), so all three sets measure the same log odds. The 38 subject nouns
pair into 19 lemmas, singular and plural (the code asserts this). Each CausalGym row comes with its mirror, so the num and
lex sets are balanced by class without a pairwise mirror of their own.
Output: experiments/tasks/agr_num/<task>__{num,lex}/{train,test}.json in CausalGym's format (the original pairs stay in
CausalGym itself). Prints, per task and split, the row count and how many original pairs already shared one lemma.
Usage: python -I experiments/tasks/make_agr_num.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import hf_hub_download  # noqa: E402

OUT = Path(__file__).resolve().parent / "agr_num"
TASKS = ["agr_sv_num_subj-relc", "agr_sv_num_obj-relc", "agr_sv_num_pp",
         "agr_refl_num_subj-relc", "agr_refl_num_obj-relc", "agr_refl_num_pp"]
CG_REV = "95349c3a5e53e2506e8b212482ea6dd784978156"     # CausalGym's pinned snapshot (experiments/lib/pins.py)
rows = {s: json.load(open(hf_hub_download("aryaman/causalgym", f"{s}.json", repo_type="dataset", revision=CG_REV))) for s in ("train", "test")}
nouns = sorted({r[k][1] for s in rows.values() for r in s if r["task"] in TASKS for k in ("base", "src")})
sing = {n + "s": n for n in nouns} | {n[:-1] + "ies": n for n in nouns if n.endswith("y")}     # actors, secretaries
plural = {n for n in nouns if n in sing}
lemma = {n: sing.get(n, n) for n in nouns}
assert len(nouns) == 38 and len(plural) == 19 and len(set(lemma.values())) == 19, (len(nouns), len(plural))
form = {(lemma[n], n in plural): n for n in nouns}
counts = {}
for task in TASKS:
    for kind in ("num", "lex"):
        (OUT / f"{task}__{kind}").mkdir(parents=True, exist_ok=True)
    for split, rs in rows.items():
        out = {"num": [], "lex": []}; same = 0
        for r in (r for r in rs if r["task"] == task):
            assert [i for i, (a, b) in enumerate(zip(r["base"], r["src"])) if a != b] in ([1], []) and len(r["base"]) == len(r["src"])
            nb, ns = r["base"][1], r["src"][1]
            assert (nb in plural) == (r["base_type"] == "plural") and (nb in plural) != (ns in plural)
            same += lemma[nb] == lemma[ns]
            for kind, noun in (("num", form[(lemma[nb], ns in plural)]), ("lex", form[(lemma[ns], nb in plural)])):
                src = list(r["base"]); src[1] = noun
                out[kind].append(dict(r, src=src, task=f"{task}__{kind}", pair_kind=kind))
        for kind, o in out.items():
            json.dump(o, open(OUT / f"{task}__{kind}" / f"{split}.json", "w"), indent=0)
        counts[f"{task} {split}"] = dict(rows=len(out["num"]), same_lemma_in_original=same)
print(json.dumps(counts, indent=1))

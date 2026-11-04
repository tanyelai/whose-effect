"""The movie template of Tigges, Hollinsworth et al. as CausalGym-format pairs, one set for every model (no per-model filter).

Source: prompts.yaml of github.com/curt-tigges/eliciting-latent-sentiment (MIT licence; the release pins commit 037cbc9 and
the file's sha256 in experiments/lib/pins.py). Their template "I thought this movie was{ADJ}, I{VRB} it. \\nConclusion: This movie is"
with their adjective and verb lists, after a start-of-text token; every adjective i is paired with every verb j (their own
prompts pair adjective i with verb i only, too few rows for one pass of DAS). Base = a positive prompt, source = the negative
prompt with the same indices (and the mirror); labels = the first positive and the first negative answer token (one label
pair). Training rows use their train adjectives, test rows their test adjectives. Their per-model one-token filter of the word
lists is not applied here (make_sentiment_templates.py applies it, per model); the readout drops a pair whose labels are not
single tokens for the model.
Output: <out dir>/sentiment/{train,test}.json (the release's files: experiments/tasks/sentiment/).
Usage: python -I experiments/tasks/make_sentiment_pairs.py <prompts.yaml> <out dir>
"""
import json
import sys
from pathlib import Path

import yaml

Y = yaml.safe_load(open(sys.argv[1])); OUT = Path(sys.argv[2]) / "sentiment"; OUT.mkdir(parents=True, exist_ok=True)
w = lambda k: [" " + str(x).strip() for x in Y[k]]  # noqa: E731
pos_ans, neg_ans = w("positive_answer_tokens")[0], w("negative_answer_tokens")[0]
pv, nv = w("positive_verbs"), w("negative_verbs")


def rows(P, N):
    out = []
    for i in range(min(len(P), len(N))):
        for j in range(min(len(pv), len(nv))):
            base = ["<|endoftext|>I thought this movie was", P[i], ", I", pv[j], " it. \nConclusion: This movie is"]
            src = ["<|endoftext|>I thought this movie was", N[i], ", I", nv[j], " it. \nConclusion: This movie is"]
            out.append(dict(base=base, src=src, base_type="positive", src_type="negative", base_label=pos_ans, src_label=neg_ans))
            out.append(dict(base=src, src=base, base_type="negative", src_type="positive", base_label=neg_ans, src_label=pos_ans))
    return out


json.dump(rows(w("positive_adjectives_train"), w("negative_adjectives_train")), open(OUT / "train.json", "w"), indent=0)
json.dump(rows(w("positive_adjectives_test"), w("negative_adjectives_test")), open(OUT / "test.json", "w"), indent=0)
print("labels:", pos_ans, neg_ans, "| train", len(json.load(open(OUT / "train.json"))), "test", len(json.load(open(OUT / "test.json"))))

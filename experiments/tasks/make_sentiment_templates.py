"""Tigges, Hollinsworth et al.'s sentiment templates as CausalGym-format pairs, per model (the paper's sentiment tasks).

Source: github.com/curt-tigges/eliciting-latent-sentiment, commit 037cbc9c18867e03da14ceace612ab7e0f9449f7 (MIT): the format
strings of utils/prompts.py (PromptType.get_format_string), and the word and answer lists that get_prompts() takes for each
template from prompts.yaml. Their list rule, followed here (PromptsConfig.get): a leading space, the words kept whose length in
the model's tokens is the template's filter length (1; 2 for the adverbs), duplicates removed. Labels: the first word of the
template's positive and negative answer lists after that rule (CLASSIFICATION: " Positive" / " Negative"); one answer pair,
since the readout reads one question, where their score averages over the pairs.
Pairs (ours; the reason: their 13-30 prompts per template are too few for one pass of DAS, and our test trains and tests within
a task):
  movie       their movie template with their own train / test adjective lists and every verb (the pairing of
              experiments/tasks/make_sentiment_pairs.py, with their filter);
  two slots   book (VRB, ADJ), res (NOUN, VRB), product (FEEL, ADJ): base from the positive lists, source from the negative
              lists, every pair (i, j); training rows have i and j even, test rows i and j odd, mixed pairs dropped;
  one slot    adverb (ADV), proper (NOUN), medical (MED): every positive word i against every negative word k, split the same
              way;
  indexed     completion, classification, res_class: prompt i fills ADJ1-ADJ3 with core adjectives i, i+1, i+2, VRB with verb i
              and ADJ4 with top adjective i (indices wrap, as their CircularList), inside each half of the core adjectives and
              verbs (training: even indices, test: odd), every positive prompt against every negative one of the same half.
              Of templates that differ only in punctuation, quotes or one word (COMPLETION/COMPLETION_2,
              CLASSIFICATION/_2/_3/_4), the first in their enum order. MULTI_SUBJECT_1 is left out (its variable is which review
              is asked about). CLASSIFICATION is kept as published and flagged: its negative ADJ4 list in this prompts.yaml
              holds positive verbs ("adore", ...), so every negative prompt ends "very adore." (their list, in their runs too).
Then 200 training and 50 test pairs with seed 94 (all if fewer), each with its mirror (base and source exchanged).
Their own use of the templates, also written: sentiment_<template>_transfer/ trains on the movie template's training rows and
tests on all of the template's rows (a direction fitted on the movie template, evaluated on the others).
Output: experiments/data/<model tag>/sentiment_templates/<task>/{train,test}.json and build.json (list sizes after the filter, rows and
labels per task; a template whose positive or negative answer list keeps no word after the filter is reported, not written).
experiments/run/data.sh runs it for every model.
Usage: python -I experiments/tasks/make_sentiment_templates.py <their prompts.yaml> --models=<hf id>,... [--tokenizer_dirs=...]
"""
import json
import random
import re
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

Y = yaml.safe_load(open(sys.argv[1])); REPO = Path(__file__).resolve().parents[2]
args = dict(a.lstrip("-").split("=", 1) for a in sys.argv[2:] if "=" in a)
MODELS = args["models"].split(","); DIRS = args["tokenizer_dirs"].split(",") if "tokenizer_dirs" in args else MODELS
E = "<|endoftext|>"
SLOT = {  # spans with the slots as None, slot lists in text order (key, filter length), answer lists
    "book": ([E + "The novel was", None, " by the book club members. The comments were universally", None,
              " \nConclusion: You will be very"], [("verbs", 1), ("comment_adjectives", 1)], "answer_adjectives"),
    "res": ([E + "This restaurant is a", None, ", you are going to", None, " it. \nConclusion: This restaurant is very"],
            [("nouns", 1), ("infinitives", 1)], "very_answers"),
    "product": ([E + "I am", None, " about this purchase, the product is", None, ". \nConclusion: I feel very"],
                [("feelings", 1), ("core_adjectives", 1)], "moods"),
    "adverb": ([E + "The traveller", None, " walked to their destination. The traveller felt very"], [("adverbs", 2)], "moods"),
    "proper": ([E + "When I hear the name", None, ", I feel very"], [("proper_nouns", 1)], "answer_tokens"),
    "medical": ([E + "I heard the doctor use the word", None, " and I felt very"], [("medical", 1)], "answer_tokens"),
}
INDEXED = {
    "completion": ("I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and overall the movie was "
                   "just very", "answer_tokens"),
    "classification": ("Review Text: 'I thought this movie was{ADJ1}, I{VRB} it. The acting was{ADJ2}, the plot was{ADJ3}, and "
                       "overall the movie was just very {ADJ4}.' \nReview Sentiment:", None),
    "res_class": ("This restaurant was{ADJ1}, I{VRB} it. The food was{ADJ2}, and the service was{ADJ3}. Overall it was just",
                  "answer_tokens"),
}
MOVIE = [E + "I thought this movie was", None, ", I", None, " it. \nConclusion: This movie is"]


def fill(spans, words):
    it = iter(words)
    return [next(it) if s is None else s for s in spans]


def cross(P, N):
    pairs = {"train": [], "test": []}
    for i in range(len(P)):
        for k in range(len(N)):
            if i % 2 == k % 2:
                pairs["train" if i % 2 == 0 else "test"].append((P[i], N[k]))
    return pairs


def build(tok):
    def w(key, length=1):
        out = []
        for x in Y[key]:
            s = " " + str(x).strip()
            if len(tok(s, add_special_tokens=False)["input_ids"]) == length and s not in out:
                out.append(s)
        return out

    def answers(key):
        if key is None:
            return " Positive", " Negative"
        p, n = w(f"positive_{key}"), w(f"negative_{key}")
        return (p[0], n[0]) if p and n else (None, None)

    jobs, sizes = {}, {}
    pv, nv = w("positive_verbs"), w("negative_verbs")
    pa, na = w("positive_adjectives_train"), w("negative_adjectives_train")
    ta, tn = w("positive_adjectives_test"), w("negative_adjectives_test")
    jobs["movie"] = ({"train": [(fill(MOVIE, (pa[i], pv[j])), fill(MOVIE, (na[i], nv[j])))
                                for i in range(min(len(pa), len(na))) for j in range(min(len(pv), len(nv)))],
                      "test": [(fill(MOVIE, (ta[i], pv[j])), fill(MOVIE, (tn[i], nv[j])))
                               for i in range(min(len(ta), len(tn))) for j in range(min(len(pv), len(nv)))]},
                     answers("answer_tokens"))
    sizes["movie"] = dict(adjectives_train=(len(pa), len(na)), adjectives_test=(len(ta), len(tn)), verbs=(len(pv), len(nv)))
    for name, (spans, slots, ans) in SLOT.items():
        lists = [(w(f"positive_{k}", L), w(f"negative_{k}", L)) for k, L in slots]
        sizes[name] = [(len(p), len(n)) for p, n in lists]
        if len(lists) == 2:
            (p1, n1), (p2, n2) = lists
            pairs = {"train": [], "test": []}
            for i in range(min(len(p1), len(n1))):
                for j in range(min(len(p2), len(n2))):
                    if i % 2 == j % 2:
                        pairs["train" if i % 2 == 0 else "test"].append((fill(spans, (p1[i], p2[j])), fill(spans, (n1[i], n2[j]))))
        else:
            (p1, n1), = lists
            pairs = cross([fill(spans, (x,)) for x in p1], [fill(spans, (x,)) for x in n1])
        jobs[name] = (pairs, answers(ans))
    for name, (fmt, ans) in INDEXED.items():
        pairs = {}
        for split, half in (("train", 0), ("test", 1)):
            def prompt(sign, i):
                adj, vrb, top = w(f"{sign}_core_adjectives")[half::2], w(f"{sign}_verbs")[half::2], w(f"{sign}_top_adjectives")
                words = dict(ADJ1=adj[i % len(adj)], ADJ2=adj[(i + 1) % len(adj)], ADJ3=adj[(i + 2) % len(adj)],
                             VRB=vrb[i % len(vrb)], ADJ4=top[i % len(top)] if top else "")
                parts = re.split(r"\{(ADJ1|ADJ2|ADJ3|ADJ4|VRB)\}", fmt)
                return [x for x in [E + parts[0]] + [words[x] if k % 2 == 0 else x for k, x in enumerate(parts[1:])] if x != ""]
            n = min(len(w("positive_core_adjectives")[half::2]), len(w("negative_core_adjectives")[half::2]))
            if not (w("positive_verbs")[half::2] and w("negative_verbs")[half::2]):
                n = 0                                                     # no verb of this half passes the filter
            pairs[split] = [(prompt("positive", i), prompt("negative", k)) for i in range(n) for k in range(n)]
        jobs[name] = (pairs, answers(ans))
    return jobs, sizes


def rows(pairs, pos, neg, task):
    out = []
    for k, (b, s) in enumerate(pairs):
        out.append(dict(base=b, src=s, base_type="positive", src_type="negative", base_label=pos, src_label=neg, task=task, pair=k))
        out.append(dict(base=s, src=b, base_type="negative", src_type="positive", base_label=neg, src_label=pos, task=task, pair=k))
    return out


for model, tdir in zip(MODELS, DIRS):
    tok = AutoTokenizer.from_pretrained(tdir)
    out = REPO / "experiments/data" / model.rstrip("/").split("/")[-1] / "sentiment_templates"
    jobs, sizes = build(tok); built = {}
    written = {}
    for name, (pairs, (pos, neg)) in jobs.items():
        task = f"sentiment_{name}"
        if pos is None:
            built[task] = "no answer word passes the filter"; continue
        rng = random.Random(94); written[task] = {}
        for split, n in (("train", 200), ("test", 50)):
            ps = pairs[split] if len(pairs[split]) <= n else rng.sample(pairs[split], n)
            written[task][split] = rows(ps, pos, neg, task)
        built[task] = dict(train=len(written[task]["train"]), test=len(written[task]["test"]), labels=(pos, neg))
    for task, sp in written.items():
        d = out / task; d.mkdir(parents=True, exist_ok=True)
        for split, rr in sp.items():
            json.dump(rr, open(d / f"{split}.json", "w"), indent=0)
        if task != "sentiment_movie":
            dt = out / f"{task}_transfer"; dt.mkdir(parents=True, exist_ok=True)
            json.dump([dict(r, task=f"{task}_transfer") for r in written["sentiment_movie"]["train"]], open(dt / "train.json", "w"), indent=0)
            json.dump([dict(r, task=f"{task}_transfer") for r in sp["train"] + sp["test"]], open(dt / "test.json", "w"), indent=0)
    json.dump(dict(source="curt-tigges/eliciting-latent-sentiment 037cbc9", model=model, list_sizes=sizes, tasks=built),
              open(out / "build.json", "w"), indent=1)
    print(model, {k: (v["train"], v["test"]) if isinstance(v, dict) else v for k, v in built.items()})

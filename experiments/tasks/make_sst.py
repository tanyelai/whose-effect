"""Stanford Sentiment Treebank sentences in the two scaffolds of Tigges, Hollinsworth et al. (the paper's SST tasks).

The SST copy in their repository (curt-tigges/eliciting-latent-sentiment, MIT licence, commit
037cbc9c18867e03da14ceace612ab7e0f9449f7, folder stanfordSentimentTreebank; Socher et al., 2013), read as data:
sentence-level labels as their utils/treebank.py get_merged_dataframe builds them (a sentence's phrase id from
dictionary_fixed.txt, its value from sentiment_labels.txt rounded to 0 or 1 with Python's round, the first of duplicated
phrases and sentences kept), and SST's own split (datasetSplit.txt: 1 train, 2 test, 3 dev; dev is not used).
Their scaffolds (utils/treebank.py apply_scaffold_to_prompts, construct_answer_tokens):
  sst_classification   "Review Text: {sentence} Review Sentiment:"     labels " Positive" / " Negative"
  sst_continuation     "{sentence} Overall the movie was just very"    labels " good" / " bad"
Base: a positive sentence, source: a negative sentence of the same split, drawn with seed 94 (no sentence twice in a split;
the same sentences in both scaffolds); 200 training and 50 test pairs, each with its mirror. These are not minimal pairs:
every word may differ. Their pairing by token count and their logit-difference filter are not used: the readout reads only
the last position, and no task here is filtered on the model.
Output: <out>/sst_{classification,continuation}/{train,test}.json (the release's files: <out> = experiments/tasks/sst).
Usage: python -I experiments/tasks/make_sst.py <their stanfordSentimentTreebank directory> <out dir>
"""
import json
import random
import sys
from pathlib import Path

SST, OUT = Path(sys.argv[1]), Path(sys.argv[2])
phrase_id = {}
for line in open(SST / "dictionary_fixed.txt", encoding="utf-8"):
    ph, pid = line.rstrip("\n").rsplit("|", 1)
    phrase_id.setdefault(ph, int(pid))                                # the first of duplicated phrases
value = {}
for k, line in enumerate(open(SST / "sentiment_labels.txt", encoding="utf-8")):
    if k == 0:
        continue                                                      # header "phrase ids|sentiment values"
    pid, v = line.strip().split("|")
    value[int(pid)] = int(round(float(v)))
sentences = {}
for k, line in enumerate(open(SST / "datasetSentences_fixed.txt", encoding="utf-8")):
    if k == 0:
        continue                                                      # header "sentence_index\tsentence"
    idx, s = line.rstrip("\n").split("\t", 1)
    sentences[int(idx)] = s
split = {}
for k, line in enumerate(open(SST / "datasetSplit.txt", encoding="utf-8")):
    if k == 0:
        continue                                                      # header "sentence_index,splitset_label"
    idx, lab = line.strip().split(",")
    split[int(idx)] = {"1": "train", "2": "test", "3": "dev"}[lab]
pool, seen = {"train": {0: [], 1: []}, "test": {0: [], 1: []}}, set()
for idx in sorted(sentences):                                         # their drop_duplicates('sentence', keep='first'),
    s = sentences[idx]                                                # over all three splits, before the split is read
    if s in seen:
        continue
    seen.add(s)
    if s not in phrase_id or split.get(idx) not in pool:
        continue
    pool[split[idx]][value[phrase_id[s]]].append(s)
print({sp: {lab: len(v) for lab, v in d.items()} for sp, d in pool.items()})
SCAF = {"classification": (lambda s: ["<|endoftext|>Review Text:", " " + s, " Review Sentiment:"], " Positive", " Negative"),
        "continuation": (lambda s: ["<|endoftext|>", s, " Overall the movie was just very"], " good", " bad")}
for name, (spans, pos, neg) in SCAF.items():
    rng = random.Random(94); d = OUT / f"sst_{name}"; d.mkdir(parents=True, exist_ok=True)
    for sp, n in (("train", 200), ("test", 50)):
        P, N = rng.sample(pool[sp][1], n), rng.sample(pool[sp][0], n)
        rows = []
        for k, (p, q) in enumerate(zip(P, N)):
            b, x = spans(p), spans(q)
            rows.append(dict(base=b, src=x, base_type="positive", src_type="negative", base_label=pos, src_label=neg,
                             task=f"sst_{name}", pair=k))
            rows.append(dict(base=x, src=b, base_type="negative", src_type="positive", base_label=neg, src_label=pos,
                             task=f"sst_{name}", pair=k))
        json.dump(rows, open(d / f"{sp}.json", "w"), indent=0)
    print(name, "written")

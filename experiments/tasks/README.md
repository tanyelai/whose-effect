# Task data

The task data that do not depend on the model, with the scripts that build them. Data that do depend on it, such as the
entities a model knows or the words that are one token for it, are built for every model by `experiments/run/data.sh` into
`experiments/data/<model>/`, and `experiments/run/check_data.py` compares each file with the hash of the paper's
(`experiments/matrix/data.sha256`).

We use every source the way its authors did. Where the method needs a change, the change, its reason and the script that
makes it are here; the paper's Appendix B.1 points to this page for each one. Each change is marked **R** when the method
requires it, with the reason, or **C** when it is our choice, in which case the authors' version is often built as well.

What the method needs from a task:

- minimal pairs, a base and a source input that differ in one variable;
- a base label and a source label of one token each, since the readout reads the log odds of one over the other;
- for Section 5's two-label tasks, the same two labels in every pair;
- training and test rows from the same task;
- for the maps inside the network, regions that line up between base and source.

Each source below gives where its data come from (link, pinned revision, checksum where there is one, licence), what the
files hold, the authors' protocol as their code runs it, what our build does, and whether the paper's numbers can be compared
with their published ones.

---

## CausalGym (Arora et al., 2024)

| | |
|---|---|
| data | `aryaman/causalgym` (Hugging Face), revision 95349c3 (`experiments/lib/pins.py`) |
| code | `github.com/aryamanarora/causalgym`, commit 0f3129f |
| licence | MIT (the Hugging Face dataset; the code repository states none) |
| files | `train.json`, `test.json`: 29 tasks; each row has `base`, `src` (lists of spans), `base_label`, `src_label`, `base_type`, `src_type`, `task` |

**Their protocol** (`data.py`, `das.py`, `train.py`):
- their train and test splits (train seed 42, test seed 1);
- every pair also stored with base and source exchanged (mirrored);
- the label is the first token of the label string;
- DAS with a linear learning-rate schedule;
- the metric is the log odds ratio.

**The build:** the 29 tasks are read as released (the readout and the maps). `make_agr_num.py` re-pairs the six
number-agreement tasks into number-only pairs (`agr_num/<task>__num/`), with lemma-only pairs as a control
(`agr_num/<task>__lex/`).

| difference | class | in the release |
|---|---|---|
| a row whose label is more than one token is dropped | R: a one-token label | the readout drops such rows; for CausalGym none is dropped in any model (the `dropped` count in `experiments/results/readout/readout_<model>_m_causalgym.json`, written by `experiments/scripts/readout.py`) |
| a constant learning rate instead of their schedule | C | stated in the paper's Appendix B.3 |
| number-only agreement (`agr_num/`) re-pairs their six number-agreement tasks | R: it isolates number from the lemma | built by `make_agr_num.py` |
| `agr_num/` pairs are not mirrored pair by pair | R: each is built from one CausalGym row, and every row of theirs comes with its mirror, so the set is balanced by class (50/50 in every test file), which is what the class-signed mean difference uses; a pairwise mirror would double the rows and DAS's steps | stated |

**Comparable with their results: yes.**
- Section 7.1 reads their published tables
  (`experiments/causalgym/causalgym_published_tables.json`, parsed by `experiments/causalgym/parse_published_tables.py`).
- Their code was re-run as released and with only the position call fixed (Appendix E.2; `experiments/run/causalgym_rerun.sh`,
  `experiments/causalgym/rerun/position_fix.patch`, scored by `experiments/score/score_causalgym_rerun.py`).
- `experiments/causalgym/` also holds three checks of the released call (Appendix E.2), which need a clone of their
  repository and their environment with pyvene (`experiments/causalgym/rerun/requirements_frozen.txt`): `index_shift_exposure.py`
  (its output is `index_shift_exposure.json`), `released_call_scope.py` (the appendix's counts of shifted cells and of reads
  past the end of a sentence) and `released_call_unit_test.py`.
- `experiments/causalgym/causalgym_fig4_darkness.json` holds the cell shades read from one of their published figures
  (npi_any_subj-relc; its `source` field names the figure), which `experiments/score/score_causalgym_map.py` compares with the
  release's DAS map.

---

## Kumon and Yanaka (2026), filler–gap and NPI constructions

| | |
|---|---|
| data | `kumoryo9/shared-mech` (Hugging Face), revision d57aeab (`experiments/lib/pins.py`) |
| code | `github.com/ynklab/shared_syntactic_mechanism`, commit 0a12355 |
| paper | arXiv 2604.22166 |
| licence | MIT (the Hugging Face dataset; the code repository states none) |
| files | one folder per construction (8 filler–gap, 8 NPI, 1 control; `make_kumon.py` leaves out FGNaturalEmbed and the control); `train.json`, `test_id.json` (in distribution), `test.json` (out of distribution; `*_label_ood` fields); rows hold `base`, `src` (lists of spans), labels, types and a `mapping` of roles to regions |

**Their protocol** (`data/load_data.py`, `activation.py`, `config.py`):
- `collect_dataset` stores every row together with base and source exchanged;
- for the test files it takes the first half of the rows, then mirrors them;
- a label of several tokens is read by its first token;
- for NPI they keep only pairs whose base has the NPI and whose source does not (`--src_base_pattern label2`), and score with
  a modified odds ratio;
- DAS is leave-one-out over constructions;
- they report results averaged over the in-distribution and out-of-distribution test sets;
- models: Pythia 1B (main), 2.8B and 6.9B; Gemma 3 1B and 12B.

**The build** (`make_kumon.py`, per model, into `experiments/data/<model>/kumon/`) follows their loader and their NPI runs: every
training row and the first half of each test file, each with its mirror; a label read by its first token; for NPI, only the
pairs in their direction (every base label " ever").

| difference | class | in the release |
|---|---|---|
| NPI pairs also in both directions (`<construction>_mirrored`) | R: the method's mirrored set | built beside their direction, labelled as ours |
| the in-distribution test file is the main test | R: DAS is evaluated in its training distribution | the out-of-distribution file is a held-out test (`_ood`, and `_ood_label` with its `*_label_ood` labels) |
| a two-label log odds instead of their NPI odds | R: the readout reads one label against another | stated |
| DAS trained per construction, not leave-one-out | R: the question is per task | stated |

Excluded by rule: `FGNaturalEmbed` (10 test rows, fewer than 20) and `CtrlCapital` (no training split).

Properties of their data that are kept and reported:
- **Shared inputs across their split.** Some NPI constructions share inputs between training and test: 38 of NPISimple's
  50 test rows, 14 of NPIDeterminermono's and 15 of NPIOnlyScope's share a string with training, and some pairs repeat with
  another label. `<construction>_disjoint` (ours, a held-out test) keeps only test rows whose strings occur in no training
  row; NPISimple then has 12 test rows, and `NPISimple_disjoint` is not in the `kumon_disjoint` group (`experiments/matrix/tasks.json`).
- **Labels.** Every NPI pair contrasts " ever" with one of five other words (" never", " still", " often", " already",
  " also"), so no construction asks a single question.
- **Pairs that differ beyond the licensor.** In NPIEmb the matrix verb also differs (" might wonder whether" / " can know
  that"); in NPISimple the subject's capital letter and space differ (" the pedestrians" / "The pedestrians").
- **Duplicates.** A few rows repeat in the training files (NPIOnlyScope 12, NPIDeterminermono 9, NPISimple 3, FGMatrixQ 2,
  NPIQuantifiers 1, as built for GPT-2).

**Comparable with their results: no.** The release does not run their code. Their main model, Pythia-1B, is one of the
paper's models.

---

## RAVEL (Huang et al., 2024)

| | |
|---|---|
| data and code | `github.com/explanare/ravel`, commit e421fc9; `data.tgz` sha256 b148f0ded1979cb0e17e301cfda41cb120517a109d3c758cc768e376946d96bc |
| licence | MIT |
| files | per entity type (city, Nobel prize winner, occupation, physical object, verb): `*_entity_attributes.json`, `*_entity_to_split.json`, `*_attribute_to_prompts.json`, `*_prompt_to_split.json`, Wikipedia prompts |

**Their protocol** (`src/utils/generate_ravel_instance.py`, `metric_utils.py`, their paper's Appendix A):
- entities and prompts are split;
- a label is the model's own continuation of the prompt (`prompt_to_output`);
- an entity is kept when the model predicts its attribute correctly;
- interchange sources are other prompts or Wikipedia text;
- the scores are Cause, Iso and their mean (disentanglement);
- the raw prompt is fed with no start-of-text token.

**The build:** two-label tasks, two values per attribute.
- `make_ravel_attributes.py` lists the candidates of every entity type and attribute (`ravel/attributes/candidates.json`):
  prompts grouped by the form of the next word, the two most frequent one-token values, and a random sample of up to 60
  entities per value and split (seed 94; RAVEL samples entities at random). `experiments/scripts/filter_known_attributes.py` keeps,
  per model, the (entity, prompt) combinations the model knows by RAVEL's criterion (its most likely next token is the label)
  and draws the pairs without replacement (`experiments/data/<model>/ravel_attributes/`, and `ravel_attributes_sot/` with a
  start-of-text token; `<task>_ood`, with test rows from RAVEL's test prompts, is a held-out test).
- `make_ravel_continent.py` lists the European and Asian cities for the continent task (`ravel/ravel_continent/cities.json`);
  `experiments/scripts/filter_known.py` keeps, per model, the cities the model places on their continent and builds the pairs
  (`experiments/data/<model>/ravel_continent/`).

| difference | class | in the release |
|---|---|---|
| two values per attribute (a binary contrast) | R: one question | stated |
| the label is the value's next token; for the Gender prompts that end in "for", " his" / " her", the word those prompts elicit; the Gender prompts that end in "gender:" form their own task (`*_gender_value`) with the value as label | R: a one-token label | built |
| the JSON prompts form tasks of their own (`*_json`), with labels without a leading space | R: one label pair per task | built |
| base and source share one prompt | R: a minimal pair with aligned regions | stated |
| the continent task keeps a city when the model prefers its continent by at least one log odds, not by RAVEL's criterion (which the attribute tasks use) | C | stated |
| a start-of-text token before the prompt | C | both built: RAVEL's raw prompt (`ravel_attributes/`) and ours (`ravel_attributes_sot/`) |
| RAVEL's validation entities used for training | C | stated |

**Comparable with their results: no.** Their scores are Cause, Iso and Disentangle, on Llama-2-7B. The release does not run
their code.

---

## Tigges, Hollinsworth et al. (2023, 2024), sentiment templates

| | |
|---|---|
| code and data | `github.com/curt-tigges/eliciting-latent-sentiment`, commit 037cbc9 (`prompts.yaml`, `utils/prompts.py`, `utils/treebank.py`); `prompts.yaml` sha256 592b699a2cfa0d3a5f489bd6167e58a0d27239cd77e7252eb4bfc2f297bd2d7a (`experiments/lib/pins.py`) |
| licence | MIT |
| files | `prompts.yaml`: word and answer lists; templates in `utils/prompts.py` (`PromptType`) |

**Their protocol:**
- every word list is filtered per model to one-token words (two tokens for adverbs);
- duplicates are removed;
- prompt i pairs adjective i with verb i, and the corrupted prompt is the next prompt;
- the movie template has train and test adjective lists; the other templates are used to evaluate a direction fitted on the
  movie template;
- the score is the logit difference averaged over the answer pairs.

**The build:**
- `make_sentiment_templates.py`: the English templates (of those that differ only in punctuation, quotes or one word, the
  first; MULTI_SUBJECT_1 left out), per model with their filter (`experiments/data/<model>/sentiment_templates/`,
  the movie template as `sentiment_movie`), plus the authors' own transfer use (`*_transfer`);
- `make_sentiment_pairs.py`: the movie template without their per-model filter, one set for every model
  (`experiments/tasks/sentiment/`);
- the re-run of their published score: `experiments/scripts/sentiment_reread.py` (the paper's Section 7.3).

| difference | class | in the release |
|---|---|---|
| `experiments/tasks/sentiment/` has no per-model one-token word filter (624–768 of its 864 training rows contain a word their filter would drop, depending on the model) | C | the per-model templates apply their filter (`sentiment_movie` there) |
| every adjective × verb pair instead of i with i | R: their ~26 training prompts are too few for one pass of DAS | stated |
| one answer pair per template (the first after their filter: " great" / " terrible" on the movie template) instead of the mean over their pairs | R: one question | stated |
| the other templates are also split into training and test rows (by the parity of the word indices) | R: the test trains and tests within a task | their transfer use is also built |
| CLASSIFICATION kept as published, though its negative prompts end in a positive verb (their `negative_top_adjectives` list) | as published | flagged here |
| `experiments/scripts/sentiment_reread.py` | follows their protocol: their one-token filter, the mean over their five answer pairs, a start-of-text token, the next prompt as the corrupted one | — |

Properties of their data that are kept and reported:
- **Label words inside the prompts.** In completion and res_class, a label word is also a filler in 23–27% of rows ("…the
  plot was terrible, and overall the movie was just very" → " terrible").
- **Verbs shared across splits.** Verbs are shared between the movie template's training and test rows, as in their split.
- **Shared words in CLASSIFICATION.** Its ADJ4 words come from the top-adjective lists, which are not split, so they appear
  in both splits (the positive list is the one word "good").
- **Few distinct prompts.** The indexed templates have few distinct prompts (14 bases in 98 training rows for GPT-2).
- **The transfer test.** It trains on the movie template's labels and tests on each template's own labels, which is their
  use: a direction fitted on one template and read on another.
- **A task that drops.** After their filter, sentiment_book has fewer than 20 test rows in five of the six small models
  (GPT-2 keeps 20). It is therefore not in the `sentiment_templates` group, whose tasks are the same in every model
  (`experiments/matrix/tasks.json`); it is in `sentiment_templates_model_specific` for the models where it keeps at least 20.

**Comparable with their results: yes, for the published sentiment score.** Section 7.3 re-runs it with their code path.

## Stanford Sentiment Treebank (Socher et al., 2013), via Tigges et al.

The SST copy in the same repository (MIT licence). `make_sst.py` builds `sst/`: sentence labels as their
`get_merged_dataframe` builds them, SST's own split, and their two scaffolds. Duplicate sentences are removed over all three
splits before the split is read, as their `drop_duplicates` does.

| difference | class | in the release |
|---|---|---|
| random positive–negative pairing, the same rows in every model, instead of pairing by token count per model | C | stated |
| 200 / 50 pairs sampled | C (cost) | stated |
| SST pairs are not minimal pairs (the two sentences may differ in every word) | — | stated in the paper |

Their logit-difference filter applies only to their 1.4B classification runs, so not filtering is their protocol for the
paper's models.

---

## IOI

**Ours** (`make_ioi_pairs.py`, `ioi/`, the paper's "IOI, ours"): one sentence frame (4 openers x 10 places x 10 objects). The
names, places and objects are our own lists, not Wang et al.'s, as the paper's Appendix B.1 says.

**Wang et al. (2023)** (`github.com/redwoodresearch/Easy-Transformer`, commit ea15315, `easy_transformer/ioi_dataset.py`,
sha256 94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231; MIT licence): 15 BABA templates and their ABBA
forms, their names, places and objects; no start-of-text token (their comment: "GPT-2 small wasn't trained this way").
`make_ioi_wang.py` builds `ioi_wang/`; its `*_ood` tasks, a held-out test, train on BABA templates 1–10 and their ABBA forms
and test on templates 11–15 and theirs.

| difference | class | in the release |
|---|---|---|
| a start-of-text token before the prompt | C | their convention is the default; `*_sot` adds the token (ours) |
| all 30 templates rather than their mixes | C, minor | stated |
| only names that are one token in every tokenizer given | R: a one-token label | stated |
| with two names (`ioi_wang_1`), five of the 50 test pairs are training frames with the names swapped | unavoidable with two names | stated |

**Comparable with their results: no.** Section 9.2 quotes Makelov et al.'s 140.7% and 123.6% (their Table 1); the release
does not re-run them.

---

## Built folders

| folder | built by | from |
|---|---|---|
| `agr_num/` | `make_agr_num.py` | CausalGym |
| `ioi/` | `make_ioi_pairs.py` | ours |
| `ioi_wang/` | `make_ioi_wang.py` | Wang et al. |
| `ravel/ravel_continent/`, `ravel/attributes/` | `make_ravel_continent.py`, `make_ravel_attributes.py` (then the per-model filters) | RAVEL |
| `sentiment/` | `make_sentiment_pairs.py` | Tigges et al. |
| `sst/` | `make_sst.py` | SST via Tigges et al. |

Per model, in `experiments/data/<model>/` (tokenizer-dependent, built by `experiments/run/data.sh`): `kumon/` (`make_kumon.py`),
`sentiment_templates/` (`make_sentiment_templates.py`), `ravel_continent/` (`experiments/scripts/filter_known.py`),
`ravel_attributes/` and `ravel_attributes_sot/` (`experiments/scripts/filter_known_attributes.py`), and `ravel_attributes_regions/`
(`experiments/scripts/ravel_regions.py`: the RAVEL tasks whose prompts come in two forms, in one region layout for their maps).

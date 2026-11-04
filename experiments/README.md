# Experiments

Everything behind the paper's results.

| folder | what it holds |
|---|---|
| `run/` | one script per experiment, issuing exactly the paper's runs; `all.sh` runs them all in order |
| `scripts/` | the experiments: the readout, the maps inside the network, the re-runs of published methods |
| `score/` | the scorers, from the runs' outputs to the files the paper reads; `score_all.sh` runs them in order |
| `matrix/` | the readout units, each model's task lists, the sets of maps, and the hashes of the paper's task data |
| `lib/` | shared code: the measurement's arithmetic, pinned versions, provenance, tolerances |
| `tests/` | checks of the scorers and the arithmetic |
| `tasks/` | the task data that do not depend on a model, with their builders and sources ([data card](tasks/)) |
| `causalgym/` | CausalGym's published tables, three checks of its released code, and `rerun/` to run that code again |

## Reproducing the paper

Linux, an NVIDIA GPU and Python 3.12 (Llama and Gemma need a Hugging Face token). From the repository's root:

```bash
bash experiments/run/setup.sh                               # .venv with the paper's pinned software
PY=.venv/bin/python python -I experiments/run/download.py   # every model and dataset, at its pinned revision
PY=.venv/bin/python bash experiments/run/all.sh             # task data, runs, scores and the paper's numbers
```

`all.sh` resumes where it stopped. For the 7B-32B models, `run/with_model.sh` keeps one model on a scratch disk at a time.

From saved outputs (`results/`, `data/`, `logs/`) to the paper:

```bash
bash experiments/score/score_all.sh && bash paper/build.sh
```

Figures need macOS (Avenir Next, Iowan Old Style) and Chrome; elsewhere use `FIGS=0`. Scorers accept only outputs made by
the code in this tree (`lib/provenance.py`).

## Where each result comes from

| the paper | runs | scorers | written into the paper by |
|---|---|---|---|
| Section 5, at the readout | `run/readout.sh`, `run/decomp.sh` | `score_readout*.py`, `score_adequacy.py`, `score_scale.py`, `cap_mechanism.py` | `numbers_main.py`, `fig_readout.py`, `fig_law.py`, `fig_trace.py`, `fig_npi_decomp.py`, `table_readout.py`, `table_scale.py` |
| Section 6, inside the network | `run/inside.sh` | `score_inside_set.py`, `inside_set_post.py`, `inside_forms_post.py`, `inside_excess_rank.py` | `numbers_main.py`, `fig_guide_b.py`, `fig_guide_d.py`, `fig_guide_f.py` |
| Section 7, benchmark scores | `run/causalgym_maps.sh`, `run/sentiment.sh`, `run/causalgym_rerun.sh` | `score_causalgym_map.py`, `score_sentiment_reread.py`, `score_causalgym_rerun.py` | `numbers_cgmap.py`, `fig_cgmap.py`, `fig_benchmarks.py`, `table_causalgym.py`, `table_sentiment.py` |
| the running example | `run/inside.sh`, `run/readout.sh` | the maps themselves | `fig_guide_a.py` to `fig_guide_f.py` |

## Tests

```bash
python -I experiments/tests/<name>_test.py
```

Sixteen run on synthetic data; `ce_law_scorer_test.py` and `instrument_tol_test.py` need your own outputs.

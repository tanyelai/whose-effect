<div align="center">

# Whose Effect Is It?

**How Objectives Inflate Learned Subspace Interventions**

Toygar Tanyel · Serkan Ayvaz

[![arXiv](https://img.shields.io/badge/arXiv-2611.05033-b31b1b.svg)](https://arxiv.org/abs/2611.05033)
[![OwnEffect](https://img.shields.io/badge/library-OwnEffect-ef6c00.svg)](owneffect/)
[![Proofs](https://img.shields.io/badge/proofs-Lean%204-7e57c2.svg)](lean/)
[![Experiments](https://img.shields.io/badge/experiments-reproduce-00897b.svg)](experiments/)
[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-2e7d32.svg)](LICENSE)

</div>

Learned interventions such as DAS often move a model's answer further than copying the whole state does. We show why, and
how to read such an effect beside the network's own.

![Figure 1 of the paper](paper/figures/fig_law.png)

## OwnEffect

```bash
pip install -e ".[stats]"
```

```python
from owneffect import OwnEffect

oe = OwnEffect(model, tokenizer)
oe.prepare(pairs)
rep = oe.report(layer=6, region=1, direction=d)
rep["readings"]["reading"]    # e.g. "excess": beyond the network's own interchange
```

## Repository

- [`owneffect/`](owneffect/): the library
- [`experiments/`](experiments/): experiments, scorers, task data, and how to reproduce the paper
- [`lean/`](lean/): the paper's theorems, machine-checked
- [`paper/`](paper/): the paper's figures, and the scripts that write its numbers, tables and figures from the results

## Citation

```bibtex
@misc{tanyel2026whose,
  title  = {Whose Effect Is It? How Objectives Inflate Learned Subspace Interventions},
  author = {Tanyel, Toygar and Ayvaz, Serkan},
  year   = {2026},
  eprint = {2611.05033},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi    = {10.48550/arXiv.2611.05033},
  url    = {https://arxiv.org/abs/2611.05033}
}
```

Apache-2.0. The task data keep their sources' licences.

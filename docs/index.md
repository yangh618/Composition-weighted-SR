# Composition-weighted Symbolic Regression (CWSR)

CWSR discovers compact, interpretable analytical expressions that map
**chemical composition → a target property** using a Monte-Carlo-Tree-Search
driven symbolic regression engine. The `cwsr` package is **task-agnostic**: the
same code runs on Matbench tasks, alloy databases, or your own composition
dataset.

## Quick start

```bash
conda env create -f environment.yml     # creates env `cwsr`
conda activate cwsr
pip install -e .                        # installs cwsr package + CLIs
```

```python
from cwsr.datasets import get_dataset
from cwsr.model import fit_dataset

ds = get_dataset("alloy_density", data_dir="examples/alloys/data")
outputs = fit_dataset(ds, output_dir="results/density",
                      var_count=3, ops=["mul", "add", "sub", "R"],
                      max_expressions=300, num_parallel=4, seed=42)
print(outputs[0]["expression"])          # best discovered expression
```

## Documentation

- **[Tutorial](tutorial.md)** — step-by-step, runnable examples.
- **[API Reference](reference.md)** — full manual for every public API.
- **[Design & roadmap](design.md)** — architecture and Alloys-SR merge plan.

## Model form

Each sample is a 118-dim atomic-fraction composition vector `comp`; a model is
`y = f(W @ comp)`, where `W` is a `(var_count, 118)` weight matrix found by the
search and `f` is a symbolic expression in the `var_count` latent variables.

## Contents

- `cwsr/` — engine (`regressor`, `mcts`, `gp`, `exp_tree`, `exp_queue`,
  `reward`, `checkpoint`) and framework (`datasets`, `model`, `predict`,
  `formula`).
- `tests/` — short-time smoke tests (alloy + Matbench).
- `examples/` — local examples (alloy database + data are bundled locally;
  see `examples/alloys/`).

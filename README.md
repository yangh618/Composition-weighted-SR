# Composition-weighted Symbolic Regression (CWSR)

CWSR discovers compact, interpretable analytical expressions that map
**chemical composition → a target property** using a Monte-Carlo-Tree-Search
(MCTS) driven symbolic-regression engine. It is **task-agnostic**: the same
code runs on Matbench benchmarks, alloy databases, or your own composition
datasets.

> The engine in this package was originally built on iMCTS but has been heavily
> reworked and flattened into `cwsr`; all imports use `cwsr.*`.

## Model form

Each sample is a 118-dim atomic-fraction composition vector `comp`. A learned
model is:

```
y = f(W @ comp)
```

- `comp` : (118,) composition (atomic fractions, ordered by atomic number)
- `W`    : (var_count, 118) tabulated per-element weights found by the search
- `f`    : a symbolic expression in `var_count` latent variables `x0 … x_{v-1}`

## Key features

- **MCTS-driven symbolic regression** engine (`cwsr.Regressor`, `cwsr.mcts`, …)
- **Task-agnostic datasets**: Matbench, alloy `.npz` databases, or any custom
  composition dataset via `cwsr.datasets.get_dataset(...)`
- Element-safe train/valid splits (every element in the data stays in training)
- Forward evaluation + **analytic gradients**, and formula→value query
- Unified training drivers, CLI entry points, and smoke tests (on `develop`)

## Documentation

- **[Tutorial](docs/tutorial.md)** — step-by-step examples
- **[API Reference](docs/reference.md)** — full manual for every public API
- **[Design & roadmap](docs/design.md)** — architecture and merge plan

## Install

Requires Python ≥ 3.10. In a conda environment:

```bash
conda env create -f environment.yml   # creates env `cwsr`
conda activate cwsr
pip install -e .
```

Or with plain pip (after installing the numerical deps):

```bash
python -m pip install -r requirements.txt
pip install -e .
```

## Quick start

### High-level (recommended)

Load any dataset by name through the registry, then fit:

```python
from cwsr.datasets import get_dataset
from cwsr.model import fit_dataset

ds = get_dataset("matbench_expt_gap", fold=0)   # or "alloy_density", or a .npz path

outputs = fit_dataset(
    ds,
    output_dir="results",
    var_count=3,                                  # latent variables
    ops=["mul", "add", "sub", "R"],               # allowed operators
    max_expressions=2000, max_depth=6,
    num_parallel=8, seed=0,
)

best = outputs[0]
print(best["expression"])                          # e.g. '0.31*x0 + 1.04*x2'
```

### Low-level engine

```python
import numpy as np
from cwsr import Regressor

# X: (n_samples, 118) compositions, y: (n_samples,) targets
model = Regressor(
    x_train=X, y_train=y,
    var_count=3, ops=["mul", "sub", "add", "div", "sqrt", "exp", "log", "R"],
    max_expressions=5000, max_depth=6, num_parallel=8,
)
simplified, raw, n_evaluations, path, outputs = model.fit(seed=0)
print(f"Best expression: {outputs[0]['expression']}")
```

See the [Tutorial](docs/tutorial.md) for loading custom `.npz` data, using the
forward/query/gradient helpers, and adding your own datasets.

## Repository layout

```
cwsr/            engine (regressor, mcts, gp, exp_tree, exp_queue, reward,
                 checkpoint) + framework (datasets, model, predict, formula)
docs/            tutorial.md, reference.md, design.md (+ MkDocs site config)
mkdocs.yml       documentation site configuration
dataloader.py    Matbench data loader (legacy top-level)
eval.py, run_cwsr.py   Matbench CLI/eval runners (legacy top-level)
examples/        local examples + alloy databases (not committed)
tests/           smoke + sanity tests (maintained on the develop branch)
```

## Supported & extensible datasets

- Matbench tasks (e.g. `matbench_expt_gap`, `matbench_glass`, …)
- Alloy databases (`alloy_density`, `alloy_hardness`, …) from `.npz` files
- Any `.npz` with `targets`, `formulas`, `target_name`, `source` keys, or a
  custom provider registered via `cwsr.datasets.register_provider`

## Versioning

Current release: **v0.1.0** (package `version` and `cwsr.__version__` aligned).

## References

- *Composition-Weighted Symbolic Regression for General-Purpose Property
  Prediction* — the method implemented by this package. (Add full citation —
  authors / journal / year — when available.)

## License

[MIT](LICENSE)

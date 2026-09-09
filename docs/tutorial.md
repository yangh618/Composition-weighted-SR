# CWSR — User Tutorial

Composition-weighted Symbolic Regression (CWSR) discovers compact, interpretable
analytical expressions that map **chemical composition → a target property**,
using a Monte-Carlo-Tree-Search (MCTS) driven search. Everything lives in the
`cwsr` package and is **task-agnostic**: the same code runs on Matbench tasks,
alloy databases, or your own composition dataset.

This tutorial assumes Python ≥ 3.10 inside the `cwsr` conda environment.

---

## 1. Concepts (read this first)

Every sample is a **composition vector** of length 118, one column per element
(atomic fractions, ordered by atomic number H..Og, summing to 1). The learned
model has the form

```
y = f(W @ comp)
```

- `comp` : (118,) composition vector
- `W`    : (var_count, 118) tabulated per-element weights found by the search
- `f`    : a symbolic expression in `var_count` latent variables `x0 ... x_{v-1}`,
           e.g. `x0 + 3.78*x1 - x2 + 1.04`

Training returns a ranked list of candidate expressions with their optimized
`W` weights and train/valid metrics (MAE, reward).

Key modules:

| Module | Purpose |
|---|---|
| `cwsr.Regressor` | The search engine (fit an expression to data) |
| `cwsr.datasets` | Task-agnostic dataset loading + splits |
| `cwsr.model` | High-level `fit_dataset` / `train` drivers |
| `cwsr.predict` | Forward evaluation, analytic gradients, formula query |
| `cwsr.formula` | formula ↔ composition helpers |

```python
import cwsr
from cwsr import Regressor, simplify_expression
from cwsr.datasets import get_dataset, list_datasets, CompositionDataset
print(list_datasets()[:5])   # e.g. ['alloy_density', 'alloy_ductility', ..., 'matbench_glass', ...]
```

---

## 2. Loading data

All data — regardless of source — is exposed as a
[`CompositionDataset`](cwsr/datasets/base.py) with `.compositions` (n,118),
`.targets` (n,), `.formulas`, `.target_name`, `.source` and `.meta`.

### (a) Bundled alloy databases

The alloy databases ship inside the example
(`examples/alloys/data/*.npz`), so no download is needed:

```python
from cwsr.datasets import get_dataset

ds = get_dataset("alloy_density", data_dir="examples/alloys/data")
print(ds)                      # density: 470 samples
print(ds.target_name)          # 'Density (g/cm^3)'
print(ds.compositions.shape)   # (470, 118)
print(ds.compositions[0].sum())# 1.0  (atomic fractions)
print(ds.formulas[:3])         # ['Al0.1CoCrFeNi', 'Al0.25CoFeNi', ...]
```

### (b) Matbench tasks

```python
# First call downloads/caches the task (and asks before downloading).
ds = get_dataset("matbench_glass", fold=0)     # small task, good for testing
# Larger examples: "matbench_dielectric", "matbench_expt_gap", ...
```

### (c) Your own .npz dataset

Any file with keys `targets`, `formulas`, `target_name`, `source` loads the
same way — just pass the path:

```python
ds = get_dataset("my_data.npz")
```

### Train/valid splitting

The framework always keeps **every chemical element present in the data in the
training split** (needed because CWSR learns per-element weights):

```python
from cwsr.datasets.base import split_dataset
train_idx, valid_idx = split_dataset(ds.compositions, ds.targets,
                                     ratio=0.8, seed=42)
```

---

## 3. Training

### (a) High-level driver (recommended)

`cwsr.model.fit_dataset` splits, fits, and writes `cwsr_outputs_<name>_<ts>.json`
under an output directory, returning the ranked outputs list.

```python
from cwsr.datasets import get_dataset
from cwsr.model import fit_dataset

ds = get_dataset("alloy_density", data_dir="examples/alloys/data")

outputs = fit_dataset(
    ds,
    output_dir="results/density",
    var_count=3,          # latent variables (keep small)
    ops=["mul", "add", "sub", "R"],   # allow multiplication, +, -, constants
    max_depth=5,
    max_expressions=300,  # search budget (larger -> better, slower)
    num_parallel=4,
    num_batches=16,
    K=10,
    seed=42,
)
```

### (b) Low-level engine API

For full control use `cwsr.Regressor` directly. `fit` returns
`(simplified_expr, raw_expr, n_evaluations, path, outputs)`; the best trained
model is `outputs[0]`.

```python
import numpy as np
from cwsr import Regressor
from cwsr.datasets.base import split_dataset

X, y = ds.compositions, ds.targets
tr, va = split_dataset(X, y, ratio=0.8, seed=42)

model = Regressor(
    x_train=X[tr], y_train=y[tr],
    x_valid=X[va], y_valid=y[va],
    var_count=3,
    ops=["mul", "add", "sub", "R"],
    max_depth=5, max_expressions=300,
    num_parallel=4, num_batches=16, seed=42,
)
simplified, raw, n_evals, path, outputs = model.fit(seed=42)

best = outputs[0]
print("best expression :", best["expression"])   # e.g. '0.31*x0 + 1.04*x2'
print("train MAE       :", best["mae"])
print("valid MAE       :", best["mae_valid"])
W = np.asarray(best["weights"], dtype=float)      # (var_count, 118)
print("weights shape   :", W.shape)
```

### (c) Understanding the output JSON

Each element of `outputs` is a dict:

```json
{
  "expression":    "0.31*x0 + 1.04*x2",
  "weights":       [[0.12, ...118 cols...], ...],
  "train_reward":  0.61,
  "valid_reward":  0.62,
  "mae":           0.96,
  "mae_valid":     0.93,
  "rank":          1
}
```

These files are what the query / eval / bootstrap tooling consume.

---

## 4. Using a trained model (prediction)

### (a) In Python — vectorized forward pass

```python
import numpy as np
from cwsr.predict import compile_expression, predict_vector

best = outputs[0]
f = compile_expression(best["expression"], var_count=3)
W = np.asarray(best["weights"], dtype=float)

yhat = predict_vector(ds.compositions, W, f)      # (n,) predictions
```

### (b) Formula → value (query)

`query` predicts a property for one chemical formula from a **refined-results
JSON file** (a list whose first element has `expression` and `weights`).
`outputs` from training already has exactly that shape, so save the best as a
refined-results file and query:

```python
import json
json.dump(outputs, open("results/density/refined_results_density.json", "w"))

from cwsr.predict.query import predict as query_predict
print(query_predict("results/density/refined_results_density.json", "Al0.25CoCrFeNi"))
```

or from the command line:

```bash
python -m cwsr.predict.query \
    --results results/density/refined_results_density.json \
    "Al0.25CoCrFeNi"
```

### (c) Analytic gradients (for design/optimisation)

```python
from cwsr.predict import compile_expression, compile_gradient_functions, predict_and_gradient

grads = compile_gradient_functions(best["expression"], var_count=3)
val, grad = predict_and_gradient(ds.compositions[0], W, f, grads)
# grad: (118,) derivative of the prediction w.r.t. each element's fraction
```

---

## 5. Adding a new composition dataset

Define a provider that returns a `CompositionDataset` and register it. Then it
is usable by every downstream tool with `get_dataset("<name>")`.

```python
from cwsr.datasets.base import CompositionDataset
from cwsr.datasets import register_provider, get_dataset

def my_provider(task, **kw):
    # ... build X (n,118) and y (n,) from whatever source you like ...
    return CompositionDataset(
        name="my_" + task,
        compositions=X, targets=y,
        target_name="my property", source="my source",
    )

register_provider("my_", my_provider)      # name-prefix style, or full names
```

A `.npz` with the standard keys works out of the box (see §2c).

---

## 6. Quick sanity / short runs

Use the included short smoke tests to confirm the whole stack works fast:

```bash
conda activate cwsr
./tests/run_short.sh                          # alloy + matbench, ~30 s
python tests/smoke_alloy.py    --property density --max-samples 150 --max-expressions 80
python tests/smoke_matbench.py --task matbench_glass --max-samples 150 --max-expressions 80
```

---

## 7. Practical tips

- **`var_count`** is the number of latent variables; 3–4 is a good default.
  Larger values make expressions/weights heavier.
- **`max_expressions`** is the search budget — the single biggest
  runtime/quality knob. Use ~200–300 for exploration, thousands for final runs.
- **`ops`** controls allowed operators. Minimal sets (`mul add sub R`) are much
  faster than transcendental ones (`exp log sqrt tanh ...`).
- Restrict `ops` / lower `max_depth` when you only need a rough model; expand
  for accuracy.
- The verbose training log prints a live "best expression / MAE" report; look
  at `outputs[0]` for the final champion with its optimized `W`.

## 8. Roadmap

Inverse design, Pareto-front optimisation, bootstrap UQ and the plotting suite
are implemented in the reference `Alloys-SR` project and are being ported into
`cwsr` (`cwsr/design`, `cwsr/validate`, `cwsr/plotting`) per `DESIGN.md`; the
public dataset/train/predict API above is stable and already covers the core
end-to-end workflow.


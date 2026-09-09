# CWSR — API Reference

Complete reference for the public Python APIs of the `cwsr` package: dataset
layer, training, prediction, formula helpers, the CLI, and the search-engine
classes. Pair with `TUTORIAL.md` (worked examples) and `DESIGN.md`
(architecture/roadmap).

Assumes Python ≥ 3.10 with the project installed (`pip install -e .`) and the
`cwsr` environment active.

---

## 0. Package layout & import map

The engine was originally built on iMCTS and has been flattened into `cwsr`;
all imports now use `cwsr.*`.

| Old (iMCTS) | Now |
|---|---|
| `iMCTS.Regressor` | `cwsr.Regressor` |
| `iMCTS.regressor` | `cwsr.regressor` |
| `iMCTS.mcts` (MCTS, MCTS_Node) | `cwsr.mcts` |
| `iMCTS.gp` (GPManager) | `cwsr.gp` |
| `iMCTS.src.exp_tree` (ExpTree) | `cwsr.exp_tree` |
| `iMCTS.src.utils.exp_queue` (Exp_Queue) | `cwsr.exp_queue` |
| `iMCTS.src.utils.reward` (Optimizer, sp_module) | `cwsr.reward` |
| `iMCTS.checkpoint` | `cwsr.checkpoint` |

Framework subpackages: `cwsr.formula`, `cwsr.datasets` (`base`, `matbench`,
`alloy`, `registry`), `cwsr.model`, `cwsr.predict` (`forward`, `query`).

---

## 1. Top-level package: `cwsr`

```python
import cwsr
from cwsr import Regressor, simplify_expression, datasets
```

Exports
- `cwsr.Regressor` — search engine (see §6).
- `cwsr.simplify_expression(exp_str, verbose=False) -> str` — clean/simplify an
  expression string via sympy.
- `cwsr.datasets` — dataset subpackage (§3).
- `cwsr.__version__` — `"0.1.0"`.

---

## 2. `cwsr.formula` — composition ↔ formula

All vectors are length-118 atomic fractions (column `Z-1` ↔ element of atomic
number `Z`; `H`..`Og`), summing to 1.

- `ELEMENT_SYMBOLS : list[str]` — 118 element symbols, index = zero-based Z.
- `form2comp(formula: str) -> np.ndarray` — parse a chemical formula
  (fractional stoichiometry OK, via pymatgen) into a normalized (118,)
  vector. Raises on unparseable formulas.
- `composition_to_formula(composition, threshold=1e-6) -> str` — vector → a
  formula string like `"Al0.25CoCrFeNi"` (fractions ≈1 written without index).
- `format_composition(composition, top_n=20, threshold=1e-6) -> str` —
  human-readable top-`top_n` summary like `"Fe0.30 Ni0.25 Cr0.20 …"`.

```python
from cwsr.formula import form2comp, composition_to_formula, format_composition
v = form2comp("FeCrCoNi")                # (118,), sums to 1 (equimolar -> each 0.25)
composition_to_formula(v)                # 'Cr0.25Fe0.25Co0.25Ni0.25' (atomic order)
format_composition(v, top_n=4)           # 'Ni0.25 Cr0.25 Co0.25 Fe0.25'
```

> Note: CWSR works on **normalized** atomic fractions (sum = 1). A formula like
> `"Al0.25CoCrFeNi"` has 5 atoms summing to 4.25, so its vector is normalized to
> `Al≈0.059 Cr≈0.235 Co≈0.235 Fe≈0.235 Ni≈0.235` on load. Use equimolar
> formulas (e.g. `"FeCrCoNi"`) if you want 0.25 round-trips.

---

## 3. `cwsr.datasets` — task-agnostic data layer

### 3.1 `base`

`CompositionDataset` (dataclass) is the single schema every tool consumes.

```python
from cwsr.datasets.base import CompositionDataset
ds = CompositionDataset(
    name="alloy_density",
    compositions=X,          # (n, 118) float
    targets=y,               # (n,)
    formulas=["Al0.1CoCrFeNi", ...],   # optional
    target_name="Density (g/cm^3)",    # optional
    source="...",            # optional
    meta={},                 # optional free-form dict
)
```

Fields / read-only properties
- `.compositions` (n,118), `.targets` (n,), `.formulas` (list|None),
  `.target_name`, `.source`, `.meta`
- `.n_samples`, `.n_features` (==118), `len(ds)`

Raises `ValueError` if compositions are not 2-D or length-mismatched.

Split helpers
- `split_dataset(compositions, targets, ratio=0.8, seed=None)
  -> (train_idx, valid_idx)` — deterministic random split that forces any
  *singleton* element (a column present in only one sample) into training.
- `ensure_train_covers_species(compositions, train_indices, valid_indices)
  -> (train_idx, valid_idx)` — the internal guarantee; rarely called directly.

### 3.2 Providers + registry (`matbench`, `alloy`, `registry`)

- `list_datasets() -> list[str]` — names currently resolvable.
- `get_dataset(name, *, data_dir="processed_data", **kwargs)
  -> CompositionDataset`
  - `name` a registered Matbench task (e.g. `"matbench_glass"`) — optionally
    `fold=0`.
  - `name` `"alloy_<property>"` (density, hardness, yield_strength,
    elongation, melting_temperature, ductility, youngs_modulus) — resolved
    from `data_dir` (`<data_dir>/<property>.npz`).
  - `name` a path to a `.npz` with keys `targets`, `formulas`,
    `target_name`, `source`.
- `register_provider(name, provider)` — register `provider(task_or_name,
  **kwargs) -> CompositionDataset`.

```python
from cwsr.datasets import get_dataset, list_datasets

print(list_datasets())                                   # includes 'alloy_density', 'matbench_glass', ...
ds_a = get_dataset("alloy_density", data_dir="examples/alloys/data")
ds_m = get_dataset("matbench_glass", fold=0)             # downloads on first use (asks first)
ds_c = get_dataset("my_data.npz")                        # custom npz
```

Constants: `cwsr.datasets.matbench.MATBENCH_TASKS` (list of task keys);
`cwsr.datasets.alloy.DEFAULT_PROPERTIES` (property → npz filename).


---

## 4. `cwsr.model` — training drivers

- `train(compositions, targets, output_prefix, split_ratio=0.8, seed=None,
  **regressor_kwargs) -> list[dict]`
  Splits (element-safe), fits a `Regressor`, writes `<output_prefix>.json`
  (the ranked outputs list) and returns it.
- `fit_dataset(dataset, output_dir=".", split_ratio=0.8, seed=None,
  **regressor_kwargs) -> list[dict]`
  Same on a `CompositionDataset`; writes
  `output_dir/cwsr_outputs_<name>_<timestamp>.json`.

`**regressor_kwargs` are forwarded to `Regressor` (see §6): always pass
`var_count`, and consider `ops`, `max_depth`, `max_expressions`,
`num_parallel`, `num_batches`, `K`.

```python
from cwsr.datasets import get_dataset
from cwsr.model import fit_dataset

ds = get_dataset("alloy_density", data_dir="examples/alloys/data")
outputs = fit_dataset(ds, output_dir="results/density",
                      var_count=3, ops=["mul", "add", "sub", "R"],
                      max_depth=5, max_expressions=300,
                      num_parallel=4, num_batches=16, seed=42)
best = outputs[0]          # ranked by valid reward/MAE
```

---

## 5. `cwsr.predict` — evaluation, gradients, query

### 5.1 `forward`

The forward model is `y = f(W @ comp)`. All helpers are expression/weights
based (dataset-agnostic). `W` is `(var_count, 118)`; `f` is a compiled
callable mapping `(n, var_count) -> (n,)`.

- `compile_expression(expression: str, var_count: int) -> Callable` — sympy
  lambdify + numba JIT. `var_count >= 1`.
- `compile_gradient_functions(expression, var_count) -> list[Callable]` —
  one compiled `df/dx_i` per variable.
- `jit_compile(func) -> Callable` — wrap an arbitrary per-row function into
  the `(n, var_count)->(n,)` numba form.
- `predict(composition(118,), weights(var_count,118), f) -> float`
- `predict_vector(compositions(n,118), weights, f) -> np.ndarray(n,)`
- `predict_and_gradient(composition, weights, f, grad_funcs)
  -> (value, gradient(118,))`
- `predict_vector_and_gradients(compositions, weights, f, grad_funcs)
  -> (values, gradients(n,118))`
- `_get_active_indices(elements) -> np.ndarray` — 1-based atomic numbers →
  0-based columns.
- `_build_composition(comp_flat, active_indices) -> (118,)` — renormalize free
  params onto active columns.
- re-exports `ELEMENT_SYMBOLS`, `composition_to_formula`, `format_composition`.

> Import note: a subset is re-exported from `cwsr.predict` (`compile_expression`,
> `compile_gradient_functions`, `jit_compile`, `predict`, `predict_vector`,
> `predict_and_gradient`, `ELEMENT_SYMBOLS`, `composition_to_formula`,
> `format_composition`). The full list above (incl.
> `predict_vector_and_gradients`, `_get_active_indices`, `_build_composition`)
> is available from `cwsr.predict.forward`.

```python
import numpy as np
from cwsr.predict.forward import (compile_expression, predict_vector,
                                  compile_gradient_functions, predict_and_gradient)

best = outputs[0]
W = np.asarray(best["weights"], dtype=float)
f = compile_expression(best["expression"], var_count=3)
yhat = predict_vector(ds.compositions, W, f)
grads = compile_gradient_functions(best["expression"], var_count=3)
val, grad = predict_and_gradient(ds.compositions[0], W, f, grads)
```

### 5.2 `query`

Predicts a property for one chemical formula from a refined-results JSON
(a list whose first element has `expression` + `weights`).

- `load_refined(results_path) -> dict` — returns `data[0]`.
- `predict(results_path, formula) -> float`
- `main(argv=None) -> int` — CLI. Run via
  `python -m cwsr.predict.query --results <file> [formula]` or the
  `cwsr-query` console script. Omitting `formula` enters an interactive loop.

```python
import json
json.dump(outputs, open("results/refined_results.json", "w"))

from cwsr.predict.query import predict as q
q("results/refined_results.json", "Al0.25CoCrFeNi")
```


---

## 6. Engine

### 6.1 `cwsr.Regressor` (main entry point)

`simplify_expression(exp_str, verbose=False) -> str`

```python
from cwsr import Regressor
model = Regressor(
    x_train=X[tr],      # (n,118) compositions
    y_train=y[tr],      # (n,)
    x_valid=X[va],      # optional
    y_valid=y[va],      # optional
    # --- latent/search ---
    var_count=None,          # latent variables; defaults to x_train.shape[1] (118) if None
    ops=None,                # operator list; vars x0.. added automatically
    max_depth=6,
    K=500,                   # MCTS exploration constant
    c=4.0, gamma=0.5,
    gp_rate=0.2,             # genetic-programming rate
    mutation_rate=0.1,
    exploration_rate=0.2,
    max_constants=10,
    max_expressions=2e6,     # search budget
    # --- optimisation ---
    optimization_method="LN_NELDERMEAD",   # nlopt algorithm name
    lbfgs_upper_bound=47.0,
    num_trials=1,
    # --- parallelism ---
    num_parallel=4,
    num_batches=16,
    # --- misc ---
    arity_dict=None, context=None, complexity=None,
    max_single_arity_ops=999,
    verbose=False,
    reward_func=None,
    seed=None,
    save_every=0, save_checkpoint_every=0, output_prefix=None,
)
```

Parameters
- `x_train` / `y_train` / `x_valid` / `y_valid` — row-wise data, shape
  `(n, 118)` / `(n,)`. Same row count in X and y per set.
- `var_count` — number of latent variables `x0..x_{var_count-1}`. **Set it
  explicitly** (3–4 typical); leaving `None` uses 118.
- `ops` — allowed operators among `add sub mul div sqrt sin cos tanh exp log
  Max Min Pow R` (variables are appended automatically). `"R"` introduces
  optimizable constants. Default is a broad set.
- `max_depth` / `max_constants` — expression tree depth cap and max constants.
- `max_expressions` — total expression evaluation budget (the dominant
  runtime/quality control).
- `K`, `c`, `gamma` — MCTS exploration parameters.
- `gp_rate`, `mutation_rate`, `exploration_rate` — genetic-programming mix.
- `num_parallel`, `num_batches`, `num_trials` — parallel processes per MCTS
  batch, batch count, and NLopt restarts.
- `optimization_method`, `lbfgs_upper_bound` — nlopt algorithm and box bound.
- `seed`, `verbose`, `reward_func`, `output_prefix`, `save_every`,
  `save_checkpoint_every`.

Methods
- `fit(seed=None, checkpoint=None)
  -> (simplified_expr, raw_expr, n_evaluations, path, outputs)` where
  `outputs` is a `list[dict]` ranked by valid MAE (best = `outputs[0]`).
- `build_MAE_loss(expr_str, X, Y) -> Callable` — objective over a parameter
  vector.
- `optimize_weights(best_expr, is_positive_init) -> (mae, mae_valid, W)` —
  NLopt-optimize weights for a fixed expression.
- `save_status(mcts) -> list[dict]` — the top expression records with weights
  and metrics.

Output record schema (each dict in the returned/JSON `outputs`):

| key | meaning |
|---|---|
| `expression` | expression string in `x0..` |
| `weights` | optimized `(var_count, 118)` weight matrix (list-of-lists) |
| `train_reward` / `valid_reward` | reward `1/(1 + MAE/σ)` |
| `mae` / `mae_valid` | mean absolute error on train / valid |
| `rank` | 1-based rank |

### 6.2 `cwsr.mcts`

- `MCTS(optimizer, gp_manager, gp_rate=0.2, mutation_rate=0.2,
  exploration_rate=0.2, K=500, c=4, gamma=0.5, verbose=False,
  succ_error_tol=1e-6, num_parallel=4, num_batches=16, num_trials=1,
  seed=None)` — the search driver. Notable attributes: `.count_num` (# eval),
  `.exp_queue`, `.path_queue`, `.root`. Method `search(exp_tree)` runs one
  MCTS iteration.
- `MCTS_Node(mcts=...)` — tree node; methods `expand(state)`, `choose()`,
  `random_child()`, `backpropagate(path, value)`, `propagate(path, value)`,
  `is_leaf()`, `ucb()`. Module-level task helpers `_rollout_task`,
  `_optimize_task`, `_evalpath_task`, `_mutation_task`, `_crossover_task`
  (used for multiprocessing; normally not called directly).

### 6.3 `cwsr.gp.GPManager`

`GPManager(ops, arity_dict, verbose=True)` — genetic-programming operators over
expression-tree paths. Methods: `mutate(state, path)`, `generate(state)`,
`crossover(state, path1, path2)`, `node_replace`, `shrink_mutate`,
`uniform_mutate`, `insert_mutate`, plus path geometry helpers
(`cal_subtree_size_at_index`, `cal_subtree_depth_at_index`,
`cal_depth_at_index`). Used internally by the search.


### 6.4 `cwsr.exp_tree`

- `ExpTreeBase(max_depth, max_single_arity_ops, max_constants, arity_dict,
  complexity, ops)` — stateful builder for one expression (a stack of
  operators). Methods: `add_op(op)`, `is_empty()`, `is_full()`,
  `is_terminal()`, `random_fill()`, `get_expression()`, `clear()`.
- `ExpTree(...)` — subclass enforcing validity checks on `add_op`/terminal.

### 6.5 `cwsr.exp_queue`

- `Queue_Base(max_size)` — bounded priority-ish queue of `(state, reward)`.
- `Exp_Queue(max_size)` — queue used to hold top expressions; `append(state,
  train_reward, valid_reward=0.0, threshold=1e-5)`, `.best()`,
  `.best_reward()`, `.random_sample()`, `len()`.

### 6.6 `cwsr.reward`

- `Optimizer(var_count, x_train, y_train, x_valid=None, y_valid=None,
  sigma=None, context=None, reward_func=None, optimization_method=
  "LN_NELDERMEAD", lbfgs_upper_bound=47.0)` — constant/weight optimization +
  reward (`reward = 1/(1 + MAE/σ)`). Methods: `optimize_constants(state,
  is_positive_init=False) -> (expr, reward, reward_valid)`, `run_nlopt(...)`,
  `compile_expression(...)`, `valid_expression(...)`, `init_parameters(state)`,
  `init_positive_parameters(state)`.
- `sp_module` — the lambdify module set (min/max/heaviside/pow + numpy) used
  to compile expressions (imported by `cwsr.predict.forward`).
- `Heaviside_vec(x)` — vectorized Heaviside used in expressions.

### 6.7 `cwsr.checkpoint`

- `save_checkpoint(filepath, mcts, regressor_state=None)` — serialize an MCTS
  run to JSON.
- `load_checkpoint(filepath) -> dict`
- `mcts_to_dict(mcts) -> dict` / `mcts_from_dict(data, mcts_instance)` —
  lower-level (de)serialization.
- `_node_to_dict`, `_flatten_tree`, `_rebuild_tree` — internals.

```python
from cwsr.checkpoint import load_checkpoint, save_checkpoint
save_checkpoint("run_ckpt.json", mcts)
state = load_checkpoint("run_ckpt.json")     # {'mcts': {...}, ...}
```

---

## 7. Console entry points

| Command | Module |
|---|---|
| `cwsr` | `run_cwsr` (Matbench CLI runner) |
| `cwsr-eval` | `eval` |
| `cwsr-query` | `cwsr.predict.query` |

Example:

```bash
cwsr-query --results results/refined_results.json "Al0.25CoCrFeNi"
```

---

## 8. Conventions & notes

- **Shapes**: compositions `(n, 118)`; weights `(var_count, 118)`;
  expression functions accept `(n, var_count) -> (n,)`.
- **Columns** are indexed by atomic number minus 1.
- The element-safe split (all elements present in the data appear in the
  training set) is mandatory for CWSR's per-element weights — always use
  `split_dataset` or `fit_dataset`.
- Output JSON from training is directly consumable by `query`, and is the same
  schema used by the `eval`/refinement and bootstrap workflows.
- `var_count` is never inferred from a wide one-hot matrix by accident: pass
  it explicitly.
- Smaller `ops` sets + lower `max_depth` speed up runs dramatically; raise
  `max_expressions` for better final expressions.
- First use of Matbench tasks downloads/caches data after an interactive
  confirmation; the alloy databases ship in `examples/alloys/data`.


# Unified CWSR Framework — Design & Alloys-SR Merge Plan

Status: Phase 1 (foundation) implemented and import-validated.
Branches: developed on `develop` of `Composition-weighted-SR`.

## Goal

Merge the analysis features prototyped in the `Alloys-SR` repository into the
`cwsr` core so that the resulting framework is **task-agnostic**: the same
training / evaluation / prediction / inverse-design / Pareto / bootstrap code
runs for *any* dataset that maps compositions (118-dim atomic-fraction
vectors) to target properties (Matbench, alloy databases, custom `.npz`, …).

`Alloys-SR` itself becomes a thin **example** (`examples/alloys/`) that
registers its preprocessed databases and drives them through the unified core.

## Architecture

```
cwsr/                  unified package: flattened engine + task-agnostic API
  regressor.py         CWSR Regressor (MCTS-driven symbolic regression)
  mcts.py / gp.py / checkpoint.py
  exp_tree.py / exp_queue.py / reward.py
  formula.py           composition<->formula helpers (form2comp, symbols, formatting)
  datasets/
    base.py            CompositionDataset container + task-agnostic train/valid split
    matbench.py        Matbench provider
    alloy.py           alloy .npz provider
    registry.py        name/path -> provider dispatch ("works for any task")
  model/
    train.py           unified train driver (thin wrapper over cwsr.Regressor)
  evaluate/
    eval.py            unified eval + weight refinement (--refine)
    metrics.py         MAE/RMSE/R2 (+ bootstrap CI helpers)
  predict/
    forward.py         jitted forward eval + analytic gradients (ported from Alloys)
    query.py           formula -> property (ported from Alloys, path-based)
  design/
    inverse.py         target / weighted-sum / Tchebycheff composition design
    pareto.py          multi-objective Pareto-front sweep
  validate/
    bootstrap.py       bootstrap UQ (parallel) + use-bootstrap-best
  plotting/            parity, Pareto, periodic-table, element plots
  cli.py               single `cwsr` CLI (train/eval/query/inverse/pareto/bootstrap)
dataloader.py          thin re-export shim (kept for backward compatibility)
eval.py                thin re-export shim (backward compat)
run_cwsr.py            thin re-export shim (backward compat)
visiualize.py          kept (periodic-table visualiser, expanded later)
examples/
  matbench/            existing Matbench examples -> driven via cwsr datasets
  alloys/              NEW: alloy database example (uses cwsr unified API)
```

### Task-agnostic data abstraction

Any dataset is described by one object with a stable schema; the whole
framework (train/eval/query/inverse/pareto/bootstrap) is written against this
schema, never against alloy- or Matbench-specific paths.

```python
@dataclass
class CompositionDataset:
    name: str            # e.g. "matbench_expt_gap", "alloy_density", "<custom>"
    compositions: np.ndarray   # (n, 118)
    targets: np.ndarray        # (n,)
    formulas: list[str] | None
    target_name: str           # property name/units
    source: str                # provenance
    meta: dict                 # var_count, units, etc.
```

`cwsr.datasets.registry.get_dataset(...)` resolves a task name / npz path to a
`CompositionDataset`, so adding a new database = adding a provider, no code
changes downstream.

## Merge map (Alloys-SR -> cwsr)

| Alloys-SR module | New home | Treatment |
|---|---|---|
| `forward.py` | `cwsr/predict/forward.py` | ported, generalized (no alloy paths) |
| `query.py` | `cwsr/predict/query.py` | ported, generalized (results-path based) |
| `inverse.py` | `cwsr/design/inverse.py` | port, generalize refined-results loading |
| `pareto.py` | `cwsr/design/pareto.py` | port, generalize |
| `bootstrap_utils.py` | `cwsr/validate/bootstrap.py` | port, generalize |
| `bootstrap_parity.py` | `cwsr/validate/bootstrap.py` | port, generalize |
| `use_bootstrap_best.py` | `cwsr/validate/bootstrap.py` | port, generalize |
| `cwsr_alloys.py` | `examples/alloys/` + `cwsr/model/train.py` | training now task-agnostic; example registers alloy npz |
| `eval.py` / `visiualize.py` | `cwsr/evaluate/eval.py`, `cwsr/plotting/` | unify; keep top-level shims |
| `preprocess.py`, `verify_data.py` | `examples/alloys/preprocess/` | example-local (data-specific) |
| `plot_*.py`, `plot_utils.py` | `cwsr/plotting/` | port; examples re-use |
| shell/slurm drivers | `examples/alloys/` | example-local |
| datasets & data | `examples/alloys/processed_data`, `experiments/` | example data (git-ignored where appropriate) |

## Phased execution (each increment lands as a commit on `develop`)

1. **Phase 1 (done here):** `cwsr/` package skeleton, `formula.py`, dataset
   layer (`base`, `matbench`, `alloy`, `registry`), `predict/forward.py` +
   `predict/query.py` (ported), `examples/alloys/` scaffold, `setup.py` +
   console-script wiring, top-level backward-compat shims, this doc. Validated
   with `python -c "import cwsr..."` and `py_compile`.
2. **Phase 2:** `cwsr/model/train.py` unified training CLI + Matbench/alloy
   runners; reconcile the two `eval.py` forks into `cwsr/evaluate/eval.py`.
3. **Phase 3:** port + generalize `design/inverse.py` and `design/pareto.py`.
4. **Phase 4:** port + generalize `validate/bootstrap.py` (bootstrap_utils +
   bootstrap_parity + use_bootstrap_best) and `evaluate/metrics.py`.
5. **Phase 5:** `cwsr/plotting/` (parity, Pareto, periodic-table, element
   plots) and migrate `visiualize.py`.
6. **Phase 6:** full `cwsr` CLI (`cwsr train/eval/query/inverse/pareto/
   bootstrap --dataset ...`), README rewrite, end-to-end alloy example smoke
   test on real processed data, remove duplicated example copies.

## Compatibility

- The engine lives inside `cwsr` as a flat set of modules (`regressor.py`,
  `mcts.py`, `gp.py`, `checkpoint.py`, `exp_tree.py`, `exp_queue.py`,
  `reward.py`). Imports use `cwsr.*` (see the package docstring for the flat
  name mapping).
- Existing top-level modules (`dataloader`, `eval`, `run_cwsr`,
  `visiualize`) stay importable as shims so nothing breaks mid-migration.

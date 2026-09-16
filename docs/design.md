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
  data.py              CompositionDataset schema + element-safe train/valid split
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
datasets/              concrete databases + dispatch (separate top-level package)
  matbench.py          Matbench provider (+ raw load_matbench / load_matbench_test)
  alloy.py             alloy .npz provider (bundled data under alloys/data/)
  alloys/data/*.npz    bundled example alloy databases (no download needed)
  registry.py          name/path -> provider dispatch ("works for any task")
analysis/              post-training analysis (top-level package, lazy submodules)
  _common.py           shared helpers (results loading, element indices, JSON)
  inverse.py           target / weighted-sum / Tchebycheff composition design
  pareto.py            Pareto front via Tchebycheff scalarization sweep
  bootstrap.py         bootstrap resampling UQ + refined-results export
scripts/               legacy command-line executables (examples, not installed)
  run_cwsr.py          Matbench CLI runner (engine-level, legacy)
  eval.py              Matbench eval + NLopt weight refinement (legacy)
  visiualize.py        PeriodicTableVisualizer used by eval.py (legacy)
  view_out.py          one-off paper plotter (training-log MAE curves)
  run.sh, eval.sh      thin shell wrappers with example argument sets
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

`datasets.registry.get_dataset(...)` resolves a task name / npz path to a
`CompositionDataset`, so adding a new database = adding a provider, no code
changes downstream.

The generic container itself lives in the framework core (`cwsr/data.py`);
`datasets/` only holds providers, the registry and the bundled databases.

## Merge map (Alloys-SR -> cwsr)

| Alloys-SR module | New home | Treatment |
|---|---|---|
| `forward.py` | `cwsr/predict/forward.py` | ported, generalized (no alloy paths) |
| `query.py` | `cwsr/predict/query.py` | ported, generalized (results-path based) |
| `inverse.py` | `analysis/inverse.py` | **ported**; expressions from any results JSON, elements by number/symbol |
| `pareto.py` | `analysis/pareto.py` | **ported**; Tchebycheff sweep + non-dominated filter |
| `bootstrap_utils.py` | `analysis/bootstrap.py` | **ported**; element-safe resampling, CIs, metrics |
| `bootstrap_parity.py` | `analysis/bootstrap.py` (CLI) | **replaced** by a task-agnostic bootstrap CLI; parity plots deferred to the plotting phase |
| `use_bootstrap_best.py` | `analysis/bootstrap.py` | **ported** as `write_refined_results` (refined-results export) |
| `cwsr_alloys.py` | `examples/alloys/` + `cwsr/model/train.py` | training now task-agnostic; example registers alloy npz |
| `eval.py` / `visiualize.py` | `cwsr/evaluate/`, `cwsr/plotting/` | unify; legacy copies kept as `scripts/` examples |
| `preprocess.py`, `verify_data.py` | `examples/alloys/preprocess/` | example-local (data-specific) |
| `plot_*.py`, `plot_utils.py` | `cwsr/plotting/` | port; examples re-use |
| shell/slurm drivers | `examples/alloys/` | example-local |
| datasets & data | `datasets/alloys/data`, `experiments/` | example data (git-ignored where appropriate) |

## Phased execution (each increment lands as a commit on `develop`)

1. **Phase 1 (done here):** `cwsr/` package skeleton, `formula.py`, dataset
   layer (`data`, `matbench`, `alloy`, `registry`), `predict/forward.py` +
   `predict/query.py` (ported), `examples/alloys/` scaffold, `setup.py` +
   console-script wiring, this doc. Validated
   with `python -c "import cwsr..."` and `py_compile`.
2. **Phase 2:** `cwsr/model/train.py` unified training CLI + Matbench/alloy
   runners; reconcile the two `eval.py` forks into `cwsr/evaluate/eval.py`.
3. **Phase 3 (done):** `analysis/inverse.py` + `analysis/pareto.py` ported and
   generalized (accepted expressions come from any results JSON, elements may
   be symbols or atomic numbers, I/O is dataset-agnostic).
4. **Phase 4 (done for bootstrap UQ):** `analysis/bootstrap.py` ports
   `bootstrap_utils` + `use_bootstrap_best` (element-safe resampling, NLopt
   weight re-optimisation, CIs, metrics, refined-results export). The
   parity-plot driver (`bootstrap_parity.py`) and `evaluate/metrics.py` remain
   with the plotting phase.
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
- The legacy Matbench runners are plain scripts under `scripts/` (`run_cwsr.py`,
  `eval.py`, `visiualize.py`) kept as examples of command-line usage — they are
  not installed entry points. All Matbench loading (including
  `load_matbench` / `load_matbench_test`) lives in `datasets/matbench.py`, so
  the repository root holds no standalone Python modules.

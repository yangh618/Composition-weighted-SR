# Pre-Publication Testing Checklist (CWSR)

Checklist of tests/QA to complete before making the project (and its docs)
public. Commands assume the `cwsr` conda env and run from the repo root;
`PY` below = `/home/huangyang/anaconda3/envs/cwsr/bin/python` (adjust for your env).

Legend: ☑ already verified in this workspace (re-run on a fresh clone);
[ ] = open item needing work.

---

## 0. Fresh-environment & install reproducibility
- [ ] Create env from scratch: `conda env create -f environment.yml` → env named `cwsr`.
- [ ] `pip install -e .` succeeds; console scripts installed:
      `command -v cwsr-query cwsr-gallery cwsr-inverse cwsr-pareto cwsr-bootstrap`.
- [ ] On a clean interpreter, `python -c "import cwsr; from cwsr import Regressor"` works
      (no reliance on repo working dir / installed package consistency).
- [ ] Confirm no build/lint leftovers get re-committed (`*.egg-info/`, numba `*.nbc/*.nbi`,
      `__pycache__/`, `build/`, `dist/`) — covered by `.gitignore`.

## 1. Import & reference hygiene
- [ ] No leftover references to the removed legacy engine package remain in code
      (all imports use `cwsr.*`).
- [ ] `grep -rn 'home/huangyang'` → remove any personal/machine-specific paths
      from code and docs (e.g. Alloys-SR sibling path, conda prefix).
- [ ] Every symbol documented in `docs/reference.md` imports:
      framework + engine classes/functions (see the validation snippet in §9).

## 2. Unit tests (automated — `tests/unit/`)
Run with `pytest tests/unit`. Coverage map (test module → contract verified):
- [x] `test_public_api.py` — documented imports/exports, `__version__` vs `setup.py`,
      lazy `analysis` import, `Regressor` construction defaults + validation errors.
- [x] `test_data.py` — `CompositionDataset` dtypes/shapes/validation, per-instance
      `meta`, `split_dataset` determinism/size/coverage + species-coverage invariant
      (ratio→1, tiny datasets, singleton moves).
- [x] `test_formula.py` — `form2comp` normalization/indexing/fractional formulas and
      invalid-input behaviour, `composition_to_formula`/`format_composition`.
- [x] `test_forward.py`, `test_forward_prediction.py`, `test_composition_helpers.py`
      — every operator (`add sub mul div Max Min Pow sqrt exp log sin cos`), constant
      broadcast, IEEE edge cases (div-by-zero → inf, `log(0)` → -inf, `sqrt(-x)` → nan),
      analytic gradients vs finite differences, element parsing/normalization.
- [x] `test_exp_tree.py`, `test_exp_queue.py` — tree construction/limits/`get_expression`
      /deterministic `random_fill`/`clear`; queue ordering, near-duplicate suppression,
      non-finite rejection, capacity eviction.
- [x] `test_reward.py` — `Heaviside_vec`, `sp_module`, `Optimizer` sigma/algorithm
      resolution, `run_nlopt` on a quadratic, parameter init shapes, `valid_expression`
      accept/reject, MAE-loss value + gradient at/away from the optimum.
- [x] `test_checkpoint.py` — `mcts_to_dict`/`mcts_from_dict` round trip (counters, queues,
      tree wiring) and `save_checkpoint`/`load_checkpoint` JSON round trip.
- [x] `test_regressor.py`, `test_regressor_search.py` — ops/arity/complexity wiring,
      rate clipping, `num_trials` handling, `save_status` output contract, full search
      contract + recovery of a known synthetic law + seed reproducibility.
- [x] `test_query.py` — refined-results loading, hand-checked predictions, CLI paths.
- [x] `test_datasets.py` — registry listing/dispatch, alloy `.npz` provider (bundled and
      synthetic), error paths; Matbench declared without downloading.
- [x] `test_analysis_*.py` — shared helpers, inverse design (exact target hits,
      unreachable-target residuals, element constraints), Pareto non-dominance/front
      span, bootstrap CIs/metrics/export/CLI.

## 3. Test suite (`tests/`)
Standard pytest layout — see "Running the suite" below for commands:

```
tests/
  conftest.py                 shared fixtures (synthetic datasets, tiny fit, tmp files)
  fixtures/                   synthetic-problem helpers + reference data
  unit/                       fast contract/unit tests (no search)
  integration/                end-to-end workflow + CLI subprocess tests
```

- [x] `tests/integration/test_end_to_end.py` — synthetic dataset → `fit_dataset`
      (CWSR search) → saved artifact → re-evaluated metrics → `cwsr-query` /
      inverse-design consumption. Runs in seconds on 24 synthetic samples.
- [x] `tests/integration/test_cli.py` — `--help` and argument-rejection for every
      module CLI plus the legacy `scripts/*.py`, and one executable
      `analysis.bootstrap` run through a real subprocess.
- [x] No test needs the network, GPU, MPI, `$HOME` or a real materials dataset;
      all artifacts are written to `tmp_path`.

### Running the suite
```bash
pytest                      # whole suite (~45 s; tiny searches included)
pytest -q                   # quiet
pytest -v                   # verbose
pytest tests/unit           # fast unit tests only
pytest tests/integration    # end-to-end + CLI only
pytest -m "not slow"        # skip the tiny symbolic-regression searches (~18 s)
pytest -k formula           # select by keyword
```
`pytest` configuration lives in `pytest.ini` (test paths, `pythonpath = .` so the
repo packages are importable without an editable install). Test dependencies are
declared as `pip install -e ".[test]"`.

### Verifying the built wheel (installed-package run)
```bash
python -m build --wheel --outdir dist
python -m venv --system-site-packages /tmp/cwsr-test
/tmp/cwsr-test/bin/python -m pip install --no-deps dist/*.whl
cd /tmp && /tmp/cwsr-test/bin/python -c "import cwsr, datasets, analysis; print(cwsr.__file__)"
# -> /tmp/cwsr-test/lib/python3.11/site-packages/cwsr/__init__.py  (not the source tree)

mkdir -p /tmp/wheel_tests && cp -r tests /tmp/wheel_tests/tests
sed 's|^pythonpath = .$|pythonpath =|' pytest.ini > /tmp/wheel_tests/pytest.ini
cd /tmp/wheel_tests && /tmp/cwsr-test/bin/python -m pytest -q
```
Running from a copy of `tests/` outside the repository guarantees the imports come
from `site-packages` rather than the working tree. Three checkout-only checks skip
in that mode (`setup.py` version string, the unpackaged `scripts/*.py` entry points).

## 4. CLI tests
- [ ] `cwsr-query --results <refined.json> "FeCrCoNi"` prints a finite value.
- [ ] `cwsr-query --results <refined.json>` (no formula) enters interactive mode and exits on `q`.
- [ ] `cwsr-query` / `cwsr-gallery` at least parse `--help` without import errors,
      and so do the legacy scripts: `python scripts/run_cwsr.py --help`,
      `python scripts/eval.py --help`.
- [ ] Verify output/params JSON written by training are valid JSON and loadable by
      `cwsr.predict.query`.
- [ ] `cwsr-inverse` / `cwsr-pareto` / `cwsr-bootstrap` parse `--help`; an
      inverse run hits a reachable target, a 2-objective Pareto sweep returns
      non-dominated points, and a small bootstrap run
      (`--n_bootstrap 5 --num_trials 1`) reports CIs with `n_successful > 0`
      and (with `--refined_output`) a refined-results file that `cwsr-query`
      can read.

## 5. Docs build & preview
- [ ] `PY -m mkdocs build --strict` exits 0 (all internal links resolve).
- [ ] `PY -m mkdocs serve -a 127.0.0.1:8000` → check `/`, `/tutorial/`,
      `/reference/`, `/design/` return HTTP 200.
- [ ] Add a CI docs job that runs the strict build on push (so it can't regress).
- [ ] Set `repo_url` in `mkdocs.yml` before publishing (adds edit/source links).

## 6. Data & licensing (publication-critical)
- [ ] Decide what is committed. `examples/` is currently **git-ignored**; the
      bundled alloy `.npz` were copied from the separate `Alloys-SR` repo and
      retain their original source licensing — do **not** publish those unless
      licensing is confirmed/attributed.
- [ ] Add/adjust `LICENSE` (MIT present) and `NOTICE`/attribution for bundled data.
- [ ] Confirm no large files / model checkpoints / `experiments/` / `results/`
      would be pushed (they are git-ignored; verify on a fresh clone with
      `git ls-files`).
- [ ] Scan for secrets/keys/tokens in tracked files.

## 7. Optional CI (recommended before public)
- [ ] Workflow that runs: `pip install -e ".[test]"` → `pytest -q` → wheel build →
      `mkdocs build --strict` on push/PR.
- [ ] Python 3.10 and 3.11 jobs (setup declares `>=3.10`).

## 8. Publication itself
- [ ] Decide branch policy: work on `develop`, merge to `main`.
- [ ] Make repo **public** (only after the above pass) OR publish docs only.
- [ ] Enable Pages: Settings → Pages → Deploy from branch `gh-pages` / root.
- [ ] Confirm live URL: `https://<user>.github.io/<repo>/`.

## 9. Quick local validation snippet (reference-hygiene)
```bash
PY=/home/huangyang/anaconda3/envs/cwsr/bin/python
$PY - <<'PY'
import cwsr
from cwsr import Regressor, simplify_expression
from datasets import get_dataset, list_datasets, register_provider, CompositionDataset
from cwsr.data import split_dataset
from cwsr.model import train, fit_dataset
from cwsr.predict import (compile_expression, compile_gradient_functions,
                          predict_vector, predict_and_gradient, jit_compile)
from cwsr.predict.forward import predict_vector_and_gradients
from cwsr.predict.query import load_refined, predict, main
from cwsr.mcts import MCTS, MCTS_Node
from cwsr.gp import GPManager
from cwsr.exp_tree import ExpTree, ExpTreeBase
from cwsr.exp_queue import Exp_Queue, Queue_Base
from cwsr.reward import Optimizer, sp_module, Heaviside_vec
from cwsr.checkpoint import save_checkpoint, load_checkpoint, mcts_to_dict, mcts_from_dict
from cwsr.plotting import periodic_table_figure, present_mask, build_gallery, main
from analysis import inverse, pareto, bootstrap
from analysis._common import load_expression_from_results, resolve_active_indices
from analysis.inverse import optimize_target, optimize_weighted_sum, optimize_tchebycheff
from analysis.pareto import compute_pareto_front_analytical
from analysis.bootstrap import (bootstrap_resample, compute_confidence_intervals,
                                compute_bootstrap_metrics, run_bootstrap,
                                load_top_expressions, write_refined_results)
print("ok:", len(list_datasets()), "datasets registered")
PY
```

## Known gaps to close before publication
- The pytest suite (§2–§3) covers the public API, dataset/split logic, expression
  compilation and operators, the queue/tree/checkpoint internals, the search
  contract, the analysis tools and the CLIs. Not yet covered: `cwsr.mcts`/
  `cwsr.gp` mutation+crossover internals in isolation (exercised indirectly by the
  search tests), `cwsr.plotting` (untracked WIP), and Matbench *download* paths
  (deliberately offline).
- **Known defect (recorded, not fixed):** `cwsr.predict.forward.predict_vector_and_gradients`
  is documented public API but is unused and broken for every `var_count` — its
  einsum subscripts (`"nvi,ij->nj"`) do not match `grads.shape == (n, var_count)`;
  the intended contraction is `"nv,vj->nj"`. Pinned by an `xfail` test in
  `tests/unit/test_forward_prediction.py`. The same helper is imported by the §9
  reference-hygiene snippet, so that snippet cannot call it.
- **Known limitation (documented by test):** `num_trials=0` passes
  `Regressor.__init__` but crashes the search in `MCTS_Node.backpropagate`
  (`TypeError: can only concatenate list (not "NoneType") to list`); see
  `tests/unit/test_regressor_search.py::test_num_trials_zero_crashes_the_search`.
- `form2comp` surfaces the underlying parser error type (pymatgen `ValueError` for
  unparsable strings, ASE `KeyError` for unknown symbols) — pinned by
  `tests/unit/test_formula.py`.
- The legacy Matbench runners (`scripts/run_cwsr.py`, `scripts/eval.py`) are
  kept as runnable examples only; they are not migrated onto the unified `cwsr`
  framework (see `docs/design.md`).
- `analysis/` ports inverse design, Pareto fronts and bootstrap UQ; the plotting
  suite (parity/Pareto/periodic-table figures, incl. the old
  `bootstrap_parity.py` driver) and the unified `cwsr` CLI are still pending
  (see `docs/design.md`).
- Alloy data + example scripts live under git-ignored `examples/`; ensure the
  published docs don't reference private/local paths a public reader can't reach.


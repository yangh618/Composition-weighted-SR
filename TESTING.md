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
      `command -v cwsr cwsr-eval cwsr-query`.
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

## 2. Engine unit tests (to add — currently only smoke-tested)
- [ ] `Regressor` recovers a known closed form on synthetic data
      (e.g. `y = x0 + 2*x1`) within tolerance using a small budget.
- [ ] `fit` returns the documented tuple; `outputs[0]` keys are exactly
      `{expression, weights, train_reward, valid_reward, mae, mae_valid, rank}`.
- [ ] `var_count=None` fallback does not silently become 118 on wide inputs
      (documented trap — assert users must pass it).
- [ ] Numeric-gradient check: `predict_and_gradient` vs finite differences.
- [ ] `split_dataset` species-coverage invariant: every element present in the
      full set is present in the training split; edge cases — `ratio→1`,
      few samples, all-valid singleton moves (regression for the earlier bug).
- [ ] `checkpoint` round trip: `save_checkpoint` → `load_checkpoint` → `mcts_from_dict`
      reproduces `count_num`/tree.
- [ ] `formula` helpers: `form2comp` normalization (sums to 1),
      `composition_to_formula`/`format_composition` round trip,
      118-element indexing.
- [ ] First-run numba JIT compilation completes without error (per module used).

## 3. Smoke / integration tests (present in `tests/`)
- [ ] `python tests/smoke_alloy.py   --property density --max-samples 150 --max-expressions 80`
- [ ] `python tests/smoke_matbench.py --task matbench_glass --max-samples 150 --max-expressions 80`
      (first run downloads the task after an interactive prompt).
- [ ] Run `./tests/run_short.sh` end-to-end in one go.
- [ ] Re-run all of the above on a **fresh clone** (not the working dir) to prove
      the package is self-contained.

## 4. CLI tests
- [ ] `cwsr-query --results <refined.json> "FeCrCoNi"` prints a finite value.
- [ ] `cwsr-query --results <refined.json>` (no formula) enters interactive mode and exits on `q`.
- [ ] `cwsr` / `cwsr-eval` at least parse `--help` without import errors.
- [ ] Verify output/params JSON written by training are valid JSON and loadable by
      `cwsr.predict.query`.

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
- [ ] Workflow that runs: import test + the two smoke tests + `mkdocs build --strict`
      on push/PR.
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
from cwsr.datasets import get_dataset, list_datasets, register_provider, CompositionDataset
from cwsr.datasets.base import split_dataset
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
print("ok:", len(list_datasets()), "datasets registered")
PY
```

## Known gaps to close before publication
- Engine internals currently have **only smoke coverage**, not unit tests (§2).
- `eval.py` / `run_cwsr.py` (top-level) still target the Matbench workflow and are
  not yet migrated onto the unified `cwsr` framework (see `docs/design.md`).
- Inverse design, Pareto, bootstrap UQ and plotting are not yet ported into `cwsr`
  (present only in the reference `Alloys-SR` project).
- Alloy data + example scripts live under git-ignored `examples/`; ensure the
  published docs don't reference private/local paths a public reader can't reach.


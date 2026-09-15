# `scripts/` — legacy command-line scripts (runnable examples)

Standalone executables kept for reference and as examples of driving CWSR from
the command line. The **maintained** interfaces are the `cwsr-query` /
`cwsr-gallery` console scripts and the Python API
(`datasets.get_dataset(...)` + `cwsr.model.fit_dataset(...)`).

| Script | Purpose |
|---|---|
| `run_cwsr.py` | Legacy Matbench CLI runner: fit on a Matbench task via `cwsr.Regressor`, write `cwsr_outputs_*` / `cwsr_params_*` JSON. |
| `eval.py` | Evaluate a `cwsr_outputs_*.json`: optional NLopt re-refinement of the weights, test MAE/accuracy, periodic-table plot. |
| `visiualize.py` | `PeriodicTableVisualizer` used by `eval.py` (superseded by `cwsr.plotting.periodic_table`). |
| `view_out.py` | One-off paper plotter: parses training logs into an MAE-vs-nodes figure. Edit `LOG_FILES` at the top to point at your runs. |
| `run.sh`, `eval.sh` | Thin wrappers around `run_cwsr.py` / `eval.py` with an example argument set. |
| `rsync.sh` | Local data-sync helper (git-ignored). |
| `README.org` | Original org-mode notes for the scripts above (superseded by the top-level `README.md`). |

## Running them

These scripts import the framework (`cwsr`, plus the Matbench loaders in
`datasets.matbench`), so make the project importable first — see
[`TESTING.md`](../TESTING.md) §0:

```bash
pip install -e .                     # recommended
python scripts/run_cwsr.py --task matbench_expt_gap --max_expressions 200
python scripts/eval.py --model_path <cwsr_outputs_*.json> --task matbench_expt_gap
```

Or, without installing, run them from the repository root with the root on the
path (`visiualize.py` is found next to `eval.py` automatically):

```bash
PYTHONPATH=. python scripts/run_cwsr.py --help
PYTHONPATH=. python scripts/eval.py --help
```

The `*.sh` wrappers resolve their sibling scripts by path, so they can be
invoked from anywhere (relative `--output_dir` paths resolve in your cwd).

Matbench tasks download/cache on first use (after an interactive confirmation).

## For new work

Prefer the task-agnostic path instead of these Matbench-specific scripts:

```python
from datasets import get_dataset
from cwsr.model import fit_dataset

ds = get_dataset("matbench_expt_gap", fold=0)      # or "alloy_density", or a .npz path
fit_dataset(ds, output_dir="results",
            var_count=3, ops=["mul", "add", "sub", "R"], max_expressions=200)
```

# Short-time CWSR smoke tests

Runnable scripts that exercise the task-agnostic `cwsr` pipeline end-to-end on
both a **Matbench** task and the **alloy database**, quickly. Each script:

1. loads the dataset through the unified `cwsr.datasets` registry
   (`get_dataset("alloy_density")`, `get_dataset("matbench_glass")`),
2. downsamples to a small subset to keep runtime short,
3. runs a small bounded CWSR fit (`cwsr.Regressor`),
4. does a forward-prediction sanity check on the discovered model,
5. writes `outputs`/`params` JSON under `tests/_out/` and reports timing +
   the best expression.

These are *smoke tests*: they confirm the whole stack runs correctly, not that
the search is accurate (use large `--max-expressions` and full data for that).

## Run everything

```bash
conda activate cwsr
./tests/run_short.sh
```

## Run individually

```bash
# Alloy database (density) — uses the bundled data under examples/alloys/data
python tests/smoke_alloy.py --property density --max-samples 150 --max-expressions 80

# Matbench — real data via cwsr.datasets; if not cached it ASKS before downloading
python tests/smoke_matbench.py --task matbench_glass --max-samples 150 --max-expressions 80
python tests/smoke_matbench.py --task matbench_expt_gap --yes   # auto-confirm download
python tests/smoke_matbench.py --task matbench_expt_gap --no    # abort if not cached
```

## Common options

| Flag | Default | Meaning |
|---|---|---|
| `--max-samples` | all | Downsample the dataset to N rows (keeps runs short) |
| `--max-expressions` | 150 | CWSR search budget |
| `--max-depth` | 5 | Max expression depth |
| `--var-count` | 3 | Number of latent variables |
| `--num-parallel` | 4 | Parallel processes |
| `--seed` | 1 | Reproducibility |
| `--output-dir` | `tests/_out` | Where JSON artifacts are written |

## Notes

- Requires the `cwsr` conda env (all numerical deps incl. `matbench`).
- Matbench data downloads from figshare on first use (~<1 MB for
  `matbench_glass`); offline runs need the task already cached
  (see `python dataloader.py`).
- For alloy, set `--data-dir` to wherever your processed `*.npz` databases
  live (default: `/home/huangyang/Git/Alloys-SR/processed_data`), or pass
  `--npz /path/to/property.npz`.

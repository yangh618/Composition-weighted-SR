"""Shared helpers for short-time CWSR smoke tests.

Each smoke test:
  1. loads a dataset through the task-agnostic `cwsr.datasets` registry,
  2. downsamples to a small subset to keep the run short,
  3. runs a bounded CWSR fit (small `max_expressions`) on that subset,
  4. does a quick forward-prediction sanity check on the discovered model,
  5. writes outputs/params JSON and reports timing + the best expression.

These are *smoke tests*: they verify the whole pipeline runs end-to-end
quickly, not that the search finds an accurate model (use larger budgets /
the full datasets for real accuracy).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Allow running scripts directly from anywhere in the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Default location of the alloy databases (self-contained example copy).
ALLOW_DATA_DIR = str(ROOT / "examples" / "alloys" / "data")

# Small default hyper-parameters so a run takes ~seconds-to-a-couple-of-minutes.
DEFAULT_OPS = ["mul", "add", "sub", "R"]
DEFAULT_HYPERPARAMS = dict(
    var_count=3,
    max_depth=5,
    max_constants=2,
    K=10,
    max_expressions=150,
    num_parallel=4,
    num_batches=16,
)


# =========================================================================
# Matbench download confirmation (ask before hitting the network)
# =========================================================================
#: Approximate download size in MB for common tasks (informational only).
MATBENCH_APPROX_MB = {
    "matbench_glass": 0.04,
    "matbench_dielectric": 0.4,
    "matbench_expt_gap": 0.9,
    "matbench_mp_gap": 70,
}


def matbench_is_cached(task: str) -> Optional[bool]:
    """Return True/False if we can tell the task is already cached, else None.

    Matbench/matminer cache task JSON under ``<matminer>/datasets/``.
    """
    try:
        import matminer
        base = Path(matminer.__file__).resolve().parent / "datasets"
        candidates = [base / f"{task}.json.gz", base / f"{task}.json"]
        return any(p.exists() for p in candidates)
    except Exception:
        return None


def confirm_matbench_download(task: str,
                              assume_yes: bool = False,
                              assume_no: bool = False) -> bool:
    """Ask the user before downloading a Matbench task that isn't cached.

    Returns ``True`` if the task is already cached (never forces a download
    refusal for cached data), otherwise honours ``assume_yes``/``assume_no``
    or prompts interactively.
    """
    cached = matbench_is_cached(task)
    if cached:
        return True
    if assume_yes:
        return True
    if assume_no:
        return False

    size = MATBENCH_APPROX_MB.get(task)
    size_txt = f" (~{size} MB)" if size is not None else ""
    print(f"[download] Matbench task '{task}' is not cached locally.")
    if not sys.stdin.isatty():
        print("[download] No terminal for confirmation; run with --yes to "
              "allow downloading.")
        return False
    ans = input(f"[download] Fetch '{task}' now{size_txt}? [y/N]: ").strip().lower()
    return ans in ("y", "yes")



def downsample(compositions: np.ndarray,
               targets: np.ndarray,
               max_samples: Optional[int],
               seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministically reduce a dataset to ``max_samples`` rows (or return all)."""
    if max_samples is None or compositions.shape[0] <= max_samples:
        return compositions, targets
    rng = np.random.default_rng(seed)
    idx = rng.permutation(compositions.shape[0])[:max_samples]
    return compositions[idx], targets[idx]


def run_short_fit(compositions: np.ndarray,
                  targets: np.ndarray,
                  output_dir: "str | Path",
                  run_name: str,
                  split_ratio: float = 0.8,
                  seed: Optional[int] = None,
                  hyperparams: Optional[dict] = None,
                  ) -> dict:
    """Fit a small CWSR model, save artifacts and return a report dict."""
    from cwsr import Regressor
    from cwsr.datasets.base import split_dataset
    from cwsr.predict.forward import compile_expression, predict

    hp = dict(DEFAULT_HYPERPARAMS)
    hp.update(hyperparams or {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, valid_idx = split_dataset(compositions, targets,
                                         ratio=split_ratio, seed=seed)
    print(f"  split: train={len(train_idx)} valid={len(valid_idx)}")

    model = Regressor(
        x_train=compositions[train_idx],
        y_train=targets[train_idx],
        x_valid=compositions[valid_idx],
        y_valid=targets[valid_idx],
        ops=hp.pop("ops", DEFAULT_OPS),
        seed=seed,
        **hp,
    )

    t0 = time.time()
    fit_result = model.fit(seed=seed)
    elapsed = time.time() - t0
    outputs = fit_result[4]  # list of top expressions w/ weights + metrics

    if not outputs:
        raise RuntimeError("CWSR fit produced no outputs; try a larger "
                           "`max_expressions` budget.")

    best = outputs[0]
    best_expr = best["expression"]
    best_weights = np.asarray(best["weights"], dtype=np.float64)

    timestamp = int(time.time())
    out_file = output_dir / f"cwsr_outputs_{run_name}_{timestamp}.json"
    param_file = output_dir / f"cwsr_params_{run_name}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(outputs, f, indent=2)
    with open(param_file, "w") as f:
        json.dump({"run_name": run_name, **hp,
                   "split_ratio": split_ratio, "seed": seed, "ops": DEFAULT_OPS},
                  f, indent=2)

    # Quick forward sanity check on one sample using the best trained model.
    var_count = best_weights.shape[0]
    f_pred = compile_expression(best_expr, var_count)
    probe = compositions[0]
    y_pred = predict(probe, best_weights, f_pred)
    finite_ok = np.isfinite(y_pred)

    report = {
        "run_name": run_name,
        "n_train": len(train_idx), "n_valid": len(valid_idx),
        "best_expression": best_expr,
        "evaluations": int(fit_result[2]),
        "elapsed_sec": float(elapsed),
        "probe_y_pred": float(y_pred),
        "finite_ok": bool(finite_ok),
        "output_file": str(out_file),
    }
    return report


def report_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("CWSR SHORT SMOKE RESULT")
    print("=" * 60)
    print(f"  run          : {report['run_name']}")
    print(f"  best expr    : {report['best_expression']}")
    print(f"  evaluations  : {report['evaluations']}")
    print(f"  elapsed      : {report['elapsed_sec']:.1f} s")
    print(f"  probe finite : {report['finite_ok']}  (y_pred={report['probe_y_pred']:.4g})")
    print(f"  outputs      : {report['output_file']}")
    print("=" * 60)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Downsample to this many samples (default: all).")
    parser.add_argument("--output-dir", type=str, default="tests/_out",
                        help="Where to write outputs/params JSON.")
    parser.add_argument("--max-expressions", type=int,
                        default=DEFAULT_HYPERPARAMS["max_expressions"])
    parser.add_argument("--max-depth", type=int,
                        default=DEFAULT_HYPERPARAMS["max_depth"])
    parser.add_argument("--var-count", type=int,
                        default=DEFAULT_HYPERPARAMS["var_count"])
    parser.add_argument("--num-parallel", type=int,
                        default=DEFAULT_HYPERPARAMS["num_parallel"])
    parser.add_argument("--seed", type=int, default=1)

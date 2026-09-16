#!/usr/bin/env python3
"""
Bootstrap Resampling (uncertainty quantification)
=================================================
Core functions for bootstrap resampling of CWSR predictions.

Ported from the reference ``Alloys-SR`` project (``bootstrap_utils.py`` +
``use_bootstrap_best.py``): the numerics are unchanged, only the imports and
paths are generalized (the engine is :class:`cwsr.Regressor`, the element-safe
split comes from :mod:`cwsr.data`, and expressions are read from any CWSR
results JSON). Run as ``cwsr-bootstrap`` or ``python -m analysis.bootstrap``.

Provides:
  - compile_expression: fast expression evaluation
  - load_top_expressions: top-N expressions from a results JSON
  - bootstrap_resample: bootstrap resampling with NLopt weight re-optimization
  - compute_confidence_intervals: percentile-based confidence intervals
  - compute_bootstrap_metrics: MAE, RMSE, R² with bootstrap uncertainty
  - write_refined_results: turn a bootstrap result into a refined-results JSON
"""

import argparse
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

from cwsr import Regressor
from cwsr.data import split_dataset

from joblib import Parallel, delayed

# Reuse core utilities from eval.py
from cwsr.predict.forward import (
    compile_expression as _forward_compile_expression,
)


from analysis._common import (
    load_expression_from_results,
    present_elements,
    save_json,
)


def compile_expression(expression: str, var_count: int):
    """Compile ``expression`` into a fast callable of ``var_count`` variables."""
    return _forward_compile_expression(expression, var_count)


warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# Default bootstrap parameters
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_CI_LEVEL = 95  # percent


def load_top_expressions(results_path: "str | Path",
                         top_n: int = 3) -> Optional[list]:
    """Load the top ``top_n`` entries of a CWSR results JSON.

    Returns a list of output dicts (each with ``expression``, ``weights``,
    ``mae``, ``mae_valid``, ...) as ranked by the training driver — index 0 is
    the best candidate. ``None`` if the file does not exist.
    """
    path = Path(results_path)
    if not path.exists():
        print(f"  WARNING: results not found: {path}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):          # a single output dict
        data = [data]
    return data[:top_n]


# ─────────────────────────────────────────────
# Bootstrap Resampling Core
# ─────────────────────────────────────────────
# Uses a fixed-expression bootstrap approach:
#   - The CWSR expression is kept fixed (from refined results)
#   - For each bootstrap iteration, the training data is resampled
#     (using split_dataset to ensure all chemical species are covered)
#   - The original weights are used as the starting point for NLopt
#     optimization on the bootstrap sample
#   - Predictions are computed on the full training and test sets

def _bootstrap_iteration(
    i: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    expression: str,
    weights_original: np.ndarray,
    var_count: int,
    num_trials: int,
    seed: int,
) -> dict:
    """
    Run a single bootstrap iteration (helper for parallel execution).

    Returns a dict with keys: 'i', 'y_pred_train', 'y_pred_test', 'weights', 'success'.
    """
    rng = np.random.default_rng(seed)
    n_features = x_train.shape[1]

    # Pre-compile the expression function
    expr_func = compile_expression(expression, var_count)

    # Use split_dataset to create a bootstrap sample that ensures
    # all chemical species are covered in the training set.
    boot_indices, _ = split_dataset(
        x_train, y_train, ratio=0.7, seed=rng.integers(0, 2**31)
    )
    x_boot = x_train[boot_indices]
    y_boot = y_train[boot_indices]
    train_indices, valid_indices = split_dataset(
        x_boot, y_boot, ratio=0.85, seed=rng.integers(0, 2**31)
    )
    x_valid = x_boot[valid_indices]
    y_valid = y_boot[valid_indices]
    x_boot = x_boot[train_indices]
    y_boot = y_boot[train_indices]

    # Create a Regressor on the bootstrap sample (same as eval.py)
    reg_boot = Regressor(
        x_train=x_boot,
        y_train=y_boot,
        x_valid=x_valid,
        y_valid=y_valid,
        var_count=var_count,
        verbose=False,
        optimization_method='LD_LBFGS',
    )

    # Re-optimize weights on bootstrap sample using NLopt optimization.
    # Use the original weights as the initial guess, then try additional
    # random initializations. Pick the one with lowest validation MAE.
    best_weights = None
    best_valid_mae = float("inf")

    for trial in range(num_trials * 2 + 1):
        is_positive_init = trial < num_trials
        try:
            if trial == 0:
                # Warm start: use original weights as initial guess
                object_func = reg_boot.build_MAE_loss(
                    expression, x_boot, y_boot
                )
                initial_guess = weights_original.flatten()
                optimized_params = reg_boot.optimizer.run_nlopt(
                    object_func, initial_guess, reg_boot.lbfgs_upper_bound,
                    xtol=1e-6, maxeval=100,
                )
                tabulated_weights = np.reshape(
                    optimized_params, (var_count, n_features)
                )
                # Compute validation MAE
                x_bar_valid = np.einsum("ni,vi->nv", x_valid, tabulated_weights)
                y_pred_valid = expr_func(x_bar_valid)
                mae_valid = np.mean(np.abs(y_pred_valid - y_valid))
            else:
                # Random initialization (standard optimize_weights)
                mae, mae_valid, tabulated_weights = reg_boot.optimize_weights(
                    expression, is_positive_init
                )

            if mae_valid < best_valid_mae:
                best_valid_mae = mae_valid
                best_weights = tabulated_weights
        except Exception:
            continue

    if best_weights is not None:
        # Compute predictions on full training and test sets
        x_bar_train = np.einsum("ni,vi->nv", x_train, best_weights)
        y_pred_train = expr_func(x_bar_train)

        x_bar_test = np.einsum("ni,vi->nv", x_test, best_weights)
        y_pred_test = expr_func(x_bar_test)

        return {
            "i": i,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "weights": best_weights,
            "success": True,
        }

    return {
        "i": i,
        "y_pred_train": None,
        "y_pred_test": None,
        "weights": None,
        "success": False,
    }


def bootstrap_resample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    expression: str,
    weights_original: np.ndarray,
    var_count: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    random_seed: Optional[int] = None,
    num_trials: int = 5,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform bootstrap resampling to estimate prediction uncertainty.

    For each bootstrap iteration:
      1. Resample the training data using split_dataset (ensures species coverage)
      2. Re-optimize the CWSR weights on the bootstrap sample via NLopt,
         starting from the original weights
      3. Compute predictions on the full training and test sets

    Parameters
    ----------
    x_train : np.ndarray, shape (n_train, n_features)
        Training composition features.
    y_train : np.ndarray, shape (n_train,)
        Training target values.
    x_test : np.ndarray, shape (n_test, n_features)
        Test composition features (for bootstrap evaluation).
    y_test : np.ndarray, shape (n_test,)
        Test target values.
    expression : str
        Symbolic expression string.
    weights_original : np.ndarray, shape (var_count, n_features)
        Original weights from refined results (used as initial guess).
    var_count : int
        Number of latent variables (usually 3).
    n_bootstrap : int
        Number of bootstrap iterations.
    random_seed : int, optional
        Random seed for reproducibility.
    num_trials : int
        Number of NLopt optimization trials per bootstrap iteration.
    n_jobs : int
        Number of parallel jobs (-1 for all CPUs, 1 for sequential).

    Returns
    -------
    y_pred_all_train : np.ndarray, shape (n_bootstrap, n_train)
        Training predictions from each bootstrap iteration.
    y_pred_all_test : np.ndarray, shape (n_bootstrap, n_test)
        Test predictions from each bootstrap iteration.
    weights_all : np.ndarray, shape (n_bootstrap, var_count, n_features)
        Fitted weights from each bootstrap iteration.
    success_mask : np.ndarray, shape (n_bootstrap,)
        Boolean mask indicating which bootstrap iterations succeeded.
    """
    rng = np.random.default_rng(random_seed)
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_features = x_train.shape[1]

    # Generate seeds for each iteration (deterministic from the master seed)
    iteration_seeds = [rng.integers(0, 2**31) for _ in range(n_bootstrap)]

    # Run bootstrap iterations in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_bootstrap_iteration)(
            i, x_train, y_train, x_test, y_test,
            expression, weights_original, var_count, num_trials,
            iteration_seeds[i],
        )
        for i in range(n_bootstrap)
    )

    # Assemble results
    y_pred_all_train = np.full((n_bootstrap, n_train), np.nan)
    y_pred_all_test = np.full((n_bootstrap, n_test), np.nan)
    weights_all = np.full((n_bootstrap, var_count, n_features), np.nan)
    success_mask = np.zeros(n_bootstrap, dtype=bool)

    for res in results:
        i = res["i"]
        if res["success"]:
            y_pred_all_train[i] = res["y_pred_train"]
            y_pred_all_test[i] = res["y_pred_test"]
            weights_all[i] = res["weights"]
            success_mask[i] = True

    return y_pred_all_train, y_pred_all_test, weights_all, success_mask


def compute_confidence_intervals(
    y_pred_all: np.ndarray,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confidence intervals from bootstrap predictions.

    Parameters
    ----------
    y_pred_all : np.ndarray, shape (n_bootstrap, n_samples)
        Predictions from each bootstrap iteration.
    ci_level : float
        Confidence level in percent (e.g., 95 for 95% CI).

    Returns
    -------
    y_pred_mean : np.ndarray, shape (n_samples,)
        Mean prediction across bootstrap iterations.
    y_pred_std : np.ndarray, shape (n_samples,)
        Standard deviation of predictions.
    ci_lower : np.ndarray, shape (n_samples,)
        Lower bound of confidence interval.
    ci_upper : np.ndarray, shape (n_samples,)
        Upper bound of confidence interval.
    """
    # Filter out NaN values (failed bootstrap iterations)
    valid_mask = ~np.isnan(y_pred_all).any(axis=1)
    y_pred_valid = y_pred_all[valid_mask]

    if len(y_pred_valid) == 0:
        raise ValueError("No successful bootstrap iterations!")

    alpha = (100 - ci_level) / 2
    lower_pct = alpha
    upper_pct = 100 - alpha

    y_pred_mean = np.mean(y_pred_valid, axis=0)
    y_pred_std = np.std(y_pred_valid, axis=0, ddof=1)
    ci_lower = np.percentile(y_pred_valid, lower_pct, axis=0)
    ci_upper = np.percentile(y_pred_valid, upper_pct, axis=0)

    return y_pred_mean, y_pred_std, ci_lower, ci_upper


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_pred_all: np.ndarray,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> dict:
    """
    Compute performance metrics with bootstrap uncertainty.

    For each bootstrap iteration, compute MAE and R², then report
    the mean and confidence intervals of these metrics.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True target values.
    y_pred_all : np.ndarray, shape (n_bootstrap, n_samples)
        Predictions from each bootstrap iteration.
    ci_level : float
        Confidence level in percent.

    Returns
    -------
    dict with keys: mae_mean, mae_ci, r2_mean, r2_ci, etc.
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(y_pred_all).any(axis=1)
    y_pred_valid = y_pred_all[valid_mask]
    n_valid_boot = len(y_pred_valid)

    if n_valid_boot == 0:
        return {
            "mae_mean": float("nan"),
            "mae_std": float("nan"),
            "mae_ci": (float("nan"), float("nan")),
            "rmse_mean": float("nan"),
            "rmse_std": float("nan"),
            "r2_mean": float("nan"),
            "r2_std": float("nan"),
            "r2_ci": (float("nan"), float("nan")),
            "n_successful": 0,
        }

    alpha = (100 - ci_level) / 2

    mae_all = np.zeros(n_valid_boot)
    r2_all = np.zeros(n_valid_boot)
    rmse_all = np.zeros(n_valid_boot)

    y_true_mean = np.mean(y_true)

    for i in range(n_valid_boot):
        residuals = y_pred_valid[i] - y_true
        mae_all[i] = np.mean(np.abs(residuals))
        rmse_all[i] = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        r2_all[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mae_mean": float(np.mean(mae_all)),
        "mae_std": float(np.std(mae_all, ddof=1)),
        "mae_ci": (
            float(np.percentile(mae_all, alpha)),
            float(np.percentile(mae_all, 100 - alpha)),
        ),
        "rmse_mean": float(np.mean(rmse_all)),
        "rmse_std": float(np.std(rmse_all, ddof=1)),
        "r2_mean": float(np.mean(r2_all)),
        "r2_std": float(np.std(r2_all, ddof=1)),
        "r2_ci": (
            float(np.percentile(r2_all, alpha)),
            float(np.percentile(r2_all, 100 - alpha)),
        ),
        "n_successful": n_valid_boot,
    }


# ─────────────────────────────────────────────
# Bootstrap orchestration + refined-results export
# ─────────────────────────────────────────────

def run_bootstrap(x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_test: np.ndarray,
                  y_test: np.ndarray,
                  expression: str,
                  weights_original: np.ndarray,
                  var_count: Optional[int] = None,
                  n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                  num_trials: int = 5,
                  n_jobs: int = 1,
                  seed: Optional[int] = None,
                  ci_level: float = DEFAULT_CI_LEVEL,
                  verbose: bool = False) -> dict:
    """Bootstrap a fixed expression and summarise its uncertainty.

    Resamples the training set ``n_bootstrap`` times (element-safe splits),
    re-optimises the element weights on each resample via NLopt, then reports
    prediction dispersion and metric confidence intervals on the training and
    evaluation arrays.

    Returns a dict with ``expression``, ``weights_original``, per-sample
    predictions/statistics, ``metrics_train``/``metrics_test`` (MAE, RMSE, R²
    with CIs), ``weights_all`` and ``success_mask``.
    """
    weights_original = np.asarray(weights_original, dtype=np.float64)
    if var_count is None:
        var_count = weights_original.shape[0]

    if verbose:
        print(f"  Bootstrapping {n_bootstrap} resamples "
              f"({num_trials} NLopt trials each, n_jobs={n_jobs})...")

    (y_pred_all_train, y_pred_all_test,
     weights_all, success_mask) = bootstrap_resample(
        x_train, y_train, x_test, y_test,
        expression, weights_original, var_count,
        n_bootstrap=n_bootstrap,
        random_seed=seed,
        num_trials=num_trials,
        n_jobs=n_jobs,
    )

    n_success = int(success_mask.sum())
    if n_success == 0:
        raise RuntimeError(
            "All bootstrap iterations failed - check the expression/weights."
        )

    mean_train, std_train, lo_train, hi_train = compute_confidence_intervals(
        y_pred_all_train, ci_level=ci_level
    )
    mean_test, std_test, lo_test, hi_test = compute_confidence_intervals(
        y_pred_all_test, ci_level=ci_level
    )

    return {
        "expression": expression,
        "weights_original": weights_original.tolist(),
        "var_count": int(var_count),
        "n_bootstrap": int(n_bootstrap),
        "n_successful": n_success,
        "ci_level": float(ci_level),
        "metrics_train": compute_bootstrap_metrics(y_train, y_pred_all_train,
                                                   ci_level=ci_level),
        "metrics_test": compute_bootstrap_metrics(y_test, y_pred_all_test,
                                                  ci_level=ci_level),
        "prediction_stats": {
            "train": {"mean": mean_train.tolist(), "std": std_train.tolist(),
                      "ci_lower": lo_train.tolist(), "ci_upper": hi_train.tolist()},
            "test": {"mean": mean_test.tolist(), "std": std_test.tolist(),
                     "ci_lower": lo_test.tolist(), "ci_upper": hi_test.tolist()},
        },
        "weights_all": weights_all.tolist(),
        "success_mask": success_mask.tolist(),
    }


def write_refined_results(bootstrap_result: dict,
                          output_path: "str | Path") -> Path:
    """Write a bootstrap result as a one-entry refined-results JSON.

    Generalizes ``use_bootstrap_best.py``: the file produced here is consumed
    by ``cwsr-query``, ``analysis.inverse`` and ``analysis.pareto``.
    """
    metrics_train = bootstrap_result.get("metrics_train", {})
    metrics_test = bootstrap_result.get("metrics_test", {})

    def _pick(metrics, key, fallback=0.0):
        """Accept both ``<key>_mean`` (bootstrap) and ``<key>``."""
        return metrics.get(f"{key}_mean", metrics.get(key, fallback))

    refined = [{
        "expression": bootstrap_result["expression"],
        "weights": bootstrap_result.get("weights_original") or [],
        "mae": _pick(metrics_train, "mae"),
        "mae_std": metrics_train.get("mae_std", 0.0),
        "mae_ci": list(metrics_train.get("mae_ci", (0.0, 0.0))),
        "rmse": _pick(metrics_train, "rmse"),
        "rmse_std": metrics_train.get("rmse_std", 0.0),
        "r2": _pick(metrics_train, "r2"),
        "r2_std": metrics_train.get("r2_std", 0.0),
        "r2_ci": list(metrics_train.get("r2_ci", (0.0, 0.0))),
        "mae_valid": _pick(metrics_test, "mae"),
        "mae_valid_std": metrics_test.get("mae_std", 0.0),
        "mae_valid_ci": list(metrics_test.get("mae_ci", (0.0, 0.0))),
        "rmse_valid": _pick(metrics_test, "rmse"),
        "rmse_valid_std": metrics_test.get("rmse_std", 0.0),
        "r2_valid": _pick(metrics_test, "r2"),
        "r2_valid_std": metrics_test.get("r2_std", 0.0),
        "r2_valid_ci": list(metrics_test.get("r2_ci", (0.0, 0.0))),
        "n_bootstrap": bootstrap_result.get("n_bootstrap", 0),
        "n_successful": bootstrap_result.get("n_successful", 0),
    }]
    return save_json(output_path, refined)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Bootstrap UQ for a discovered CWSR expression on any dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # bootstrap the best expression of a training run on a bundled alloy database
  python -m analysis.bootstrap --dataset alloy_density \\
      --results gallery/results/alloy_density/cwsr_outputs_alloy_density_*.json \\
      --n_bootstrap 50 --jobs 4 --output results/bootstrap_density.json

  # also export a refined-results file for query / inverse / pareto
  python -m analysis.bootstrap --dataset alloy_density --results <json> \\
      --refined_output results/refined_results_density.json
        """,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name registered in `datasets` (e.g. alloy_density, "
             "matbench_glass) or a path to a processed .npz file.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="CWSR results JSON (cwsr_outputs_*.json or refined results).",
    )
    parser.add_argument(
        "--expr_idx",
        type=int,
        default=0,
        help="Which entry of --results to bootstrap (default: 0 = best)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory holding the *.npz databases (alloy datasets only)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold (Matbench datasets only, default: 0)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples used as the bootstrap/training pool (default: 0.8)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap resamples (default: {DEFAULT_N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=5,
        help="NLopt weight-optimisation trials per resample (default: 5)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers (-1 = all CPUs; default: 1)",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help=f"Confidence level in percent (default: {DEFAULT_CI_LEVEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the bootstrap JSON (default: bootstrap_<stem>.json "
             "next to --results)",
    )
    parser.add_argument(
        "--refined_output",
        type=str,
        default=None,
        help="Also write a one-entry refined-results JSON consumable by "
             "cwsr-query / analysis.inverse / analysis.pareto",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        default=False,
        help="Only print the summary; do not write any JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    # Lazy imports: the library above stays usable without `datasets`.
    import datasets
    from cwsr.data import split_dataset

    # ── Load the dataset (registered name or .npz path) ──
    kwargs = {}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.dataset.startswith("matbench_"):
        kwargs["fold"] = args.fold
    ds = datasets.get_dataset(args.dataset, **kwargs)

    train_idx, eval_idx = split_dataset(ds.compositions, ds.targets,
                                        ratio=args.split_ratio, seed=args.seed)
    x_train, y_train = ds.compositions[train_idx], ds.targets[train_idx]
    x_eval, y_eval = ds.compositions[eval_idx], ds.targets[eval_idx]

    # ── Load the expression to bootstrap ──
    expression, weights = load_expression_from_results(args.results, args.expr_idx)

    print("=" * 60)
    print("BOOTSTRAP UNCERTAINTY QUANTIFICATION")
    print("=" * 60)
    print(f"  Dataset   : {ds.name} ({ds.n_samples} samples: "
          f"{len(train_idx)} bootstrap / {len(eval_idx)} evaluation)")
    print(f"  Results   : {args.results} [expr_idx={args.expr_idx}]")
    print(f"  Expression: {expression}")
    print(f"  Resamples : {args.n_bootstrap} (trials={args.num_trials}, "
          f"jobs={args.jobs}, CI={args.ci_level:g}%)")
    print()

    result = run_bootstrap(
        x_train, y_train, x_eval, y_eval,
        expression, weights,
        n_bootstrap=args.n_bootstrap,
        num_trials=args.num_trials,
        n_jobs=args.jobs,
        seed=args.seed,
        ci_level=args.ci_level,
        verbose=args.verbose,
    )

    for label, metrics in (("bootstrap", result["metrics_train"]),
                           ("evaluation", result["metrics_test"])):
        print(f"  [{label}] MAE  {metrics['mae_mean']:.4g} "
              f"(CI {metrics['mae_ci'][0]:.4g} - {metrics['mae_ci'][1]:.4g})")
        print(f"  [{label}] RMSE {metrics['rmse_mean']:.4g} "
              f"(std {metrics['rmse_std']:.3g})")
        print(f"  [{label}] R2   {metrics['r2_mean']:.4f} "
              f"(CI {metrics['r2_ci'][0]:.4f} - {metrics['r2_ci'][1]:.4f})")
    print(f"  Successful resamples: {result['n_successful']}/{result['n_bootstrap']}")

    if args.no_save:
        print("\n(--no_save: nothing written)")
        return 0

    out = Path(args.output) if args.output else (
        Path(args.results).parent / f"bootstrap_{Path(args.results).stem}.json"
    )
    save_json(out, result)
    print(f"\nBootstrap result saved to: {out}")

    if args.refined_output:
        refined_path = write_refined_results(result, args.refined_output)
        print(f"Refined results saved to:  {refined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

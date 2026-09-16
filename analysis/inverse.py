#!/usr/bin/env python3
"""
Optimize: Composition Optimization with Discovered Expressions
===============================================================
Given discovered symbolic expressions (from CWSR), find optimal compositions.

Ported from the reference ``Alloys-SR`` project (``inverse.py``): the numerics
are unchanged, only the imports and paths are generalized to this framework
(the forward helpers come from :mod:`cwsr.predict.forward`, and expressions are
read from any CWSR results JSON). Run as ``cwsr-inverse`` or
``python -m analysis.inverse``.

Two modes:
  Mode 1 — Target (--target):
      Find a composition that achieves a specific target property value.
      y = f(W @ comp) → target

  Mode 2 — Weighted-sum (--obj_weights):
      Find a composition that optimizes a weighted sum of multiple properties.
      minimize  sum_i obj_weights[i] * f_i(composition)
      where obj_weights[i] > 0 means maximize f_i, obj_weights[i] < 0 means minimize f_i.

Usage:
    # Mode 1: Target optimization (single property)
    python -m analysis.inverse --refined_results density:refined_results_density.json --target 8.0

    # Mode 2: Weighted-sum (maximize hardness, minimize density)
    python -m analysis.inverse \\
        --refined_results hardness:refined_results_hardness.json \\
        --refined_results density:refined_results_density.json \\
        --obj_weights 1.0 -1.0

    # Constrain to specific elements
    python -m analysis.inverse --refined_results density:refined_results_density.json --target 8.0 \\
        --elements 22 23 24 25 26 27 28 29 40 41 42 72 73 74
"""

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from cwsr.predict.forward import (
    compile_expression,
    compile_gradient_functions,
    predict,
    predict_and_gradient,
    _build_composition,
    _get_active_indices,
    ELEMENT_SYMBOLS,
    composition_to_formula,
    format_composition,
)

from analysis._common import (
    describe_elements,
    load_expression_from_results,
    resolve_active_indices,
    save_json,
)


# =========================================================================
# Objective Functions — Mode 1: Target
# =========================================================================

def _objective_target(comp_flat, weights, expression_func, target, active_indices):
    """Minimize |prediction - target|."""
    comp = _build_composition(comp_flat, active_indices)
    pred = predict(comp, weights, expression_func)
    return float(np.abs(pred - target))


def _objective_target_smooth(comp_flat, weights, expression_func, target, active_indices):
    """Minimize (prediction - target)^2. Smooth for gradient-based optimization."""
    comp = _build_composition(comp_flat, active_indices)
    pred = predict(comp, weights, expression_func)
    return float((pred - target) ** 2)


def _objective_target_smooth_grad(comp_flat, weights, expression_func, grad_funcs,
                                   target, active_indices):
    """Gradient of (prediction - target)^2 w.r.t. comp_flat."""
    comp = _build_composition(comp_flat, active_indices)
    pred, grad = predict_and_gradient(comp, weights, expression_func, grad_funcs)

    x_sq = comp_flat ** 2
    sum_sq = x_sq.sum()

    grad_c = grad[active_indices]
    grad_x = 2.0 * comp_flat / sum_sq * (
        grad_c - np.dot(grad_c, comp[active_indices])
    )

    return 2.0 * (pred - target) * grad_x


# =========================================================================
# Objective Functions — Mode 2: Weighted-Sum (original, kept for backward compat)
# =========================================================================

def _objective_weighted_sum(comp_flat, element_weights_list, expression_funcs,
                             obj_weights, active_indices):
    """
    Maximize: sum_i obj_weights[i] * f_i(composition)
    obj_weights[i] > 0 → maximize f_i, obj_weights[i] < 0 → minimize f_i.

    Returns negative of the weighted sum so that scipy's minimize
    actually maximizes the intended objective.
    """
    comp = _build_composition(comp_flat, active_indices)
    total = 0.0
    for j in range(len(expression_funcs)):
        pred = predict(comp, element_weights_list[j], expression_funcs[j])
        total += obj_weights[j] * pred
    return float(-total)


def _objective_weighted_sum_grad(comp_flat, element_weights_list, expression_funcs,
                                  grad_funcs_list, obj_weights, active_indices):
    """Gradient of negated weighted-sum objective w.r.t. comp_flat."""
    comp = _build_composition(comp_flat, active_indices)

    x_sq = comp_flat ** 2
    sum_sq = x_sq.sum()

    total_grad = np.zeros(len(comp_flat))

    for j in range(len(expression_funcs)):
        _, grad = predict_and_gradient(
            comp, element_weights_list[j], expression_funcs[j], grad_funcs_list[j]
        )
        grad_c = grad[active_indices]
        grad_x = 2.0 * comp_flat / sum_sq * (
            grad_c - np.dot(grad_c, comp[active_indices])
        )
        total_grad += obj_weights[j] * grad_x

    return -total_grad


# =========================================================================
# Objective Functions — Mode 3: Augmented Weighted Tchebycheff
# =========================================================================
# The weighted-sum method fails for non-convex Pareto fronts (e.g., when
# the model has 1/sqrt(Max(...)) terms that create "holes" in the landscape).
# The augmented weighted Tchebycheff method can handle non-convex fronts by
# minimizing the weighted max deviation from an ideal point:
#
#   minimize  max_i [ w_i * |f_i(x) - z_i*| ]  +  rho * sum_i w_i * |f_i(x) - z_i*|
#
# where:
#   z_i* = ideal point (utopian objective values)
#   w_i  = scalarization weights (positive, sum to 1)
#   rho  = small augmentation parameter (e.g., 1e-6)
#
# For obj_weights[i] > 0 (maximize f_i): z_i* = f_i_max (upper bound)
# For obj_weights[i] < 0 (minimize f_i): z_i* = f_i_min (lower bound)

def _objective_tchebycheff(comp_flat, element_weights_list, expression_funcs,
                            scalar_weights, obj_weights, ideal_point, active_indices,
                            rho=1e-6):
    """
    Augmented weighted Tchebycheff objective.
    
    scalar_weights : np.ndarray, shape (n_obj,)
        Positive scalarization weights (sum to 1) for the Tchebycheff metric.
    obj_weights : np.ndarray, shape (n_obj,)
        Signed objective weights: positive → maximize, negative → minimize.
    ideal_point : np.ndarray, shape (n_obj,)
        Utopian point: for maximize objectives, this is the max value;
        for minimize objectives, this is the min value.
    rho : float
        Augmentation parameter (very small, e.g. 1e-6). Keeps the metric
        smooth near the ideal point without pulling solutions away from
        the true Pareto front.
    """
    comp = _build_composition(comp_flat, active_indices)
    
    # Compute property values
    vals = np.zeros(len(expression_funcs))
    for j in range(len(expression_funcs)):
        vals[j] = predict(comp, element_weights_list[j], expression_funcs[j])
    
    # Compute normalized deviations from ideal point
    # For maximize (obj_weights[j] > 0): deviation = ideal - val (we want val close to ideal)
    # For minimize (obj_weights[j] < 0): deviation = val - ideal
    deviations = np.zeros(len(expression_funcs))
    for j in range(len(expression_funcs)):
        if obj_weights[j] > 0:  # maximize
            deviations[j] = ideal_point[j] - vals[j]
        else:  # minimize
            deviations[j] = vals[j] - ideal_point[j]
    
    # Ensure non-negative deviations (ideal point should be utopian)
    deviations = np.maximum(deviations, 0.0)
    
    # Weighted Tchebycheff: max_i [w_i * deviation_i] + rho * sum_i [w_i * deviation_i]
    weighted_devs = scalar_weights * deviations
    tchebycheff = np.max(weighted_devs) + rho * np.sum(weighted_devs)
    
    return float(tchebycheff)


def _objective_tchebycheff_grad(comp_flat, element_weights_list, expression_funcs,
                                 grad_funcs_list, scalar_weights, obj_weights,
                                 ideal_point, active_indices, rho=1e-6):
    """
    Gradient of the augmented weighted Tchebycheff objective w.r.t. comp_flat.
    """
    comp = _build_composition(comp_flat, active_indices)
    
    x_sq = comp_flat ** 2
    sum_sq = x_sq.sum()
    
    # Compute property values and gradients
    vals = np.zeros(len(expression_funcs))
    grads_c = []  # gradients w.r.t. composition (active elements only)
    for j in range(len(expression_funcs)):
        val, grad = predict_and_gradient(
            comp, element_weights_list[j], expression_funcs[j], grad_funcs_list[j]
        )
        vals[j] = val
        grad_c = grad[active_indices]
        # Transform gradient from composition space to comp_flat space
        grad_x = 2.0 * comp_flat / sum_sq * (
            grad_c - np.dot(grad_c, comp[active_indices])
        )
        grads_c.append(grad_x)
    
    # Compute deviations and their gradients
    # For maximize (obj_weights[j] > 0): deviation = ideal - val, grad_dev = -grad_val
    # For minimize (obj_weights[j] < 0): deviation = val - ideal, grad_dev = grad_val
    deviations = np.zeros(len(expression_funcs))
    dev_grads = []
    for j in range(len(expression_funcs)):
        if obj_weights[j] > 0:  # maximize
            deviations[j] = ideal_point[j] - vals[j]
            dev_grads.append(-grads_c[j])
        else:  # minimize
            deviations[j] = vals[j] - ideal_point[j]
            dev_grads.append(grads_c[j])
    
    deviations = np.maximum(deviations, 0.0)
    
    # Find which objective gives the max weighted deviation
    weighted_devs = scalar_weights * deviations
    max_idx = np.argmax(weighted_devs)
    
    # Gradient of Tchebycheff term: gradient of max_i [w_i * deviation_i]
    # This is w_max * grad_deviation_max (where max is attained)
    # If multiple objectives attain the max, we use subgradient (pick one)
    total_grad = scalar_weights[max_idx] * dev_grads[max_idx]
    
    # Gradient of augmentation term: rho * sum_i [w_i * deviation_i]
    for j in range(len(expression_funcs)):
        if deviations[j] > 0:
            total_grad += rho * scalar_weights[j] * dev_grads[j]
    
    return total_grad


# =========================================================================
# Optimization Routines — Mode 1: Target
# =========================================================================

def optimize_target(
    weights: np.ndarray,
    expression_func,
    target: float,
    grad_funcs: Optional[list] = None,
    active_elements: Optional[List[int]] = None,
    n_trials: int = 50,
    method: str = "L-BFGS-B",
    verbose: bool = False,
) -> Tuple[np.ndarray, float, float]:
    """
    Mode 1: Optimize composition to achieve a target property value.

    Parameters
    ----------
    weights : np.ndarray, shape (var_count, 118)
        Tabulated weights from CWSR.
    expression_func : callable
        Compiled expression function.
    target : float
        Target property value.
    grad_funcs : list of callable, optional
        Gradient functions. If provided, uses L-BFGS-B with analytical gradients.
    active_elements : list of int, optional
        1-based atomic numbers of allowed elements.
    n_trials : int
        Number of random restarts.
    method : str
        'L-BFGS-B' (gradient-based) or 'Nelder-Mead' (gradient-free).
    verbose : bool
        Whether to print progress.

    Returns
    -------
    composition : np.ndarray, shape (118,)
        Optimized composition vector.
    predicted : float
        Predicted property value.
    error : float
        Absolute error |predicted - target|.
    """
    active_indices = _get_active_indices(active_elements)
    n_active = len(active_indices)

    best_comp = None
    best_error = float("inf")
    best_pred = None

    if verbose:
        print(f"  Active elements: {n_active}")
        print(f"  Optimization trials: {n_trials}")

    for trial in range(n_trials):
        alpha = np.random.uniform(0.1, 5.0)
        init_comp = np.random.dirichlet(np.full(n_active, alpha))

        if method == "L-BFGS-B" and grad_funcs is not None:
            def obj_func(x):
                return _objective_target_smooth(
                    x, weights, expression_func, target, active_indices,
                )

            def grad_func(x):
                return _objective_target_smooth_grad(
                    x, weights, expression_func, grad_funcs, target, active_indices,
                )

            # Bounds on comp_flat: keep values reasonable since they get squared
            # in _build_composition.
            bounds = [(0.0, 10.0)] * n_active

            try:
                res = minimize(
                    obj_func, init_comp,
                    method="L-BFGS-B",
                    jac=grad_func,
                    bounds=bounds,
                    options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8},
                )
                opt_flat = res.x
            except Exception:
                continue
        else:
            res = minimize(
                _objective_target, init_comp,
                args=(weights, expression_func, target, active_indices),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-12},
            )
            opt_flat = res.x

        comp = _build_composition(opt_flat, active_indices)
        pred = predict(comp, weights, expression_func)
        error = float(np.abs(pred - target))

        if error < best_error:
            best_error = error
            best_comp = comp.copy()
            best_pred = pred

        if verbose and (trial + 1) % 10 == 0:
            print(f"    Trial {trial + 1}/{n_trials}: best error = {best_error:.6g}")

    return best_comp, best_pred, best_error


# =========================================================================
# Optimization Routines — Mode 2: Weighted-Sum
# =========================================================================

def optimize_weighted_sum(
    element_weights_list: list,
    expression_funcs: list,
    grad_funcs_list: list,
    obj_weights: np.ndarray,
    active_indices: np.ndarray,
    n_multistart: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mode 2: Optimize a weighted-sum scalarization for a single set of objective weights.

    Parameters
    ----------
    element_weights_list : list of np.ndarray
        List of CWSR weight matrices for each property.
    expression_funcs : list of callable
        List of compiled expression functions.
    grad_funcs_list : list of list of callable
        List of gradient function lists for each property.
    obj_weights : np.ndarray, shape (n_obj,)
        Signed objective weights: positive → maximize, negative → minimize.
    active_indices : np.ndarray, shape (n_active,)
        Indices of active elements.
    n_multistart : int
        Number of random starting points.
    seed : int, optional
        Random seed.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    composition : np.ndarray, shape (118,)
        Optimized composition vector.
    values : np.ndarray, shape (n_obj,)
        Property values at the optimized composition.
    """
    n_active = len(active_indices)
    rng = np.random.default_rng(seed)

    best_comp = None
    best_fun = np.inf
    best_vals = None

    for trial in range(n_multistart):
        alpha = rng.uniform(0.1, 5.0, size=n_active)
        x0 = rng.dirichlet(alpha)

        def obj_func(x):
            return _objective_weighted_sum(
                x, element_weights_list, expression_funcs,
                obj_weights, active_indices,
            )

        def grad_func(x):
            return _objective_weighted_sum_grad(
                x, element_weights_list, expression_funcs, grad_funcs_list,
                obj_weights, active_indices,
            )

        # Bounds on comp_flat: keep values reasonable since they get squared
        # in _build_composition. comp_flat^2 / sum(comp_flat^2) gives the
        # composition fraction, so comp_flat in [0, 10] is more than enough.
        bounds = [(0.0, 10.0)] * n_active

        try:
            result = minimize(
                obj_func, x0,
                method="L-BFGS-B",
                jac=grad_func,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-6},
            )
        except Exception:
            continue

        if result.fun < best_fun:
            best_fun = result.fun
            x_opt = result.x
            comp_opt = _build_composition(x_opt, active_indices)

            vals_opt = np.zeros(len(expression_funcs))
            for j in range(len(expression_funcs)):
                vals_opt[j] = predict(comp_opt, element_weights_list[j], expression_funcs[j])

            best_comp = comp_opt.copy()
            best_vals = vals_opt.copy()

    if best_comp is None:
        return np.zeros(118), np.full(len(expression_funcs), np.nan)

    return best_comp, best_vals


# =========================================================================
# Optimization Routines — Mode 3: Augmented Weighted Tchebycheff
# =========================================================================

def optimize_tchebycheff(
    element_weights_list: list,
    expression_funcs: list,
    grad_funcs_list: list,
    obj_weights: np.ndarray,
    active_indices: np.ndarray,
    n_multistart: int = 10,
    seed: Optional[int] = None,
    verbose: bool = False,
    rho: float = 1e-6,
    scalar_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mode 3: Optimize using augmented weighted Tchebycheff scalarization.

    This method can handle non-convex Pareto fronts where the weighted-sum
    method fails. It minimizes the weighted max deviation from an ideal point:

        minimize  max_i [ w_i * deviation_i ]  +  rho * sum_i [ w_i * deviation_i ]

    where deviation_i = |f_i(x) - z_i*| is the distance from the ideal point.

    Parameters
    ----------
    element_weights_list : list of np.ndarray
        List of CWSR weight matrices for each property.
    expression_funcs : list of callable
        List of compiled expression functions.
    grad_funcs_list : list of list of callable
        List of gradient function lists for each property.
    obj_weights : np.ndarray, shape (n_obj,)
        Signed objective weights: positive → maximize, negative → minimize.
    active_indices : np.ndarray, shape (n_active,)
        Indices of active elements.
    n_multistart : int
        Number of random starting points.
    seed : int, optional
        Random seed.
    verbose : bool
        Whether to print progress.
    rho : float
        Augmentation parameter (very small positive value, e.g. 1e-6).
        Keeps the Tchebycheff metric smooth near the ideal point without
        pulling solutions away from the true Pareto front.
    scalar_weights : np.ndarray, shape (n_obj,), optional
        Positive scalarization weights (sum to 1) for the Tchebycheff metric.
        If None, derived from |obj_weights| / sum(|obj_weights|).

    Returns
    -------
    composition : np.ndarray, shape (118,)
        Optimized composition vector.
    values : np.ndarray, shape (n_obj,)
        Property values at the optimized composition.
    """
    n_obj = len(expression_funcs)
    n_active = len(active_indices)
    rng = np.random.default_rng(seed)

    # ── Step 1: Compute the ideal point via single-objective optimization ──
    if verbose:
        print("  Computing ideal point (single-objective optima)...")

    ideal_point = np.zeros(n_obj)
    for j in range(n_obj):
        # Create a weight vector that only optimizes objective j
        single_weight = np.zeros(n_obj)
        single_weight[j] = obj_weights[j]

        _, vals = optimize_weighted_sum(
            element_weights_list, expression_funcs, grad_funcs_list,
            single_weight, active_indices,
            n_multistart=n_multistart,
            seed=rng.integers(0, 2**31) if seed is not None else None,
            verbose=False,
        )

        if np.any(np.isnan(vals)):
            if verbose:
                print(f"    WARNING: Single-objective optimization failed for objective {j}")
            ideal_point[j] = 0.0
        else:
            ideal_point[j] = vals[j]

        if verbose:
            arrow = "↑" if obj_weights[j] > 0 else "↓"
            print(f"    Objective {j} {arrow}: {ideal_point[j]:.6g}")

    # ── Step 2: Optimize using Tchebycheff scalarization ──
    if scalar_weights is None:
        scalar_weights = np.abs(obj_weights) / np.abs(obj_weights).sum()

    if verbose:
        print(f"  Scalarization weights: {scalar_weights}")
        print(f"  Ideal point: {ideal_point}")

    best_comp = None
    best_fun = np.inf
    best_vals = None

    for trial in range(n_multistart):
        alpha = rng.uniform(0.1, 5.0, size=n_active)
        x0 = rng.dirichlet(alpha)

        def obj_func(x):
            return _objective_tchebycheff(
                x, element_weights_list, expression_funcs,
                scalar_weights, obj_weights, ideal_point, active_indices, rho,
            )

        def grad_func(x):
            return _objective_tchebycheff_grad(
                x, element_weights_list, expression_funcs, grad_funcs_list,
                scalar_weights, obj_weights, ideal_point, active_indices, rho,
            )

        bounds = [(0.0, 10.0)] * n_active

        try:
            result = minimize(
                obj_func, x0,
                method="L-BFGS-B",
                jac=grad_func,
                bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-6},
            )
        except Exception:
            continue

        if result.fun < best_fun:
            best_fun = result.fun
            x_opt = result.x
            comp_opt = _build_composition(x_opt, active_indices)

            vals_opt = np.zeros(len(expression_funcs))
            for j in range(len(expression_funcs)):
                vals_opt[j] = predict(comp_opt, element_weights_list[j], expression_funcs[j])

            best_comp = comp_opt.copy()
            best_vals = vals_opt.copy()

    if best_comp is None:
        return np.zeros(118), np.full(len(expression_funcs), np.nan)

    return best_comp, best_vals


# =========================================================================
# Main CLI
# =========================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimize: Composition Optimization with Discovered Expressions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Target optimization (single property)
  python -m analysis.inverse --refined_results density:refined_results_density.json --target 8.0

  # Mode 2: Weighted-sum (maximize hardness, minimize density)
  python -m analysis.inverse \\
      --refined_results hardness:refined_results_hardness.json \\
      --refined_results density:refined_results_density.json \\
      --obj_weights 1.0 -1.0

  # Constrain to specific elements
  python -m analysis.inverse --refined_results density:refined_results_density.json --target 8.0 \\
      --elements 22 23 24 25 26 27 28 29 40 41 42 72 73 74
        """,
    )

    parser.add_argument(
        "--refined_results",
        type=str,
        action="append",
        required=True,
        dest="refined_results_list",
        help="Property definitions in format: name:path_to_json "
             "e.g., density:refined_results_density.json "
             "Use multiple times for multiple objectives (Mode 2).",
    )
    parser.add_argument(
        "--expr_idx",
        type=int,
        default=0,
        help="Index of the expression in each refined_results to use "
             "(default: 0 = best)",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Elements to include: atomic numbers (e.g., 22) or chemical symbols "
             "(e.g., Ti). Mixed types allowed. (default: all 118)",
    )

    # Mode 1: target
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Target property value to optimize for (Mode 1).",
    )

    # Mode 2: weighted-sum
    parser.add_argument(
        "--obj_weights",
        type=float,
        nargs="+",
        default=None,
        help="Signed objective weights: positive → maximize, negative → minimize. "
             "One per property (Mode 2).",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random restart optimization trials (default: 10; "
             "recommend 20+ for better convergence)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="L-BFGS-B",
        choices=["Nelder-Mead", "L-BFGS-B"],
        help="Optimization method (default: L-BFGS-B; uses analytical gradients)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Minimum fraction to include in formula output (default: 0.001)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=8,
        help="Number of top elements to display (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # ── Parse property definitions ──
    properties = []
    for entry in args.refined_results_list:
        parts = entry.split(":")
        if len(parts) != 2:
            parser.error(
                f"Invalid format: '{entry}'. Expected 'name:path'"
            )
        name, path_str = parts
        properties.append({"name": name, "path": path_str})

    n_obj = len(properties)

    # ── Determine mode ──
    mode_target = args.target is not None
    mode_weighted = args.obj_weights is not None

    if mode_target and mode_weighted:
        parser.error("Cannot specify both --target and --obj_weights. Choose one mode.")
    if not mode_target and not mode_weighted:
        parser.error("Must specify either --target (Mode 1) or --obj_weights (Mode 2).")

    if mode_target:
        if n_obj != 1:
            parser.error("Mode 1 (--target) requires exactly one property.")
        if args.method == "L-BFGS-B":
            print("Note: L-BFGS-B with analytical gradients will be used for Mode 1.")
    else:
        if n_obj < 1:
            parser.error("Mode 2 (--obj_weights) requires at least 1 property.")
        if len(args.obj_weights) != n_obj:
            parser.error(
                f"Number of obj_weights ({len(args.obj_weights)}) must match "
                f"number of objectives ({n_obj})"
            )

    print("=" * 60)
    if mode_target:
        print("OPTIMIZATION: Target Value")
    else:
        print("OPTIMIZATION: Weighted-Sum")
    print("=" * 60)
    for j, p in enumerate(properties):
        print(f"  {p['name']}")
        print(f"    File: {p['path']}")
    if mode_weighted:
        obj_weights = np.array(args.obj_weights, dtype=np.float64)
        for j, p in enumerate(properties):
            arrow = "↑ maximize" if obj_weights[j] > 0 else "↓ minimize"
            print(f"    Objective: {arrow}")
    # ── Determine active element indices ──
    active_indices = resolve_active_indices(args.elements)
    print(f"Active elements ({len(active_indices)}): "
          f"{describe_elements(active_indices)}")
    print()

    # ── Load and compile expressions ──
    expressions = []
    expression_strs = []
    element_weights_list = []

    try:
        for p in properties:
            expression, element_weights = load_expression_from_results(
                p["path"], args.expr_idx
            )
            expressions.append(
                compile_expression(expression, element_weights.shape[0])
            )
            expression_strs.append(expression)
            element_weights_list.append(element_weights)

            if args.verbose:
                print(f"  Loaded '{p['name']}': {expression} "
                      f"(element_weights: {element_weights.shape[0]}"
                      f"×{element_weights.shape[1]})")
    except (FileNotFoundError, IndexError) as exc:
        print(f"Error: {exc}")
        return

    # ── Compile gradient functions ──
    print("Compiling analytical gradient functions...")
    grad_funcs_list = []
    for j, p in enumerate(properties):
        var_count = element_weights_list[j].shape[0]
        grads = compile_gradient_functions(expression_strs[j], var_count)
        grad_funcs_list.append(grads)
        if args.verbose:
            print(f"  Compiled {len(grads)} gradient functions for '{p['name']}'")

    # ── Run optimization ──
    if mode_target:
        # ── Mode 1: Target optimization ──
        target = args.target
        print(f"\n{'='*60}")
        print(f"TARGET: {properties[0]['name']} = {target}")
        print(f"{'='*60}")

        comp, pred, error = optimize_target(
            weights=element_weights_list[0],
            expression_func=expressions[0],
            target=target,
            grad_funcs=grad_funcs_list[0] if args.method == "L-BFGS-B" else None,
            active_elements=args.elements,
            n_trials=args.trials,
            method=args.method,
            verbose=args.verbose,
        )

        if comp is None:
            print(f"  WARNING: Optimization failed for target {target}.")
            return

        formula = composition_to_formula(comp, threshold=args.threshold)
        formatted = format_composition(comp, top_n=args.top_n, threshold=args.threshold)

        print(f"\n  Optimized Composition:")
        print(f"    Formula: {formula}")
        print(f"    Top elements: {formatted}")
        print(f"    Predicted: {pred:.6g}")
        print(f"    Target:    {target}")
        print(f"    Error:     {error:.6g}")

        if args.output:
            output_data = {
                "metadata": {
                    "mode": "target",
                    "property": properties[0]["name"],
                    "target": float(target),
                    "file": properties[0]["path"],
                    "active_elements": args.elements,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "composition": {
                    "formula": formula,
                    "vector": comp.tolist(),
                    "detail": [
                        {"element": ELEMENT_SYMBOLS[Z], "atomic_number": Z + 1,
                         "fraction": float(comp[Z])}
                        for Z in range(118) if comp[Z] > args.threshold
                    ],
                },
                "result": {
                    "predicted": float(pred),
                    "target": float(target),
                    "error": float(error),
                },
            }
            output_path = save_json(args.output, output_data)
            print(f"\nResults saved to: {output_path}")

    else:
        # ── Mode 2: Weighted-sum optimization ──
        obj_weights = np.array(args.obj_weights, dtype=np.float64)

        print(f"\nOptimizing with obj_weights: "
              f"{dict(zip([p['name'] for p in properties], obj_weights))}")

        comp, vals = optimize_weighted_sum(
            element_weights_list, expressions, grad_funcs_list,
            obj_weights, active_indices,
            n_multistart=args.trials,
            seed=args.seed,
            verbose=args.verbose,
        )

        if np.any(np.isnan(vals)):
            print("  WARNING: Optimization failed.")
            return

        formula = composition_to_formula(comp, threshold=args.threshold)
        formatted = format_composition(comp, top_n=args.top_n, threshold=args.threshold)

        print(f"\n  Optimized Composition:")
        print(f"    Formula: {formula}")
        print(f"    Top elements: {formatted}")
        for j in range(n_obj):
            arrow = "↑" if obj_weights[j] > 0 else "↓"
            print(f"    {properties[j]['name']} {arrow}: {vals[j]:.6g}")

        if args.output:
            output_data = {
                "metadata": {
                    "mode": "weighted_sum",
                    "objectives": [
                        {"name": properties[j]["name"],
                         "obj_weight": float(obj_weights[j]),
                         "file": properties[j]["path"]}
                        for j in range(n_obj)
                    ],
                    "active_elements": args.elements,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "composition": {
                    "formula": formula,
                    "vector": comp.tolist(),
                    "detail": [
                        {"element": ELEMENT_SYMBOLS[Z], "atomic_number": Z + 1,
                         "fraction": float(comp[Z])}
                        for Z in range(118) if comp[Z] > args.threshold
                    ],
                },
                "properties": {
                    properties[j]["name"]: {
                        "value": float(vals[j]),
                        "obj_weight": float(obj_weights[j]),
                    }
                    for j in range(n_obj)
                },
            }
            output_path = save_json(args.output, output_data)
            print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

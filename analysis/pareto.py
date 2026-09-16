#!/usr/bin/env python3
"""
Pareto Front Optimization
==========================
Given multiple discovered expressions (from CWSR) for different properties,
find Pareto-optimal compositions by sweeping weighted-sum scalarizations.

The sweep generates Pareto-optimal points by:
  1. Sweeping scalarization weights (w1, w2, ..., wk) over the simplex
  2. For each weight vector, optimizing: minimize sum_i w_i * obj_weights[i] * f_i(comp)
  3. Collecting all optimized points

Objective weights are signed: positive → maximize, negative → minimize.

Ported from the reference ``Alloys-SR`` project (``pareto.py``): the numerics
are unchanged, only the imports and paths are generalized to this framework
(the sweep scalarizes with :func:`analysis.inverse.optimize_tchebycheff` and
reads expressions from any CWSR results JSON). Run as ``cwsr-pareto`` or
``python -m analysis.pareto``.

Usage:
    python -m analysis.pareto \\
        --refined_results hardness:path/to/hardness.json \\
        --refined_results density:path/to/density.json \\
        --obj_weights 1.0 -1.0 \\
        --elements 22 23 24 25 26 27 28 29 40 41 42 72 73 74 \\
        --n_pareto_points 100 --trials 10 --output pareto.json
"""

import argparse
import time
from typing import Optional, Tuple

import numpy as np

from cwsr.predict.forward import (
    compile_expression,
    compile_gradient_functions,
    ELEMENT_SYMBOLS,
    composition_to_formula,
    format_composition,
    _get_active_indices,
)

from analysis.inverse import optimize_tchebycheff

from analysis._common import (
    describe_elements,
    load_expression_from_results,
    resolve_active_indices,
    save_json,
)


# =========================================================================
# Pareto Front via Weighted-Sum Sweep
# =========================================================================

def compute_pareto_front_analytical(
    element_weights_list: list,
    expression_funcs: list,
    grad_funcs_list: list,
    obj_weights: np.ndarray,
    active_indices: np.ndarray,
    n_pareto_points: int = 100,
    seed: Optional[int] = None,
    n_multistart: int = 10,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pareto front by sweeping weighted-sum scalarizations.

    For each set of scalarization weights (one per objective), we optimize:
        minimize  sum_i obj_weights[i] * f_i(composition)
    where obj_weights[i] > 0 means maximize f_i, obj_weights[i] < 0 means minimize f_i.

    For 2 objectives, we sweep a single parameter t in [0, 1]:
        w = [t, 1-t]
    For 3+ objectives, we sample random weight vectors from the simplex.

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
    active_indices : np.ndarray
        0-based indices of active elements.
    n_pareto_points : int
        Number of Pareto-optimal points to generate.
    seed : int, optional
        Random seed for multistart initialization.
    n_multistart : int
        Number of random starting points for each optimization.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    pareto_compositions : np.ndarray, shape (n_pareto, 118)
        Pareto-optimal composition vectors.
    pareto_values : np.ndarray, shape (n_pareto, n_obj)
        Property values at Pareto-optimal points.
    """
    n_obj = len(expression_funcs)
    rng = np.random.default_rng(seed)

    try:
        from tqdm import tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    # Generate weight vectors
    if n_obj == 2:
        weight_vectors = np.column_stack([
            np.linspace(0.0, 1.0, n_pareto_points),
            np.linspace(1.0, 0.0, n_pareto_points),
        ])
    else:
        # weights sampled from (0,1) and normalized to sum to 1
        weight_vectors = rng.dirichlet(np.ones(n_obj), size=n_pareto_points)

    if verbose:
        print(f"  Weighted-sum scalarization: "
              f"{'sweeping' if n_obj == 2 else 'sampling'} "
              f"{n_pareto_points} weight vectors...")
        import sys; sys.stdout.flush()

    all_comps_list = []
    all_vals_list = []

    iterator = tqdm(range(n_pareto_points), desc="  Sweep", unit="pt") \
        if _has_tqdm and verbose else range(n_pareto_points)
    for idx in iterator:
        w = weight_vectors[idx]
        # Use augmented weighted Tchebycheff scalarization instead of
        # weighted-sum to handle non-convex Pareto fronts (e.g., when
        # expressions contain 1/sqrt(Max(...)) terms that create
        # "holes" in the objective landscape).
        # The sweep weight w is passed as scalar_weights to control
        # which region of the Pareto front to explore.
        comp, vals = optimize_tchebycheff(
            element_weights_list, expression_funcs, grad_funcs_list,
            obj_weights, active_indices,
            n_multistart=n_multistart,
            seed=rng.integers(0, 2**31) if seed is not None else None,
            verbose=False,
            scalar_weights=w,
        )

        if np.all(np.isfinite(vals)):
            all_comps_list.append(comp)
            all_vals_list.append(vals)

    if len(all_comps_list) == 0:
        if verbose:
            print("  WARNING: No feasible points found.")
        return np.empty((0, 118)), np.empty((0, n_obj))

    all_comps = np.array(all_comps_list)
    all_vals = np.array(all_vals_list)

    # Remove duplicates
    _, unique_idx = np.unique(np.round(all_vals, decimals=10), axis=0, return_index=True)
    pareto_comps = all_comps[unique_idx]
    pareto_vals = all_vals[unique_idx]

    # ── Filter: keep only non-dominated points ──
    # A point p dominates q if for all objectives:
    #   obj_weights[j] * p[j] >= obj_weights[j] * q[j]   (strictly > for at least one)
    # For maximize (obj_weights[j] > 0): higher p[j] is better
    # For minimize (obj_weights[j] < 0): lower p[j] is better
    # So we compare signed values: obj_weights[j] * val[j]
    n_pts = len(pareto_vals)
    signed_vals = pareto_vals * obj_weights[np.newaxis, :]  # shape (n_pts, n_obj)

    is_dominated = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        if is_dominated[i]:
            continue
        # Check if point i dominates any other point j
        for j in range(i + 1, n_pts):
            if is_dominated[j]:
                continue
            diff = signed_vals[i] - signed_vals[j]
            if np.all(diff >= -1e-10) and np.any(diff > 1e-10):
                # i dominates j
                is_dominated[j] = True
            elif np.all(diff <= 1e-10) and np.any(diff < -1e-10):
                # j dominates i
                is_dominated[i] = True
                break

    n_before = len(pareto_vals)
    pareto_comps = pareto_comps[~is_dominated]
    pareto_vals = pareto_vals[~is_dominated]
    n_after = len(pareto_vals)

    if verbose:
        print(f"  Pareto front points (raw): {n_before}")
        print(f"  Pareto front points (non-dominated): {n_after}")

    return pareto_comps, pareto_vals


# =========================================================================
# Main CLI
# =========================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Pareto Front: Sweep weights to find Pareto-optimal compositions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2-objective: maximize hardness, minimize density
  python -m analysis.pareto \\
      --refined_results hardness:path/to/hardness.json \\
      --refined_results density:path/to/density.json \\
      --obj_weights 1.0 -1.0 \\
      --elements 22 23 24 25 26 27 28 29 40 41 42 72 73 74 \\
      --n_pareto_points 100 --trials 10 --output pareto.json

  # 3-objective: maximize hardness, maximize yield strength, minimize density
  python -m analysis.pareto \\
      --refined_results hardness:path/hardness.json \\
      --refined_results yield_strength:path/ys.json \\
      --refined_results density:path/density.json \\
      --obj_weights 1.0 1.0 -1.0 \\
      --n_pareto_points 200 --trials 20
        """,
    )

    parser.add_argument(
        "--refined_results",
        type=str,
        action="append",
        required=True,
        dest="refined_results_list",
        help="Property definitions in format: name:path_to_json "
             "e.g., hardness:refined_hardness.json "
             "Use multiple times for multiple objectives.",
    )
    parser.add_argument(
        "--obj_weights",
        type=float,
        nargs="+",
        required=True,
        help="Signed objective weights: positive → maximize, negative → minimize. "
             "One per property.",
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
    parser.add_argument(
        "--n_pareto_points",
        type=int,
        default=100,
        help="Number of Pareto-optimal points to generate (default: 100)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of random restart optimization trials per weight set (default: 10; "
             "recommend 20+ for better convergence)",
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
    obj_weights = np.array(args.obj_weights, dtype=np.float64)

    if len(obj_weights) != n_obj:
        parser.error(
            f"Number of obj_weights ({len(obj_weights)}) must match "
            f"number of objectives ({n_obj})"
        )

    print("=" * 60)
    print("PARETO FRONT: Multi-Objective Alloy Design")
    print("=" * 60)
    print(f"Objectives ({n_obj}):")
    for j, p in enumerate(properties):
        arrow = "↑ maximize" if obj_weights[j] > 0 else "↓ minimize"
        print(f"  {p['name']}: {arrow}")
        print(f"    File: {p['path']}")
    if args.elements:
        # accept atomic numbers and/or chemical symbols
        print(f"Active elements: {describe_elements(_get_active_indices(args.elements))}")
    else:
        print(f"Active elements: All 118")
    print()

    # ── Determine active element indices ──
    active_indices = resolve_active_indices(args.elements)

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

    # ── Compute Pareto front via weighted-sum sweep ──
    print(f"\n{'='*60}")
    print("PARETO FRONT VIA WEIGHTED-SUM OPTIMIZATION")
    print(f"{'='*60}")

    pareto_compositions, pareto_values = \
        compute_pareto_front_analytical(
            element_weights_list=element_weights_list,
            expression_funcs=expressions,
            grad_funcs_list=grad_funcs_list,
            obj_weights=obj_weights,
            active_indices=active_indices,
            n_pareto_points=args.n_pareto_points,
            seed=args.seed,
            n_multistart=args.trials,
            verbose=True,
        )

    n_pareto = len(pareto_compositions)
    if n_pareto == 0:
        print("  WARNING: Weighted-sum optimization found no feasible points.")
        return

    # ── Rank Pareto points ──
    vmin = pareto_values.min(axis=0)
    vmax = pareto_values.max(axis=0)
    vrange = vmax - vmin
    vrange[vrange == 0] = 1.0
    normalized = (pareto_values - vmin) / vrange
    for j, w in enumerate(obj_weights):
        if w > 0:  # maximize → invert normalized value
            normalized[:, j] = 1.0 - normalized[:, j]
    scores = normalized.sum(axis=1)
    sorted_idx = np.argsort(scores)

    # ── Display all Pareto-optimal compositions ──
    print(f"\n{'='*60}")
    print(f"All {n_pareto} Pareto-Optimal Compositions")
    print(f"{'='*60}")

    for rank in range(n_pareto):
        idx = sorted_idx[rank]
        comp = pareto_compositions[idx]
        vals = pareto_values[idx]

        formula = composition_to_formula(comp, threshold=args.threshold)
        formatted = format_composition(comp, top_n=8, threshold=args.threshold)

        print(f"\n  Rank {rank + 1}:")
        print(f"    Formula: {formula}")
        print(f"    Top elements: {formatted}")
        for j in range(n_obj):
            arrow = "↑" if obj_weights[j] > 0 else "↓"
            print(f"    {properties[j]['name']} {arrow}: {vals[j]:.6g}")

    # ── Save results ──
    if args.output:
        pareto_entries = []
        for rank in range(n_pareto):
            idx = sorted_idx[rank]
            comp = pareto_compositions[idx]
            vals = pareto_values[idx]

            pareto_entries.append({
                "rank": rank + 1,
                "formula": composition_to_formula(comp, threshold=args.threshold),
                "composition_vector": comp.tolist(),
                "composition_detail": [
                    {"element": ELEMENT_SYMBOLS[Z], "atomic_number": Z + 1,
                     "fraction": float(comp[Z])}
                    for Z in range(118) if comp[Z] > args.threshold
                ],
                "properties": {
                    properties[j]["name"]: {
                        "value": float(vals[j]),
                        "obj_weight": float(obj_weights[j]),
                    }
                    for j in range(n_obj)
                },
            })

        output_data = {
            "metadata": {
                "n_objectives": n_obj,
                "objectives": [
                    {"name": properties[j]["name"],
                     "obj_weight": float(obj_weights[j]),
                     "file": properties[j]["path"]}
                    for j in range(n_obj)
                ],
                "n_pareto_optimal": int(n_pareto),
                "active_elements": args.elements,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "statistics": {
                "property_ranges": {
                    properties[j]["name"]: {
                        "min": float(pareto_values[:, j].min()),
                        "max": float(pareto_values[:, j].max()),
                        "mean": float(pareto_values[:, j].mean()),
                        "std": float(pareto_values[:, j].std()),
                    }
                    for j in range(n_obj)
                },
                "pareto_count": int(n_pareto),
            },
            "pareto_front": pareto_entries,
        }

        output_path = save_json(args.output, output_data)
        print(f"\nResults saved to: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

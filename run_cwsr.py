#!/usr/bin/env python3
"""
Tabulated Invariant Symbolic Regression (TISR) Runner

This script provides a command-line interface to run symbolic regression
experiments using the TISR framework on Matbench datasets.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np

from iMCTS import Regressor
from dataloader import load_matbench, load_matbench_test


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='Run Tabulated Invariant Symbolic Regression (TISR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cwsr.py
  python run_cwsr.py --task matbench_mp_gap --max_expressions 1000
  python run_cwsr.py --ops mul sub add div sqrt --output_dir ./results
        """
    )

    # Dataset parameters
    dataset_group = parser.add_argument_group('Dataset Parameters')
    dataset_group.add_argument(
        '--task',
        type=str,
        default='matbench_expt_gap',
        choices=[
            'matbench_dielectric', 'matbench_expt_gap', 'matbench_expt_is_metal',
            'matbench_glass', 'matbench_jdft2d', 'matbench_log_gvrh',
            'matbench_log_kvrh', 'matbench_mp_e_form', 'matbench_mp_gap',
            'matbench_mp_is_metal', 'matbench_perovskites', 'matbench_phonons',
            'matbench_steels'
        ],
        help='Matbench task to run (default: matbench_expt_gap)'
    )
    dataset_group.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Cross-validation fold number (0-4). Use negative to combine train+test for that fold (default: 0)'
    )
    dataset_group.add_argument(
        '--split_ratio',
        type=float,
        default=0.8,
        help='Train/validation split ratio (0.0-1.0, default: 0.8)'
    )

    # Model architecture parameters
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument(
        '--ops',
        type=str,
        nargs='+',
        default=['mul', 'sub', 'add', 'div', 'sqrt', 'exp', 'log', 'R'],
        help='List of operations to use. Available: add, sub, mul, div, Max, Min, '
             'sqrt, sin, cos, exp, log, tanh. Note: Max, Min must be capitalized.'
    )
    model_group.add_argument(
        '--var_count',
        type=int,
        default=4,
        help='Number of input variables (default: 4)'
    )

    # Search algorithm parameters
    search_group = parser.add_argument_group('Search Algorithm')
    search_group.add_argument(
        '--K',
        type=int,
        default=10,
        help='MCTS exploration parameter (default: 10)'
    )
    search_group.add_argument(
        '--gp_rate',
        type=float,
        default=0.00,
        help='Genetic programming rate (0.0-1.0, default: 0.00)'
    )
    search_group.add_argument(
        '--mutation_rate',
        type=float,
        default=0.2,
        help='Mutation rate (0.0-1.0, default: 0.2)'
    )
    search_group.add_argument(
        '--exploration_rate',
        type=float,
        default=0.2,
        help='Exploration rate (0.0-1.0, default: 0.2)'
    )
    search_group.add_argument(
        '--max_constants',
        type=int,
        default=6,
        help='Maximum number of constants (default: 6)'
    )
    search_group.add_argument(
        '--max_depth',
        type=int,
        default=6,
        help='Maximum expression tree depth (default: 6)'
    )
    search_group.add_argument(
        '--max_expressions',
        type=int,
        default=200,
        help='Maximum number of expressions to evaluate (default: 200)'
    )

    # Optimization parameters
    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument(
        '--optimization_method',
        type=str,
        default='LD_LBFGS',
        help='NLopt optimization method (default: LD_LBFGS)'
    )
    opt_group.add_argument(
        '--num_parallel',
        type=int,
        default=8,
        help='Number of parallel processes (default: 8)'
    )
    opt_group.add_argument(
        '--num_batches',
        type=int,
        default=64,
        help='Number of batches (default: 64)'
    )
    opt_group.add_argument(
        '--num_trials',
        type=int,
        default=1,
        help='Number of experiments in optimization with different initializations. (default: 1)'
    )
    opt_group.add_argument(
        '--lbfgs_upper_bound',
        type=float,
        default=47.0,
        help='Upper bound for LBFGS optimization parameters (default: 47.0)'
    )

    # Runtime parameters
    runtime_group = parser.add_argument_group('Runtime')
    runtime_group.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output (default: True)'
    )
    runtime_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    runtime_group.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save output files (default: .)'
    )
    runtime_group.add_argument(
        '--save_every',
        type=int,
        default=0,
        help='Save intermediate results every N expression evaluations. 0 disables intermediate saving (default: 0)'
    )
    runtime_group.add_argument(
        '--param_file',
        type=str,
        default=None,
        help='JSON file containing parameter overrides. CLI arguments take precedence.'
    )

    return parser


def split_dataset(compositions: np.ndarray, targets: np.ndarray, ratio: float = 0.8, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split dataset into train/validation sets while ensuring that every
    element species present in the validation set also appears in the training set.

    Parameters
    ----------
    compositions : np.ndarray, shape (n_samples, n_features)
        Composition vectors (atomic fractions, one column per element).
    targets : np.ndarray, shape (n_samples,)
        Target values.
    ratio : float, default 0.8
        Target fraction of samples for the training set.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    train_indices : np.ndarray
        Indices for the training set.
    valid_indices : np.ndarray
        Indices for the validation set.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(targets)
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_size = max(1, int(n * ratio))
    train_indices = set(indices[:train_size].tolist())
    valid_indices = set(indices[train_size:].tolist())

    # Boolean mask of which species each sample contains
    has_species = compositions > 0  # shape (n_samples, n_features)

    # Species present in current train / valid
    train_species = set(np.where(has_species[list(train_indices)].sum(axis=0) > 0)[0])
    valid_species = set(np.where(has_species[list(valid_indices)].sum(axis=0) > 0)[0])

    missing_species = valid_species - train_species

    # Greedily move valid samples to train until all missing species are covered
    while missing_species and valid_indices:
        best_idx = None
        best_covered = set()
        # Find the valid sample that covers the most missing species
        for idx in valid_indices:
            covered = set(np.where(has_species[idx])[0]) & missing_species
            if len(covered) > len(best_covered):
                best_idx = idx
                best_covered = covered
            if len(best_covered) == len(missing_species):
                break  # can't do better than covering all remaining

        if best_idx is None or not best_covered:
            break  # no valid sample can help

        valid_indices.remove(best_idx)
        train_indices.add(best_idx)
        missing_species -= best_covered

    return np.array(sorted(train_indices), dtype=int), np.array(sorted(valid_indices), dtype=int)

def check_species_coverage(compositions: np.ndarray, train_indices: np.ndarray, valid_indices: np.ndarray) -> bool:
    """
    Check if every species present in the validation set is also present in the training set.

    Parameters
    ----------
    compositions : np.ndarray, shape (n_samples, n_features)
        Composition vectors (atomic fractions, one column per element).
    train_indices : np.ndarray
        Indices for the training set.
    valid_indices : np.ndarray
        Indices for the validation set.

    Returns
    -------
    bool
        True if coverage is complete, False otherwise.
    """
    has_species = compositions > 0  # shape (n_samples, n_features)

    train_species = set(np.where(has_species[train_indices].sum(axis=0) > 0)[0])
    valid_species = set(np.where(has_species[valid_indices].sum(axis=0) > 0)[0])

    missing_species = valid_species - train_species

    if missing_species:
        print(f"Warning: The following species are present in the validation set but missing from the training set: {missing_species}")
        return False

    return True


def main():
    """Main execution function."""
    # Pre-parse to get param_file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--param_file', type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = create_parser()

    # Load parameters from JSON file if provided
    if pre_args.param_file and Path(pre_args.param_file).exists():
        with open(pre_args.param_file, 'r') as f:
            json_params = json.load(f)
        # Only use keys that correspond to known arguments
        known_args = {action.dest for action in parser._actions}
        filtered_params = {k: v for k, v in json_params.items() if k in known_args}
        parser.set_defaults(**filtered_params)

    args = parser.parse_args(remaining)
    args.param_file = pre_args.param_file

    # Validate output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility as early as possible
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Load dataset
    if args.fold < 0:
        effective_fold = abs(args.fold)
        fold_label = f"{effective_fold} (train+test)"
    else:
        effective_fold = args.fold
        fold_label = str(effective_fold)

    print(f"Loading dataset: {args.task}, fold {fold_label}")

    if effective_fold not in range(5):
        print(f"Error loading dataset: fold must be 0-4 (got {args.fold})")
        return

    try:
        if args.fold < 0:
            train_compositions, train_targets = load_matbench(args.task, effective_fold)
            test_compositions, test_targets = load_matbench_test(
                args.task,
                effective_fold,
                include_target=True,
            )
            compositions = np.concatenate([train_compositions, test_compositions], axis=0)
            targets = np.concatenate([train_targets, test_targets], axis=0)
        else:
            compositions, targets = load_matbench(args.task, effective_fold)

        print(f"Dataset loaded: {compositions.shape[0]} samples, {compositions.shape[1]} features")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize model
    print("Initializing symbolic regression model...")
    
    # split into train/valid ensuring train covers all valid compositions
    train_indices, valid_indices = split_dataset(
        compositions, targets, ratio=args.split_ratio, seed=args.seed
    )
    x_train = compositions[train_indices]
    y_train = targets[train_indices]
    x_valid = compositions[valid_indices]
    y_valid = targets[valid_indices]

    # Build output prefix for intermediate saves
    timestamp = int(time.time())
    output_prefix = str(output_dir / f"cwsr_outputs_{args.task}_fold{effective_fold}_{timestamp}")

    model = Regressor(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        ops=args.ops,
        verbose=args.verbose,
        var_count=args.var_count,
        K=args.K,
        gp_rate=args.gp_rate,
        mutation_rate=args.mutation_rate,
        max_constants=args.max_constants,
        exploration_rate=args.exploration_rate,
        max_depth=args.max_depth,
        max_expressions=args.max_expressions,
        optimization_method=args.optimization_method,
        num_parallel=args.num_parallel,
        num_batches=args.num_batches,
        num_trials=args.num_trials,
        lbfgs_upper_bound=args.lbfgs_upper_bound,
        seed=args.seed,
        save_every=args.save_every,
        output_prefix=output_prefix,
    )

    # Run symbolic regression
    print("Starting symbolic regression search...")
    # sym_exp, vec_exp, evaluations, path, outputs = model.fit()    
    try:
        sym_exp, vec_exp, evaluations, path, outputs = model.fit(seed=args.seed)
    except Exception as e:
        print(f"Error during symbolic regression: {e}")
        return

    # Save outputs
    timestamp = int(time.time())
    output_filename = output_dir / f"cwsr_outputs_{args.task}_fold{effective_fold}_{timestamp}.json"
    param_filename = output_dir / f"cwsr_params_{args.task}_fold{effective_fold}_{timestamp}.json"

    with open(param_filename, 'w') as f:
        json.dump(vars(args), f, indent=2)
        
    try:
        with open(output_filename, 'w') as f:
            json.dump(outputs, f, indent=2)
    except Exception as e:
        print(f"Error saving outputs: {e}")
        return

    # Print results
    print("\n" + "="*60)
    print("SYMBOLIC REGRESSION COMPLETED")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Fold: {fold_label}")
    print(f"Best Expression: {sym_exp}")
    print(f"Evaluations: {evaluations}")
    print(f"Output saved to: {output_filename}")
    print("="*60)


if __name__ == "__main__":
    main()

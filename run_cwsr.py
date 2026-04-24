#!/usr/bin/env python3
"""
Tabulated Invariant Symbolic Regression (TISR) Runner

This script provides a command-line interface to run symbolic regression
experiments using the TISR framework on Matbench datasets.
"""

import argparse
import json
import time
from pathlib import Path

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
  python run_tisr.py
  python run_tisr.py --task matbench_mp_gap --max_expressions 1000
  python run_tisr.py --ops mul sub add div sqrt --output_dir ./results
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

    # Runtime parameters
    runtime_group = parser.add_argument_group('Runtime')
    runtime_group.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output (default: True)'
    )
    runtime_group.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save output files (default: .)'
    )

    return parser


def main():
    """Main execution function."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model = Regressor(
        x_train=compositions,
        y_train=targets,
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
    )

    # Run symbolic regression
    print("Starting symbolic regression search...")
    # sym_exp, vec_exp, evaluations, path, outputs = model.fit()    
    try:
        sym_exp, vec_exp, evaluations, path, outputs = model.fit()
    except Exception as e:
        print(f"Error during symbolic regression: {e}")
        return

    # Save outputs
    timestamp = int(time.time())
    output_filename = output_dir / f"tisr_outputs_{args.task}_fold{effective_fold}_{timestamp}.json"

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

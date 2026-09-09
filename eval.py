import numpy as np
from typing import Tuple, Callable
import nlopt
from sympy import symbols, diff, lambdify
import json
import numba
from cwsr.reward import sp_module
from dataloader import load_matbench, load_matbench_test
import argparse
import matplotlib.pyplot as plt
from visiualize import PeriodicTableVisualizer

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
        choices=range(5),
        help='Cross-validation fold number (0-4, default: 0)'
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
        '--model_path',
        type=str,
        default='.',
        help='Directory to load saved model'
    )
    runtime_group.add_argument(
        '--is_refine',
        action='store_true',
        default=False,
        help='Enable weight refinement (default: False)'
    )

    return parser

def compile_expression(args, expression) -> Callable | None:

    # build expression
    v_len = args.var_count
    vs = [symbols(f'x{i}') for i in range(v_len)]
    xs = vs
    sympy_expr = expression
    f_pred_const = lambdify(xs, sympy_expr, modules=sp_module)
    f_pred_const = jit_compile(f_pred_const)
    return f_pred_const

def jit_compile(func: Callable) -> Callable:
    func_jit = numba.njit(func, fastmath=True)

    #@numba.njit(fastmath=True, cache=True)
    def wrapper(x_bar: np.ndarray) -> np.ndarray:
        out = func_jit(*(x_bar.T))
        return out

    return wrapper

def build_expr_and_gradient(args, expression) -> Tuple[Callable, Callable]:
    """Build symbolic expression and its gradient function.

    Returns
    -------
    (sympy_expr, f_pred, f_grad)
        sympy_expr: a simplified symbolic expression
        f_pred: function that evaluates the expression given input x
        f_grad: function that evaluates the gradient of the expression w.r.t. x
    """
    v_len = args.var_count
    vs = [symbols(f'x{i}') for i in range(v_len)]
    xs = vs
    sympy_expr = expression

    #print(sympy_expr, xs)
    grad_v = [diff(sympy_expr, v) for v in vs]
    grad_v = [g for g in grad_v]

    #print("Building gradients...")        
    grad_v_func = [jit_compile(lambdify(xs, str(grad_v_expr), modules=sp_module)) 
                for grad_v_expr in grad_v] if v_len > 0 else None

    return sympy_expr, grad_v_func

def build_MAELoss_func(args, f_pred_const, grads) -> Callable:
    v_len = args.var_count
    w_len = v_len * 118   # Total length of tabulated weights (118 per variable)
    x_train = args.x_train
    y_train = args.y_train

    # Unpack gradient functions for variables, real constants, and complex constants
    grad_v_func = grads
    #@profile
    def loss_func(p, grad):
        # Reshape the first part of parameters into tabulated weights matrix
        # Shape: (v_len, 118) where v_len is number of variables
        tabulated_weights = np.reshape(p[:w_len], (v_len, 118))

        # Compute weighted sum of input features using tabulated weights
        # x_bar shape: (n_samples, v_len)
        x_bar = np.einsum("ni,vi->nv", x_train, tabulated_weights)

        # Evaluate the symbolic expression with the current parameters
        y_pred = f_pred_const(x_bar)

        if not np.all(np.isfinite(y_pred)):
            return np.inf

        # Compute Mean Absolute Error
        mae = np.mean(np.abs(y_pred - y_train))

        # Compute gradients if gradient array is provided (for gradient-based optimization)
        if grad.size > 0:
            # Compute gradients for tabulated weights (variables)
            for idx, func in enumerate(grad_v_func):
                grad_v = func(x_bar)  # Gradient of expression w.r.t. variable idx

                if not np.all(np.isfinite(grad_v)):
                    return np.inf
                
                # Chain rule: dMAE/dw = mean(sign(y_pred - y_true) * dy_pred/dx_bar * dx_bar/dw)
                # dx_bar/dw involves the input features x_train
                grad[idx*118:(idx+1)*118] = np.mean(
                    (np.sign(y_pred - y_train) * grad_v)[:, None] * x_train,
                    axis=0
                )

        return mae
    return loss_func

def optimize_weights(args, best_expr):
    # Optimize for tabulated weights
    initial_guess = np.abs(np.random.randn(args.var_count * 118))+ 1e-3
    n_params = len(initial_guess)
    # Create an NLopt optimizer object with the specified algorithm.
    opt = nlopt.opt(args.optimization_method, n_params)
    # Set the objective function to be maximized. We minimize the negative reward.
    f_pred_const = compile_expression(args, best_expr)
    _, grads = build_expr_and_gradient(args, best_expr)
    object_func = build_MAELoss_func(args, f_pred_const, grads)
    opt.set_min_objective(object_func)

    # Set a relative tolerance for the optimization.
    opt.set_xtol_rel(1e-6)
    # Set a maximum number of evaluations to prevent infinite loops.
    opt.set_maxeval(100)
    # --- Start: Set bounds for the optimization parameters ---
    # Set a uniform lower and upper bound for all parameters.
    # This is equivalent to the `bounds` parameter in `differential_evolution`.
    bounds = np.array([-47.0] * n_params)
    opt.set_lower_bounds(bounds)
    opt.set_upper_bounds(-bounds) # Using -bounds to get [10.0, 10.0, ...]
    # --- End: Set bounds ---

    tabulated_weights = np.reshape(
        opt.optimize(initial_guess),
        (args.var_count, 118)
    )

    return tabulated_weights

def parity_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def main():
    parser = create_parser()
    args = parser.parse_args()

    x_train, y_train = load_matbench(args.task, fold=args.fold)
    x_test, y_test = load_matbench_test(args.task, fold=args.fold, include_target=True)
    x_test = np.concatenate([x_train, x_test], axis=0)
    y_test = np.concatenate([y_train, y_test], axis=0)
    args.x_train = x_train
    args.y_train = y_train
    args.x_test = x_test
    args.y_test = y_test

    is_presents = np.sum(x_train, axis=0) > 0

    # Check if model_path is a file path or JSON string
    try:
        # Try to load as JSON first
        data = json.loads(args.model_path)
    except json.JSONDecodeError:
        # If it's not valid JSON, treat it as a file path
        try:
            with open(args.model_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {args.model_path} not found.")
            return
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {args.model_path}: {e}")
            return

    best_expr = data[0]['expression']
    weights = data[0]['weights']

    if args.is_refine or weights is None:
        print("Refining weights...")
        weights = optimize_weights(args, best_expr)

    f_pred = compile_expression(args, best_expr)
    #print(x_test.shape, weights.shape)
    x_bar = np.einsum("ni,vi->nv", x_test, weights)
    
    y_pred = f_pred(x_bar)
    mae = np.mean(np.abs(y_pred - y_test))
    rate = np.sum(np.abs(y_pred - y_test) < 0.5) / len(y_test)
    print(f"Test accuracy: {rate*100:.2f}%")
    print(f"Test MAE: {mae:.6f}")
    # if y_test is not None:
    #     parity_plot(y_test, y_pred)
    ptable = PeriodicTableVisualizer()
    weights = np.array(weights).T
    ptable.plot_table(str(best_expr), weights, is_presents)
    ptable.show()

if __name__ == "__main__":
    main()
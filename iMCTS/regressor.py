import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sympy import sympify, expand, expand_log
import time
import random
from iMCTS.mcts import MCTS
from iMCTS.src import ExpTree, Optimizer
from iMCTS.gp import GPManager
import nlopt
from sympy import symbols, diff, lambdify
import gc
from iMCTS.src.utils.reward import jit_compile, sp_module

def simplify_expression(exp_str: str, verbose: bool = False) -> str:
    """Simplify mathematical expression string without relying on class methods."""
    try:
        cleaned = exp_str.replace("[", "").replace("]", "")
        expr = sympify(cleaned)
        expanded = expand(expand_log(expr, force=True))
        return str(expanded)
    except Exception as e:
        if verbose:
            print(f"Simplification failed: {e}")
        return exp_str

class Regressor:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        var_count: None,
        ops: List[str] = None,
        arity_dict: Dict[str, int] = None,
        context: Dict = None,
        complexity: Dict=None,
        max_depth: int = 6,
        K: int = 500,
        c: float = 4.0,
        gamma: float = 0.5,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.1,
        exploration_rate: float = 0.2,
        max_single_arity_ops: int = 999,
        max_constants: int = 10,
        max_expressions: float = 2e6,
        verbose: bool = False,
        reward_func: Optional[Callable] = None,
        optimization_method: str = 'LN_NELDERMEAD',
        num_parallel: int = 4,
        num_batches: int = 16,
        num_trials: int = 1,
    ):
        """
        Symbolic Regression Regressor with MCTS optimization

        Parameters:
        x_train (np.ndarray): Training data features of shape (n_features, n_samples)
        y_train (np.ndarray): Training data labels of shape (n_samples,)
        seed (int, optional): Random seed for reproducibility
        """
        # Input validation
        self._validate_inputs(x_train, y_train, max_depth)

        # Initialize core components
        self.x_train = x_train
        self.y_train = y_train
        self.verbose = verbose
        self.reward_func = reward_func
        self.current_best_expr = None
        self.current_y_pred = None
        self.tabulated_weights = None

        # Initialize operations and context
        if not var_count:
            var_count = x_train.shape[1]
        self.var_count = var_count
        self.ops = self._init_operations(ops, var_count)
        self.arity_dict = self._init_arity_dict(arity_dict, var_count)
        self.complexity = self._init_complexity(complexity, var_count)
        self.global_context = self._init_context(context)
        self.num_trials = num_trials
        # Initialize optimization parameters
        self._init_optimization_params(
            max_depth,
            K,
            c,
            gamma,
            gp_rate,
            mutation_rate,
            exploration_rate,
            max_single_arity_ops,
            max_constants,
            max_expressions,
            num_parallel,
            num_batches,
            num_trials
        )

        self.sigma = float(np.std(y_train)) if y_train is not None else 1.0
        # Initialize core components
        self.optimizer = Optimizer(
            var_count,
            x_train,
            y_train,
            self.sigma,
            self.global_context,
            reward_func,
            optimization_method
        )
        
        self.exp_tree = self._create_exp_tree()

    def fit(self, seed: int = None) -> Tuple[str, str, int, int, List[Dict]]:
        """Perform symbolic regression search"""
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        with np.errstate(all='ignore'):
            mcts = self._create_mcts()
            self.start_time = time.time()
            outputs = self.find_best(mcts)
            exp_str, reward = mcts.exp_queue.best()

            return (
                simplify_expression(exp_str, self.verbose),
                exp_str,
                mcts.count_num,
                mcts.path_queue.best()[0],
                outputs
            )
    
    def find_best(self, mcts: MCTS):
        """Find best expression within given time limit"""
        import time
        search_num = 0
        # Track last report time (store on instance so future extensions can reuse)
        last_report_time = getattr(self, '_last_report_time', self.start_time)
        REPORT_INTERVAL = 60.0  # seconds
        while mcts.count_num < self.max_expressions:
            search_num += 1
            best_reward = mcts.search(self.exp_tree)
            self.print_simple(mcts)
            now = time.time()
            # Time-based periodic status report (every ~10s)
            if self.verbose and (now - last_report_time >= REPORT_INTERVAL):
                self.print_status(mcts)
                last_report_time = now
                self._last_report_time = last_report_time

            # Exit if exceeding 48 hours total runtime
            # if now - self.start_time > 172800:  # 48 hours = 172800 seconds
            #     if self.verbose:
            #         self.print_status(mcts)
            #     break

            # Success criterion reached (reward close enough to 1)
            if 1 - best_reward < mcts.succ_error_tol:
                if self.verbose:
                    self.print_status(mcts)
                break
            gc.collect()
            
        outputs = self.save_status(mcts)
        return outputs

    def build_MAE_loss(self, expr_str: str) -> Callable:
        """Build MAE loss function for given expression string"""
        #print(expr_str)
        x_train = self.x_train
        y_train = self.y_train
        v_len = self.var_count        
        vs = [symbols(f'x{i}') for i in range(self.var_count)]
        f_pred = jit_compile(lambdify(vs, expr_str, modules=sp_module))
        grad_v = [diff(expr_str, v) for v in vs]
        grad_v_func = [
            jit_compile(lambdify(vs, str(grad_v_expr), modules=sp_module)) 
            for grad_v_expr in grad_v
            ] if self.var_count > 0 else None

        def mae_loss(params: np.ndarray, grad) -> float:
            tabulated_weights = np.reshape(params, (v_len, 118))
            x_bar = np.einsum("ni,vi->nv", x_train, tabulated_weights)
            y_pred = f_pred(x_bar)
            if not np.all(np.isfinite(y_pred)):
                return np.inf

            if grad.size > 0:
                for idx, func in enumerate(grad_v_func):
                    grad_v = func(x_bar)
                    if not np.all(np.isfinite(grad_v)):
                        return np.inf
                    grad[idx*118:(idx+1)*118] = np.mean(
                        (np.sign(y_pred - y_train) * grad_v)[:, None] * x_train, 
                        axis=0
                        )

            mae = np.mean(np.abs(y_pred - y_train))
            return mae
        return mae_loss

    def optimize_weights(self, best_expr, is_positive_init):
        # Optimize for tabulated weights
        if is_positive_init:
            initial_guess = np.abs(np.random.randn(self.var_count * 118)) * np.sqrt(5)
        else:
            initial_guess = np.random.randn(self.var_count * 118) * np.sqrt(5)
        n_params = len(initial_guess)
        # Create an NLopt optimizer object with the specified algorithm.
        opt = nlopt.opt(self.optimizer.optimization_method, n_params)
        # Set the objective function to be maximized. We minimize the negative reward.
        object_func = self.build_MAE_loss(str(best_expr))
        opt.set_min_objective(object_func)

        # Set a relative tolerance for the optimization.
        opt.set_xtol_rel(1e-6)
        # Set a maximum number of evaluations to prevent infinite loops.
        opt.set_maxeval(200)
        # --- Start: Set bounds for the optimization parameters ---
        # Set a uniform lower and upper bound for all parameters.
        # This is equivalent to the `bounds` parameter in `differential_evolution`.
        bounds = np.array([-47.0] * n_params)
        opt.set_lower_bounds(bounds)
        opt.set_upper_bounds(-bounds) # Using -bounds to get [10.0, 10.0, ...]
        # --- End: Set bounds ---

        optimized_params = opt.optimize(initial_guess)
        tabulated_weights = np.reshape(
            optimized_params,
            (self.var_count, 118)
        )
        y_pred = object_func(optimized_params, np.array([]))  # Get predictions after optimization

        return y_pred, tabulated_weights

    def print_simple(self, mcts) -> None:
        best_expr, best_reward = mcts.exp_queue.best()
        mae = (1/best_reward-1) * self.sigma if best_reward > 0 else float('inf')
        report = [
            "\n\033[1;36m=== Symbolic Regression Progress Report ===\033[0m",
            f"\033[1mEvaluated Expressions:\033[0m {mcts.count_num}",
            "\n\033[1mTop Performance Metrics:\033[0m",
            f"  \033[32mBest Expression:\033[0m \n{best_expr}",
            f"  \033[33mReward (↑):\033[0m {best_reward:.3e} | MAE (↓): {mae:.3e}",
            "\n\033[1mExploration Profile:\033[0m",
            f"  Total Nodes: {mcts.total_nodes} | Active Branches: {len(mcts.root.children)}",
            f"  Exploration Rate: {self.exploration_rate:.2f} | Mutation Rate: {self.mutation_rate:.2f}"
        ]
        print("\n".join(report))

    def save_status(self, mcts) -> List[Dict]:
        """Save and return the top 10 expressions with their optimized weights and metrics"""
        outputs = []
        for i in range(min(10, len(mcts.exp_queue.list))):
            # Get the top expressions
            best_expr, best_reward = mcts.exp_queue.list[i]
            y_true = self.y_train
            entry = {}

            min_mae = float('inf')
            weights = None
            y_pred = None
            for i in range((self.num_trials + 4) * 2):
                is_positive_init = (i < self.num_trials + 4)
                try:
                    y_pred, tabulated_weights = self.optimize_weights(best_expr, is_positive_init)
                    mae = np.mean(np.abs(y_pred - y_true))
                    if mae < min_mae:
                        min_mae = mae
                        weights = tabulated_weights
                except Exception:
                    #print(f"Weight optimization failed for expression {best_expr}: {e}")
                    continue
                    
            # Update instance variables for the best expression
            self.current_best_expr = best_expr
            self.current_y_pred = y_pred
            self.tabulated_weights = weights

            # Store results in entry
            entry['expression'] = str(best_expr)  # Ensure it's a string
            entry['weights'] = weights.tolist() if weights is not None else None
            entry['reward'] = float(best_reward)  # Ensure float
            entry['mae'] = float(min_mae)  # Ensure float
            entry['rank'] = i + 1  # Add rank (1-based)

            outputs.append(entry)

        return outputs


    # Initialization helper methods
    def _validate_inputs(self, x_train, y_train, max_depth):
        """Validate input parameters"""
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Mismatched dimensions between x_train and y_train")
            
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

    def _init_operations(self, ops, var_count) -> List[str]:
        """Initialize operations list with variables"""
        default_ops = ['add', 'sub', 'mul', 'div', 'Max', 'Min', 'Pow',
                       'sqrt', 'sin', 'cos', 'exp', 'log']
        ops = ops or default_ops
        return ops + [f'x{i}' for i in range(var_count)]

    def _init_arity_dict(self, arity_dict, var_count) -> Dict[str, int]:
        """Initialize arity dictionary with variables"""
        default_arity = {
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'Max': 2, 'Min': 2, 'Pow': 2,
            'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'tanh': 1,
            'sqrt': 1,
            'R': 0
        }
        arity_dict = arity_dict or default_arity.copy()
        arity_dict.update({f'x{i}': 0 for i in range(var_count)})
        return arity_dict

    def _init_context(self, context) -> Dict:
        """Initialize evaluation context"""
        default_context = {
            'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide,
            'Max': np.maximum, 'Min': np.minimum, 'Pow': np.power,
            'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt,
            'exp': np.exp, 'log': np.log, 'tanh': np.tanh,
        }
        return {**default_context, **(context or {})}

    def _init_complexity(self, complexity: Optional[Dict[str, int]], var_count: int) -> Dict[str, int]:
        """
        Initialize operator complexity values for time-cost estimation.

        Higher complexity values indicate operations that take longer to optimize.
        Used to quickly estimate computational cost during expression evaluation.

        Parameters:
        complexity (Optional[Dict[str, int]]): Custom complexity overrides
        var_count (int): Number of input variables

        Returns:
        Dict[str, int]: Dictionary mapping operators to their complexity values
        """
        default_complexity = {
            # Basic arithmetic (low complexity)
            'add': 1,
            'sub': 1,
            'mul': 2,
            'div': 2,

            # Comparison operations (high complexity due to optimization challenges)
            'Max': 100,
            'Min': 100,
            'Pow': 50,

            # Trigonometric and transcendental functions
            'sin': 2,
            'cos': 2,
            'tanh': 5,
            'sqrt': 4,
            'exp': 10,
            'log': 10,

            # Constants (increase optimization parameters)
            'R': 10,
        }

        # Variables (moderate complexity)
        default_complexity.update({f'x{i}': 10 for i in range(var_count)})

        return {**default_complexity, **(complexity or {})}


    def _init_optimization_params(self, max_depth, K, c, gamma, gp_rate,
                                 mutation_rate, exploration_rate,
                                 max_single_arity_ops, max_constants, max_expressions, num_parallel,
                                 num_batches, num_trials):
        """Initialize optimization parameters with validation"""
        self.max_depth = max_depth
        self.K = K
        self.c = c
        self.gamma = gamma
        self.gp_rate = np.clip(gp_rate, 0.0, 1.0)
        self.mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
        self.exploration_rate = np.clip(exploration_rate, 0.0, 1.0)
        self.max_single_arity_ops = max_single_arity_ops
        self.max_constants = max_constants
        self.max_expressions = max_expressions
        self.num_parallel = num_parallel
        self.num_batches = num_batches
        self.num_trials = num_trials

    def _create_exp_tree(self):
        """Create expression tree instance"""
        T = ExpTree(
            max_depth=self.max_depth,
            max_single_arity_ops=self.max_single_arity_ops,
            max_constants=self.max_constants,
            arity_dict=self.arity_dict,
            complexity=self.complexity,
            ops=self.ops
        )
        return T

    def _create_mcts(self):
        """Dynamically create new MCTS instance"""
        return MCTS(
            optimizer=self.optimizer,
            gp_manager=GPManager(self.ops, self.arity_dict),
            gp_rate=self.gp_rate,
            mutation_rate=self.mutation_rate,
            exploration_rate=self.exploration_rate,
            K=self.K,
            c=self.c,
            gamma=self.gamma,
            num_parallel=self.num_parallel,
            num_batches=self.num_batches,
            num_trials=self.num_trials,
            verbose=self.verbose
        )

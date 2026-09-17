import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sympy import sympify, expand, expand_log
import time
import random
import json
import warnings
from cwsr.mcts import MCTS
from cwsr.exp_tree import ExpTree
from cwsr.reward import Optimizer
from cwsr.gp import GPManager
import gc


#: SymPy function names the engine's op set can represent (name -> engine op).
_FUNCTION_OPS = {"sqrt": "sqrt", "exp": "exp", "log": "log", "sin": "sin",
                 "cos": "cos", "tanh": "tanh", "Abs": "Abs",
                 "Max": "Max", "Min": "Min"}

#: Constructor settings captured by :meth:`Regressor.state_dict` (and restored by
#: :meth:`Regressor.from_state`). ``reward_func`` is deliberately absent (a
#: callable cannot be serialized).
_STATE_CONFIG_KEYS = ("var_count", "ops", "max_depth", "K", "c", "gamma",
                      "gp_rate", "mutation_rate", "exploration_rate",
                      "max_single_arity_ops", "max_constants", "max_expressions",
                      "num_parallel", "num_batches", "num_trials",
                      "lbfgs_upper_bound", "optimization_method", "seed",
                      "verbose", "save_every", "save_checkpoint_every",
                      "output_prefix")


def _is_variable(op: str) -> bool:
    """True for the engine's variable tokens (``x0``, ``x1``, ...)."""
    return op.startswith("x") and op[1:].isdigit()



def _expression_to_path(expression: str) -> List[str]:
    """Convert a symbolic expression into the engine's operator path.

    The returned list is the pre-order operator sequence an
    :class:`~cwsr.exp_tree.ExpTree` consumes (``add``, ``mul``, ``Pow``,
    ``sqrt``, ``x0``, ``R``, ...), which is what the genetic-programming pool
    (:attr:`~cwsr.mcts.MCTS.path_queue`) stores. Numeric literals become the
    optimizable constant token ``R`` (they are re-optimised on the next search),
    ``x{i}`` stays a variable, n-ary sums/products are folded into binary
    ``add``/``mul`` chains, and ``a - b`` / ``a / b`` are expressed as
    ``add``/``mul`` with an ``R`` operand — the constant absorbs the sign or the
    reciprocal, so the shape still fits the engine's operator set.

    Raises ``ValueError`` for nodes the engine cannot represent (unknown
    symbols or functions).
    """
    from sympy import Add, Max, Min, Mul, Number, Pow, Rational, Symbol, sympify
    from sympy.core.function import Function

    def walk(node) -> List[str]:
        if isinstance(node, Number):
            return ["R"]
        if isinstance(node, Symbol):
            name = str(node)
            if name.startswith("x") and name[1:].isdigit():
                return [name]
            if name[:1] in ("R", "C"):      # latent constant symbols
                return ["R"]
            raise ValueError(f"unsupported symbol '{name}'")
        if isinstance(node, Add):
            args = list(node.args)
            ops = walk(args[0])
            for arg in args[1:]:
                ops = ["add"] + ops + walk(arg)
            return ops
        if isinstance(node, Mul):
            args = list(node.args)
            ops = walk(args[0])
            for arg in args[1:]:
                ops = ["mul"] + ops + walk(arg)
            return ops
        if isinstance(node, Pow):
            base, exponent = node.args
            if exponent == Rational(1, 2):
                return ["sqrt"] + walk(base)
            return ["Pow"] + walk(base) + walk(exponent)
        if isinstance(node, (Max, Min)):
            args = list(node.args)
            op = "Max" if isinstance(node, Max) else "Min"
            ops = walk(args[0])
            for arg in args[1:]:
                ops = [op] + ops + walk(arg)
            return ops
        if isinstance(node, Function):
            name = node.func.__name__
            if name not in _FUNCTION_OPS:
                raise ValueError(f"unsupported function '{name}'")
            return [_FUNCTION_OPS[name]] + walk(node.args[0])
        raise ValueError(f"unsupported expression node '{node}'")

    ops = walk(sympify(expression))
    if not ops:
        raise ValueError(f"empty expression: {expression!r}")
    return ops


def _warn_if_no_output_prefix(save_every: int,
                              save_checkpoint_every: int,
                              output_prefix: Optional[str]) -> None:
    """Warn when the periodic-save flags cannot do anything.

    ``save_every`` / ``save_checkpoint_every`` write ``<output_prefix>_step*.json``
    and ``<output_prefix>_ckpt_step*.json`` files, so they need an
    ``output_prefix``. Without one the saves are silently skipped otherwise,
    which looks like the feature is broken.
    """
    if output_prefix:
        return
    requested = []
    if save_every and save_every > 0:
        requested.append(f"save_every={save_every}")
    if save_checkpoint_every and save_checkpoint_every > 0:
        requested.append(f"save_checkpoint_every={save_checkpoint_every}")
    if requested:
        warnings.warn(
            f"{' and '.join(requested)} requested but output_prefix is not set: "
            "no intermediate results or checkpoints will be written. Pass "
            "output_prefix='<path/prefix>' (or use cwsr.model.fit_dataset, which "
            "sets it) to persist them.",
            UserWarning,
            stacklevel=3,
        )

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
        x_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        var_count: int = None,
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
        lbfgs_upper_bound: float = 47.0,
        seed: int = None,
        save_every: int = 0,
        save_checkpoint_every: int = 0,
        output_prefix: str = None,
        warm_start: Optional[List[Dict]] = None,
        run_meta: Optional[Dict] = None,
    ):
        """
        Symbolic Regression Regressor with MCTS optimization

        Parameters:
        x_train (np.ndarray): Training data features of shape (n_features, n_samples)
        y_train (np.ndarray): Training data labels of shape (n_samples,)
        seed (int, optional): Random seed for reproducibility
        output_prefix (str, optional): Where to persist run artifacts. ``fit`` writes
            ``<output_prefix>_final.json`` (ranked results) and
            ``<output_prefix>_ckpt_final.json`` (resumable MCTS state) at the end of
            every run, plus the periodic ``*_step<N>.json`` / ``*_ckpt_step<N>.json``
            files when the two flags below are set. Without it nothing is written.
        save_every (int, optional): Write ranked results every N evaluated
            expressions (requires ``output_prefix``).
        save_checkpoint_every (int, optional): Write a resumable MCTS checkpoint
            every N evaluated expressions (requires ``output_prefix``).
        warm_start (list[dict], optional): Previously discovered results (the
            ``outputs`` list of an earlier run) to seed the new search with. Their
            expressions are queued as the starting champions and, when the shape
            can be rebuilt, their operator paths also join the genetic-programming
            pool. See :meth:`from_results`.
        run_meta (dict, optional): Free-form provenance recorded in
            :meth:`state_dict` / the final checkpoint (e.g. dataset name).
        """
        # Input validation
        self._validate_inputs(x_train, y_train, max_depth)
        if x_valid is not None and y_valid is not None:
            self._validate_inputs(x_valid, y_valid, max_depth)

        # Initialize core components
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
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
        # Store LBFGS bound for weight optimization
        self.lbfgs_upper_bound = lbfgs_upper_bound
        # Set seed in __init__ so any pre-fit randomness is also reproducible
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Intermediate-save settings
        self.save_every = save_every
        self.save_checkpoint_every = save_checkpoint_every
        self.output_prefix = output_prefix
        # Persistence / restart bookkeeping
        self.optimization_method = optimization_method
        self.run_meta = dict(run_meta or {})
        #: Loaded results (list of dicts with expression/rewards) used as a warm start.
        self.warm_start: List[Dict] = list(warm_start or [])
        #: Previous results attached by :meth:`load` / :meth:`from_state`.
        self.results: Optional[List[Dict]] = None
        #: MCTS state attached by :meth:`load`; pass it to ``fit(checkpoint=...)``.
        self.checkpoint: Optional[Dict] = None
        # Initialize core components
        self.optimizer = Optimizer(
            var_count,
            x_train,
            y_train,
            x_valid,
            y_valid,
            self.sigma,
            self.global_context,
            reward_func,
            optimization_method,
            self.lbfgs_upper_bound
        )
        
        self.exp_tree = self._create_exp_tree()

    def fit(self, seed: int = None, checkpoint: dict = None) -> Tuple[str, str, int, int, List[Dict]]:
        """Perform symbolic regression search

        Persistence (only when ``output_prefix`` is set):
          * ``<output_prefix>_step<N>.json``      — ranked results every ``save_every`` expressions
          * ``<output_prefix>_ckpt_step<N>.json`` — resumable checkpoint every
            ``save_checkpoint_every`` expressions: MCTS state *plus* the full run state
            (hyperparameters/config, data, provenance) under the ``regressor`` key; ranked
            results are not duplicated here (they are in the sibling ``_step<N>.json``)
          * ``<output_prefix>_final.json``        — the finished run's ranked results
          * ``<output_prefix>_ckpt_final.json``   — the finished run's MCTS state *plus* a full
            ``regressor`` state (config + data + results + provenance), so the run can be rebuilt
            and restarted with :meth:`load` / :meth:`resume`.

        Restarting:
          * ``fit(checkpoint=...)`` continues an MCTS state (see :meth:`resume`);
          * a model built by :meth:`from_results` (or a state carrying ``results``)
            warm-starts the new search with the previous champions.

        Without ``output_prefix`` nothing is written (a ``UserWarning`` is raised
        if the save flags were requested).
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        _warn_if_no_output_prefix(self.save_every, self.save_checkpoint_every,
                                  self.output_prefix)

        with np.errstate(all='ignore'):
            mcts = self._create_mcts()
            if checkpoint is not None:
                from cwsr.checkpoint import mcts_from_dict
                mcts_from_dict(checkpoint, mcts)
                print(f"[Checkpoint] Resumed MCTS from {mcts.count_num} evaluations")
            elif self.warm_start:
                self._seed_from_warm_start(mcts)
                print(f"[Warm start] Seeded {len(mcts.exp_queue)} expression(s) / "
                      f"{len(mcts.path_queue)} path(s) from loaded results")
            self.start_time = time.time()
            outputs = self.find_best(mcts)
            exp_str, _, _ = mcts.exp_queue.best()
            self._save_final(mcts, outputs)

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
        last_saved_interval = 0
        last_checkpoint_interval = 0
        while mcts.count_num < self.max_expressions:
            search_num += 1
            best_reward = mcts.search(self.exp_tree)
            self.print_simple(mcts)

            # Intermediate save based on expression count
            if self.save_every > 0 and self.output_prefix:
                current_interval = mcts.count_num // self.save_every
                if current_interval > last_saved_interval:
                    self._maybe_save_intermediate(mcts)
                    last_saved_interval = current_interval

            # Checkpoint save based on expression count
            if self.save_checkpoint_every > 0 and self.output_prefix:
                current_ckpt_interval = mcts.count_num // self.save_checkpoint_every
                if current_ckpt_interval > last_checkpoint_interval:
                    self._maybe_save_checkpoint(mcts)
                    last_checkpoint_interval = current_ckpt_interval
                    
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

    def build_MAE_loss(self, expr_str: str, X, Y) -> Callable:
        """Build MAE loss function for given expression string using Optimizer methods."""
        class MockState:
            constant_count = 0
            real_constant_count = 0

        state = MockState()
        # Use a non-zero dummy guess for the validity check to avoid log(0) etc.
        # ensure a positive guess to avoid issues with sqrt, log, etc. during validity check
        dummy_guess = np.abs(np.random.randn(self.var_count * 118) * np.sqrt(5))
        f_pred_const = self.optimizer.valid_expression(expr_str, state, dummy_guess)
        if f_pred_const is None:
            raise ValueError(f"Expression produced non-finite values: {expr_str}")

        _, grads = self.optimizer.build_expr_and_gradient(expr_str, state)
        return self.optimizer.build_MAELoss_func(state, X, Y, f_pred_const, grads)

    def optimize_weights(self, best_expr, is_positive_init):
        # Optimize for tabulated weights
        if is_positive_init:
            initial_guess = np.abs(np.random.randn(self.var_count * 118)) * np.sqrt(5)
        else:
            initial_guess = np.random.randn(self.var_count * 118) * np.sqrt(5)
        object_func = self.build_MAE_loss(str(best_expr), self.x_train, self.y_train)
        optimized_params = self.optimizer.run_nlopt(
            object_func, initial_guess, self.lbfgs_upper_bound, xtol=1e-6, maxeval=200
        )
        tabulated_weights = np.reshape(
            optimized_params,
            (self.var_count, 118)
        )
        mae = object_func(optimized_params, np.array([]))  # Get predictions after optimization
        # If no validation data, use training data for validation MAE as well (to avoid errors)
        if self.x_valid is None or self.y_valid is None:
            object_func_valid = self.build_MAE_loss(str(best_expr), self.x_train, self.y_train)
        else:
            object_func_valid = self.build_MAE_loss(str(best_expr), self.x_valid, self.y_valid)          
        mae_valid = object_func_valid(optimized_params, np.array([]))
        return mae, mae_valid, tabulated_weights

    def print_simple(self, mcts) -> None:
        best_expr, best_train_reward, best_valid_reward = mcts.exp_queue.best()
        train_mae = (1/best_train_reward-1) * self.sigma if best_train_reward > 0 else float('inf')
        valid_mae = (1/best_valid_reward-1) * self.sigma if best_valid_reward > 0 else float('inf')
        report = [
            "\n\033[1;36m=== Symbolic Regression Progress Report ===\033[0m",
            f"\033[1mEvaluated Expressions:\033[0m {mcts.count_num}",
            "\n\033[1mTop Performance Metrics:\033[0m",
            f"  \033[32mBest Expression:\033[0m \n{best_expr}",
            f"  \033[33mTrain Reward (↑):\033[0m {best_train_reward:.3e} | MAE (↓): {train_mae:.3e}",
            f"  \033[33mValid Reward (↑):\033[0m {best_valid_reward:.3e} | MAE (↓): {valid_mae:.3e}",
            "\n\033[1mExploration Profile:\033[0m",
            f"  Total Nodes: {mcts.total_nodes} | Active Branches: {len(mcts.root.children)}",
            f"  Exploration Rate: {self.exploration_rate:.2f} | Mutation Rate: {self.mutation_rate:.2f}"
        ]
        print("\n".join(report))

    def state_dict(self, mcts: Optional[MCTS] = None,
                   outputs: Optional[List[Dict]] = None) -> Dict:
        """Serializable description of this run: config + data + results + meta.

        This is what the checkpoints embed under the ``regressor`` key — the
        periodic ones written by :meth:`_maybe_save_checkpoint` (with
        ``outputs=None``, since ranked results are written to the sibling
        ``_step<N>.json``) and the final one written by :meth:`_save_final` (with
        the ranked results) — and what :meth:`from_state` consumes, so a saved run
        can be rebuilt — and restarted — without re-deriving the hyperparameters.
        """
        config = {
            "var_count": int(self.var_count),
            "ops": [op for op in self.ops if not _is_variable(op)],
            "max_depth": int(self.max_depth),
            "K": int(self.K),
            "c": float(self.c),
            "gamma": float(self.gamma),
            "gp_rate": float(self.gp_rate),
            "mutation_rate": float(self.mutation_rate),
            "exploration_rate": float(self.exploration_rate),
            "max_single_arity_ops": int(self.max_single_arity_ops),
            "max_constants": int(self.max_constants),
            "max_expressions": self.max_expressions,
            "num_parallel": int(self.num_parallel),
            "num_batches": int(self.num_batches),
            "num_trials": int(self.num_trials),
            "lbfgs_upper_bound": float(self.lbfgs_upper_bound),
            "optimization_method": str(self.optimization_method),
            "seed": self.seed,
            "verbose": bool(self.verbose),
            "save_every": int(self.save_every),
            "save_checkpoint_every": int(self.save_checkpoint_every),
            "output_prefix": self.output_prefix,
        }
        data = {
            "x_train": np.asarray(self.x_train).tolist(),
            "y_train": np.asarray(self.y_train).tolist(),
            "x_valid": None if self.x_valid is None else np.asarray(self.x_valid).tolist(),
            "y_valid": None if self.y_valid is None else np.asarray(self.y_valid).tolist(),
        }
        meta = dict(self.run_meta)
        if mcts is not None:
            meta.update({"count_num": int(mcts.count_num),
                         "best_reward": float(mcts.best_reward),
                         "total_nodes": int(mcts.total_nodes)})
        return {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "data": data,
            "results": outputs,
            "meta": meta,
        }

    @classmethod
    def from_state(cls, state: Dict, **overrides) -> "Regressor":
        """Rebuild a :class:`Regressor` from a :meth:`state_dict` payload.

        Keyword ``overrides`` replace individual settings — e.g.
        ``max_expressions=20000`` to keep searching longer, or ``output_prefix=``
        to redirect the artifacts — and may also provide ``x_train``/``y_train``
        for a state that carries no data. ``reward_func`` cannot be serialized,
        so pass it explicitly if the original run used one. Previous results are
        attached as ``model.results`` and used as the warm start.
        """
        config = dict(state.get("config") or {})
        data = dict(state.get("data") or {})
        config.update({key: value for key, value in overrides.items()
                       if key in _STATE_CONFIG_KEYS})

        x_train = overrides.get("x_train", data.get("x_train"))
        y_train = overrides.get("y_train", data.get("y_train"))
        if x_train is None or y_train is None:
            raise ValueError(
                "state carries no training data: pass x_train=/y_train= as "
                "overrides (or use Regressor.from_results with your dataset)"
            )
        x_valid = overrides.get("x_valid", data.get("x_valid"))
        y_valid = overrides.get("y_valid", data.get("y_valid"))

        kwargs = {key: config[key] for key in _STATE_CONFIG_KEYS
                  if key in config and config[key] is not None}
        model = cls(
            x_train=np.asarray(x_train, dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.float64),
            x_valid=None if x_valid is None else np.asarray(x_valid, dtype=np.float64),
            y_valid=None if y_valid is None else np.asarray(y_valid, dtype=np.float64),
            **kwargs,
        )
        model.results = state.get("results")
        if model.results and not model.warm_start:
            model.warm_start = list(model.results)
        return model

    @classmethod
    def load(cls, path, **overrides) -> "Regressor":
        """Load a saved run (checkpoint file **or** already-parsed dict).

        Every checkpoint written by :meth:`fit` — the periodic
        ``<prefix>_ckpt_step<N>.json`` files and the final
        ``<prefix>_ckpt_final.json`` — carries both the MCTS state and the full
        ``regressor`` state, so the returned model is ready to continue::

            model = Regressor.load("results/run_ckpt_final.json")
            model.fit(seed=0, checkpoint=model.checkpoint)   # or Regressor.resume(path)

        Periodic checkpoints deliberately omit the ranked results (they are in the
        sibling ``_step<N>.json``), so ``model.results`` is ``None`` for those and
        the previous champions are carried by the restored MCTS queues instead.

        Raises ``ValueError`` for files that are not checkpoints (a results-only
        JSON has no data) — use :meth:`from_results` for those.
        """
        if isinstance(path, dict):
            payload = path
        else:
            from cwsr.checkpoint import load_checkpoint
            payload = load_checkpoint(path)
        if "regressor" not in payload:
            raise ValueError(
                f"{path!r} contains no 'regressor' state, so it is not a checkpoint "
                "written by Regressor.fit. For a results JSON use "
                "Regressor.from_results(path, x_train=..., y_train=...)."
            )
        model = cls.from_state(payload["regressor"], **overrides)
        model.checkpoint = payload.get("mcts")
        return model

    @classmethod
    def resume(cls, path, seed: Optional[int] = None, **overrides):
        """Load a checkpoint and continue its search.

        Shorthand for ``Regressor.load(path, **overrides).fit(seed=seed,
        checkpoint=model.checkpoint)``; returns the same tuple as :meth:`fit`.
        ``overrides`` can extend the run (``max_expressions=20000``) or redirect
        its artifacts (``output_prefix=...``).
        """
        model = cls.load(path, **overrides)
        return model.fit(seed=seed, checkpoint=model.checkpoint)

    @classmethod
    def from_results(cls, results_path, x_train, y_train,
                     x_valid=None, y_valid=None,
                     warm_start_top_n: int = 3, **overrides) -> "Regressor":
        """Rebuild a model from a saved results JSON and warm-start a new search.

        Results files carry expressions and weights but no data, so the data is
        supplied by the caller (typically the dataset the model was trained on).
        The top ``warm_start_top_n`` entries become the starting champions: they
        are queued as evaluated results and — when their operator shape can be
        rebuilt within ``max_depth``/the op set — their paths also join the
        genetic-programming pool, so the new search mutates the previous law
        instead of starting from nothing.
        """
        with open(results_path, "r") as handle:
            entries = json.load(handle)
        if isinstance(entries, dict):
            entries = [entries]
        warm_start = list(entries[:warm_start_top_n])

        model = cls(
            x_train=np.asarray(x_train, dtype=np.float64),
            y_train=np.asarray(y_train, dtype=np.float64),
            x_valid=None if x_valid is None else np.asarray(x_valid, dtype=np.float64),
            y_valid=None if y_valid is None else np.asarray(y_valid, dtype=np.float64),
            warm_start=warm_start,
            **overrides,
        )
        model.results = entries
        return model

    def _seed_from_warm_start(self, mcts: MCTS) -> None:
        """Seed the queues from :attr:`warm_start` so a loaded model keeps competing.

        Expressions always join the results queue (they then appear in the final
        ranking); operator paths join the genetic-programming pool only when the
        engine can rebuild them within ``max_depth`` and the current op set.
        """
        for entry in self.warm_start:
            expression = entry.get("expression")
            if not expression:
                continue
            train_reward = float(entry.get("train_reward") or 0.0)
            valid_reward = float(entry.get("valid_reward") or 0.0)
            mcts.exp_queue.append(str(expression), train_reward, valid_reward)
            mcts.best_reward = max(mcts.best_reward, valid_reward)

            path = None
            try:
                path = _expression_to_path(str(expression))
            except Exception:
                path = None
            if path and self._path_is_valid(path):
                mcts.path_queue.append(path, train_reward, valid_reward)
            else:
                warnings.warn(
                    f"warm-start expression {expression!r} cannot be rebuilt as an "
                    "operator path (op set / max_depth limits): it is kept as a "
                    "candidate result but will not seed the genetic-programming pool.",
                    UserWarning, stacklevel=3,
                )

    def _path_is_valid(self, path: List[str]) -> bool:
        """Dry-run a path on a fresh tree to be sure the engine can build it."""
        template = ExpTree(max_depth=self.max_depth,
                           max_single_arity_ops=self.max_single_arity_ops,
                           max_constants=self.max_constants,
                           arity_dict=dict(self.arity_dict),
                           complexity=dict(self.complexity),
                           ops=list(self.ops))
        try:
            for op in path:
                template.add_op(op)
            return template.is_terminal()
        except Exception:
            return False

    def _maybe_save_intermediate(self, mcts: MCTS) -> None:
        """Save intermediate results to a JSON file if save_every is enabled."""
        if self.save_every <= 0 or not self.output_prefix:
            return
        outputs = self.save_status(mcts)
        filename = f"{self.output_prefix}_step{mcts.count_num}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(outputs, f, indent=2)
            print(f"  [Intermediate save] {filename}")
        except Exception as e:
            print(f"  [Intermediate save failed] {e}")

    def _maybe_save_checkpoint(self, mcts: MCTS) -> None:
        """Save a resumable checkpoint every ``save_checkpoint_every`` evaluations.

        Like the final checkpoint, the periodic one carries the **full run state**
        under the ``regressor`` key (hyperparameters/config, training data and
        provenance/meta), not just the MCTS state, so ``Regressor.load`` /
        ``Regressor.resume`` can rebuild and continue a run from *any* checkpoint
        file — which is what the tutorial documents.

        Ranked results are deliberately **not** re-computed here (that costs
        ``num_trials``-many weight optimisations per entry); they are written to
        the sibling ``<prefix>_step<N>.json`` by ``save_every``, and the resumed
        search keeps its champions through the restored MCTS queues.
        """
        if self.save_checkpoint_every <= 0 or not self.output_prefix:
            return
        from cwsr.checkpoint import save_checkpoint
        filename = f"{self.output_prefix}_ckpt_step{mcts.count_num}.json"
        try:
            save_checkpoint(filename, mcts, regressor_state=self.state_dict(mcts))
            print(f"  [Checkpoint save] {filename}")
        except Exception as e:
            print(f"  [Checkpoint save failed] {e}")

    def _save_final(self, mcts: MCTS, outputs: List[Dict]) -> None:
        """Persist the finished run: ranked results + a resumable checkpoint.

        Reuses the ``outputs`` already produced by :meth:`find_best`, so no
        additional weight optimisation is performed. No-op when
        ``output_prefix`` is not set (see :func:`_warn_if_no_output_prefix`).
        """
        if not self.output_prefix:
            return
        from cwsr.checkpoint import save_checkpoint

        results_file = f"{self.output_prefix}_final.json"
        checkpoint_file = f"{self.output_prefix}_ckpt_final.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(outputs, f, indent=2)
            print(f"  [Final save] {results_file}")
        except Exception as e:
            print(f"  [Final save failed] {e}")
        try:
            save_checkpoint(checkpoint_file, mcts,
                            regressor_state=self.state_dict(mcts, outputs))
            print(f"  [Final checkpoint] {checkpoint_file}")
        except Exception as e:
            print(f"  [Final checkpoint failed] {e}")

    def save_status(self, mcts) -> List[Dict]:
        """Save and return the top 10 expressions with their optimized weights and metrics"""
        outputs = []
        for i in range(min(10, len(mcts.exp_queue.list))):
            # Get the top expressions
            best_expr, train_reward, best_reward = mcts.exp_queue.list[i]
            entry = {}

            min_mae = float('inf')
            min_mae_valid = float('inf')
            weights = None
            y_pred = None
            for j in range((self.num_trials + 4) * 2):
                is_positive_init = (j < self.num_trials + 4)
                try:
                    mae, mae_valid, tabulated_weights = self.optimize_weights(best_expr, is_positive_init)
                    if mae_valid < min_mae_valid:
                        min_mae = mae
                        min_mae_valid = mae_valid
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
            entry['train_reward'] = float(train_reward)  # Ensure float
            entry['valid_reward'] = float(best_reward)  # Ensure float
            entry['mae'] = float(min_mae)  # Ensure float
            entry['mae_valid'] = float(min_mae_valid)  # Ensure float
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
            verbose=self.verbose,
            seed=self.seed,
        )

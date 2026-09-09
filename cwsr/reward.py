import numpy as np
from typing import Tuple, Callable, Any
import nlopt
from sympy import symbols, diff, lambdify, sympify
from memory_profiler import profile
import numba
import functools
from sympy import Min, Max, Piecewise

@numba.njit
def Heaviside_vec(x):
    out = np.empty_like(x)
    for i in range(x.size):
        if x[i] > 0.0:
            out[i] = 1.0
        elif x[i] < 0.0:
            out[i] = 0.0
        else:
            out[i] = 0.5   # standard convention
    return out

# Custom module that handles Min/Max properly for numba
sp_module = [{
    # SymPy names - use safe functions that work with numba
    "Min": np.minimum,
    "Max": np.maximum,

    # Other functions
    "Heaviside": Heaviside_vec,
    "Pow": np.power,
    "Abs": np.abs,
    "log": np.log,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}, np]

    
def jit_compile(func: Callable) -> Callable:
    func_jit = numba.njit(func, fastmath=True)

    #@numba.njit(fastmath=True, cache=True)
    def wrapper(x_bar: np.ndarray) -> np.ndarray:
        out = func_jit(*(x_bar.T))
        return out

    return wrapper

# def lower_minmax(expr):
#     # Replace Max
#     expr = expr.replace(Max, lambda a, b: Piecewise((a, a >= b), (b, True)))
#     # Replace Min
#     expr = expr.replace(Min, lambda a, b: Piecewise((a, a <= b), (b, True)))
#     return expr


class Optimizer:
    """Optimize symbolic expression constants and compute reward.

    Performance oriented adjustments:
    - Avoid repeated regex compilation inside loops (use str.replace for exact tokens).
    - Use Numba accelerated RMSE-based reward.
    - Early exits for simple / non-optimizable states.
    - Reduce attribute / global lookups in hot sections via local bindings.
    """

    def __init__(self, 
                 var_count: int, 
                 x_train: np.ndarray, 
                 y_train: np.ndarray, 
                 x_valid: np.ndarray = None,
                 y_valid: np.ndarray = None,
                 sigma: float = None, 
                 context: dict[str, Any] = None,
                 reward_func: Callable = None,
                 optimization_method: str = 'LN_NELDERMEAD',
                 lbfgs_upper_bound: float = 47.0,
                 ):
        self.var_count = var_count
        self.lbfgs_upper_bound = lbfgs_upper_bound
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        # Precompute target std (σ). If y_train is None, default to 1.0 to avoid div-by-zero.
        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = float(np.std(y_train)) if y_train is not None else 1.0
        self.context = context
        # Optimization method for constant optimization
        self.optimization_method = getattr(nlopt, optimization_method)

    #@profile
    def optimize_constants(self, state, is_positive_init: False) -> Tuple[str, float]:
        """Optimize constants in the state's expression and compute reward.

        Returns
        -------
        (expression, reward)
            The (possibly) constant-substituted expression and its computed reward.
        """
        opt_max_step, opt_tol = 100, 1e-6
        #opt_max_step, opt_tol = 100, 1e-4
        expression: str = state.get_expression()
        # print(expression, state.constant_count)

        # Quick rejection: if expression already contains invalid tokens, skip optimization entirely.
        if any(tok in expression for tok in ("zoo", "nan", "inf")):
            if state.constant_count > 0:
                return expression, 0.0, 0.0
            return expression, 0.0, 0.0

        grads = None
        f_pred_const = None

        try:
            # Initialization of parameters
            if is_positive_init:
                initial_guess = self.init_positive_parameters(state)
            else:
                initial_guess = self.init_parameters(state)
                
            n_params = len(initial_guess)

            # Validate expression before optimization
            # if test_mae is inf/nan, skip optimization
            # Bulding gradients and evaluating over training set can be costly,
            # so we do a quick check first.
            f_pred_const = self.valid_expression(expression, state, initial_guess)
            if f_pred_const is None:
                return expression, 0.0, 0.0
        
            # Build once (slightly faster than eval of string each param iteration inside minimize)
            # NOTE: context is trusted upstream. If untrusted, this is a code injection risk.
            expression, grads = self.build_expr_and_gradient(expression, state)
            object_func = self.build_MAELoss_func(state, self.x_train, self.y_train, f_pred_const, grads)

            # Run the optimization using the generic NLopt runner.
            optimized_params = self.run_nlopt(object_func, initial_guess, self.lbfgs_upper_bound, xtol=opt_tol, maxeval=opt_max_step)

        # except nlopt.Failure as e:
        #     print(f"NLopt optimization failed: {e}")
        #    return expression, 0.0
        except Exception:
            #print(f"An unexpected error occurred during NLopt optimization: {e}")
            return expression, 0.0, 0.0 # train_reward, valid_reward are both 0 if failed

        # Compile final expression to a single lambda for evaluation.
        try:
            expression = self.compile_expression(expression, state, optimized_params)
        except Exception:
            return expression, 0.0, 0.0 # train_reward, valid_reward are both 0 if failed

        mae = object_func(optimized_params, np.empty(0))
        reward = 1.0 / (1.0 + mae/self.sigma)
        # print(expression, float(reward))
        
        if self.x_valid is None:
            return expression, float(reward), float(reward)
        else:
            object_func = self.build_MAELoss_func(state, self.x_valid, self.y_valid, f_pred_const, grads)
            mae = object_func(optimized_params, np.empty(0))
            reward_valid = 1.0 / (1.0 + mae/self.sigma)
            return expression, float(reward), float(reward_valid)

    def run_nlopt(self, objective_func, initial_guess, bounds, xtol=1e-6, maxeval=200):
        """Generic NLopt optimization runner shared across modules."""
        n_params = len(initial_guess)
        opt = nlopt.opt(self.optimization_method, n_params)
        opt.set_min_objective(objective_func)
        opt.set_xtol_rel(xtol)
        opt.set_maxeval(maxeval)
        bounds_arr = np.array([-bounds] * n_params)
        opt.set_lower_bounds(bounds_arr)
        opt.set_upper_bounds(-bounds_arr)
        return opt.optimize(initial_guess)

    def build_MAELoss_func(self, state, x_train, y_train, f_pred_const, grads) -> Callable:
        """
        Build a Mean Absolute Error (MAE) loss function for NLopt optimization.

        This function constructs a closure that computes the MAE between predicted
        and actual target values, along with gradients for gradient-based optimization.
        The loss function is used to optimize the tabulated weights and constants
        in the symbolic expression during symbolic regression.

        Parameters
        ----------
        state : object
            The current state containing expression and constant information
        f_pred_const : Callable
            Function that evaluates the symbolic expression given input features
        grads : tuple
            Tuple of gradient functions for variables (grad_v_func), real constants (grad_r_func),
            and complex constants (grad_c_func)

        Returns
        -------
        Callable
            A loss function that takes optimization parameters and computes MAE + gradients
        """
        v_len = self.var_count  # Number of input variables
        w_len = v_len * 118     # Total length of tabulated weights (118 per variable)

        # Unpack gradient functions for variables, real constants, and complex constants
        grad_v_func, grad_r_func, grad_c_func = grads
        #@profile
        def loss_func(p, grad):
            """
            Compute MAE loss and gradients for a given parameter vector.

            Parameters
            ----------
            p : np.ndarray
                Flattened parameter vector containing tabulated weights and constants
            grad : np.ndarray
                Output array to store computed gradients (modified in-place)

            Returns
            -------
            float
                Mean Absolute Error between predictions and true values
            """
            # Reshape the first part of parameters into tabulated weights matrix
            # Shape: (v_len, 118) where v_len is number of variables
            tabulated_weights = np.reshape(p[:w_len], (v_len, 118))

            # Compute weighted sum of input features using tabulated weights
            # x_bar shape: (n_samples, v_len)
            x_bar = np.einsum("ni,vi->nv", x_train, tabulated_weights)

            # If there are constants to optimize, append them to the feature matrix
            if state.constant_count > 0:
                constants = np.tile(p[w_len:], (x_bar.shape[0], 1))  # Tile constants for each sample
                x_bar = np.concatenate((x_bar, constants), axis=1)   # Append to features

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

                # Compute gradients for real constants
                if grad_r_func is not None:
                    for idx, func in enumerate(grad_r_func):
                        grad_r = func(x_bar)  # Gradient of expression w.r.t. real constant
                        if not np.all(np.isfinite(grad_r)):
                            return np.inf
                        grad[w_len + idx] = np.mean(np.sign(y_pred - y_train) * grad_r, axis=0)

                # Note: Complex constant gradients are handled separately in parameter optimization
            return mae
        return loss_func

    def compile_expression(self, expression, state, optimized_params) -> Callable:
        """
        Compile the symbolic expression with optimized constants into an executable function.

        This function takes the symbolic expression from the state, substitutes the optimized
        constant values (both real and complex), and compiles it into a JAX-accelerated
        function for fast evaluation during reward computation.

        Parameters
        ----------
        state : object
            The current state containing expression and constant information
        optimized_params : np.ndarray
            Array of optimized parameters containing tabulated weights, real constants,
            and complex constants (real and imaginary parts)

        Returns
        -------
        tuple
            (sympy_expr, f_pred_const) where:
            - sympy_expr: the simplified symbolic expression with constants substituted
            - f_pred_const: JAX-compiled function for fast evaluation of the expression
        """
        v_len = self.var_count      # Number of input variables
        w_len = v_len * 118         # Total length of tabulated weights
        r_len = state.real_constant_count   # Number of real constants
        c_len = state.constant_count - r_len  # Number of complex constants

        # Create symbolic variables for input features and constants
        vs = [symbols(f'x{i}') for i in range(v_len)]
        rs = [symbols(f'R{i}') for i in range(r_len)]
        cs = [symbols(f'C{i}') for i in range(c_len)]
        
        # Convert string expression to symbolic expression
        sympy_expr = sympify(expression)
        #print(expression, sympy_expr)
        # sympy_expr = expression
        
        # Create substitution dictionary for constants
        val_dict = {}
        
        # Handle complex constants (stored as real + imaginary parts in params)
        if c_len:
            # Extract complex constants from optimized parameters
            # Real parts: w_len + r_len to w_len + r_len + c_len
            # Imaginary parts: 118 + r_len + c_len to 118 + r_len + 2*c_len
            complex_constants = (optimized_params[w_len + r_len:w_len + r_len + c_len] + 
                               1j * optimized_params[118 + r_len + c_len:118 + r_len + 2 * c_len])
            for idx, c in enumerate(complex_constants):
                val_dict[cs[idx]] = c
        
        # Handle real constants
        if r_len:
            real_constants = optimized_params[w_len: w_len + r_len]
            for idx, r in enumerate(real_constants):
                val_dict[rs[idx]] = float(r)
        
        # Substitute constants into the symbolic expression
        sympy_expr = sympy_expr.subs(val_dict)
        
        return sympy_expr

    #@profile    
    def valid_expression(self, expression, state, p) -> Callable | None:

        # build expression
        v_len = self.var_count
        r_len = state.real_constant_count
        c_len = state.constant_count - r_len

        vs = [symbols(f'x{i}') for i in range(v_len)]
        rs = [symbols(f'R{i}') for i in range(r_len)]
        cs = [symbols(f'C{i}') for i in range(c_len)]
        xs = vs + rs + cs
        #print(expression)
        #sympy_expr = sympify(expression)
        sympy_expr = expression

        #print("Building functions...")
        #print(sympy_expr, xs, rs)
        f_pred_const = lambdify(xs, sympy_expr, modules=sp_module)
        f_pred_const = jit_compile(f_pred_const)

        # test expression validity
        v_len = self.var_count
        w_len = v_len * 118
        x_train = self.x_train  # Capture as local variable for closure

        tabulated_weights = np.reshape(p[:w_len], (v_len, 118))
        x_bar = np.einsum("ni,vi->nv", x_train, tabulated_weights)

        if state.constant_count > 0:
            constants = np.tile(p[w_len:], (x_bar.shape[0], 1))
            x_bar = np.concatenate((x_bar, constants), axis=1)
        y_pred = f_pred_const(x_bar)
        #print(y_pred.shape, x_bar.shape)
        res = f_pred_const if np.all(np.isfinite(y_pred)) else None
        return res


    def build_expr_and_gradient(self, expression, state) -> Tuple[Callable, Callable]:
        """Build symbolic expression and its gradient function.

        Returns
        -------
        (sympy_expr, f_pred, f_grad)
            sympy_expr: a simplified symbolic expression
            f_pred: function that evaluates the expression given input x
            f_grad: function that evaluates the gradient of the expression w.r.t. x
        """
        v_len = self.var_count
        r_len = state.real_constant_count
        c_len = state.constant_count - r_len

        vs = [symbols(f'x{i}') for i in range(v_len)]
        rs = [symbols(f'R{i}') for i in range(r_len)]
        cs = [symbols(f'C{i}') for i in range(c_len)]
        xs = vs + rs + cs
        #sympy_expr = sympify(expression)
        sympy_expr = expression

        #print(sympy_expr, xs)
        grad_v = [diff(sympy_expr, v) for v in vs]
        grad_r = [diff(sympy_expr, r) for r in rs]
        grad_c = [diff(sympy_expr, c) for c in cs]
        grad_v = [g for g in grad_v]
        grad_r = [g for g in grad_r]
        grad_c = [g for g in grad_c]

        #print("Building gradients...")        
        grad_v_func = [jit_compile(lambdify(xs, str(grad_v_expr), modules=sp_module)) 
                    for grad_v_expr in grad_v] if v_len > 0 else None
        grad_r_func = [jit_compile(lambdify(xs, str(grad_r_expr), modules=sp_module)) 
                    for grad_r_expr in grad_r] if r_len > 0 else None
        grad_c_func = [jit_compile(lambdify(xs, str(grad_c_expr), modules=sp_module)) 
                    for grad_c_expr in grad_c] if c_len > 0 else None

        return sympy_expr, (grad_v_func, grad_r_func, grad_c_func)
    
    def init_parameters(self, state) -> np.ndarray:
        """Initialize parameters for optimization.

        Returns
        -------
        params: np.ndarray
            Flattened array of initial parameters (complex real/imag + real constants + tabulated weights)
        """
        constant_count = state.constant_count
        r_len = state.real_constant_count
        c_len = constant_count - r_len

        if c_len:
            guess_c = np.random.randn(c_len * 2)
        else:
            guess_c = np.empty(0)
        if r_len:
            guess_r = np.random.randn(r_len) * np.sqrt(5)
        else:
            guess_r = np.empty(0)
        # positive initial weights to avoid error in log/sqrt
        guess_w = np.random.randn(118 * self.var_count) * np.sqrt(5)
        initial_guess = np.concatenate((guess_w, guess_r, guess_c))
        return initial_guess

    def init_positive_parameters(self, state) -> np.ndarray:
        """Initialize parameters for optimization.

        Returns
        -------
        params: np.ndarray
            Flattened array of initial parameters (complex real/imag + real constants + tabulated weights)
        """
        constant_count = state.constant_count
        r_len = state.real_constant_count
        c_len = constant_count - r_len

        if c_len:
            guess_c = np.random.randn(c_len * 2)
        else:
            guess_c = np.empty(0)
        if r_len:
            guess_r = np.abs(np.random.randn(r_len)) * np.sqrt(5)
        else:
            guess_r = np.empty(0)
        # positive initial weights to avoid error in log/sqrt
        guess_w = np.abs(np.random.randn(118 * self.var_count)) * np.sqrt(5)
        initial_guess = np.concatenate((guess_w, guess_r, guess_c))
        return initial_guess

"""Unified forward model: evaluation + analytic gradients.

The composition-weighted forward model is::

    y = f(W @ comp)

where ``comp`` is a normalized 118-dim composition vector, ``W`` is the
(var_count, 118) tabulated weight matrix learned by CWSR, and ``f`` is the
symbolic expression in variables ``x0 .. x_{var_count-1}``.

This module is dataset-agnostic: everything is defined in terms of an
expression string, a weight matrix and a composition vector.
"""

from __future__ import annotations

from typing import Callable, List

import numba
import numpy as np
from sympy import diff, lambdify, symbols, sympify

from cwsr.formula import (  # re-exported for convenience
    ELEMENT_SYMBOLS,
    composition_to_formula,
    format_composition,
)

__all__ = [
    "jit_compile",
    "compile_expression",
    "compile_gradient_functions",
    "predict",
    "predict_vector",
    "predict_and_gradient",
    "predict_vector_and_gradients",
    "_get_active_indices",
    "_build_composition",
    "ELEMENT_SYMBOLS",
    "composition_to_formula",
    "format_composition",
]


# =========================================================================
# JIT compilation
# =========================================================================
def jit_compile(func: Callable) -> Callable:
    """Compile ``func`` with Numba for fast column-wise array evaluation.

    The input ``x_bar`` has shape (n, var_count); columns are unpacked as
    positional arguments so it works for any ``var_count >= 1``.
    """
    func_jit = numba.njit(func, fastmath=True)

    def wrapper(x_bar: np.ndarray) -> np.ndarray:
        out = func_jit(*(x_bar.T))
        # Constant expressions (e.g. lambdify returning a scalar 0) must still
        # produce a per-sample array.
        if np.ndim(out) == 0:
            out = np.full(x_bar.shape[0], out)
        return out

    return wrapper


# =========================================================================
# Expression compilation
# =========================================================================
def compile_expression(expression: str, var_count: int) -> Callable:
    """Compile a symbolic expression string into a jitted callable.

    Parameters
    ----------
    expression : str
        Expression in variables ``x0 .. x_{var_count-1}``.
    var_count : int
        Number of input variables (>= 1).

    Returns
    -------
    callable
        ``f(x_bar)`` with ``x_bar`` of shape (n, var_count) -> (n,).
    """
    if var_count < 1:
        raise ValueError(f"var_count must be >= 1, got {var_count}")
    return jit_compile(_lambdify(expression, var_count))


def compile_gradient_functions(expression: str,
                               var_count: int) -> List[Callable]:
    """Compile the analytic partial derivatives df/dx_i for each variable."""
    if var_count < 1:
        raise ValueError(f"var_count must be >= 1, got {var_count}")
    vs = [symbols(f"x{i}") for i in range(var_count)]
    expr = sympify(expression)
    return [jit_compile(lambdify(vs, diff(expr, vs[i]),
                                 modules=_sp_module()))
            for i in range(var_count)]


# =========================================================================
# Forward prediction
# =========================================================================
def _latent(weights: np.ndarray, composition: np.ndarray) -> np.ndarray:
    """x = W @ comp, shape (var_count,)."""
    return np.einsum("vi,i->v", weights, composition)


def predict(composition: np.ndarray,
            weights: np.ndarray,
            expression_func: Callable) -> float:
    """Predicted property value ``f(W @ composition)`` for one composition."""
    x_bar = _latent(weights, composition)
    return float(expression_func(x_bar[np.newaxis, :])[0])


def predict_vector(compositions: np.ndarray,
                   weights: np.ndarray,
                   expression_func: Callable) -> np.ndarray:
    """Vectorized prediction over an (n, 118) array of compositions."""
    x_bar = np.einsum("vi,ni->nv", weights, compositions)
    return expression_func(x_bar)


def _sp_module():
    """Lazy import of the cwsr lambdify module set (avoid circular import)."""
    from cwsr.reward import sp_module
    return sp_module


def _lambdify(expression: str, var_count: int):
    vs = [symbols(f"x{i}") for i in range(var_count)]
    return lambdify(vs, expression, modules=_sp_module())

# =========================================================================
# Gradients
# =========================================================================
def predict_and_gradient(composition: np.ndarray,
                         weights: np.ndarray,
                         expression_func: Callable,
                         grad_funcs: List[Callable],
                         ) -> "tuple[float, np.ndarray]":
    """Value and composition-gradient ``df/dcomp`` (shape (118,))."""
    x_bar_2d = _latent(weights, composition)[np.newaxis, :]
    value = float(expression_func(x_bar_2d)[0])
    grad_x = np.array([g(x_bar_2d)[0] for g in grad_funcs])
    gradient = np.einsum("i,ij->j", grad_x, weights)
    return value, gradient


def predict_vector_and_gradients(compositions: np.ndarray,
                                 weights: np.ndarray,
                                 expression_func: Callable,
                                 grad_funcs: List[Callable],
                                 ) -> "tuple[np.ndarray, np.ndarray]":
    """Vectorized (values, gradients) over an (n, 118) array."""
    x_bar = np.einsum("vi,ni->nv", weights, compositions)
    values = expression_func(x_bar)
    grads = np.stack([g(x_bar) for g in grad_funcs], axis=1)  # (n, var_count)
    return values, np.einsum("nvi,ij->nj", grads, weights)


# =========================================================================
# Composition parameterisation helpers (for constrained optimisation)
# =========================================================================
def _get_active_indices(elements) -> np.ndarray:
    """Map 1-based atomic numbers (``--elements 22 23 ...``) to 0-based columns."""
    arr = np.atleast_1d(np.asarray(elements, dtype=int))
    if arr.ndim != 1:
        raise ValueError("elements must be a flat list of atomic numbers")
    return arr - 1


def _build_composition(comp_flat: np.ndarray,
                       active_indices: np.ndarray) -> np.ndarray:
    """Rebuild a (118,) normalized composition from free parameters.

    ``comp_flat`` are the (unnormalised) fractions on the active columns; the
    returned vector has non-active columns set to zero and active columns
    normalized to sum to one.
    """
    comp = np.zeros(118, dtype=np.float64)
    comp[active_indices] = np.asarray(comp_flat, dtype=np.float64)
    norm = comp[active_indices].sum()
    if norm > 0:
        comp[active_indices] /= norm
    return comp


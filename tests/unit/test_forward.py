"""Unit tests for expression compilation and the supported operator set."""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.predict.forward import (compile_expression,
                                  compile_gradient_functions, jit_compile)


def _xbar(x0, x1=None):
    """Build an ``(n, var_count)`` argument for a compiled expression."""
    if x1 is None:
        return np.asarray(x0, dtype=np.float64).reshape(-1, 1)
    return np.column_stack([np.asarray(x0, dtype=np.float64),
                            np.asarray(x1, dtype=np.float64)])


def test_compile_expression_rejects_invalid_var_count():
    for var_count in (0, -1):
        with pytest.raises(ValueError, match="var_count must be >= 1"):
            compile_expression("x0", var_count)
        with pytest.raises(ValueError, match="var_count must be >= 1"):
            compile_gradient_functions("x0", var_count)


def test_compile_expression_returns_per_sample_array_for_constants():
    """Constant expressions must broadcast to one value per sample."""
    out = compile_expression("5", var_count=1)(_xbar([1.0, 2.0, 3.0]))
    assert out.shape == (3,)
    assert np.allclose(out, 5.0)


@pytest.mark.parametrize("expression,expected", [
    ("x0 + x1", [3.0, 7.0]),
    ("x0 - x1", [-1.0, -1.0]),
    ("x0 * x1", [2.0, 12.0]),
    ("x0 / x1", [0.5, 0.75]),
    ("Max(x0, x1)", [2.0, 4.0]),
    ("Min(x0, x1)", [1.0, 3.0]),
    ("Pow(x0, 2.0)", [1.0, 9.0]),
    ("sqrt(x0)", [1.0, np.sqrt(3.0)]),
    ("exp(x0)", [np.e, np.exp(3.0)]),
    ("log(x0)", [0.0, np.log(3.0)]),
    ("sin(x0)", [np.sin(1.0), np.sin(3.0)]),
    ("cos(x0)", [np.cos(1.0), np.cos(3.0)]),
])
def test_supported_operators_evaluate_consistently(expression, expected):
    """Every operator the search can build must compile and evaluate.

    ``Max``/``Min``/``Pow`` are the capitalised SymPy spellings the expression
    tree emits; the rest are plain NumPy functions (see ``cwsr.reward.sp_module``).
    Inputs are ``x0 = [1, 3]`` and ``x1 = [2, 4]``.
    """
    func = compile_expression(expression, var_count=2)
    out = func(_xbar([1.0, 3.0], [2.0, 4.0]))
    assert out.shape == (2,)
    assert np.allclose(out, np.asarray(expected, dtype=np.float64), rtol=1e-12)


def test_nested_expression_evaluates():
    func = compile_expression("sqrt(x0) + Max(x1, 0.5) * Pow(x0, 2.0)", var_count=2)
    x0 = np.array([4.0, 9.0])
    x1 = np.array([0.5, 2.0])
    assert np.allclose(func(_xbar(x0, x1)), np.sqrt(x0) + np.maximum(x1, 0.5) * x0 ** 2)


@pytest.mark.parametrize("expression,value,kind", [
    ("1/x0", 0.0, "inf"),        # division by zero -> +inf
    ("log(x0)", 0.0, "-inf"),    # log(0) -> -inf
    ("log(x0)", -2.0, "nan"),    # log of a negative number -> nan
    ("sqrt(x0)", -1.0, "nan"),   # sqrt of a negative number -> nan
    ("exp(x0)", 1000.0, "inf"),  # overflow -> inf
])
def test_numerical_edge_cases_follow_ieee_conventions(expression, value, kind):
    """Documented current behaviour: singular points yield inf/nan, never raise."""
    func = compile_expression(expression, var_count=1)
    with np.errstate(all="ignore"):
        out = np.asarray(func(_xbar([value])))
    assert out.shape == (1,)
    if kind == "inf":
        assert np.isposinf(out[0])
    elif kind == "-inf":
        assert np.isneginf(out[0])
    else:
        assert np.isnan(out[0])


def test_jit_compile_maps_rows_to_values():
    raw = jit_compile(lambda x0: x0 ** 2)
    assert np.allclose(raw(_xbar([1.0, 2.0])), [1.0, 4.0])


def test_gradient_functions_are_analytic_partials():
    grads = compile_gradient_functions("x0*x1 + x1", var_count=2)
    assert len(grads) == 2
    point = _xbar([2.0], [3.0])
    assert grads[0](point)[0] == pytest.approx(3.0)   # d/dx0 = x1
    assert grads[1](point)[0] == pytest.approx(3.0)   # d/dx1 = x0 + 1


def test_expression_compilation_is_deterministic():
    first = compile_expression("x0 + 2.0", 1)(_xbar([1.0, 2.0]))
    second = compile_expression("x0 + 2.0", 1)(_xbar([1.0, 2.0]))
    assert np.array_equal(first, second)
    assert np.allclose(first, [3.0, 4.0])

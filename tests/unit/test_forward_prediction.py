"""Unit tests for forward prediction and analytic gradients."""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.predict.forward import (compile_expression,
                                  compile_gradient_functions, predict,
                                  predict_and_gradient, predict_vector,
                                  predict_vector_and_gradients)
from tests.fixtures import synthetic as S


def test_predict_matches_hand_calculated_values(fe_ni_weights):
    """Toy model ``y = Fe_fraction + 0.5 * Ni_fraction`` (row 0 = Fe, row 1 = Ni)."""
    weights = fe_ni_weights
    func = compile_expression("x0 + 0.5*x1", weights.shape[0])

    equimolar = S.compositions_from_fractions({"Fe": 0.5, "Ni": 0.5})
    pure_fe = S.compositions_from_fractions({"Fe": 1.0})
    pure_ni = S.compositions_from_fractions({"Ni": 1.0})

    assert predict(equimolar, weights, func) == pytest.approx(0.75)
    assert predict(pure_fe, weights, func) == pytest.approx(1.0)
    assert predict(pure_ni, weights, func) == pytest.approx(0.5)


def test_predict_vector_matches_per_sample_predict(fe_ni_compositions, fe_ni_weights):
    func = compile_expression("x0 + 0.5*x1", fe_ni_weights.shape[0])
    vector = predict_vector(fe_ni_compositions, fe_ni_weights, func)
    manual = np.array([predict(c, fe_ni_weights, func) for c in fe_ni_compositions])
    assert vector.shape == (len(fe_ni_compositions),)
    assert np.allclose(vector, manual)


def test_prediction_is_deterministic(fe_ni_compositions, fe_ni_weights):
    func = compile_expression("x0 + 0.5*x1", 2)
    first = predict_vector(fe_ni_compositions, fe_ni_weights, func)
    second = predict_vector(fe_ni_compositions, fe_ni_weights, func)
    assert np.array_equal(first, second)


def test_gradients_match_finite_differences(fe_ni_compositions):
    """Analytic ``df/dcomp`` must agree with one-sided finite differences.

    For ``y = x0**2 + sqrt(x0 + 1)`` with ``x0 = Fe_fraction``.
    """
    weights = S.element_selector_weights(("Fe",))
    expression = "x0*x0 + sqrt(x0 + 1.0)"
    func = compile_expression(expression, 1)
    grads = compile_gradient_functions(expression, 1)

    composition = fe_ni_compositions[0]
    value, gradient = predict_and_gradient(composition, weights, func, grads)

    eps = 1e-6
    numerical = np.zeros(118)
    for column in range(118):
        shifted = composition.copy()
        shifted[column] += eps
        numerical[column] = (predict(shifted, weights, func) - value) / eps

    assert gradient.shape == (118,)
    mask = numerical != 0
    assert mask.any(), "finite-difference reference is all zeros"
    assert np.allclose(gradient[mask], numerical[mask], atol=1e-5, rtol=1e-4)


@pytest.mark.xfail(
    reason="known defect in cwsr.predict.forward.predict_vector_and_gradients: "
           "einsum subscripts 'nvi,ij->nj' do not match grads.shape (n, var_count); "
           "the intended contraction is 'nv,vj->nj'",
    strict=False,
)
def test_vector_gradient_helper_agrees_with_per_sample_gradients(fe_ni_compositions):
    """Regression note (defect recorded, production code deliberately unchanged).

    Problem    : ``predict_vector_and_gradients`` is documented public API
                 (``docs/reference.md`` §5, listed in ``forward.__all__``) but is
                 not called anywhere in the framework.
    Expected   : vectorized ``(values, gradients)`` matching
                 ``predict_and_gradient`` per sample.
    Current    : ``grads`` is shaped ``(n, var_count)`` while the einsum requests
                 ``"nvi,ij->nj"`` (needs a 3-D operand), so the helper raises
                 ``ValueError: einstein sum subscripts string contains too many
                 subscripts for operand 0`` for *every* ``var_count``.
    """
    weights = S.element_selector_weights(("Fe",))
    expression = "sqrt(x0 + 1.0) * x0"
    func = compile_expression(expression, 1)
    grads = compile_gradient_functions(expression, 1)

    values, gradients = predict_vector_and_gradients(fe_ni_compositions, weights,
                                                     func, grads)
    value0, grad0 = predict_and_gradient(fe_ni_compositions[0], weights, func, grads)

    assert values.shape == (len(fe_ni_compositions),)
    assert gradients.shape == (len(fe_ni_compositions), 118)
    assert values[0] == pytest.approx(value0)
    assert np.allclose(gradients[0], grad0)


def test_linear_model_gradient_is_the_weight_matrix_row():
    """For ``y = x0`` the gradient w.r.t. the composition is exactly ``W[0]``."""
    weights = S.element_selector_weights(("Fe",))       # one row == one variable
    func = compile_expression("x0", 1)
    grads = compile_gradient_functions("x0", 1)
    composition = S.compositions_from_fractions({"Fe": 0.3, "Ni": 0.7})

    value, gradient = predict_and_gradient(composition, weights, func, grads)
    assert value == pytest.approx(0.3)
    assert gradient.shape == (118,)
    assert np.allclose(gradient, weights[0])

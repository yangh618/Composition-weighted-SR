"""Unit tests for inverse design (``analysis.inverse``).

The toy problem is analytically solvable, so the optimiser's answers can be
checked against hand-computed values instead of snapshots:
``y = Fe_fraction + 0.5 * Ni_fraction`` over the active set {Fe, Ni}.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.inverse import (optimize_target, optimize_tchebycheff,
                              optimize_weighted_sum)
from cwsr.predict.forward import (compile_expression,
                                  compile_gradient_functions)
from tests.fixtures import synthetic as S

EXPRESSION = "x0 + 0.5*x1"
ACTIVE = ["Fe", "Ni"]


@pytest.fixture
def toy_model(fe_ni_weights):
    func = compile_expression(EXPRESSION, fe_ni_weights.shape[0])
    grads = compile_gradient_functions(EXPRESSION, fe_ni_weights.shape[0])
    return fe_ni_weights, func, grads


@pytest.mark.parametrize("target,tolerance", [
    (0.9, 1e-6),      # interior target: hit essentially exactly
    (0.75, 1e-6),
    (0.5, 1e-3),      # boundary target (Ni = 1): limited by the optimiser tolerance
])
def test_optimize_target_reaches_reachable_values(toy_model, target, tolerance):
    weights, func, grads = toy_model
    composition, prediction, error = optimize_target(
        weights, func, target, grads, active_elements=ACTIVE, n_trials=5,
    )

    assert composition.shape == (118,)
    assert composition.sum() == pytest.approx(1.0, abs=1e-8)
    assert prediction == pytest.approx(target, abs=tolerance)
    assert error <= tolerance
    # mass only on the allowed elements
    assert set(np.flatnonzero(composition > 1e-9)) <= {S.FE, S.NI}


def test_optimize_target_reports_the_residual_for_unreachable_targets(toy_model):
    """``Fe + 0.5 * Ni`` over {Fe, Ni} spans [0.5, 1.0], so 0.25 is unreachable.

    Current behaviour: the optimiser returns the closest feasible composition and
    reports the residual error instead of raising.
    """
    weights, func, grads = toy_model
    composition, prediction, error = optimize_target(
        weights, func, 0.25, grads, active_elements=ACTIVE, n_trials=5,
    )

    assert prediction == pytest.approx(0.5, abs=1e-3)
    assert error == pytest.approx(0.25, abs=1e-3)
    assert composition.sum() == pytest.approx(1.0, abs=1e-8)


def test_optimize_target_is_deterministic_for_a_fixed_seed(toy_model):
    weights, func, grads = toy_model
    first, prediction_first, _ = optimize_target(weights, func, 0.75, grads,
                                                 active_elements=ACTIVE, n_trials=3)
    second, prediction_second, _ = optimize_target(weights, func, 0.75, grads,
                                                   active_elements=ACTIVE, n_trials=3)
    assert prediction_first == pytest.approx(prediction_second)
    assert np.allclose(first, second)


def test_optimize_target_rejects_unknown_elements(toy_model):
    weights, func, grads = toy_model
    with pytest.raises(ValueError):
        optimize_target(weights, func, 0.5, grads, active_elements=["Zz"])


def test_optimize_target_without_gradients_falls_back_gracefully(toy_model):
    weights, func, _ = toy_model
    composition, prediction, _ = optimize_target(
        weights, func, 0.75, None, active_elements=ACTIVE, n_trials=3,
        method="Nelder-Mead",
    )
    assert prediction == pytest.approx(0.75, abs=1e-4)
    assert set(np.flatnonzero(composition > 1e-9)) <= {S.FE, S.NI}


@pytest.fixture
def two_property_toy():
    """Property A = Fe fraction (maximise), property B = Ni fraction (minimise)."""
    weights = S.element_selector_weights(("Fe", "Ni"))
    funcs = [compile_expression("x0", 1), compile_expression("x0", 1)]
    grads = [compile_gradient_functions("x0", 1), compile_gradient_functions("x0", 1)]
    return [weights[0:1], weights[1:2]], funcs, grads


def test_weighted_sum_finds_the_analytic_optimum(two_property_toy):
    """maximise Fe while minimising Ni -> the optimum is pure Fe."""
    weights_list, funcs, grads = two_property_toy
    composition, values = optimize_weighted_sum(
        weights_list, funcs, grads, np.array([1.0, -1.0]), np.array([S.FE, S.NI]),
        n_multistart=3, seed=0,
    )

    assert composition.shape == (118,)
    assert composition.sum() == pytest.approx(1.0, abs=1e-8)
    assert values == pytest.approx([1.0, 0.0], abs=1e-4)


def test_tchebycheff_matches_the_analytic_optimum(two_property_toy):
    weights_list, funcs, grads = two_property_toy
    composition, values = optimize_tchebycheff(
        weights_list, funcs, grads, np.array([1.0, -1.0]), np.array([S.FE, S.NI]),
        n_multistart=3, seed=0,
    )

    assert composition.sum() == pytest.approx(1.0, abs=1e-8)
    assert values == pytest.approx([1.0, 0.0], abs=1e-4)
    # only the two active elements carry mass
    assert set(np.flatnonzero(composition > 1e-9)) <= {S.FE, S.NI}


def test_tchebycheff_supports_scalar_weights_and_rho(two_property_toy):
    weights_list, funcs, grads = two_property_toy
    _, values = optimize_tchebycheff(
        weights_list, funcs, grads, np.array([1.0, -1.0]), np.array([S.FE, S.NI]),
        n_multistart=2, seed=0, rho=1e-4, scalar_weights=np.array([0.5, 0.5]),
    )
    assert np.all(np.isfinite(values))


def test_multi_objective_helpers_are_reproducible(two_property_toy):
    weights_list, funcs, grads = two_property_toy
    first = optimize_tchebycheff(weights_list, funcs, grads, np.array([1.0, -1.0]),
                                 np.array([S.FE, S.NI]), n_multistart=3, seed=7)
    second = optimize_tchebycheff(weights_list, funcs, grads, np.array([1.0, -1.0]),
                                  np.array([S.FE, S.NI]), n_multistart=3, seed=7)
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])

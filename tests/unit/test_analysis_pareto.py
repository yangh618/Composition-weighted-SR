"""Unit tests for the Pareto front tool (``analysis.pareto``).

Toy trade-off: maximise ``Fe_fraction`` *and* ``Ni_fraction`` over {Fe, Ni}.
Every composition on the ``Fe + Ni = 1`` simplex is non-dominated, so the exact
front is known analytically and the returned points can be checked directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.pareto import compute_pareto_front_analytical
from cwsr.predict.forward import (compile_expression,
                                  compile_gradient_functions)
from tests.fixtures import synthetic as S


@pytest.fixture
def two_property_toy():
    weights = S.element_selector_weights(("Fe", "Ni"))
    funcs = [compile_expression("x0", 1), compile_expression("x0", 1)]
    grads = [compile_gradient_functions("x0", 1), compile_gradient_functions("x0", 1)]
    return [weights[0:1], weights[1:2]], funcs, grads


def is_non_dominated(values: np.ndarray, weights: np.ndarray) -> bool:
    """True if no point is dominated by another (signed values, maximisation)."""
    signed = values * np.sign(weights)[np.newaxis, :]
    for i in range(len(signed)):
        for j in range(len(signed)):
            if i == j:
                continue
            if np.all(signed[j] >= signed[i]) and np.any(signed[j] > signed[i]):
                return False
    return True


def test_front_is_non_dominated_and_spans_the_simplex(two_property_toy):
    weights_list, funcs, grads = two_property_toy
    compositions, values = compute_pareto_front_analytical(
        weights_list, funcs, grads, np.array([1.0, 1.0]), np.array([S.FE, S.NI]),
        n_pareto_points=6, seed=0, n_multistart=2, verbose=False,
    )

    assert compositions.shape[1] == 118
    assert values.shape[1] == 2
    assert len(values) >= 2, "a genuine trade-off must yield several front points"

    # structural invariant: Pareto fronts contain no dominated points
    assert is_non_dominated(values, np.array([1.0, 1.0]))

    # analytic: every point lies on Fe + Ni = 1 and both extremes are reachable
    assert np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
    assert values[:, 0].max() == pytest.approx(1.0, abs=1e-4)
    assert values[:, 1].max() == pytest.approx(1.0, abs=1e-4)

    # every returned composition is a valid normalized composition over {Fe, Ni}
    assert np.allclose(compositions.sum(axis=1), 1.0, atol=1e-8)
    for composition in compositions:
        assert set(np.flatnonzero(composition > 1e-9)) <= {S.FE, S.NI}


def test_front_is_reproducible_for_a_fixed_seed(two_property_toy):
    weights_list, funcs, grads = two_property_toy
    kwargs = dict(obj_weights=np.array([1.0, 1.0]), active_indices=np.array([S.FE, S.NI]),
                  n_pareto_points=5, seed=11, n_multistart=2, verbose=False)
    first_comps, first_vals = compute_pareto_front_analytical(weights_list, funcs, grads, **kwargs)
    second_comps, second_vals = compute_pareto_front_analytical(weights_list, funcs, grads, **kwargs)
    assert np.allclose(first_vals, second_vals)
    assert np.allclose(first_comps, second_comps)


def test_conflicting_weights_collapse_to_the_single_ideal_point(two_property_toy):
    """maximise Fe / minimise Ni: the ideal point is achievable -> one point.

    This documents current behaviour for a *non-conflicting* pair of objectives
    (the sweep legitimately finds the same optimum for every scalarisation, and
    duplicates are removed).
    """
    weights_list, funcs, grads = two_property_toy
    _, values = compute_pareto_front_analytical(
        weights_list, funcs, grads, np.array([1.0, -1.0]), np.array([S.FE, S.NI]),
        n_pareto_points=6, seed=0, n_multistart=2, verbose=False,
    )
    assert values.shape == (1, 2)
    assert values[0] == pytest.approx([1.0, 0.0], abs=1e-4)


def test_three_objective_sweep_returns_finite_points():
    """3+ objectives use Dirichlet-sampled scalarisation weights."""
    weights = S.element_selector_weights(("Fe", "Ni", "Co"))
    funcs = [compile_expression("x0", 1) for _ in range(3)]
    grads = [compile_gradient_functions("x0", 1) for _ in range(3)]
    weights_list = [weights[i:i + 1] for i in range(3)]

    compositions, values = compute_pareto_front_analytical(
        weights_list, funcs, grads, np.ones(3), np.array([S.FE, S.NI, S.CO]),
        n_pareto_points=4, seed=3, n_multistart=1, verbose=False,
    )

    assert values.shape[1] == 3
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(compositions))
    assert values.min() > -1e-6

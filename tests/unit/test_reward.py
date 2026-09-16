"""Unit tests for the reward/optimisation core (``cwsr.reward``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cwsr.reward import Heaviside_vec, Optimizer, sp_module
from tests.fixtures import synthetic as S


def test_heaviside_uses_the_half_at_zero_convention():
    out = Heaviside_vec(np.array([-2.0, -1e-9, 0.0, 1e-9, 3.0]))
    assert np.allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_sp_module_exposes_the_symbolic_function_set():
    """``sp_module`` is the (module_set, numpy) pair handed to ``lambdify``."""
    assert isinstance(sp_module, (list, tuple)) and len(sp_module) == 2
    functions, numpy_module = sp_module
    assert numpy_module is np
    for name in ("Min", "Max", "Heaviside", "Pow", "Abs", "log", "exp", "sqrt",
                 "sin", "cos", "tan"):
        assert name in functions, f"sp_module is missing {name}"


def test_optimizer_reads_sigma_from_targets(fe_ni_compositions):
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    X = fe_ni_compositions[:4]
    default = Optimizer(var_count=1, x_train=X, y_train=y)
    assert default.sigma == pytest.approx(float(np.std(y)))

    explicit = Optimizer(var_count=1, x_train=X, y_train=y, sigma=2.5)
    assert explicit.sigma == pytest.approx(2.5)


def test_optimizer_resolves_the_nlopt_algorithm(fe_ni_compositions):
    import nlopt

    X = fe_ni_compositions[:2]
    y = np.zeros(2)
    optimizer = Optimizer(var_count=1, x_train=X, y_train=y,
                          optimization_method="LD_LBFGS")
    assert optimizer.optimization_method == nlopt.LD_LBFGS
    with pytest.raises(AttributeError):
        Optimizer(var_count=1, x_train=X, y_train=y,
                  optimization_method="NOT_AN_ALGORITHM")


def test_run_nlopt_minimizes_a_quadratic(fe_ni_compositions):
    """Sanity check of the shared NLopt wrapper: minimise ``(x - 3)**2``."""
    optimizer = Optimizer(var_count=1, x_train=fe_ni_compositions[:2],
                          y_train=np.zeros(2), optimization_method="LD_LBFGS")

    def objective(params, grad):
        if grad.size > 0:
            grad[:] = 2.0 * (params - 3.0)
        return float((params[0] - 3.0) ** 2)

    solution = optimizer.run_nlopt(objective, np.array([-10.0]), bounds=47.0,
                                   xtol=1e-10, maxeval=500)
    assert solution[0] == pytest.approx(3.0, abs=1e-4)


def test_parameter_initialisation_shapes_and_positivity(fe_ni_compositions):
    state = SimpleNamespace(constant_count=1, real_constant_count=1)
    optimizer = Optimizer(var_count=2, x_train=fe_ni_compositions[:2],
                          y_train=np.zeros(2))

    np.random.seed(0)
    params = optimizer.init_parameters(state)
    # 2 latent variables * 118 weights + 1 real constant + 0 complex constants
    assert params.shape == (2 * 118 + 1,)

    np.random.seed(0)
    positive = optimizer.init_positive_parameters(state)
    assert positive.shape == (2 * 118 + 1,)
    assert np.all(np.asarray(positive[:2 * 118]) >= 0)


def test_valid_expression_returns_callable_for_finite_predictions(fe_ni_compositions):
    optimizer = Optimizer(var_count=1, x_train=fe_ni_compositions[:4],
                          y_train=np.zeros(4))
    state = SimpleNamespace(constant_count=0, real_constant_count=0)
    guess = np.abs(np.random.randn(118))

    func = optimizer.valid_expression("x0", state, guess)
    assert callable(func)
    assert np.all(np.isfinite(func(np.zeros((2, 1)) + 1.0)))


def test_valid_expression_returns_none_when_predictions_are_not_finite():
    """A singular expression on all-zero features is rejected (``log(0)``)."""
    zeros = np.zeros((4, 118))
    optimizer = Optimizer(var_count=1, x_train=zeros, y_train=np.zeros(4))
    state = SimpleNamespace(constant_count=0, real_constant_count=0)

    with np.errstate(all="ignore"):
        assert optimizer.valid_expression("log(x0)", state, np.zeros(118)) is None


def test_mae_loss_is_zero_at_the_true_weights_and_tracks_a_shift():
    """``build_MAELoss_func`` returns MAE; a uniform weight shift adds |shift|."""
    from cwsr import Regressor

    X, y, true_weights = S.tiny_linear_problem(n=16)
    model = Regressor(x_train=X, y_train=y, var_count=1, ops=["mul"],
                      max_depth=2, max_expressions=1, num_parallel=1, seed=0)
    loss = model.build_MAE_loss("x0", X, y)      # params = flattened weights only

    assert float(loss(true_weights, np.array([]))) == pytest.approx(0.0, abs=1e-12)
    assert float(loss(true_weights + 0.1, np.array([]))) == pytest.approx(0.1, rel=1e-9)

    gradient = np.zeros_like(true_weights)
    loss(true_weights, gradient)
    assert np.allclose(gradient, 0.0, atol=1e-9)          # zero at the minimum
    gradient[:] = 0.0
    loss(true_weights + 0.1, gradient)
    assert np.linalg.norm(gradient) > 0.0                  # non-zero away from it

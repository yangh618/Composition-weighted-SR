"""Unit tests for the search engine wrapper (``cwsr.Regressor``) — setup and
the search-free helpers. The end-to-end search lives in
``test_regressor_search.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cwsr import Regressor
from tests.fixtures import synthetic as S

#: The ranked-output contract documented in TESTING.md / docs/reference.md.
DOCUMENTED_OUTPUT_KEYS = {"expression", "weights", "train_reward",
                          "valid_reward", "mae", "mae_valid", "rank"}


@pytest.fixture
def small_model():
    X, y, _ = S.tiny_linear_problem(n=12)
    return Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                     var_count=1, ops=["add", "mul"], max_depth=3,
                     num_parallel=1, seed=0)


def test_initialization_wires_ops_context_and_tree(small_model):
    model = small_model

    assert model.var_count == 1
    assert {"add", "mul", "x0"} <= set(model.ops)
    assert model.arity_dict["add"] == 2 and model.arity_dict["x0"] == 0
    assert model.global_context["add"] is np.add
    assert model.complexity["add"] == 1
    assert model.exp_tree.max_depth == model.max_depth == 3
    assert model.optimizer.var_count == model.var_count


def test_rates_are_clipped_to_the_unit_interval():
    X, y, _ = S.tiny_linear_problem(n=6)
    model = Regressor(x_train=X, y_train=y, var_count=1, ops=["mul"],
                      gp_rate=2.0, mutation_rate=-1.0, exploration_rate=5.0,
                      num_parallel=1, seed=0)
    assert model.gp_rate == 1.0
    assert model.mutation_rate == 0.0
    assert model.exploration_rate == 1.0


def test_variables_are_appended_for_each_latent_dimension():
    X, y, _ = S.tiny_linear_problem(n=6)
    model = Regressor(x_train=X, y_train=y, var_count=3, ops=["mul"],
                      max_depth=2, num_parallel=1, seed=0)
    assert {f"x{i}" for i in range(3)} <= set(model.ops)
    assert all(model.arity_dict[f"x{i}"] == 0 for i in range(3))


def test_build_mae_loss_rejects_expressions_with_non_finite_predictions():
    """``log(0)`` on all-zero features must be rejected, not silently used."""
    zeros = np.zeros((4, 118))
    model = Regressor(x_train=zeros, y_train=np.zeros(4), var_count=1,
                      ops=["mul"], max_depth=2, num_parallel=1, seed=0)
    with np.errstate(all="ignore"):
        with pytest.raises(ValueError, match="non-finite"):
            model.build_MAE_loss("log(x0)", zeros, np.zeros(4))


def test_optimize_weights_returns_finite_metrics_and_a_weight_matrix():
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, var_count=1,
                      ops=["mul"], max_depth=2, num_parallel=1, seed=0)

    mae, mae_valid, weights = model.optimize_weights("x0", is_positive_init=True)

    assert np.isfinite(mae) and mae >= 0.0
    assert np.isfinite(mae_valid) and mae_valid >= 0.0
    assert weights.shape == (1, 118) and np.all(np.isfinite(weights))


def test_save_status_output_contract():
    """``save_status`` yields the ranked output dicts documented in TESTING.md."""
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, var_count=1,
                      ops=["mul"], max_depth=2, num_parallel=1, num_trials=1,
                      seed=0)
    mcts = model._create_mcts()
    mcts.exp_queue.append("x0", 0.5, 0.5)

    outputs = model.save_status(mcts)

    assert len(outputs) == 1
    entry = outputs[0]
    assert set(entry) == DOCUMENTED_OUTPUT_KEYS
    assert entry["rank"] == 1
    assert entry["expression"] == "x0"
    assert np.asarray(entry["weights"]).shape == (1, 118)
    assert np.isfinite(entry["mae"]) and np.isfinite(entry["mae_valid"])


def test_optimize_weights_without_validation_data_falls_back_to_training():
    X, y, _ = S.tiny_linear_problem(n=8)
    model = Regressor(x_train=X, y_train=y, var_count=1, ops=["mul"],
                      max_depth=2, num_parallel=1, seed=0)
    mae, mae_valid, weights = model.optimize_weights("x0", is_positive_init=False)
    assert weights.shape == (1, 118)
    assert np.isfinite(mae) and np.isfinite(mae_valid)

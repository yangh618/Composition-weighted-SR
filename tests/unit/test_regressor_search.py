"""End-to-end search tests for ``cwsr.Regressor`` / ``cwsr.model.fit_dataset``.

These run a *minimal* search (a few seconds, deterministic seed) through the
session-scoped ``fitted_run`` fixture, and assert invariants plus the exact
recovery of a known synthetic law. They are marked ``slow`` but still run by
default — the whole point of the suite is to exercise the real pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from cwsr import Regressor
from tests.conftest import TINY_SEARCH_KWARGS
from tests.unit.test_regressor import DOCUMENTED_OUTPUT_KEYS

pytestmark = pytest.mark.slow


def test_fit_outputs_contract(fitted_run):
    outputs = fitted_run.outputs

    assert outputs, "the search produced no candidate expressions"
    for index, entry in enumerate(outputs, start=1):
        assert set(entry) == DOCUMENTED_OUTPUT_KEYS
        assert entry["rank"] == index
        assert isinstance(entry["expression"], str) and entry["expression"]
        weights = np.asarray(entry["weights"])
        assert weights.shape == (1, 118) and np.all(np.isfinite(weights))
        assert np.isfinite(entry["mae"]) and np.isfinite(entry["mae_valid"])
        assert entry["mae"] >= 0.0 and entry["mae_valid"] >= 0.0


def test_fit_recovers_the_known_linear_law(fitted_run):
    """The synthetic target is exactly ``X @ w``: the search must recover it.

    Tolerance note: ``Regressor.optimize_weights`` runs NLopt with
    ``maxeval=200`` over ``var_count * 118`` parameters, so the fitted weights
    reproduce the law to ~1e-5 (observed ~5e-6) rather than to machine
    precision. ``1e-4`` is therefore the honest accuracy claim here, not an
    arbitrarily loosened bound.
    """
    from cwsr.predict.forward import compile_expression, predict_vector

    best = fitted_run.outputs[0]
    weights = np.asarray(best["weights"])
    predictions = predict_vector(fitted_run.compositions, weights,
                                 compile_expression(best["expression"], 1))

    assert np.allclose(predictions, fitted_run.targets, atol=1e-4)
    assert best["mae_valid"] <= 1e-4


def test_fit_is_reproducible_for_a_fixed_seed(fitted_run, tmp_path):
    """The same seed must reproduce the same ranked expression list."""
    from cwsr.model import fit_dataset

    repeat = fit_dataset(fitted_run.dataset, output_dir=tmp_path, seed=0,
                         **TINY_SEARCH_KWARGS)

    assert [entry["expression"] for entry in repeat] == \
           [entry["expression"] for entry in fitted_run.outputs]
    assert [entry["rank"] for entry in repeat] == \
           [entry["rank"] for entry in fitted_run.outputs]


def test_repeated_fits_on_the_same_model_agree(fitted_run):
    """Re-fitting one ``Regressor`` instance must not change the outcome."""
    X, y = fitted_run.compositions, fitted_run.targets
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, seed=0,
                      **TINY_SEARCH_KWARGS)
    first = model.fit(seed=0)[4]
    second = model.fit(seed=0)[4]
    assert [entry["expression"] for entry in first] == \
           [entry["expression"] for entry in second]


def test_num_trials_zero_crashes_the_search(fitted_run):
    """Regression note: documents current behaviour, does not endorse it.

    Problem      : ``num_trials=0`` is accepted by ``Regressor.__init__``.
    Expected     : a validation error, or a working (if poor) search.
    Current      : the first rollout returns ``None`` as its path and
                   ``MCTS_Node.backpropagate`` raises
                   ``TypeError: can only concatenate list (not "NoneType")``.
    Discovered   : while building this test suite (no code was changed).
    """
    X, y = fitted_run.compositions, fitted_run.targets
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, var_count=1,
                      ops=["add", "mul"], max_depth=3, max_expressions=2,
                      num_batches=1, num_trials=0, num_parallel=1,
                      optimization_method="LD_LBFGS", seed=0)
    with pytest.raises(TypeError, match="concatenate"):
        model.fit(seed=0)

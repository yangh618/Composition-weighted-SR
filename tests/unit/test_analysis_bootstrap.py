"""Unit tests for the bootstrap UQ metrics, result loading and exporting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis._common import load_expression_from_results
from analysis.bootstrap import (compute_bootstrap_metrics,
                                compute_confidence_intervals,
                                load_top_expressions, write_refined_results)


def test_confidence_intervals_match_numpy():
    predictions = np.array([[1.0, 2.0], [1.2, 2.2], [0.8, 1.8]])

    mean, std, lower, upper = compute_confidence_intervals(predictions, ci_level=95)

    assert np.allclose(mean, predictions.mean(axis=0))
    assert np.allclose(std, predictions.std(axis=0, ddof=1))
    assert np.allclose(lower, np.percentile(predictions, 2.5, axis=0))
    assert np.allclose(upper, np.percentile(predictions, 97.5, axis=0))
    assert np.all(lower <= mean) and np.all(mean <= upper)


def test_confidence_intervals_ignore_failed_resamples():
    predictions = np.array([[1.0], [np.nan], [3.0]])
    mean, std, _, _ = compute_confidence_intervals(predictions)
    assert mean[0] == pytest.approx(2.0)
    assert std[0] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_confidence_intervals_raise_when_all_resamples_failed():
    with pytest.raises(ValueError, match="No successful bootstrap iterations"):
        compute_confidence_intervals(np.full((3, 2), np.nan))


def test_metrics_for_a_perfect_prediction():
    truth = np.array([1.0, 2.0, 3.0])
    perfect = np.tile(truth, (2, 1))

    metrics = compute_bootstrap_metrics(truth, perfect)

    assert metrics["mae_mean"] == pytest.approx(0.0)
    assert metrics["rmse_mean"] == pytest.approx(0.0)
    assert metrics["r2_mean"] == pytest.approx(1.0)
    assert metrics["n_successful"] == 2
    assert metrics["mae_ci"] == pytest.approx((0.0, 0.0))


def test_metrics_are_computed_per_resample():
    """A constant offset must give MAE/RMSE equal to the offset, R2 < 1."""
    truth = np.array([0.0, 1.0, 2.0])
    predictions = np.array([truth + 1.0, truth + 3.0])

    metrics = compute_bootstrap_metrics(truth, predictions)

    assert metrics["mae_mean"] == pytest.approx(2.0)
    assert metrics["rmse_mean"] == pytest.approx(2.0)
    # CIs are percentile-based over the per-resample metrics
    assert metrics["mae_ci"] == pytest.approx(
        tuple(np.percentile([1.0, 3.0], [2.5, 97.5])))
    assert metrics["mae_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert metrics["r2_mean"] < 1.0
    assert metrics["r2_ci"][0] <= metrics["r2_mean"] <= metrics["r2_ci"][1]


def test_metrics_with_no_successful_resample_are_nan():
    metrics = compute_bootstrap_metrics(np.array([1.0]), np.array([[np.nan]]))
    assert metrics["n_successful"] == 0
    assert np.isnan(metrics["mae_mean"]) and np.isnan(metrics["r2_mean"])


def test_load_top_expressions_limits_and_handles_missing_files(tmp_path,
                                                              refined_results_file):
    assert len(load_top_expressions(refined_results_file, top_n=1)) == 1
    assert load_top_expressions(refined_results_file)[0]["expression"] == "x0 + 0.5*x1"
    assert load_top_expressions(tmp_path / "missing.json") is None


def test_write_refined_results_is_consumable_by_the_toolchain(tmp_path, fe_ni_weights):
    """The exported file must round-trip through the shared results loader."""
    bootstrap_result = {
        "expression": "x0 + 0.5*x1",
        "weights_original": fe_ni_weights.tolist(),
        "n_bootstrap": 8,
        "n_successful": 7,
        "metrics_train": {"mae_mean": 0.01, "mae_std": 0.002, "mae_ci": (0.008, 0.012),
                          "r2_mean": 0.99, "r2_std": 0.001, "r2_ci": (0.98, 0.999)},
        "metrics_test": {"mae_mean": 0.02, "mae_std": 0.003, "mae_ci": (0.015, 0.025),
                         "r2_mean": 0.95, "r2_std": 0.002, "r2_ci": (0.94, 0.96)},
    }
    path = write_refined_results(bootstrap_result, tmp_path / "refined.json")

    payload = json.loads(path.read_text())
    assert payload[0]["mae"] == pytest.approx(0.01)
    assert payload[0]["mae_valid"] == pytest.approx(0.02)
    assert payload[0]["n_successful"] == 7

    expression, weights = load_expression_from_results(path)
    assert expression == "x0 + 0.5*x1"
    assert weights.shape == (2, 118)


def test_write_refined_results_accepts_plain_metric_keys(tmp_path, fe_ni_weights):
    """Both ``<key>`` and ``<key>_mean`` metric spellings are supported."""
    path = write_refined_results(
        {"expression": "x0", "weights_original": fe_ni_weights.tolist(),
         "metrics_train": {"mae": 0.5}, "metrics_test": {"mae": 0.7}},
        tmp_path / "plain.json",
    )
    payload = json.loads(path.read_text())
    assert payload[0]["mae"] == pytest.approx(0.5)
    assert payload[0]["mae_valid"] == pytest.approx(0.7)

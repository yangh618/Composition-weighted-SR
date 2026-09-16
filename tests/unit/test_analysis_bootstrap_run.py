"""Unit tests for ``analysis.bootstrap.run_bootstrap`` and its CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.bootstrap import main, run_bootstrap
from tests.fixtures import synthetic as S


@pytest.fixture(scope="module")
def bootstrap_result():
    """One tiny bootstrap run over the exactly-linear synthetic problem."""
    compositions, targets, weights = S.tiny_linear_problem(n=24)
    np.random.seed(0)
    return run_bootstrap(
        compositions[:18], targets[:18], compositions[18:], targets[18:],
        "x0", weights.reshape(1, 118), var_count=1,
        n_bootstrap=3, num_trials=1, n_jobs=1, seed=0,
    )


def test_run_bootstrap_result_contract(bootstrap_result):
    assert set(bootstrap_result) >= {"expression", "weights_original", "var_count",
                                     "n_bootstrap", "n_successful", "ci_level",
                                     "metrics_train", "metrics_test",
                                     "prediction_stats", "weights_all", "success_mask"}
    assert bootstrap_result["expression"] == "x0"
    assert bootstrap_result["n_bootstrap"] == 3
    assert bootstrap_result["n_successful"] >= 1
    assert np.asarray(bootstrap_result["weights_original"]).shape == (1, 118)
    assert np.asarray(bootstrap_result["weights_all"]).shape == (3, 1, 118)
    assert len(bootstrap_result["success_mask"]) == 3


def test_run_bootstrap_metrics_are_ordered_and_finite(bootstrap_result):
    for key in ("metrics_train", "metrics_test"):
        metrics = bootstrap_result[key]
        assert np.isfinite(metrics["mae_mean"]) and metrics["mae_mean"] >= 0.0
        assert metrics["mae_ci"][0] <= metrics["mae_mean"] <= metrics["mae_ci"][1]
        assert np.isfinite(metrics["r2_mean"])
        assert metrics["r2_ci"][0] <= metrics["r2_mean"] <= metrics["r2_ci"][1]


def test_run_bootstrap_uncertainty_is_tiny_for_an_exact_model(bootstrap_result):
    """The synthetic target is exactly ``x0``: resamples should stay near-perfect."""
    assert bootstrap_result["metrics_test"]["mae_mean"] <= 1e-6
    stats = bootstrap_result["prediction_stats"]["test"]
    assert np.allclose(stats["mean"], np.asarray(stats["ci_lower"]), atol=1e-6)


def test_run_bootstrap_is_reproducible_with_a_fixed_numpy_seed():
    compositions, targets, weights = S.tiny_linear_problem(n=20)
    kwargs = dict(expression="x0", weights_original=weights.reshape(1, 118),
                  var_count=1, n_bootstrap=2, num_trials=1, n_jobs=1, seed=5)

    np.random.seed(0)
    first = run_bootstrap(compositions[:15], targets[:15], compositions[15:],
                          targets[15:], **kwargs)
    np.random.seed(0)
    second = run_bootstrap(compositions[:15], targets[:15], compositions[15:],
                           targets[15:], **kwargs)

    assert first["success_mask"] == second["success_mask"]
    assert np.allclose(first["weights_all"], second["weights_all"])
    assert first["metrics_test"]["mae_mean"] == pytest.approx(
        second["metrics_test"]["mae_mean"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("argv", [[], ["--dataset", "alloy_density"]])
def test_cli_requires_its_arguments(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_cli_runs_a_tiny_bootstrap_end_to_end(tmp_path, linear_npz,
                                              refined_results_file, capsys):
    output = tmp_path / "bootstrap.json"
    refined = tmp_path / "refined_from_bootstrap.json"

    code = main([
        "--dataset", str(linear_npz),
        "--results", str(refined_results_file),
        "--n_bootstrap", "2",
        "--num_trials", "1",
        "--jobs", "1",
        "--seed", "0",
        "--output", str(output),
        "--refined_output", str(refined),
    ])

    printed = capsys.readouterr().out
    assert code == 0
    assert "BOOTSTRAP UNCERTAINTY QUANTIFICATION" in printed
    assert "MAE" in printed

    payload = json.loads(output.read_text())
    assert payload["n_bootstrap"] == 2 and payload["n_successful"] >= 1

    refined_payload = json.loads(refined.read_text())
    assert refined_payload[0]["expression"] == "x0 + 0.5*x1"
    assert len(refined_payload[0]["weights"]) == 2


def test_cli_no_save_writes_nothing(tmp_path, linear_npz, refined_results_file,
                                    capsys):
    before = set(tmp_path.iterdir())
    code = main([
        "--dataset", str(linear_npz),
        "--results", str(refined_results_file),
        "--n_bootstrap", "2", "--num_trials", "1", "--jobs", "1",
        "--no_save",
    ])
    assert code == 0
    assert "--no_save" in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == before

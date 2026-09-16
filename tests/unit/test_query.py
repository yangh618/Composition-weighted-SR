"""Unit tests for the query tool (``cwsr.predict.query``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cwsr.predict.query import load_refined, main, predict


def test_load_refined_returns_the_first_entry(refined_results_file):
    entry = load_refined(refined_results_file)
    assert entry["expression"] == "x0 + 0.5*x1"
    assert np.asarray(entry["weights"]).shape == (2, 118)


def test_load_refined_error_paths(tmp_path, refined_results_file):
    with pytest.raises(FileNotFoundError, match="Refined results not found"):
        load_refined(tmp_path / "missing.json")

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="No refined results"):
        load_refined(empty)


def test_predict_matches_hand_calculated_values(refined_results_file):
    """``y = Fe_fraction + 0.5 * Ni_fraction`` for the toy refined model."""
    assert predict(refined_results_file, "Fe") == pytest.approx(1.0)
    assert predict(refined_results_file, "Ni") == pytest.approx(0.5)
    assert predict(refined_results_file, "Fe0.5Ni0.5") == pytest.approx(0.75)
    assert predict(refined_results_file, "Fe2O3") == pytest.approx(0.4)


def test_predict_is_deterministic(refined_results_file):
    first = predict(refined_results_file, "FeCoNi")
    second = predict(refined_results_file, "FeCoNi")
    assert first == second


def test_cli_prints_the_prediction_and_returns_zero(refined_results_file, capsys):
    code = main(["--results", str(refined_results_file), "Fe0.5Ni0.5"])
    assert code == 0
    assert "0.7500" in capsys.readouterr().out


def test_cli_requires_the_results_argument(refined_results_file):
    with pytest.raises(SystemExit) as excinfo:
        main(["Fe0.5Ni0.5"])
    assert excinfo.value.code == 2


def test_cli_reports_a_missing_results_file(tmp_path, capsys):
    code = main(["--results", str(tmp_path / "nope.json"), "Fe"])
    assert code == 1
    assert "Error" in capsys.readouterr().out


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_interactive_mode_exits_on_quit(refined_results_file, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "q")
    assert main(["--results", str(refined_results_file)]) == 0
    assert "CWSR Query Tool" in capsys.readouterr().out

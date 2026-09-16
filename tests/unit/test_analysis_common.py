"""Unit tests for the shared analysis helpers (``analysis._common``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis._common import (describe_elements, load_expression_from_results,
                              load_expression_specs, present_elements,
                              resolve_active_indices, save_json)
from tests.fixtures import synthetic as S


def test_load_expression_from_results_reads_a_ranked_list(refined_results_file):
    expression, weights = load_expression_from_results(refined_results_file)
    assert expression == "x0 + 0.5*x1"
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (2, 118) and weights.dtype == np.float64
    assert weights[0, S.FE] == 1.0 and weights[1, S.NI] == 1.0


def test_load_expression_from_results_accepts_a_single_dict(tmp_path):
    path = tmp_path / "single.json"
    path.write_text(json.dumps({"expression": "x0", "weights": [[0.0] * 118]}))
    expression, weights = load_expression_from_results(path)
    assert expression == "x0" and weights.shape == (1, 118)


def test_load_expression_from_results_honours_the_index(tmp_path, fe_ni_weights):
    path = tmp_path / "two.json"
    path.write_text(json.dumps([
        {"expression": "x0", "weights": fe_ni_weights.tolist()},
        {"expression": "x1", "weights": fe_ni_weights.tolist()},
    ]))
    assert load_expression_from_results(path, index=1)[0] == "x1"


def test_load_expression_from_results_error_paths(tmp_path, refined_results_file):
    with pytest.raises(FileNotFoundError, match="Results file not found"):
        load_expression_from_results(tmp_path / "missing.json")
    with pytest.raises(IndexError, match="out of range"):
        load_expression_from_results(refined_results_file, index=5)

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    with pytest.raises(IndexError, match="No expression entries"):
        load_expression_from_results(empty)


def test_load_expression_specs_collects_named_models(refined_results_file):
    models = load_expression_specs([("density", refined_results_file)], index=0)
    assert len(models) == 1
    model = models[0]
    assert set(model) == {"name", "path", "expression", "weights"}
    assert model["name"] == "density"
    assert model["expression"] == "x0 + 0.5*x1"
    assert model["weights"].shape == (2, 118)


def test_resolve_active_indices_accepts_symbols_numbers_and_none():
    assert np.array_equal(resolve_active_indices(None), np.arange(118))
    assert resolve_active_indices(["Fe", "Ni"]).tolist() == [S.FE, S.NI]
    assert resolve_active_indices([26]).tolist() == [S.FE]
    with pytest.raises(ValueError):
        resolve_active_indices(["Zz"])


def test_describe_elements_is_human_readable():
    assert describe_elements([S.FE]) == "Fe(Z=26)"
    assert describe_elements([S.FE, S.NI]) == "Fe(Z=26), Ni(Z=28)"


def test_present_elements_finds_columns_used_in_the_data(fe_ni_compositions):
    present = present_elements(fe_ni_compositions)
    assert present.tolist() == [S.FE, S.NI]


def test_save_json_creates_parents_and_returns_the_path(tmp_path):
    target = tmp_path / "nested" / "dir" / "payload.json"
    returned = save_json(target, {"a": 1, "b": [1.0, 2.0]})

    assert returned == target and target.exists()
    assert json.loads(target.read_text()) == {"a": 1, "b": [1.0, 2.0]}
    assert "\n" in target.read_text()          # indented, not a single line

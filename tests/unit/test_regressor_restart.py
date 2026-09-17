"""Tests for loading and restarting saved runs (checkpoint or results file).

Covers the three ways back into a trained model:
``Regressor.load`` (full checkpoint), ``Regressor.resume`` (continue the search)
and ``Regressor.from_results`` (warm-start a new search from a results JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cwsr import Regressor
from cwsr.exp_queue import combine_rewards
from cwsr.regressor import _expression_to_path
from tests.fixtures import synthetic as S

#: Smallest configuration that still runs a real search.
TINY = dict(var_count=1, ops=["add", "mul", "sqrt"], max_depth=4,
            max_expressions=2, num_batches=1, num_trials=1, num_parallel=1,
            optimization_method="LD_LBFGS", seed=0)


# ---------------------------------------------------------------------------
# Expression -> operator path (what lets a loaded model rejoin the GP phase)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expression,expected", [
    ("x0", ["x0"]),
    ("sqrt(x0)", ["sqrt", "x0"]),
    ("3*x0", ["mul", "R", "x0"]),
    ("x0 + x1", ["add", "x0", "x1"]),
    ("x0 - 0.5*x1", ["add", "x0", "mul", "R", "x1"]),
    ("x0*x1", ["mul", "x0", "x1"]),
    ("Pow(x0, 2)", ["Pow", "x0", "R"]),
    ("Max(x0, x1)", ["Max", "x0", "x1"]),
    ("sqrt(x0)*(x1 + 3.63214665208763)", ["mul", "sqrt", "x0", "add", "R", "x1"]),
    ("sqrt(x2*(sqrt(x0) + x0))", ["sqrt", "mul", "x2", "add", "x0", "sqrt", "x0"]),
])
def test_expression_to_path_map(expression, expected):
    """Documented mapping: literals -> ``R``, n-ary folds, sign/reciprocal absorbed."""
    assert _expression_to_path(expression) == expected


@pytest.mark.parametrize("expression", ["bogus_symbol", "unknown_fn(x0)", ""])
def test_expression_to_path_rejects_unrepresentable_input(expression):
    with pytest.raises(ValueError):
        _expression_to_path(expression)


# ---------------------------------------------------------------------------
# state_dict / from_state (no search needed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_model():
    X, y, _ = S.tiny_linear_problem(n=12)
    return Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                     run_meta={"dataset": "toy", "n_samples": 12}, **TINY)


def test_state_dict_captures_config_data_and_provenance(tiny_model):
    state = tiny_model.state_dict()

    assert set(state) == {"created", "config", "data", "results", "meta"}
    config = state["config"]
    assert config["var_count"] == 1
    assert config["ops"] == ["add", "mul", "sqrt"]        # variables are re-derived
    assert config["max_depth"] == 4
    assert config["optimization_method"] == "LD_LBFGS"
    assert config["seed"] == 0
    assert state["data"]["x_train"] and state["data"]["y_train"]
    assert state["meta"] == {"dataset": "toy", "n_samples": 12}
    assert state["results"] is None


def test_state_dict_round_trip_rebuilds_an_equivalent_model(tiny_model):
    rebuilt = Regressor.from_state(tiny_model.state_dict())

    assert rebuilt.var_count == tiny_model.var_count
    assert rebuilt.ops == tiny_model.ops
    assert rebuilt.max_depth == tiny_model.max_depth
    assert rebuilt.optimization_method == "LD_LBFGS"
    assert np.array_equal(rebuilt.x_train, tiny_model.x_train)
    assert np.array_equal(rebuilt.y_train, tiny_model.y_train)
    assert rebuilt.exp_tree.max_depth == tiny_model.exp_tree.max_depth


def test_from_state_applies_overrides(tiny_model):
    rebuilt = Regressor.from_state(tiny_model.state_dict(),
                                   max_expressions=123, output_prefix="/tmp/run")
    assert rebuilt.max_expressions == 123
    assert rebuilt.output_prefix == "/tmp/run"


def test_from_state_requires_data_when_the_state_has_none():
    with pytest.raises(ValueError, match="no training data"):
        Regressor.from_state({"config": {"var_count": 1, "ops": ["mul"]},
                              "data": {}})


def test_state_dict_records_mcts_summary_when_given(tiny_model):
    mcts = tiny_model._create_mcts()
    mcts.exp_queue.append("x0", 0.5, 0.5)
    state = tiny_model.state_dict(mcts=mcts, outputs=[{"expression": "x0"}])

    assert state["meta"]["count_num"] == 0
    assert state["meta"]["total_nodes"] == 1
    assert state["results"] == [{"expression": "x0"}]


# ---------------------------------------------------------------------------
# load / resume from a real checkpoint
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> dict:
    """One tiny persisted run: results JSON + self-describing final checkpoint."""
    output_dir = tmp_path_factory.mktemp("trained_run")
    prefix = output_dir / "run"
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix),
                      run_meta={"dataset": "toy_linear"}, **TINY)
    outputs = model.fit(seed=0)[4]
    return {"dir": output_dir, "prefix": prefix, "outputs": outputs,
            "X": X, "y": y,
            "results": Path(f"{prefix}_final.json"),
            "checkpoint": Path(f"{prefix}_ckpt_final.json")}


def test_checkpoint_is_self_describing(trained_run):
    payload = json.loads(trained_run["checkpoint"].read_text())

    assert set(payload) == {"mcts", "regressor"}
    state = payload["regressor"]
    assert state["meta"]["dataset"] == "toy_linear"
    assert state["meta"]["count_num"] > 0
    assert state["data"]["x_train"] and state["config"]["ops"] == ["add", "mul", "sqrt"]
    assert state["results"] == json.loads(json.dumps(trained_run["outputs"]))


def test_load_rebuilds_the_model_and_attaches_the_checkpoint(trained_run):
    model = Regressor.load(trained_run["checkpoint"])

    assert model.var_count == 1
    assert np.asarray(model.x_train).shape == trained_run["X"].shape
    assert np.allclose(np.asarray(model.x_train), trained_run["X"])
    assert model.results == json.loads(json.dumps(trained_run["outputs"]))
    assert model.checkpoint is not None and model.checkpoint["count_num"] > 0


def test_load_accepts_an_already_parsed_checkpoint(trained_run):
    payload = json.loads(trained_run["checkpoint"].read_text())
    assert Regressor.load(payload).var_count == 1


def test_load_rejects_a_results_only_file(trained_run):
    with pytest.raises(ValueError, match="from_results"):
        Regressor.load(trained_run["results"])


def test_resume_continues_the_search(trained_run, capsys):
    simplified, raw, evaluations, path, outputs = Regressor.resume(
        trained_run["checkpoint"], seed=0, max_expressions=3)

    assert "Resumed MCTS from" in capsys.readouterr().out
    previous = json.loads(trained_run["checkpoint"].read_text())["mcts"]["count_num"]
    assert evaluations >= previous
    assert outputs and outputs[0]["expression"]


# ---------------------------------------------------------------------------
# Warm start from a results file
# ---------------------------------------------------------------------------
def test_from_results_warm_starts_a_new_search(trained_run, capsys):
    model = Regressor.from_results(trained_run["results"], trained_run["X"],
                                   trained_run["y"], trained_run["X"],
                                   trained_run["y"], **TINY)
    assert len(model.warm_start) == 3
    assert model.warm_start[0]["expression"] == trained_run["outputs"][0]["expression"]

    simplified, raw, evaluations, path, outputs = model.fit(seed=0)

    printed = capsys.readouterr().out
    assert "[Warm start] Seeded" in printed
    # the loaded champion is retained as a candidate in the new ranking
    assert any(entry["expression"] == trained_run["outputs"][0]["expression"]
               for entry in outputs)


def test_from_results_respects_warm_start_top_n(trained_run):
    model = Regressor.from_results(trained_run["results"], trained_run["X"],
                                   trained_run["y"], warm_start_top_n=1, **TINY)
    assert len(model.warm_start) == 1


def test_seed_from_warm_start_populates_both_queues(tmp_path):
    """With a rebuildable path, the loaded champion also enters the GP pool."""
    import warnings as _warnings

    X, y, _ = S.tiny_linear_problem(n=6)
    entry = {"expression": "0.5*x0", "train_reward": 0.4, "valid_reward": 0.45}
    model = Regressor(x_train=X, y_train=y, warm_start=[entry],
                      **{**TINY, "ops": ["add", "mul", "R"]})
    mcts = model._create_mcts()

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")          # no warning: the path is rebuildable
        model._seed_from_warm_start(mcts)

    assert len(mcts.exp_queue) == 1 and len(mcts.path_queue) == 1
    assert mcts.path_queue.best()[0] == ["mul", "R", "x0"]
    # loaded champions are scored with the *current* ranking criterion (the
    # train/valid mix), not with the validation reward alone
    expected = combine_rewards(entry["train_reward"], entry["valid_reward"],
                               model.valid_reward_weight)
    assert mcts.best_reward == pytest.approx(expected)
    assert mcts.best_reward != pytest.approx(entry["valid_reward"])


def test_warm_start_warns_when_the_path_cannot_be_rebuilt(tmp_path):
    """A literal constant needs the ``R`` token; without it only results are seeded."""
    X, y, _ = S.tiny_linear_problem(n=12)
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"expression": "0.5*x0",
                                    "train_reward": 0.4, "valid_reward": 0.4}]))
    model = Regressor.from_results(results, X, y,
                                   **{**TINY, "ops": ["add", "mul"],
                                      "max_expressions": 1})

    with pytest.warns(UserWarning, match="cannot be rebuilt as an operator path"):
        simplified, raw, evaluations, path, outputs = model.fit(seed=0)

    # the loaded candidate is still reported, and the run completed
    assert evaluations >= 1

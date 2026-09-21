"""Tests for standalone hyperparameters persistence (save + reload).

``Regressor.save_hyperparameters`` writes a run's settings to their own JSON file
(``<output_prefix>_hyperparams.json``, or an explicit path) and
``Regressor.load_hyperparameters`` / ``Regressor.from_hyperparameters`` read them
back — so a search configuration can be archived and replayed on other data
without carrying a checkpoint along. The mapping written is exactly the ``config``
entry of ``Regressor.state_dict``, i.e. the hyperparameters the checkpoints have
always embedded.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cwsr import Regressor
from cwsr.regressor import (HYPERPARAMETERS_FORMAT, HYPERPARAMETERS_SUFFIX,
                            _STATE_CONFIG_KEYS)
from tests.fixtures import synthetic as S

#: Smallest configuration that still runs a real search.
TINY = dict(var_count=1, ops=["add", "mul"], max_depth=3, max_expressions=2,
            num_batches=1, num_trials=1, num_parallel=1,
            optimization_method="LD_LBFGS", seed=0)


@pytest.fixture
def small_model():
    """An untrained model with provenance, and no ``output_prefix``."""
    X, y, _ = S.tiny_linear_problem(n=12)
    return Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                     run_meta={"dataset": "toy", "n_samples": 12}, **TINY)


# ---------------------------------------------------------------------------
# What counts as "the hyperparameters"
# ---------------------------------------------------------------------------
def test_hyperparameters_are_the_checkpoint_config(small_model):
    """The standalone mapping is the very one the checkpoints embed."""
    params = small_model.hyperparameters()

    assert params == small_model.state_dict()["config"]
    assert set(params) == set(_STATE_CONFIG_KEYS)
    assert params["var_count"] == 1
    assert params["ops"] == ["add", "mul"]            # variables are re-derived
    assert params["max_depth"] == 3
    assert params["optimization_method"] == "LD_LBFGS"
    assert params["seed"] == 0
    assert "reward_func" not in params                # a callable cannot be saved
    json.dumps(params)                                # already JSON-serializable


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def test_save_hyperparameters_defaults_to_the_output_prefix(tmp_path):
    X, y, _ = S.tiny_linear_problem(n=12)
    prefix = tmp_path / "run"
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix), **TINY)

    written = model.save_hyperparameters()

    assert written == f"{prefix}{HYPERPARAMETERS_SUFFIX}"
    payload = json.loads(Path(written).read_text())
    assert set(payload) == {"format", "created", "config", "meta"}
    assert payload["format"] == HYPERPARAMETERS_FORMAT
    assert payload["created"]                                  # a timestamp
    assert payload["meta"] == {}
    assert payload["config"] == model.hyperparameters()
    assert payload["config"]["output_prefix"] == str(prefix)


def test_save_hyperparameters_accepts_an_explicit_path(tmp_path, small_model):
    """An explicit path works without an ``output_prefix``."""
    target = tmp_path / "params.json"

    assert small_model.save_hyperparameters(str(target)) == str(target)
    assert json.loads(target.read_text())["meta"] == {"dataset": "toy",
                                                      "n_samples": 12}


def test_save_hyperparameters_requires_a_path_or_a_prefix(small_model):
    assert small_model.output_prefix is None
    with pytest.raises(ValueError, match="output_prefix is not set"):
        small_model.save_hyperparameters()


def test_the_file_records_every_setting_that_drives_the_search(tmp_path):
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      valid_reward_weight=0.25, save_every=5,
                      save_checkpoint_every=7, output_prefix="/tmp/run", **TINY)

    params = Regressor.load_hyperparameters(model.save_hyperparameters(
        str(tmp_path / "params.json")))

    assert params["valid_reward_weight"] == pytest.approx(0.25)
    assert params["save_every"] == 5
    assert params["save_checkpoint_every"] == 7
    assert params["output_prefix"] == "/tmp/run"
    assert params["seed"] == 0


# ---------------------------------------------------------------------------
# Reloading
# ---------------------------------------------------------------------------
def test_load_hyperparameters_round_trips_a_file(tmp_path, small_model):
    written = small_model.save_hyperparameters(str(tmp_path / "params.json"))

    assert Regressor.load_hyperparameters(written) == small_model.hyperparameters()


def test_load_hyperparameters_reads_a_state_dict_and_a_save_file(tmp_path,
                                                                small_model):
    """Both the parsed save file and a state dict are accepted."""
    written = small_model.save_hyperparameters(str(tmp_path / "params.json"))
    payload = json.loads(Path(written).read_text())

    assert Regressor.load_hyperparameters(payload) == small_model.hyperparameters()
    assert Regressor.load_hyperparameters(small_model.state_dict()) == \
           small_model.hyperparameters()


def test_load_hyperparameters_rejects_a_payload_without_a_config(tmp_path):
    """A results-only JSON (the saved *model*) is not a hyperparameters file."""
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"expression": "x0", "rank": 1}]))

    with pytest.raises(ValueError, match="carries no 'config'"):
        Regressor.load_hyperparameters(results)


# ---------------------------------------------------------------------------
# Rebuilding a model from saved hyperparameters
# ---------------------------------------------------------------------------
def test_from_hyperparameters_replays_the_settings_on_new_data(tmp_path, small_model):
    written = small_model.save_hyperparameters(str(tmp_path / "params.json"))
    other_X, other_y, _ = S.tiny_linear_problem(n=8)      # different dataset

    rebuilt = Regressor.from_hyperparameters(written, other_X, other_y)

    assert rebuilt.var_count == small_model.var_count
    assert rebuilt.ops == small_model.ops
    assert rebuilt.max_depth == small_model.max_depth
    assert rebuilt.optimization_method == "LD_LBFGS"
    assert rebuilt.valid_reward_weight == small_model.valid_reward_weight
    assert rebuilt.seed == 0
    # ... on the data given here, and nothing is invented beyond the settings
    assert np.array_equal(rebuilt.x_train, np.asarray(other_X, dtype=np.float64))
    assert rebuilt.x_valid is None
    assert rebuilt.results is None and rebuilt.warm_start == []
    assert rebuilt.checkpoint is None          # a fresh search, not a resume


def test_from_hyperparameters_applies_overrides(tmp_path, small_model):
    written = small_model.save_hyperparameters(str(tmp_path / "params.json"))

    rebuilt = Regressor.from_hyperparameters(
        written, small_model.x_train, small_model.y_train,
        max_expressions=123, output_prefix="/tmp/run", valid_reward_weight=0.0)

    assert rebuilt.max_expressions == 123
    assert rebuilt.output_prefix == "/tmp/run"
    assert rebuilt.valid_reward_weight == 0.0


def test_from_hyperparameters_accepts_an_already_parsed_file(tmp_path, small_model):
    written = small_model.save_hyperparameters(str(tmp_path / "params.json"))
    payload = json.loads(Path(written).read_text())

    rebuilt = Regressor.from_hyperparameters(payload, small_model.x_train,
                                             small_model.y_train)

    assert rebuilt.max_depth == small_model.max_depth
    assert rebuilt.hyperparameters()["ops"] == ["add", "mul"]


# ---------------------------------------------------------------------------
# A real run records its settings (one tiny search, shared by the tests below)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run_with_prefix(tmp_path_factory) -> SimpleNamespace:
    """One tiny search with an ``output_prefix`` and run provenance."""
    output_dir = tmp_path_factory.mktemp("hyperparams_run")
    prefix = output_dir / "run"
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix), run_meta={"dataset": "toy"},
                      **TINY)
    outputs = model.fit(seed=0)[4]
    return SimpleNamespace(dir=output_dir, prefix=prefix, model=model,
                           outputs=outputs)


def test_fit_writes_the_hyperparameters_file(run_with_prefix):
    """A run with an ``output_prefix`` always leaves its settings on disk."""
    written = Path(f"{run_with_prefix.prefix}{HYPERPARAMETERS_SUFFIX}")

    assert written.exists(), "fit did not write the hyperparameters file"
    payload = json.loads(written.read_text())
    assert payload["format"] == HYPERPARAMETERS_FORMAT
    assert payload["created"]
    assert payload["meta"] == {"dataset": "toy"}
    assert payload["config"] == run_with_prefix.model.hyperparameters()
    # the settings are reloadable straight from the run's own artifact
    assert Regressor.load_hyperparameters(written) == \
           run_with_prefix.model.hyperparameters()


def test_load_hyperparameters_also_reads_a_checkpoint(run_with_prefix):
    """A checkpoint written by ``fit`` carries the same hyperparameters."""
    from cwsr.checkpoint import load_checkpoint

    checkpoint_file = f"{run_with_prefix.prefix}_ckpt_final.json"
    embedded = load_checkpoint(checkpoint_file)["regressor"]["config"]

    assert Regressor.load_hyperparameters(checkpoint_file) == embedded
    assert Regressor.load_hyperparameters(
        f"{run_with_prefix.prefix}{HYPERPARAMETERS_SUFFIX}") == embedded


def test_a_run_can_be_replayed_from_its_hyperparameters_file(run_with_prefix,
                                                             tmp_path, capsys):
    """The recorded settings rebuild a working model (fresh search, new budget)."""
    replay = tmp_path / "replay"
    model = Regressor.from_hyperparameters(
        f"{run_with_prefix.prefix}{HYPERPARAMETERS_SUFFIX}",
        run_with_prefix.model.x_train, run_with_prefix.model.y_train,
        run_with_prefix.model.x_valid, run_with_prefix.model.y_valid,
        max_expressions=1, output_prefix=str(replay))

    assert model.checkpoint is None
    assert model.max_expressions == 1

    _, _, evaluations, _, outputs = model.fit(seed=0)

    assert "Resumed MCTS" not in capsys.readouterr().out   # a fresh search
    assert evaluations >= 1
    assert outputs and outputs[0]["rank"] == 1
    # the replayed run records its own settings next to its artifacts
    assert Path(f"{replay}{HYPERPARAMETERS_SUFFIX}").exists()

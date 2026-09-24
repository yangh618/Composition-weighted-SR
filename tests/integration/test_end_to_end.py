"""Minimal end-to-end workflow test.

synthetic dataset -> ``CompositionDataset`` -> ``fit_dataset`` (CWSR search) ->
saved artifact -> evaluation -> downstream consumers (query / inverse design).

Reuses the session-scoped ``fitted_run`` fixture, so the (tiny) search runs once
for the whole suite. No real materials data, no network, no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_pipeline_writes_a_loadable_artifact(fitted_run):
    artifact = fitted_run.artifact

    assert artifact.exists()
    assert artifact.name.startswith("cwsr_outputs_toy_linear_")
    assert artifact.parent == fitted_run.output_dir

    payload = json.loads(artifact.read_text())
    assert isinstance(payload, list) and payload
    # the artifact is exactly the returned outputs (JSON round-trip safe)
    assert payload == json.loads(json.dumps(fitted_run.outputs))


def test_saved_model_metrics_match_a_recomputed_split(fitted_run):
    """Artifact metrics are split-based; recompute them on the same splits.

    ``Regressor.optimize_weights`` reports the MAE of the **training split** in
    ``mae`` and of the validation split in ``mae_valid`` (the loss is built on
    ``self.x_train/self.y_train`` and ``self.x_valid/self.y_valid``). The suite's
    fit uses ``seed=0`` and ``split_ratio=0.8``, which is reproducible here.

    The fitted weights reproduce the exactly-linear synthetic target to ~5e-6
    (NLopt ``maxeval=200`` in the weight stage), hence the 1e-4 accuracy bound.
    """
    from cwsr.data import split_dataset
    from cwsr.predict.forward import compile_expression, predict_vector

    entry = json.loads(fitted_run.artifact.read_text())[0]
    compositions, targets = fitted_run.compositions, fitted_run.targets
    train_idx, valid_idx = split_dataset(compositions, targets, ratio=0.8, seed=0)

    predictions = predict_vector(compositions,
                                 np.asarray(entry["weights"], dtype=float),
                                 compile_expression(entry["expression"], 1))

    mae_train = float(np.mean(np.abs(predictions[train_idx] - targets[train_idx])))
    mae_valid = float(np.mean(np.abs(predictions[valid_idx] - targets[valid_idx])))

    assert mae_train == pytest.approx(entry["mae"], abs=1e-9)
    assert mae_valid == pytest.approx(entry["mae_valid"], abs=1e-9)
    assert float(np.mean(np.abs(predictions - targets))) <= 1e-4


def test_artifact_flows_into_query_and_inverse_design(fitted_run):
    from analysis._common import load_expression_from_results
    from analysis.inverse import optimize_target
    from cwsr.predict.forward import (compile_expression,
                                      compile_gradient_functions)
    from cwsr.predict.query import predict as query_predict

    expression, weights = load_expression_from_results(fitted_run.artifact)
    assert expression and weights.shape == (1, 118)

    # formula -> property through the documented query path
    value = query_predict(fitted_run.artifact, "FeCoNi")
    assert np.isfinite(value)

    # the trained model supports inverse design on a reachable target
    func = compile_expression(expression, 1)
    grads = compile_gradient_functions(expression, 1)
    target = float(fitted_run.targets.mean())
    composition, prediction, error = optimize_target(
        weights, func, target, grads,
        active_elements=["Fe", "Co", "Ni", "Cr"], n_trials=3,
    )

    assert composition.sum() == pytest.approx(1.0, abs=1e-8)
    assert error <= 1e-3
    assert prediction == pytest.approx(target, abs=1e-3)


def test_every_training_run_is_persisted(fitted_run):
    """``fit_dataset`` sets an ``output_prefix``, so the finished run is on disk.

    Artifacts: ``cwsr_outputs_<name>_<ts>.json`` (the model, written by
    ``cwsr.model.train``) plus ``…_hyperparams.json``, ``…_final.json`` and
    ``…_ckpt_final.json`` (written by ``Regressor.fit``). The final checkpoint
    must be resumable.
    """
    from cwsr.checkpoint import load_checkpoint

    names = {path.name for path in fitted_run.output_dir.iterdir()}
    assert any(name.endswith("_hyperparams.json") for name in names), names
    assert any(name.endswith("_final.json") for name in names), names
    assert any(name.endswith("_ckpt_final.json") for name in names), names

    # ... and the hyperparameters file reloads to the same settings
    from cwsr import Regressor

    hyperparams_file = next(path for path in fitted_run.output_dir.iterdir()
                            if path.name.endswith("_hyperparams.json"))
    assert Regressor.load_hyperparameters(hyperparams_file) == \
           Regressor.load_hyperparameters(
               next(path for path in fitted_run.output_dir.iterdir()
                    if path.name.endswith("_ckpt_final.json")))

    checkpoint_file = next(path for path in fitted_run.output_dir.iterdir()
                           if path.name.endswith("_ckpt_final.json"))
    checkpoint = load_checkpoint(checkpoint_file)
    assert checkpoint["mcts"]["count_num"] > 0
    assert isinstance(checkpoint["mcts"]["tree"], list)

    # the run state is embedded and carries dataset provenance from fit_dataset
    from cwsr.data import split_dataset

    state = checkpoint["regressor"]
    assert state["meta"]["dataset"] == fitted_run.dataset.name
    assert state["meta"]["n_samples"] == fitted_run.dataset.n_samples
    assert state["config"]["var_count"] == 1
    # the stored data is the run's training split (fit_dataset uses ratio=0.8)
    train_idx, _ = split_dataset(fitted_run.dataset.compositions,
                                 fitted_run.dataset.targets, ratio=0.8, seed=0)
    assert len(state["data"]["x_train"]) == len(train_idx)
    assert np.allclose(np.asarray(state["data"]["x_train"]),
                       fitted_run.dataset.compositions[train_idx])


def test_train_helper_writes_its_output_file(tmp_path):
    """``cwsr.model.train`` writes ``<prefix>.json`` (documented artifact name)."""
    from cwsr.data import split_dataset
    from cwsr.model import train
    from tests.fixtures import synthetic as S

    compositions, targets, _ = S.tiny_linear_problem(n=8)
    train_idx, valid_idx = split_dataset(compositions, targets, ratio=0.75, seed=0)
    assert len(train_idx) + len(valid_idx) == 8

    prefix = tmp_path / "run"
    outputs = train(compositions, targets, output_prefix=str(prefix), seed=0,
                    var_count=1, ops=["add", "mul"], max_depth=2,
                    max_expressions=2, num_batches=1, num_trials=1,
                    num_parallel=1, optimization_method="LD_LBFGS")

    written = tmp_path / "run.json"
    assert written.exists()
    assert json.loads(written.read_text()) == json.loads(json.dumps(outputs))


def test_a_resumed_run_can_be_resumed_again(tmp_path):
    """Regression: resuming *twice* used to crash with ``Invalid op``.

    Loading a checkpoint rebuilds the MCTS tree from its flat representation, so
    a wrong breadth-first child indexing silently re-wired the branches. The next
    selection then replayed an operator onto an ``ExpTree`` state that could no
    longer accept it (``ValueError: Invalid op, not in available ops``), which
    broke every resume beyond the first. The checkpoint written *by a resumed
    run* must therefore be resumable again.
    """
    from cwsr import Regressor
    from tests.fixtures import synthetic as S

    X, y, _ = S.tiny_linear_problem(n=12)
    prefix = tmp_path / "run"
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix), var_count=1,
                      ops=["add", "mul", "sqrt"], max_depth=4,
                      max_expressions=4, num_batches=2, num_trials=1,
                      num_parallel=1, save_checkpoint_every=2,
                      optimization_method="LD_LBFGS", seed=0)
    model.fit(seed=0)

    # first resume: from the checkpoint of the original run
    _, _, evaluations, _, _ = Regressor.resume(f"{prefix}_ckpt_final.json",
                                               seed=0, max_expressions=6)
    assert evaluations > 0

    # second resume: from a checkpoint written *by the resumed run*
    written = sorted(tmp_path.glob("run_ckpt_step*.json"),
                     key=lambda path: int(path.stem.rsplit("step", 1)[1]))
    assert len(written) >= 2, [path.name for path in written]

    _, _, evaluations, _, outputs = Regressor.resume(written[-1], seed=0,
                                                     max_expressions=8)
    assert evaluations >= json.loads(written[-1].read_text())["mcts"]["count_num"]
    assert outputs and outputs[0]["expression"]

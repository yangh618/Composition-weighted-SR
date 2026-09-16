"""Tests for run persistence: warnings, periodic saves, final artifacts, resume.

Context: ``save_every`` / ``save_checkpoint_every`` require ``output_prefix``;
without it the saves were silently skipped, and ``fit`` never persisted a
finished run at all. These tests pin the (documented) behaviour now in place.
"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cwsr import Regressor
from cwsr.checkpoint import load_checkpoint
from cwsr.regressor import _warn_if_no_output_prefix
from tests.fixtures import synthetic as S

#: Smallest configuration that still runs a real search.
TINY = dict(var_count=1, ops=["add", "mul"], max_depth=3, max_expressions=3,
            num_batches=1, num_trials=1, num_parallel=1,
            optimization_method="LD_LBFGS", seed=0)


@contextmanager
def no_warnings_allowed():
    """Fail if any warning is emitted (negative assertions)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ---------------------------------------------------------------------------
# The silent no-op must now be visible
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("save_every,save_checkpoint_every", [(0, 0), (None, None)])
def test_no_warning_when_no_save_was_requested(save_every, save_checkpoint_every):
    """``0`` (the default) and ``None`` both mean "no periodic saves"."""
    with no_warnings_allowed():
        _warn_if_no_output_prefix(save_every, save_checkpoint_every, None)


def test_no_warning_when_output_prefix_is_set():
    with no_warnings_allowed():
        _warn_if_no_output_prefix(5, 5, "/tmp/run")


@pytest.mark.parametrize("save_every,checkpoint_every,match", [
    (5, 0, "save_every=5"),
    (0, 5, "save_checkpoint_every=5"),
    (5, 5, "save_every=5 and save_checkpoint_every=5"),
])
def test_warns_when_save_flags_have_no_output_prefix(save_every, checkpoint_every,
                                                     match):
    with pytest.warns(UserWarning, match=match):
        _warn_if_no_output_prefix(save_every, checkpoint_every, None)


def test_fit_warns_about_unused_save_flags():
    """The warning must reach a user of a real run, not only the helper."""
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      save_every=5, save_checkpoint_every=5,
                      **{**TINY, "max_expressions": 1})
    with pytest.warns(UserWarning, match="output_prefix is not set"):
        model.fit(seed=0)


# ---------------------------------------------------------------------------
# Periodic + final artifacts (one tiny search, shared by the assertions below)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def saved_run(tmp_path_factory) -> SimpleNamespace:
    """One tiny search with periodic saves enabled and an ``output_prefix``.

    ``save_every=2`` triggers exactly one periodic save (each save runs
    ``save_status``, i.e. ten NLopt weight optimisations, so keeping the
    interval coarse keeps the suite fast).
    """
    output_dir = tmp_path_factory.mktemp("saved_run")
    prefix = output_dir / "run"
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix), save_every=2,
                      save_checkpoint_every=2, **TINY)
    outputs = model.fit(seed=0)[4]
    return SimpleNamespace(dir=output_dir, prefix=prefix, outputs=outputs)


def test_periodic_saves_are_written_when_a_prefix_is_set(saved_run):
    step_files = sorted(p for p in saved_run.dir.iterdir()
                        if p.name.startswith("run_step"))
    checkpoint_files = sorted(p for p in saved_run.dir.iterdir()
                              if p.name.startswith("run_ckpt_step"))

    assert step_files, "no periodic results files were written"
    assert checkpoint_files, "no periodic checkpoints were written"
    # periodic files must be valid artifacts of their documented shape
    payload = json.loads(step_files[-1].read_text())
    assert isinstance(payload, list) and payload[0]["rank"] == 1
    assert load_checkpoint(checkpoint_files[-1])["mcts"]["count_num"] > 0


def test_final_results_and_checkpoint_are_written(saved_run):
    results_file = Path(f"{saved_run.prefix}_final.json")
    checkpoint_file = Path(f"{saved_run.prefix}_ckpt_final.json")

    assert results_file.exists(), "the finished run's results were not persisted"
    assert checkpoint_file.exists(), "the finished run's checkpoint was not persisted"

    assert json.loads(results_file.read_text()) == \
           json.loads(json.dumps(saved_run.outputs))

    checkpoint = load_checkpoint(checkpoint_file)
    assert "mcts" in checkpoint
    assert checkpoint["mcts"]["count_num"] > 0
    assert isinstance(checkpoint["mcts"]["tree"], list)


def test_final_checkpoint_resumes_a_following_run(saved_run, capsys):
    """The written checkpoint is usable: ``fit(checkpoint=...)`` resumes from it."""
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      **{**TINY, "max_expressions": 1})
    resumed = load_checkpoint(f"{saved_run.prefix}_ckpt_final.json")["mcts"]

    evaluations = model.fit(seed=0, checkpoint=resumed)[2]

    assert "Resumed MCTS from" in capsys.readouterr().out
    assert evaluations >= resumed["count_num"]


def test_no_files_are_written_without_a_prefix(tmp_path, monkeypatch):
    """Default in-memory use of ``Regressor`` must not touch the filesystem."""
    monkeypatch.chdir(tmp_path)
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, **TINY)
    model.fit(seed=0)
    assert list(tmp_path.iterdir()) == []


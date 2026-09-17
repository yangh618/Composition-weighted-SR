"""Tests for the mixed train/valid reward used to rank search candidates.

``Regressor(valid_reward_weight=α)`` makes the search optimise
``α · valid_reward + (1 − α) · train_reward`` instead of ranking by the validation
reward alone (``α = 1``, the original CWSR behaviour **and the shipped default** —
mixing is opt-in). The coefficient drives the expression/path queues, the
within-batch trial selection, the tree backpropagation and the "solved" criterion;
both raw rewards stay in the results so reporting is unchanged.
"""

from __future__ import annotations

import json

import pytest

from cwsr import Regressor
from cwsr.exp_queue import combine_rewards
from cwsr.regressor import DEFAULT_VALID_REWARD_WEIGHT
from tests.fixtures import synthetic as S

#: Smallest configuration that still runs a real search.
TINY = dict(var_count=1, ops=["add", "mul"], max_depth=3, num_batches=3,
            num_trials=1, num_parallel=1, optimization_method="LD_LBFGS")


def _fit(alpha, *, max_expressions=4, seed=0, n=24):
    X, y, _ = S.tiny_linear_problem(n=n)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      max_expressions=max_expressions, seed=seed,
                      valid_reward_weight=alpha, **TINY)
    outputs = model.fit(seed=seed)[4]
    return model, outputs


# ---------------------------------------------------------------------------
# The knob itself
# ---------------------------------------------------------------------------
def test_default_ranks_by_the_validation_reward_alone():
    """The shipped default is "full validation MAE": α = 1.0 (mixing is opt-in)."""
    assert DEFAULT_VALID_REWARD_WEIGHT == 1.0

    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y, **TINY)

    assert model.valid_reward_weight == 1.0
    mcts = model._create_mcts()
    assert mcts.valid_reward_weight == 1.0
    assert mcts.exp_queue.valid_weight == 1.0
    # ... so the score *is* the validation reward, i.e. valid MAE alone decides
    assert mcts.score(0.9, 0.2) == pytest.approx(0.2)
    # and the mix is an explicit opt-in
    mixed = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      valid_reward_weight=0.5, **TINY)
    assert mixed._create_mcts().score(0.9, 0.2) == pytest.approx(0.55)


@pytest.mark.parametrize("given, expected", [(1.0, 1.0), (0.0, 0.0), (0.25, 0.25),
                                             (1.5, 1.0), (-2.0, 0.0)])
def test_the_coefficient_is_validated_and_clipped(given, expected):
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      valid_reward_weight=given, **TINY)

    assert model.valid_reward_weight == expected


def test_the_coefficient_reaches_the_mcts_and_its_queues():
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      valid_reward_weight=0.25, K=7, **TINY)

    mcts = model._create_mcts()

    assert mcts.valid_reward_weight == 0.25
    assert mcts.exp_queue.valid_weight == 0.25
    assert mcts.path_queue.valid_weight == 0.25
    assert mcts.score(0.4, 0.8) == pytest.approx(0.5)   # 0.25 * 0.8 + 0.75 * 0.4


# ---------------------------------------------------------------------------
# The search really optimises the mixed score
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
def test_search_best_reward_is_the_mixed_score_of_its_best_entry(alpha):
    X, y, _ = S.tiny_linear_problem(n=24)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      max_expressions=4, seed=0, valid_reward_weight=alpha, **TINY)
    mcts = model._create_mcts()

    best_reward = mcts.search(model._create_exp_tree())
    train_reward, valid_reward = mcts.exp_queue.best_reward()

    assert best_reward == pytest.approx(
        combine_rewards(train_reward, valid_reward, alpha))
    if alpha == 1.0:      # the original semantics: valid reward alone
        assert best_reward == pytest.approx(valid_reward)
    if alpha == 0.0:      # training reward alone
        assert best_reward == pytest.approx(train_reward)


def test_trial_selection_uses_the_mixed_score():
    """Within a batch, the kept trial is the one with the better *blend*."""
    X, y, _ = S.tiny_linear_problem(n=12)

    def chosen(alpha):
        model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                          max_expressions=1, seed=0, valid_reward_weight=alpha,
                          **{**TINY, "num_trials": 2, "num_batches": 1})
        mcts = model._create_mcts()
        # trial 1 wins on train, trial 2 wins on valid
        results = [("train_best", 0.9, 0.2, ["x0"]),
                   ("valid_best", 0.2, 0.9, ["x0"])]
        return mcts._select_best_trials(results)[0]

    train_best, valid_best = chosen(0.0), chosen(1.0)
    mixed = chosen(0.5)

    assert train_best[0] == "train_best" and train_best[1] == pytest.approx(0.9)
    assert valid_best[0] == "valid_best" and valid_best[2] == pytest.approx(0.9)
    # at the midpoint both trials score 0.55, so the first one keeps the slot
    assert mixed[0] == "train_best"


def test_warm_start_champions_are_scored_with_the_current_criterion():
    """Loaded results are re-scored with the α in force, not with their raw valid reward."""
    X, y, _ = S.tiny_linear_problem(n=6)
    entry = {"expression": "0.5*x0", "train_reward": 0.4, "valid_reward": 0.45}

    def best_reward(alpha):
        model = Regressor(x_train=X, y_train=y, warm_start=[entry],
                          valid_reward_weight=alpha,
                          **{**TINY, "ops": ["add", "mul", "R"]})
        mcts = model._create_mcts()
        model._seed_from_warm_start(mcts)
        return mcts.best_reward

    # default (α = 1.0): the validation reward decides, so the mixed score *is* 0.45
    assert best_reward(1.0) == pytest.approx(0.45)
    # explicit blend: 0.5 * 0.45 + 0.5 * 0.4
    assert best_reward(0.5) == pytest.approx(0.425)
    assert best_reward(0.0) == pytest.approx(0.4)


def test_ranked_results_keep_both_raw_rewards():
    """Mixing changes the ranking, not the recorded train/valid rewards."""
    _, outputs = _fit(0.3)

    assert outputs
    for entry in outputs:
        assert 0.0 <= entry["train_reward"] <= 1.0
        assert 0.0 <= entry["valid_reward"] <= 1.0
        assert entry["mae"] >= 0.0 and entry["mae_valid"] >= 0.0
        assert set(entry) == {"expression", "weights", "train_reward", "valid_reward",
                              "mae", "mae_valid", "rank"}


# ---------------------------------------------------------------------------
# Persistence of the coefficient
# ---------------------------------------------------------------------------
def test_state_dict_records_the_coefficient():
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      valid_reward_weight=0.2, **TINY)

    assert model.state_dict()["config"]["valid_reward_weight"] == pytest.approx(0.2)


def test_checkpoints_carry_and_restore_the_coefficient(tmp_path):
    prefix = tmp_path / "run"
    X, y, _ = S.tiny_linear_problem(n=12)
    model = Regressor(x_train=X, y_train=y, x_valid=X, y_valid=y,
                      output_prefix=str(prefix), save_checkpoint_every=2,
                      max_expressions=4, seed=0, valid_reward_weight=0.75, **TINY)
    model.fit(seed=0)

    checkpoint = json.loads((tmp_path / "run_ckpt_final.json").read_text())
    assert checkpoint["regressor"]["config"]["valid_reward_weight"] == pytest.approx(0.75)

    restored = Regressor.load(tmp_path / "run_ckpt_final.json")
    assert restored.valid_reward_weight == pytest.approx(0.75)
    assert restored._create_mcts().valid_reward_weight == pytest.approx(0.75)

    # and the override path still works
    overridden = Regressor.load(tmp_path / "run_ckpt_final.json",
                                valid_reward_weight=1.0)
    assert overridden.valid_reward_weight == pytest.approx(1.0)

"""Unit tests for the prioritized expression queue (``cwsr.exp_queue``)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cwsr.exp_queue import (DEFAULT_VALID_REWARD_WEIGHT, Exp_Queue, Queue_Base,
                            combine_rewards)


def test_exp_queue_is_a_queue_base():
    assert issubclass(Exp_Queue, Queue_Base)


def test_empty_queue_contract():
    queue = Exp_Queue(max_size=3)
    assert queue.is_empty() and len(queue) == 0
    assert queue.best() == (None, None)
    assert queue.best_reward() == (0.0, 0.0)
    assert queue.random_sample() == (None, None)


def test_append_orders_entries_by_valid_reward_descending():
    queue = Exp_Queue(max_size=4)
    assert queue.append("a", 0.1, 0.5) is True
    assert queue.append("b", 0.2, 0.9) is True
    assert queue.append("c", 0.3, 0.7) is True

    assert [entry[0] for entry in queue.list] == ["b", "c", "a"]
    assert len(queue) == 3 and not queue.is_empty()
    assert queue.best()[0] == "b"
    assert queue.best_reward() == pytest.approx((0.2, 0.9))
    assert queue.min_reward == pytest.approx(0.5)


def test_near_duplicate_rewards_are_suppressed():
    """Entries whose valid reward differs by < threshold are rejected."""
    queue = Exp_Queue(max_size=4)
    assert queue.append("a", 0.1, 0.5) is True
    assert queue.append("duplicate", 0.1, 0.500005) is False
    assert queue.append("distinct", 0.1, 0.51) is True
    assert len(queue) == 2


@pytest.mark.parametrize("bad_reward", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_rewards_are_rejected(bad_reward):
    queue = Exp_Queue(max_size=2)
    assert queue.append("bad", 0.0, bad_reward) is False
    assert queue.is_empty()


def test_capacity_evicts_the_worst_entry():
    queue = Exp_Queue(max_size=3)
    for name, reward in (("a", 0.5), ("b", 0.9), ("c", 0.7)):
        queue.append(name, 0.0, reward)
    assert [entry[0] for entry in queue.list] == ["b", "c", "a"]

    assert queue.append("d", 0.0, 0.95) is True
    assert [entry[0] for entry in queue.list] == ["d", "b", "c"]
    assert queue.min_reward == pytest.approx(0.7)
    # an entry that cannot beat the current minimum is dropped
    assert queue.append("e", 0.0, 0.1) is False
    assert len(queue) == 3


def test_iteration_and_random_sample_yield_stored_entries():
    queue = Exp_Queue(max_size=3)
    queue.append("a", 0.1, 0.5)
    queue.append("b", 0.2, 0.9)
    assert {entry[0] for entry in queue} == {"a", "b"}

    random.seed(0)
    sample = queue.random_sample()
    assert sample in [(entry) for entry in queue.list]
    assert isinstance(sample[2], float)


def test_rewards_stay_finite_and_sorted_under_many_inserts():
    queue = Exp_Queue(max_size=10)
    rng = np.random.default_rng(0)
    for i, reward in enumerate(rng.uniform(0, 1, size=50)):
        queue.append(f"item{i}", 0.0, float(reward))
    rewards = [entry[2] for entry in queue.list]
    assert len(rewards) == 10
    assert rewards == sorted(rewards, reverse=True)
    assert queue.min_reward == pytest.approx(rewards[-1])


# ---------------------------------------------------------------------------
# Reward mixing: rank by a train/valid blend instead of valid alone
# ---------------------------------------------------------------------------
def test_combine_rewards_blends_train_and_valid():
    # weight 1.0 keeps the original "valid reward only" behaviour ...
    assert combine_rewards(0.2, 0.8, 1.0) == pytest.approx(0.8)
    # ... weight 0.0 is the training reward alone ...
    assert combine_rewards(0.2, 0.8, 0.0) == pytest.approx(0.2)
    # ... and 0.5 mixes them equally
    assert combine_rewards(0.2, 0.8, 0.5) == pytest.approx(0.5)
    assert combine_rewards(0.4, 0.6, 0.25) == pytest.approx(0.45)


@pytest.mark.parametrize("weight, expected", [(1.7, 0.8), (-3.0, 0.2), (2.0, 0.8)])
def test_combine_rewards_clips_the_weight(weight, expected):
    assert combine_rewards(0.2, 0.8, weight) == pytest.approx(expected)


def test_queue_default_weight_ranks_by_valid_reward():
    assert DEFAULT_VALID_REWARD_WEIGHT == 1.0
    queue = Exp_Queue(max_size=3)
    assert queue.valid_weight == 1.0
    queue.append("a", 0.9, 0.1)
    queue.append("b", 0.1, 0.5)
    assert queue.best()[0] == "b"           # valid wins at weight 1.0
    assert queue.score(0.9, 0.1) == pytest.approx(0.1)


def test_queue_can_rank_by_the_training_reward():
    queue = Exp_Queue(max_size=3, valid_weight=0.0)
    queue.append("train_best", 0.9, 0.1)
    queue.append("valid_best", 0.1, 0.5)
    assert queue.best()[0] == "train_best"
    assert queue.min_reward == pytest.approx(0.1)   # the valid-best entry's score


def test_queue_rank_order_depends_on_the_weight():
    # (state, train_reward, valid_reward); note the blends below are pairwise
    # distinct for every weight so the duplicate filter never interferes
    entries = [("both_good", 0.8, 0.8), ("train_heavy", 0.9, 0.3),
               ("valid_heavy", 0.2, 0.9)]

    def order(weight):
        queue = Exp_Queue(max_size=4, valid_weight=weight)
        for state, train, valid in entries:
            assert queue.append(state, train, valid) is True
        return [entry[0] for entry in queue.list]

    assert order(1.0) == ["valid_heavy", "both_good", "train_heavy"]
    assert order(0.0) == ["train_heavy", "both_good", "valid_heavy"]
    assert order(0.5) == ["both_good", "train_heavy", "valid_heavy"]


def test_duplicate_suppression_uses_the_combined_score():
    """Two entries with the same *combined* score are treated as duplicates."""
    mixed = Exp_Queue(max_size=4, valid_weight=0.5)
    assert mixed.append("low_valid", 0.8, 0.2) is True
    assert mixed.append("high_valid", 0.2, 0.8) is False   # identical blend (0.5)
    assert len(mixed) == 1

    # ... while the pure-valid queue keeps them apart
    valid_only = Exp_Queue(max_size=4, valid_weight=1.0)
    assert valid_only.append("low_valid", 0.8, 0.2) is True
    assert valid_only.append("high_valid", 0.2, 0.8) is True
    assert len(valid_only) == 2


def test_capacity_eviction_uses_the_combined_score():
    queue = Exp_Queue(max_size=2, valid_weight=0.5)
    queue.append("weak", 0.1, 0.2)        # blend 0.15
    queue.append("strong", 0.4, 0.4)      # blend 0.40
    assert queue.append("stronger", 0.5, 0.5) is True     # blend 0.50 evicts "weak"
    assert [entry[0] for entry in queue.list] == ["stronger", "strong"]
    assert queue.append("weakest", 0.0, 0.0) is False

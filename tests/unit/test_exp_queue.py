"""Unit tests for the prioritized expression queue (``cwsr.exp_queue``)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cwsr.exp_queue import Exp_Queue, Queue_Base


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

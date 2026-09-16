"""Unit tests for MCTS checkpointing (``cwsr.checkpoint``).

The serialisation routines only need an MCTS-like object (``root``,
``exp_queue``, ``path_queue``, ``count_num``, ``best_reward``, ``total_nodes``),
so these tests build a tiny stub tree instead of running a search.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Tuple

import pytest

from cwsr.checkpoint import (load_checkpoint, mcts_from_dict, mcts_to_dict,
                             save_checkpoint)
from cwsr.exp_queue import Exp_Queue
from cwsr.mcts import MCTS_Node


def make_stub_mcts(seed_rewards: Tuple[float, ...] = (0.3, 0.6)) -> SimpleNamespace:
    """A two-node MCTS-like object with populated queues."""
    mcts = SimpleNamespace(
        exp_queue=Exp_Queue(max_size=8),
        path_queue=Exp_Queue(max_size=8),
        count_num=7,
        best_reward=0.6,
        total_nodes=1,
        root=None,
    )
    root = MCTS_Node(mcts=mcts, parent=None, move="")
    child = MCTS_Node(mcts=mcts, parent=root, move="add")
    root.children.append(child)
    root.visits, root.value, root.is_terminal = 3, 0.4, False
    root.unexpanded_moves = ["mul", "sub"]
    child.visits, child.value, child.is_terminal = 1, 0.6, True
    child.unexpanded_moves = []
    mcts.root = root
    mcts.total_nodes = 2

    for i, reward in enumerate(seed_rewards):
        mcts.exp_queue.append(f"expr{i}", reward / 2, reward)
        mcts.path_queue.append(["add", f"x{i}"], reward / 2, reward)
    return mcts


def test_mcts_to_dict_captures_state_queues_and_tree():
    mcts = make_stub_mcts()
    data = mcts_to_dict(mcts)

    assert set(data) == {"count_num", "best_reward", "total_nodes",
                         "exp_queue", "path_queue", "tree"}
    assert data["count_num"] == 7
    assert data["best_reward"] == pytest.approx(0.6)
    assert data["total_nodes"] == 2

    assert len(data["tree"]) == 2
    root_entry, child_entry = data["tree"]
    assert root_entry["move"] == "" and root_entry["children_indices"] == [1]
    assert root_entry["unexpanded_moves"] == ["mul", "sub"]
    assert child_entry["move"] == "add" and child_entry["is_terminal"] is True

    # queue entries are serialised with stringified states
    assert data["exp_queue"][0]["state"] == "expr1"       # best valid reward first
    assert data["path_queue"][0]["state"] == ["add", "x1"]


def test_mcts_round_trip_restores_counters_queues_and_tree():
    mcts = make_stub_mcts()
    restored = make_stub_mcts(seed_rewards=())
    restored.exp_queue = mcts.exp_queue
    restored.path_queue = mcts.path_queue

    mcts_from_dict(mcts_to_dict(mcts), restored)

    assert restored.count_num == mcts.count_num
    assert restored.best_reward == pytest.approx(mcts.best_reward)
    assert restored.total_nodes == mcts.total_nodes

    assert len(restored.exp_queue) == len(mcts.exp_queue) == 2
    assert [entry[0] for entry in restored.exp_queue.list] == \
           [entry[0] for entry in mcts.exp_queue.list]
    assert len(restored.path_queue) == 2

    root = restored.root
    assert root.move == "" and len(root.children) == 1
    assert root.visits == 3 and root.value == pytest.approx(0.4)
    assert root.unexpanded_moves == ["mul", "sub"]

    child = root.children[0]
    assert child.parent is root and child.move == "add"
    assert child.visits == 1 and child.is_terminal is True


def test_save_and_load_checkpoint_round_trip(tmp_path):
    mcts = make_stub_mcts()
    path = tmp_path / "checkpoint.json"
    regressor_state = {"var_count": 1, "seed": 0}

    save_checkpoint(str(path), mcts, regressor_state=regressor_state)
    payload = json.loads(path.read_text())

    assert payload["mcts"] == mcts_to_dict(mcts)
    assert payload["regressor"] == regressor_state

    loaded = load_checkpoint(str(path))
    assert loaded["regressor"] == regressor_state
    assert loaded["mcts"]["count_num"] == 7


def test_save_checkpoint_without_regressor_state(tmp_path):
    path = tmp_path / "checkpoint_only_mcts.json"
    save_checkpoint(str(path), make_stub_mcts())
    assert set(json.loads(path.read_text())) == {"mcts"}


def test_rebuilding_an_empty_tree_returns_none():
    from cwsr.checkpoint import _rebuild_tree

    assert _rebuild_tree([], make_stub_mcts()) is None

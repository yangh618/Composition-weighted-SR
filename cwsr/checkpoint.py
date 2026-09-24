"""Checkpoint save/load utilities for MCTS and Regressor state."""

import json
from collections import deque
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np


def _node_to_dict(node) -> Dict[str, Any]:
    """Serialize an MCTS_Node to a dict (without parent reference)."""
    return {
        "move": node.move,
        "visits": node.visits,
        "value": node.value,
        "is_terminal": node.is_terminal,
        "unexpanded_moves": list(node.unexpanded_moves),
        "children_indices": [],  # filled in by _flatten_tree
    }


def _flatten_tree(root_node) -> List[Dict[str, Any]]:
    """Flatten the MCTS tree into a list of node dicts via BFS.

    Nodes are stored in level order and ``children_indices`` records the
    position of each child in the returned list. Because the traversal is
    breadth-first, a child always lands *after* the nodes that were already
    queued, so its index is ``len(nodes) + len(queue)`` at enqueue time; every
    node is therefore claimed by exactly one parent and a node's child indices
    are strictly increasing. :func:`_rebuild_tree` relies on those invariants.
    """
    nodes: List[Dict[str, Any]] = []
    queue = deque([root_node])
    while queue:
        current = queue.popleft()
        node_dict = _node_to_dict(current)
        nodes.append(node_dict)
        children_indices = []
        for child in current.children:
            # everything still in the queue is written to ``nodes`` before this
            # child, hence the offset by ``len(queue)``
            children_indices.append(len(nodes) + len(queue))
            queue.append(child)
        node_dict["children_indices"] = children_indices
    return nodes


def _indices_are_consistent(nodes: List[Dict[str, Any]]) -> bool:
    """True when ``children_indices`` describes a tree in BFS order.

    A breadth-first flattening guarantees that a child index is strictly
    greater than its parent's position, in range, and claimed by a single
    parent. Checkpoints written by the buggy flattening (every child of a node
    collapsed onto the same index) break the last two rules, so they are
    detected here and recovered by :func:`_children_indices_for`.
    """
    seen = set()
    for i, nd in enumerate(nodes):
        for child_idx in nd.get("children_indices", ()):
            if isinstance(child_idx, bool) or not isinstance(child_idx, int):
                return False
            if child_idx <= i or child_idx >= len(nodes) or child_idx in seen:
                return False
            seen.add(child_idx)
    return True


def _children_indices_for(nodes: List[Dict[str, Any]]) -> List[List[int]]:
    """Child indices for every node, reconstructed when the stored ones are broken.

    Node dicts only carry the *count* of children reliably (the list length),
    and the flat list is always in BFS order, so the original topology can be
    recovered when ``children_indices`` cannot be trusted: the children of a
    node are the next unclaimed entries of the list.
    """
    if _indices_are_consistent(nodes):
        return [list(nd.get("children_indices", ())) for nd in nodes]

    indices: List[List[int]] = []
    next_index = 1
    for nd in nodes:
        count = len(nd.get("children_indices", ()))
        indices.append(list(range(next_index, next_index + count)))
        next_index += count
    return indices


def _rebuild_tree(nodes: List[Dict[str, Any]], mcts_instance):
    """Rebuild an MCTS tree from a flat list of node dicts."""
    from cwsr.mcts import MCTS_Node

    if not nodes:
        return None

    children_indices = _children_indices_for(nodes)

    # Create all nodes first
    mcts_nodes = [
        MCTS_Node(mcts=mcts_instance, parent=None, move=nd["move"])
        for nd in nodes
    ]

    # Wire up parent/children and restore attributes
    for i, nd in enumerate(nodes):
        node = mcts_nodes[i]
        node.visits = nd["visits"]
        node.value = nd["value"]
        node.is_terminal = nd["is_terminal"]
        node.unexpanded_moves = list(nd["unexpanded_moves"])
        for child_idx in children_indices[i]:
            if not 0 < child_idx < len(mcts_nodes):
                continue  # truncated/hand-edited data: drop the dangling child
            child = mcts_nodes[child_idx]
            child.parent = node
            node.children.append(child)

    return mcts_nodes[0]


def mcts_to_dict(mcts) -> Dict[str, Any]:
    """Serialize an MCTS instance to a dictionary."""
    # Serialize queues: each entry is (state, train_reward, valid_reward)
    exp_queue_data = [
        {"state": str(entry[0]), "train_reward": float(entry[1]), "valid_reward": float(entry[2])}
        for entry in mcts.exp_queue.list
    ]
    path_queue_data = [
        {"state": list(entry[0]), "train_reward": float(entry[1]), "valid_reward": float(entry[2])}
        for entry in mcts.path_queue.list
    ]

    tree_data = _flatten_tree(mcts.root)

    return {
        "count_num": int(mcts.count_num),
        "best_reward": float(mcts.best_reward),
        "total_nodes": int(mcts.total_nodes),
        "exp_queue": exp_queue_data,
        "path_queue": path_queue_data,
        "tree": tree_data,
    }


def mcts_from_dict(data: Dict[str, Any], mcts_instance):
    """Restore an MCTS instance from a dictionary (modifies mcts_instance in-place)."""
    from sortedcontainers import SortedList

    mcts_instance.count_num = data["count_num"]
    mcts_instance.best_reward = data["best_reward"]
    mcts_instance.total_nodes = data["total_nodes"]

    # Restore exp_queue
    mcts_instance.exp_queue.list.clear()
    mcts_instance.exp_queue._reward_values.clear()
    mcts_instance.exp_queue.min_reward = float("-inf")
    for entry in data["exp_queue"]:
        mcts_instance.exp_queue.append(
            entry["state"], entry["train_reward"], entry["valid_reward"]
        )

    # Restore path_queue
    mcts_instance.path_queue.list.clear()
    mcts_instance.path_queue._reward_values.clear()
    mcts_instance.path_queue.min_reward = float("-inf")
    for entry in data["path_queue"]:
        mcts_instance.path_queue.append(
            entry["state"], entry["train_reward"], entry["valid_reward"]
        )

    # Restore tree
    root = _rebuild_tree(data["tree"], mcts_instance)
    mcts_instance.root = root


def save_checkpoint(filepath: str, mcts, regressor_state: Optional[Dict] = None):
    """Save MCTS and optional Regressor state to a JSON file."""
    checkpoint = {
        "mcts": mcts_to_dict(mcts),
    }
    if regressor_state is not None:
        checkpoint["regressor"] = regressor_state

    with open(filepath, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load a checkpoint dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

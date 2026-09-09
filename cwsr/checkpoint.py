"""Checkpoint save/load utilities for MCTS and Regressor state."""

import json
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
        "children_indices": [],  # filled later
    }


def _flatten_tree(root_node) -> List[Dict[str, Any]]:
    """Flatten the MCTS tree into a list of node dicts via BFS."""
    nodes = []
    index_map = {id(root_node): 0}
    queue = [root_node]
    while queue:
        current = queue.pop(0)
        node_dict = _node_to_dict(current)
        nodes.append(node_dict)
        for child in current.children:
            index_map[id(child)] = len(nodes)
            queue.append(child)
        # Set children indices for the current node
        node_dict["children_indices"] = [index_map[id(child)] for child in current.children]
    return nodes


def _rebuild_tree(nodes: List[Dict[str, Any]], mcts_instance):
    """Rebuild an MCTS tree from a flat list of node dicts."""
    from cwsr.mcts import MCTS_Node

    if not nodes:
        return None

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
        for child_idx in nd["children_indices"]:
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

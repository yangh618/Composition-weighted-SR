"""Unit tests for the expression tree (``cwsr.exp_tree``)."""

from __future__ import annotations

import random

import pytest

from cwsr.exp_tree import ExpTree, ExpTreeBase

ARITY = {"add": 2, "mul": 2, "sqrt": 1, "R": 0, "x0": 0, "x1": 0}
COMPLEXITY = {op: 1.0 for op in ARITY}
OPS = list(ARITY)


def make_tree(max_depth: int = 4,
              max_constants: int = 2,
              max_single_arity_ops: int = 2) -> ExpTree:
    """A small tree over ``add``/``mul``/``sqrt``/``R``/``x0``/``x1``."""
    return ExpTree(
        max_depth=max_depth,
        max_single_arity_ops=max_single_arity_ops,
        max_constants=max_constants,
        arity_dict=dict(ARITY),
        complexity=dict(COMPLEXITY),
        ops=list(OPS),
    )


def test_is_exp_tree_base_subclass():
    assert issubclass(ExpTree, ExpTreeBase)


def test_initial_state():
    tree = make_tree()
    assert tree.is_empty() and tree.is_full() and not tree.is_terminal()
    assert tree.depth == 0 and tree.length == 0
    assert sorted(tree.available_ops) == sorted(OPS)


def test_manual_construction_emits_infix_expression():
    tree = make_tree()
    tree.add_op("add")
    assert tree.root_op == "add" and tree.depth == 1 and not tree.is_terminal()

    tree.add_op("x0")
    tree.add_op("x1")
    assert tree.is_terminal()
    assert tree.get_expression() == "(x0 + x1)"
    assert tree.length == 3


def test_nested_unary_and_binary_construction():
    tree = make_tree()
    for op in ("mul", "sqrt", "x0", "add", "x1", "R"):
        tree.add_op(op)
    assert tree.is_terminal()
    assert tree.get_expression() == "(sqrt(x0) * (x1 + R0))"


def test_max_depth_restricts_available_ops_to_leaves():
    tree = make_tree(max_depth=2)
    tree.add_op("add")
    assert sorted(tree.available_ops) == ["R", "x0", "x1"]


def test_unknown_operator_is_rejected():
    tree = make_tree()
    with pytest.raises(ValueError, match="Invalid op"):
        tree.add_op("bogus")


def test_constant_budget_is_enforced_by_available_ops():
    """With ``max_constants=1`` a second constant is rejected.

    Current behaviour: the budget is enforced by filtering ``available_ops``
    (plus a defensive check in ``add_op_common``), so the error is reported as
    an invalid-op ``ValueError`` rather than a dedicated "limit reached" one.
    """
    tree = make_tree(max_constants=1)
    tree.add_op("add")
    tree.add_op("R")
    assert "R" not in tree.available_ops
    with pytest.raises(ValueError, match="Invalid op"):
        tree.add_op("R")


def test_single_arity_budget_is_enforced_by_available_ops():
    tree = make_tree(max_single_arity_ops=1)
    tree.add_op("add")
    tree.add_op("sqrt")
    assert "sqrt" not in tree.available_ops
    with pytest.raises(ValueError, match="Invalid op"):
        tree.add_op("sqrt")


def test_get_expression_requires_a_complete_tree():
    tree = make_tree()
    with pytest.raises(ValueError, match="incomplete or empty"):
        tree.get_expression()
    tree.add_op("add")
    with pytest.raises(ValueError, match="incomplete or empty"):
        tree.get_expression()


def test_random_fill_returns_state_ops_and_complexity():
    """``random_fill`` returns ``(state, op_list, complexity)``.

    Note: the signature is annotated ``-> List[str]`` but the engine unpacks the
    3-tuple (``cwsr/mcts.py``: ``filled_state, path, complex = ...``); the tuple
    contract is the one that must hold.
    """
    random.seed(0)
    tree = make_tree()
    result = tree.random_fill()

    assert isinstance(result, tuple) and len(result) == 3
    filled_state, op_list, complexity = result
    assert filled_state is tree
    assert isinstance(op_list, list) and op_list
    assert complexity == pytest.approx(sum(COMPLEXITY[op] for op in op_list))
    assert tree.is_terminal()
    assert tree.get_expression()


def test_random_fill_is_reproducible_under_a_fixed_python_seed():
    random.seed(0)
    first = make_tree()
    first.random_fill()
    random.seed(0)
    second = make_tree()
    second.random_fill()

    assert first.op_list == second.op_list
    assert first.get_expression() == second.get_expression()


def test_clear_resets_the_tree():
    random.seed(0)
    tree = make_tree()
    tree.random_fill()
    tree.clear()

    assert tree.is_empty() and not tree.is_terminal()
    assert tree.depth == 0 and tree.length == 0 and tree.constant_count == 0
    assert sorted(tree.available_ops) == sorted(OPS)

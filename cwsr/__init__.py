"""Composition-weighted Symbolic Regression (CWSR) — unified engine + framework.

The ``cwsr`` package bundles the CWSR symbolic-regression *engine* (originally
built on iMCTS but since heavily reworked and flattened here) together with a
task-agnostic framework for training, evaluation, querying, inverse design,
Pareto optimisation and bootstrap UQ over any composition -> property dataset.

Convenience imports::

    from cwsr import Regressor, simplify_expression
    from cwsr.datasets import get_dataset, CompositionDataset
    from cwsr.mcts import MCTS
    from cwsr.exp_tree import ExpTree
    from cwsr.exp_queue import Exp_Queue
    from cwsr.reward import Optimizer, sp_module
"""

from cwsr import datasets  # noqa: F401
from cwsr.regressor import Regressor, simplify_expression  # noqa: F401

__all__ = [
    "datasets",
    "Regressor",
    "simplify_expression",
]
__version__ = "0.1.0"


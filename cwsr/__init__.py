"""Composition-weighted Symbolic Regression (CWSR) — unified engine + framework.

The ``cwsr`` package bundles the CWSR symbolic-regression *engine* — an
MCTS-driven search core — together with a
task-agnostic framework for training, evaluation, querying, inverse design,
Pareto optimisation and bootstrap UQ over any composition -> property dataset.

The task-agnostic data layer lives in the separate top-level ``datasets``
package (see ``datasets.get_dataset``).

Convenience imports::

    from cwsr import Regressor, simplify_expression
    from datasets import get_dataset, CompositionDataset
    from cwsr.mcts import MCTS
    from cwsr.exp_tree import ExpTree
    from cwsr.exp_queue import Exp_Queue
    from cwsr.reward import Optimizer, sp_module
"""

from cwsr.regressor import Regressor, simplify_expression  # noqa: F401

__all__ = [
    "Regressor",
    "simplify_expression",
]
__version__ = "0.1.0"


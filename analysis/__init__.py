"""Analysis tools built on the CWSR framework.

A task-agnostic layer for post-training analysis — the counterpart of
:mod:`cwsr` (engine + core framework) and :mod:`datasets` (data providers):

* :mod:`analysis.inverse` — composition design from a discovered expression:
  hit a target value, or optimise a weighted-sum / augmented weighted
  Tchebycheff scalarization of several properties;
* :mod:`analysis.pareto` — analytical 2-objective Pareto front from two
  discovered expressions;
* :mod:`analysis.bootstrap` — bootstrap resampling UQ over the training set and
  selection of the most robust expression across bootstrap runs.

Every entry point consumes the same artifacts as the rest of the framework
(``cwsr_outputs_*.json`` / refined-results JSON) and a dataset name or path
resolved through :func:`datasets.get_dataset`, so nothing here is
alloy- or Matbench-specific.

Console scripts: ``cwsr-inverse``, ``cwsr-pareto``, ``cwsr-bootstrap``.

Submodules are imported lazily (they pull in scipy/joblib), so ``import
analysis`` stays cheap::

    from analysis import inverse, pareto, bootstrap
"""

from __future__ import annotations

import importlib
from typing import List

__all__: List[str] = ["bootstrap", "inverse", "pareto"]

_SUBMODULES = ("bootstrap", "inverse", "pareto")


def __getattr__(name: str):
    """Lazily import the submodules on first attribute access (PEP 562)."""
    if name in _SUBMODULES:
        return importlib.import_module(f"analysis.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES))

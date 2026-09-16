"""Shared helpers for the :mod:`analysis` tools (inverse / Pareto / bootstrap).

Small, dependency-light utilities that all three components need: loading a
discovered expression plus its tabulated element weights from a CWSR results
JSON, resolving the active-element index set, and writing result payloads.

These helpers know nothing about alloys or Matbench — only about the framework
conventions (``cwsr_outputs_*.json`` / refined-results files whose entries carry
``expression`` and ``weights``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from cwsr.predict.forward import ELEMENT_SYMBOLS, _get_active_indices

__all__ = [
    "load_expression_from_results",
    "load_expression_specs",
    "resolve_active_indices",
    "describe_elements",
    "present_elements",
    "save_json",
]


def load_expression_from_results(results_path: "str | Path",
                                 index: int = 0,
                                 ) -> Tuple[str, np.ndarray]:
    """Load ``(expression, weights)`` from a CWSR results JSON.

    Parameters
    ----------
    results_path : str | Path
        JSON written by :func:`cwsr.model.fit_dataset` / ``train`` (a ranked
        list of output dicts), a refined-results file with the same shape, or a
        single output dict.
    index : int
        Which entry to take (``0`` = best/rank 1).

    Returns
    -------
    (expression, weights) : (str, np.ndarray)
        The expression string and its ``(var_count, 118)`` element weights.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):          # a single output dict
        data = [data]
    if len(data) == 0:
        raise IndexError(f"No expression entries in {path}")
    if index >= len(data):
        raise IndexError(
            f"Expression index {index} out of range for {path} "
            f"(has {len(data)} entries)"
        )

    entry = data[index]
    return entry["expression"], np.asarray(entry["weights"], dtype=np.float64)


def load_expression_specs(specs: Sequence[Tuple[str, "str | Path"]],
                          index: int = 0,
                          ) -> List[dict]:
    """Load several ``(name, results_path)`` pairs into model dicts.

    Returns a list of ``{"name", "path", "expression", "weights"}`` dicts, in
    the order given — the shared input format of the inverse-design and Pareto
    routines in this package.
    """
    models = []
    for name, path in specs:
        expression, weights = load_expression_from_results(path, index)
        models.append({"name": name, "path": str(path),
                       "expression": expression, "weights": weights})
    return models


def resolve_active_indices(elements: Optional[Sequence] = None) -> np.ndarray:
    """Indices of the elements to optimize over (default: all 118).

    ``elements`` entries may be atomic numbers (``22``) or symbols (``"Ti"``).
    """
    if elements is None:
        return np.arange(118, dtype=int)
    return _get_active_indices(list(elements))


def describe_elements(indices: Sequence[int]) -> str:
    """``'Ti(Z=22), V(Z=23)'``-style labels for element indices."""
    return ", ".join(
        f"{ELEMENT_SYMBOLS[int(i)]}(Z={int(i) + 1})" for i in indices
    )


def present_elements(compositions: np.ndarray) -> np.ndarray:
    """Indices of elements that actually occur in ``(n, 118)`` compositions."""
    return np.flatnonzero(np.asarray(compositions, dtype=np.float64).sum(axis=0) > 0)


def save_json(path: "str | Path", payload: Any) -> Path:
    """Write ``payload`` as indented JSON (creating parent directories)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

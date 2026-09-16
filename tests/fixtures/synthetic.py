"""Shared synthetic fixtures for the CWSR test suite.

Nothing here touches the network, the user's home directory, large datasets or
the GPU: everything is a small, deterministic, hand-checkable synthetic problem
built from :func:`numpy.random.default_rng` with fixed seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Element columns used by the synthetic problems (0-based, H = 0 .. Og = 117).
FE, CO, NI, CR, AL, O, H = 25, 26, 27, 23, 12, 7, 0

#: Symbols accepted by ``--elements`` / ``_parse_elements``.
ELEMENT_INDEX: Dict[str, int] = {
    "H": H, "O": O, "Al": AL, "Cr": CR, "Fe": FE, "Co": CO, "Ni": NI,
}


def compositions_from_fractions(fractions: Dict[str, float]) -> np.ndarray:
    """Build a single normalized (118,) composition vector from fractions.

    >>> compositions_from_fractions({"Fe": 0.5, "Ni": 0.5})[FE]
    0.5
    """
    vec = np.zeros(118, dtype=np.float64)
    for symbol, fraction in fractions.items():
        vec[ELEMENT_INDEX[symbol]] = fraction
    return vec / vec.sum()


def composition_matrix(n: int,
                       elements: Sequence[str],
                       seed: int = 0) -> np.ndarray:
    """``(n, 118)`` matrix of random Dirichlet mixtures over ``elements``."""
    rng = np.random.default_rng(seed)
    cols = [ELEMENT_INDEX[symbol] for symbol in elements]
    matrix = np.zeros((n, 118), dtype=np.float64)
    matrix[:, cols] = rng.dirichlet(np.ones(len(cols)), size=n)
    return matrix


def linear_targets(compositions: np.ndarray,
                   weights: np.ndarray) -> np.ndarray:
    """Targets that are exactly linear in the composition: ``y = X @ weights``."""
    return compositions @ np.asarray(weights, dtype=np.float64)


def selector_weights(rows: Sequence[Sequence[float]],
                     cols: Sequence[int],
                     var_count: int | None = None) -> np.ndarray:
    """``(var_count, 118)`` weight matrix with explicit per-row column values."""
    var_count = len(rows) if var_count is None else var_count
    weights = np.zeros((var_count, 118), dtype=np.float64)
    for row, values in enumerate(rows):
        for col, value in zip(cols, values):
            weights[row, col] = value
    return weights


def element_selector_weights(symbols: Sequence[str]) -> np.ndarray:
    """``(len(symbols), 118)`` weights picking one element fraction per row.

    ``W = element_selector_weights(["Fe", "Ni"])`` makes ``pred = x0 + x1``
    equal ``Fe_fraction + Ni_fraction``.
    """
    weights = np.zeros((len(symbols), 118), dtype=np.float64)
    for row, symbol in enumerate(symbols):
        weights[row, ELEMENT_INDEX[symbol]] = 1.0
    return weights


def write_refined_results(path: Path,
                          expression: str,
                          weights: np.ndarray,
                          **extra) -> Path:
    """Write a one-entry refined-results JSON (the file ``cwsr-query`` reads)."""
    entry = {
        "expression": expression,
        "weights": np.asarray(weights, dtype=np.float64).tolist(),
        "mae": 0.0,
        "mae_valid": 0.0,
        "rank": 1,
    }
    entry.update(extra)
    path.write_text(json.dumps([entry], indent=2))
    return path


def write_npz_database(path: Path,
                       formulas: List[str],
                       targets: List[float],
                       target_name: str = "toy property",
                       source: str = "unit-test") -> Path:
    """Write a processed database in the documented ``.npz`` layout."""
    np.savez(
        path,
        targets=np.asarray(targets, dtype=np.float64),
        formulas=np.asarray(formulas),
        target_name=target_name,
        source=source,
    )
    return path


def tiny_linear_problem(n: int = 24,
                        elements: Tuple[str, ...] = ("Fe", "Co", "Ni", "Cr"),
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiny exactly-linear regression problem the engine can recover.

    Returns ``(X, y, w_true)`` with ``y = X @ w_true`` exactly, so a successful
    search must reach MAE ≈ 0 with the expression ``x0``.
    """
    compositions = composition_matrix(n, elements, seed=seed)
    weights = np.zeros(118, dtype=np.float64)
    weights[[ELEMENT_INDEX[symbol] for symbol in elements]] = [0.3, 0.5, 0.2, 0.1]
    return compositions, linear_targets(compositions, weights), weights

"""Core dataset schema and train/valid split logic (framework-wide).

A :class:`CompositionDataset` is the single, framework-wide description of any
composition -> property task. Downstream modules (train, eval, query, inverse,
Pareto, bootstrap) are written against this schema only, never against a
specific database layout, which is what makes the framework task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CompositionDataset:
    """A normalized composition -> target dataset.

    Attributes
    ----------
    name : str
        Stable identifier, e.g. ``"matbench_expt_gap"`` or ``"alloy_density"``.
    compositions : np.ndarray, shape (n, 118)
        Atomic-fraction vectors, one column per element (H..Og).
    targets : np.ndarray, shape (n,)
        Target property values.
    formulas : list[str] | None
        Optional source chemical formulas per sample.
    target_name : str
        Property name / units, used for labels.
    source : str
        Provenance description.
    meta : dict
        Free-form metadata (``var_count``, units, task category, ...).
    """

    name: str
    compositions: np.ndarray
    targets: np.ndarray
    formulas: Optional[List[str]] = None
    target_name: str = ""
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.compositions = np.asarray(self.compositions, dtype=np.float64)
        self.targets = np.asarray(self.targets, dtype=np.float64)
        if self.compositions.ndim != 2:
            raise ValueError(f"compositions must be 2D, got {self.compositions.shape}")
        if self.compositions.shape[0] != self.targets.shape[0]:
            raise ValueError(
                "compositions/targets length mismatch: "
                f"{self.compositions.shape[0]} vs {self.targets.shape[0]}"
            )

    @property
    def n_features(self) -> int:
        return self.compositions.shape[1]

    @property
    def n_samples(self) -> int:
        return self.compositions.shape[0]

    def __len__(self) -> int:
        return self.n_samples


def ensure_train_covers_species(compositions: np.ndarray,
                                train_indices: np.ndarray,
                                valid_indices: np.ndarray,
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Force any 'singleton' species into the training split.

    A species (composition column) is a singleton if it appears in exactly one
    sample of the whole dataset. Because CWSR uses per-element tabulated
    weights it must see every element during training, so such samples must not
    be allowed to land in the validation split.

    Returns adjusted (train_indices, valid_indices).
    """
    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)

    # Columns that are nonzero in exactly one sample.
    present = compositions > 0.0
    col_counts = present.sum(axis=0)
    singleton_cols = np.where(col_counts == 1)[0]
    if singleton_cols.size == 0:
        return train_indices, valid_indices

    # Rows that hold a singleton species.
    singleton_rows = np.where(present[:, singleton_cols].any(axis=1))[0]

    # Move any singleton-row currently in the validation set into training.
    move = np.isin(valid_indices, singleton_rows)
    if move.any():
        moved = valid_indices[move]
        valid_indices = valid_indices[~move]
        train_indices = np.concatenate([train_indices, moved])
    return train_indices, valid_indices


def split_dataset(compositions: np.ndarray,
                  targets: np.ndarray,
                  ratio: float = 0.8,
                  seed: Optional[int] = None,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/valid split ensuring all species are represented.

    Parameters
    ----------
    compositions : np.ndarray, shape (n, 118)
    targets : np.ndarray, shape (n,)
    ratio : float
        Fraction of samples used for training (0 < ratio < 1).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    (train_indices, valid_indices) : (np.ndarray, np.ndarray)
    """
    n = compositions.shape[0]
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(round(ratio * n)))
    train_indices = perm[:n_train]
    valid_indices = perm[n_train:]

    return ensure_train_covers_species(compositions, train_indices, valid_indices)

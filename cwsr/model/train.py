"""Unified training driver.

Runs the CWSR search on any :class:`~cwsr.datasets.base.CompositionDataset`
(Matbench, alloy databases, custom .npz) via the shared :class:`cwsr.Regressor`
engine, saving the same ``cwsr_params_*`` / ``cwsr_outputs_*`` JSON artifacts
used by the rest of the framework (eval/refinement, query, bootstrap, ...).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from cwsr.datasets.base import CompositionDataset, split_dataset

__all__ = ["train", "fit_dataset"]


def train(compositions: np.ndarray,
          targets: np.ndarray,
          output_prefix: str,
          split_ratio: float = 0.8,
          seed: Optional[int] = None,
          **regressor_kwargs,
          ) -> dict:
    """Split the data, run the CWSR search and save params+outputs JSON.

    Returns the ``outputs`` dict from ``Regressor.fit`` (also written to
    ``<output_prefix>.json``).
    """
    from cwsr import Regressor

    # Ensure the training split covers every element present in the data.
    train_idx, valid_idx = split_dataset(compositions, targets,
                                         ratio=split_ratio, seed=seed)

    model = Regressor(
        x_train=compositions[train_idx],
        y_train=targets[train_idx],
        x_valid=compositions[valid_idx],
        y_valid=targets[valid_idx],
        seed=seed,
        output_prefix=output_prefix,
        **regressor_kwargs,
    )
    checkpoint = None
    sym_exp, vec_exp, evaluations, path, outputs = model.fit(
        seed=seed, checkpoint=checkpoint
    )

    with open(f"{output_prefix}.json", "w") as f:
        json.dump(outputs, f, indent=2)
    return outputs


def fit_dataset(dataset: CompositionDataset,
                output_dir: "str | Path" = ".",
                split_ratio: float = 0.8,
                seed: Optional[int] = None,
                **regressor_kwargs) -> dict:
    """Train on a :class:`CompositionDataset`, saving results under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    prefix = output_dir / f"cwsr_outputs_{dataset.name}_{timestamp}"
    return train(dataset.compositions, dataset.targets,
                 output_prefix=str(prefix),
                 split_ratio=split_ratio, seed=seed, **regressor_kwargs)

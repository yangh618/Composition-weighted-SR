"""Matbench dataset provider for the unified framework.

Wraps the Matbench benchmark tasks (composition/property) into
:class:`~cwsr.datasets.base.CompositionDataset` objects so the same unified
training/eval code can run on them. Matbench data is downloaded and cached by
the `matbench` package itself.
"""

from __future__ import annotations

from typing import List, Optional

from cwsr.datasets.base import CompositionDataset

# Valid Matbench task keys (composition-based).
MATBENCH_TASKS: List[str] = [
    "matbench_dielectric", "matbench_expt_gap", "matbench_expt_is_metal",
    "matbench_glass", "matbench_jdft2d", "matbench_log_gvrh",
    "matbench_log_kvrh", "matbench_mp_e_form", "matbench_mp_gap",
    "matbench_mp_is_metal", "matbench_perovskites", "matbench_phonons",
    "matbench_steels",
]


def load_matbench_dataset(task_key: str,
                          fold: int = 0,
                          split: str = "train",
                          include_target: bool = True,
                          ) -> CompositionDataset:
    """Build a :class:`CompositionDataset` from a Matbench task.

    Parameters
    ----------
    task_key : str
        One of :data:`MATBENCH_TASKS`.
    fold : int
        Cross-validation fold (0-4).
    split : str
        ``"train"`` (train+validation) or ``"test"``.
    include_target : bool
        Whether to load test targets (only relevant for ``split="test"``).

    Notes
    -----
    Loading requires the optional `matbench` package and its (legacy) data
    stack, which is intentionally not part of the core install; see
    ``requirements.txt``.
    """
    # Lazy import so the core package does not hard-require matbench.
    from dataloader import load_matbench, load_matbench_test

    if task_key not in MATBENCH_TASKS:
        raise ValueError(f"Unknown matbench task '{task_key}'. "
                         f"Valid: {MATBENCH_TASKS}")

    if split == "train":
        compositions, targets = load_matbench(task_key, fold=fold)
        formulas = None
    else:
        compositions, targets = load_matbench_test(
            task_key, fold=fold, include_target=include_target
        )
        formulas = None

    meta = {"dataset": "matbench", "task": task_key, "fold": fold,
            "split": split, "var_count": None}
    return CompositionDataset(
        name=task_key,
        compositions=compositions,
        targets=targets,
        formulas=formulas,
        target_name=task_key,
        source=f"matbench:{task_key}",
        meta=meta,
    )


def matbench_provider(task: str,
                      fold: int = 0,
                      **kwargs) -> CompositionDataset:
    return load_matbench_dataset(task, fold=fold, **kwargs)

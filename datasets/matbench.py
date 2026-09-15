"""Matbench dataset provider for the unified framework.

Wraps the Matbench benchmark tasks (composition/property) into
:class:`~cwsr.data.CompositionDataset` objects so the same unified
training/eval code can run on them, and owns the raw Matbench loaders
(:func:`load_matbench` for train+validation, :func:`load_matbench_test` for the
held-out test split). Matbench data is downloaded and cached by the `matbench`
package itself.

Loading needs the optional ``matbench`` package and its (legacy) data stack,
which is intentionally not part of the core install (see ``requirements.txt``);
those imports are therefore deferred to call time.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from cwsr.data import CompositionDataset
from cwsr.formula import form2comp

# Valid Matbench task keys (composition-based).
MATBENCH_TASKS: List[str] = [
    "matbench_dielectric", "matbench_expt_gap", "matbench_expt_is_metal",
    "matbench_glass", "matbench_jdft2d", "matbench_log_gvrh",
    "matbench_log_kvrh", "matbench_mp_e_form", "matbench_mp_gap",
    "matbench_mp_is_metal", "matbench_perovskites", "matbench_phonons",
    "matbench_steels",
]


def _load_task(task_key: str, fold: int):
    """Validate ``task_key`` and return the loaded Matbench task object."""
    if task_key not in MATBENCH_TASKS:
        raise ValueError(f"Invalid task_key '{task_key}'. "
                         f"Must be one of {MATBENCH_TASKS}")

    # Lazy imports so the core package does not hard-require matbench.
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = getattr(mb, task_key)
    task.load()
    return task


def _compositions_from_inputs(inputs) -> np.ndarray:
    """Convert Matbench inputs (structure objects or formula strings) to (n, 118).

    Structure-like entries are converted via ASE atomic numbers; anything else
    is treated as a formula string and parsed with
    :func:`cwsr.formula.form2comp`.
    """
    from tqdm import tqdm

    compositions = np.zeros((len(inputs), 118), dtype=np.float64)
    for i, entry in tqdm(enumerate(inputs), total=len(inputs)):
        if hasattr(entry, "to_ase_atoms"):
            atoms = entry.to_ase_atoms()
            atomic_nums = atoms.get_atomic_numbers()
            # Composition vector from atomic fractions.
            compositions[i] = (np.sum(np.eye(118)[atomic_nums - 1], axis=0)
                               / len(atomic_nums))
        else:
            compositions[i] = form2comp(str(entry))
    return compositions


def load_matbench(task_key: str,
                  fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load train+validation data for a Matbench task/fold.

    Parameters
    ----------
    task_key : str
        One of :data:`MATBENCH_TASKS` (e.g. ``"matbench_mp_gap"``).
    fold : int
        Cross-validation fold (0-4).

    Returns
    -------
    (compositions, targets) : (np.ndarray, np.ndarray)
        ``(n, 118)`` normalized composition vectors and ``(n,)`` targets.
    """
    task = _load_task(task_key, fold)
    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    return _compositions_from_inputs(train_inputs), np.array(train_outputs)


def load_matbench_test(task_key: str,
                       fold: int = 0,
                       include_target: bool = False,
                       ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load the held-out test split for a Matbench task/fold.

    Parameters
    ----------
    task_key : str
        One of :data:`MATBENCH_TASKS`.
    fold : int
        Cross-validation fold (0-4).
    include_target : bool
        Whether to return test targets (``None`` if ``False``).

    Returns
    -------
    (compositions, targets) : (np.ndarray, np.ndarray | None)
        ``(n, 118)`` normalized composition vectors and ``(n,)`` targets.
    """
    task = _load_task(task_key, fold)

    if include_target:
        test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
    else:
        test_inputs = task.get_test_data(fold, include_target=False)
        test_outputs = None

    compositions = _compositions_from_inputs(test_inputs)
    targets = None if test_outputs is None else np.array(test_outputs)
    return compositions, targets


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

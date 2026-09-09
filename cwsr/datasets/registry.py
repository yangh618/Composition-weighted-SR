"""Dataset provider registry (task-agnostic dispatch).

Register any provider (a callable ``(task, **kw) -> CompositionDataset``) under
a name or name-prefix, then resolve datasets with :func:`get_dataset`. This is
the single seam for adding new composition->property databases to the
framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from cwsr.datasets.alloy import DEFAULT_PROPERTIES, load_alloy_dataset
from cwsr.datasets.base import CompositionDataset
from cwsr.datasets.matbench import MATBENCH_TASKS, matbench_provider

Provider = Callable[..., CompositionDataset]

#: Registered providers: name -> callable.
_PROVIDERS: Dict[str, Provider] = {}


def register_provider(name: str, provider: Provider) -> None:
    """Register a dataset provider under ``name``."""
    _PROVIDERS[name] = provider


def _auto_register_builtins() -> None:
    if _PROVIDERS:
        return
    # Matbench: one provider handles all matbench_* tasks.
    for task in MATBENCH_TASKS:
        register_provider(task, matbench_provider)
    # Alloy databases: alloy_<property> names.
    for prop in DEFAULT_PROPERTIES:
        register_provider(f"alloy_{prop}", load_alloy_dataset)


def get_dataset(name: str, *, data_dir: "str | Path" = "processed_data",
                **kwargs) -> CompositionDataset:
    """Resolve ``name`` to a :class:`CompositionDataset`.

    ``name`` may be:
      * a registered task name (``"matbench_expt_gap"``, ``"alloy_density"``);
      * a path to a ``.npz`` file with the standard keys (parsed as an alloy
        database).
    """
    _auto_register_builtins()

    if name in _PROVIDERS:
        # Matbench providers need data_dir untouched; alloy providers use it.
        if name.startswith("alloy_"):
            return _PROVIDERS[name](name[len("alloy_"):], data_dir=data_dir,
                                    **kwargs)
        return _PROVIDERS[name](name, **kwargs)

    # Fallback: treat as a path to a processed npz database.
    p = Path(name)
    if p.exists() and p.suffix == ".npz":
        return load_alloy_dataset(str(p))

    raise ValueError(
        f"Unknown dataset '{name}'. Registered: {list_datasets()}\n"
        "Or pass the path to a processed .npz file."
    )


def list_datasets() -> List[str]:
    """List all currently registered dataset names."""
    _auto_register_builtins()
    return sorted(_PROVIDERS.keys())

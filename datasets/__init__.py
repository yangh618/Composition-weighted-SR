"""Concrete dataset providers, the provider registry, and bundled databases.

The generic container/schema itself lives in the framework core
(:mod:`cwsr.data`); this package holds everything database-specific:

* :mod:`datasets.alloy` — preprocessed alloy ``.npz`` databases (+ the bundled
  ``alloys/data`` databases);
* :mod:`datasets.matbench` — Matbench task provider;
* :mod:`datasets.registry` — name/path -> provider dispatch.

The core schema is re-exported here so ``from datasets import
CompositionDataset`` keeps working.
"""

from cwsr.data import (
    CompositionDataset,
    split_dataset,
    ensure_train_covers_species,
)
from datasets.registry import (
    get_dataset,
    list_datasets,
    register_provider,
)

__all__ = [
    "CompositionDataset",
    "split_dataset",
    "ensure_train_covers_species",
    "get_dataset",
    "list_datasets",
    "register_provider",
]

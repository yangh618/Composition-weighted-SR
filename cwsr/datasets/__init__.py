"""Dataset containers and task-agnostic helpers for the unified CWSR framework."""

from cwsr.datasets.base import (
    CompositionDataset,
    split_dataset,
    ensure_train_covers_species,
)
from cwsr.datasets.registry import (
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

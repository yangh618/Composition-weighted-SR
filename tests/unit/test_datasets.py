"""Unit tests for the dataset registry and providers.

Offline only: no Matbench download, no network access. The Matbench provider is
checked through its declared task list, not by fetching data.
"""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.data import CompositionDataset
from datasets import get_dataset, list_datasets, register_provider
from datasets.alloy import DEFAULT_DATA_DIR, DEFAULT_PROPERTIES, load_alloy_dataset
from datasets.matbench import MATBENCH_TASKS
from tests.fixtures import synthetic as S


def test_registry_lists_the_builtin_providers():
    names = list_datasets()
    assert names == sorted(names)
    assert len(names) >= len(MATBENCH_TASKS) + len(DEFAULT_PROPERTIES)
    assert {"alloy_density", "alloy_hardness", "matbench_glass"} <= set(names)


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset("definitely_not_a_dataset")


def test_register_provider_dispatch():
    """A registered provider receives the *full* registry name as its task."""
    def provider(task, **kwargs):
        return CompositionDataset(name=f"custom_{task}",
                                  compositions=np.zeros((2, 118)),
                                  targets=np.zeros(2),
                                  target_name="custom target")

    register_provider("custom_toy", provider)
    dataset = get_dataset("custom_toy")
    assert dataset.name == "custom_custom_toy"
    assert dataset.n_samples == 2 and dataset.n_features == 118


def test_alloy_provider_reads_the_documented_npz_layout(synthetic_npz):
    dataset = get_dataset(str(synthetic_npz))

    assert dataset.name == "alloy_toy_property"
    assert dataset.n_samples == 4 and dataset.n_features == 118
    assert dataset.target_name == "toy property"
    assert dataset.source == "unit-test"
    assert list(dataset.formulas) == ["FeCoNi", "Fe2O3", "Ni", "CrFe"]
    assert dataset.targets.tolist() == [1.0, 2.0, 3.0, 4.0]
    # compositions are built from the formulas and normalized
    assert np.allclose(dataset.compositions.sum(axis=1), 1.0)


def test_alloy_provider_error_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="Processed data file not found"):
        load_alloy_dataset("density", data_dir=tmp_path)

    # a path that does not exist is not silently accepted by the registry
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset(str(tmp_path / "missing.npz"))


def test_alloy_provider_resolves_property_names_inside_data_dir(tmp_path):
    S.write_npz_database(tmp_path / "density.npz", formulas=["FeNi"],
                         targets=[1.0], target_name="density")
    dataset = load_alloy_dataset("density", data_dir=tmp_path)
    assert dataset.name == "alloy_density" and dataset.n_samples == 1


def test_alloy_property_table_maps_names_to_files():
    assert DEFAULT_PROPERTIES["density"] == "density.npz"
    assert DEFAULT_PROPERTIES["youngs_modulus"] == "youngs_modulus.npz"
    assert set(DEFAULT_PROPERTIES) == {"density", "ductility", "elongation",
                                       "hardness", "melting_temperature",
                                       "yield_strength", "youngs_modulus"}


@pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / DEFAULT_PROPERTIES["density"]).exists(),
    reason="bundled alloy databases are not present in this checkout",
)
def test_bundled_alloy_database_loads():
    dataset = get_dataset("alloy_density")
    assert dataset.name == "alloy_density"
    assert dataset.n_samples > 0 and dataset.n_features == 118
    assert np.allclose(dataset.compositions.sum(axis=1), 1.0)
    assert dataset.target_name  # property label is populated


def test_matbench_tasks_are_declared_without_downloading_anything():
    assert len(MATBENCH_TASKS) == 13
    assert all(task.startswith("matbench_") for task in MATBENCH_TASKS)
    assert set(MATBENCH_TASKS) <= set(list_datasets())

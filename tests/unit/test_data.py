"""Unit tests for the framework data schema and the element-safe split."""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.data import (CompositionDataset, ensure_train_covers_species,
                       split_dataset)
from tests.fixtures import synthetic as S


# ---------------------------------------------------------------------------
# CompositionDataset
# ---------------------------------------------------------------------------
def test_dataset_normalizes_dtypes_and_reports_shapes():
    X = S.composition_matrix(5, ("Fe", "Ni"), seed=1)
    y = np.arange(5, dtype=np.int32)
    dataset = CompositionDataset(name="toy", compositions=X.astype(np.float32),
                                 targets=y)

    assert dataset.compositions.dtype == np.float64
    assert dataset.targets.dtype == np.float64
    assert dataset.compositions.shape == (5, 118)
    assert dataset.n_samples == 5 == len(dataset)
    assert dataset.n_features == 118
    assert dataset.formulas is None and dataset.target_name == "" and dataset.source == ""


def test_dataset_rejects_non_2d_compositions_and_length_mismatch():
    with pytest.raises(ValueError, match="compositions must be 2D"):
        CompositionDataset(name="bad", compositions=np.zeros(118), targets=np.zeros(1))
    with pytest.raises(ValueError, match="length mismatch"):
        CompositionDataset(name="bad", compositions=np.zeros((3, 118)),
                           targets=np.zeros(2))


def test_dataset_meta_is_not_shared_between_instances():
    """``meta`` uses ``default_factory``: instances must not share one dict."""
    a = CompositionDataset(name="a", compositions=np.zeros((1, 118)), targets=np.zeros(1))
    b = CompositionDataset(name="b", compositions=np.zeros((1, 118)), targets=np.zeros(1))
    a.meta["var_count"] = 3
    assert b.meta == {}


# ---------------------------------------------------------------------------
# ensure_train_covers_species
# ---------------------------------------------------------------------------
def test_singleton_species_is_forced_into_training():
    """The documented invariant: a species present in exactly one sample must
    not end up in the validation split (per-element weights need it in training).
    """
    X = np.zeros((4, 118))
    X[:, S.FE] = 1.0
    X[:, S.NI] = 1.0
    X[3, :] = 0.0
    X[3, S.AL] = 1.0          # Al appears exactly once -> singleton row 3

    train, valid = ensure_train_covers_species(X, np.array([0, 1]), np.array([2, 3]))

    assert 3 not in valid
    assert set(train) == {0, 1, 3}
    assert set(valid) == {2}


def test_singleton_already_in_training_leaves_split_unchanged():
    X = np.zeros((3, 118))
    X[:, S.FE] = 1.0
    X[2, :] = 0.0
    X[2, S.AL] = 1.0
    train, valid = ensure_train_covers_species(X, np.array([0, 2]), np.array([1]))
    assert train.tolist() == [0, 2] and valid.tolist() == [1]


def test_no_singleton_species_returns_split_unchanged():
    X = S.composition_matrix(6, ("Fe", "Ni"), seed=2)
    train, valid = ensure_train_covers_species(X, np.array([0, 1, 2]), np.array([3, 4, 5]))
    assert train.tolist() == [0, 1, 2] and valid.tolist() == [3, 4, 5]


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------
def test_split_is_deterministic_for_a_fixed_seed():
    X, y, _ = S.tiny_linear_problem(n=20)
    first = split_dataset(X, y, ratio=0.8, seed=42)
    second = split_dataset(X, y, ratio=0.8, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    other = split_dataset(X, y, ratio=0.8, seed=43)
    assert not np.array_equal(first[0], other[0])


def test_split_is_disjoint_and_covers_every_sample():
    X, y, _ = S.tiny_linear_problem(n=21)
    train, valid = split_dataset(X, y, ratio=0.7, seed=0)
    assert set(train.tolist()).isdisjoint(valid.tolist())
    assert sorted(train.tolist() + valid.tolist()) == list(range(21))
    assert len(train) == round(0.7 * 21)


def test_split_keeps_every_present_element_in_training():
    """Element coverage is the core requirement of the CWSR split."""
    X, y, _ = S.tiny_linear_problem(n=24)
    X[23, :] = 0.0
    X[23, S.AL] = 1.0                     # Al occurs once, at index 23
    train, _ = split_dataset(X, y, ratio=0.5, seed=1)
    assert 23 in train.tolist()
    train_present = (X[train] > 0).any(axis=0)
    assert train_present[S.AL]


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5])
def test_split_rejects_out_of_range_ratio(ratio: float):
    X, y, _ = S.tiny_linear_problem(n=6)
    with pytest.raises(ValueError, match="ratio must be in"):
        split_dataset(X, y, ratio=ratio, seed=0)


def test_split_edge_case_tiny_dataset_still_returns_a_nonempty_train_set():
    X, y, _ = S.tiny_linear_problem(n=3)
    train, valid = split_dataset(X, y, ratio=0.34, seed=0)   # round(1.02) -> 1
    assert len(train) == 1
    assert sorted(train.tolist() + valid.tolist()) == [0, 1, 2]

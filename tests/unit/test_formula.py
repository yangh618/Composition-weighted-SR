"""Unit tests for composition <-> formula handling (``cwsr.formula``)."""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.formula import (ELEMENT_SYMBOLS, composition_to_formula, form2comp,
                          format_composition)
from tests.fixtures import synthetic as S


def test_element_symbols_table():
    assert len(ELEMENT_SYMBOLS) == 118
    assert ELEMENT_SYMBOLS[0] == "H" and ELEMENT_SYMBOLS[117] == "Og"
    assert len(set(ELEMENT_SYMBOLS)) == 118


def test_form2comp_normalizes_to_atomic_fractions():
    vec = form2comp("Fe2O3")
    assert vec.shape == (118,) and vec.dtype == np.float64
    assert vec.sum() == pytest.approx(1.0)
    assert vec[S.FE] == pytest.approx(2 / 5)
    assert vec[S.O] == pytest.approx(3 / 5)
    assert np.count_nonzero(vec) == 2


@pytest.mark.parametrize("formula,index,expected", [
    ("H", 0, 1.0),          # single element -> column 0 holds the full mass
    ("Fe", S.FE, 1.0),
    ("FeCoNi", S.FE, 1 / 3),
])
def test_form2comp_index_mapping(formula, index, expected):
    assert form2comp(formula)[index] == pytest.approx(expected)


def test_form2comp_handles_fractional_stoichiometry():
    """Fractional formulas are normalized (documented in docs/tutorial.md)."""
    vec = form2comp("Al0.25CoCrFeNi")
    assert vec.sum() == pytest.approx(1.0)
    assert vec[S.AL] == pytest.approx(0.25 / 4.25)
    assert vec[S.FE] == pytest.approx(1.0 / 4.25)


@pytest.mark.parametrize("formula,exception", [
    ("not_a_formula", ValueError),
    ("", ValueError),
    ("Xx9", KeyError),      # unknown element symbol -> ASE lookup KeyError
])
def test_form2comp_rejects_invalid_formulas(formula: str, exception):
    """Invalid input always raises, but the exception type follows the source.

    Unparsable strings surface pymatgen's ``ValueError``; a syntactically valid
    formula with an unknown symbol surfaces a ``KeyError`` from the ASE
    atomic-number lookup. Both are documented here as current behaviour.
    """
    with pytest.raises(exception):
        form2comp(formula)


def test_composition_to_formula_round_trip_and_order():
    # elements are emitted in order of increasing atomic number
    assert composition_to_formula(form2comp("FeCoNi")) == "Fe0.3333Co0.3333Ni0.3333"
    assert composition_to_formula(form2comp("FeCrCoNi")) == "Cr0.25Fe0.25Co0.25Ni0.25"


def test_composition_to_formula_writes_unit_fractions_without_coefficient():
    assert composition_to_formula(form2comp("Fe")) == "Fe"


def test_composition_to_formula_threshold_and_empty_vector():
    vec = np.zeros(118)
    vec[S.FE] = 0.5
    vec[S.NI] = 1e-9
    assert composition_to_formula(vec) == "Fe0.5"
    assert composition_to_formula(vec, threshold=0.6) == ""
    assert composition_to_formula(np.zeros(118)) == ""


def test_format_composition_is_sorted_and_limited():
    vec = np.zeros(118)
    vec[S.FE], vec[S.NI], vec[S.CR] = 0.6, 0.3, 0.1
    assert format_composition(vec) == "Fe0.6 Ni0.3 Cr0.1"
    assert format_composition(vec, top_n=2) == "Fe0.6 Ni0.3"
    assert format_composition(vec, threshold=0.2) == "Fe0.6 Ni0.3"


def test_formula_round_trip_preserves_the_element_set():
    for formula in ("FeCoNi", "Fe2O3", "CrFe", "H2O"):
        vec = form2comp(formula)
        rendered = composition_to_formula(vec)
        expected = {ELEMENT_SYMBOLS[i] for i in np.flatnonzero(vec > 1e-6)}
        assert {s for s in expected if s in rendered} == expected, f"{formula}: {rendered}"

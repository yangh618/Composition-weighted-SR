"""Unit tests for the composition parameterisation helpers.

These back the inverse-design / Pareto tools (``analysis.*``): they map element
identifiers to columns and rebuild normalized compositions from the free
parameters of a constrained optimisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cwsr.predict.forward import (ELEMENT_SYMBOLS, _build_composition,
                                  _get_active_indices, _parse_elements,
                                  composition_to_formula)
from tests.fixtures import synthetic as S


def test_parse_elements_accepts_symbols_and_atomic_numbers():
    assert _parse_elements([22, 23]).tolist() == [21, 22]
    assert _parse_elements(["Ti", "v"]).tolist() == [21, 22]
    assert _parse_elements(["Fe", 28]).tolist() == [S.FE, S.NI]   # 28 == Ni


@pytest.mark.parametrize("bad", [["Xx"], [0], [119]])
def test_parse_elements_rejects_unknown_identifiers(bad):
    with pytest.raises(ValueError):
        _parse_elements(bad)


def test_get_active_indices_defaults_to_all_118_and_validates():
    assert np.array_equal(_get_active_indices(None), np.arange(118))
    assert _get_active_indices(["Fe", "Ni"]).tolist() == [S.FE, S.NI]
    with pytest.raises(ValueError, match="No valid active elements"):
        _get_active_indices([])


def test_build_composition_normalizes_active_columns_only():
    built = _build_composition(np.array([2.0, 6.0]), np.array([S.FE, S.NI]))
    assert built.shape == (118,)
    assert built.sum() == pytest.approx(1.0)
    assert built[S.FE] == pytest.approx(0.25) and built[S.NI] == pytest.approx(0.75)
    assert built[S.CO] == 0.0
    # a zero sum leaves the vector at zeros instead of dividing by zero
    assert np.allclose(_build_composition(np.zeros(2), np.array([S.FE, S.NI])), 0.0)


def test_forward_reexports_formula_helpers():
    from cwsr.formula import composition_to_formula as formula_version

    assert ELEMENT_SYMBOLS[0] == "H"
    assert formula_version is composition_to_formula

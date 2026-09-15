"""Composition <-> chemical-formula helpers (task-agnostic).

These utilities work on normalized 118-dimensional atomic-fraction vectors
(columns indexed by atomic number, H=0 ... Og=117) and are shared by every
dataset provider and downstream tool in the framework: :func:`form2comp`
(formula -> vector, used when loading databases) and
:func:`composition_to_formula` / :func:`format_composition` (vector -> readable
formula).
"""

from __future__ import annotations

from typing import List

import numpy as np

# Element symbols ordered by atomic number (Z=1..118), index 0 == H.
ELEMENT_SYMBOLS: List[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def form2comp(formula: str) -> np.ndarray:
    """Convert a chemical formula into a normalized 118-dim composition vector.

    This is the canonical formula -> composition converter, used by the dataset
    providers (``datasets.alloy``, ``datasets.matbench``) and by
    :mod:`cwsr.predict.query`.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``"FeCrCoNi"`` or ``"Al0.25CoCrFeNi"``.

    Returns
    -------
    np.ndarray, shape (118,)
        Atomic fractions summing to 1, indexed by zero-based atomic number
        (H=0 ... Og=117).

    Notes
    -----
    Parsing is delegated to pymatgen so fractional stoichiometries such as
    ``"Ag0.5Ge1Pb1.75S4"`` are handled robustly; ``pymatgen``/``ase`` are
    imported lazily so importing :mod:`cwsr.formula` stays dependency-light.
    """
    from ase.data import atomic_numbers
    from pymatgen.core import Composition

    comp = Composition(formula)
    vec = np.zeros(118, dtype=np.float64)
    for element, amount in comp.get_el_amt_dict().items():
        vec[atomic_numbers[element] - 1] = amount
    return vec / vec.sum()


def composition_to_formula(composition: np.ndarray,
                           threshold: float = 1e-6) -> str:
    """Format a composition vector as a chemical formula string.

    Parameters
    ----------
    composition : np.ndarray, shape (118,)
        Atomic-fraction vector.
    threshold : float
        Minimum fraction for an element to be included.

    Returns
    -------
    str
        E.g. ``"Al0.25CoCrFeNi"`` (fractions of ~1 are written without a
        coefficient).
    """
    parts = []
    for Z in range(118):
        frac = float(composition[Z])
        if frac > threshold:
            symbol = ELEMENT_SYMBOLS[Z]
            if abs(frac - 1.0) < 1e-6:
                parts.append(symbol)
            else:
                parts.append(f"{symbol}{frac:.4g}")
    return "".join(parts)


def format_composition(composition: np.ndarray,
                       top_n: int = 20,
                       threshold: float = 1e-6) -> str:
    """Human-readable summary of a composition, e.g. ``"Fe0.30 Ni0.25 Cr0.20"``."""
    parts: List[str] = []
    for idx in np.argsort(composition)[::-1]:
        frac = float(composition[idx])
        if frac > threshold and len(parts) < top_n:
            parts.append(f"{ELEMENT_SYMBOLS[idx]}{frac:.4g}")
    return " ".join(parts)

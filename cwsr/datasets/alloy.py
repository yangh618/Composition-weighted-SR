"""Alloy database dataset provider.

Reads the preprocessed ``.npz`` databases (``targets``, ``formulas``,
``target_name``, ``source`` keys) used by the alloys example and exposes them
through the shared :class:`~cwsr.datasets.base.CompositionDataset` schema.
Because only the loaders are dataset-specific, any other property/DB stored in
the same npz layout can be added without touching downstream code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from cwsr.datasets.base import CompositionDataset
from cwsr.formula import form2comp

# Default property name -> npz filename mapping (mirrors the alloy databases).
DEFAULT_PROPERTIES: Dict[str, str] = {
    "density": "density.npz",
    "ductility": "ductility.npz",
    "elongation": "elongation.npz",
    "hardness": "hardness.npz",
    "melting_temperature": "melting_temperature.npz",
    "yield_strength": "yield_strength.npz",
    "youngs_modulus": "youngs_modulus.npz",
}


def _parse_formula_vecs(formulas: np.ndarray) -> np.ndarray:
    """Convert an array of formula strings into (n, 118) composition vectors."""
    return np.stack([form2comp(str(f)) for f in formulas])


def load_alloy_dataset(property_name: str,
                       data_dir: "str | Path" = "processed_data",
                       ) -> CompositionDataset:
    """Load a preprocessed alloy ``.npz`` into a :class:`CompositionDataset`.

    Parameters
    ----------
    property_name : str
        One of :data:`DEFAULT_PROPERTIES` (e.g. ``"density"``), or the path to
        an arbitrary ``.npz`` file with the standard keys.
    data_dir : str | Path
        Directory containing the preprocessed ``.npz`` files.
    """
    data_dir = Path(data_dir)

    if property_name in DEFAULT_PROPERTIES:
        npz_path = data_dir / DEFAULT_PROPERTIES[property_name]
    else:
        # Treat the argument as an explicit path to an npz file.
        npz_path = Path(property_name)

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {npz_path}\n"
            f"Run the alloys example preprocessing first."
        )

    data = np.load(npz_path, allow_pickle=True)
    formulas = list(data["formulas"])
    compositions = _parse_formula_vecs(np.asarray(formulas))
    targets = np.asarray(data["targets"], dtype=np.float64)
    target_name = str(data["target_name"])
    source = str(data["source"])

    return CompositionDataset(
        name=f"alloy_{Path(npz_path).stem}",
        compositions=compositions,
        targets=targets,
        formulas=formulas,
        target_name=target_name,
        source=source,
        meta={"dataset": "alloy", "property": Path(npz_path).stem,
              "npz_path": str(npz_path)},
    )

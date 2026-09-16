"""Shared pytest fixtures for the CWSR test suite.

All fixtures are deterministic, offline and tiny. The only "expensive" one is
``fitted_run``, which performs a *minimal* CWSR search (shared session-wide so
the whole suite pays for it once).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fixtures import synthetic as S

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Minimal-but-valid search configuration (verified to converge on the exactly
#: linear synthetic problem while keeping the whole suite fast).
TINY_SEARCH_KWARGS = dict(
    var_count=1,
    ops=["add", "mul"],
    max_depth=3,
    max_expressions=4,
    num_batches=2,
    num_trials=1,
    num_parallel=1,
    optimization_method="LD_LBFGS",
)


@pytest.fixture
def fe_ni_compositions() -> np.ndarray:
    """(24, 118) mixtures of Fe and Ni only (no other element present)."""
    return S.composition_matrix(24, ("Fe", "Ni"), seed=0)


@pytest.fixture
def fe_ni_weights() -> np.ndarray:
    """(2, 118) selector weights: row 0 = Fe fraction, row 1 = Ni fraction."""
    return S.element_selector_weights(("Fe", "Ni"))


@pytest.fixture
def refined_results_file(tmp_path: Path, fe_ni_weights: np.ndarray) -> Path:
    """A refined-results JSON for the model ``y = Fe_fraction + 0.5 * Ni_fraction``.

    Hand-checkable: ``"Fe0.5Ni0.5"`` -> 0.75, ``"Fe"`` -> 1.0, ``"Ni"`` -> 0.5.
    """
    return S.write_refined_results(
        tmp_path / "refined_results_toy.json",
        expression="x0 + 0.5*x1",
        weights=fe_ni_weights,
    )


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    """A tiny processed ``.npz`` database in the documented layout."""
    return S.write_npz_database(
        tmp_path / "toy_property.npz",
        formulas=["FeCoNi", "Fe2O3", "Ni", "CrFe"],
        targets=[1.0, 2.0, 3.0, 4.0],
    )


@pytest.fixture
def linear_npz(tmp_path: Path) -> Path:
    """A 24-sample Fe/Ni database consistent with :func:`refined_results_file`.

    Targets are ``Fe_fraction + 0.5 * Ni_fraction``, so bootstrap resampling and
    the query/inverse tools have a real (if small) dataset to work with.
    """
    rng = np.random.default_rng(0)
    fractions = rng.dirichlet(np.ones(2), size=24)
    formulas = [f"Fe{fe:.4f}Ni{ni:.4f}" for fe, ni in fractions]
    targets = [float(fe + 0.5 * ni) for fe, ni in fractions]
    return S.write_npz_database(tmp_path / "linear_iface.npz", formulas=formulas,
                                targets=targets, target_name="toy linear")


@pytest.fixture
def synthetic_dataset(synthetic_npz: Path):
    """The ``CompositionDataset`` loaded from :func:`synthetic_npz`."""
    from datasets import get_dataset

    return get_dataset(str(synthetic_npz))


@pytest.fixture(scope="session")
def fitted_run(tmp_path_factory) -> SimpleNamespace:
    """Run one minimal end-to-end search and keep everything it produced.

    Session-scoped: the tiny search costs a few seconds (mostly first-call numba
    JIT + NLopt weight optimisation), so the suite pays for it once.
    """
    from cwsr.data import CompositionDataset
    from cwsr.model import fit_dataset

    compositions, targets, weights = S.tiny_linear_problem()
    dataset = CompositionDataset(
        name="toy_linear",
        compositions=compositions,
        targets=targets,
        target_name="toy linear target",
        source="unit-test",
    )
    output_dir = tmp_path_factory.mktemp("cwsr_fit")
    outputs = fit_dataset(dataset, output_dir=output_dir, seed=0,
                          **TINY_SEARCH_KWARGS)
    artifacts = sorted(output_dir.glob("cwsr_outputs_toy_linear_*.json"))
    assert artifacts, f"fit_dataset wrote no artifact into {output_dir}"

    return SimpleNamespace(
        dataset=dataset,
        compositions=compositions,
        targets=targets,
        true_weights=weights,
        outputs=outputs,
        artifact=artifacts[0],
        output_dir=output_dir,
    )


@pytest.fixture
def tiny_composition() -> np.ndarray:
    """(118,) equimolar FeCoNiCr composition."""
    return S.compositions_from_fractions({"Fe": 0.25, "Co": 0.25,
                                          "Ni": 0.25, "Cr": 0.25})


def load_json(path: Path):
    """Load a JSON artifact (small helper used across integration tests)."""
    with open(path) as handle:
        return json.load(handle)

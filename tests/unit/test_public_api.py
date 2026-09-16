"""Public API contract for the CWSR framework (how a user imports and uses it).

These tests deliberately import through the public entry points
(``cwsr``, ``cwsr.data``, ``cwsr.model``, ``datasets``, ``analysis``) rather
than private internals.
"""

from __future__ import annotations

import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import synthetic as S

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_is_declared_and_matches_setup_py():
    """``cwsr.__version__`` must stay aligned with the packaged version."""
    import cwsr

    setup_py = REPO_ROOT / "setup.py"
    if not setup_py.exists():
        pytest.skip("source-checkout only: setup.py is not part of the installed package")

    assert isinstance(cwsr.__version__, str) and cwsr.__version__
    declared = re.search(r'version\s*=\s*"([^"]+)"', setup_py.read_text())
    assert declared is not None, "setup.py has no version=..."
    assert cwsr.__version__ == declared.group(1)


def test_top_level_exports():
    import cwsr

    assert set(cwsr.__all__) == {"Regressor", "simplify_expression"}
    for name in cwsr.__all__:
        assert hasattr(cwsr, name), f"cwsr.{name} is missing"
    assert inspect.isclass(cwsr.Regressor)
    assert callable(cwsr.simplify_expression)


def test_simplify_expression_returns_str():
    from cwsr import simplify_expression

    simplified = simplify_expression("x0 + x0")
    assert isinstance(simplified, str) and simplified


@pytest.mark.parametrize("module", [
    "cwsr.data",
    "cwsr.formula",
    "cwsr.regressor",
    "cwsr.mcts",
    "cwsr.gp",
    "cwsr.exp_tree",
    "cwsr.exp_queue",
    "cwsr.reward",
    "cwsr.checkpoint",
    "cwsr.model",
    "cwsr.predict",
    "cwsr.predict.forward",
    "cwsr.predict.query",
    "datasets",
    "datasets.registry",
    "datasets.alloy",
    "datasets.matbench",
    "analysis",
    "analysis._common",
])
def test_documented_modules_are_importable(module: str):
    assert importlib.import_module(module) is not None


def test_public_names_exist():
    from cwsr.checkpoint import (load_checkpoint, mcts_from_dict, mcts_to_dict,
                                 save_checkpoint)
    from cwsr.data import (CompositionDataset, ensure_train_covers_species,
                           split_dataset)
    from cwsr.exp_queue import Exp_Queue, Queue_Base
    from cwsr.exp_tree import ExpTree, ExpTreeBase
    from cwsr.formula import (ELEMENT_SYMBOLS, composition_to_formula,
                              form2comp, format_composition)
    from cwsr.mcts import MCTS, MCTS_Node
    from cwsr.model import fit_dataset, train
    from cwsr.predict import (compile_expression, compile_gradient_functions,
                              jit_compile, predict_and_gradient, predict_vector)
    from cwsr.predict.query import load_refined, predict
    from cwsr.reward import Heaviside_vec, Optimizer, sp_module

    assert len(ELEMENT_SYMBOLS) == 118
    for obj in (CompositionDataset, Exp_Queue, ExpTree, MCTS, MCTS_Node,
                Optimizer, Queue_Base, ExpTreeBase):
        assert inspect.isclass(obj)
    for func in (split_dataset, ensure_train_covers_species, form2comp,
                 composition_to_formula, format_composition, train, fit_dataset,
                 compile_expression, compile_gradient_functions, jit_compile,
                 predict_vector, predict_and_gradient, load_refined, predict,
                 Heaviside_vec, save_checkpoint, load_checkpoint, mcts_to_dict,
                 mcts_from_dict):
        assert callable(func)
    assert sp_module is not None


def test_datasets_reexports_the_core_schema():
    """``from datasets import CompositionDataset`` keeps working (back-compat)."""
    import datasets
    from cwsr.data import CompositionDataset, split_dataset

    assert datasets.CompositionDataset is CompositionDataset
    assert datasets.split_dataset is split_dataset


def test_analysis_exposes_the_three_tools_lazily():
    import analysis

    assert set(analysis.__all__) == {"bootstrap", "inverse", "pareto"}
    for name in analysis.__all__:
        module = getattr(analysis, name)
        assert inspect.ismodule(module)
        assert module.__name__ == f"analysis.{name}"
    with pytest.raises(AttributeError):
        analysis.does_not_exist


def test_analysis_import_is_cheap_in_a_fresh_interpreter():
    """Importing ``analysis`` must not pull scipy/joblib (documented design)."""
    code = (
        "import sys, analysis; "
        "print('scipy' in sys.modules, 'joblib' in sys.modules, 'sklearn' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False False False"


def test_regressor_construction_defaults_and_validation():
    from cwsr import Regressor

    X, y, _ = S.tiny_linear_problem(n=8)
    model = Regressor(x_train=X, y_train=y)

    assert model.var_count == X.shape[1]
    assert model.max_depth == 6
    assert model.num_trials == 1
    assert model.sigma == pytest.approx(float(np.std(y)))
    # ops are defaulted and extended with one variable per latent dimension
    assert "add" in model.ops and "x0" in model.ops
    assert model.arity_dict["x0"] == 0
    assert model.exp_tree.max_depth == model.max_depth

    with pytest.raises(ValueError, match="Mismatched dimensions"):
        Regressor(x_train=X, y_train=y[:-1])
    with pytest.raises(ValueError, match="max_depth"):
        Regressor(x_train=X, y_train=y, max_depth=0)

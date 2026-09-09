"""Fast, training-free sanity checks for the cwsr package.

Validates imports/reference hygiene and the numerically testable public APIs
(formula helpers, element-safe splits, forward/analytic gradients) without
running the (slow) MCTS search. Run directly or via tests/run_tests.sh.

Usage:  python tests/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok - {msg}")


def check_imports() -> None:
    import cwsr  # noqa: F401
    from cwsr import Regressor, simplify_expression  # noqa: F401
    from cwsr.datasets import (get_dataset, list_datasets, register_provider,  # noqa: F401
                               CompositionDataset)
    from cwsr.datasets.base import split_dataset  # noqa: F401
    from cwsr.model import train, fit_dataset  # noqa: F401
    from cwsr.predict import (compile_expression, compile_gradient_functions,  # noqa: F401
                              predict_vector, predict_and_gradient, jit_compile)
    from cwsr.predict.forward import predict_vector_and_gradients  # noqa: F401
    from cwsr.predict.query import load_refined, predict, main  # noqa: F401
    from cwsr.mcts import MCTS, MCTS_Node  # noqa: F401
    from cwsr.gp import GPManager  # noqa: F401
    from cwsr.exp_tree import ExpTree, ExpTreeBase  # noqa: F401
    from cwsr.exp_queue import Exp_Queue, Queue_Base  # noqa: F401
    from cwsr.reward import Optimizer, sp_module, Heaviside_vec  # noqa: F401
    from cwsr.checkpoint import (save_checkpoint, load_checkpoint,  # noqa: F401
                                 mcts_to_dict, mcts_from_dict)
    assert isinstance(Regressor, type)
    _check(len(list_datasets()) >= 1, "datasets registry populated")
    print("  ok - all public symbols import")


def check_formula() -> None:
    from cwsr.formula import (ELEMENT_SYMBOLS, form2comp,
                              composition_to_formula, format_composition)
    _check(len(ELEMENT_SYMBOLS) == 118, "118 element symbols")
    v = form2comp("FeCrCoNi")
    _check(abs(v.sum() - 1.0) < 1e-9, "form2comp normalizes to sum=1")
    _check(v.shape == (118,), "composition vector length 118")
    _check(np.count_nonzero(v) == 4, "equimolar FeCrCoNi -> 4 nonzero columns")
    # FeCrCoNi round-trips as equimolar fractions.
    rt = composition_to_formula(v)
    _check(all(x in rt for x in ("0.25",)), "round trip keeps equimolar 0.25 fractions")


def check_split() -> None:
    from cwsr.datasets.base import split_dataset, ensure_train_covers_species
    rng = np.random.default_rng(0)
    # Synthetic: a singleton element in the last sample (only sample with col 3).
    X = rng.uniform(0.05, 0.4, size=(40, 118))
    X = X / X.sum(axis=1, keepdims=True)
    X[39] = 0.0
    X[39, 3] = 1.0  # singleton element present only in sample 39
    y = rng.normal(size=40)
    tr, va = split_dataset(X, y, ratio=0.7, seed=1)
    # Invariant: sample 39 (the only holder of col 3) must be in training.
    _check(39 in tr, "singleton-element sample forced into training split")
    _check(len(np.intersect1d(tr, va)) == 0, "splits are disjoint")
    # Edge: high ratio still yields a valid, exhaustive split.
    tr2, va2 = split_dataset(X, y, ratio=0.9, seed=1)
    _check(len(tr2) + len(va2) == 40 and len(np.intersect1d(tr2, va2)) == 0,
           "high-ratio split is valid and exhaustive")


def check_gradients() -> None:
    from cwsr.predict import compile_expression, compile_gradient_functions, predict_and_gradient
    var_count = 2
    W = np.zeros((var_count, 118))
    W[0, 0] = 1.0  # x0 = comp[0]
    W[1, 1] = 2.0  # x1 = 2*comp[1] -> f = x0 + 2*x1 = comp0 + 4*comp1
    f = compile_expression("x0 + 2*x1", var_count)
    grads = compile_gradient_functions("x0 + 2*x1", var_count)
    comp = np.zeros(118)
    comp[0], comp[1] = 0.3, 0.2
    val, grad = predict_and_gradient(comp, W, f, grads)
    _check(abs(val - (0.3 + 4 * 0.2)) < 1e-6, "forward value matches closed form")
    # finite-difference check of the composition gradient
    h = 1e-6
    for idx in (0, 1):
        e = np.zeros(118); e[idx] = h
        vp = f(np.einsum("vi,i->v", W, comp + e)[None, :])[0]
        vm = f(np.einsum("vi,i->v", W, comp - e)[None, :])[0]
        fd = (vp - vm) / (2 * h)
        _check(abs(fd - grad[idx]) < 1e-4, f"gradient[{idx}] matches finite diff ({grad[idx]:.6f})")


def check_query_cli() -> None:
    import json
    import subprocess
    import tempfile
    weights = np.zeros((1, 118)); weights[0, 28] = 10.0  # pure Cu -> 10*x0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([{"expression": "x0", "weights": weights.tolist()}], fh)
        path = fh.name
    proc = subprocess.run([sys.executable, "-m", "cwsr.predict.query",
                           "--results", path, "Cu"],
                          capture_output=True, text=True)
    _check(proc.returncode == 0, "query CLI returns 0")
    _check(abs(float(proc.stdout.strip()) - 10.0) < 1e-6, "query CLI predicts 10.0 for pure Cu")


def main() -> int:
    print("== fast sanity checks ==")
    check_imports()
    check_formula()
    check_split()
    check_gradients()
    check_query_cli()
    print("\nAll fast sanity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

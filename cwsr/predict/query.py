"""Query a discovered CWSR model: chemical formula -> predicted property.

Dataset-agnostic: it reads a "refined results" JSON file (a list whose first
entry carries ``expression`` and ``weights``) and predicts a property value for
any chemical formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from cwsr.formula import form2comp
from cwsr.predict.forward import compile_expression

__all__ = ["load_refined", "predict", "main"]


def load_refined(results_path: "str | Path") -> dict:
    """Load the best (first) expression record from a refined-results file.

    The file is expected to be a JSON list of entries, each with at least
    ``expression`` and ``weights`` keys.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Refined results not found: {results_path}")
    with open(results_path) as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No refined results in {results_path}")
    return data[0]


def predict(results_path: "str | Path", formula: str) -> float:
    """Predict the property value for ``formula`` using the given results file."""
    refined = load_refined(results_path)
    expression = refined["expression"]
    weights = refined["weights"]
    if hasattr(weights, "shape"):
        weights = weights
    else:
        import numpy as np
        weights = np.asarray(weights, dtype=np.float64)

    var_count = weights.shape[0]
    f_pred = compile_expression(expression, var_count)

    composition = form2comp(formula)
    x_bar = np_einsum(weights, composition)
    return float(f_pred(x_bar[np.newaxis, :])[0])


def np_einsum(weights, composition):
    import numpy as np
    return np.einsum("vi,i->v", weights, composition)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a CWSR model for a property given a chemical formula.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  cwsr query --results refined_results_density.json "Al0.2235Cr0.5651Co0.0176Nb0.1546Mo0.0386"
  cwsr query --results refined_results_density.json "Al0.25CoCrFeNi"
        """,
    )
    parser.add_argument("--results", required=True,
                        help="Path to refined-results JSON (list of {expression, weights}).")
    parser.add_argument("formula", nargs="?", type=str, default=None,
                        help="Chemical formula, e.g. Al0.25CoCrFeNi. Omit for interactive mode.")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.formula is None:
        # Interactive loop.
        print("CWSR Query Tool")
        print("=" * 40)
        while True:
            try:
                formula = input("Enter chemical formula (or 'q' to quit): ").strip()
                if formula.lower() in ("q", "quit", "exit"):
                    break
                if not formula:
                    continue
                value = predict(args.results, formula)
                print(f"  {formula}: {value:.4f}")
            except Exception as e:  # noqa: BLE001
                print(f"  Error: {e}")
        return 0

    try:
        value = predict(args.results, args.formula)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        return 1
    print(f"{value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

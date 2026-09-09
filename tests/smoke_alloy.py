#!/usr/bin/env python3
"""Short-time CWSR smoke test on the alloy database.

Loads an alloy property database (.npz) through the task-agnostic
`cwsr.datasets` registry, downsamples it, runs a small CWSR fit and
sanity-checks the result. The databases live in the `Alloys-SR` sibling repo
(or anywhere you pass with --data-dir).

Usage
-----
python tests/smoke_alloy.py --property density --data-dir /path/to/Alloys-SR/processed_data --max-samples 150
python tests/smoke_alloy.py --npz /path/to/hardness.npz --max-expressions 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    ALLOW_DATA_DIR,
    add_cli_args,
    downsample,
    report_summary,
    run_short_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short-time CWSR smoke test on the alloy database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--property", type=str, default=None,
                     help="Registered alloy property, e.g. density | hardness.")
    src.add_argument("--npz", type=str, default=None,
                     help="Explicit path to a processed .npz database.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dir holding alloy *.npz (default: the copy under "
                             "examples/alloys/data).")
    add_cli_args(parser)
    args = parser.parse_args()

    from cwsr.datasets import get_dataset

    if args.data_dir:
        data_dir = args.data_dir
    elif Path(ALLOW_DATA_DIR).is_dir():
        data_dir = ALLOW_DATA_DIR  # local example copy (self-contained)
    else:
        data_dir = "/home/huangyang/Git/Alloys-SR/processed_data"  # sibling fallback
    if args.property:
        name = f"alloy_{args.property}"
        ds = get_dataset(name, data_dir=data_dir)
    else:
        ds = get_dataset(args.npz)

    print(f"[load] '{ds.name}' via cwsr.datasets from {ds.meta.get('npz_path')}")
    print(f"[load] {ds.n_samples} samples x {ds.n_features}, "
          f"target={ds.target_name!r}")

    X, y = downsample(ds.compositions, ds.targets,
                      max_samples=args.max_samples, seed=args.seed)
    print(f"[downsample] using {len(X)} samples")

    hyperparams = {
        "max_expressions": args.max_expressions,
        "max_depth": args.max_depth,
        "var_count": args.var_count,
        "num_parallel": args.num_parallel,
    }
    report = run_short_fit(X, y, output_dir=args.output_dir,
                           run_name=f"smoke_{ds.name}",
                           seed=args.seed, hyperparams=hyperparams)
    report_summary(report)

    assert report["finite_ok"], "forward prediction not finite"
    assert report["best_expression"], "no expression discovered"
    print("\nSMOKE TEST PASSED (alloy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

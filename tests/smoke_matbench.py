#!/usr/bin/env python3
"""Short-time CWSR smoke test on a real Matbench task.

Loads a Matbench task through the task-agnostic `cwsr.datasets` registry,
downsamples it, runs a small CWSR fit and sanity-checks the result.

Note
----
The first run downloads/caches the Matbench task data (figshare). If you are
offline and the data is not already cached, run `python dataloader.py` first
or provide an already-cached task.

Usage
-----
python tests/smoke_matbench.py --task matbench_glass --max-samples 150
python tests/smoke_matbench.py --task matbench_dielectric --max-expressions 80
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import (
    add_cli_args,
    confirm_matbench_download,
    downsample,
    report_summary,
    run_short_fit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short-time CWSR smoke test on real Matbench data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="matbench_glass",
                        help="Matbench task key (default: matbench_glass).")
    parser.add_argument("--fold", type=int, default=0)
    dl = parser.add_mutually_exclusive_group()
    dl.add_argument("--yes", action="store_true", dest="yes",
                    help="Auto-confirm dataset download if not cached.")
    dl.add_argument("--no", action="store_true", dest="no",
                    help="Never download; abort if the task isn't cached.")
    add_cli_args(parser)
    args = parser.parse_args()

    from cwsr.datasets import get_dataset

    if not confirm_matbench_download(args.task, assume_yes=args.yes,
                                     assume_no=args.no):
        print(f"[abort] dataset '{args.task}' not available offline and "
              f"download not confirmed.\n"
              f"  Re-run with --yes to allow the download, or prime the cache "
              f"with: python dataloader.py")
        return 1

    print(f"[load] matbench task '{args.task}' via cwsr.datasets ...")
    ds = get_dataset(args.task, fold=args.fold)
    print(f"[load] {ds.name}: {ds.n_samples} samples x {ds.n_features}, "
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
    print("\nSMOKE TEST PASSED (matbench)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

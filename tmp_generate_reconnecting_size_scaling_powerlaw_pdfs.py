#!/usr/bin/env python3
"""Generate standard flowchart-style panels for each reconnecting size.

Individual-size classification, fitting, and plotting belong to
``Plotting.truncated_powerlaw_reversibility_flowchart.generate_single_size_panels``.
This script only discovers the size-scaling runs and calls that entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Management.reconnectionJobSelection import discover_simulation_jobs
from Plotting.standardPowerlaw import SIZE_SCALING_MIN_SIZE
from Plotting.truncated_powerlaw_reversibility_flowchart import generate_single_size_panels


DATA_ROOT = Path("/Volumes/data/MTS2D_output/sizeScalingJobs")
OUTPUT_DIR = Path(__file__).resolve().parent / "output/pdf/reconnecting_size_scaling/flowchart_panels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, action="append", help="Select a size; repeat for more than one.")
    parser.add_argument("--largest-first", action="store_true")
    parser.add_argument("--skip-pvalue", action="store_true", help="Explicitly omit the Clauset bootstrap and 95% intervals.")
    args = parser.parse_args()

    jobs = discover_simulation_jobs(args.data_root, job_type="size-scaling")
    by_size = {}
    for job in jobs:
        if job.size < SIZE_SCALING_MIN_SIZE:
            continue
        if args.size is None or job.size in args.size:
            path = job.folder / "macroData.csv"
            if path.is_file():
                by_size.setdefault(job.size, []).append(path)
    if not by_size or (args.size and set(args.size) != set(by_size)):
        raise FileNotFoundError(f"No macroData.csv files for requested sizes {args.size} in {args.data_root}")

    for size, paths in sorted(by_size.items(), reverse=args.largest_first):
        output = args.output_dir / f"L{size}"
        generate_single_size_panels(
            sorted(paths), system_size=size, output_dir=output,
            evaluate_pvalue=not args.skip_pvalue,
        )
        print(f"Saved L={size} panels to {output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract reconnection VTUs from a selected family of simulation jobs."""

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Management.reconnectionJobSelection import discover_simulation_jobs
from Management.vtuBeforeReconnectionExtraction import extract_simulation


def parse_args() -> argparse.Namespace:
    default_root = Path("/Volumes/data/MTS2D_output/sizeScalingJobs")
    default_executable = (
        Path(__file__).resolve().parents[2] / "MTS2D/build-release/MTS2D"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help=f"Root containing job folders (default: {default_root}).",
    )
    parser.add_argument(
        "--job-type",
        choices=("size-scaling",),
        default="size-scaling",
        help="Job family to process (default: size-scaling).",
    )
    parser.add_argument("--size", type=int, help="Only process this system size L.")
    parser.add_argument(
        "--executable",
        type=Path,
        default=default_executable,
        help=f"MTS2D executable (default: {default_executable}).",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_simulation_jobs(
        args.data_root,
        job_type=args.job_type,
        size=args.size,
        require_dumps=True,
    )
    if not jobs:
        raise FileNotFoundError(
            f"No completed {args.job_type} job folders with dumps found in "
            f"{Path(args.data_root).expanduser().resolve()}"
        )
    print(f"Found {len(jobs)} {args.job_type} job(s) to process.", flush=True)
    total_dumps = 0
    for job in jobs:
        print(
            f"Processing L={job.size}, seed={job.seed}: {job.folder.name}",
            flush=True,
        )
        results = extract_simulation(
            job.folder,
            args.executable,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        total_dumps += len(results)
    print(f"Finished: {total_dumps} dump(s) across {len(jobs)} job(s).")


if __name__ == "__main__":
    main()

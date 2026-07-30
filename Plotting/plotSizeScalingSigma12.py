#!/usr/bin/env python3
"""Plot the live-safe macroscopic average sigma12 curves for size-scaling jobs."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Management.reconnectionJobSelection import discover_simulation_jobs
from Plotting.reconnectingEnergyJumpAndElementDistribution import (
    determine_yield_load,
    read_live_macro_snapshot,
)


def plot_sigma12(
    data_root: Path,
    size: Optional[int] = None,
    output_folder: Optional[Path] = None,
    formats: Optional[List[str]] = None,
) -> Path:
    jobs = discover_simulation_jobs(
        data_root,
        job_type="size-scaling",
        size=size,
        require_dumps=True,
    )
    if not jobs:
        raise FileNotFoundError("No size-scaling jobs with dumps were found.")
    formats = ["pdf"] if formats is None else formats
    if not formats or any(extension not in {"pdf", "png"} for extension in formats):
        raise ValueError("formats must contain 'pdf', 'png', or both.")

    if output_folder is None:
        output_folder = Path(data_root).expanduser().resolve() / "plots"
    output_folder = Path(output_folder).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    size_tag = "all" if size is None else str(size)
    output_stem = output_folder / f"size_scaling_L{size_tag}_sigma12_vs_load"

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, job in enumerate(jobs):
        snapshot = read_live_macro_snapshot(job.folder / "macroData.csv")
        yield_load = determine_yield_load(snapshot)
        maximum = snapshot["avg_sigma12"].idxmax()
        color = colors(index % 10)
        label = f"L={job.size}, seed={job.seed}"
        ax.plot(
            snapshot["load"],
            snapshot["avg_sigma12"],
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=label,
        )
        ax.scatter(
            [snapshot.loc[maximum, "load"]],
            [snapshot.loc[maximum, "avg_sigma12"]],
            color=color,
            marker="o",
            s=22,
            zorder=3,
        )
        if not np.isclose(yield_load, snapshot.loc[maximum, "load"]):
            raise RuntimeError(
                f"Yield-load mismatch for {job.folder.name}: "
                f"{yield_load} != {snapshot.loc[maximum, 'load']}"
            )

    ax.set_xlabel(r"Load $\gamma$")
    ax.set_ylabel(r"Average stress $\sigma_{12}$")
    ax.set_title(r"Size-scaling macroscopic stress curves")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize="small", ncol=2, loc="best")
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)
    return output_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Volumes/data/MTS2D_output/sizeScalingJobs"),
    )
    parser.add_argument("--size", type=int, help="Only plot this system size.")
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "png"), default=["pdf"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size is not None and args.size <= 0:
        raise ValueError("--size must be positive")
    plot_sigma12(args.data_root, args.size, args.output_folder, args.formats)


if __name__ == "__main__":
    main()

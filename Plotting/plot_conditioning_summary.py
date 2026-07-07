from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Plotting.plot_element_condition_decomposition import (
    plot_records_on_axes,
    read_csv_records,
)
from Plotting.stiffness_geometry_schematic import plot_schematic


def make_figure(
    *,
    current_csv: Path,
    reference_csv: Path,
    integer_shear: int,
    local_shear: float,
    out_pdf: Path,
    out_png: Path,
) -> None:
    current_records = read_csv_records(current_csv)
    reference_records = read_csv_records(reference_csv)

    fig = plt.figure(figsize=(8.6, 9.4))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.05, 3.0],
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.055,
        hspace=0.22,
    )
    top_grid = outer[0].subgridspec(1, 2, wspace=-0.18)
    bottom_grid = outer[1].subgridspec(3, 2, hspace=0.28, wspace=0.27)

    schematic_axes = [fig.add_subplot(top_grid[0, i]) for i in range(2)]
    plot_schematic(schematic_axes, integer_shear=integer_shear, local_shear=local_shear)

    current_axes = [fig.add_subplot(bottom_grid[i, 0]) for i in range(3)]
    reference_axes = [fig.add_subplot(bottom_grid[i, 1]) for i in range(3)]
    plot_records_on_axes(
        current_records,
        current_axes,
        mode="current",
        column_title="distorted current geometry",
        row_titles=False,
    )
    plot_records_on_axes(
        reference_records,
        reference_axes,
        mode="reference",
        column_title="distorted reference geometry",
        row_titles=False,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the current/reference schematic with the two conditioning plots."
    )
    parser.add_argument(
        "--current-csv",
        type=Path,
        default=Path("Plots/no_minimization_current_condition_decomposition.csv"),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("Plots/no_minimization_reference_condition_decomposition.csv"),
    )
    parser.add_argument("--integer-shear", type=int, default=2)
    parser.add_argument("--local-shear", type=float, default=0.5)
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_summary.pdf"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_summary.png"),
    )
    args = parser.parse_args()

    make_figure(
        current_csv=args.current_csv,
        reference_csv=args.reference_csv,
        integer_shear=args.integer_shear,
        local_shear=args.local_shear,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
    )
    print(args.out_pdf)
    print(args.out_png)


if __name__ == "__main__":
    main()

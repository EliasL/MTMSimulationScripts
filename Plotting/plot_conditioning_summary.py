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
    condition_quantities,
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
    graphs_pdf: Path,
    graphs_png: Path,
    geometry_pdf: Path,
    geometry_png: Path,
) -> None:
    current_records = read_csv_records(current_csv)
    reference_records = read_csv_records(reference_csv)
    n_condition_rows = len(condition_quantities("current"))

    fig, schematic_axes = plt.subplots(1, 2, figsize=(8.6, 2.7))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.92, bottom=0.08, wspace=-0.18)
    plot_schematic(
        schematic_axes,
        integer_shear=integer_shear,
        local_shear=local_shear,
        scale=1.0,
    )
    geometry_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(geometry_pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(geometry_png, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    fig = plt.figure(figsize=(7.224, 6.72))
    grid = fig.add_gridspec(
        n_condition_rows,
        2,
        left=0.075,
        right=0.985,
        top=0.955,
        bottom=0.07,
        hspace=0.28,
        wspace=0.27,
    )
    current_axes = [fig.add_subplot(grid[i, 0]) for i in range(n_condition_rows)]
    reference_axes = [fig.add_subplot(grid[i, 1]) for i in range(n_condition_rows)]
    plot_records_on_axes(
        current_records,
        current_axes,
        mode="current",
        column_title=r"Distorted current geometry: $\gamma_{\mathrm{ref}}=0$",
        row_titles=False,
    )
    plot_records_on_axes(
        reference_records,
        reference_axes,
        mode="reference",
        column_title=r"Distorted reference geometry: $\gamma_{\mathrm{ref}}\neq0$",
        row_titles=False,
    )

    graphs_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(graphs_pdf)
    fig.savefig(graphs_png, dpi=220)
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
        "--graphs-pdf",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_graphs.pdf"),
    )
    parser.add_argument(
        "--graphs-png",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_graphs.png"),
    )
    parser.add_argument(
        "--geometry-pdf",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_geometry.pdf"),
    )
    parser.add_argument(
        "--geometry-png",
        type=Path,
        default=Path("Plots/no_minimization_conditioning_geometry.png"),
    )
    args = parser.parse_args()

    make_figure(
        current_csv=args.current_csv,
        reference_csv=args.reference_csv,
        integer_shear=args.integer_shear,
        local_shear=args.local_shear,
        graphs_pdf=args.graphs_pdf,
        graphs_png=args.graphs_png,
        geometry_pdf=args.geometry_pdf,
        geometry_png=args.geometry_png,
    )
    print(args.graphs_pdf)
    print(args.graphs_png)
    print(args.geometry_pdf)
    print(args.geometry_png)


if __name__ == "__main__":
    main()

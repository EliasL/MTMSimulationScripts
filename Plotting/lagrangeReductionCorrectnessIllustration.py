"""Illustrate the four decompositions induced by Lagrange reduction."""

from __future__ import annotations

import numpy as np

from MTMath.reduction import lagrange_reduction
from Plotting.plasticReductionCorrectnessIllustration import (
    OUT,
    TOTAL_F,
    apply_style,
    decomposition_data_from_M,
    print_summary,
    save_figure,
)


OUTPUT_STEM = OUT / "lagrange_reduction_correctness_illustration"
COLUMN_LINESTYLES = ("-", "--")


def decomposition_data():
    """Return the Lagrange base reduction and its four quadrant variants."""
    _, base_M = lagrange_reduction(TOTAL_F.T @ TOTAL_F)
    data = decomposition_data_from_M(base_M)

    quadrants = [branch["quadrant"] for branch in data["branches"]]
    if quadrants != [0, 1, 2, 3]:
        raise RuntimeError(
            f"Expected Lagrange variants in quadrants [0, 1, 2, 3], got {quadrants}"
        )
    return data


def main():
    apply_style()
    data = decomposition_data()
    png_path, pdf_path = save_figure(
        data,
        OUTPUT_STEM,
        column_linestyles=COLUMN_LINESTYLES,
    )
    print_summary(data)
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


if __name__ == "__main__":
    main()

"""Illustrate the determinant-one slice projected onto the Poincare disk.

The highlighted curve is the ``C12 = 0`` slice of the positive-definite
surface ``det(C) = 1``.  In the side-view coordinates

    p = (C11 - C22) / 2,    u = (C11 + C22) / 2,

that slice is the upper branch of ``u**2 - p**2 = 1``.  The projection used
throughout the project is stereographic projection from ``(p, u) = (0, -1)``
onto ``u = 0``.  The unit disk viewed edge-on is therefore the line segment
``-1 <= x_P <= 1``.

Run this file from the repository root to write PNG and PDF copies to
``Plots/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from MTMath.poincareEnergy import C2PoincareDisk


BLUE = "#2369a8"
ORANGE = "#d95f02"
GRAY = "#6f7782"
LIGHT_GRAY = "#c4c9d0"
BLACK = "#20252b"


def determinant_one_slice(s: np.ndarray) -> np.ndarray:
    """Return ``C = diag(exp(s), exp(-s))`` for a vector of parameters."""

    s = np.asarray(s, dtype=float)
    C = np.zeros(s.shape + (2, 2), dtype=float)
    C[..., 0, 0] = np.exp(s)
    C[..., 1, 1] = np.exp(-s)
    return C


def projection_slice(s: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return side-view coordinates and Poincare coordinates for the slice."""

    C = determinant_one_slice(s)
    p = 0.5 * (C[..., 0, 0] - C[..., 1, 1])
    u = 0.5 * (C[..., 0, 0] + C[..., 1, 1])
    x_p, y_p = C2PoincareDisk(C)
    return p, u, x_p, y_p


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    """Add a compact arrow without changing the data limits."""

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            shrinkA=0,
            shrinkB=0,
            **kwargs,
        )
    )


def make_figure(
    *,
    out_pdf: Path,
    out_png: Path,
    s_max: float = 2.8,
    dpi: int = 240,
) -> tuple[Path, Path]:
    """Create and save the side-view projection schematic."""

    s = np.linspace(-s_max, s_max, 500)
    p, u, _, _ = projection_slice(s)

    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.90)

    # The line u=0 is the disk seen edge-on.  Its length is exactly two.
    ax.plot(
        [-1.0, 1.0],
        [0.0, 0.0],
        color=ORANGE,
        linewidth=2.1,
        solid_capstyle="round",
        zorder=5,
    )
    ax.scatter(
        [-1.0, 1.0],
        [0.0, 0.0],
        s=16,
        color=BLACK,
        edgecolor="white",
        linewidth=0.55,
        zorder=7,
    )
    ax.text(-0.06, 0.06, r"$\mathcal{P}$", fontsize=11, color=BLACK,
            ha="right", va="bottom")

    # Projection center and the determinant-one cross-section.
    projection_center = (0.0, -1.0)
    ax.scatter(*projection_center, s=18, color=BLACK, zorder=8)
    ax.text(0.13, -1.08, r"$P$", fontsize=10, color=BLACK,
            ha="left", va="top")

    ax.plot(p, u, color=BLUE, linewidth=2.1, zorder=4)

    # A point away from the det(C)=1 surface, still on the C12=0 slice, is
    # radially normalized back onto the determinant-one curve first.
    example_s = 0.72
    outside_s = example_s
    outside_scale = 1.55
    C_outside = outside_scale * determinant_one_slice(np.asarray(outside_s))
    C_on_det_one = C_outside / np.sqrt(
        C_outside[0, 0] * C_outside[1, 1] - C_outside[0, 1] ** 2
    )
    outside_p = 0.5 * (C_outside[0, 0] - C_outside[1, 1])
    outside_u = 0.5 * (C_outside[0, 0] + C_outside[1, 1])
    det_one_p = 0.5 * (C_on_det_one[0, 0] - C_on_det_one[1, 1])
    det_one_u = 0.5 * (C_on_det_one[0, 0] + C_on_det_one[1, 1])
    ax.plot(
        [0.0, outside_p],
        [0.0, outside_u],
        color=GRAY,
        linewidth=0.95,
        linestyle=(0, (3.0, 3.0)),
        zorder=2,
    )
    _arrow(
        ax,
        (outside_p, outside_u),
        (det_one_p, det_one_u),
        color=GRAY,
        linewidth=1.0,
        zorder=6,
    )
    ax.scatter([outside_p, det_one_p], [outside_u, det_one_u], s=14, color=BLACK, zorder=8)
    ax.text(
        outside_p,
        outside_u + 0.10,
        r"$\det\mathbf{C}>1$",
        fontsize=9.5,
        color=BLACK,
        ha="center",
        va="bottom",
    )

    # Limiting projection rays through the two disk endpoints.  They meet the
    # determinant-one curve only in the limit as |p| tends to infinity.
    for sign in (-1.0, 1.0):
        ax.plot(
            [projection_center[0], sign * 4.0],
            [projection_center[1], 3.0],
            color=LIGHT_GRAY,
            linewidth=0.9,
            linestyle=(0, (3.0, 3.0)),
            zorder=1,
        )

    # A handful of rays make the projection construction visible while the
    # highlighted curve still reads as one continuous line.
    sample_s = np.array([-1.45, -example_s, 0.0, example_s, 1.45])
    sample_p, sample_u, sample_x, _ = projection_slice(sample_s)
    for p_value, u_value, x_value in zip(sample_p, sample_u, sample_x):
        ax.plot(
            [projection_center[0], p_value],
            [projection_center[1], u_value],
            color=LIGHT_GRAY,
            linewidth=0.9,
            linestyle=(0, (3.0, 3.0)),
            zorder=1,
        )
        ax.scatter([p_value], [u_value], s=13, color=BLACK, zorder=7)
        ax.scatter([x_value], [0.0], s=13, color=BLACK, zorder=7)

    ax.text(0.0, 1.85, r"$\det\widehat{\mathbf{C}}=1$, $\widehat{C}_{12}=0$",
            color=BLUE, fontsize=11, ha="center", va="center")
    ax.text(0.0, 1.58, r"$u^2-p^2=1$",
            color=BLUE, fontsize=10, ha="center", va="center")

    ax.set_xlabel(r"$p=(\widehat{C}_{11}-\widehat{C}_{22})/2$", fontsize=10)
    ax.set_ylabel(r"$u=(\widehat{C}_{11}+\widehat{C}_{22})/2$", fontsize=10)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.35, 2.7)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="0.78", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.88", linewidth=0.8, zorder=0)
    ax.grid(False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_pdf, out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=ROOT / "Plots" / "proincareDiskProjectionFigure.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "Plots" / "proincareDiskProjectionFigure.png",
    )
    args = parser.parse_args()

    out_pdf, out_png = make_figure(out_pdf=args.out_pdf, out_png=args.out_png)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()

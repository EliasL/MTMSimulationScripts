"""Illustrate lattice and Cartesian metric charts for a triangular reference.

The script deliberately distinguishes the lattice-coordinate Gram matrix
``G = [C]_(A0)`` from the Cartesian component matrix ``C_E = [C]_E``.  It
does not execute on import.  Run it explicitly to create the PDF and PNG.
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

from MTMath.poincareEnergy import C2PoincareDisk, drawC, drawPoincareGrid, prepPoincareFig


GRID_SIZE = 420
INK = "#1B1B1B"
GRID = "#7B8794"
ORANGE = "#D95F02"
BLUE = "#2369A8"
TEAL = "#008B8B"
PURPLE = "#7A5195"
LIGHT = "#D8DEE5"


def triangular_basis() -> np.ndarray:
    """Return the area-normalised triangular reference basis Q_triangle."""

    return (4.0 / 3.0) ** 0.25 * np.array(
        [[1.0, 0.5], [0.0, np.sqrt(3.0) / 2.0]]
    )


def triangular_metric() -> np.ndarray:
    """Return Q_triangle.T @ Q_triangle, checking its unit determinant."""

    Q = triangular_basis()
    G = Q.T @ Q
    if not np.isclose(np.linalg.det(Q), 1.0):
        raise RuntimeError("The triangular reference basis must have determinant one")
    if not np.isclose(np.linalg.det(G), 1.0):
        raise RuntimeError("The triangular metric must have determinant one")
    return G


def _disk_point(C: np.ndarray) -> tuple[float, float]:
    """Map a positive-definite metric to data coordinates on a disk axis."""

    x, y = C2PoincareDisk(C)
    return (
        float(x) * GRID_SIZE / 2.0 + GRID_SIZE / 2.0,
        float(y) * GRID_SIZE / 2.0 + GRID_SIZE / 2.0,
    )


def _draw_disk(ax: plt.Axes, *, transformation: np.ndarray | None, title: str) -> None:
    """Draw a Poincare disk with the reusable Lagrange-reduction tiling."""

    prepPoincareFig(
        ax=ax,
        grid_size=GRID_SIZE,
        withGrid=False,
        withYieldSurface=False,
        minimalTicks=True,
    )
    drawPoincareGrid(
        ax=ax,
        grid_size=GRID_SIZE,
        depth=2,
        transformation=transformation,
        c=GRID,
        alpha=0.62,
        linewidth=0.62,
        zorder=1,
    )
    ax.set_title(title, fontsize=11, color=INK, pad=8)


def _mark(ax: plt.Axes, C: np.ndarray, label: str, *, color: str, offset: tuple[float, float]) -> None:
    """Mark a metric in the displayed Cartesian-coordinate chart."""

    drawC(
        ax=ax,
        C=C,
        grid_size=GRID_SIZE,
        c=color,
        scatter=True,
        s=34,
        edgecolors="white",
        linewidths=0.8,
        zorder=20,
    )
    x, y = _disk_point(C)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=10,
        color=INK,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        zorder=21,
    )


def _arrow(
    ax: plt.Axes,
    vector: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
    label_offset: tuple[float, float],
) -> None:
    """Draw one labelled vector from the origin."""

    ax.add_patch(
        FancyArrowPatch(
            (0.0, 0.0),
            tuple(vector),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            zorder=5,
        )
    )
    ax.text(
        vector[0] + label_offset[0],
        vector[1] + label_offset[1],
        label,
        fontsize=9.2,
        color=color,
        ha="left" if label_offset[0] >= 0 else "right",
        va="bottom" if label_offset[1] >= 0 else "top",
        zorder=6,
    )


def _draw_lattice_action(
    ax: plt.Axes,
    Q: np.ndarray,
    M_60: np.ndarray,
    N_60: np.ndarray,
) -> None:
    """Show that lattice relabelling and Cartesian rotation give the same basis."""

    transformed = Q @ M_60
    if not np.allclose(transformed, N_60 @ Q):
        raise RuntimeError("A0 M_60 and N_60 A0 must describe the same vectors")

    coefficients = np.arange(-2, 3)
    lattice_points = np.array(
        [Q @ np.array([i, j], dtype=float) for i in coefficients for j in coefficients]
    )

    ax.scatter(
        lattice_points[:, 0],
        lattice_points[:, 1],
        s=10,
        color=LIGHT,
        edgecolors="none",
        zorder=0,
    )
    ax.axhline(0.0, color=LIGHT, linewidth=0.7, zorder=0)
    ax.axvline(0.0, color=LIGHT, linewidth=0.7, zorder=0)

    _arrow(
        ax,
        Q[:, 0],
        color=BLUE,
        linestyle="-",
        linewidth=2.1,
        label=r"$\mathbf a_1^0$",
        label_offset=(0.04, -0.05),
    )
    _arrow(
        ax,
        Q[:, 1],
        color=ORANGE,
        linestyle="-",
        linewidth=2.1,
        label=r"$\mathbf a_2^0$",
        label_offset=(0.04, 0.04),
    )
    _arrow(
        ax,
        transformed[:, 0],
        color=TEAL,
        linestyle="--",
        linewidth=1.55,
        label=r"$\mathbf a_1^{0\prime}$",
        label_offset=(-0.05, 0.14),
    )
    _arrow(
        ax,
        transformed[:, 1],
        color=PURPLE,
        linestyle="--",
        linewidth=1.55,
        label=r"$\mathbf a_2^{0\prime}$",
        label_offset=(-0.04, 0.04),
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-0.42, 1.55)
    ax.axis("off")
    ax.set_title(
        "Same triangular lattice: "
        r"$\mathbf A^{0\prime}=\mathbf A^0\mathbf M_{60}$"
        "\n"
        r"$\mathbf A^{0\prime}=\mathbf N_{60}\mathbf A^0$",
        fontsize=10.2,
        color=INK,
        pad=2,
    )
    ax.text(
        0.5,
        -0.02,
        r"$\mathbf N_{60}=\mathbf Q_\triangle\mathbf M_{60}"
        r"\mathbf Q_\triangle^{-1}=\mathbf R(60^\circ)$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.2,
        color=INK,
    )


def _diagram_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    label_offset: tuple[float, float],
) -> None:
    """Draw a labelled arrow in axes coordinates."""

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.25},
    )
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    ax.text(
        midpoint[0] + label_offset[0],
        midpoint[1] + label_offset[1],
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.8,
        color=INK,
    )


def _draw_commutative_diagram(ax: plt.Axes) -> None:
    """Show the compatible Cartesian and lattice-coordinate symmetry actions."""

    ax.axis("off")
    ax.set_title("The two symmetry actions commute", fontsize=10.2, color=INK, pad=2)

    positions = {
        "C": (0.18, 0.73),
        "G": (0.82, 0.73),
        "CN": (0.18, 0.25),
        "GM": (0.82, 0.25),
    }
    labels = {
        "C": r"$[\mathbf C]_{\mathbf E}$",
        "G": r"$\mathbf G$",
        "CN": r"$[\mathbf C_{\mathbf N}]_{\mathbf E}$",
        "GM": r"$\mathbf M^\mathsf{T}\mathbf G\mathbf M$",
    }
    for key, position in positions.items():
        ax.text(
            *position,
            labels[key],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=INK,
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": LIGHT},
        )

    _diagram_arrow(
        ax,
        (0.29, 0.73),
        (0.70, 0.73),
        r"$\mathbf Q_\triangle^\mathsf{T}(\,\cdot\,)\mathbf Q_\triangle$",
        label_offset=(0.0, 0.10),
    )
    _diagram_arrow(
        ax,
        (0.18, 0.63),
        (0.18, 0.35),
        r"$\mathbf N^\mathsf{T}(\,\cdot\,)\mathbf N$",
        label_offset=(0.15, 0.0),
    )
    _diagram_arrow(
        ax,
        (0.82, 0.63),
        (0.82, 0.35),
        r"$\mathbf M^\mathsf{T}(\,\cdot\,)\mathbf M$",
        label_offset=(-0.16, 0.0),
    )
    _diagram_arrow(
        ax,
        (0.29, 0.25),
        (0.70, 0.25),
        r"$\mathbf Q_\triangle^\mathsf{T}(\,\cdot\,)\mathbf Q_\triangle$",
        label_offset=(0.0, -0.10),
    )


def make_figure(out_pdf: Path, out_png: Path, dpi: int = 300) -> tuple[Path, Path]:
    """Render the two metric charts and their compatible symmetry actions."""

    Q = triangular_basis()
    Q_inv = np.linalg.inv(Q)
    G_triangle = triangular_metric()
    M_60 = np.array([[0.0, -1.0], [1.0, 1.0]])
    N_60 = Q @ M_60 @ Q_inv
    expected_rotation = np.array(
        [[0.5, -np.sqrt(3.0) / 2.0], [np.sqrt(3.0) / 2.0, 0.5]]
    )
    if not np.allclose(N_60, expected_rotation):
        raise RuntimeError("Q_triangle M_60 Q_triangle^-1 is not the expected rotation")
    if not np.allclose(M_60.T @ G_triangle @ M_60, G_triangle):
        raise RuntimeError("M_60 must leave the triangular Gram matrix invariant")
    if not np.allclose(N_60.T @ N_60, np.eye(2)):
        raise RuntimeError("The Cartesian representative N_60 must be orthogonal")

    C_square_in_triangular_reference = Q_inv.T @ np.eye(2) @ Q_inv
    if not np.allclose(Q_inv.T @ G_triangle @ Q_inv, np.eye(2)):
        raise RuntimeError("The triangular Gram vertex must map to C_E = I")

    fig = plt.figure(figsize=(9.6, 7.2), layout="constrained")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(3.25, 1.55),
        width_ratios=(1.0, 0.30, 1.0),
        hspace=0.04,
        wspace=0.02,
    )
    ax_standard = fig.add_subplot(grid[0, 0])
    ax_mapping = fig.add_subplot(grid[0, 1])
    ax_reference = fig.add_subplot(grid[0, 2])
    ax_lattice = fig.add_subplot(grid[1, 0])
    ax_diagram = fig.add_subplot(grid[1, 1:])

    _draw_disk(
        ax_standard,
        transformation=None,
        title="Lattice-metric chart\n" r"$\mathbf G=[\mathbf C]_{\mathbf A^0}$",
    )
    _mark(
        ax_standard,
        np.eye(2),
        r"$\mathbf I$ (square vertex)",
        color=BLUE,
        offset=(7, 7),
    )
    _mark(
        ax_standard,
        G_triangle,
        r"$\mathbf G_\triangle$ (triangular vertex)",
        color=ORANGE,
        offset=(8, -10),
    )

    # The plotting helper applies T.T @ G @ T.  With T = Q^-1, the standard
    # lattice-metric tiling is displayed in the Cartesian triangular-reference
    # chart C_E = Q^-T G Q^-1.
    _draw_disk(
        ax_reference,
        transformation=Q_inv,
        title="Cartesian tensor chart\n" r"$[\mathbf C]_{\mathbf E}$ for a triangular reference",
    )
    _mark(
        ax_reference,
        np.eye(2),
        r"$[\mathbf C]_{\mathbf E}=\mathbf I$"
        "\n"
        r"(undeformed triangle)",
        color=ORANGE,
        offset=(7, 7),
    )
    _mark(
        ax_reference,
        C_square_in_triangular_reference,
        r"square vertex",
        color=BLUE,
        offset=(8, -10),
    )

    ax_mapping.axis("off")
    ax_mapping.annotate(
        "",
        xy=(0.94, 0.61),
        xytext=(0.06, 0.61),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.3},
    )
    ax_mapping.text(
        0.5,
        0.69,
        r"$[\mathbf C]_{\mathbf E}$"
        "\n"
        r"$=\mathbf Q_\triangle^{-\mathsf{T}}\mathbf G\mathbf Q_\triangle^{-1}$",
        transform=ax_mapping.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=INK,
    )
    ax_mapping.annotate(
        "",
        xy=(0.06, 0.39),
        xytext=(0.94, 0.39),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": TEAL, "lw": 1.3},
    )
    ax_mapping.text(
        0.5,
        0.31,
        r"$\mathbf G$"
        "\n"
        r"$=\mathbf Q_\triangle^{\mathsf{T}}[\mathbf C]_{\mathbf E}\mathbf Q_\triangle$",
        transform=ax_mapping.transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK,
    )

    _draw_lattice_action(ax_lattice, Q, M_60, N_60)
    _draw_commutative_diagram(ax_diagram)

    fig.suptitle(
        "The same triangular metric in lattice and Cartesian coordinates",
        fontsize=13,
        color=INK,
        y=1.02,
    )

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
        default=ROOT / "Plots" / "triangularReferencePoincareFigure.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "Plots" / "triangularReferencePoincareFigure.png",
    )
    args = parser.parse_args()
    out_pdf, out_png = make_figure(args.out_pdf, args.out_png)
    print(out_pdf)
    print(out_png)


if __name__ == "__main__":
    main()

"""Draw the MTS2D periodic-cell simple-shear loading protocol.

Panel (b) isolates the effect of changing the periodic cell: real-node
coordinates are held fixed while the periodic images are rebuilt with the new
cell vectors. This is an explanatory intermediate, not a separately evaluated
MTS2D state: ``applyAffineStep`` immediately applies the same affine map to the
real-node solver guess. Panel (c) shows the resulting pristine affine lattice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


NAVY = "#263858"
NAVY_LIGHT = "#52627d"
IMAGE = "#d94a45"
CELL = "#388a80"
TEXT = "#18243a"


@dataclass(frozen=True)
class State:
    gamma: float
    move_real_nodes: bool


def shear(gamma: float) -> np.ndarray:
    """MTS2D ``getShear(gamma, theta=0)``."""

    return np.array([[1.0, gamma], [0.0, 1.0]])


def point(col: int, row: int, state: State, n: int) -> np.ndarray:
    """Return a centered real-node or periodic-image position.

    Image coordinates obey ``x_image = x_real + H m``, where
    ``H = shear(gamma) @ diag(n, n)`` and ``m`` is an integer image index.
    """

    base_col = col % n
    base_row = row % n
    image_index = np.array(
        [(col - base_col) // n, (row - base_row) // n], dtype=float
    )
    center = 0.5 * (n - 1)
    real_position = np.array([base_col, base_row], dtype=float)
    transformation = shear(state.gamma)
    if state.move_real_nodes:
        real_position = transformation @ real_position
    image_shift = transformation @ (n * image_index)
    return real_position + image_shift - center


def unique_mesh_edges(n: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Triangulate each periodic element pair exactly once.

    The real cell is extended only across its top and right boundaries. The
    equivalent bottom and left copies are omitted to avoid duplicate elements.
    """

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def add(a: tuple[int, int], b: tuple[int, int]) -> None:
        edges.add(tuple(sorted((a, b))))

    for row in range(n):
        for col in range(n):
            bl = (col, row)
            br = (col + 1, row)
            tl = (col, row + 1)
            tr = (col + 1, row + 1)
            add(bl, br)
            add(bl, tl)
            add(br, tr)
            add(tl, tr)
            add(tl, br)
    return sorted(edges)


def image_ring(n: int) -> list[tuple[int, int]]:
    indices = {(col, n) for col in range(n + 1)}
    indices.update({(n, row) for row in range(n)})
    return sorted(indices)


def draw_panel(
    ax: plt.Axes,
    state: State,
    n: int = 3,
) -> None:
    for (c0, r0), (c1, r1) in unique_mesh_edges(n):
        p0 = point(c0, r0, state, n)
        p1 = point(c1, r1, state, n)
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=NAVY_LIGHT,
            linewidth=1.25,
            alpha=0.68,
            solid_capstyle="round",
            zorder=1,
        )

    # Start half a lattice spacing outside the real-node array, apply the same
    # origin-anchored deformation as MTS2D, then translate only for display.
    center = 0.5 * (n - 1)
    reference_corners = np.array(
        [
            [-0.5, -0.5],
            [n - 0.5, -0.5],
            [n - 0.5, n - 0.5],
            [-0.5, n - 0.5],
            [-0.5, -0.5],
        ]
    )
    boundary = (shear(state.gamma) @ reference_corners.T).T - center
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color=CELL,
        linewidth=2.15,
        linestyle=(0, (5, 4)),
        zorder=5,
    )

    real_indices = [(col, row) for row in range(n) for col in range(n)]
    real_points = np.vstack([point(col, row, state, n) for col, row in real_indices])
    ax.scatter(
        real_points[:, 0],
        real_points[:, 1],
        s=55,
        facecolor=NAVY,
        edgecolor="white",
        linewidth=0.55,
        zorder=6,
    )

    image_points = np.vstack([point(col, row, state, n) for col, row in image_ring(n)])
    ax.scatter(
        image_points[:, 0],
        image_points[:, 1],
        s=58,
        facecolor="white",
        edgecolor=IMAGE,
        linewidth=2.0,
        zorder=7,
    )

    ax.set_xlim(-1.72, 2.56)
    ax.set_ylim(-1.66, 2.24)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def add_stage_arrow(fig: plt.Figure, left_ax: plt.Axes, right_ax: plt.Axes, label: str) -> None:
    left_box = left_ax.get_position()
    right_box = right_ax.get_position()
    y = 0.54 * left_box.y0 + 0.46 * left_box.y1
    x0 = left_box.x1 + 0.003
    x1 = right_box.x0 - 0.003
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.6,
            color="black",
            joinstyle="miter",
            capstyle="butt",
            zorder=20,
        )
    )
    fig.text(
        0.5 * (x0 + x1),
        y + 0.030,
        label,
        ha="center",
        va="bottom",
        color="black",
        fontsize=9.0,
    )


def create_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": r"\usepackage{amsmath}",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9), facecolor="white")
    gamma = 0.13
    draw_panel(
        axes[0],
        State(gamma=0.0, move_real_nodes=False),
    )
    draw_panel(
        axes[1],
        State(gamma=gamma, move_real_nodes=False),
    )
    draw_panel(
        axes[2],
        State(gamma=gamma, move_real_nodes=True),
    )

    fig.subplots_adjust(left=0.012, right=0.988, top=0.82, bottom=0.13, wspace=0.055)
    add_stage_arrow(fig, axes[0], axes[1], "load cell")
    add_stage_arrow(fig, axes[1], axes[2], "relax")
    for ax, panel_label in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(
            0.5,
            -0.065,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=TEXT,
            fontsize=10.0,
            clip_on=False,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=NAVY,
            markeredgecolor="white",
            label="real nodes",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor="white",
            markeredgecolor=IMAGE,
            markeredgewidth=1.8,
            label="ghost nodes",
        ),
        Line2D(
            [0],
            [0],
            color=CELL,
            linewidth=2.0,
            linestyle=(0, (5, 4)),
            label="periodic cell",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=9.0,
        handlelength=2.3,
        columnspacing=1.9,
    )
    return fig


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = repo_root / "output" / "pdf"
    output_dir = repo_root / "output"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = "periodic_boundary_loading_protocol"
    fig = create_figure()
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

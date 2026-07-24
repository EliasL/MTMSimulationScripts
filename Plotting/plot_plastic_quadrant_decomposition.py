"""Compact figures for the two plastic-reduction decompositions.

The visual language follows the Lagrange-reduction visualizer: the two
columns of a matrix are shown as a solid and a dashed teal basis vector.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MTMath.energyFunction import SShear, rotation
from MTMath.poincareEnergy import C2PoincareDisk, plot_reduction_history
from MTMath.poincareTiling import plasticReductionBFS
from Plotting.matrix_visualization import draw_matrix_columns


OUT = ROOT / "Plots" / "plastic_reduction"

TEAL = "#008C95"
BLUE = "#2171B5"
ORANGE = "#E67E22"
NEUTRAL = "#59636E"
GRID = "#D8DEE5"
TEXT = "#1F2933"


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def decomposition_data():
    total_F = SShear(1.3, s_conponent=(0, 1))
    candidate_Cs, paths = plasticReductionBFS(
        total_F.T @ total_F,
        max_depth=5,
        plot=False,
        return_paths=True,
    )
    representatives = []
    for candidate_index in range(len(candidate_Cs)):
        path = next(
            result
            for result in paths
            if result["candidate_index"] == candidate_index
        )
        M = np.asarray(path["M"], dtype=float)
        representatives.append(
            {
                "path": path["path"],
                "M": M,
                "F_e": total_F @ M,
                "F_p": np.linalg.inv(M),
            }
        )
    return total_F, representatives, paths


MOVE_MATRICES = {
    "U+": np.array([[1.0, 1.0], [0.0, 1.0]]),
    "U-": np.array([[1.0, -1.0], [0.0, 1.0]]),
    "L+": np.array([[1.0, 0.0], [1.0, 1.0]]),
    "L-": np.array([[1.0, 0.0], [-1.0, 1.0]]),
}


def _path_history(C0, path):
    """Return the metric after every unit shear in one BFS path."""
    M = np.eye(2)
    history = [np.asarray(C0, dtype=float)]
    for move in path:
        M = M @ MOVE_MATRICES[move]
        history.append(M.T @ C0 @ M)
    return np.stack(history)


def _draw_reduction_paths(ax, total_F, short, long):
    """Draw the two representative BFS paths in a compact Poincare disk."""
    resolution = 420
    C0 = total_F.T @ total_F
    short_history = _path_history(C0, short["path"])
    long_history = _path_history(C0, long["path"])

    plot_reduction_history(
        total_F,
        ax=ax,
        histories=(),
        resolution=resolution,
        grid_depth=5,
        show_grid=True,
        show_colorbar=False,
        show_legend=False,
        show_axes=False,
        lagrange_color=TEXT,
        plastic_color=NEUTRAL,
        grid_color="#7C8792",
        linewidth=1.4,
        white_background=True,
    )
    if ax.images:
        ax.images[0].set_alpha(0.20)
    for collection in reversed(ax.collections):
        if isinstance(collection, PathCollection):
            collection.remove()
            break

    for history, color, linestyle in (
        (short_history, BLUE, "-"),
        (long_history, ORANGE, "--"),
    ):
        x, y = C2PoincareDisk(history)
        points = np.column_stack(
            (
                x * resolution / 2 + resolution / 2,
                y * resolution / 2 + resolution / 2,
            )
        )
        for start, end in zip(points[:-1], points[1:]):
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=1.45,
                    linestyle=linestyle,
                    color=color,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=5,
                )
            )
        ax.scatter(
            *points[-1],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=7,
        )

    start_x, start_y = C2PoincareDisk(C0)
    ax.scatter(
        start_x * resolution / 2 + resolution / 2,
        start_y * resolution / 2 + resolution / 2,
        s=30,
        color=NEUTRAL,
        edgecolor=TEXT,
        linewidth=0.8,
        zorder=8,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=4.8,
                markerfacecolor=NEUTRAL,
                markeredgecolor=TEXT,
                markeredgewidth=0.8,
                linestyle="none",
                label=r"Original $\mathbf{F}$",
            ),
            Line2D([0], [0], color=BLUE, linewidth=1.5, label="short"),
            Line2D(
                [0],
                [0],
                color=ORANGE,
                linewidth=1.5,
                linestyle="--",
                label="long",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.25,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor=GRID,
        fontsize=6.5,
    )
    ax.set_title(r"Poincar\'e disk: BFS paths", color=TEXT, pad=3)


def plot_decompositions(total_F, representatives):
    short, long = sorted(representatives, key=lambda result: len(result["path"]))
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.35))
    fig.subplots_adjust(
        left=0.045,
        right=0.99,
        bottom=0.13,
        top=0.96,
        wspace=0.16,
        hspace=0.34,
    )

    _draw_reduction_paths(axes[0, 0], total_F, short, long)
    draw_matrix_columns(
        axes[0, 1],
        short["F_p"],
        limits=None,
        title=r"Short: $\mathbf{F}_{\!p}^{(s)}$",
    )
    draw_matrix_columns(
        axes[0, 2],
        long["F_p"],
        limits=None,
        title=r"Long: $\mathbf{F}_{\!p}^{(\ell)}$",
    )
    draw_matrix_columns(
        axes[1, 0],
        total_F,
        limits=None,
        title=r"Original $\mathbf{F}$",
    )
    draw_matrix_columns(
        axes[1, 1],
        short["F_e"],
        limits=None,
        title=r"Short: $\mathbf{F}_{\!e}^{(s)}$",
    )
    draw_matrix_columns(
        axes[1, 2],
        long["F_e"],
        limits=None,
        title=r"Long: $\mathbf{F}_{\!e}^{(\ell)}$",
    )

    legend = [
        Line2D(
            [0],
            [0],
            color=TEAL,
            linewidth=2.1,
            linestyle="-",
            label=r"first column $\mathbf{A}_{:1}$",
        ),
        Line2D(
            [0],
            [0],
            color=TEAL,
            linewidth=2.1,
            linestyle="--",
            label=r"second column $\mathbf{A}_{:2}$",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=2.4,
        columnspacing=2.0,
    )
    return fig


def _rotation_snapshots(ax, matrix, theta, label):
    transformed = matrix @ rotation(theta)
    draw_matrix_columns(
        ax,
        transformed,
        limits=(-1.25, 1.25),
        title=label,
    )


def plot_reference_basis_rotation():
    base_F = SShear(0.4, s_conponent=(0, 1))
    base_C = base_F.T @ base_F

    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.2))
    snapshot_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    poincare_ax = fig.add_subplot(grid[1, :])

    for ax, theta, label in zip(
        snapshot_axes,
        (0.0, np.pi / 2.0, np.pi),
        (r"$\theta=0$", r"$\theta=\pi/2$", r"$\theta=\pi$"),
    ):
        _rotation_snapshots(ax, base_F, theta, label)

    theta = np.linspace(0.0, np.pi, 501)
    Q = rotation(theta)
    C_path = np.einsum("...ji,jk,...kl->...il", Q, base_C, Q)
    x_path, y_path = C2PoincareDisk(C_path)

    poincare_ax.add_patch(
        plt.Circle((0.0, 0.0), 1.0, fill=False, color=GRID, linewidth=1.0)
    )
    poincare_ax.axhline(0.0, color=GRID, linewidth=0.8)
    poincare_ax.axvline(0.0, color=GRID, linewidth=0.8)
    poincare_ax.plot(x_path, y_path, color=BLUE, linewidth=2.0)

    for fraction in (0.18, 0.43, 0.68, 0.93):
        index = int(fraction * (len(theta) - 1))
        before = max(0, index - 4)
        poincare_ax.add_patch(
            FancyArrowPatch(
                (x_path[before], y_path[before]),
                (x_path[index], y_path[index]),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.4,
                color=BLUE,
            )
        )

    start = np.array(C2PoincareDisk(base_C), dtype=float)
    half_Q = rotation(np.pi / 2.0)
    half_C = half_Q.T @ base_C @ half_Q
    half = np.array(C2PoincareDisk(half_C), dtype=float)
    poincare_ax.scatter(*start, s=38, color=BLUE, zorder=4)
    poincare_ax.scatter(*half, s=38, color=ORANGE, zorder=4)
    poincare_ax.annotate(
        r"$\mathbf{C}(0)=\mathbf{C}(\pi)$",
        xy=start,
        xytext=(12, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )
    poincare_ax.annotate(
        r"$\mathbf{C}(\pi/2)$: opposite elastic quadrant",
        xy=half,
        xytext=(-12, -10),
        textcoords="offset points",
        ha="right",
        va="top",
    )
    poincare_ax.set_xlim(-0.32, 0.32)
    poincare_ax.set_ylim(-0.27, 0.27)
    poincare_ax.set_aspect("equal", adjustable="box")
    poincare_ax.set_xlabel(r"$x_p$")
    poincare_ax.set_ylabel(r"$y_p$")
    poincare_ax.set_title(
        r"$\mathbf{C}(\theta)=\mathbf{Q}(\theta)^T\mathbf{C}(0)\mathbf{Q}(\theta)$ closes after $\theta=\pi$",
        color=TEXT,
        pad=5,
    )
    for spine in poincare_ax.spines.values():
        spine.set_color(GRID)

    fig.suptitle(
        "Orthogonal change of reference basis: one material index versus two",
        fontsize=11,
        color=TEXT,
    )
    return fig


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    total_F, representatives, _ = decomposition_data()

    figures = {
        "elastic_plastic_factors": plot_decompositions(total_F, representatives),
        "reference_basis_rotation": plot_reference_basis_rotation(),
    }
    for stem, fig in figures.items():
        fig.savefig(
            OUT / f"{stem}.pdf",
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        fig.savefig(
            OUT / f"{stem}.png",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

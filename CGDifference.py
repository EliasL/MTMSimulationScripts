#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
_CACHE_DIR = ROOT_DIR / ".cache"
if "MPLCONFIGDIR" not in os.environ:
    (_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_CACHE_DIR / "matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

from MTMath.meshUtils import perfect_grid_nodes
from MTMath.poincareEnergy import C2Plane, drawTriangularElasticDomain, prepPoincareFig


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

PLOTS_DIR = ROOT_DIR / "Plots"
OUTPUT_PATH = PLOTS_DIR / "CGDifference.pdf"

N_CELLS = 3
SHEAR_MAX = 0.5
N_SHEAR_STEPS = 120
GRID_SIZE = 320

RIGHT_COLOR = "#0072B2"
LEFT_COLOR = "#D55E00"
ORIGINAL_COLOR = "#1A1A1A"
TRIANGULAR_REGION_COLOR = "#009E73"


def simple_shear_F(gamma: float) -> np.ndarray:
    return np.array([[1.0, gamma], [0.0, 1.0]], dtype=float)


def sheared_nodes(nodes: np.ndarray, gamma: float) -> np.ndarray:
    return nodes @ simple_shear_F(gamma).T


def triangular_elements(n_cells: int, diagonal: str) -> np.ndarray:
    n = n_cells + 1
    elements = []
    for j in range(n_cells):
        for i in range(n_cells):
            bl = j * n + i
            br = bl + 1
            tl = (j + 1) * n + i
            tr = tl + 1

            if diagonal == "major":
                # Major diagonal: upper-left to bottom-right.
                elements.append([tl, bl, br])
                elements.append([tl, br, tr])
            elif diagonal == "minor":
                elements.append([bl, br, tr])
                elements.append([bl, tr, tl])
            else:
                raise ValueError(f"Unknown diagonal orientation: {diagonal}")

    return np.asarray(elements, dtype=int)


def mesh_edges(elements: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in elements:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(sorted(edges), dtype=int)


def mesh_segments(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray([[nodes[i], nodes[j]] for i, j in edges], dtype=float)


def gram_from_two_shortest_edges(nodes: np.ndarray, element: np.ndarray) -> np.ndarray:
    a, b, c = [int(index) for index in element]
    edges = [
        (a, b, nodes[b] - nodes[a]),
        (a, c, nodes[c] - nodes[a]),
        (b, c, nodes[c] - nodes[b]),
    ]
    shortest = sorted(edges, key=lambda edge: float(edge[2] @ edge[2]))[:2]

    first_start, first_end, _ = shortest[0]
    second_start, second_end, _ = shortest[1]
    shared = {first_start, first_end} & {second_start, second_end}
    if len(shared) != 1:
        raise RuntimeError("The two shortest triangle edges should share one vertex.")

    origin = shared.pop()
    vectors = []
    for start, end, _ in shortest:
        other = end if start == origin else start
        vectors.append(nodes[other] - nodes[origin])

    V = np.column_stack(vectors)
    return V.T @ V


def element_deformation_gradients(
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    gradients = []
    for element in elements:
        X = reference_nodes[element]
        x = current_nodes[element]
        dX = np.column_stack((X[1] - X[0], X[2] - X[0]))
        dx = np.column_stack((x[1] - x[0], x[2] - x[0]))
        gradients.append(dx @ np.linalg.inv(dX))
    return np.asarray(gradients)


def assert_uniform_matrices(
    matrices: np.ndarray,
    name: str,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    if matrices.shape[0] <= 1:
        return
    reference = matrices[0]
    if np.allclose(matrices, reference, rtol=rtol, atol=atol):
        return
    max_diff = float(np.max(np.abs(matrices - reference)))
    raise AssertionError(f"{name} is not uniform over the mesh; max difference {max_diff:.3e}")


def C_path(nodes: np.ndarray, elements: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    matrices = []
    for gamma in gammas:
        current_nodes = sheared_nodes(nodes, float(gamma))
        F_values = element_deformation_gradients(nodes, current_nodes, elements)
        C_values = F_values.swapaxes(-1, -2) @ F_values
        assert_uniform_matrices(C_values, rf"C at gamma={gamma:.6g}")
        matrices.append(C_values[0])
    return np.asarray(matrices)


def G_values_for_elements(
    nodes: np.ndarray,
    elements: np.ndarray,
    gamma: float,
) -> np.ndarray:
    current_nodes = sheared_nodes(nodes, float(gamma))
    return np.asarray(
        [gram_from_two_shortest_edges(current_nodes, element) for element in elements]
    )


def G_path(nodes: np.ndarray, elements: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    matrices = []
    for gamma in gammas:
        G_values = G_values_for_elements(nodes, elements, float(gamma))
        assert_uniform_matrices(G_values, rf"G at gamma={gamma:.6g}")
        matrices.append(G_values[0])
    return np.asarray(matrices)


def matrix_path_to_pixels(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = C2Plane(matrices)
    return x * GRID_SIZE / 2.0 + GRID_SIZE / 2.0, y * GRID_SIZE / 2.0 + GRID_SIZE / 2.0


def add_trajectory_arrow(ax, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    ax.plot(x, y, color=color, linewidth=2.2, label=label, zorder=5)
    ax.scatter(x[0], y[0], s=18, color=ORIGINAL_COLOR, zorder=7)
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) <= 1e-12:
        return
    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-6], y[-6]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16),
        zorder=8,
    )


def plot_disk_paths(
    ax,
    right_matrices: np.ndarray,
    left_matrices: np.ndarray,
    title: str,
    show_ylabel: bool,
    highlight_triangular_region: bool = False,
) -> None:
    prepPoincareFig(
        ax=ax,
        grid_size=GRID_SIZE,
        withGrid=True,
        withYieldSurface=False,
        transformation=None,
        minimalTicks=False,
    )

    if highlight_triangular_region:
        drawTriangularElasticDomain(
            ax=ax,
            grid_size=GRID_SIZE,
            zoom=1,
            c=TRIANGULAR_REGION_COLOR,
            linewidth=1.15,
            alpha=0.55,
            transformation=None,
            zorder=2,
        )

    x_right, y_right = matrix_path_to_pixels(right_matrices)
    x_left, y_left = matrix_path_to_pixels(left_matrices)
    add_trajectory_arrow(ax, x_right, y_right, RIGHT_COLOR, "Right shear")
    add_trajectory_arrow(ax, x_left, y_left, LEFT_COLOR, "Left shear")
    ax.set_title(title, pad=8)
    if not show_ylabel:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def plot_mesh_panel(ax, nodes: np.ndarray, elements: np.ndarray, title: str) -> None:
    edges = mesh_edges(elements)
    right_nodes = sheared_nodes(nodes, SHEAR_MAX)
    left_nodes = sheared_nodes(nodes, -SHEAR_MAX)

    for deformed, color in ((right_nodes, RIGHT_COLOR), (left_nodes, LEFT_COLOR)):
        ax.add_collection(
            LineCollection(
                mesh_segments(deformed, edges),
                colors=color,
                linewidths=1.1,
                alpha=0.28,
                zorder=1,
            )
        )

    ax.add_collection(
        LineCollection(
            mesh_segments(nodes, edges),
            colors=ORIGINAL_COLOR,
            linewidths=1.25,
            zorder=3,
        )
    )

    top_y = float(nodes[:, 1].max()) + 0.18
    top_mid = np.array([0.5 * (nodes[:, 0].min() + nodes[:, 0].max()), top_y])
    arrow_targets = {
        "Right shear": top_mid + np.array([SHEAR_MAX * nodes[:, 1].max(), 0.0]),
        "Left shear": top_mid - np.array([SHEAR_MAX * nodes[:, 1].max(), 0.0]),
    }
    for label, target in arrow_targets.items():
        color = RIGHT_COLOR if label == "Right shear" else LEFT_COLOR
        ax.add_patch(
            FancyArrowPatch(
                posA=top_mid,
                posB=target,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2.0,
                color=color,
                zorder=5,
            )
        )

    all_nodes = np.vstack([left_nodes, nodes, right_nodes])
    pad = 0.28
    ax.set_xlim(all_nodes[:, 0].min() - pad, all_nodes[:, 0].max() + pad)
    ax.set_ylim(all_nodes[:, 1].min() - pad, top_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure() -> plt.Figure:
    nodes = perfect_grid_nodes((N_CELLS + 1, N_CELLS + 1))
    right_gammas = np.linspace(0.0, SHEAR_MAX, N_SHEAR_STEPS)
    left_gammas = np.linspace(0.0, -SHEAR_MAX, N_SHEAR_STEPS)
    assert np.isclose(right_gammas[-1], SHEAR_MAX)
    assert np.isclose(left_gammas[-1], -SHEAR_MAX)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.4, 7.4),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.22, "hspace": 0.34},
    )

    row_specs = [
        ("major", "Major mesh"),
        ("minor", "Minor mesh"),
    ]
    for row, (diagonal, label) in enumerate(row_specs):
        elements = triangular_elements(N_CELLS, diagonal)
        c_right = C_path(nodes, elements, right_gammas)
        c_left = C_path(nodes, elements, left_gammas)
        g_right = G_path(nodes, elements, right_gammas)
        g_left = G_path(nodes, elements, left_gammas)

        plot_mesh_panel(axes[row, 0], nodes, elements, label)
        plot_disk_paths(
            axes[row, 1],
            c_right,
            c_left,
            r"$\mathbf{C}=\mathbf{F}^{\mathsf{T}}\mathbf{F}$",
            show_ylabel=True,
        )
        plot_disk_paths(
            axes[row, 2],
            g_right,
            g_left,
            r"$\mathbf{G}$",
            show_ylabel=False,
            highlight_triangular_region=True,
        )

    handles = [
        Line2D([0], [0], color=RIGHT_COLOR, lw=2.2, label="Right shear"),
        Line2D([0], [0], color=LEFT_COLOR, lw=2.2, label="Left shear"),
        Line2D([0], [0], color=ORIGINAL_COLOR, lw=1.4, label="Reference mesh"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        r"Affine simple shear: metric tensor $\mathbf{C}$ versus element Gram matrix $\mathbf{G}$",
        y=1.035,
    )
    return fig


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

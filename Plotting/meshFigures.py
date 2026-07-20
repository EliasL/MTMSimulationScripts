"""Small, paper-style figures illustrating mesh kinematics.

The first figure in this collection shows why an element-local deformation
gradient is not invariant under an edge flip when the new child triangles are
assigned fresh square-half reference configurations.

Run this file directly to write PNG and PDF copies to ``Plots/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from MTMath.meshUtils import (
    element_deformation_gradients,
    structured_triangular_mesh,
)
from Plotting.mesh_plotting import MeshFigure, MeshStyle


REFERENCE_COLOR = "#0072B2"
CURRENT_COLOR = "#D55E00"
TRIANGLE_CONNECTIVITY = np.array([[0, 1, 2]], dtype=int)


@dataclass(frozen=True)
class EdgeFlipState:
    """Geometry and element-local deformation gradients at one stage."""

    title: str
    current_nodes: np.ndarray
    connectivity: np.ndarray
    reference_elements: np.ndarray
    deformation_gradients: np.ndarray

    @property
    def current_elements(self) -> np.ndarray:
        return self.current_nodes[self.connectivity]


@dataclass(frozen=True)
class EdgeFlipExample:
    """The initial, sheared, and reconnected states used by the figure."""

    shear: float
    applied_deformation_gradient: np.ndarray
    states: tuple[EdgeFlipState, EdgeFlipState, EdgeFlipState]


def simple_shear_matrix(gamma: float) -> np.ndarray:
    """Return the horizontal simple-shear deformation gradient."""

    return np.array([[1.0, gamma], [0.0, 1.0]], dtype=float)


def apply_deformation(
    points: np.ndarray,
    deformation_gradient: np.ndarray,
) -> np.ndarray:
    """Apply a homogeneous two-dimensional deformation to row-wise points."""

    points = np.asarray(points, dtype=float)
    deformation_gradient = np.asarray(deformation_gradient, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (n_points, 2), got {points.shape}.")
    if deformation_gradient.shape != (2, 2):
        raise ValueError(
            "deformation_gradient must have shape (2, 2), "
            f"got {deformation_gradient.shape}."
        )
    return points @ deformation_gradient.T


def _opposite_longest_edge_vertex(triangle: np.ndarray) -> int:
    """Return the vertex opposite the longest edge, matching ``TElement``."""

    opposite_edge_lengths_squared = np.empty(3, dtype=float)
    for vertex in range(3):
        edge = triangle[(vertex + 1) % 3] - triangle[(vertex + 2) % 3]
        opposite_edge_lengths_squared[vertex] = edge @ edge
    return int(np.argmax(opposite_edge_lengths_squared))


def closest_square_reference_triangle(current_triangle: np.ndarray) -> np.ndarray:
    """Choose the oriented unit-square half closest to a current triangle.

    This mirrors ``closestSquareReferenceNodes`` in MTS2D: the vertex opposite
    the current longest edge is placed at a square corner, and the four square
    orientations and both adjacent-leg assignments are compared by dot product.
    """

    current_triangle = np.asarray(current_triangle, dtype=float)
    if current_triangle.shape != (3, 2):
        raise ValueError(
            "current_triangle must have shape (3, 2), "
            f"got {current_triangle.shape}."
        )

    angle_index = _opposite_longest_edge_vertex(current_triangle)
    adjacent_index_1 = (angle_index + 1) % 3
    adjacent_index_2 = (angle_index + 2) % 3
    current_leg_1 = (
        current_triangle[adjacent_index_1] - current_triangle[angle_index]
    )
    current_leg_2 = (
        current_triangle[adjacent_index_2] - current_triangle[angle_index]
    )

    # This is the [0, 1]^2 translation of the centered square used by MTS2D.
    # A common origin keeps the reference configuration fixed across panels;
    # the translation has no effect on the calculated deformation gradient.
    candidates = (
        ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
    )

    best_score = -np.inf
    best_reference = np.empty((3, 2), dtype=float)
    for angle_corner, adjacent_corner_1, adjacent_corner_2 in candidates:
        angle_corner = np.asarray(angle_corner, dtype=float)
        adjacent_corner_1 = np.asarray(adjacent_corner_1, dtype=float)
        adjacent_corner_2 = np.asarray(adjacent_corner_2, dtype=float)
        reference_leg_1 = adjacent_corner_1 - angle_corner
        reference_leg_2 = adjacent_corner_2 - angle_corner

        score_12 = (
            current_leg_1 @ reference_leg_1 + current_leg_2 @ reference_leg_2
        )
        score_21 = (
            current_leg_1 @ reference_leg_2 + current_leg_2 @ reference_leg_1
        )
        score = max(score_12, score_21)
        if score <= best_score:
            continue

        candidate_reference = np.empty((3, 2), dtype=float)
        candidate_reference[angle_index] = angle_corner
        if score_12 >= score_21:
            candidate_reference[adjacent_index_1] = adjacent_corner_1
            candidate_reference[adjacent_index_2] = adjacent_corner_2
        else:
            candidate_reference[adjacent_index_1] = adjacent_corner_2
            candidate_reference[adjacent_index_2] = adjacent_corner_1
        best_score = score
        best_reference = candidate_reference

    return best_reference


def deformation_gradients_from_elements(
    reference_elements: np.ndarray,
    current_elements: np.ndarray,
) -> np.ndarray:
    """Calculate ``F`` for disconnected element-wise reference geometries."""

    reference_elements = np.asarray(reference_elements, dtype=float)
    current_elements = np.asarray(current_elements, dtype=float)
    if reference_elements.shape != current_elements.shape:
        raise ValueError(
            "reference_elements and current_elements must have the same shape, "
            f"got {reference_elements.shape} and {current_elements.shape}."
        )
    if reference_elements.ndim != 3 or reference_elements.shape[1:] != (3, 2):
        raise ValueError(
            "element arrays must have shape (n_elements, 3, 2), "
            f"got {reference_elements.shape}."
        )

    disconnected_connectivity = np.arange(reference_elements.size // 2).reshape(-1, 3)
    return element_deformation_gradients(
        reference_elements.reshape(-1, 2),
        current_elements.reshape(-1, 2),
        disconnected_connectivity,
    )


def build_edge_flip_example(shear: float = 0.75) -> EdgeFlipExample:
    """Build the three states in the edge-flip deformation-gradient example."""

    reference_nodes, minor_connectivity = structured_triangular_mesh(
        (2, 2), diagonal="minor"
    )
    _, major_connectivity = structured_triangular_mesh((2, 2), diagonal="major")
    applied_deformation_gradient = simple_shear_matrix(shear)
    loaded_nodes = apply_deformation(reference_nodes, applied_deformation_gradient)

    initial_reference_elements = reference_nodes[minor_connectivity]
    initial_current_elements = reference_nodes[minor_connectivity]
    initial_state = EdgeFlipState(
        title="Initial",
        current_nodes=reference_nodes,
        connectivity=minor_connectivity,
        reference_elements=initial_reference_elements,
        deformation_gradients=deformation_gradients_from_elements(
            initial_reference_elements, initial_current_elements
        ),
    )

    loaded_current_elements = loaded_nodes[minor_connectivity]
    loaded_state = EdgeFlipState(
        title="Simple shear",
        current_nodes=loaded_nodes,
        connectivity=minor_connectivity,
        reference_elements=initial_reference_elements,
        deformation_gradients=deformation_gradients_from_elements(
            initial_reference_elements, loaded_current_elements
        ),
    )

    flipped_current_elements = loaded_nodes[major_connectivity]
    flipped_reference_elements = np.stack(
        [
            closest_square_reference_triangle(current_triangle)
            for current_triangle in flipped_current_elements
        ]
    )
    flipped_state = EdgeFlipState(
        title="After edge flip",
        current_nodes=loaded_nodes,
        connectivity=major_connectivity,
        reference_elements=flipped_reference_elements,
        deformation_gradients=deformation_gradients_from_elements(
            flipped_reference_elements, flipped_current_elements
        ),
    )

    return EdgeFlipExample(
        shear=shear,
        applied_deformation_gradient=applied_deformation_gradient,
        states=(initial_state, loaded_state, flipped_state),
    )


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _format_matrix_entry(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    label: str,
    color: str = "black",
) -> None:
    """Draw a compact 2x2 matrix without requiring an external LaTeX install."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {matrix.shape}.")

    entries = [[_format_matrix_entry(value) for value in row] for row in matrix]
    matrix_tex = (
        r"\begin{bmatrix}"
        + entries[0][0]
        + " & "
        + entries[0][1]
        + r" \\ "
        + entries[1][0]
        + " & "
        + entries[1][1]
        + r"\end{bmatrix}"
    )
    ax.text(
        0.5,
        0.24,
        rf"${label}{matrix_tex}$",
        transform=ax.transAxes,
        fontsize=18,
        color=color,
        ha="center",
        va="center",
    )


def _draw_geometry_panel(
    ax: plt.Axes,
    state: EdgeFlipState,
    *,
    flipped: bool,
    xlim: tuple[float, float],
) -> None:
    current_style = MeshStyle(
        color=CURRENT_COLOR,
        face_alpha=0.14,
        linewidth=2.15,
        node_facecolor="white",
        node_edgecolor=CURRENT_COLOR,
        node_size=34,
        node_linewidth=1.5,
        zorder=2,
    )
    reference_style = MeshStyle(
        color=REFERENCE_COLOR,
        face_alpha=0.0,
        linewidth=1.8,
        linestyle=(0, (4.0, 3.0)),
        draw_faces=False,
        draw_nodes=False,
        zorder=5,
    )

    mesh = MeshFigure(ax)
    mesh.draw_mesh(state.current_nodes, state.connectivity, style=current_style)

    if flipped:
        for reference_element in state.reference_elements:
            mesh.draw_mesh(
                reference_element,
                TRIANGLE_CONNECTIVITY,
                style=reference_style,
            )
    else:
        reference_nodes = np.empty_like(state.current_nodes)
        for element_indices, reference_element in zip(
            state.connectivity, state.reference_elements
        ):
            reference_nodes[element_indices] = reference_element
        mesh.draw_mesh(reference_nodes, state.connectivity, style=reference_style)

    mesh.configure_axis(
        xlim=xlim,
        ylim=(-0.18, 1.18),
        equal_aspect=True,
        hide_axes=True,
    )


def make_edge_flip_deformation_gradient_figure(
    shear: float = 0.75,
) -> tuple[plt.Figure, np.ndarray, EdgeFlipExample]:
    """Create the 2-row by 2-column edge-flip illustration."""

    _configure_matplotlib()
    example = build_edge_flip_example(shear)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 4.2),
        gridspec_kw={
            "height_ratios": (0.55, 1.45),
            "hspace": 0.0,
            "wspace": 0.08,
        },
    )

    displayed_states = example.states[1:]
    matrix_labels = (
        r"\mathbf{F}^{-}=",
        r"\mathbf{F}^{+}=",
    )
    for column, state in enumerate(displayed_states):
        top_axis = axes[0, column]
        top_axis.text(
            0.04,
            0.92,
            rf"\textbf{{({chr(ord('a') + column)})}}",
            transform=top_axis.transAxes,
            fontsize=24,
            ha="left",
            va="top",
        )
        top_axis.set_xlim(0, 1)
        top_axis.set_ylim(0, 1)
        top_axis.axis("off")
        _draw_matrix(
            top_axis,
            state.deformation_gradients[0],
            label=matrix_labels[column],
            color="black",
        )
        _draw_geometry_panel(
            axes[1, column],
            state,
            flipped=(column == 1),
            xlim=(-0.18, 1.93),
        )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                linestyle=(0, (4.0, 3.0)),
                linewidth=1.8,
                label=r"reference configuration $X_e$",
            ),
            Line2D(
                [0],
                [0],
                color=CURRENT_COLOR,
                linestyle="-",
                linewidth=2.15,
                label=r"current configuration $x_e$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.155),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=2.8,
    )
    figure.subplots_adjust(left=0.025, right=0.985, top=0.98, bottom=0.08)
    return figure, axes, example


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the deformation-gradient edge-flip schematic."
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.75,
        help="Applied simple-shear amount (default: 0.75).",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "Plots" / "edge_flip_deformation_gradient",
        help="Output path without an extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    figure, _, _ = make_edge_flip_deformation_gradient_figure(args.shear)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

"""Small, paper-style figures illustrating mesh kinematics.

The first figure in this collection shows why an element-local deformation
gradient changes under an edge flip even when the current lattice geometry is
unchanged.  Every element is compared with one canonical reference triangle;
the two element-local representatives are related by a fixed lattice basis
change ``Q``.

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
CURRENT_COLORS = ("#D55E00", "#009E73")
# Kept as a public alias for callers that used the original single colour.
CURRENT_COLOR = CURRENT_COLORS[0]
CANONICAL_REFERENCE_TRIANGLE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=float,
)
TRIANGLE_CONNECTIVITY = np.array([[0, 1, 2]], dtype=int)


@dataclass(frozen=True)
class EdgeFlipState:
    """Geometry and element-local deformation gradients at one stage."""

    title: str
    current_nodes: np.ndarray
    connectivity: np.ndarray
    canonical_reference_element: np.ndarray
    deformation_gradients: np.ndarray

    @property
    def current_elements(self) -> np.ndarray:
        return self.current_nodes[self.connectivity]

    @property
    def reference_elements(self) -> np.ndarray:
        """Return the canonical reference triangle for each element.

        This broadcasted compatibility view makes the single-reference
        construction explicit: the returned triangles are identical, rather
        than independently selected for each current element.
        """

        return np.broadcast_to(
            self.canonical_reference_element,
            (len(self.connectivity), 3, 2),
        )


@dataclass(frozen=True)
class EdgeFlipExample:
    """The initial, sheared, and reconnected states used by the figure."""

    shear: float
    applied_deformation_gradient: np.ndarray
    canonical_reference_element: np.ndarray
    lattice_basis_change: np.ndarray
    states: tuple[EdgeFlipState, EdgeFlipState, EdgeFlipState]

    @property
    def Q(self) -> np.ndarray:
        """The fixed lattice basis change relating the two element ``F``s."""

        return self.lattice_basis_change

    @property
    def F(self) -> np.ndarray:
        """Calculated pre-flip deformation gradient of current element 1."""

        return self.states[1].deformation_gradients[0]


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


def deformation_gradients_from_canonical_reference(
    canonical_reference_element: np.ndarray,
    current_elements: np.ndarray,
) -> np.ndarray:
    """Calculate all element gradients from one canonical reference triangle."""

    canonical_reference_element = np.asarray(
        canonical_reference_element,
        dtype=float,
    )
    current_elements = np.asarray(current_elements, dtype=float)
    if canonical_reference_element.shape != (3, 2):
        raise ValueError(
            "canonical_reference_element must have shape (3, 2), "
            f"got {canonical_reference_element.shape}."
        )
    if current_elements.ndim != 3 or current_elements.shape[1:] != (3, 2):
        raise ValueError(
            "current_elements must have shape (n_elements, 3, 2), "
            f"got {current_elements.shape}."
        )

    reference_elements = np.broadcast_to(
        canonical_reference_element,
        current_elements.shape,
    )
    return deformation_gradients_from_elements(reference_elements, current_elements)


def paired_deformation_gradients(
    F: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Return ``F^(1) = F`` and ``F^(2) = F Q`` for one state."""

    F = np.asarray(F, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if F.shape != (2, 2) or Q.shape != (2, 2):
        raise ValueError(
            f"F and Q must both have shape (2, 2), got {F.shape} and {Q.shape}."
        )
    return np.stack((F, F @ Q))


def build_edge_flip_example(shear: float = 0.75) -> EdgeFlipExample:
    """Build the three states in the canonical-reference edge-flip example.

    The two triangles use one canonical reference triangle.  ``F`` is
    calculated from current element 1 before the flip, while ``Q`` is
    calculated from the pair.  The second element then satisfies
    ``F^(2) = F^(1) @ Q`` before and after the flip.
    """

    reference_nodes, minor_connectivity = structured_triangular_mesh(
        (2, 2), diagonal="minor"
    )
    _, major_connectivity = structured_triangular_mesh((2, 2), diagonal="major")
    applied_deformation_gradient = simple_shear_matrix(shear)
    loaded_nodes = apply_deformation(reference_nodes, applied_deformation_gradient)
    canonical_reference_element = CANONICAL_REFERENCE_TRIANGLE.copy()

    initial_current_elements = reference_nodes[minor_connectivity]
    initial_deformation_gradients = deformation_gradients_from_canonical_reference(
        canonical_reference_element,
        initial_current_elements,
    )
    loaded_current_elements = loaded_nodes[minor_connectivity]
    loaded_deformation_gradients = deformation_gradients_from_canonical_reference(
        canonical_reference_element,
        loaded_current_elements,
    )
    flipped_current_elements = loaded_nodes[major_connectivity]
    flipped_deformation_gradients = deformation_gradients_from_canonical_reference(
        canonical_reference_element,
        flipped_current_elements,
    )
    lattice_basis_change = np.linalg.solve(
        loaded_deformation_gradients[0],
        loaded_deformation_gradients[1],
    )

    initial_state = EdgeFlipState(
        title="Initial",
        current_nodes=reference_nodes,
        connectivity=minor_connectivity,
        canonical_reference_element=canonical_reference_element,
        deformation_gradients=initial_deformation_gradients,
    )

    loaded_state = EdgeFlipState(
        title="Simple shear",
        current_nodes=loaded_nodes,
        connectivity=minor_connectivity,
        canonical_reference_element=canonical_reference_element,
        deformation_gradients=loaded_deformation_gradients,
    )

    flipped_state = EdgeFlipState(
        title="After edge flip",
        current_nodes=loaded_nodes,
        connectivity=major_connectivity,
        canonical_reference_element=canonical_reference_element,
        deformation_gradients=flipped_deformation_gradients,
    )

    # F^(1) is calculated directly from the current geometry in each state.
    # Q is calculated before the flip and relates F^(1) to F^(2) in both
    # topologies; F^(1)+ is deliberately not assigned from Q.
    if not all(
        np.allclose(
            state.deformation_gradients,
            paired_deformation_gradients(
                state.deformation_gradients[0],
                lattice_basis_change,
            ),
        )
        for state in (initial_state, loaded_state, flipped_state)
    ):
        raise RuntimeError(
            "The element gradients no longer satisfy F^(1)=F and "
            "F^(2)=F^(1)Q before and after the edge flip."
        )

    return EdgeFlipExample(
        shear=shear,
        applied_deformation_gradient=applied_deformation_gradient,
        canonical_reference_element=canonical_reference_element,
        lattice_basis_change=lattice_basis_change,
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
    x: float = 0.5,
    y: float = 0.24,
    fontsize: float = 14.0,
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
        x,
        y,
        rf"${label}{matrix_tex}$",
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
    )


def _draw_geometry_panel(
    ax: plt.Axes,
    state: EdgeFlipState,
    *,
    xlim: tuple[float, float],
) -> None:
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
    for element_index, element_indices in enumerate(state.connectivity):
        color = CURRENT_COLORS[element_index % len(CURRENT_COLORS)]
        current_style = MeshStyle(
            color=color,
            face_alpha=0.14,
            linewidth=2.15,
            node_facecolor="white",
            node_edgecolor=color,
            node_size=34,
            node_linewidth=1.5,
            zorder=2 + element_index,
        )
        mesh.draw_mesh(
            state.current_nodes,
            element_indices[None, :],
            style=current_style,
        )

    # Draw one, and only one, reference representative in every panel.
    mesh.draw_mesh(
        state.canonical_reference_element,
        TRIANGLE_CONNECTIVITY,
        style=reference_style,
    )

    mesh.configure_axis(
        xlim=xlim,
        ylim=(-0.18, 1.18),
        equal_aspect=True,
        hide_axes=True,
    )


def make_edge_flip_deformation_gradient_figure(
    shear: float = 0.75,
) -> tuple[plt.Figure, np.ndarray, EdgeFlipExample]:
    """Create the canonical-reference edge-flip illustration.

    Each column contains the two element-local representatives.  The second
    is displayed as ``F Q`` to make clear that ``Q`` survives reconnection
    while the base representative changes.
    """

    _configure_matplotlib()
    example = build_edge_flip_example(shear)
    figure = plt.figure(figsize=(9.2, 5.6))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=(0.72, 1.28),
        width_ratios=(1.0, 0.30, 1.0),
        hspace=0.0,
        wspace=0.02,
    )
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = figure.add_subplot(grid[0, 0])
    axes[0, 1] = figure.add_subplot(grid[0, 2])
    axes[1, 0] = figure.add_subplot(grid[1, 0])
    axes[1, 1] = figure.add_subplot(grid[1, 2])
    center_axis = figure.add_subplot(grid[1, 1])
    center_axis.axis("off")

    displayed_states = example.states[1:]
    matrix_labels = (
        (
            r"\mathbf{F}^{(1)-}=\mathbf{F}=",
            r"\mathbf{F}^{(2)-}=\mathbf{F}\mathbf{Q}=",
        ),
        (
            r"\mathbf{F}^{(1)+}=",
            r"\mathbf{F}^{(2)+}=\mathbf{F}^{(1)+}\mathbf{Q}=",
        ),
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
        for element_index, color in enumerate(CURRENT_COLORS):
            matrix_x = (
                (0.26, 0.80),
                (0.20, 0.74),
            )[column][element_index]
            _draw_matrix(
                top_axis,
                state.deformation_gradients[element_index],
                label=matrix_labels[column][element_index],
                color=color,
                x=matrix_x,
                y=0.28,
                fontsize=12.5,
            )
        _draw_geometry_panel(
            axes[1, column],
            state,
            xlim=(-0.18, 1.93),
        )

    _draw_matrix(
        center_axis,
        example.F,
        label="",
        color="black",
        x=0.68,
        y=0.57,
        fontsize=12.5,
    )
    _draw_matrix(
        center_axis,
        example.Q,
        label="",
        color="black",
        x=0.68,
        y=0.30,
        fontsize=12.5,
    )
    center_axis.text(
        0.02,
        0.57,
        r"$\mathbf{F}=$",
        transform=center_axis.transAxes,
        fontsize=17.0,
        ha="left",
        va="center",
    )
    center_axis.text(
        0.02,
        0.30,
        r"$\mathbf{Q}=$",
        transform=center_axis.transAxes,
        fontsize=17.0,
        ha="left",
        va="center",
    )

    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=REFERENCE_COLOR,
                linestyle=(0, (4.0, 3.0)),
                linewidth=1.8,
                label=r"canonical reference element",
            ),
            Line2D(
                [0],
                [0],
                color=CURRENT_COLORS[0],
                linestyle="-",
                linewidth=2.15,
                label=r"current element 1",
            ),
            Line2D(
                [0],
                [0],
                color=CURRENT_COLORS[1],
                linestyle="-",
                linewidth=2.15,
                label=r"current element 2",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.175),
        ncol=3,
        frameon=False,
        fontsize=14.0,
        handlelength=2.8,
    )
    figure.subplots_adjust(left=0.015, right=0.99, top=0.98, bottom=0.20)
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

"""Illustrate MTS2D's four-candidate reference-element selection.

After a reconnection, MTS2D keeps the slightly deformed current triangle and
tests the four unit-square corner triangles used by
``closestSquareReferenceNodes`` in ``src/Mesh/tElement.cpp``.  The candidate
whose two legs have the largest total alignment with the current legs is
selected.  This figure shows every candidate, the resulting deformation
gradient, and the winning choice for one representative current triangle.

Run from the repository root, for example::

    MPLBACKEND=Agg MPLCONFIGDIR=.matplotlib-cache PYTHONPATH=. \\
      ./.venv/bin/python Plotting/reference_element_selection_schematic.py
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from Plotting.meshFigures import (
    CURRENT_COLOR,
    REFERENCE_COLOR,
    TRIANGLE_CONNECTIVITY,
    _configure_matplotlib,
    _format_matrix_entry,
)
from Plotting.mesh_plotting import MeshFigure, MeshStyle


SELECTION_BACKGROUND = "#EEF7EA"


@dataclass(frozen=True)
class Candidate:
    """One corner-triangle candidate, in MTS2D's source-code order."""

    label: str
    corner_names: str
    reference_nodes: np.ndarray


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    score: float
    reference_nodes: np.ndarray
    current_nodes: np.ndarray
    deformation_gradient: np.ndarray


def current_triangle() -> np.ndarray:
    """Return a gently distorted, counterclockwise current triangle.

    Its largest angle is at node zero, so it uses the same angle-node branch
    as the selection routine in MTS2D.  The two vectors deliberately favour
    the upper-right square corner (candidate C).
    """

    return np.array(
        [[0.0, 0.0], [-1.08, 0.18], [-0.16, -0.96]], dtype=float
    )


def source_order_candidates() -> tuple[Candidate, ...]:
    """Return the four candidates from ``closestSquareReferenceNodes``.

    The node sequence is ``angleCorner, adjacentCorner1, adjacentCorner2``;
    it intentionally follows the C++ source before the later CCW ordering
    step.  This lets the score below be a direct transcription of the code.
    """

    return (
        Candidate(
            "A",
            "ADB",
            np.array([[-0.5, 0.5], [0.5, 0.5], [-0.5, -0.5]], dtype=float),
        ),
        Candidate(
            "B",
            "ADC",
            np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]], dtype=float),
        ),
        Candidate(
            "C",
            "BCA",
            np.array([[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5]], dtype=float),
        ),
        Candidate(
            "D",
            "BCD",
            np.array([[0.5, -0.5], [-0.5, -0.5], [0.5, 0.5]], dtype=float),
        ),
    )


def _counterclockwise_pair(
    reference_nodes: np.ndarray, current_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MTS2D's CCW reference-ordering convention to a node pair."""

    reference_edges = (reference_nodes[1:] - reference_nodes[0]).T
    if np.linalg.det(reference_edges) > 0.0:
        return reference_nodes, current_nodes
    return reference_nodes[[0, 2, 1]], current_nodes[[0, 2, 1]]


def deformation_gradient(reference_nodes: np.ndarray, current_nodes: np.ndarray) -> np.ndarray:
    """Return ``F`` satisfying ``F D_ref = D_cur`` for one linear triangle."""

    d_ref = (reference_nodes[1:] - reference_nodes[0]).T
    d_current = (current_nodes[1:] - current_nodes[0]).T
    return d_current @ np.linalg.inv(d_ref)


def evaluate_candidates(nodes: np.ndarray | None = None) -> tuple[CandidateResult, ...]:
    """Evaluate the exact two-dot-product score used by MTS2D."""

    source_current_nodes = current_triangle() if nodes is None else np.asarray(nodes, dtype=float)
    if source_current_nodes.shape != (3, 2):
        raise ValueError("nodes must have shape (3, 2)")
    u, v = source_current_nodes[1] - source_current_nodes[0], source_current_nodes[2] - source_current_nodes[0]

    results = []
    for candidate in source_order_candidates():
        reference_nodes = candidate.reference_nodes
        leg1, leg2 = reference_nodes[1] - reference_nodes[0], reference_nodes[2] - reference_nodes[0]
        score12 = float(u @ leg1 + v @ leg2)
        score21 = float(u @ leg2 + v @ leg1)
        if score12 >= score21:
            paired_reference = reference_nodes.copy()
        else:
            paired_reference = reference_nodes[[0, 2, 1]].copy()
        # ``closestSquareReferenceNodes`` changes reference positions only;
        # ``orderNodes`` subsequently applies the same CCW reordering to both
        # node arrays.  Keeping the current nodes intact here matches that
        # source-code sequence exactly.
        paired_current = source_current_nodes.copy()
        paired_reference, paired_current = _counterclockwise_pair(paired_reference, paired_current)
        results.append(
            CandidateResult(
                candidate=candidate,
                score=max(score12, score21),
                reference_nodes=paired_reference,
                current_nodes=paired_current,
                deformation_gradient=deformation_gradient(paired_reference, paired_current),
            )
        )
    return tuple(results)


def _bottom_aligned(nodes: np.ndarray) -> np.ndarray:
    """Translate a triangle to a shared baseline without changing its ``F``.

    The four reference corners point in different directions.  Centring each
    triangle horizontally and putting its lowest point on ``y=0`` makes every
    overlaid pair easy to compare while retaining its actual orientation.
    """

    nodes = np.asarray(nodes, dtype=float)
    return nodes - np.array([nodes[:, 0].mean(), nodes[:, 1].min()])


def polar_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the orthogonal and right-stretch factors ``R, U`` of ``F=R U``."""

    matrix = np.array(matrix, dtype=float, copy=True, order="C")
    if matrix.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {matrix.shape}.")
    right_cauchy_green = np.array(
        [
            [
                matrix[0, 0] ** 2 + matrix[1, 0] ** 2,
                matrix[0, 0] * matrix[0, 1] + matrix[1, 0] * matrix[1, 1],
            ],
            [
                matrix[0, 0] * matrix[0, 1] + matrix[1, 0] * matrix[1, 1],
                matrix[0, 1] ** 2 + matrix[1, 1] ** 2,
            ],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(right_cauchy_green)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("polar decomposition requires a nonsingular matrix.")
    stretch = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    orthogonal = matrix @ np.linalg.inv(stretch)
    return orthogonal, stretch


def _matrix_tex(matrix: np.ndarray) -> str:
    """Return a compact MathText representation of a 2x2 matrix."""

    entries = [[_format_matrix_entry(value) for value in row] for row in matrix]
    return (
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


def _draw_polar_factors(ax, matrix: np.ndarray, label: str) -> None:
    """Draw ``F = R U`` using ``R`` for the orthogonal polar factor."""

    orthogonal, stretch = polar_decomposition(matrix)
    factor = r"\mathbf{R}"
    ax.text(
        0.5,
        0.58,
        rf"$\mathbf{{F}}^{{({label})}}={factor}^{{({label})}}\mathbf{{U}}^{{({label})}}$",
        transform=ax.transAxes,
        fontsize=10.0,
        color="black",
        ha="center",
        va="center",
    )
    ax.text(
        0.30,
        0.30,
        rf"${factor}^{{({label})}}$",
        transform=ax.transAxes,
        fontsize=7.5,
        color="black",
        ha="center",
        va="center",
    )
    ax.text(
        0.72,
        0.30,
        rf"$\mathbf{{U}}^{{({label})}}$",
        transform=ax.transAxes,
        fontsize=7.5,
        color="black",
        ha="center",
        va="center",
    )
    ax.text(
        0.30,
        0.06,
        rf"${_matrix_tex(orthogonal)}$",
        transform=ax.transAxes,
        fontsize=7.0,
        color="black",
        ha="center",
        va="center",
    )
    ax.text(
        0.72,
        0.06,
        rf"${_matrix_tex(stretch)}$",
        transform=ax.transAxes,
        fontsize=7.0,
        color="black",
        ha="center",
        va="center",
    )


def _draw_pair(ax, result: CandidateResult) -> None:
    reference_style = MeshStyle(
        color=REFERENCE_COLOR,
        face_alpha=0.0,
        linewidth=1.8,
        linestyle=(0, (4.0, 3.5)),
        draw_faces=False,
        draw_nodes=False,
        zorder=5,
    )
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
    mesh = MeshFigure(ax)
    mesh.draw_mesh(
        _bottom_aligned(result.current_nodes),
        TRIANGLE_CONNECTIVITY,
        style=current_style,
    )
    mesh.draw_mesh(
        _bottom_aligned(result.reference_nodes),
        TRIANGLE_CONNECTIVITY,
        style=reference_style,
    )
    mesh.configure_axis(
        xlim=(-0.95, 0.95),
        ylim=(-0.10, 1.30),
        equal_aspect=True,
        hide_axes=True,
    )


def make_figure(*, out_pdf: Path, out_png: Path) -> tuple[CandidateResult, ...]:
    """Write the four-candidate reference-selection schematic."""

    _configure_matplotlib()
    results = evaluate_candidates()
    selected_index = int(np.argmax([result.score for result in results]))
    figure = plt.figure(figsize=(8.8, 3.2))
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=(0.60, 1.38),
        hspace=0.02,
        wspace=0.018,
        left=0.03,
        right=0.97,
        bottom=0.12,
        top=0.965,
    )
    matrix_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    pair_axes = [figure.add_subplot(grid[1, index]) for index in range(4)]

    for index, (matrix_ax, pair_ax, result) in enumerate(zip(matrix_axes, pair_axes, results)):
        if index == selected_index:
            # Keep the selected-state tint inside the geometric panel, with a
            # small inset at its top and bottom so it does not feel oversized.
            pair_ax.add_patch(
                Rectangle(
                    (0.0, 0.03),
                    1.0,
                    0.94,
                    transform=pair_ax.transAxes,
                    facecolor=SELECTION_BACKGROUND,
                    edgecolor="none",
                    zorder=-10,
                )
            )
        matrix_ax.text(
            0.5,
            0.96,
            rf"\textbf{{({chr(ord('a') + index)})}}",
            transform=matrix_ax.transAxes,
            fontsize=10.5,
            color="black",
            ha="center",
            va="top",
        )
        _draw_polar_factors(
            matrix_ax,
            result.deformation_gradient,
            chr(ord("a") + index),
        )
        matrix_ax.axis("off")
        _draw_pair(pair_ax, result)

    figure.legend(
        handles=(
            Line2D([0], [0], color=REFERENCE_COLOR, linestyle=(0, (4.0, 3.0)), linewidth=1.8, label="reference element"),
            Line2D([0], [0], color=CURRENT_COLOR, linewidth=2.0, label="current element"),
        ),
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.09), fontsize=9.0,
        frameon=False, handlelength=2.3, columnspacing=2.1,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Crop only the exported canvas.  This preserves the deliberate GridSpec
    # geometry and figure-level legend placement, unlike ``tight_layout``.
    figure.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-pdf", type=Path, default=Path("Plots/reference_element_selection_schematic.pdf"))
    parser.add_argument("--out-png", type=Path, default=Path("Plots/reference_element_selection_schematic.png"))
    args = parser.parse_args()
    results = make_figure(out_pdf=args.out_pdf, out_png=args.out_png)
    selected = max(results, key=lambda result: result.score)
    print(args.out_pdf)
    print(args.out_png)
    print(f"Selected candidate: {selected.candidate.label} (score={selected.score:.3f})")


if __name__ == "__main__":
    main()

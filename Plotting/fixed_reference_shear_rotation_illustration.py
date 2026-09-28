"""Show how a stale element reference misreads a shear after an edge flip.

The default source is the archived 8x8 double-dislocation inspection run.  It
uses the before/after snapshots of element 48 at gamma=1, where an edge flip
changes the element's reference triangle.  The plot deliberately compares the
same post-flip current triangle against two choices of reference geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting.matrix_visualization import draw_matrix_columns
from Plotting.mesh_plotting import MeshFigure, MeshStyle


DEFAULT_HISTORY = (
    ROOT
    / "MTS2D_reversibility_dump"
    / "test_data"
    / "doubleDislocation8x8Inspection"
    / "element_48_matrix_history.json"
)
DEFAULT_OUTPUT_STEM = (
    ROOT / "Plots" / "reconnecting_reference" / "fixed_reference_shear_rotation"
)
TRIANGLE = np.array([[0, 1, 2]], dtype=int)

TEXT = "#20262E"
MUTED = "#66717B"
GRID = "#CBD2D8"
REFERENCE = "#087E8B"
CURRENT = "#D95F02"
CORRECT = "#2E7D32"
STALE = "#A23E48"
COLUMN_COLORS = (REFERENCE, CURRENT)
MATRIX_LIMITS = (-1.42, 1.42)


def as_matrix(value, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {matrix.shape}.")
    return matrix


def as_triangle(value, *, name: str) -> np.ndarray:
    triangle = np.asarray(value, dtype=float)
    if triangle.shape != (3, 2):
        raise ValueError(f"{name} must have shape (3, 2), got {triangle.shape}.")
    return triangle


def edge_matrix(nodes: np.ndarray) -> np.ndarray:
    """Return the two edge vectors from local node zero as matrix columns."""
    return (nodes[1:] - nodes[0]).T


def deformation_from(reference_nodes: np.ndarray, current_nodes: np.ndarray) -> np.ndarray:
    reference_edges = edge_matrix(reference_nodes)
    if abs(np.linalg.det(reference_edges)) < 1e-12:
        raise ValueError("The selected reference triangle is degenerate.")
    return edge_matrix(current_nodes) @ np.linalg.inv(reference_edges)


def polar_angle(matrix: np.ndarray) -> float:
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        raise ValueError("Expected an orientation-preserving matrix for its polar angle.")
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def format_number(value: float) -> str:
    rounded = 0.0 if abs(value) < 5e-4 else round(float(value), 2)
    return f"{rounded:g}"


def matrix_latex(matrix: np.ndarray) -> str:
    matrix = as_matrix(matrix, name="matrix")
    return (
        r"\begin{pmatrix}"
        f"{format_number(matrix[0, 0])} & {format_number(matrix[0, 1])} \\\\ "
        f"{format_number(matrix[1, 0])} & {format_number(matrix[1, 1])}"
        r"\end{pmatrix}"
    )


def format_angle(angle: float) -> str:
    if abs(abs(angle) - 180.0) < 0.1:
        return r"180^\circ"
    return f"{angle:.1f}^\\circ"


def load_flip_data(history_path: Path, gamma: float) -> dict[str, np.ndarray | float]:
    payload = json.loads(history_path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{history_path} does not contain non-empty 'rows' data.")

    matches = [row for row in rows if np.isclose(float(row["gamma"]), gamma)]
    if len(matches) != 1:
        raise ValueError(f"Expected one row at gamma={gamma:g}, found {len(matches)}.")
    row = matches[0]
    if not row.get("element_changed", False):
        raise ValueError(f"No reference-changing edge flip was recorded at gamma={gamma:g}.")

    first_after = rows[0].get("after")
    before = row.get("before")
    after = row.get("after")
    if not all(isinstance(state, dict) for state in (first_after, before, after)):
        raise ValueError("The selected row is missing before/after element snapshots.")

    reference_initial = as_triangle(first_after["reference_nodes"], name="initial reference")
    reference_updated = as_triangle(after["reference_nodes"], name="updated reference")
    current_before = as_triangle(before["current_nodes"], name="pre-flip current triangle")
    current_after = as_triangle(after["current_nodes"], name="post-flip current triangle")
    H_after = as_matrix(after["H"], name="post-flip H")

    F_before = deformation_from(
        as_triangle(before["reference_nodes"], name="pre-flip reference"),
        current_before,
    )
    F_updated = deformation_from(reference_updated, current_after)
    F_fixed = deformation_from(reference_initial, current_after)
    T_updated = F_updated @ H_after

    return {
        "gamma": float(row["gamma"]),
        "reference_initial": reference_initial,
        "reference_updated": reference_updated,
        "current_before": current_before,
        "current_after": current_after,
        "F_before": F_before,
        "F_updated": F_updated,
        "F_fixed": F_fixed,
        "H_after": H_after,
        "T_updated": T_updated,
    }


def centered(nodes: np.ndarray) -> np.ndarray:
    return nodes - nodes.mean(axis=0)


def draw_geometry_pair(
    ax,
    *,
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    title: str,
    result_label: str,
    result_color: str,
) -> None:
    reference_nodes = centered(reference_nodes)
    current_nodes = centered(current_nodes)
    reference_style = MeshStyle(
        color=REFERENCE,
        face_alpha=0.04,
        linewidth=2.0,
        linestyle="--",
        node_size=30,
        node_linewidth=0.0,
        zorder=2,
    )
    current_style = MeshStyle(
        color=CURRENT,
        face_alpha=0.12,
        linewidth=2.0,
        linestyle="-",
        node_size=30,
        node_linewidth=0.0,
        zorder=3,
    )
    mesh = MeshFigure(ax)
    mesh.draw_mesh(reference_nodes, TRIANGLE, style=reference_style)
    mesh.draw_mesh(current_nodes, TRIANGLE, style=current_style)
    for index, (point, label) in enumerate(zip(reference_nodes, ("0", "1", "2"))):
        direction = point / max(np.linalg.norm(point), 1e-12)
        ax.text(
            *(point + 0.10 * direction),
            rf"$X_{label}$",
            color=REFERENCE,
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
        )
    for point, label in zip(current_nodes, ("0", "1", "2")):
        direction = point / max(np.linalg.norm(point), 1e-12)
        ax.text(
            *(point + 0.19 * direction),
            rf"$x_{label}$",
            color=CURRENT,
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
        )
    ax.set_title(title, fontsize=10, color=TEXT, pad=8, fontweight="semibold")
    ax.text(
        0.5,
        -0.10,
        result_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=result_color,
        fontweight="semibold",
    )
    ax.set_xlim(-0.92, 0.92)
    ax.set_ylim(-0.92, 0.92)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
def draw_matrix_panel(
    ax,
    *,
    matrix: np.ndarray,
    title: str,
    expression: str,
    interpretation: str,
    color: str = TEXT,
) -> None:
    draw_matrix_columns(
        ax,
        matrix,
        limits=MATRIX_LIMITS,
        colors=COLUMN_COLORS,
        linestyles=("-", "-"),
        linewidth=2.1,
        mutation_scale=12,
        grid_color=GRID,
        show_ticks=False,
    )
    ax.set_title(title, fontsize=9.2, pad=8, color=color, fontweight="semibold")
    ax.text(
        0.5,
        -0.12,
        rf"${expression} \simeq {matrix_latex(matrix)}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.4,
        color=TEXT,
    )
    ax.text(
        0.5,
        -0.30,
        interpretation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color=color,
    )


def make_figure(data: dict[str, np.ndarray | float]):
    gamma = float(data["gamma"])
    F_before = np.asarray(data["F_before"])
    H_after = np.asarray(data["H_after"])
    F_updated = np.asarray(data["F_updated"])
    T_updated = np.asarray(data["T_updated"])
    F_fixed = np.asarray(data["F_fixed"])

    fig = plt.figure(figsize=(14.2, 9.8), facecolor="white")
    fig.text(
        0.5,
        0.972,
        "A fixed element reference turns a shear into a rotation after an edge flip",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="semibold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.942,
        (
            r"Archived double-dislocation inspection, element 48 at "
            rf"$\gamma={gamma:g}$; the right two geometry panels use the identical post-flip current triangle."
        ),
        ha="center",
        va="top",
        fontsize=10,
        color=MUTED,
    )

    grid = fig.add_gridspec(
        2,
        1,
        left=0.05,
        right=0.98,
        top=0.88,
        bottom=0.19,
        height_ratios=(1.04, 1.0),
        hspace=0.52,
    )
    geometry_grid = grid[0].subgridspec(1, 3, wspace=0.28)
    draw_geometry_pair(
        fig.add_subplot(geometry_grid[0, 0]),
        reference_nodes=np.asarray(data["reference_initial"]),
        current_nodes=np.asarray(data["current_before"]),
        title=rf"Immediately before the flip, $\gamma={gamma:g}^-$",
        result_label=r"$F^-\simeq S_x$ (simple shear)",
        result_color=CORRECT,
    )
    draw_geometry_pair(
        fig.add_subplot(geometry_grid[0, 1]),
        reference_nodes=np.asarray(data["reference_initial"]),
        current_nodes=np.asarray(data["current_after"]),
        title=r"Same post-flip triangle with frozen $\mathbf{X}^0$",
        result_label=r"$F_{\rm fixed}^+\simeq-I$ (rotation)",
        result_color=STALE,
    )
    draw_geometry_pair(
        fig.add_subplot(geometry_grid[0, 2]),
        reference_nodes=np.asarray(data["reference_updated"]),
        current_nodes=np.asarray(data["current_after"]),
        title=r"Same post-flip triangle with updated $\mathbf{X}^+$",
        result_label=r"$F^+\simeq I$; the shear is retained in $H^+$",
        result_color=CORRECT,
    )

    matrix_grid = grid[1].subgridspec(1, 5, wspace=0.56)
    draw_matrix_panel(
        fig.add_subplot(matrix_grid[0, 0]),
        matrix=F_before,
        title=r"pre-flip $F^-$",
        expression=r"F^-",
        interpretation=f"polar angle ${format_angle(polar_angle(F_before))}$\nnot a pure rotation",
        color=CORRECT,
    )
    draw_matrix_panel(
        fig.add_subplot(matrix_grid[0, 1]),
        matrix=H_after,
        title=r"updated reference branch $H^+$",
        expression=r"H^+",
        interpretation="stores the unit shear\nacross the edge flip",
        color=CORRECT,
    )
    draw_matrix_panel(
        fig.add_subplot(matrix_grid[0, 2]),
        matrix=F_updated,
        title=r"updated local $F^+$",
        expression=r"F^+",
        interpretation="the refreshed triangle\nis locally undeformed",
        color=CORRECT,
    )
    draw_matrix_panel(
        fig.add_subplot(matrix_grid[0, 3]),
        matrix=T_updated,
        title=r"correct total $T^+=F^+H^+$",
        expression=r"T^+",
        interpretation=f"polar angle ${format_angle(polar_angle(T_updated))}$\nstill the original shear",
        color=CORRECT,
    )
    fixed_axis = fig.add_subplot(matrix_grid[0, 4])
    draw_matrix_panel(
        fixed_axis,
        matrix=F_fixed,
        title=r"stale-reference $F_{\rm fixed}^+$",
        expression=r"F_{\rm fixed}^+",
        interpretation=f"polar angle ${format_angle(polar_angle(F_fixed))}$\npure half-turn rotation",
        color=STALE,
    )
    fixed_axis.set_facecolor("#FFF6F5")
    for spine in fixed_axis.spines.values():
        spine.set_color(STALE)
        spine.set_linewidth(1.25)

    fig.text(
        0.5,
        0.100,
        (
            r"Correct tracking: $T^+=F^+H^+\simeq S_x$, matching the pre-flip shear.  "
            r"If the reference is frozen at $\mathbf{X}^0$, the same post-flip nodes instead give "
            r"$F_{\rm fixed}^+=D\mathbf{x}^+\,(D\mathbf{X}^0)^{-1}\simeq-I$."
        ),
        ha="center",
        va="center",
        fontsize=10.2,
        color=TEXT,
    )
    fig.text(
        0.5,
        0.064,
        (
            r"Here $D\mathbf{x}=[\mathbf{x}_1-\mathbf{x}_0\;\;\mathbf{x}_2-\mathbf{x}_0]$; "
            r"the reference/current triangles are independently centred only for display."
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        color=MUTED,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    data = load_flip_data(args.history, args.gamma)
    figure = make_figure(data)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()

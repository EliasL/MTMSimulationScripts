"""Illustrate which elastic quadrant recovers a known plastic jump."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MTMath.reduction import elastic_domain_quadrant
from Plotting.matrix_visualization import draw_matrix_columns


TOTAL_F = np.array([[-0.43, 1.21], [-1.19, 1.02]], dtype=float)
KNOWN_PLASTIC_F = np.array([[1.0, -1.0], [0.0, 1.0]])

M1 = np.array([[1.0, 0.0], [0.0, -1.0]])
M2 = np.array([[0.0, 1.0], [1.0, 0.0]])

OUT = ROOT / "Plots" / "plastic_reduction"
OUTPUT_STEM = OUT / "plastic_reduction_correctness_illustration"
PLOT_LIMIT = float(np.max(np.abs(TOTAL_F)))

COLUMN_COLORS = ("#087E8B", "#D95F02")
COLUMN_LINESTYLES = ("-", "-")
TEXT = "#20262E"
MUTED = "#66717B"
GRID = "#CBD2D8"
CORRECT = "#2E7D32"
LAGRANGE_M0_COLOR = "#9DFA9B"
CORRECT_ARROW_LINEWIDTH = 2.5
LAGRANGE_M0_ARROW_LINEWIDTH = 2.5
LAGRANGE_M0_MUTATION_SCALE = 22


def decomposition_data():
    """Return the known factors and all four symmetry-related reductions."""
    base_M = np.linalg.inv(KNOWN_PLASTIC_F)
    data = decomposition_data_from_M(base_M)

    matching_quadrants = [
        branch["quadrant"] for branch in data["branches"] if branch["matches"]
    ]
    if matching_quadrants != [3]:
        raise RuntimeError(
            f"Expected quadrant 3 to be the unique match, got {matching_quadrants}"
        )
    return data


def decomposition_data_from_M(base_M):
    """Build the four quadrant decompositions generated from ``base_M``."""
    base_M = np.asarray(base_M, dtype=float)
    if base_M.shape != (2, 2):
        raise ValueError(f"base_M must have shape (2, 2), got {base_M.shape}")

    known_F_p = KNOWN_PLASTIC_F.copy()
    known_F_e = TOTAL_F @ np.linalg.inv(known_F_p)
    if not np.allclose(known_F_e @ known_F_p, TOTAL_F):
        raise RuntimeError("Known elastic-plastic factors do not reconstruct F")

    symmetry_branches = (
        (r"\mathbf{M}", np.eye(2), "recovers the known factors"),
        (
            r"\mathbf{M}\mathbf{m}_1",
            M1,
            "elastic column 2 changes sign",
        ),
        (
            r"\mathbf{M}\mathbf{m}_2",
            M2,
            "elastic columns 1 and 2 are exchanged",
        ),
        (
            r"\mathbf{M}\mathbf{m}_1\mathbf{m}_2",
            M1 @ M2,
            "elastic columns are exchanged with a sign change",
        ),
    )

    branches = []
    for index, (expression, symmetry, description) in enumerate(symmetry_branches):
        reduction_M = base_M @ symmetry
        F_e = TOTAL_F @ reduction_M
        F_p = np.linalg.inv(reduction_M)
        quadrant = int(elastic_domain_quadrant(F_e.T @ F_e))
        if index == 3:
            label = r"$\mathbf{M}_0^{(c)}=\mathbf{M}\mathbf{m}_1\mathbf{m}_2$"
            decomposition_superscript = "0,c"
        else:
            label = rf"$\mathbf{{M}}_{{{quadrant}}}={expression}$"
            decomposition_superscript = str(quadrant)
        matches = np.allclose(F_e, known_F_e) and np.allclose(
            F_p, known_F_p
        )
        branches.append(
            {
                "label": label,
                "decomposition_superscript": decomposition_superscript,
                "description": description,
                "quadrant": quadrant,
                "M": reduction_M,
                "F_e": F_e,
                "F_p": F_p,
                "preferred": index == 0,
                "lagrange": index == 3,
                "matches": matches,
                "elastic_error": np.linalg.norm(F_e - known_F_e),
                "plastic_error": np.linalg.norm(F_p - known_F_p),
            }
        )

    return {
        "F": TOTAL_F.copy(),
        "known_F_e": known_F_e,
        "known_F_p": known_F_p,
        "base_M": base_M,
        "branches": branches,
    }


def counterclockwise_branch(data):
    """Return the extra counter-clockwise partner of the M0 branch."""

    symmetry = M2 @ M1
    reduction_M = data["base_M"] @ symmetry
    F_e = data["F"] @ reduction_M
    F_p = np.linalg.inv(reduction_M)
    return {
        "label": r"$\mathbf{M}_0^{(cc)}=\mathbf{M}\mathbf{m}_2\mathbf{m}_1$",
        "decomposition_superscript": "0,cc",
        "description": "counter-clockwise quarter-turn from M",
        "quadrant": int(elastic_domain_quadrant(F_e.T @ F_e)),
        "M": reduction_M,
        "F_e": F_e,
        "F_p": F_p,
        "preferred": False,
        "lagrange": False,
        "matches": False,
        "elastic_error": np.linalg.norm(F_e - data["known_F_e"]),
        "plastic_error": np.linalg.norm(F_p - data["known_F_p"]),
    }


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _matrix_axis(fig, position, matrix, title, column_linestyles):
    ax = fig.add_axes(position)
    draw_matrix_columns(
        ax,
        matrix,
        limits=(-PLOT_LIMIT, PLOT_LIMIT),
        title=title,
        colors=COLUMN_COLORS,
        linestyles=column_linestyles,
        linewidth=2.35,
        mutation_scale=12,
        grid_color=GRID,
        origin_color=TEXT,
        show_ticks=False,
    )
    ax.axhline(0.0, color=GRID, linewidth=0.55, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.55, zorder=0)
    return ax


def make_figure(data=None, *, column_linestyles=COLUMN_LINESTYLES):
    """Create the forward construction and four reverse decompositions."""
    if data is None:
        data = decomposition_data()

    fig = plt.figure(figsize=(11.2, 7.0))
    fig.text(
        0.365,
        0.33,
        r"$\mathbf{M}=(\mathbf{F}_p^\star)^{-1}$"
        "\n"
        r"$\mathbf{F}_e^{(q)}=\mathbf{F}\mathbf{M}_q$, "
        r"$\mathbf{F}_p^{(q)}=\mathbf{M}_q^{-1}$",
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.2,
        color=TEXT,
    )

    known_width = 0.13
    known_height = 0.18
    elastic_axis = _matrix_axis(
        fig,
        [0.10, 0.625, known_width, known_height],
        data["known_F_e"],
        r"$\mathbf{F}_e^\star$",
        column_linestyles,
    )
    plastic_axis = _matrix_axis(
        fig,
        [0.10, 0.355, known_width, known_height],
        data["known_F_p"],
        r"$\mathbf{F}_p^\star$",
        column_linestyles,
    )
    product_axis = _matrix_axis(
        fig,
        [0.29, 0.475, 0.15, 0.21],
        data["F"],
        r"$\mathbf{F}=\mathbf{F}_e^\star\mathbf{F}_p^\star$",
        column_linestyles,
    )
    stacked_center = (
        elastic_axis.get_position().x1,
        0.5
        * (
            elastic_axis.get_position().y0
            + elastic_axis.get_position().height / 2
            + plastic_axis.get_position().y0
            + plastic_axis.get_position().height / 2
        ),
    )
    product_box = product_axis.get_position()
    fig.add_artist(
        FancyArrowPatch(
            stacked_center,
            (product_box.x0 - 0.01, product_box.y0 + product_box.height / 2),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=MUTED,
        )
    )

    elastic_x = 0.58
    plastic_x = 0.695
    branch_width = 0.11
    branch_height = 0.11
    row_bottoms = (0.80, 0.65, 0.50, 0.35, 0.20)
    equation_x = elastic_x - 0.095
    equation_arrow_x = equation_x - 0.01
    arrow_start = (product_box.x1, product_box.y0 + product_box.height / 2)
    right_branches = (
        list(data["branches"][:-1])
        + [counterclockwise_branch(data), data["branches"][-1]]
    )
    curvatures = (-0.18, -0.09, 0.0, 0.09, 0.18)

    for index, (branch, row_bottom) in enumerate(
        zip(right_branches, row_bottoms)
    ):
        row_center = row_bottom + branch_height / 2
        _matrix_axis(
            fig,
            [elastic_x, row_bottom, branch_width, branch_height],
            branch["F_e"],
            None,
            column_linestyles,
        )
        _matrix_axis(
            fig,
            [plastic_x, row_bottom, branch_width, branch_height],
            branch["F_p"],
            None,
            column_linestyles,
        )
        fig.text(
            elastic_x + branch_width / 2,
            row_bottom + branch_height + 0.008,
            rf"$\mathbf{{F}}_e^{{({branch['decomposition_superscript']})}}$",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )
        fig.text(
            plastic_x + branch_width / 2,
            row_bottom + branch_height + 0.008,
            rf"$\mathbf{{F}}_p^{{({branch['decomposition_superscript']})}}$",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )

        arrow_end = (equation_arrow_x, row_center)
        curvature = curvatures[index]
        connectionstyle = f"arc3,rad={curvature}"
        if branch["preferred"]:
            fig.add_artist(
                FancyArrowPatch(
                    arrow_start,
                    arrow_end,
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=CORRECT_ARROW_LINEWIDTH,
                    color=CORRECT,
                    connectionstyle=connectionstyle,
                    zorder=-1,
                )
            )
        else:
            branch_is_lagrange = branch.get("lagrange", False)
            branch_color = LAGRANGE_M0_COLOR if branch_is_lagrange else TEXT
            branch_linewidth = (
                LAGRANGE_M0_ARROW_LINEWIDTH
                if branch_is_lagrange
                else 1.25
            )
            branch_mutation_scale = (
                LAGRANGE_M0_MUTATION_SCALE
                if branch_is_lagrange
                else 13
            )
            fig.add_artist(
                FancyArrowPatch(
                    arrow_start,
                    arrow_end,
                    transform=fig.transFigure,
                    arrowstyle="-",
                    linewidth=branch_linewidth,
                    linestyle="-",
                    color=branch_color,
                    connectionstyle=connectionstyle,
                    zorder=-1,
                )
            )
            fig.add_artist(
                FancyArrowPatch(
                    arrow_start,
                    arrow_end,
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=branch_mutation_scale,
                    linewidth=0,
                    edgecolor=branch_color,
                    facecolor=branch_color,
                    connectionstyle=connectionstyle,
                    zorder=-1,
                )
            )
        fig.text(
            equation_x,
            row_center,
            branch["label"],
            ha="left",
            va="center",
            fontsize=9.5,
            color=TEXT,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    legend = [
        Line2D(
            [0],
            [0],
            color=COLUMN_COLORS[0],
            linestyle=column_linestyles[0],
            linewidth=2.35,
            label=r"$\mathbf{A}_{:1}$",
        ),
        Line2D(
            [0],
            [0],
            color=COLUMN_COLORS[1],
            linestyle=column_linestyles[1],
            linewidth=2.35,
            label=r"$\mathbf{A}_{:2}$",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.365, row_bottoms[0] + branch_height / 2 + 0.015),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        labelspacing=0.45,
    )
    return fig


def print_summary(data):
    for branch in data["branches"]:
        print(
            f"quadrant {branch['quadrant']}: "
            f"match={branch['matches']}, "
            f"elastic error={branch['elastic_error']:.6g}, "
            f"plastic error={branch['plastic_error']:.6g}"
        )
        print(branch["M"])


def save_figure(data, output_stem, *, column_linestyles=COLUMN_LINESTYLES):
    """Save one correctness illustration as both PNG and PDF."""
    fig = make_figure(data, column_linestyles=column_linestyles)
    OUT.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    apply_style()
    data = decomposition_data()
    png_path, pdf_path = save_figure(data, OUTPUT_STEM)
    print_summary(data)
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


if __name__ == "__main__":
    main()

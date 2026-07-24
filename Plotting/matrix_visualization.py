"""Reusable Matplotlib helpers for drawing 2D matrices by their columns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter


DEFAULT_COLUMN_COLORS = ("#008C95", "#008C95")
DEFAULT_COLUMN_LINESTYLES = ("-", "--")
DEFAULT_GRID_COLOR = "#D8DEE5"
DEFAULT_ORIGIN_COLOR = "#1F2933"


def draw_matrix_columns(
    ax,
    matrix,
    *,
    limits=None,
    title=None,
    colors: Sequence[str] = DEFAULT_COLUMN_COLORS,
    linestyles: Sequence[str] = DEFAULT_COLUMN_LINESTYLES,
    linewidth=2.1,
    mutation_scale=12,
    grid_color=DEFAULT_GRID_COLOR,
    origin_color=DEFAULT_ORIGIN_COLOR,
    show_ticks=True,
):
    """Draw the two columns of a 2x2 matrix as independently styled arrows.

    Keeping the column colors fixed makes swaps, sign changes, and rotations
    visually traceable across a sequence of related matrices.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"matrix must have shape (2, 2), got {matrix.shape}")
    if len(colors) != 2 or len(linestyles) != 2:
        raise ValueError("colors and linestyles must each contain two entries")

    def arrowhead_near_boundary(vector):
        if limits is None:
            return False
        x_min, x_max = sorted(limits)
        y_min, y_max = sorted(limits)
        boundary_band = 0.03 * max(x_max - x_min, y_max - y_min)
        return (
            min(vector[0] - x_min, x_max - vector[0]) <= boundary_band
            or min(vector[1] - y_min, y_max - vector[1]) <= boundary_band
        )

    arrows = []
    for vector, color, linestyle in zip(
        (matrix[:, 0], matrix[:, 1]), colors, linestyles
    ):
        clipped_dashed_head = (
            linestyle not in ("-", "solid") and arrowhead_near_boundary(vector)
        )
        if clipped_dashed_head:
            # Preserve the usual dashed FancyArrowPatch everywhere except at a
            # clipped endpoint.  There, its dashed head outline breaks into
            # little spurs, so only the terminal head is redrawn solid.
            ax.plot(
                [0.0, vector[0]],
                [0.0, vector[1]],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                dash_capstyle="butt",
                zorder=3,
            )
            arrow_start = 0.94 * vector
            arrow_linewidth = linewidth
            arrow_linestyle = "-"
        else:
            arrow_start = (0.0, 0.0)
            arrow_linewidth = linewidth
            arrow_linestyle = linestyle

        arrow = FancyArrowPatch(
            tuple(arrow_start),
            tuple(vector),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=arrow_linewidth,
            linestyle=arrow_linestyle,
            color=color,
            shrinkA=0 if clipped_dashed_head else 2,
            shrinkB=0 if clipped_dashed_head else 2,
            zorder=3,
        )
        ax.add_patch(arrow)
        arrows.append(arrow)

    ax.scatter([0.0], [0.0], s=10, color=origin_color, zorder=4)
    if limits is None:
        ax.margins(0.10)
        ax.autoscale_view()
    else:
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)

    def half_step_ticks(bounds):
        lower, upper = sorted(bounds)
        first = np.ceil((lower - 1e-12) * 2.0) / 2.0
        last = np.floor((upper + 1e-12) * 2.0) / 2.0
        return np.arange(first, last + 0.25, 0.5)

    def half_step_label(value, _position):
        return f"{0.0 if abs(value) < 1e-12 else value:g}"

    if show_ticks:
        ax.set_xticks(half_step_ticks(ax.get_xlim()))
        ax.set_yticks(half_step_ticks(ax.get_ylim()))
        ax.xaxis.set_major_formatter(FuncFormatter(half_step_label))
        ax.yaxis.set_major_formatter(FuncFormatter(half_step_label))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(length=2.5, width=0.7, pad=1.5)
    if title is not None:
        ax.set_title(title, pad=7, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_color(grid_color)

    return tuple(arrows)

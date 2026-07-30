"""Relate the eight discrete decompositions to oriented Poincare disks."""

from __future__ import annotations

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from MTMath.poincareEnergy import (
    C2PoincareDisk,
    drawPoincareGrid,
    generate_elastic_quadrant_grid,
)
from MTMath.reduction import lagrange_reduction_history, plastic_reduction_history
from Plotting.matrix_visualization import draw_matrix_columns
from Plotting.plasticReductionAllDecompositionsIllustration import (
    decomposition_table_data,
)
from Plotting.plasticReductionCorrectnessIllustration import (
    COLUMN_COLORS,
    GRID,
    OUT,
    PLOT_LIMIT,
    TEXT,
    apply_style,
)


OUTPUT_STEM = OUT / "plastic_reduction_determinant_quadrants_illustration"
DISK_RESOLUTION = 840
DISK_GAP = -40
DISK_ALPHA = 0.62
CMAP_BY_DETERMINANT = {1: "coolwarm", -1: "viridis"}
PANEL_COLOR_STRENGTH = DISK_ALPHA
HISTORY_F = np.array([[-0.43, 1.21], [-1.19, 1.02]], dtype=float)
LAGRANGE_PATH_COLOR = "#008A27"
ELASTIC_PATH_COLOR = "#69D86E"
INITIAL_MARKER_COLOR = "#C62828"


def quadrant_palette(determinant):
    """Return the four quadrant colors for one determinant sector."""
    determinant = int(determinant)
    if determinant not in CMAP_BY_DETERMINANT:
        raise ValueError("determinant must be +1 or -1")
    return plt.colormaps[CMAP_BY_DETERMINANT[determinant]](
        np.linspace(0.0, 1.0, 4)
    )


def disk_display_palette(determinant):
    """Return the palette used to render one opaque/transparent disk."""
    palette = quadrant_palette(determinant).copy()
    if int(determinant) == 1:
        palette[:, :3] = 1.0 - DISK_ALPHA * (1.0 - palette[:, :3])
        palette[:, 3] = 1.0
    return palette


def panel_background_color(decomposition, strength=PANEL_COLOR_STRENGTH):
    """Return the decomposition's disk-quadrant color."""
    palette = quadrant_palette(decomposition["determinant"])
    color = np.asarray(palette[decomposition["quadrant"], :3], dtype=float)
    return tuple(1.0 - strength * (1.0 - color))


def _text_color_for_background(background_color):
    """Choose readable panel labels for the saturated quadrant colors."""
    red, green, blue = np.asarray(background_color, dtype=float)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.46 else TEXT


def grouped_disk_points(data, determinant):
    """Group expressions that map to the same Poincare point."""
    groups = defaultdict(list)
    for decomposition in data:
        if decomposition["determinant"] != determinant:
            continue
        C_e = decomposition["F_e"].T @ decomposition["F_e"]
        x, y = C2PoincareDisk(C_e)
        key = tuple(np.round([float(x), float(y)], decimals=12))
        groups[key].append(decomposition["expression"])
    return dict(groups)


def _disk_origin(determinant):
    """Return the lower-left coordinate of one determinant disk."""
    if int(determinant) == 1:
        return 0.0, DISK_RESOLUTION + DISK_GAP
    if int(determinant) == -1:
        return 0.0, 0.0
    raise ValueError("determinant must be +1 or -1")


def _draw_grid_at_offset(ax, x_offset, y_offset, zorder=1):
    """Draw the reusable Poincare grid translated to one disk's location."""
    line_start = len(ax.lines)
    drawPoincareGrid(
        ax=ax,
        grid_size=DISK_RESOLUTION,
        depth=5,
        c="#65717D",
        alpha=0.42,
        linewidth=0.42,
        zorder=zorder,
    )
    clip = Circle(
        (
            x_offset + DISK_RESOLUTION / 2,
            y_offset + DISK_RESOLUTION / 2,
        ),
        DISK_RESOLUTION / 2,
        transform=ax.transData,
    )
    for line in ax.lines[line_start:]:
        line.set_xdata(np.asarray(line.get_xdata(), dtype=float) + x_offset)
        line.set_ydata(np.asarray(line.get_ydata(), dtype=float) + y_offset)
        line.set_clip_path(clip)


def _draw_quadrant_disk(ax, quadrant_grid, determinant):
    palette = disk_display_palette(determinant)
    x_offset, y_offset = _disk_origin(determinant)
    quadrant_cmap = colors.ListedColormap(
        palette,
        name=f"{CMAP_BY_DETERMINANT[determinant]}_quadrants",
    )
    # Keep the rectangular raster transparent outside the circular disk so an
    # opaque foreground disk does not erase the exposed part of the other one.
    quadrant_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    quadrant_norm = colors.BoundaryNorm(np.arange(-0.5, 4.5), 4)

    ax.imshow(
        quadrant_grid,
        origin="lower",
        extent=(
            x_offset,
            x_offset + DISK_RESOLUTION,
            y_offset,
            y_offset + DISK_RESOLUTION,
        ),
        interpolation="nearest",
        cmap=quadrant_cmap,
        norm=quadrant_norm,
        alpha=1.0 if determinant == 1 else DISK_ALPHA,
        # The opaque positive disk masks the lower disk's outline and grid
        # inside the overlap, while remaining transparent outside its circle.
        zorder=5 if determinant == 1 else 0,
    )
    _draw_grid_at_offset(
        ax,
        x_offset,
        y_offset,
        zorder=6 if determinant == 1 else 1,
    )
    circle_center = (
        x_offset + DISK_RESOLUTION / 2,
        y_offset + DISK_RESOLUTION / 2,
    )
    ax.add_patch(
        Circle(
            circle_center,
            DISK_RESOLUTION / 2,
            fill=False,
            edgecolor=TEXT,
            linewidth=0.9,
            zorder=7 if determinant == 1 else 4,
        )
    )

    sign = "+" if determinant > 0 else "-"
    title_y = (
        y_offset + DISK_RESOLUTION + 9
        if determinant > 0
        else y_offset - 9
    )
    ax.text(
        circle_center[0],
        title_y,
        rf"$\det\mathbf{{M}}={sign}1$",
        ha="center",
        va="bottom" if determinant > 0 else "top",
        color=TEXT,
        fontsize=10.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 0.8,
        },
        zorder=12,
    )


def _draw_factor(
    ax,
    matrix,
    label,
    background_color,
    label_position="bottom",
):
    ax.set_facecolor(background_color)
    label_color = _text_color_for_background(background_color)
    arrows = draw_matrix_columns(
        ax,
        matrix,
        limits=(-PLOT_LIMIT, PLOT_LIMIT),
        colors=COLUMN_COLORS,
        linestyles=("-", "-"),
        linewidth=2.15,
        mutation_scale=11,
        grid_color=GRID,
        origin_color=TEXT,
        show_ticks=False,
    )
    for arrow in arrows:
        arrow.set_path_effects(
            [
                patheffects.Stroke(linewidth=4.7, foreground="white"),
                patheffects.Normal(),
            ]
        )
    ax.axhline(0.0, color=GRID, linewidth=0.5, zorder=0)
    ax.axvline(0.0, color=GRID, linewidth=0.5, zorder=0)
    label_y = 0.93 if label_position == "top" else 0.07
    label_va = "top" if label_position == "top" else "bottom"
    ax.text(
        0.06,
        label_y,
        label,
        transform=ax.transAxes,
        ha="left",
        va=label_va,
        fontsize=9.5,
        color=label_color,
        bbox={
            "facecolor": background_color,
            "edgecolor": "none",
            "pad": 0.8,
        },
        zorder=6,
    )


def reduction_history_data(F=HISTORY_F):
    """Return both reduction histories with their determinant sheet at each step."""
    F = np.asarray(F, dtype=float)
    if F.shape != (2, 2):
        raise ValueError("F must have shape (2, 2)")
    if not np.all(np.isfinite(F)) or abs(np.linalg.det(F)) <= 1e-12:
        raise ValueError("F must be finite and invertible")

    C = F.T @ F
    initial_determinant = int(np.sign(np.linalg.det(F)))
    lagrange_history, lagrange_transforms = lagrange_reduction_history(
        C,
        return_transforms=True,
    )
    lagrange_determinants = [initial_determinant]
    determinant = initial_determinant
    for transform in lagrange_transforms:
        determinant *= int(round(np.linalg.det(transform)))
        lagrange_determinants.append(determinant)

    plastic_history = plastic_reduction_history(C)
    plastic_determinants = [initial_determinant] * len(plastic_history)
    return {
        "lagrange": {
            "history": lagrange_history,
            "determinants": tuple(lagrange_determinants),
            "transforms": tuple(lagrange_transforms),
        },
        "elastic": {
            "history": plastic_history,
            "determinants": tuple(plastic_determinants),
            "transforms": (),
        },
    }


def _history_point(C, determinant):
    x, y = C2PoincareDisk(C)
    x_offset, y_offset = _disk_origin(determinant)
    return np.array(
        [
            x * DISK_RESOLUTION / 2
            + x_offset
            + DISK_RESOLUTION / 2,
            y * DISK_RESOLUTION / 2
            + y_offset
            + DISK_RESOLUTION / 2,
        ]
    )


def _draw_history_arrow(ax, start, end, color, linewidth=2.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=9,
        path_effects=[
            patheffects.Stroke(linewidth=linewidth + 2.4, foreground="white"),
            patheffects.Normal(),
        ],
    )
    ax.add_patch(arrow)


def _transform_label(transform):
    matrix = np.asarray(transform, dtype=float)
    candidates = (
        (np.array([[1, 0], [0, -1]], dtype=float), r"$\mathbf{m}_1$"),
        (np.array([[0, 1], [1, 0]], dtype=float), r"$\mathbf{m}_2$"),
        (np.array([[1, -1], [0, 1]], dtype=float), r"$\mathbf{m}_3$"),
    )
    for candidate, label in candidates:
        if np.array_equal(matrix, candidate):
            return label
    return None


def _elastic_transform_label(history, step):
    """Name the unit shear used by one elastic-reduction history step."""
    before = np.asarray(history[step], dtype=float)
    after = np.asarray(history[step + 1], dtype=float)
    a, b, c = before[0, 0], before[1, 1], before[0, 1]
    if a <= b:
        n = int(round((after[0, 1] - c) / a))
        return rf"$\mathbf{{E}}_{{12}}({n})$"
    n = int(round((after[0, 1] - c) / b))
    return rf"$\mathbf{{E}}_{{21}}({n})$"


def _draw_history_paths(ax, F=HISTORY_F):
    """Overlay Lagrange and elastic histories on the two determinant disks."""
    histories = reduction_history_data(F)
    initial_point = None
    for name, color in (
        ("lagrange", LAGRANGE_PATH_COLOR),
        ("elastic", ELASTIC_PATH_COLOR),
    ):
        entry = histories[name]
        points = [
            _history_point(C, determinant)
            for C, determinant in zip(
                entry["history"], entry["determinants"]
            )
        ]
        if initial_point is None:
            initial_point = points[0]
        for step, (start, end) in enumerate(zip(points[:-1], points[1:])):
            _draw_history_arrow(ax, start, end, color)
            if name == "lagrange":
                label = _transform_label(entry["transforms"][step])
            else:
                label = _elastic_transform_label(entry["history"], step)
            if label is not None:
                crosses_disks = (
                    entry["determinants"][step]
                    != entry["determinants"][step + 1]
                )
                fraction = 0.30 if crosses_disks else 0.50
                midpoint = start + fraction * (end - start)
                ax.annotate(
                    label,
                    xy=midpoint,
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10.0,
                    color=TEXT,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.82,
                        "pad": 0.7,
                    },
                    zorder=11,
                )
        ax.scatter(
            np.asarray(points)[:, 0],
            np.asarray(points)[:, 1],
            s=30,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=10,
        )
        ax.scatter(
            [points[0][0]],
            [points[0][1]],
            s=46,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=11,
        )

    ax.scatter(
        [initial_point[0]],
        [initial_point[1]],
        s=20,
        facecolor=INITIAL_MARKER_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=13,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=LAGRANGE_PATH_COLOR,
            linewidth=2.0,
            label="Lagrange reduction",
        ),
        Line2D(
            [0],
            [0],
            color=ELASTIC_PATH_COLOR,
            linewidth=2.0,
            label="Plastic reduction",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6.2,
            markerfacecolor=INITIAL_MARKER_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=r"Initial $\mathbf F$",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.955),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#9AA3AA",
        framealpha=1.0,
        fontsize=8.0,
        handlelength=2.0,
        columnspacing=1.2,
    )
    legend.set_zorder(20)
    return histories


def make_figure(data=None, history_F=HISTORY_F):
    """Create two determinant disks with reduction histories above eight factors."""
    if data is None:
        data = decomposition_table_data()

    fig = plt.figure(figsize=(7.45, 5.88))
    master = fig.add_gridspec(
        1,
        3,
        width_ratios=(2.0, 1.0, 1.0),
        left=0.015,
        right=0.995,
        bottom=0.025,
        top=0.975,
        wspace=0.10,
    )

    quadrant_grid = generate_elastic_quadrant_grid(
        resolution=DISK_RESOLUTION,
    )
    disk_ax = fig.add_subplot(master[0, 0])
    for determinant in (-1, 1):
        _draw_quadrant_disk(disk_ax, quadrant_grid, determinant)
    _draw_history_paths(disk_ax, history_F)
    disk_ax.set_xlim(-18, DISK_RESOLUTION + 18)
    disk_ax.set_ylim(-18, 2 * DISK_RESOLUTION + DISK_GAP + 18)
    disk_ax.set_aspect("equal", adjustable="box")
    disk_ax.set_xticks([])
    disk_ax.set_yticks([])
    disk_ax.set_frame_on(False)

    table_grid = master[0, 1:].subgridspec(
        4,
        2,
        wspace=0.06,
        hspace=0.12,
    )
    titles = []
    for index, decomposition in enumerate(data):
        row, column = divmod(index, 2)
        pair = table_grid[row, column].subgridspec(1, 2, wspace=0.015)
        elastic_ax = fig.add_subplot(pair[0, 0])
        plastic_ax = fig.add_subplot(pair[0, 1])
        background_color = panel_background_color(decomposition)
        label_position = "top" if index in (1, 4) else "bottom"
        _draw_factor(
            elastic_ax,
            decomposition["F_e"],
            r"$\mathbf{F}_e$",
            background_color,
            label_position=label_position,
        )
        _draw_factor(
            plastic_ax,
            decomposition["F_p"],
            r"$\mathbf{F}_p$",
            background_color,
            label_position=label_position,
        )
        titles.append((elastic_ax, plastic_ax, decomposition["expression"]))

    fig.canvas.draw()
    m_pair = titles[0]
    mr_pair = titles[1]
    ms_pair = titles[4]
    mrs_pair = titles[5]

    def _between_title_positions(left_pair, right_pair):
        left_box = left_pair[0].get_position(), left_pair[1].get_position()
        right_box = right_pair[0].get_position(), right_pair[1].get_position()
        center = (left_box[0].x0 + right_box[1].x1) / 2
        y = max(
            left_box[0].y1,
            left_box[1].y1,
            right_box[0].y1,
            right_box[1].y1,
        ) + 0.009
        return center, y

    r_x, r_y = _between_title_positions(m_pair, mr_pair)
    fig.text(
        r_x,
        r_y,
        r"$\mathbf{r}=\mathbf{m}_1\mathbf{m}_2$",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=TEXT,
    )
    s_x, s_y = _between_title_positions(ms_pair, mrs_pair)
    fig.text(
        s_x,
        s_y,
        r"$\mathbf{s}=\mathbf{m}_1$",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=TEXT,
    )
    for elastic_ax, plastic_ax, expression in titles:
        elastic_box = elastic_ax.get_position()
        plastic_box = plastic_ax.get_position()
        fig.text(
            (elastic_box.x0 + plastic_box.x1) / 2,
            max(elastic_box.y1, plastic_box.y1) + 0.009,
            expression,
            ha="center",
            va="bottom",
            fontsize=10.2,
            color=TEXT,
        )
    return fig


def main():
    apply_style()
    data = decomposition_table_data()
    fig = make_figure(data)
    OUT.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_STEM.with_suffix(".png")
    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


if __name__ == "__main__":
    main()

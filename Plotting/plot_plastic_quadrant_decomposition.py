"""Compact figures for the two plastic-reduction decompositions."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
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
from MTMath.reduction import plastic_reduction_history
from Plotting.matrix_visualization import draw_matrix_columns
from Plotting.plasticReductionCorrectnessIllustration import (
    COLUMN_COLORS,
    GRID as FACTOR_GRID,
    TEXT as FACTOR_TEXT,
    apply_style,
)


OUT = ROOT / "Plots" / "plastic_reduction"

BLUE = "#2171B5"
ORANGE = "#E67E22"
SHORT_PATH_COLOR = "#2ca02c"
LONG_PATH_COLOR = "#d62728"
OQ_PATH_COLOR = "#5B9BD5"
NEUTRAL = "#59636E"
GRID = FACTOR_GRID
TEXT = FACTOR_TEXT
SHORT_FACTOR_BACKGROUND = "#DDF2E0"
LONG_FACTOR_BACKGROUND = "#F7DDDD"
OQ_FACTOR_BACKGROUND = "#DCEBF7"
ORIGINAL_FACTOR_BACKGROUND = "#F3F5F6"
MAX_ALTERNATIVE_DEPTH = 15


def _style():
    apply_style()
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
    short_path, long_path, paths = _shortest_reduction_and_alternative(
        total_F.T @ total_F,
    )

    return total_F, _representatives_from_paths(total_F, (short_path, long_path)), paths


def _representatives_from_paths(total_F, selected_paths):
    representatives = []
    for path in selected_paths:
        M = np.asarray(path["M"], dtype=float)
        representatives.append(
            {
                "path": path["path"],
                "M": M,
                "F_e": total_F @ M,
                "F_p": np.linalg.inv(M),
            }
        )
    return representatives


MOVE_MATRICES = {
    "U+": np.array([[1.0, 1.0], [0.0, 1.0]]),
    "U-": np.array([[1.0, -1.0], [0.0, 1.0]]),
    "L+": np.array([[1.0, 0.0], [1.0, 1.0]]),
    "L-": np.array([[1.0, 0.0], [-1.0, 1.0]]),
}

QUARTER_TURNS = (
    np.array([[0, -1], [1, 0]], dtype=int),
    np.array([[0, 1], [-1, 0]], dtype=int),
)


def _move_labels_from_history(history):
    labels = []
    for current_C, next_C in zip(history[:-1], history[1:]):
        matching_labels = [
            label
            for label, move in MOVE_MATRICES.items()
            if np.allclose(
                move.T @ current_C @ move,
                next_C,
                atol=1e-12,
                rtol=1e-12,
            )
        ]
        if len(matching_labels) != 1:
            raise RuntimeError(
                "Could not uniquely identify a plastic-reduction unit shear."
            )
        labels.append(matching_labels[0])
    return tuple(labels)


def _shortest_reduction_and_alternative(C0, max_depth=MAX_ALTERNATIVE_DEPTH):
    """Return PR and the shortest path to the same metric with a new lift."""
    C0 = np.asarray(C0, dtype=float)
    reduction_history, reduction_M = plastic_reduction_history(
        C0,
        return_M=True,
    )
    reduction_path = _move_labels_from_history(reduction_history)
    reduction_endpoint = reduction_history[-1]
    last_paths = []

    for search_depth in range(1, max_depth + 1):
        _candidate_Cs, paths = plasticReductionBFS(
            C0,
            max_depth=search_depth,
            plot=False,
            return_paths=True,
            allow_no_center=True,
        )
        last_paths = paths
        globally_shorter_paths = [
            path for path in paths if path["depth"] < len(reduction_path)
        ]
        if globally_shorter_paths:
            raise RuntimeError(
                "Plastic reduction was not a globally shortest path into the "
                "elastic domain."
            )
        endpoint_paths = [
            path
            for path in paths
            if np.allclose(
                path["C"],
                reduction_endpoint,
                atol=1e-12,
                rtol=1e-12,
            )
        ]
        reduction_matches = [
            path for path in endpoint_paths if path["path"] == reduction_path
        ]
        alternative_lifts = [
            path
            for path in endpoint_paths
            if not np.array_equal(
                np.asarray(path["M"], dtype=int),
                np.asarray(reduction_M, dtype=int),
            )
        ]
        if reduction_matches and alternative_lifts:
            reduction_result = reduction_matches[0]
            # The depth loop guarantees minimal length. Preserve BFS move order
            # as the deterministic tie-breaker when several lifts appear at
            # that same depth.
            alternative_result = alternative_lifts[0]
            if not np.array_equal(
                np.asarray(reduction_result["M"], dtype=int),
                np.asarray(reduction_M, dtype=int),
            ):
                raise RuntimeError(
                    "BFS and plastic reduction accumulated different transforms."
                )
            if not np.array_equal(
                np.asarray(alternative_result["M"], dtype=int),
                -np.asarray(reduction_M, dtype=int),
            ):
                raise RuntimeError(
                    "The alternative lift is not the expected 180-degree rotation."
                )
            return reduction_result, alternative_result, paths

    raise RuntimeError(
        "No path with a distinct lift reached the plastic-reduction endpoint by "
        f"depth {max_depth}. Last search returned {len(last_paths)} terminal paths."
    )


def _shortest_quarter_turn_match(
    C0,
    reduction_result,
    max_depth=MAX_ALTERNATIVE_DEPTH,
):
    """Return the shortest BFS lift differing from PR by a quarter turn."""
    reduction_M = np.asarray(reduction_result["M"], dtype=int)
    target_lifts = tuple(reduction_M @ turn for turn in QUARTER_TURNS)
    for search_depth in range(1, max_depth + 1):
        _candidate_Cs, paths = plasticReductionBFS(
            C0,
            max_depth=search_depth,
            plot=False,
            return_paths=True,
            allow_no_center=True,
        )
        for path in paths:
            path_M = np.asarray(path["M"], dtype=int)
            if any(np.array_equal(path_M, target) for target in target_lifts):
                return path, paths
    raise RuntimeError(
        "No quarter-turn match reached the elastic domain by "
        f"depth {max_depth}."
    )


def match2_decomposition_data(total_F=None):
    """Return PR plus its shortest quarter- and half-turn BFS matches."""
    if total_F is None:
        total_F = SShear(1.3, s_conponent=(0, 1))
    C0 = total_F.T @ total_F
    reduction_path, half_turn_path, paths = (
        _shortest_reduction_and_alternative(C0)
    )
    quarter_turn_path, _quarter_turn_paths = _shortest_quarter_turn_match(
        C0,
        reduction_path,
    )
    return (
        total_F,
        _representatives_from_paths(
            total_F,
            (reduction_path, quarter_turn_path, half_turn_path),
        ),
        paths,
    )


def same_length_decomposition_data(total_F, long_path):
    """Build the second comparison from two steps into the original Long path."""
    prefix_matrix = np.eye(2)
    for move in long_path["path"][:2]:
        prefix_matrix = prefix_matrix @ MOVE_MATRICES[move]
    intermediate_F = total_F @ prefix_matrix
    reduction_path, alternative_path, paths = _shortest_reduction_and_alternative(
        intermediate_F.T @ intermediate_F,
    )
    return (
        intermediate_F,
        _representatives_from_paths(
            intermediate_F,
            (reduction_path, alternative_path),
        ),
        paths,
    )


def _path_history(C0, path):
    """Return the metric after every unit shear in one BFS path."""
    M = np.eye(2)
    history = [np.asarray(C0, dtype=float)]
    for move in path:
        M = M @ MOVE_MATRICES[move]
        history.append(M.T @ C0 @ M)
    return np.stack(history)


def _draw_reduction_paths(
    ax,
    total_F,
    short,
    long,
    *,
    path_labels=("Short", "Long"),
):
    """Draw two representative BFS paths in a compact Poincare disk."""
    _draw_reduction_path_set(
        ax,
        total_F,
        (short, long),
        path_labels=path_labels,
        linestyles=("-", "--"),
    )


def _draw_reduction_path_set(
    ax,
    total_F,
    representatives,
    *,
    path_labels,
    linestyles,
    path_colors=None,
):
    """Draw PR and one or more rotationally related BFS paths."""
    if not (
        len(representatives) == len(path_labels) == len(linestyles)
    ):
        raise ValueError("Paths, labels, and linestyles must have equal length.")
    if path_colors is None:
        path_colors = (SHORT_PATH_COLOR,) + (LONG_PATH_COLOR,) * (
            len(representatives) - 1
        )
    if len(path_colors) != len(representatives):
        raise ValueError("Paths and path_colors must have equal length.")
    resolution = 420
    C0 = total_F.T @ total_F
    histories = [
        _path_history(C0, representative["path"])
        for representative in representatives
    ]

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

    for path_index, (history, linestyle) in enumerate(
        zip(histories, linestyles)
    ):
        color = path_colors[path_index]
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
        if path_index > 0 and np.allclose(
            history[-1],
            histories[0][-1],
            atol=1e-12,
            rtol=1e-12,
        ):
            ax.scatter(
                *points[-1],
                s=8,
                color=path_colors[0],
                edgecolor="none",
                zorder=8,
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
    path_handles = [
        Line2D(
            [0],
            [0],
            color=path_colors[index],
            linewidth=1.5,
            linestyle=linestyle,
            label=label,
        )
        for index, (label, linestyle) in enumerate(
            zip(path_labels, linestyles)
        )
    ]
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
                label="Initial",
            ),
            *path_handles,
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
    ax.set_title("BFS paths", color=TEXT, pad=3)


def _vector_limit(matrices):
    return max(
        1.0,
        max(
            float(np.linalg.norm(np.asarray(matrix, dtype=float), axis=0).max())
            for matrix in matrices
        ),
    )


def _draw_rotation_annotation(ax, degrees=180):
    """Mark a red panel as a rotation of its green counterpart."""
    ax.add_patch(
        FancyArrowPatch(
            (0.86, 0.15),
            (0.72, 0.13),
            transform=ax.transAxes,
            connectionstyle="arc3,rad=-0.9",
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.85,
            color=FACTOR_TEXT,
            shrinkA=0,
            shrinkB=0,
            zorder=7,
        )
    )
    ax.text(
        0.67,
        0.25,
        rf"${degrees}^\circ$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=FACTOR_TEXT,
        bbox={
            "facecolor": ax.get_facecolor(),
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 0.3,
        },
        zorder=8,
    )


def _draw_factor_panel(
    ax,
    matrix,
    *,
    label,
    background_color,
    vector_limit,
    reference_matrix=None,
    rotation_degrees=None,
    panel_limit=None,
):
    """Draw a factor pair panel using the determinant-quadrant visual style."""
    limit = vector_limit if panel_limit is None else panel_limit
    ax.set_facecolor(background_color)
    if reference_matrix is not None:
        reference_arrows = draw_matrix_columns(
            ax,
            reference_matrix,
            limits=(-limit, limit),
            colors=COLUMN_COLORS,
            linestyles=("-", "-"),
            linewidth=0.8,
            mutation_scale=6,
            grid_color=FACTOR_GRID,
            origin_color=background_color,
            show_ticks=False,
        )
        for arrow in reference_arrows:
            arrow.set_alpha(0.30)
            arrow.set_zorder(1)
    arrows = draw_matrix_columns(
        ax,
        matrix,
        limits=(-limit, limit),
        colors=COLUMN_COLORS,
        linestyles=("-", "-"),
        linewidth=2.15,
        mutation_scale=12,
        grid_color=FACTOR_GRID,
        origin_color=FACTOR_TEXT,
        show_ticks=False,
    )
    for arrow in arrows:
        arrow.set_path_effects(
            [
                patheffects.Stroke(linewidth=4.7, foreground="white"),
                patheffects.Normal(),
            ]
    )
    ax.axhline(0.0, color=FACTOR_GRID, linewidth=0.55, zorder=0)
    ax.axvline(0.0, color=FACTOR_GRID, linewidth=0.55, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    if rotation_degrees is not None:
        _draw_rotation_annotation(ax, rotation_degrees)
    ax.text(
        0.06,
        0.93,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=FACTOR_TEXT,
        bbox={
            "facecolor": background_color,
            "edgecolor": "none",
            "pad": 0.8,
        },
        zorder=6,
    )


def plot_decompositions(
    total_F,
    representatives,
    *,
    path_labels=("Short", "Long"),
    path_titles=None,
):
    if path_titles is None:
        path_titles = path_labels
    short, long = representatives
    if len(short["path"]) > len(long["path"]):
        raise ValueError("The first representative must be the shortest path.")
    factor_matrices = (
        short["F_p"],
        long["F_p"],
        total_F,
        short["F_e"],
        long["F_e"],
    )
    vector_limit = _vector_limit(factor_matrices)
    fig = plt.figure(figsize=(6.2, 2.35))
    master = fig.add_gridspec(
        1,
        6,
        width_ratios=(1.50, 0.85, 0.85, 0.85, 0.85, 0.85),
        left=0.02,
        right=0.99,
        bottom=0.06,
        top=0.82,
        wspace=0.035,
    )

    disk_ax = fig.add_subplot(master[0, 0])
    original_ax = fig.add_subplot(master[0, 1])
    short_elastic_ax = fig.add_subplot(master[0, 2])
    short_plastic_ax = fig.add_subplot(master[0, 3])
    long_elastic_ax = fig.add_subplot(master[0, 4])
    long_plastic_ax = fig.add_subplot(master[0, 5])

    _draw_reduction_paths(
        disk_ax,
        total_F,
        short,
        long,
        path_labels=path_labels,
    )
    _draw_factor_panel(
        original_ax,
        total_F,
        label=r"$\mathbf{F}$",
        background_color=ORIGINAL_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
    )
    _draw_factor_panel(
        short_elastic_ax,
        short["F_e"],
        label=r"$\mathbf{F}_e$",
        background_color=SHORT_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
        panel_limit=1.0,
    )
    _draw_factor_panel(
        short_plastic_ax,
        short["F_p"],
        label=r"$\mathbf{F}_p$",
        background_color=SHORT_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
    )
    _draw_factor_panel(
        long_elastic_ax,
        long["F_e"],
        label=r"$\mathbf{F}_e$",
        background_color=LONG_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
        reference_matrix=short["F_e"],
        rotation_degrees=180,
        panel_limit=1.0,
    )
    _draw_factor_panel(
        long_plastic_ax,
        long["F_p"],
        label=r"$\mathbf{F}_p$",
        background_color=LONG_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
        reference_matrix=short["F_p"],
        rotation_degrees=180,
    )

    fig.canvas.draw()
    title_axes = [
        (original_ax, "Initial"),
    ]
    for left_ax, right_ax, title in (
        (short_elastic_ax, short_plastic_ax, path_titles[0]),
        (long_elastic_ax, long_plastic_ax, path_titles[1]),
    ):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        title_axes.append(
            (
                (left_box.x0 + right_box.x1) / 2.0,
                max(left_box.y1, right_box.y1) + 0.014,
                title,
            )
        )
    original_box = original_ax.get_position()
    title_axes[0] = (
        (original_box.x0 + original_box.x1) / 2.0,
        original_box.y1 + 0.014,
        title_axes[0][1],
    )
    for x, y, title in title_axes:
        fig.text(
            x,
            y,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=FACTOR_TEXT,
        )
    return fig


def plot_decompositions_match2(
    total_F,
    representatives,
    *,
    path_labels=("Short", "OQ", "Long"),
    path_titles=("Short", "Other quadrant (OQ)", "Long"),
    path_colors=(SHORT_PATH_COLOR, OQ_PATH_COLOR, LONG_PATH_COLOR),
    factor_backgrounds=(
        SHORT_FACTOR_BACKGROUND,
        OQ_FACTOR_BACKGROUND,
        LONG_FACTOR_BACKGROUND,
    ),
):
    """Plot PR beside its shortest 90- and 180-degree rotational matches."""
    short, match_90, match_180 = representatives
    factor_matrices = (
        total_F,
        short["F_e"],
        short["F_p"],
        match_90["F_e"],
        match_90["F_p"],
        match_180["F_e"],
        match_180["F_p"],
    )
    vector_limit = _vector_limit(factor_matrices)
    fig = plt.figure(figsize=(6.2, 2.7))
    master = fig.add_gridspec(
        2,
        6,
        width_ratios=(1.50, 0.85, 0.85, 0.85, 0.68, 0.68),
        left=0.02,
        right=0.99,
        bottom=0.05,
        top=0.92,
        wspace=0.035,
        hspace=0.24,
    )

    disk_ax = fig.add_subplot(master[:, 0])
    original_ax = fig.add_subplot(master[:, 1])
    short_elastic_ax = fig.add_subplot(master[:, 2])
    short_plastic_ax = fig.add_subplot(master[:, 3])
    match_90_elastic_ax = fig.add_subplot(master[0, 4])
    match_90_plastic_ax = fig.add_subplot(master[0, 5])
    match_180_elastic_ax = fig.add_subplot(master[1, 4])
    match_180_plastic_ax = fig.add_subplot(master[1, 5])

    _draw_reduction_path_set(
        disk_ax,
        total_F,
        representatives,
        path_labels=path_labels,
        linestyles=("-", "-", "--"),
        path_colors=path_colors,
    )
    _draw_factor_panel(
        original_ax,
        total_F,
        label=r"$\mathbf{F}$",
        background_color=ORIGINAL_FACTOR_BACKGROUND,
        vector_limit=vector_limit,
    )
    _draw_factor_panel(
        short_elastic_ax,
        short["F_e"],
        label=r"$\mathbf{F}_e$",
        background_color=factor_backgrounds[0],
        vector_limit=vector_limit,
        panel_limit=1.0,
    )
    _draw_factor_panel(
        short_plastic_ax,
        short["F_p"],
        label=r"$\mathbf{F}_p$",
        background_color=factor_backgrounds[0],
        vector_limit=vector_limit,
    )
    for axes, representative, degrees, background_color in (
        (
            (match_90_elastic_ax, match_90_plastic_ax),
            match_90,
            90,
            factor_backgrounds[1],
        ),
        (
            (match_180_elastic_ax, match_180_plastic_ax),
            match_180,
            180,
            factor_backgrounds[2],
        ),
    ):
        for ax, key, label, reference in (
            (axes[0], "F_e", r"$\mathbf{F}_e$", short["F_e"]),
            (axes[1], "F_p", r"$\mathbf{F}_p$", short["F_p"]),
        ):
            _draw_factor_panel(
                ax,
                representative[key],
                label=label,
                background_color=background_color,
                vector_limit=vector_limit,
                reference_matrix=reference,
                rotation_degrees=degrees,
                panel_limit=1.0 if key == "F_e" else None,
            )

    fig.canvas.draw()
    title_positions = []
    for ax, title in ((original_ax, "Initial"),):
        box = ax.get_position()
        title_positions.append(
            ((box.x0 + box.x1) / 2.0, box.y1 + 0.012, title)
        )
    for left_ax, right_ax, title, offset in (
        (short_elastic_ax, short_plastic_ax, path_titles[0], 0.012),
        (match_90_elastic_ax, match_90_plastic_ax, path_titles[1], 0.008),
        (match_180_elastic_ax, match_180_plastic_ax, path_titles[2], 0.008),
    ):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        title_positions.append(
            (
                (left_box.x0 + right_box.x1) / 2.0,
                max(left_box.y1, right_box.y1) + offset,
                title,
            )
        )
    for x, y, title in title_positions:
        fig.text(
            x,
            y,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            color=FACTOR_TEXT,
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
    _, long_representative = representatives
    same_length_F, same_length_representatives, _ = (
        same_length_decomposition_data(total_F, long_representative)
    )
    match2_F, match2_representatives, _ = match2_decomposition_data()
    other_initial_match2_F, other_initial_match2_representatives, _ = (
        match2_decomposition_data(same_length_F)
    )

    figures = {
        "elastic_plastic_factors": plot_decompositions(total_F, representatives),
        "elastic_plastic_factors_match2": plot_decompositions_match2(
            match2_F,
            match2_representatives,
        ),
        "elastic_plastic_factors_match2_other_initial": plot_decompositions_match2(
            other_initial_match2_F,
            other_initial_match2_representatives,
        ),
        "same_length_path": plot_decompositions(
            same_length_F,
            same_length_representatives,
            path_labels=("Short", "Long"),
            path_titles=("Short", "Long"),
        ),
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

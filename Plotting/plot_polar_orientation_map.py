#!/usr/bin/env python3
"""Plot element-wise square-crystal orientations from the polar part of ``F`` or ``T``.

The mesh rendering is deliberately delegated to ``Plotting.pyplotFunctions``:
it is the same periodic, locally-unwrapped rendering used by the animation
plots.  Only the orientation field and the inset are specific to this figure.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Rectangle


SIMULATION_SCRIPTS = Path("/Users/eliaslundheim/work/PhD/SimulationScripts")
SIMULATION = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l0.15,1e-05,3.0PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
NO_RECONNECTION_SIMULATION = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l0.15,1e-05,2.0PBCt3epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
SQUARE_SYMMETRY_ANGLE = np.pi / 2.0
LOAD_PATTERN = re.compile(r"_load=([-+0-9.eE]+)_")


def closest_vtu(simulation: Path, target_gamma: float) -> tuple[Path, float]:
    """Return the saved VTU whose explicit load is nearest ``target_gamma``."""
    candidates = []
    for path in (simulation / "data").glob("*.vtu"):
        match = LOAD_PATTERN.search(path.name)
        if match is None:
            continue
        gamma = float(match.group(1))
        candidates.append((abs(gamma - target_gamma), gamma, path))
    if not candidates:
        raise FileNotFoundError(f"No load-labelled VTUs found below {simulation / 'data'}")
    _, gamma, path = min(candidates)
    return path, gamma


def square_crystal_orientation(matrices: np.ndarray, matrix_name: str) -> np.ndarray:
    """Return ``arg(R) mod pi/2`` for one cell-wise ``matrix_name = R U``."""
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{matrix_name} contains non-finite matrix entries.")

    left, _, right_transpose = np.linalg.svd(matrices)
    rotation = left @ right_transpose
    determinants = np.linalg.det(rotation)
    if np.any(determinants <= 0.0):
        raise ValueError(f"The polar factor of {matrix_name} contains a reflection.")

    angles = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return np.mod(angles, SQUARE_SYMMETRY_ANGLE)


def deformed_crystal_axis_orientation(
    matrices: np.ndarray, matrix_name: str, axis_index: int
) -> np.ndarray:
    """Return the lab angle of ``matrix_name e_i``, modulo square symmetry."""
    if not np.all(np.isfinite(matrices)):
        raise ValueError(f"{matrix_name} contains non-finite matrix entries.")
    if axis_index not in (0, 1):
        raise ValueError("axis_index must be 0 or 1.")
    crystal_axis = matrices[:, :, axis_index]
    if np.any(np.linalg.norm(crystal_axis, axis=1) <= np.finfo(float).eps):
        raise ValueError(f"{matrix_name} maps a crystal axis to a zero-length vector.")
    return np.mod(
        np.arctan2(crystal_axis[:, 1], crystal_axis[:, 0]), SQUARE_SYMMETRY_ANGLE
    )


def longest_edge_orientation(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    load: float | None = None,
    box_size: float | None = None,
    subtract_angle: float = 0.0,
) -> np.ndarray:
    """Return the lab angle of every triangle's longest periodic edge."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n_nodes, 2).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n_elements, 3).")
    vertices = points[triangles]
    edges = vertices[:, (1, 2, 0)] - vertices[:, (0, 1, 2)]
    if (load is None) != (box_size is None):
        raise ValueError("load and box_size must either both be set or both be None.")
    if not np.isfinite(subtract_angle):
        raise ValueError("subtract_angle must be finite.")
    if load is None:
        minimum_image_edges = edges
    else:
        periodic_images = np.array(
            [
                (x_copy * box_size + load * y_copy * box_size, y_copy * box_size)
                for x_copy in (-1, 0, 1)
                for y_copy in (-1, 0, 1)
            ]
        )
        candidate_edges = edges[:, :, None, :] - periodic_images[None, None, :, :]
        candidate_lengths_squared = np.sum(candidate_edges**2, axis=-1)
        nearest_image = np.argmin(candidate_lengths_squared, axis=2)
        minimum_image_edges = candidate_edges[
            np.arange(len(triangles))[:, None],
            np.arange(3)[None, :],
            nearest_image,
        ]
    lengths_squared = np.sum(minimum_image_edges**2, axis=-1)
    longest_edge = np.argmax(lengths_squared, axis=1)
    edge_vectors = minimum_image_edges[np.arange(len(triangles)), longest_edge]
    if np.any(np.linalg.norm(edge_vectors, axis=1) <= np.finfo(float).eps):
        raise ValueError("An element has no non-zero periodic edge.")
    return np.mod(
        np.arctan2(edge_vectors[:, 1], edge_vectors[:, 0]) - subtract_angle,
        SQUARE_SYMMETRY_ANGLE,
    )


def three_by_three_periodic_nodes(
    points: np.ndarray, load: float, box_size: float
) -> np.ndarray:
    """Tile the unchanged nodes and merge only coincident periodic copies."""
    shifts = np.array(
        [
            (x_copy * box_size + load * y_copy * box_size, y_copy * box_size)
            for x_copy in (-1, 0, 1)
            for y_copy in (-1, 0, 1)
        ]
    )
    tiled_points = (points[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    keys = np.rint(tiled_points / 1.0e-10).astype(np.int64)
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return tiled_points[np.sort(first_indices)]


def triangles_intersecting_viewport(
    points: np.ndarray, triangles: np.ndarray, viewport: tuple[float, float, float, float]
) -> np.ndarray:
    """Return triangles that intersect the closed Cartesian viewport."""
    xmin, xmax, ymin, ymax = viewport
    vertices = points[triangles]
    visible = (
        (vertices[:, :, 0].max(axis=1) >= xmin)
        & (vertices[:, :, 0].min(axis=1) <= xmax)
        & (vertices[:, :, 1].max(axis=1) >= ymin)
        & (vertices[:, :, 1].min(axis=1) <= ymax)
    )
    if not np.any(visible):
        raise RuntimeError("No triangles intersect the requested viewport.")
    return triangles[visible]


def draw_longest_edge_field(
    axis,
    data,
    points: np.ndarray,
    triangles: np.ndarray,
    orientation: np.ndarray,
    *,
    cmap,
    norm,
    tile_periodically: bool,
) -> None:
    """Draw one flat-shaded longest-edge field on an existing axis."""
    if tile_periodically:
        draw_tiled_orientation(
            axis,
            data,
            triangles,
            orientation,
            cmap=cmap,
            norm=norm,
            show_nodes=False,
        )
    else:
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        axis.tripcolor(
            triangulation,
            facecolors=orientation,
            shading="flat",
            cmap=cmap,
            norm=norm,
            edgecolors="none",
        )


def draw_comparison_orientation_field(
    axis,
    data,
    points: np.ndarray,
    triangles: np.ndarray,
    orientation: np.ndarray,
    *,
    cmap,
    norm,
    tile_periodically: bool,
    locally_unwrapped: bool,
    show_nodes: bool = False,
) -> None:
    """Draw a comparison field, preserving element values across periodic seams."""
    if not locally_unwrapped:
        draw_longest_edge_field(
            axis,
            data,
            points,
            triangles,
            orientation,
            cmap=cmap,
            norm=norm,
            tile_periodically=tile_periodically,
        )
        return
    if not tile_periodically:
        raise ValueError("Locally unwrapped fields require periodic tiling.")
    if orientation.shape != (len(triangles),):
        raise ValueError("One orientation is required for every triangle.")

    from Plotting.pyplotFunctions import _plot_periodic_unwrapped_mesh_elements

    _plot_periodic_unwrapped_mesh_elements(
        axis,
        points,
        triangles,
        orientation,
        norm,
        cmap,
        data,
        "none",
        cull_to_view=True,
    )
    if show_nodes:
        draw_periodic_nodes(axis, data, points)


def save_longest_edge_map(
    data,
    gamma: float,
    points: np.ndarray,
    triangles: np.ndarray,
    orientation: np.ndarray,
    output: Path,
    *,
    tile_periodically: bool,
) -> None:
    """Save one seam-free longest-edge map using the supplied connectivity."""
    from Plotting.pyplotFunctions import draw_periodic_shear_box

    box_size = float(data.size[0])
    viewport = (0.0, box_size * (1.0 + gamma), 0.0, box_size)
    cmap = plt.get_cmap("twilight_shifted")
    norm = Normalize(vmin=0.0, vmax=SQUARE_SYMMETRY_ANGLE)
    figure = plt.figure(figsize=(9.0, 5.6), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(24, 1), wspace=0.06)
    axis = figure.add_subplot(grid[0])
    color_axis = figure.add_subplot(grid[1])
    axis.set_xlim(*viewport[:2])
    axis.set_ylim(*viewport[2:])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    draw_longest_edge_field(
        axis,
        data,
        points,
        triangles,
        orientation,
        cmap=cmap,
        norm=norm,
        tile_periodically=tile_periodically,
    )
    draw_periodic_shear_box(axis, origin=(0.0, 0.0), load=gamma, box_size=box_size)
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.set_ticks((0.0, np.pi / 4.0, np.pi / 2.0))
    colorbar.set_ticklabels((r"$0^\circ$", r"$45^\circ$", r"$90^\circ$"))
    colorbar.set_label(r"$\theta(\mathrm{longest})\bmod 90^\circ$")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600)
    plt.close(figure)


def draw_longest_edge_delaunay_comparison(
    vtu_path: Path, gamma: float, before_output: Path, after_output: Path
) -> None:
    """Save stored and Delaunay connectivity maps without moving any node."""
    if not SIMULATION_SCRIPTS.is_dir():
        raise FileNotFoundError(f"SimulationScripts directory not found: {SIMULATION_SCRIPTS}")
    sys.path.insert(0, str(SIMULATION_SCRIPTS))
    from Plotting.dataFunctions import VTUData
    from scipy.spatial import Delaunay

    data = VTUData(vtu_path)
    points = np.asarray(data.get_nodes()[:, :2], dtype=float)
    stored_triangles = np.asarray(data.get_connectivity(), dtype=int)
    box_size = float(data.size[0])
    if data.size[0] != data.size[1]:
        raise ValueError("This comparison requires a square periodic cell.")
    viewport = (0.0, box_size * (1.0 + gamma), 0.0, box_size)

    stored_orientation = longest_edge_orientation(
        points, stored_triangles, load=data.load, box_size=box_size
    )
    tiled_points = three_by_three_periodic_nodes(points, data.load, box_size)
    delaunay_triangles = Delaunay(tiled_points).simplices
    visible_delaunay_triangles = triangles_intersecting_viewport(
        tiled_points, delaunay_triangles, viewport
    )
    delaunay_orientation = longest_edge_orientation(
        tiled_points, visible_delaunay_triangles
    )

    save_longest_edge_map(
        data,
        gamma,
        points,
        stored_triangles,
        stored_orientation,
        before_output,
        tile_periodically=True,
    )
    save_longest_edge_map(
        data,
        gamma,
        tiled_points,
        visible_delaunay_triangles,
        delaunay_orientation,
        after_output,
        tile_periodically=False,
    )


def draw_reconnection_comparison(
    no_reconnection_vtu: Path,
    no_reconnection_gamma: float,
    reconnecting_vtu: Path,
    reconnecting_gamma: float,
    output: Path,
    *,
    target_gamma: float,
    orientation: str = "longest-edge",
) -> None:
    """Draw a shared orientation comparison with and without reconnection."""
    if not SIMULATION_SCRIPTS.is_dir():
        raise FileNotFoundError(f"SimulationScripts directory not found: {SIMULATION_SCRIPTS}")
    sys.path.insert(0, str(SIMULATION_SCRIPTS))
    from Plotting.dataFunctions import VTUData
    from Plotting.pyplotFunctions import draw_periodic_shear_box
    from scipy.spatial import Delaunay

    no_reconnection_data = VTUData(no_reconnection_vtu)
    reconnecting_data = VTUData(reconnecting_vtu)
    no_reconnection_points = np.asarray(no_reconnection_data.get_nodes()[:, :2], dtype=float)
    reconnecting_points = np.asarray(reconnecting_data.get_nodes()[:, :2], dtype=float)
    no_reconnection_stored = np.asarray(no_reconnection_data.get_connectivity(), dtype=int)
    reconnecting_triangles = np.asarray(reconnecting_data.get_connectivity(), dtype=int)
    box_size = float(no_reconnection_data.size[0])
    if no_reconnection_data.size != reconnecting_data.size:
        raise ValueError("The comparison requires equal square cell sizes.")
    if no_reconnection_data.size[0] != no_reconnection_data.size[1]:
        raise ValueError("The comparison requires a square periodic cell.")
    viewport = (0.0, box_size * (1.0 + target_gamma), 0.0, box_size)

    if orientation == "longest-edge":
        no_reconnection_tiled = three_by_three_periodic_nodes(
            no_reconnection_points, no_reconnection_data.load, box_size
        )
        no_reconnection_triangles = triangles_intersecting_viewport(
            no_reconnection_tiled,
            Delaunay(no_reconnection_tiled).simplices,
            viewport,
        )
        no_reconnection_orientation = longest_edge_orientation(
            no_reconnection_tiled,
            no_reconnection_triangles,
            subtract_angle=np.pi / 4.0,
        )
        reconnecting_orientation = longest_edge_orientation(
            reconnecting_points,
            reconnecting_triangles,
            load=reconnecting_data.load,
            box_size=box_size,
            subtract_angle=np.pi / 4.0,
        )
        no_reconnection_plot_points = no_reconnection_tiled
        no_reconnection_plot_triangles = no_reconnection_triangles
        no_reconnection_tile_periodically = False
        reconnecting_tile_periodically = True
        locally_unwrapped = False
        orientation_label = (
            r"$[\theta(\mathrm{longest})-\pi/4]\bmod \pi/2$"
        )
    elif orientation == "reduced-polar":
        no_reconnection_orientation = square_crystal_orientation(
            no_reconnection_data.get_F_e(), "F_e"
        )
        reconnecting_orientation = square_crystal_orientation(
            reconnecting_data.get_F_e(), "F_e"
        )
        no_reconnection_plot_points = no_reconnection_points
        no_reconnection_plot_triangles = no_reconnection_stored
        no_reconnection_tile_periodically = True
        reconnecting_tile_periodically = True
        locally_unwrapped = True
        orientation_label = (
            r"$\theta_e=\arg(\mathbf{R}_e)"
            r"\bmod \pi/2$"
        )
    else:
        raise ValueError(f"Unsupported comparison orientation: {orientation}")

    cmap = plt.get_cmap("twilight_shifted")
    norm = Normalize(vmin=0.0, vmax=SQUARE_SYMMETRY_ANGLE)
    figure, (left_axis, right_axis) = plt.subplots(
        1, 2, figsize=(11.0, 4.3), constrained_layout=False
    )
    figure.subplots_adjust(left=0.0, right=0.995, bottom=0.20, top=0.91, wspace=0.02)
    color_axis = figure.add_axes((0.17, 0.045, 0.66, 0.055))
    for axis in (left_axis, right_axis):
        axis.set_xlim(*viewport[:2])
        axis.set_ylim(*viewport[2:])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    left_axis.set_title("(a) No reconnection")
    right_axis.set_title("(b) With reconnection")

    draw_comparison_orientation_field(
        left_axis,
        no_reconnection_data,
        no_reconnection_plot_points,
        no_reconnection_plot_triangles,
        no_reconnection_orientation,
        cmap=cmap,
        norm=norm,
        tile_periodically=no_reconnection_tile_periodically,
        locally_unwrapped=locally_unwrapped,
    )
    draw_comparison_orientation_field(
        right_axis,
        reconnecting_data,
        reconnecting_points,
        reconnecting_triangles,
        reconnecting_orientation,
        cmap=cmap,
        norm=norm,
        tile_periodically=reconnecting_tile_periodically,
        locally_unwrapped=locally_unwrapped,
    )
    draw_periodic_shear_box(
        left_axis,
        origin=(0.0, 0.0),
        load=no_reconnection_gamma,
        box_size=box_size,
    )
    draw_periodic_shear_box(
        right_axis,
        origin=(0.0, 0.0),
        load=reconnecting_gamma,
        box_size=box_size,
    )
    add_mesh_insets(
        left_axis,
        no_reconnection_data,
        no_reconnection_plot_points,
        no_reconnection_plot_triangles,
        no_reconnection_orientation,
        box_size=box_size,
        gamma=target_gamma,
        cmap=cmap,
        norm=norm,
        tile_periodically=no_reconnection_tile_periodically,
        locally_unwrapped=locally_unwrapped,
        zoom_x_shifts=(0.02, -0.01),
        bottom_right_local_center=(0.5, 0.32),
    )
    add_mesh_insets(
        right_axis,
        reconnecting_data,
        reconnecting_points,
        reconnecting_triangles,
        reconnecting_orientation,
        box_size=box_size,
        gamma=target_gamma,
        cmap=cmap,
        norm=norm,
        tile_periodically=reconnecting_tile_periodically,
        locally_unwrapped=locally_unwrapped,
        bottom_right_local_center=(0.45, 0.07),
        bottom_right_zoom_half_width=12.0,
        bottom_right_inset_width=0.375,
    )

    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks((0.0, np.pi / 4.0, np.pi / 2.0))
    colorbar.set_ticklabels((r"$0$", r"$\pi/4$", r"$\pi/2$"))
    colorbar.set_label(orientation_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.0)
    plt.close(figure)


def get_matrix(data, name: str) -> np.ndarray:
    """Read a cell-wise 2-by-2 matrix using its explicit VTU component names."""
    components = data.get_matrix_components(name)
    return np.stack(
        (
            np.stack((components[(1, 1)], components[(1, 2)]), axis=-1),
            np.stack((components[(2, 1)], components[(2, 2)]), axis=-1),
        ),
        axis=-2,
    )


def draw_tiled_orientation(
    ax,
    data,
    triangles: np.ndarray,
    angles: np.ndarray,
    *,
    cmap,
    norm,
    show_nodes: bool,
) -> None:
    """Draw raw triangles in the centre cell and its eight periodic neighbours.

    ``tripcolor(..., facecolors=...)`` intentionally retains one orientation
    per element; no node averaging or interpolated shading is applied.
    """
    points = np.asarray(data.get_nodes()[:, :2], dtype=float)
    if points.shape != (len(points), 2):
        raise ValueError("points must have shape (n_nodes, 2).")
    if angles.shape != (len(triangles),):
        raise ValueError("One orientation is required for every triangle.")
    box_size = float(data.size[0])
    for x_copy in (-1, 0, 1):
        for y_copy in (-1, 0, 1):
            shift_y = y_copy * box_size
            shift_x = x_copy * box_size
            triangulation = mtri.Triangulation(
                points[:, 0] + shift_x + data.load * shift_y,
                points[:, 1] + shift_y,
                triangles,
            )
            ax.tripcolor(
                triangulation,
                facecolors=angles,
                shading="flat",
                cmap=cmap,
                norm=norm,
                # This is the large-mesh setting used by the animation plotter.
                # It leaves each element flat-coloured, but rasterises no edges
                # between neighbouring triangles.
                edgecolors="none",
            )
    if show_nodes:
        draw_periodic_nodes(ax, data, points)


def draw_periodic_nodes(axis, data, points: np.ndarray) -> None:
    """Draw white node markers over the visible periodic mesh copies."""
    box_size = float(data.size[0])
    for x_copy in (-1, 0, 1):
        for y_copy in (-1, 0, 1):
            shift_y = y_copy * box_size
            shift_x = x_copy * box_size
            axis.scatter(
                points[:, 0] + shift_x + data.load * shift_y,
                points[:, 1] + shift_y,
                s=3.0,
                c="white",
                edgecolors="none",
                zorder=11,
            )


def add_mesh_insets(
    axis,
    data,
    points: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    *,
    box_size: float,
    gamma: float,
    cmap,
    norm,
    tile_periodically: bool,
    locally_unwrapped: bool = False,
    zoom_x_shifts: tuple[float, float] = (0.0, 0.0),
    zoom_half_width: float = 8.0,
    bottom_right_local_center: tuple[float, float] = (0.5, 0.3),
    bottom_right_zoom_half_width: float | None = None,
    bottom_right_inset_width: float = 0.25,
) -> None:
    """Add the three mesh zooms and their connector lines."""
    viewport = (0.0, box_size * (1.0 + gamma), 0.0, box_size)
    if len(zoom_x_shifts) != 2 or not np.all(np.isfinite(zoom_x_shifts)):
        raise ValueError("zoom_x_shifts must contain two finite offsets.")
    if locally_unwrapped and not tile_periodically:
        raise ValueError("Locally unwrapped insets require periodic tiling.")
    if bottom_right_zoom_half_width is None:
        bottom_right_zoom_half_width = zoom_half_width
    if not np.isfinite(bottom_right_zoom_half_width) or bottom_right_zoom_half_width <= 0.0:
        raise ValueError("bottom_right_zoom_half_width must be positive and finite.")
    if not np.isfinite(bottom_right_inset_width) or not 0.0 < bottom_right_inset_width < 1.0:
        raise ValueError("bottom_right_inset_width must be between 0 and 1.")
    bottom_right_local_center = np.asarray(bottom_right_local_center, dtype=float)
    if bottom_right_local_center.shape != (2,) or not np.all(
        np.isfinite(bottom_right_local_center)
    ):
        raise ValueError("bottom_right_local_center must contain two finite coordinates.")
    upper_x = (0.5 + 0.05 + zoom_x_shifts[0]) * viewport[1]
    lower_x = (0.2 + zoom_x_shifts[1]) * viewport[1]
    zoom_center = (upper_x, 0.5 * box_size)
    zoom_viewports = [
        (
            zoom_center[0] - zoom_half_width,
            zoom_center[0] + zoom_half_width,
            zoom_center[1] - zoom_half_width,
            zoom_center[1] + zoom_half_width,
        ),
        (
            lower_x - zoom_half_width,
            lower_x + zoom_half_width,
            0.1 * box_size - zoom_half_width,
            0.1 * box_size + zoom_half_width,
        ),
        (
            bottom_right_local_center[0] * viewport[1]
            + zoom_half_width
            - 2.0 * bottom_right_zoom_half_width,
            bottom_right_local_center[0] * viewport[1] + zoom_half_width,
            bottom_right_local_center[1] * box_size - zoom_half_width,
            bottom_right_local_center[1] * box_size + zoom_half_width,
        ),
    ]
    inset_axes = [
        axis.inset_axes((0.71, 0.56, 0.25, 0.35)),
        axis.inset_axes((0.04, 0.56, 0.25, 0.35)),
        axis.inset_axes((0.96 - bottom_right_inset_width, 0.04, bottom_right_inset_width, 0.35)),
    ]
    for inset_axis, inset_viewport in zip(inset_axes, zoom_viewports, strict=True):
        inset_axis.set_zorder(30)
        inset_axis.patch.set_facecolor("white")
        inset_axis.patch.set_alpha(1.0)
        inset_axis.set_xlim(*inset_viewport[:2])
        inset_axis.set_ylim(*inset_viewport[2:])
        inset_axis.set_aspect("equal", adjustable="box")
        for spine in inset_axis.spines.values():
            spine.set_edgecolor("white")
        if locally_unwrapped:
            draw_comparison_orientation_field(
                inset_axis,
                data,
                points,
                triangles,
                angles,
                cmap=cmap,
                norm=norm,
                tile_periodically=tile_periodically,
                locally_unwrapped=True,
                show_nodes=True,
            )
        elif tile_periodically:
            draw_tiled_orientation(
                inset_axis,
                data,
                triangles,
                angles,
                cmap=cmap,
                norm=norm,
                show_nodes=True,
            )
        else:
            inset_axis.tripcolor(
                mtri.Triangulation(points[:, 0], points[:, 1], triangles),
                facecolors=angles,
                shading="flat",
                cmap=cmap,
                norm=norm,
                edgecolors="none",
            )
            inset_axis.scatter(
                points[:, 0], points[:, 1], s=3.0, c="white", edgecolors="none", zorder=11
            )
        inset_axis.set_xticks([])
        inset_axis.set_yticks([])

    for inset_viewport in zoom_viewports:
        axis.add_patch(
            Rectangle(
                (inset_viewport[0], inset_viewport[2]),
                inset_viewport[1] - inset_viewport[0],
                inset_viewport[3] - inset_viewport[2],
                fill=False,
                edgecolor="white",
                linewidth=0.8,
                zorder=25,
            )
        )
    for inset_corner, zoom_corner in zip(
        (
            (zoom_viewports[0][0], zoom_viewports[0][2]),
            (zoom_viewports[0][0], zoom_viewports[0][3]),
        ),
        (
            (zoom_viewports[0][1], zoom_viewports[0][2]),
            (zoom_viewports[0][1], zoom_viewports[0][3]),
        ),
        strict=True,
    ):
        axis.add_artist(
            ConnectionPatch(
                xyA=inset_corner,
                coordsA=inset_axes[0].transData,
                xyB=zoom_corner,
                coordsB=axis.transData,
                color="white",
                linewidth=0.6,
                zorder=26,
            )
        )
    for inset_corner, zoom_corner in zip(
        (
            (zoom_viewports[1][0], zoom_viewports[1][2]),
            (zoom_viewports[1][1], zoom_viewports[1][2]),
        ),
        (
            (zoom_viewports[1][0], zoom_viewports[1][3]),
            (zoom_viewports[1][1], zoom_viewports[1][3]),
        ),
        strict=True,
    ):
        axis.add_artist(
            ConnectionPatch(
                xyA=inset_corner,
                coordsA=inset_axes[1].transData,
                xyB=zoom_corner,
                coordsB=axis.transData,
                color="white",
                linewidth=0.6,
                zorder=26,
            )
        )

    for inset_corner, zoom_corner in zip(
        (
            (zoom_viewports[2][0], zoom_viewports[2][2]),
            (zoom_viewports[2][0], zoom_viewports[2][3]),
        ),
        (
            (zoom_viewports[2][1], zoom_viewports[2][2]),
            (zoom_viewports[2][1], zoom_viewports[2][3]),
        ),
        strict=True,
    ):
        axis.add_artist(
            ConnectionPatch(
                xyA=inset_corner,
                coordsA=inset_axes[2].transData,
                xyB=zoom_corner,
                coordsB=axis.transData,
                color="white",
                linewidth=0.6,
                zorder=26,
            )
        )


def draw_orientation_map(
    vtu_path: Path,
    gamma: float,
    output: Path,
    *,
    matrix_name: str,
    orientation_measure: str = "polar",
    zoom_half_width: float = 8.0,
    zoom_center: tuple[float, float] | None = None,
) -> None:
    """Draw one element-wise orientation map and a mesh-resolved inset."""
    if not SIMULATION_SCRIPTS.is_dir():
        raise FileNotFoundError(f"SimulationScripts directory not found: {SIMULATION_SCRIPTS}")
    sys.path.insert(0, str(SIMULATION_SCRIPTS))
    from Plotting.dataFunctions import VTUData
    from Plotting.pyplotFunctions import draw_periodic_shear_box

    data = VTUData(vtu_path)
    triangles = np.asarray(data.get_connectivity(), dtype=int)
    box_size = float(data.size[0])
    if data.size[0] != data.size[1]:
        raise ValueError("This square-crystal orientation plot requires a square cell.")
    points = np.asarray(data.get_nodes()[:, :2], dtype=float)
    if orientation_measure == "longest-edge":
        orientation = longest_edge_orientation(
            points, triangles, load=data.load, box_size=box_size
        )
        orientation_label = r"$\theta(\mathrm{longest})\bmod 90^\circ$"
    else:
        if matrix_name == "F":
            matrices = data.get_F()
            matrix_label = r"F"
        elif matrix_name == "T":
            # This is the total ``T`` from the manuscript: F_E F_P.  The legacy
            # VTU fields named T.. instead encode a branch-history quantity and
            # must not be substituted for F_P here.
            matrices = data.get_F_e() @ get_matrix(data, "F_P")
            matrix_label = r"T=F_E F_P"
        else:
            raise ValueError(f"Unsupported orientation matrix: {matrix_name}")
        if orientation_measure == "polar":
            orientation = square_crystal_orientation(matrices, matrix_name)
            orientation_label = rf"$\arg(R)\bmod 90^\circ$,  ${matrix_label}=RU$"
        elif orientation_measure == "crystal-axis":
            orientation = deformed_crystal_axis_orientation(matrices, matrix_name, 0)
            orientation_label = rf"$\arg({matrix_label}e_1)\bmod 90^\circ$"
        elif orientation_measure == "crystal-axis-2":
            orientation = deformed_crystal_axis_orientation(matrices, matrix_name, 1)
            orientation_label = rf"$\arg({matrix_label}e_2)\bmod 90^\circ$"
        else:
            raise ValueError(f"Unsupported orientation measure: {orientation_measure}")

    cmap = plt.get_cmap("twilight_shifted")
    norm = Normalize(vmin=0.0, vmax=SQUARE_SYMMETRY_ANGLE)
    viewport = (0.0, box_size * (1.0 + gamma), 0.0, box_size)
    if zoom_center is None:
        zoom_center = (
            (0.5 + 0.05) * box_size * (1.0 + gamma),
            0.5 * box_size,
        )
    zoom_center = np.asarray(zoom_center, dtype=float)
    if zoom_center.shape != (2,) or not np.all(np.isfinite(zoom_center)):
        raise ValueError("zoom_center must contain two finite laboratory coordinates.")
    zoom_viewport = (
        zoom_center[0] - zoom_half_width,
        zoom_center[0] + zoom_half_width,
        zoom_center[1] - zoom_half_width,
        zoom_center[1] + zoom_half_width,
    )
    lower_left_zoom_center = np.array((0.2 * viewport[1], 0.1 * box_size))
    lower_left_zoom_viewport = (
        lower_left_zoom_center[0] - zoom_half_width,
        lower_left_zoom_center[0] + zoom_half_width,
        lower_left_zoom_center[1] - zoom_half_width,
        lower_left_zoom_center[1] + zoom_half_width,
    )

    figure = plt.figure(figsize=(9.0, 5.6), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(24, 1), wspace=0.06)
    axis = figure.add_subplot(grid[0])
    color_axis = figure.add_subplot(grid[1])
    # Shifted right by 0.1 in axes coordinates.  The width is reduced just
    # enough to keep the inset fully inside the main panel.
    inset = axis.inset_axes((0.71, 0.56, 0.25, 0.35))
    lower_left_inset = axis.inset_axes((0.04, 0.56, 0.25, 0.35))
    axis.set_xlim(*viewport[:2])
    axis.set_ylim(*viewport[2:])
    axis.set_aspect("equal", adjustable="box")
    draw_tiled_orientation(
        axis,
        data,
        triangles,
        orientation,
        cmap=cmap,
        norm=norm,
        show_nodes=False,
    )
    for inset_axis, inset_viewport in (
        (inset, zoom_viewport),
        (lower_left_inset, lower_left_zoom_viewport),
    ):
        inset_axis.set_zorder(30)
        inset_axis.patch.set_facecolor("white")
        inset_axis.patch.set_alpha(1.0)
        inset_axis.set_xlim(*inset_viewport[:2])
        inset_axis.set_ylim(*inset_viewport[2:])
        inset_axis.set_aspect("equal", adjustable="box")
        draw_tiled_orientation(
            inset_axis,
            data,
            triangles,
            orientation,
            cmap=cmap,
            norm=norm,
            show_nodes=True,
        )
    draw_periodic_shear_box(axis, origin=(0.0, 0.0), load=gamma, box_size=box_size)

    for inset_axis in (inset, lower_left_inset):
        inset_axis.set_xticks([])
        inset_axis.set_yticks([])
    for inset_viewport in (zoom_viewport, lower_left_zoom_viewport):
        axis.add_patch(
            Rectangle(
                (inset_viewport[0], inset_viewport[2]),
                2.0 * zoom_half_width,
                2.0 * zoom_half_width,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                zorder=25,
            )
        )
    for corner in ((zoom_viewport[0], zoom_viewport[3]), (zoom_viewport[1], zoom_viewport[2])):
        axis.add_artist(
            ConnectionPatch(
                xyA=corner,
                coordsA=inset.transData,
                xyB=corner,
                coordsB=axis.transData,
                color="black",
                linewidth=0.6,
                zorder=26,
            )
        )
    for inset_corner, zoom_corner in zip(
        (
            (lower_left_zoom_viewport[0], lower_left_zoom_viewport[2]),
            (lower_left_zoom_viewport[1], lower_left_zoom_viewport[2]),
        ),
        (
            (lower_left_zoom_viewport[0], lower_left_zoom_viewport[3]),
            (lower_left_zoom_viewport[1], lower_left_zoom_viewport[3]),
        ),
        strict=True,
    ):
        axis.add_artist(
            ConnectionPatch(
                xyA=inset_corner,
                coordsA=lower_left_inset.transData,
                xyB=zoom_corner,
                coordsB=axis.transData,
                color="black",
                linewidth=0.6,
                zorder=26,
            )
        )

    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_axis)
    colorbar.set_ticks((0.0, np.pi / 4.0, np.pi / 2.0))
    colorbar.set_ticklabels((r"$0^\circ$", r"$45^\circ$", r"$90^\circ$"))
    colorbar.set_label(orientation_label)
    axis.set_xticks([])
    axis.set_yticks([])

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--simulation", type=Path, default=SIMULATION)
    parser.add_argument("--field", choices=("F", "T", "both"), default="F")
    parser.add_argument(
        "--orientation-measure",
        choices=("polar", "crystal-axis", "crystal-axis-2", "longest-edge"),
        default="polar",
        help="Use polar rotation, F e_1, F e_2, or the longest mesh-edge direction.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "Plots",
    )
    parser.add_argument(
        "--delaunay-longest-edge-comparison",
        action="store_true",
        help="Compare stored connectivity with a Delaunay reconstruction of the 3x3 node tiling.",
    )
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--reconnection-comparison",
        action="store_true",
        help="Plot no-reconnection/Delaunay on the left and reconnecting mesh on the right.",
    )
    comparison_group.add_argument(
        "--reduced-polar-reconnection-comparison",
        action="store_true",
        help="Plot the elastic polar rotation R_e of F_e in both meshes.",
    )
    parser.add_argument("--zoom-half-width", type=float, default=8.0)
    parser.add_argument(
        "--zoom-center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Laboratory-coordinate centre of the inset; defaults to the cell centre.",
    )
    args = parser.parse_args()
    if args.zoom_half_width <= 0.0:
        raise ValueError("zoom-half-width must be positive.")
    if args.reconnection_comparison or args.reduced_polar_reconnection_comparison:
        no_reconnection_vtu, no_reconnection_gamma = closest_vtu(
            NO_RECONNECTION_SIMULATION, args.gamma
        )
        reconnecting_vtu, reconnecting_gamma = closest_vtu(SIMULATION, args.gamma)
        orientation = (
            "reduced-polar"
            if args.reduced_polar_reconnection_comparison
            else "longest-edge"
        )
        output = args.output_directory / (
            "reduced_polar_reconnection_comparison_gamma_0p7.png"
            if orientation == "reduced-polar"
            else "longest_edge_reconnection_comparison_gamma_0p7.png"
        )
        draw_reconnection_comparison(
            no_reconnection_vtu,
            no_reconnection_gamma,
            reconnecting_vtu,
            reconnecting_gamma,
            output,
            target_gamma=args.gamma,
            orientation=orientation,
        )
        print(
            f"Rendered {orientation} no-reconnection and reconnecting maps at gamma="
            f"{no_reconnection_gamma:.5f}/{reconnecting_gamma:.5f} to {output}"
        )
        return
    vtu_path, gamma = closest_vtu(args.simulation, args.gamma)
    if args.delaunay_longest_edge_comparison:
        before_output = args.output_directory / "longest_edge_stored_connectivity_gamma_0p7.png"
        after_output = args.output_directory / "longest_edge_delaunay_connectivity_gamma_0p7.png"
        draw_longest_edge_delaunay_comparison(
            vtu_path, gamma, before_output, after_output
        )
        print(
            f"Rendered stored and Delaunay longest-edge maps from {vtu_path.name} "
            f"at gamma={gamma:.5f} to {before_output} and {after_output}"
        )
        return
    fields = ("F", "T") if args.field == "both" else (args.field,)
    for matrix_name in fields:
        matrix_suffix = "mesh" if args.orientation_measure == "longest-edge" else matrix_name
        output = args.output_directory / (
            f"{args.orientation_measure.replace('-', '_')}_orientation_"
            f"{matrix_suffix}_gamma_0p7.png"
        )
        draw_orientation_map(
            vtu_path,
            gamma,
            output,
            matrix_name=matrix_name,
            orientation_measure=args.orientation_measure,
            zoom_half_width=args.zoom_half_width,
            zoom_center=args.zoom_center,
        )
        print(f"Rendered {matrix_name} from {vtu_path.name} at gamma={gamma:.5f} to {output}")


if __name__ == "__main__":
    main()

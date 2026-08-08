"""Reusable real-space helpers for comparing a small set of VTU mesh states.

This module is intentionally independent of the reversibility classification.
It owns the geometry-sensitive work that other mesh plots may also reuse:

* strict state correspondence checks;
* periodic-image alignment;
* displacement and cell-energy differences;
* activity-based zoom selection; and
* energy backgrounds with amplified displacement arrows.

The public interfaces are defined now; implementations are left explicit for
the follow-up pass.  Unexpected topology or field layouts must raise errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
import numpy as np

from Plotting.vtuDataForSylvain import VTUData


class TopologyRelation(str, Enum):
    """How two mesh states may be compared."""

    IDENTICAL = "identical"
    RECONNECTED = "reconnected"


@dataclass(frozen=True)
class MeshState:
    """One VTU state reduced to the arrays needed by event plots."""

    path: Path
    points: np.ndarray
    triangles: np.ndarray
    reference_indices: np.ndarray
    point_fields: Mapping[str, np.ndarray]
    cell_fields: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class EventDisplacements:
    """Displacements associated with the five-state reversibility protocol."""

    forward_relaxation: np.ndarray  # x2 - x1
    backward_relaxation: np.ndarray  # x4 - x3
    closure_residual: np.ndarray  # x4 - x0


@dataclass(frozen=True)
class ZoomRegion:
    """Periodic-aware Cartesian viewport and the activity it contains."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    activity_fraction: float
    center: tuple[float, float]


@dataclass(frozen=True)
class ArrowScale:
    """Explicit amplification applied to physical displacement vectors."""

    amplification: float
    physical_key_length: float
    target_element_fraction: float


@dataclass(frozen=True)
class FieldGeometry:
    """Geometry used to draw a cell field."""

    kind: str
    triangles: np.ndarray | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    values: np.ndarray | None = None


def load_mesh_state(path: Path) -> MeshState:
    """Load one VTU through ``vtuDataForSylvain.VTUData``.

    The implementation should retain vector point fields such as
    ``displacement`` and scalar cell fields such as ``energy_field``, ``nrm3``
    and ``deltaNrm3``.  Scalar-only ``VTUData.field`` is insufficient for all
    of these arrays, so access to its underlying meshio object is expected.
    """

    data = VTUData(path)
    mesh = data.mesh
    cells = getattr(mesh, "cells_dict", {})
    if "triangle" not in cells:
        raise ValueError(f"Expected triangle cells in {path}, found {list(cells)}.")
    triangles = np.asarray(cells["triangle"], dtype=int)
    points = np.asarray(data.points[:, :2], dtype=float)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Invalid triangle shape {triangles.shape} in {path}.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Non-finite points in {path}.")
    reference_indices = np.asarray(mesh.point_data.get("refIndex"), dtype=int).reshape(-1)
    if reference_indices.shape != (len(points),):
        raise ValueError(f"Invalid refIndex shape {reference_indices.shape} in {path}.")

    point_fields = {
        name: np.asarray(values)
        for name, values in mesh.point_data.items()
    }
    cell_fields = {}
    for name, blocks in mesh.cell_data.items():
        if not isinstance(blocks, (list, tuple)) or len(blocks) != 1:
            raise ValueError(f"Cell field {name!r} has an unsupported block layout.")
        values = np.asarray(blocks[0])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        cell_fields[name] = values
    if "energy_field" not in cell_fields:
        raise KeyError(f"Missing cell energy_field in {path}.")
    if "deltaNrm3" not in cell_fields or "nrm3" not in cell_fields:
        raise KeyError(f"Missing m3 fields in {path}.")
    return MeshState(
        path=Path(path),
        points=points,
        triangles=triangles,
        reference_indices=reference_indices,
        point_fields=point_fields,
        cell_fields=cell_fields,
    )


def determine_topology_relation(first: MeshState, second: MeshState) -> TopologyRelation:
    """Return IDENTICAL only for equal point identities and connectivity."""

    if first.reference_indices.shape != second.reference_indices.shape:
        return TopologyRelation.RECONNECTED
    if not np.array_equal(first.reference_indices, second.reference_indices):
        return TopologyRelation.RECONNECTED
    if first.triangles.shape != second.triangles.shape:
        return TopologyRelation.RECONNECTED
    if np.array_equal(first.triangles, second.triangles):
        return TopologyRelation.IDENTICAL
    return TopologyRelation.RECONNECTED


def align_periodic_states(
    states: Mapping[str, MeshState], *, load_by_state: Mapping[str, float], box_size: float
) -> Mapping[str, MeshState]:
    """Put all states into consistent periodic images using ``refIndex``.

    Duplicate periodic image nodes must agree after alignment.  The routine
    should keep one physical arrow per reference index while preserving enough
    tiled geometry to draw elements crossing the periodic boundary.
    """

    if not states:
        raise ValueError("At least one mesh state is required.")
    names = list(states)
    reference = states[names[0]].reference_indices
    for name in names[1:]:
        state = states[name]
        if not np.array_equal(reference, state.reference_indices):
            raise ValueError(
                f"State {name!r} has different point reference indices; "
                "node correspondence must be implemented explicitly."
            )
    # The saved MTS2D states retain the same periodic-image ordering.  Keep the
    # coordinates unchanged here and make the correspondence check explicit;
    # the renderer removes only rigid mean translation from differences.
    del load_by_state, box_size
    return dict(states)


def calculate_event_displacements(states: Mapping[str, MeshState]) -> EventDisplacements:
    """Calculate x2-x1, x4-x3 and x4-x0 after removing mean translation."""

    required = {
        "state0_min_gamma",
        "state1_affine_gamma_plus",
        "state2_relaxed_gamma_plus",
        "state3_affine_gamma_minus",
        "state4_relaxed_gamma",
    }
    missing = required.difference(states)
    if missing:
        raise KeyError(f"Missing event states: {sorted(missing)}")
    state0 = states["state0_min_gamma"]
    state1 = states["state1_affine_gamma_plus"]
    state2 = states["state2_relaxed_gamma_plus"]
    state3 = states["state3_affine_gamma_minus"]
    state4 = states["state4_relaxed_gamma"]
    if not all(state.points.shape == state0.points.shape for state in states.values()):
        raise ValueError("Event states have different point shapes.")

    def centered(first, second):
        difference = np.asarray(second - first, dtype=float)
        difference -= np.mean(difference, axis=0, keepdims=True)
        return difference

    return EventDisplacements(
        forward_relaxation=centered(state1.points, state2.points),
        backward_relaxation=centered(state3.points, state4.points),
        closure_residual=centered(state0.points, state4.points),
    )


def calculate_forward_m3_change(states: Mapping[str, MeshState]) -> np.ndarray:
    """Return the forward plastic mask on state 2.

    Cross-check ``state2.deltaNrm3`` against ``state2.nrm3-state0.nrm3`` when
    topology is unchanged.  A forward change remains plastic even if state 4
    returns to the state-0 branch.
    """

    state0 = states["state0_min_gamma"]
    state2 = states["state2_relaxed_gamma_plus"]
    relation = determine_topology_relation(state0, state2)
    delta = np.asarray(state2.cell_fields["deltaNrm3"], dtype=float)
    if relation is TopologyRelation.IDENTICAL:
        direct = np.asarray(state2.cell_fields["nrm3"]) - np.asarray(
            state0.cell_fields["nrm3"]
        )
        if delta.shape != direct.shape:
            raise ValueError("deltaNrm3 and nrm3 difference have different shapes.")
        # deltaNrm3 counts elements that changed during the forward
        # minimization, while nrm3(state2)-nrm3(state0) is only a net branch
        # difference.  They need not have identical masks when local m3 moves
        # are later undone within the same forward minimization.
    return delta != 0


def calculate_energy_change_field(
    first: MeshState,
    second: MeshState,
    *,
    relation: TopologyRelation,
    common_grid_resolution: int = 400,
) -> tuple[np.ndarray, object]:
    """Return a spatial representation of ``E(first)-E(second)``.

    For identical topology, direct cell-wise subtraction is allowed.  For a
    reconnecting pair, cell indices have no physical correspondence: project
    both piecewise-constant fields onto the same periodic Cartesian grid and
    subtract there.  Never silently subtract reconnected cell arrays by index.

    The second return value is plotting geometry: a triangulation/polygon set
    for identical topology or a grid extent for projected data.
    """

    first_energy = np.asarray(first.cell_fields["energy_field"], dtype=float)
    second_energy = np.asarray(second.cell_fields["energy_field"], dtype=float)
    if relation is TopologyRelation.IDENTICAL:
        if first_energy.shape != second_energy.shape:
            raise ValueError("Identical-topology energy fields have different shapes.")
        return first_energy - second_energy, FieldGeometry(
            kind="triangles", triangles=first.triangles
        )

    # Reconnecting meshes require a spatial projection.  This uses nearest
    # cell-centre sampling as a deterministic first implementation; it can be
    # replaced by a conservative polygon/grid projection after visual review.
    from scipy.spatial import cKDTree

    def cell_centres(state):
        return state.points[state.triangles].mean(axis=1)

    first_centres = cell_centres(first)
    second_centres = cell_centres(second)
    all_points = np.vstack((first.points, second.points))
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    x = np.linspace(xmin, xmax, common_grid_resolution)
    y = np.linspace(ymin, ymax, common_grid_resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sample_points = np.column_stack((xx.ravel(), yy.ravel()))
    first_nearest = cKDTree(first_centres).query(sample_points)[1]
    second_nearest = cKDTree(second_centres).query(sample_points)[1]
    grid_values = (
        first_energy[first_nearest] - second_energy[second_nearest]
    ).reshape(xx.shape)
    return grid_values, FieldGeometry(kind="grid", x=x, y=y, values=grid_values)


def choose_activity_zoom(
    state: MeshState,
    displacement: np.ndarray,
    *,
    forward_m3_change: np.ndarray | None,
    energy_change: np.ndarray | None,
    activity_fraction: float = 0.8,
    padding_element_lengths: float = 4.0,
) -> ZoomRegion:
    """Choose a periodic-aware viewport containing the dominant event region.

    Plastic events should include all forward-changed m3 elements.  Otherwise
    use displacement-squared activity, with energy decrease only as a secondary
    tie-breaker.  The chosen region must be reproducible, not hand tuned.
    """

    points = np.asarray(state.points, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if displacement.shape != points.shape:
        raise ValueError("Displacement and point arrays have different shapes.")
    if forward_m3_change is not None:
        changed_cells = np.flatnonzero(np.asarray(forward_m3_change, dtype=bool))
    else:
        changed_cells = np.array([], dtype=int)
    if changed_cells.size:
        active_points = np.unique(state.triangles[changed_cells].ravel())
        weights = np.ones(active_points.size, dtype=float)
    else:
        weights = np.sum(displacement**2, axis=1)
        finite = np.isfinite(weights)
        if not np.any(finite & (weights > 0)):
            raise ValueError("No nonzero displacement activity available for zoom.")
        ranked = np.flatnonzero(finite)[np.argsort(weights[finite])[::-1]]
        cumulative = np.cumsum(weights[ranked])
        target = activity_fraction * cumulative[-1]
        active_points = ranked[cumulative <= target]
        if active_points.size == 0:
            active_points = ranked[:1]
        weights = weights[active_points]

    selected = points[active_points]
    edges = points[state.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1)]
    edge_lengths = np.linalg.norm(edges[0::2] - edges[1::2], axis=1)
    element_length = float(np.median(edge_lengths[edge_lengths > 0]))
    if not np.isfinite(element_length) or element_length <= 0:
        raise ValueError("Could not infer a positive mesh element length.")
    xmin, ymin = np.min(selected, axis=0) - padding_element_lengths * element_length
    xmax, ymax = np.max(selected, axis=0) + padding_element_lengths * element_length
    return ZoomRegion(
        xlim=(float(xmin), float(xmax)),
        ylim=(float(ymin), float(ymax)),
        activity_fraction=float(activity_fraction),
        center=(float(np.average(selected[:, 0], weights=weights)), float(np.average(selected[:, 1], weights=weights))),
    )


def _choose_weighted_density_zoom(
    coordinates: np.ndarray,
    weights: np.ndarray,
    domain_points: np.ndarray,
    *,
    convolution_width: float,
    maximum_width: float,
    grid_spacing: float,
    central_margin_fraction: float,
    height_scale: float,
) -> ZoomRegion:
    """Choose a fixed-aspect viewport from a spatially weighted density."""

    coordinates = np.asarray(coordinates, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    domain_points = np.asarray(domain_points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Density coordinates must have shape (N, 2).")
    if coordinates.shape[0] != weights.size:
        raise ValueError("Density coordinates and weights have different lengths.")
    if domain_points.ndim != 2 or domain_points.shape[1] != 2:
        raise ValueError("Domain points must have shape (N, 2).")
    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite.")
    if not np.isfinite(convolution_width) or convolution_width <= 0:
        raise ValueError("convolution_width must be positive and finite.")
    if not np.isfinite(maximum_width) or maximum_width <= 0:
        raise ValueError("maximum_width must be positive and finite.")
    if not 0 <= central_margin_fraction < 0.5:
        raise ValueError("central_margin_fraction must lie in [0, 0.5).")
    if not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("height_scale must be positive and finite.")

    valid = np.all(np.isfinite(coordinates), axis=1) & np.isfinite(weights)
    if not np.any(valid & (weights > 0)):
        raise ValueError("No nonzero finite density weight is available for zooming.")
    coordinates = coordinates[valid]
    weights = weights[valid]

    xmin, ymin = np.min(domain_points, axis=0)
    xmax, ymax = np.max(domain_points, axis=0)
    span_x = float(xmax - xmin)
    span_y = float(ymax - ymin)
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Mesh has a non-positive spatial extent.")
    x_edges = np.arange(
        np.floor(xmin / grid_spacing) * grid_spacing,
        np.ceil(xmax / grid_spacing) * grid_spacing + grid_spacing,
        grid_spacing,
    )
    y_edges = np.arange(
        np.floor(ymin / grid_spacing) * grid_spacing,
        np.ceil(ymax / grid_spacing) * grid_spacing + grid_spacing,
        grid_spacing,
    )
    if len(x_edges) < 2 or len(y_edges) < 2:
        raise ValueError("Closure-density grid has too few cells.")

    density, _, _ = np.histogram2d(
        coordinates[:, 0], coordinates[:, 1],
        bins=(x_edges, y_edges), weights=weights,
    )
    from scipy.ndimage import uniform_filter

    window_cells = max(1, int(round(convolution_width / grid_spacing)))
    density = uniform_filter(
        density, size=(window_cells, window_cells), mode="constant"
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    width = min(float(maximum_width), span_x)
    height = min(width * span_y / span_x * height_scale, span_y)
    valid_x = (
        (x_centers >= xmin + width / 2)
        & (x_centers <= xmax - width / 2)
        & (x_centers >= xmin + central_margin_fraction * span_x)
        & (x_centers <= xmax - central_margin_fraction * span_x)
    )
    valid_y = (
        (y_centers >= ymin + height / 2)
        & (y_centers <= ymax - height / 2)
        & (y_centers >= ymin + central_margin_fraction * span_y)
        & (y_centers <= ymax - central_margin_fraction * span_y)
    )
    allowed = valid_x[:, None] & valid_y[None, :]
    scores = np.where(allowed, density, -np.inf)
    if not np.any(np.isfinite(scores)):
        scores = density
    index = np.unravel_index(np.nanargmax(scores), scores.shape)
    center = (float(x_centers[index[0]]), float(y_centers[index[1]]))

    x0 = min(max(center[0] - width / 2, xmin), xmax - width)
    y0 = min(max(center[1] - height / 2, ymin), ymax - height)
    return ZoomRegion(
        xlim=(float(x0), float(x0 + width)),
        ylim=(float(y0), float(y0 + height)),
        activity_fraction=1.0,
        center=center,
    )


def choose_closure_density_zoom(
    state: MeshState,
    closure_residual: np.ndarray,
    *,
    convolution_width: float = 10.0,
    maximum_width: float = 20.0,
    grid_spacing: float = 1.0,
    central_margin_fraction: float = 0.20,
) -> ZoomRegion:
    """Center a fixed-aspect zoom on the strongest closure-residual region."""

    points = np.asarray(state.points, dtype=float)
    residual = np.asarray(closure_residual, dtype=float)
    if residual.shape != points.shape:
        raise ValueError("Closure residual and mesh points have different shapes.")
    _, unique = np.unique(state.reference_indices, return_index=True)
    unique = np.sort(unique)
    return _choose_weighted_density_zoom(
        points[unique],
        np.linalg.norm(residual[unique], axis=1),
        points,
        convolution_width=convolution_width,
        maximum_width=maximum_width,
        grid_spacing=grid_spacing,
        central_margin_fraction=central_margin_fraction,
        height_scale=1.0,
    )


def choose_energy_density_zoom(
    state: MeshState,
    energy_change: np.ndarray,
    geometry: FieldGeometry,
    *,
    convolution_width: float = 10.0,
    maximum_width: float = 20.0,
    grid_spacing: float = 1.0,
    central_margin_fraction: float = 0.20,
    height_scale: float = 0.8,
) -> ZoomRegion:
    """Center a zoom on the strongest local absolute energy difference.

    The energy magnitude is binned and smoothed only to select the viewport;
    the smoothed density is not shown in the event figure.  Both positive and
    negative energy differences therefore contribute to the selected region.
    """

    if geometry.kind == "triangles":
        triangles = np.asarray(geometry.triangles, dtype=int)
        values = np.asarray(energy_change, dtype=float).reshape(-1)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Triangle geometry must have shape (N, 3).")
        if values.shape != (len(triangles),):
            raise ValueError("Triangle energy values have an unexpected shape.")
        coordinates = np.mean(state.points[triangles], axis=1)
    elif geometry.kind == "grid":
        x = np.asarray(geometry.x, dtype=float)
        y = np.asarray(geometry.y, dtype=float)
        values = np.asarray(geometry.values, dtype=float)
        if values.shape != (len(y), len(x)):
            raise ValueError("Grid energy values do not match grid coordinates.")
        xx, yy = np.meshgrid(x, y, indexing="xy")
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))
        values = values.ravel()
    else:
        raise ValueError(f"Unknown energy geometry kind {geometry.kind!r}.")
    return _choose_weighted_density_zoom(
        coordinates,
        np.abs(values),
        state.points,
        convolution_width=convolution_width,
        maximum_width=maximum_width,
        grid_spacing=grid_spacing,
        central_margin_fraction=central_margin_fraction,
        height_scale=height_scale,
    )


def choose_arrow_scale(
    displacement: np.ndarray,
    *,
    element_length: float,
    target_element_fraction: float = 1.0 / 3.0,
) -> ArrowScale:
    """Choose and round an amplification from the 95th displacement percentile."""

    magnitudes = np.linalg.norm(np.asarray(displacement, dtype=float), axis=1)
    positive = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    if positive.size == 0:
        return ArrowScale(1.0, 0.0, target_element_fraction)
    q95 = float(np.quantile(positive, 0.95))
    raw = target_element_fraction * element_length / q95
    exponent = np.floor(np.log10(raw))
    mantissa = raw / 10**exponent
    rounded_mantissa = min((1.0, 2.0, 5.0, 10.0), key=lambda value: abs(value - mantissa))
    amplification = float(rounded_mantissa * 10**exponent)
    return ArrowScale(amplification, q95, target_element_fraction)


def plot_energy_change_background(
    ax,
    state: MeshState,
    energy_change: np.ndarray,
    geometry: object,
    *,
    zoom: ZoomRegion,
    symmetric_limit: float | None = None,
    show_mesh_edges: bool = True,
    mesh_edge_color: str | None = None,
    rasterized: bool = False,
):
    """Draw a coolwarm field with zero at the neutral midpoint.

    ``mesh_edge_color="face"`` gives each triangular element an outline in
    its own face color, keeping the element boundaries visible without adding
    a separate black activity or m3 outline convention.
    """

    if mesh_edge_color == "face" and not show_mesh_edges:
        raise ValueError("mesh_edge_color='face' requires show_mesh_edges=True.")

    if geometry.kind == "triangles":
        triangulation = mtri.Triangulation(
            state.points[:, 0], state.points[:, 1], geometry.triangles
        )
        limit = symmetric_limit or float(np.nanmax(np.abs(energy_change)))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        mesh = ax.tripcolor(
            triangulation,
            facecolors=energy_change,
            cmap="coolwarm",
            norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="flat",
            edgecolors=mesh_edge_color if mesh_edge_color is not None else "none",
            linewidth=0.12 if mesh_edge_color is not None else 0.0,
        )
        if rasterized:
            mesh.set_rasterized(True)
        if show_mesh_edges and mesh_edge_color is None:
            edge_artists = ax.triplot(
                triangulation, color="black", linewidth=0.08, alpha=0.22
            )
            if rasterized:
                for artist in edge_artists:
                    artist.set_rasterized(True)
    elif geometry.kind == "grid":
        limit = symmetric_limit or float(np.nanmax(np.abs(energy_change)))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        mesh = ax.pcolormesh(
            geometry.x,
            geometry.y,
            geometry.values,
            cmap="coolwarm",
            norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            shading="auto",
        )
        if rasterized:
            mesh.set_rasterized(True)
    else:
        raise ValueError(f"Unknown field geometry kind {geometry.kind!r}.")
    ax.set_xlim(*zoom.xlim)
    ax.set_ylim(*zoom.ylim)
    ax.set_aspect("equal", adjustable="box")
    return mesh


def plot_displacement_arrows(
    ax,
    origins: np.ndarray,
    displacement: np.ndarray,
    *,
    arrow_scale: ArrowScale,
    zoom: ZoomRegion,
    show_key: bool = True,
    key_label: str | None = None,
    key_position: tuple[float, float] = (0.05, 0.90),
    rasterized: bool = False,
):
    """Overlay amplified arrows and optionally an unscaled physical key."""

    origins = np.asarray(origins, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if origins.shape != displacement.shape or origins.ndim != 2 or origins.shape[1] != 2:
        raise ValueError("Origins and displacements must both have shape (N, 2).")
    inside = (
        (origins[:, 0] >= zoom.xlim[0])
        & (origins[:, 0] <= zoom.xlim[1])
        & (origins[:, 1] >= zoom.ylim[0])
        & (origins[:, 1] <= zoom.ylim[1])
    )
    vectors = displacement * arrow_scale.amplification
    quiver = ax.quiver(
        origins[inside, 0],
        origins[inside, 1],
        vectors[inside, 0],
        vectors[inside, 1],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0018,
        headwidth=5.5,
        headlength=6.0,
        headaxislength=5.5,
        alpha=0.82,
        pivot="mid",
        # Do not discard short vectors.  Exact zero vectors are still passed
        # to Matplotlib, but cannot have a visible direction or length.
        minlength=0,
        minshaft=0,
        zorder=5,
    )
    if rasterized:
        quiver.set_rasterized(True)
    if show_key and arrow_scale.physical_key_length > 0:
        # Draw the scale as ordinary axes artists instead of a QuiverKey.
        # QuiverKey contains its own artists and can be reordered internally;
        # these explicit artists are guaranteed to sit above the quiver.
        ax.add_patch(
            Rectangle(
                (0.02, 0.84),
                0.48,
                0.12,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="0.35",
                linewidth=0.45,
                alpha=1.0,
                zorder=1000,
            )
        )
        arrow_x, arrow_y = key_position
        ax.text(
            arrow_x,
            arrow_y,
            key_label or rf"$|\rightarrow| = {arrow_scale.physical_key_length:.1e}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6,
            color="black",
            zorder=1001,
        )
    return quiver


def plot_zoom_locator(ax, state: MeshState, zoom: ZoomRegion) -> None:
    """Draw the full periodic cell with the automatically selected viewport."""

    triangulation = mtri.Triangulation(
        state.points[:, 0], state.points[:, 1], state.triangles
    )
    ax.triplot(triangulation, color="0.55", linewidth=0.08, alpha=0.35)
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (zoom.xlim[0], zoom.ylim[0]),
            zoom.xlim[1] - zoom.xlim[0],
            zoom.ylim[1] - zoom.ylim[0],
            fill=False,
            color="C3",
            linewidth=1.2,
        )
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.set_title("full mesh / selected region")

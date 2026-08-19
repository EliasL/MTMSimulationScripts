#!/usr/bin/env python3
"""Render the largest saved edge-flip events as mesh/Poincare triptychs.

The default job is the currently running 200x200 edge-flip reversibility run
that restarted from load 1.  By default the script selects the globally
largest recorded equilibrium-energy drops and requires the corresponding five
reversibility-protocol VTUs.  It refuses to substitute smaller sampled events.

Use ``--saved-only`` only for a preview from the periodically saved protocol
folders.  Exact unsaved events first need a targeted replay that writes their
five protocol states.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib import patheffects
from matplotlib.ticker import ScalarFormatter

from MTMath.poincareEnergy import C2PoincareDisk, drawCScatter, prepPoincareFig
from MTMath.reduction import plastic_reduction
from Plotting import meshEventPlotting as mesh_plot
from Plotting.dataFunctions import VTUData
from Plotting.pyplotFunctions import (
    add_padding,
    calculate_shifts,
    plot_binned_poincare_displacement_field,
    plot_mesh,
)
from Plotting.real_space_events.acquisition import state_paths_from_directory
from Plotting.real_space_events.models import EventStatePaths
from Plotting.reconnectingEnergyJumpAndElementDistribution import (
    read_live_macro_snapshot,
)


DEFAULT_JOB = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
STATE_DIRECTORY_PATTERN = re.compile(
    r"(?:rev|irrev|elastic_replay)_drop_l_(?P<load>[0-9.eE+-]+)$"
)
SIZE_PATTERN = re.compile(r"s(?P<x>\d+)x(?P<y>\d+)")
DISK_GRID_SIZE = 320
DENSITY_GRID_SIZE = 200


@dataclass(frozen=True)
class SelectedEvent:
    rank: int
    load_step: int
    load_increment: float
    box_size: float
    start_load: float
    target_load: float
    energy_drop: float
    state_directory: Path
    state_paths: EventStatePaths


def _macro_events(job_directory: Path) -> tuple[pd.DataFrame, float]:
    """Return recorded negative equilibrium-energy changes and the step size."""

    macro = read_live_macro_snapshot(
        Path(job_directory) / "macroData.csv",
        columns=("load_step", "load", "total_energy_change"),
    ).copy()
    for column in macro:
        macro[column] = pd.to_numeric(macro[column], errors="raise")
    if macro.empty:
        raise ValueError(f"No complete macro-data rows found in {job_directory}.")
    load = macro["load"].to_numpy(dtype=float)
    increments = np.diff(load)
    increments = increments[np.isfinite(increments) & (increments > 0)]
    if increments.size == 0:
        raise ValueError(f"Could not infer a positive load increment from {job_directory}.")
    increment = float(np.median(increments))
    if not np.allclose(increments, increment, rtol=1e-8, atol=1e-12):
        raise ValueError("The macro-data load increment is not constant.")
    events = macro[np.isfinite(macro["total_energy_change"]) & (macro["total_energy_change"] < 0)].copy()
    if events.empty:
        raise ValueError(f"No negative equilibrium-energy changes found in {job_directory}.")
    events["energy_drop"] = -events["total_energy_change"]
    return events.sort_values("energy_drop", ascending=False).reset_index(drop=True), increment


def _periodic_box_size(job_directory: Path) -> float:
    """Read the square periodic-cell side length from the simulation name."""

    match = SIZE_PATTERN.search(Path(job_directory).name)
    if match is None:
        raise ValueError(f"Could not find an sLxL size in {job_directory}.")
    x_size = int(match.group("x"))
    y_size = int(match.group("y"))
    if x_size != y_size:
        raise ValueError(f"The periodic renderer requires a square cell, got {x_size}x{y_size}.")
    if x_size <= 0:
        raise ValueError("Periodic box size must be positive.")
    return float(x_size)


def _saved_states_by_start_load(event_root: Path) -> dict[float, tuple[Path, EventStatePaths]]:
    """Resolve complete five-state protocol folders by their saved start load."""

    if not event_root.is_dir():
        raise FileNotFoundError(f"No reversibility-state directory found at {event_root}.")
    result = {}
    for directory in event_root.iterdir():
        if not directory.is_dir():
            continue
        match = STATE_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        start_load = float(match.group("load"))
        if start_load in result:
            raise ValueError(f"Duplicate protocol-state start load {start_load:.8g} in {event_root}.")
        result[start_load] = (directory, state_paths_from_directory(directory))
    if not result:
        raise RuntimeError(f"No complete five-state directories found in {event_root}.")
    return result


def _state_entry_for_start_load(
    saved_states: dict[float, tuple[Path, EventStatePaths]], start_load: float, increment: float
) -> tuple[Path, EventStatePaths] | None:
    tolerance = max(1e-10, increment * 1e-5)
    matches = [value for load, value in saved_states.items() if abs(load - start_load) <= tolerance]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous saved protocol state for start load {start_load:.8g}.")
    return matches[0] if matches else None


def select_events(job_directory: Path, *, number: int, saved_only: bool) -> list[SelectedEvent]:
    """Select the largest drops, never replacing missing global selections silently."""

    if number <= 0:
        raise ValueError("number must be positive.")
    events, increment = _macro_events(job_directory)
    box_size = _periodic_box_size(job_directory)
    saved_states = _saved_states_by_start_load(Path(job_directory) / "data" / "reversibilityData")
    selected = []
    missing = []
    for rank, row in events.iterrows():
        start_load = float(row["load"] - increment)
        state_entry = _state_entry_for_start_load(saved_states, start_load, increment)
        if state_entry is None:
            missing.append((int(rank) + 1, int(row["load_step"]), start_load, float(row["energy_drop"])))
            if not saved_only and len(selected) + len(missing) == number:
                break
            continue
        directory, state_paths = state_entry
        selected.append(
            SelectedEvent(
                rank=int(rank) + 1,
                load_step=int(row["load_step"]),
                load_increment=increment,
                box_size=box_size,
                start_load=start_load,
                target_load=float(row["load"]),
                energy_drop=float(row["energy_drop"]),
                state_directory=directory,
                state_paths=state_paths,
            )
        )
        if len(selected) == number:
            break

    if saved_only:
        if len(selected) < number:
            raise RuntimeError(f"Only {len(selected)} complete saved protocol events are available; requested {number}.")
        return selected
    if missing:
        details = "; ".join(
            f"rank {rank} (step {step}, gamma={load:.5f}, Delta E={drop:.6g})"
            for rank, step, load, drop in missing
        )
        raise RuntimeError(
            "The requested globally largest events have no complete five-state "
            f"protocol snapshots: {details}. Replay those exact target loads before "
            "rendering, or use --saved-only explicitly for a non-equivalent preview."
        )
    if len(selected) != number:
        raise RuntimeError(f"Expected {number} selected events, found {len(selected)}.")
    return selected


def _plain_tick_formatter() -> ScalarFormatter:
    """Format mesh colourbar ticks as ordinary decimal numbers."""

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    return formatter


def _finite_upper_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot determine a colour limit from an empty finite field.")
    limit = float(np.percentile(finite, percentile))
    if not np.isfinite(limit):
        raise ValueError("The percentile colour limit is not finite.")
    return limit


def _add_density_bin_square(axis: plt.Axes, *, zoom: float) -> None:
    """Show one density-histogram bin in the lower-right disk corner.

    Histogram bins are defined in Poincare coordinates and then projected to
    the disk image coordinates used by ``prepPoincareFig``.  A one-bin square
    therefore has the same displayed size at different zooms even though its
    Poincare-coordinate width changes with the zoom.
    """

    if not np.isfinite(zoom) or zoom < 1:
        raise ValueError("Poincare zoom must be finite and at least one.")
    radius = 1.0 / zoom
    scale = zoom * DISK_GRID_SIZE / 2.0
    bin_width = 2.0 * radius / DENSITY_GRID_SIZE
    x_disk = 0.82 * radius
    y_disk = -0.94 * radius
    x_plot = DISK_GRID_SIZE / 2.0 + scale * x_disk
    y_plot = DISK_GRID_SIZE / 2.0 + scale * y_disk
    plot_width = scale * bin_width
    axis.add_patch(
        Rectangle(
            (x_plot, y_plot),
            plot_width,
            plot_width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=110,
        )
    )


def _total_T_and_triangle_keys(
    path: Path, *, load_increment: float
) -> tuple[np.ndarray, dict[tuple[int, int, int], int]]:
    """Return total ``T`` and stable reference-index keys for one state."""

    data = VTUData(path, load_increment=load_increment)
    T = np.asarray(data.get_T(), dtype=float)
    if T.ndim != 3 or T.shape[1:] != (2, 2):
        raise ValueError(f"Expected one 2x2 T per element in {path}, got {T.shape}.")

    triangles = np.asarray(data.mesh.cells_dict.get("triangle"), dtype=int)
    reference_indices = np.asarray(data.get_point_data("refIndex"), dtype=int).reshape(-1)
    if triangles.shape != (len(T), 3):
        raise ValueError(f"Triangle/matrix count mismatch in {path}.")
    keys = [tuple(sorted(reference_indices[triangle].tolist())) for triangle in triangles]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Triangle reference keys are not unique in {path}.")
    return T, dict(zip(keys, range(len(keys)), strict=True))


def _metric_from_total_T(T: np.ndarray, *, source: Path) -> np.ndarray:
    """Return and validate the positive-definite metric induced by total ``T``."""

    T = np.asarray(T, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (2, 2):
        raise ValueError(f"Expected one 2x2 T per element in {source}, got {T.shape}.")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"Total T contains non-finite values in {source}.")
    metric = np.swapaxes(T, -1, -2) @ T
    determinant = np.linalg.det(metric)
    if not np.all(np.isfinite(determinant) & (determinant > 0)):
        raise ValueError(f"T-derived metric is not positive definite in {source}.")
    return metric


def _disk_coordinates(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = C2PoincareDisk(metric)
    valid = np.isfinite(x) & np.isfinite(y) & (x * x + y * y < 1.0 + 1e-10)
    if not np.all(valid):
        raise ValueError("A T-derived metric could not be mapped inside the Poincare disk.")
    return np.asarray(x), np.asarray(y)


def _prepare_poincare_axis(axis: plt.Axes, *, zoom: float) -> None:
    """Draw the Poincare background for one fixed zoom."""

    if not np.isfinite(zoom) or zoom < 1:
        raise ValueError("Poincare zoom must be finite and at least one.")
    prepPoincareFig(
        grid_size=DISK_GRID_SIZE,
        zoom=zoom,
        ax=axis,
        withGrid=True,
        withYieldSurface=False,
        minimalTicks=False,
    )
    axis.set_aspect("equal", adjustable="box")


def _plot_full_poincare_disk(
    axis: plt.Axes,
    after_path: Path,
    *,
    load_increment: float,
) -> tuple[object, int]:
    """Show density-coloured new-equilibrium total-``T`` points."""

    after_T, _ = _total_T_and_triangle_keys(
        after_path, load_increment=load_increment
    )
    after_metric = _metric_from_total_T(after_T, source=after_path)
    _prepare_poincare_axis(axis, zoom=1)
    scatter = drawCScatter(
        axis,
        after_metric,
        DISK_GRID_SIZE,
        zoom=1,
        density_method="hist",
        density_grid_size=DENSITY_GRID_SIZE,
        show_colorbar=False,
        alpha=0.65,
        zorder=4,
    )
    if scatter is None:
        raise RuntimeError("The new-equilibrium total-T cloud is empty.")
    _add_density_bin_square(axis, zoom=1)
    return scatter, len(after_T)


def _plot_almost_plastically_reduced_transition(
    axis: plt.Axes,
    before_path: Path,
    after_path: Path,
    *,
    load_increment: float,
    density_vmax: int,
    vector_colorbar_axis: plt.Axes | None,
    show_vector_colorbar: bool = True,
    draw_arrows: bool = True,
    return_metadata: bool = False,
) -> int | tuple[int, list[dict]]:
    """Reduce the previous branch, then advance it by the measured total-``T`` change.

    For each persistent triangle, ``M`` reduces only the previous equilibrium:
    ``T_before_reduced = T_before @ M``.  The new point is then the direct
    image under ``T_after @ inv(T_before)``.  It is intentionally not reduced
    a second time, even if this final operation leaves the fundamental elastic
    well.  Arrow colours use the increment relative to identity,
    ``Delta_T = T_after @ inv(T_before) - I``.
    """

    before_T, before_keys = _total_T_and_triangle_keys(
        before_path, load_increment=load_increment
    )
    after_T, after_keys = _total_T_and_triangle_keys(
        after_path, load_increment=load_increment
    )
    before_metric = _metric_from_total_T(before_T, source=before_path)
    common = sorted(before_keys.keys() & after_keys.keys())
    if not common:
        raise RuntimeError("No persistent triangles can be compared between the equilibrium states.")
    before_indices = np.asarray([before_keys[key] for key in common], dtype=int)
    after_indices = np.asarray([after_keys[key] for key in common], dtype=int)

    reduced_before_metric, reduction_M = plastic_reduction(
        before_metric[before_indices], compute_M=True
    )
    # Algebraically, the new representative is
    # (T_after @ inv(T_before)) @ (T_before @ M) = T_after @ M.  The matrix
    # increment used for colour is the change relative to the identity,
    # Delta T = T_after @ inv(T_before) - I.  This removes the identity
    # baseline (whose Frobenius norm is sqrt(2)) and measures the actual step.
    try:
        before_transpose = np.swapaxes(before_T[before_indices], -1, -2)
        after_transpose = np.swapaxes(after_T[after_indices], -1, -2)
        total_T_increment = np.linalg.solve(before_transpose, after_transpose)
        total_T_increment = np.swapaxes(total_T_increment, -1, -2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not solve for the total-T increment.") from exc
    delta_T = total_T_increment - np.eye(2)
    delta_T_frobenius = np.linalg.norm(delta_T, axis=(-2, -1))
    if not np.all(np.isfinite(delta_T_frobenius)):
        raise ValueError("The total-T transformation has non-finite Frobenius norms.")
    advanced_after_T = after_T[after_indices] @ reduction_M
    advanced_after_metric = _metric_from_total_T(advanced_after_T, source=after_path)

    before_x, before_y = _disk_coordinates(reduced_before_metric)
    after_x, after_y = _disk_coordinates(advanced_after_metric)
    _prepare_poincare_axis(axis, zoom=2)
    drawCScatter(
        axis,
        advanced_after_metric,
        DISK_GRID_SIZE,
        zoom=2,
        density_method="hist",
        density_grid_size=DENSITY_GRID_SIZE,
        show_colorbar=False,
        vmax=density_vmax,
        alpha=0.65,
        zorder=4,
    )
    zoom = 2
    dx = after_x - before_x
    dy = after_y - before_y
    field_result = plot_binned_poincare_displacement_field(
        axis,
        before_x,
        before_y,
        dx,
        dy,
        grid_size=DISK_GRID_SIZE,
        zoom=zoom,
        bins=40,
        min_count=5,
        min_coherence=0.0,
        show_colorbar=show_vector_colorbar,
        colorbar_axes=vector_colorbar_axis,
        color_values=delta_T_frobenius,
        colorbar_label=r"mean $\|\Delta\mathbf{T}\|_F$",
        min_vector_length=0.1,
        colorbar_log=False,
        colorbar_max_quantile=0.95,
        vector_length_from_color=True,
        vector_length_scale=0.49,
        direction_split_otsu=True,
        draw_arrows=draw_arrows,
        return_metadata=return_metadata,
    )
    if return_metadata:
        _quiver, _shown_count, _populated_count, arrow_metadata = field_result
        for record in arrow_metadata:
            source_indices = record["source_indices"]
            record["before_element_indices"] = before_indices[source_indices].copy()
            record["after_element_indices"] = after_indices[source_indices].copy()
    else:
        arrow_metadata = []
    result = len(advanced_after_T)
    return (result, arrow_metadata) if return_metadata else result


def _draw_selected_poincare_arrows(
    axis: plt.Axes,
    arrows: list[dict],
    colors: list[str],
    *,
    zoom: float,
) -> object:
    """Draw already-selected Poincare arrows using the shared disk scale."""

    if len(arrows) != len(colors) or not arrows:
        raise ValueError("Each selected Poincare arrow must have one display colour.")
    scale = float(zoom) * DISK_GRID_SIZE / 2.0
    center = float(DISK_GRID_SIZE) / 2.0
    quiver = axis.quiver(
        center + scale * np.array([arrow["x"] for arrow in arrows]),
        center + scale * np.array([arrow["y"] for arrow in arrows]),
        scale * np.array([arrow["u"] for arrow in arrows]),
        scale * np.array([arrow["v"] for arrow in arrows]),
        color=colors,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="tail",
        width=0.006,
        headwidth=4.0,
        headlength=5.0,
        headaxislength=4.2,
        zorder=7,
    )
    quiver.set_path_effects(
        [patheffects.withStroke(linewidth=0.8, foreground="white")]
    )
    return quiver


def _render_three_arrow_mesh_figure(
    event: SelectedEvent,
    state0: mesh_plot.MeshState,
    state2: mesh_plot.MeshState,
    output_directory: Path,
    *,
    density_vmax: int,
    dpi: int,
) -> Path:
    """Render the selected-arrow disk beside before/after coloured meshes."""

    arrow_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    all_points = np.vstack((state0.points[:, :2], state2.points[:, :2]))
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    viewport = add_padding((x_min, x_max, y_min, y_max), 0.03)
    mesh_cmap = ListedColormap(["#eeeeee", *arrow_colors])
    mesh_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), mesh_cmap.N)

    figure = plt.figure(figsize=(15.0, 5.2))
    disk_axis = figure.add_axes((0.035, 0.18, 0.29, 0.72))
    before_axis = figure.add_axes((0.355, 0.18, 0.29, 0.72))
    after_axis = figure.add_axes((0.675, 0.18, 0.29, 0.72))

    _point_count, arrow_metadata = _plot_almost_plastically_reduced_transition(
        disk_axis,
        event.state_paths.state0_min_gamma,
        event.state_paths.state2_relaxed_gamma_plus,
        load_increment=event.load_increment,
        density_vmax=density_vmax,
        vector_colorbar_axis=None,
        show_vector_colorbar=False,
        draw_arrows=False,
        return_metadata=True,
    )
    if len(arrow_metadata) < 3:
        raise RuntimeError(
            f"Only {len(arrow_metadata)} visible arrows are available; three are required."
        )
    selected_arrows = sorted(
        arrow_metadata, key=lambda arrow: arrow["length"], reverse=True
    )[:3]
    before_labels = np.zeros(len(state0.triangles), dtype=int)
    after_labels = np.zeros(len(state2.triangles), dtype=int)
    for label, arrow in enumerate(selected_arrows, start=1):
        before_indices = np.unique(np.asarray(arrow["before_element_indices"], dtype=int))
        after_indices = np.unique(np.asarray(arrow["after_element_indices"], dtype=int))
        if before_indices.size == 0 or after_indices.size == 0:
            raise ValueError("A selected arrow has no corresponding mesh elements.")
        if np.any(before_indices < 0) or np.any(before_indices >= len(before_labels)):
            raise IndexError("A selected before-arrow element index is out of range.")
        if np.any(after_indices < 0) or np.any(after_indices >= len(after_labels)):
            raise IndexError("A selected after-arrow element index is out of range.")
        if np.any(before_labels[before_indices]) or np.any(after_labels[after_indices]):
            raise ValueError("Selected arrows share mesh elements unexpectedly.")
        before_labels[before_indices] = label
        after_labels[after_indices] = label

    # ``plot_mesh`` reads the same triangle ordering used by the T arrays.
    # Refuse to colour by position if a future VTU reader changes that order.
    for path, state in (
        (event.state_paths.state0_min_gamma, state0),
        (event.state_paths.state2_relaxed_gamma_plus, state2),
    ):
        reader_connectivity = np.asarray(
            VTUData(path, load_increment=event.load_increment).get_connectivity(),
            dtype=int,
        )
        if not np.array_equal(reader_connectivity, state.triangles):
            raise ValueError(
                f"Triangle ordering differs between mesh readers for {path}."
            )
    _draw_selected_poincare_arrows(
        disk_axis, selected_arrows, arrow_colors, zoom=2
    )
    disk_axis.set_title(r"(a) reduced $\mathbf{T}^{\mathsf{T}}\mathbf{T}$")
    disk_axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=f"arrow {index}")
            for index, color in enumerate(arrow_colors, start=1)
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
    )

    for axis, path, load, labels, title in (
        (
            before_axis,
            event.state_paths.state0_min_gamma,
            event.start_load,
            before_labels,
            "(b) before",
        ),
        (
            after_axis,
            event.state_paths.state2_relaxed_gamma_plus,
            event.target_load,
            after_labels,
            "(c) after",
        ),
    ):
        plot_mesh(
            path,
            mesh_property="selected arrow elements",
            ax=axis,
            add_colorbar=False,
            add_rombus=True,
            periodic_load=load,
            periodic_box_size=event.box_size,
            load_increment=event.load_increment,
            cartesian_viewport_culling=True,
            cartesian_viewport=viewport,
            field_override=labels,
            field_cmap=mesh_cmap,
            field_norm=mesh_norm,
            field_background_color=(0.15, 0.15, 0.15, 0.35),
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])

    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (
        f"rank{event.rank:03d}_gamma{event.target_load:.5f}_three_arrows_meshes.png"
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return output


def render_event(
    event: SelectedEvent, output_directory: Path, *, dpi: int
) -> tuple[Path, Path, Path]:
    """Render energy, Poincare, and selected-arrow mesh figures for one event."""

    state0 = mesh_plot.load_mesh_state(event.state_paths.state0_min_gamma)
    state2 = mesh_plot.load_mesh_state(event.state_paths.state2_relaxed_gamma_plus)
    energy_change, geometry = mesh_plot.calculate_periodic_energy_change_field(
        state0,
        state2,
        first_load=event.start_load,
        second_load=event.target_load,
        box_size=event.box_size,
        common_grid_resolution=400,
    )
    values = energy_change if geometry.kind == "triangles" else np.asarray(geometry.values)
    energy_field = np.asarray(state2.cell_fields["energy_field"], dtype=float)
    energy_min = float(np.nanmin(energy_field))
    energy_max = _finite_upper_percentile(energy_field, 99.0)
    if energy_max <= energy_min:
        raise ValueError("The 99th-percentile energy limit is not above the energy minimum.")
    energy_limit = _finite_upper_percentile(np.abs(values), 99.0)
    if not np.isfinite(energy_limit) or energy_limit <= 0:
        raise ValueError(
            f"No finite nonzero cell-energy change in {event.state_directory}."
        )

    x_min, y_min = np.min(state2.points, axis=0)
    x_max, y_max = np.max(state2.points, axis=0)
    x_min, x_max, y_min, y_max = add_padding((x_min, x_max, y_min, y_max), 0.03)
    viewport = (x_min, x_max, y_min, y_max)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"rank{event.rank:03d}_gamma{event.target_load:.5f}"

    # Energy figure: the two real-space fields in one row.
    energy_figure = plt.figure(figsize=(14.0, 5.0))
    mesh_width = 0.40
    mesh_height = mesh_width * 14.0 / 7.0 * (y_max - y_min) / (x_max - x_min)
    if mesh_height <= 0 or mesh_height >= 0.82:
        raise ValueError("The mesh row cannot accommodate this viewport aspect ratio.")
    new_energy_axis = energy_figure.add_axes((0.040, 0.20, 0.44, mesh_height))
    mesh_axis = energy_figure.add_axes((0.520, 0.20, 0.44, mesh_height))
    new_energy_colorbar_axis = energy_figure.add_axes((0.080, 0.08, 0.36, 0.022))
    energy_change_colorbar_axis = energy_figure.add_axes((0.560, 0.08, 0.36, 0.022))

    new_energy_axis, new_energy_cmap, new_energy_norm = plot_mesh(
        event.state_paths.state2_relaxed_gamma_plus,
        mesh_property="energy",
        e_lims=(energy_min, energy_max),
        ax=new_energy_axis,
        add_colorbar=False,
        add_rombus=True,
        periodic_load=event.target_load,
        periodic_box_size=event.box_size,
        load_increment=event.load_increment,
        cartesian_viewport_culling=True,
        cartesian_viewport=viewport,
    )
    new_energy_axis.set_title(r"(a) $E^{(2)}$")
    new_energy_mappable = ScalarMappable(cmap=new_energy_cmap, norm=new_energy_norm)
    new_energy_mappable.set_array(state2.cell_fields["energy_field"])
    new_energy_colorbar = energy_figure.colorbar(
        new_energy_mappable, cax=new_energy_colorbar_axis, orientation="horizontal"
    )
    new_energy_colorbar.set_label(r"$E^{(e)}$")
    new_energy_colorbar.formatter = _plain_tick_formatter()
    new_energy_colorbar.update_ticks()

    mesh_axis.set_xlim(x_min, x_max)
    mesh_axis.set_ylim(y_min, y_max)
    tiling_data = VTUData(
        event.state_paths.state2_relaxed_gamma_plus,
        load_increment=event.load_increment,
    )
    tiling_data.BC = "PBC"
    tiling_data.load = event.target_load
    tiling_data.size = (int(event.box_size), int(event.box_size))
    periodic_shifts = calculate_shifts(mesh_axis, tiling_data)
    mappable = mesh_plot.plot_deformed_periodic_energy_change_background(
        mesh_axis,
        state2,
        energy_change,
        geometry,
        load=event.target_load,
        box_size=event.box_size,
        periodic_shifts=periodic_shifts,
        symmetric_limit=energy_limit,
        rasterized=True,
    )
    mesh_axis.set_title(r"(b) $E^{(0)}-E^{(2)}$")
    colorbar = energy_figure.colorbar(
        mappable, cax=energy_change_colorbar_axis, orientation="horizontal"
    )
    colorbar.set_label(r"$\Delta E$")
    colorbar.formatter = _plain_tick_formatter()
    colorbar.update_ticks()
    for axis in (new_energy_axis, mesh_axis):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])

    energy_png = output_directory / f"{stem}_energy.png"
    energy_figure.savefig(energy_png, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(energy_figure)

    # Poincare figure: the full and reduced disks in one row.
    disk_figure = plt.figure(figsize=(14.0, 7.0))
    disk_axis = disk_figure.add_axes((0.040, 0.18, 0.44, 0.72))
    reduced_disk_axis = disk_figure.add_axes((0.520, 0.18, 0.44, 0.72))
    density_colorbar_axis = disk_figure.add_axes((0.040, 0.060, 0.44, 0.022))
    vector_colorbar_axis = disk_figure.add_axes((0.520, 0.060, 0.44, 0.022))
    density_scatter, density_vmax = _plot_full_poincare_disk(
        disk_axis,
        event.state_paths.state2_relaxed_gamma_plus,
        load_increment=event.load_increment,
    )
    density_colorbar = disk_figure.colorbar(
        density_scatter,
        cax=density_colorbar_axis,
        orientation="horizontal",
    )
    disk_axis.set_title(r"(a) $\mathbf{T}^{\mathsf{T}}\mathbf{T}$")
    reduced_point_count = _plot_almost_plastically_reduced_transition(
        reduced_disk_axis,
        event.state_paths.state0_min_gamma,
        event.state_paths.state2_relaxed_gamma_plus,
        load_increment=event.load_increment,
        density_vmax=density_vmax,
        vector_colorbar_axis=vector_colorbar_axis,
    )
    density_bin_width_c = 2.0 / DENSITY_GRID_SIZE
    density_bin_width_d = 1.0 / DENSITY_GRID_SIZE
    density_colorbar.set_label(
        "Bin counts "
        f"(N_c={int(density_vmax)}, N_d={reduced_point_count}; "
        f"{DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE}; "
        f"bin width={density_bin_width_c:g} in a, {density_bin_width_d:g} in b)"
    )
    reduced_disk_axis.set_title(r"(b) reduced $\mathbf{T}^{\mathsf{T}}\mathbf{T}$")

    disk_png = output_directory / f"{stem}_disks.png"
    disk_figure.savefig(disk_png, dpi=dpi)
    plt.close(disk_figure)
    three_arrows_meshes_png = _render_three_arrow_mesh_figure(
        event,
        state0,
        state2,
        output_directory,
        density_vmax=density_vmax,
        dpi=dpi,
    )
    return energy_png, disk_png, three_arrows_meshes_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--top", type=int, default=5, help="Number of largest energy drops to render.")
    parser.add_argument(
        "--saved-only",
        action="store_true",
        help="Preview from complete sampled state folders; do not use this for the global top events.",
    )
    parser.add_argument(
        "--output-directory", type=Path,
        default=ROOT / "Plots" / "reconnecting_largest_energy_events",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")

    events = select_events(args.job, number=args.top, saved_only=args.saved_only)
    manifest_rows = []
    for event in events:
        energy_output, disk_output, three_arrows_meshes_output = render_event(
            event, args.output_directory, dpi=args.dpi
        )
        manifest_rows.append({
            "global_rank": event.rank,
            "load_step": event.load_step,
            "start_load": event.start_load,
            "target_load": event.target_load,
            "equilibrium_energy_drop": event.energy_drop,
            "state_directory": str(event.state_directory),
            "output_energy_png": str(energy_output),
            "output_disks_png": str(disk_output),
            "output_three_arrows_meshes_png": str(three_arrows_meshes_output),
        })
        print(energy_output)
        print(disk_output)
        print(three_arrows_meshes_output)
    manifest_path = args.output_directory / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(manifest_path)


if __name__ == "__main__":
    main()

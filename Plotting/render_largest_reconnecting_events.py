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
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib import patheffects
from matplotlib.ticker import ScalarFormatter

from MTMath.energyFunction import ContiEnergy, F_from_C
from MTMath.poincareEnergy import (
    C2PoincareDisk,
    approximate_ellipticity_boundary,
    drawCScatter,
    poincareDisk2C,
    prepPoincareFig,
)
from MTMath.reduction import plastic_reduction
from Plotting import meshEventPlotting as mesh_plot
from Plotting.dataFunctions import VTUData
from Plotting.pyplotFunctions import (
    add_padding,
    calculate_shifts,
    draw_periodic_element_outlines,
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
POINCARE_ARROW_COUNT = 50
POINCARE_ARROW_LENGTH_SCALE = 0.245
ELLIPTICITY_BOUNDARY_RESOLUTION = 320
ELLIPTICITY_BOUNDARY_ANGLES = 100
ELLIPTICITY_STRESS_SAMPLES = 2000


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


@dataclass(frozen=True)
class PoincareTransition:
    """Single-element Poincare transition data for one reconnection event."""

    after_metric: np.ndarray
    before_x: np.ndarray
    before_y: np.ndarray
    after_x: np.ndarray
    after_y: np.ndarray
    delta_T_frobenius: np.ndarray


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


def _require_edge_flip_slot_correspondence(job_directory: Path) -> None:
    """Reject jobs whose VTU cell order is not the MTS2D edge-flip identity."""

    config_path = Path(job_directory) / "config.conf"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Edge-flip element correspondence requires the job config: "
            f"{config_path}."
        )
    match = re.search(
        r"^\s*reconnectionMethod\s*=\s*(?P<method>\S+)\s*$",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(
            f"Could not find reconnectionMethod in {config_path}."
        )
    if match.group("method") != "edgeFlip":
        raise ValueError(
            "This renderer identifies elements through MTS2D edge-flip slots; "
            f"{config_path} specifies {match.group('method')!r}."
        )


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


def select_events(
    job_directory: Path, *, number: int | None, saved_only: bool
) -> list[SelectedEvent]:
    """Select the largest drops, never replacing missing global selections silently."""

    if number is not None and number <= 0:
        raise ValueError("number must be positive.")
    if number is None and not saved_only:
        raise ValueError("Selecting every event is only defined for --saved-only.")
    _require_edge_flip_slot_correspondence(job_directory)
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
            if (
                not saved_only
                and number is not None
                and len(selected) + len(missing) == number
            ):
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
        if number is not None and len(selected) == number:
            break

    if saved_only:
        if number is not None and len(selected) < number:
            raise RuntimeError(f"Only {len(selected)} complete saved protocol events are available; requested {number}.")
        if not selected:
            raise RuntimeError("No complete saved protocol events are available.")
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
    if number is None or len(selected) != number:
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


def _total_T(path: Path, *, load_increment: float) -> np.ndarray:
    """Return one total-``T`` matrix for every serialized mesh element."""

    data = VTUData(path, load_increment=load_increment)
    T = np.asarray(data.get_T(), dtype=float)
    if T.ndim != 3 or T.shape[1:] != (2, 2):
        raise ValueError(f"Expected one 2x2 T per element in {path}, got {T.shape}.")

    triangles = np.asarray(data.mesh.cells_dict.get("triangle"), dtype=int)
    if triangles.shape != (len(T), 3):
        raise ValueError(f"Triangle/matrix count mismatch in {path}.")
    return T


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

    after_T = _total_T(after_path, load_increment=load_increment)
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


def _loss_of_ellipticity_boundary() -> np.ndarray:
    """Return the longest resolved loss-of-ellipticity curve in disk coordinates."""

    boundary = np.asarray(
        approximate_ellipticity_boundary(
            E_func=ContiEnergy,
            beta=-0.25,
            K=4,
            resolution=ELLIPTICITY_BOUNDARY_RESOLUTION,
            n_angles=ELLIPTICITY_BOUNDARY_ANGLES,
            zoom=1,
            refine_axis_extrema=True,
            refinement_n_angles=720,
        ),
        dtype=float,
    )
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 3:
        raise ValueError("Could not resolve a loss-of-ellipticity boundary curve.")
    if not np.all(np.isfinite(boundary)):
        raise ValueError("The loss-of-ellipticity boundary contains non-finite points.")
    return boundary


def _draw_poincare_loss_of_ellipticity_limit(
    axis: plt.Axes,
    boundary: np.ndarray,
    *,
    zoom: float,
) -> None:
    """Overlay the fundamental-domain ellipticity limit on a disk axis."""

    scale = zoom * DISK_GRID_SIZE / 2.0
    center = DISK_GRID_SIZE / 2.0
    line = axis.plot(
        center + scale * boundary[:, 0],
        center + scale * boundary[:, 1],
        color="#d62728",
        linewidth=1.3,
        zorder=6,
        label="loss of ellipticity",
    )[0]
    line.set_path_effects(
        [patheffects.withStroke(linewidth=2.5, foreground="white", alpha=0.75)]
    )


def _stress_coordinates_from_vtu(path: Path, *, load_increment: float) -> tuple[np.ndarray, np.ndarray]:
    """Read Cauchy stress coordinates from one new-equilibrium VTU."""

    data = VTUData(path, load_increment=load_increment)
    sigma12 = np.asarray(data.get_cell_data("sigma12"), dtype=float).reshape(-1)
    sigma11 = np.asarray(data.get_cell_data("sigma11"), dtype=float).reshape(-1)
    sigma22 = np.asarray(data.get_cell_data("sigma22"), dtype=float).reshape(-1)
    if not (sigma12.shape == sigma11.shape == sigma22.shape):
        raise ValueError(f"Cauchy stress components have incompatible shapes in {path}.")
    normal_difference = 0.5 * (sigma22 - sigma11)
    if not np.all(np.isfinite(sigma12)) or not np.all(np.isfinite(normal_difference)):
        raise ValueError(f"Cauchy stress contains non-finite values in {path}.")
    return sigma12, normal_difference


def _resample_closed_curve(curve: np.ndarray, sample_count: int) -> np.ndarray:
    """Resample a 2-D closed curve at uniformly spaced arc-length positions."""

    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3:
        raise ValueError("A closed curve must contain at least three 2-D points.")
    if not np.all(np.isfinite(curve)):
        raise ValueError("Cannot resample a curve containing non-finite points.")
    if not isinstance(sample_count, int) or sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three.")

    closed = np.vstack([curve, curve[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= 0:
        raise ValueError("The ellipticity boundary has zero arc length.")
    targets = np.linspace(0.0, total_length, sample_count, endpoint=False)
    return np.column_stack(
        [
            np.interp(targets, cumulative, closed[:, coordinate])
            for coordinate in range(2)
        ]
    )


def _stress_loss_of_ellipticity_limit(
    boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the disk boundary and return its stress-plot coordinates.

    ``approximate_ellipticity_boundary`` returns points in Poincare coordinates;
    those coordinates must not be plotted directly on the stress axes.  Each
    sampled point is instead converted back to the metric ``C``, reconstructed
    as ``F``, and evaluated with the same Cauchy-stress routine used for the
    VTU elements.  The return order is deliberately the stress-plot order:
    ``x=(sigma22-sigma11)/2`` and ``y=sigma12``.
    """

    disk_samples = _resample_closed_curve(boundary, ELLIPTICITY_STRESS_SAMPLES)
    C_boundary = poincareDisk2C(disk_samples[:, 0], disk_samples[:, 1])
    F_boundary = F_from_C(C_boundary)
    sigma = np.asarray(ContiEnergy.cauchy_from_F(F_boundary), dtype=float)
    if sigma.shape != (ELLIPTICITY_STRESS_SAMPLES, 2, 2):
        raise ValueError(
            "The sampled ellipticity boundary produced an unexpected stress shape."
        )
    stress_x = 0.5 * (sigma[:, 1, 1] - sigma[:, 0, 0])
    stress_y = sigma[:, 0, 1]
    if not np.all(np.isfinite(stress_x)) or not np.all(np.isfinite(stress_y)):
        raise ValueError("The sampled stress-space ellipticity limit is non-finite.")
    return stress_x, stress_y


def _accumulate_poincare_density_histogram(
    histogram: np.ndarray, x: np.ndarray, y: np.ndarray, *, zoom: float
) -> int:
    """Add one batch of disk points to a fixed zoomed density histogram."""

    histogram = np.asarray(histogram)
    if histogram.shape != (DENSITY_GRID_SIZE, DENSITY_GRID_SIZE):
        raise ValueError("The Poincare density histogram has an unexpected shape.")
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Poincare density coordinates must have matching shapes.")
    radius = 1.0 / zoom
    valid = np.isfinite(x) & np.isfinite(y) & (x * x + y * y <= radius * radius)
    if not np.any(valid):
        return 0
    scale = DENSITY_GRID_SIZE / (2.0 * radius)
    ix = np.clip(((x[valid] + radius) * scale).astype(int), 0, DENSITY_GRID_SIZE - 1)
    iy = np.clip(((y[valid] + radius) * scale).astype(int), 0, DENSITY_GRID_SIZE - 1)
    np.add.at(histogram, (iy, ix), 1)
    return int(np.count_nonzero(valid))


def _draw_poincare_density_histogram(
    axis: plt.Axes, histogram: np.ndarray, *, zoom: float
) -> object:
    """Draw a logarithmic, all-event density image in Poincare plot coordinates."""

    if not np.any(histogram > 0):
        raise RuntimeError("The pooled new-equilibrium total-T cloud is empty.")

    return _draw_poincare_grid_image(
        axis,
        np.ma.masked_equal(histogram, 0),
        zoom=zoom,
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=max(2, int(np.max(histogram)))),
        alpha=0.70,
        zorder=4,
    )


def _draw_poincare_grid_image(
    axis: plt.Axes,
    values: np.ndarray | np.ma.MaskedArray,
    *,
    zoom: float,
    cmap: str | object,
    norm: Normalize | LogNorm | None,
    alpha: float = 1.0,
    zorder: int = 4,
) -> object:
    """Draw a Poincare-coordinate grid with the density-plot zoom convention.

    The image grid spans ``[-1/zoom, 1/zoom]`` in Poincare coordinates, while
    the axis always spans the fixed ``DISK_GRID_SIZE`` canvas used by
    :func:`prepPoincareFig`.  Keeping this conversion here ensures scalar
    heatmaps and density histograms occupy precisely the same disk area.
    """

    values = np.ma.asarray(values)
    if values.ndim != 2:
        raise ValueError("A Poincare grid image must be two-dimensional.")
    if not np.isfinite(zoom) or zoom < 1:
        raise ValueError("Poincare zoom must be finite and at least one.")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie between zero and one.")
    scale = zoom * DISK_GRID_SIZE / 2.0
    center = DISK_GRID_SIZE / 2.0
    radius = 1.0 / zoom
    return axis.imshow(
        values,
        origin="lower",
        extent=(center - scale * radius, center + scale * radius,
                center - scale * radius, center + scale * radius),
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=zorder,
    )


def _draw_stress_density_histogram(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
) -> tuple[object, int]:
    """Draw a 200x200 logarithmic stress density image with white empty bins."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Stress density coordinates must have matching shapes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Stress density coordinates must be finite.")
    xmin, xmax, ymin, ymax = extent
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Stress density extent must have increasing bounds.")
    histogram, _, _ = np.histogram2d(
        x,
        y,
        bins=(DENSITY_GRID_SIZE, DENSITY_GRID_SIZE),
        range=((xmin, xmax), (ymin, ymax)),
    )
    histogram = histogram.astype(int, copy=False)
    if not np.any(histogram > 0):
        raise RuntimeError("The stress cloud is empty.")
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")
    image = axis.imshow(
        np.ma.masked_equal(histogram.T, 0),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(2, int(np.max(histogram)))),
        zorder=1,
    )
    return image, int(np.count_nonzero(histogram))


def _render_pooled_poincare_flow(
    output_path: Path,
    *,
    zoom: float,
    density_histogram: np.ndarray,
    visible_density_count: int,
    total_element_count: int,
    before_x: np.ndarray,
    before_y: np.ndarray,
    after_x: np.ndarray,
    after_y: np.ndarray,
    delta_T_frobenius: np.ndarray,
    boundary: np.ndarray,
    dpi: int,
    direction_split_otsu: bool = False,
) -> Path:
    """Render one pooled Poincare flow view at a requested zoom.

    ``direction_split_otsu`` uses the shared circular-angle Otsu split within
    every populated spatial bin, allowing the bin to contribute two quivers.
    """

    flow_figure = plt.figure(figsize=(10.0, 9.4))
    flow_axis = flow_figure.add_axes((0.08, 0.17, 0.84, 0.76))
    density_colorbar_axis = flow_figure.add_axes((0.10, 0.095, 0.38, 0.022))
    vector_colorbar_axis = flow_figure.add_axes((0.54, 0.095, 0.38, 0.022))
    _prepare_poincare_axis(flow_axis, zoom=zoom)
    density_image = _draw_poincare_density_histogram(
        flow_axis, density_histogram, zoom=zoom
    )
    _draw_poincare_loss_of_ellipticity_limit(flow_axis, boundary, zoom=zoom)
    plot_binned_poincare_displacement_field(
        flow_axis,
        before_x,
        before_y,
        after_x - before_x,
        after_y - before_y,
        grid_size=DISK_GRID_SIZE,
        zoom=zoom,
        bins=30,
        min_count=50,
        min_coherence=0.0,
        color_values=delta_T_frobenius,
        colorbar_axes=vector_colorbar_axis,
        colorbar_label=r"mean $\|\Delta\mathbf{T}\|_F$",
        min_vector_length=0.0,
        colorbar_log=direction_split_otsu,
        vector_length_from_color=False,
        vector_length_scale=2.5 if direction_split_otsu else 5.0,
        arrow_width=0.003,
        arrow_headwidth=2.8,
        arrow_headlength=3.4,
        direction_split_otsu=direction_split_otsu,
    )
    density_colorbar = flow_figure.colorbar(
        density_image, cax=density_colorbar_axis, orientation="horizontal"
    )
    density_colorbar.set_label(
        "Bin counts "
        f"(N={total_element_count}; N_{{visible}}={visible_density_count}; "
        f"{DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE})"
    )
    flow_axis.legend(loc="upper left", frameon=True)
    flow_figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(flow_figure)
    return output_path


def render_aggregate_largest_event_summary(
    events: list[SelectedEvent],
    output_directory: Path,
    *,
    dpi: int,
    direction_split_otsu: bool = False,
) -> tuple[Path, Path, Path]:
    """Pool saved events into Poincare-flow and stress-space figures."""

    if not events:
        raise ValueError("At least one saved event is required for an aggregate plot.")
    output_directory.mkdir(parents=True, exist_ok=True)
    boundary = _loss_of_ellipticity_boundary()
    number = len(events)
    stress_limit_x, stress_limit_y = _stress_loss_of_ellipticity_limit(boundary)
    full_disk_zoom = 1.0
    zoomed_disk_zoom = 2.0
    full_density_histogram = np.zeros(
        (DENSITY_GRID_SIZE, DENSITY_GRID_SIZE), dtype=int
    )
    zoomed_density_histogram = np.zeros(
        (DENSITY_GRID_SIZE, DENSITY_GRID_SIZE), dtype=int
    )
    before_x_parts = []
    before_y_parts = []
    after_x_parts = []
    after_y_parts = []
    delta_T_parts = []
    sigma_xy_parts = []
    normal_difference_parts = []
    total_element_count = 0
    full_visible_density_count = 0
    zoomed_visible_density_count = 0
    for event_index, event in enumerate(events):
        transition = _single_element_poincare_transition(
            event.state_paths.state0_min_gamma,
            event.state_paths.state2_relaxed_gamma_plus,
            load_increment=event.load_increment,
        )
        sigma_xy, normal_difference = _stress_coordinates_from_vtu(
            event.state_paths.state2_relaxed_gamma_plus,
            load_increment=event.load_increment,
        )
        if len(sigma_xy) != len(transition.after_metric):
            raise ValueError(
                "The new-equilibrium stress and total-T arrays do not have the same "
                f"number of elements for {event.state_directory}."
            )
        total_element_count += len(transition.after_metric)
        full_visible_density_count += _accumulate_poincare_density_histogram(
            full_density_histogram,
            transition.after_x,
            transition.after_y,
            zoom=full_disk_zoom,
        )
        zoomed_visible_density_count += _accumulate_poincare_density_histogram(
            zoomed_density_histogram,
            transition.after_x,
            transition.after_y,
            zoom=zoomed_disk_zoom,
        )
        before_x_parts.append(np.asarray(transition.before_x, dtype=np.float32))
        before_y_parts.append(np.asarray(transition.before_y, dtype=np.float32))
        after_x_parts.append(np.asarray(transition.after_x, dtype=np.float32))
        after_y_parts.append(np.asarray(transition.after_y, dtype=np.float32))
        delta_T_parts.append(
            np.asarray(transition.delta_T_frobenius, dtype=np.float32)
        )
        sigma_xy_parts.append(np.asarray(sigma_xy, dtype=np.float32))
        normal_difference_parts.append(np.asarray(normal_difference, dtype=np.float32))
    before_x = np.concatenate(before_x_parts)
    before_y = np.concatenate(before_y_parts)
    after_x = np.concatenate(after_x_parts)
    after_y = np.concatenate(after_y_parts)
    delta_T_frobenius = np.concatenate(delta_T_parts)
    sigma_xy = np.concatenate(sigma_xy_parts)
    normal_difference = np.concatenate(normal_difference_parts)
    del (
        before_x_parts,
        before_y_parts,
        after_x_parts,
        after_y_parts,
        delta_T_parts,
        sigma_xy_parts,
        normal_difference_parts,
    )
    if not (
        len(before_x)
        == len(after_x)
        == len(delta_T_frobenius)
        == total_element_count
        == len(sigma_xy)
        == len(normal_difference)
    ):
        raise ValueError("The pooled Poincare transition arrays have inconsistent sizes.")

    flow_label = "poincare_otsu_flow" if direction_split_otsu else "poincare_mean_flow"
    flow_png = _render_pooled_poincare_flow(
        output_directory / f"aggregate_all{number:03d}_{flow_label}.png",
        zoom=full_disk_zoom,
        density_histogram=full_density_histogram,
        visible_density_count=full_visible_density_count,
        total_element_count=total_element_count,
        before_x=before_x,
        before_y=before_y,
        after_x=after_x,
        after_y=after_y,
        delta_T_frobenius=delta_T_frobenius,
        boundary=boundary,
        dpi=dpi,
        direction_split_otsu=direction_split_otsu,
    )
    zoomed_flow_png = _render_pooled_poincare_flow(
        output_directory / f"aggregate_all{number:03d}_{flow_label}_zoomed.png",
        zoom=zoomed_disk_zoom,
        density_histogram=zoomed_density_histogram,
        visible_density_count=zoomed_visible_density_count,
        total_element_count=total_element_count,
        before_x=before_x,
        before_y=before_y,
        after_x=after_x,
        after_y=after_y,
        delta_T_frobenius=delta_T_frobenius,
        boundary=boundary,
        dpi=dpi,
        direction_split_otsu=direction_split_otsu,
    )

    stress_figure, stress_axis = plt.subplots(figsize=(9.5, 8.2))
    scatter_extent = max(
        float(np.max(np.abs(normal_difference))),
        float(np.max(np.abs(sigma_xy))),
    )
    if not np.isfinite(scatter_extent) or scatter_extent <= 0:
        raise ValueError("The stress scatter has no finite nonzero extent.")
    scatter_extent *= 1.02
    stress_extent = (-scatter_extent, scatter_extent, -scatter_extent, scatter_extent)
    stress_density, visible_stress_bins = _draw_stress_density_histogram(
        stress_axis,
        normal_difference,
        sigma_xy,
        extent=stress_extent,
    )
    stress_axis.plot(
        stress_limit_x,
        stress_limit_y,
        color="#d62728",
        linewidth=1.5,
        label="loss of ellipticity",
        zorder=4,
    )
    stress_axis.set_xlabel(r"$(\sigma_{22}-\sigma_{11})/2$")
    stress_axis.set_ylabel(r"$\sigma_{xy}$")
    stress_axis.set_xlim(-scatter_extent, scatter_extent)
    stress_axis.set_ylim(-scatter_extent, scatter_extent)
    stress_axis.set_aspect("equal", adjustable="box")
    stress_colorbar = stress_figure.colorbar(stress_density, ax=stress_axis)
    stress_colorbar.set_label(
        f"Bin counts (N={total_element_count}; "
        f"nonzero bins={visible_stress_bins}; {DENSITY_GRID_SIZE}x{DENSITY_GRID_SIZE})"
    )
    stress_axis.legend(loc="best", frameon=True)
    stress_figure.tight_layout()
    stress_png = output_directory / f"aggregate_all{number:03d}_stress_scatter.png"
    stress_figure.savefig(stress_png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(stress_figure)
    return flow_png, zoomed_flow_png, stress_png


def _single_element_poincare_transition(
    before_path: Path,
    after_path: Path,
    *,
    load_increment: float,
) -> PoincareTransition:
    """Return the total-``T`` transition for persistent edge-flip slots.

    The simulation VTU field named ``T`` is the plastic component ``T_p`` in
    the present notation.  ``_total_T`` combines it with ``F_e``.  We reduce
    only the previous equilibrium and then advance it with the total-T change;
    the advanced point is deliberately not reduced again.
    """

    before_T = _total_T(before_path, load_increment=load_increment)
    after_T = _total_T(after_path, load_increment=load_increment)
    if before_T.shape != after_T.shape:
        raise ValueError(
            "Edge-flip slot correspondence requires the same number of elements "
            "before and after reconnection."
        )
    before_metric = _metric_from_total_T(before_T, source=before_path)
    element_indices = np.arange(len(before_T), dtype=int)
    reduced_before_metric, reduction_M = plastic_reduction(
        before_metric[element_indices], compute_M=True
    )
    # Delta T = T_after @ inv(T_before) - I.  Solving the transposed system
    # evaluates the right multiplication without explicitly inverting T_before.
    try:
        before_transpose = np.swapaxes(before_T[element_indices], -1, -2)
        after_transpose = np.swapaxes(after_T[element_indices], -1, -2)
        total_T_increment = np.linalg.solve(before_transpose, after_transpose)
        total_T_increment = np.swapaxes(total_T_increment, -1, -2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not solve for the total-T increment.") from exc
    delta_T_frobenius = np.linalg.norm(
        total_T_increment - np.eye(2), axis=(-2, -1)
    )
    if not np.all(np.isfinite(delta_T_frobenius)):
        raise ValueError("The total-T transformation has non-finite Frobenius norms.")
    advanced_after_metric = _metric_from_total_T(
        after_T[element_indices] @ reduction_M, source=after_path
    )
    before_x, before_y = _disk_coordinates(reduced_before_metric)
    after_x, after_y = _disk_coordinates(advanced_after_metric)
    return PoincareTransition(
        after_metric=advanced_after_metric,
        before_x=before_x,
        before_y=before_y,
        after_x=after_x,
        after_y=after_y,
        delta_T_frobenius=delta_T_frobenius,
    )


def _plot_single_element_plastically_reduced_transition(
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
    """Plot the 50 largest single-element total-``T`` transitions.

    For each persistent edge-flip element slot, ``M`` reduces only the previous
    equilibrium: ``T_before_reduced = T_before @ M``.  The new point is then
    the direct image under ``T_after @ inv(T_before)``.  It is intentionally not
    reduced a second time, even if this final operation leaves the fundamental
    elastic well.  No spatial binning or directional averaging is performed:
    arrows are the individual element transitions with the 50 largest
    ``||Delta_T||_F``, where ``Delta_T = T_after @ inv(T_before) - I``.
    """

    transition = _single_element_poincare_transition(
        before_path, after_path, load_increment=load_increment
    )
    # MTS2D's Mesh::flipEdge reconstructs the changed pair in the original
    # ``e1i``/``e2i`` slots, and writeMeshToVtu serializes cells in that same
    # ``elementIndex`` order.  The node-triplet is intentionally *not* an
    # identity here: it changes exactly for the elements affected by a flip.
    advanced_after_metric = transition.after_metric
    before_x, before_y = transition.before_x, transition.before_y
    after_x, after_y = transition.after_x, transition.after_y
    delta_T_frobenius = transition.delta_T_frobenius
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
    if len(delta_T_frobenius) < POINCARE_ARROW_COUNT:
        raise ValueError(
            f"At least {POINCARE_ARROW_COUNT} element transitions are required; "
            f"found {len(delta_T_frobenius)}."
        )
    selected_indices = np.argsort(delta_T_frobenius, kind="stable")[
        -POINCARE_ARROW_COUNT:
    ][::-1]
    selected_color_values = delta_T_frobenius[selected_indices]

    # Keep the established matrix-based arrow-length convention, but use each
    # element's own direction.  The calibration puts ||Delta T|| and the
    # Poincare displacement on the same typical scale.  The current scale is
    # half of the previous 0.49 display factor.
    selected_disk_lengths = np.hypot(dx[selected_indices], dy[selected_indices])
    nonzero = selected_disk_lengths > 0
    if not np.any(nonzero):
        raise ValueError(
            f"The {POINCARE_ARROW_COUNT} largest Delta-T elements have no "
            "Poincare displacement."
        )
    length_calibration = float(
        np.nanmedian(
            selected_disk_lengths[nonzero] / selected_color_values[nonzero]
        )
    )
    if not np.isfinite(length_calibration) or length_calibration <= 0:
        raise ValueError("Could not calibrate single-element matrix-based vector lengths.")
    arrow_lengths = (
        selected_color_values * length_calibration * POINCARE_ARROW_LENGTH_SCALE
    )
    arrow_u = np.divide(
        dx[selected_indices],
        selected_disk_lengths,
        out=np.zeros_like(selected_disk_lengths),
        where=nonzero,
    ) * arrow_lengths
    arrow_v = np.divide(
        dy[selected_indices],
        selected_disk_lengths,
        out=np.zeros_like(selected_disk_lengths),
        where=nonzero,
    ) * arrow_lengths

    color_limit = float(np.max(selected_color_values))
    color_floor = float(np.min(selected_color_values))
    if not np.isfinite(color_floor) or not np.isfinite(color_limit):
        raise ValueError("The selected Delta-T norms must be finite.")
    if color_limit == color_floor:
        color_limit = float(np.nextafter(color_floor, np.inf))
    color_norm = Normalize(vmin=color_floor, vmax=color_limit)
    color_map = plt.get_cmap("viridis")
    arrow_colors = color_map(color_norm(selected_color_values))
    arrow_colors[:, 3] = 1.0

    arrow_metadata = []
    for metadata_index, element_index in enumerate(selected_indices):
        record = {
            "x": float(before_x[element_index]),
            "y": float(before_y[element_index]),
            "u": float(arrow_u[metadata_index]),
            "v": float(arrow_v[metadata_index]),
            "length": float(arrow_lengths[metadata_index]),
            "color_value": float(selected_color_values[metadata_index]),
            "color": np.asarray(arrow_colors[metadata_index], dtype=float).copy(),
            "before_element_indices": np.asarray([element_index], dtype=int),
            "after_element_indices": np.asarray([element_index], dtype=int),
        }
        arrow_metadata.append(record)
    if draw_arrows:
        _draw_selected_poincare_arrows(
            axis,
            arrow_metadata,
            [record["color"] for record in arrow_metadata],
            zoom=zoom,
        )
    if show_vector_colorbar:
        mappable = ScalarMappable(norm=color_norm, cmap=color_map)
        mappable.set_array(selected_color_values)
        if vector_colorbar_axis is None:
            axis.figure.colorbar(
                mappable,
                ax=axis,
                orientation="horizontal",
                label=r"$\|\Delta\mathbf{T}\|_F$",
            )
        else:
            axis.figure.colorbar(
                mappable,
                cax=vector_colorbar_axis,
                orientation="horizontal",
                label=r"$\|\Delta\mathbf{T}\|_F$",
            )
    result = len(advanced_after_metric)
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
    for arrow_index in sorted(
        range(len(arrows)), key=lambda index: arrows[index]["length"]
    ):
        overlay = axis.quiver(
            [center + scale * arrows[arrow_index]["x"]],
            [center + scale * arrows[arrow_index]["y"]],
            [scale * arrows[arrow_index]["u"]],
            [scale * arrows[arrow_index]["v"]],
            color=[colors[arrow_index]],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="tail",
            width=0.006,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.2,
            zorder=8,
        )
        overlay.set_path_effects(
            [patheffects.withStroke(linewidth=0.8, foreground="white")]
        )
    return quiver


def _forward_mesh_displacement_field(
    event: SelectedEvent,
    state0: mesh_plot.MeshState,
    state2: mesh_plot.MeshState,
) -> tuple[mesh_plot.MeshState, np.ndarray, float]:
    """Load the protocol displacement field and its typical element length."""

    states = {
        "state0_min_gamma": state0,
        "state1_affine_gamma_plus": mesh_plot.load_mesh_state(
            event.state_paths.state1_affine_gamma_plus
        ),
        "state2_relaxed_gamma_plus": state2,
        "state3_affine_gamma_minus": mesh_plot.load_mesh_state(
            event.state_paths.state3_affine_gamma_minus
        ),
        "state4_relaxed_gamma": mesh_plot.load_mesh_state(
            event.state_paths.state4_relaxed_gamma
        ),
    }
    displacements = mesh_plot.calculate_event_displacements(
        states,
        periodic_vectors=(
            np.array([event.box_size, 0.0]),
            np.array([event.target_load * event.box_size, event.box_size]),
        ),
    )
    state1 = states["state1_affine_gamma_plus"]
    triangles = state2.triangles
    edge_points = state2.points[triangles[:, [0, 1, 1, 2]].reshape(-1)]
    edge_lengths = np.linalg.norm(edge_points[0::2] - edge_points[1::2], axis=1)
    finite_edges = edge_lengths[np.isfinite(edge_lengths) & (edge_lengths > 0)]
    if finite_edges.size == 0:
        raise ValueError("Cannot determine a finite mesh element length for arrows.")
    return (
        state1,
        displacements.forward_relaxation,
        float(np.median(finite_edges)),
    )


def _periodic_relative_vectors(
    point: np.ndarray,
    center: np.ndarray,
    *,
    load: float,
    box_size: float,
) -> np.ndarray:
    """Return nearby periodic images of ``point - center`` in the sheared cell."""

    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    if point.shape != (2,) or center.shape != (2,):
        raise ValueError("Periodic window positions must be two-dimensional.")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(center)):
        raise ValueError("Periodic window positions must be finite.")
    if not np.isfinite(load) or not np.isfinite(box_size) or box_size <= 0:
        raise ValueError("load must be finite and box_size must be positive.")

    delta = point - center
    a = np.array([box_size, 0.0])
    b = np.array([load * box_size, box_size])
    nearest_j = int(np.rint(delta[1] / box_size))
    relative = []
    for j in range(nearest_j - 2, nearest_j + 3):
        nearest_i = int(np.rint((delta[0] - j * b[0]) / box_size))
        for i in range(nearest_i - 2, nearest_i + 3):
            relative.append(delta - i * a - j * b)
    return np.asarray(relative, dtype=float)


def _group_arrows_into_periodic_windows(
    arrows: list[dict],
    after_centres: np.ndarray,
    *,
    load: float,
    box_size: float,
    half_window: float = 7.5,
    minimum_separation_in_window_widths: float = 2.0,
    window_count: int = 3,
) -> list[dict]:
    """Assign descending single-element arrows to separated local windows.

    The largest arrow anchors the first window.  Before all requested windows
    exist, an arrow inside a current window joins its nearest matching window;
    an outside arrow creates a new window only when its periodic distance from
    every existing center is at least the requested number of full window
    widths.  Too-close outside arrows are discarded.  Once all windows exist,
    the first arrow outside every window ends the scan.
    """

    after_centres = np.asarray(after_centres, dtype=float)
    if after_centres.ndim != 2 or after_centres.shape[1] != 2:
        raise ValueError("after_centres must have shape (number of elements, 2).")
    if not np.all(np.isfinite(after_centres)):
        raise ValueError("after_centres must be finite.")
    if not arrows:
        raise ValueError("At least one arrow is required for window grouping.")
    if not np.isfinite(half_window) or half_window <= 0:
        raise ValueError("half_window must be finite and positive.")
    if (
        not np.isfinite(minimum_separation_in_window_widths)
        or minimum_separation_in_window_widths <= 0
    ):
        raise ValueError("Window separation must be finite and positive.")
    if not isinstance(window_count, int) or window_count <= 0:
        raise ValueError("window_count must be a positive integer.")

    sorted_arrows = sorted(
        arrows, key=lambda arrow: float(arrow["color_value"]), reverse=True
    )
    full_window_width = 2.0 * half_window
    minimum_center_distance = (
        minimum_separation_in_window_widths * full_window_width
    )
    windows: list[dict] = []
    for arrow in sorted_arrows:
        element_indices = np.asarray(
            arrow["after_element_indices"], dtype=int
        ).reshape(-1)
        if element_indices.shape != (1,):
            raise ValueError(
                "Single-element window grouping requires one after element per arrow."
            )
        element_index = int(element_indices[0])
        if element_index < 0 or element_index >= len(after_centres):
            raise IndexError("Arrow element index lies outside after_centres.")
        position = after_centres[element_index]

        distances = []
        visible_windows = []
        for window_index, window in enumerate(windows):
            relative = _periodic_relative_vectors(
                position,
                window["center"],
                load=load,
                box_size=box_size,
            )
            distances.append(float(np.min(np.linalg.norm(relative, axis=1))))
            visible = np.any(
                (np.abs(relative[:, 0]) <= half_window)
                & (np.abs(relative[:, 1]) <= half_window)
            )
            if visible:
                visible_windows.append(window_index)

        if visible_windows:
            nearest_visible = min(
                visible_windows, key=lambda index: distances[index]
            )
            windows[nearest_visible]["arrows"].append(arrow)
            continue
        if len(windows) == window_count:
            break
        if distances and any(
            distance < minimum_center_distance for distance in distances
        ):
            continue
        windows.append(
            {
                "center": np.asarray(position, dtype=float).copy(),
                "arrows": [arrow],
            }
        )

    if len(windows) != window_count:
        raise RuntimeError(
            f"Only {len(windows)} spatially separated windows were found among "
            f"the {len(sorted_arrows)} selected arrows; {window_count} are required."
        )
    return windows


def _window_element_indices(window: dict, key: str) -> np.ndarray:
    """Return the unique element slots assigned to one arrow window."""

    if key not in {"before_element_indices", "after_element_indices"}:
        raise ValueError("Unsupported window element-index key.")
    arrays = [np.asarray(arrow[key], dtype=int).reshape(-1) for arrow in window["arrows"]]
    if not arrays or any(array.shape != (1,) for array in arrays):
        raise ValueError("Every window arrow must correspond to exactly one element.")
    return np.unique(np.concatenate(arrays))


def _render_three_arrow_mesh_figure(
    event: SelectedEvent,
    state0: mesh_plot.MeshState,
    state2: mesh_plot.MeshState,
    output_directory: Path,
    *,
    density_vmax: int,
    dpi: int,
    window_axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> Path:
    """Render three separated arrow windows and their local before/after meshes."""

    if window_axes is not None and len(window_axes) != 2:
        raise ValueError("Exactly two full-mesh axes are required for window outlines.")

    arrow_colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    figure = plt.figure(figsize=(15.0, 11.0))
    disk_axis = figure.add_axes((0.035, 0.18, 0.29, 0.64))

    _point_count, arrow_metadata = _plot_single_element_plastically_reduced_transition(
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
    half_window = 7.5
    after_centres = mesh_plot.periodic_triangle_centres(
        state2, load=event.target_load, box_size=event.box_size
    )
    arrow_windows = _group_arrows_into_periodic_windows(
        arrow_metadata,
        after_centres,
        load=event.target_load,
        box_size=event.box_size,
        half_window=half_window,
        minimum_separation_in_window_widths=2.0,
        window_count=3,
    )

    # ``plot_mesh`` reads the same triangle ordering used by the T arrays.
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

    window_arrows = []
    window_arrow_colors = []
    for window, color in zip(arrow_windows, arrow_colors, strict=True):
        window_arrows.extend(window["arrows"])
        window_arrow_colors.extend([color] * len(window["arrows"]))
    _draw_selected_poincare_arrows(
        disk_axis, window_arrows, window_arrow_colors, zoom=2
    )
    disk_axis.set_title(r"(a) reduced $\mathbf{T}^{\mathsf{T}}\mathbf{T}$")
    disk_axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=f"window {index}")
            for index, color in enumerate(arrow_colors, start=1)
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
    )

    field_state, field_displacement, element_length = _forward_mesh_displacement_field(
        event, state0, state2
    )
    displacement_periodic_vectors = (
        np.array([event.box_size, 0.0]),
        np.array([event.target_load * event.box_size, event.box_size]),
    )
    row_height = 0.27 * 0.70
    row_gap = 0.003
    row_block_height = 3.0 * row_height + 2.0 * row_gap
    row_bottom = 0.50 - 0.5 * row_block_height
    row_y = tuple(
        row_bottom + offset * (row_height + row_gap) for offset in (2, 1, 0)
    )
    for row, (window, color, y_position) in enumerate(
        zip(arrow_windows, arrow_colors, row_y, strict=True), start=1
    ):
        before_indices = _window_element_indices(window, "before_element_indices")
        after_indices = _window_element_indices(window, "after_element_indices")
        if before_indices.size == 0 or after_indices.size == 0:
            raise ValueError("A selected window has no corresponding mesh elements.")
        window_center_x, window_center_y = map(float, window["center"])
        if window_axes is not None:
            for energy_axis in window_axes:
                energy_axis.add_patch(
                    Rectangle(
                        (window_center_x - half_window, window_center_y - half_window),
                        2.0 * half_window,
                        2.0 * half_window,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.05,
                        linestyle=(0, (3, 2)),
                        alpha=0.95,
                        zorder=20,
                    )
                )
        before_labels = np.zeros(len(state0.triangles), dtype=int)
        after_labels = np.zeros(len(state2.triangles), dtype=int)
        before_labels[before_indices] = 1
        after_labels[after_indices] = 1
        mesh_cmap = ListedColormap(["#eeeeee", color])
        mesh_norm = BoundaryNorm(np.array([-0.5, 0.5, 1.5]), mesh_cmap.N)
        # ``plot_mesh`` enforces equal data scaling.  Use square-sized
        # rectangles here so Matplotlib does not shrink wide containers and
        # leave an unintended gap between the before/after panels.
        mesh_axis_width = row_height * figure.get_figheight() / figure.get_figwidth()
        before_axis = figure.add_axes(
            (0.385, y_position, mesh_axis_width, row_height)
        )
        after_axis = figure.add_axes(
            (0.385 + mesh_axis_width + 0.003, y_position, mesh_axis_width, row_height)
        )
        viewport = (
            window_center_x - half_window,
            window_center_x + half_window,
            window_center_y - half_window,
            window_center_y + half_window,
        )
        local_zoom = mesh_plot.ZoomRegion(
            xlim=(viewport[0], viewport[1]),
            ylim=(viewport[2], viewport[3]),
            activity_fraction=1.0,
            center=(window_center_x, window_center_y),
        )
        base_arrow_scale = mesh_plot.choose_zoom_arrow_scale(
            field_state.points,
            field_displacement,
            zoom=local_zoom,
            element_length=element_length,
            target_element_fraction=1.0 / 3.0,
            periodic_vectors=displacement_periodic_vectors,
            reference_indices=field_state.reference_indices,
        )
        if base_arrow_scale.physical_key_length <= 0:
            raise ValueError(
                f"Window {row} contains no nonzero displacement vectors."
            )
        # Preserve the established twofold display amplification, but choose
        # it independently from the vectors visible in each row.
        row_arrow_scale = mesh_plot.ArrowScale(
            amplification=2.0 * base_arrow_scale.amplification,
            physical_key_length=base_arrow_scale.physical_key_length,
            target_element_fraction=base_arrow_scale.target_element_fraction,
        )
        for column, axis, path, load, labels, center_x, center_y in (
            (
                "before",
                before_axis,
                event.state_paths.state0_min_gamma,
                event.start_load,
                before_labels,
                window_center_x,
                window_center_y,
            ),
            (
                "after",
                after_axis,
                event.state_paths.state2_relaxed_gamma_plus,
                event.target_load,
                after_labels,
                window_center_x,
                window_center_y,
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
                unwrap_periodic_triangles=True,
            )
            # Keep the current window's elements filled, but outline any other
            # selected window groups that happen to enter this same local
            # viewport.  The helper uses refIndex-aware unwrapping and sheared
            # PBC tiling, so outlines remain correct at cell boundaries.
            outline_state = state0 if column == "before" else state2
            for other_row, other_color in enumerate(arrow_colors):
                if other_row == row - 1:
                    continue
                other_indices = _window_element_indices(
                    arrow_windows[other_row],
                    "before_element_indices"
                    if column == "before"
                    else "after_element_indices",
                )
                draw_periodic_element_outlines(
                    axis,
                    outline_state,
                    other_indices,
                    load=load,
                    box_size=event.box_size,
                    viewport=viewport,
                    color=other_color,
                    linewidth=0.8,
                    linestyle=(0, (3, 2)),
                    zorder=19,
                )
            mesh_plot.plot_displacement_arrows(
                axis,
                field_state.points,
                field_displacement,
                arrow_scale=row_arrow_scale,
                zoom=local_zoom,
                show_key=(column == "before"),
                key_label=rf"$|\rightarrow| = {row_arrow_scale.physical_key_length:.1e}$",
                rasterized=True,
                periodic_vectors=displacement_periodic_vectors,
                reference_indices=field_state.reference_indices,
            )
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 1:
                axis.set_title(column, fontsize=10, pad=2)

    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / (
        f"rank{event.rank:03d}_gamma{event.target_load:.5f}_three_arrows_local_meshes.png"
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
    new_energy_axis = energy_figure.add_axes((0.030, 0.20, 0.47, mesh_height))
    mesh_axis = energy_figure.add_axes((0.500, 0.20, 0.47, mesh_height))
    # Keep compact bars close to their corresponding mesh without letting the
    # two bars approach one another.
    new_energy_colorbar_axis = energy_figure.add_axes((0.250, 0.14, 0.20, 0.022))
    energy_change_colorbar_axis = energy_figure.add_axes((0.550, 0.14, 0.20, 0.022))
    new_energy_axis.set_anchor("E")
    mesh_axis.set_anchor("W")

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
    reduced_point_count = _plot_single_element_plastically_reduced_transition(
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
        window_axes=(new_energy_axis, mesh_axis),
    )
    energy_png = output_directory / f"{stem}_energy.png"
    energy_figure.savefig(energy_png, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(energy_figure)
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
        "--aggregate-only",
        action="store_true",
        help=(
            "Pool the selected saved events into a mean Poincare-flow figure and "
            "a sigma_xy versus (sigma22-sigma11)/2 scatter, without mesh plots."
        ),
    )
    parser.add_argument(
        "--aggregate-otsu",
        action="store_true",
        help=(
            "Split directions in each pooled Poincare bin with the circular "
            "Otsu cut, allowing up to two branch-mean quivers per bin."
        ),
    )
    parser.add_argument(
        "--all-saved",
        action="store_true",
        help="Use every complete saved reversibility-protocol event, rather than --top.",
    )
    parser.add_argument(
        "--output-directory", type=Path,
        default=ROOT / "Plots" / "reconnecting_largest_energy_events",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=234,
        help="PNG resolution in dots per inch (30%% above the previous 180 dpi output).",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")

    if args.all_saved and not args.saved_only:
        raise ValueError("--all-saved requires --saved-only.")
    events = select_events(
        args.job,
        number=None if args.all_saved else args.top,
        saved_only=args.saved_only,
    )
    if args.aggregate_only:
        flow_output, zoomed_flow_output, stress_output = render_aggregate_largest_event_summary(
            events,
            args.output_directory,
            dpi=args.dpi,
            direction_split_otsu=args.aggregate_otsu,
        )
        print(flow_output)
        print(zoomed_flow_output)
        print(stress_output)
        return
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

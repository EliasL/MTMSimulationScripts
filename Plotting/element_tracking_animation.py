"""Animations for a single-element matrix history and its local mesh."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from MTMath.poincareEnergy import drawPoincareGrid, prepPoincareFig
from Plotting.element_tracking import ElementMatrixHistory, read_matrix_component
from Plotting.pyplotFunctions import wrap_periodic_mesh
from Plotting.vtuDataForSylvain import VTUData


@dataclass(frozen=True)
class GammaTimeline:
    """Uniform-load frame timing shared by Poincare and mesh animations."""

    node_history_indices: np.ndarray
    node_coordinates: np.ndarray
    path_history_indices: np.ndarray
    frame_loads: np.ndarray
    t_node_counts: np.ndarray
    path_lower_indices: np.ndarray
    path_progress: np.ndarray
    center_coordinates: np.ndarray
    mesh_history_indices: np.ndarray

    def __post_init__(self) -> None:
        frame_count = len(self.frame_loads)
        arrays = (
            self.t_node_counts,
            self.path_lower_indices,
            self.path_progress,
            self.center_coordinates,
            self.mesh_history_indices,
        )
        if any(len(array) != frame_count for array in arrays):
            raise ValueError("All frame-wise timeline arrays must have equal length.")
        if self.node_coordinates.shape != (len(self.node_history_indices), 2):
            raise ValueError("node_coordinates must have shape (number of nodes, 2).")
        if len(self.node_history_indices) < 2:
            raise ValueError("At least two distinct matrix states are required.")
        if len(self.path_history_indices) < 2:
            raise ValueError("At least two distinct loads are required.")
        if np.any(np.diff(self.frame_loads) <= 0):
            raise ValueError("Frame loads must be strictly increasing.")
        if np.any((self.path_progress < 0) | (self.path_progress > 1)):
            raise ValueError("path_progress must remain inside [0, 1].")
        invalid_node_count = (self.t_node_counts < 1) | (
            self.t_node_counts > len(self.node_history_indices)
        )
        if np.any(invalid_node_count):
            raise ValueError("t_node_counts contains an invalid node count.")

    @property
    def frame_count(self) -> int:
        return len(self.frame_loads)

    @property
    def node_complex(self) -> np.ndarray:
        return self.node_coordinates[:, 0] + 1j * self.node_coordinates[:, 1]

    @property
    def center_complex(self) -> np.ndarray:
        return self.center_coordinates[:, 0] + 1j * self.center_coordinates[:, 1]


@dataclass(frozen=True)
class MeshNeighborhood:
    """One target-centred finite-element neighbourhood."""

    outer_polygons: tuple[np.ndarray, ...]
    neighbor_polygons: tuple[np.ndarray, ...]
    target_polygon: np.ndarray
    source_path: Path

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        polygons = self.outer_polygons + self.neighbor_polygons + (self.target_polygon,)
        points = np.concatenate(polygons)
        return points.min(axis=0), points.max(axis=0)


@dataclass(frozen=True)
class PeriodicMeshSnapshot:
    """One full mesh wrapped into a square periodic cell."""

    polygons: np.ndarray
    energies: np.ndarray
    region_outlines: tuple[np.ndarray, ...]
    source_path: Path

    def __post_init__(self) -> None:
        if self.polygons.ndim != 3 or self.polygons.shape[1:] != (3, 2):
            raise ValueError("polygons must have shape (number of triangles, 3, 2).")
        if self.energies.shape != (len(self.polygons),):
            raise ValueError("One energy value is required for every plotted triangle.")
        if not np.all(np.isfinite(self.polygons)) or not np.all(np.isfinite(self.energies)):
            raise ValueError(f"Non-finite periodic-mesh data in {self.source_path}")


def load_element_matrix_histories(
    vtu_paths: Sequence[str | Path],
    element_index: int,
    matrix_names: Sequence[str],
    *,
    progress_every: int = 50,
) -> tuple[ElementMatrixHistory, ...]:
    """Load several matrix histories while opening each VTU only once."""
    paths = tuple(Path(path) for path in vtu_paths)
    names = tuple(matrix_names)
    if not paths or not names or len(set(names)) != len(names):
        raise ValueError("Paths and unique matrix names are required.")
    if element_index < 0:
        raise ValueError("element_index must be non-negative.")
    matrices = {name: np.empty((len(paths), 2, 2), dtype=float) for name in names}
    for state_index, path in enumerate(paths):
        data = VTUData(path)
        for name in names:
            for i in (1, 2):
                for j in (1, 2):
                    values = read_matrix_component(data, name, i, j, symmetric=False)
                    if element_index >= len(values):
                        raise IndexError(
                            f"Element {element_index} is outside {name} in {path}."
                        )
                    matrices[name][state_index, i - 1, j - 1] = values[element_index]
        if progress_every and (state_index + 1) % progress_every == 0:
            print(f"loaded matrix histories: {state_index + 1}/{len(paths)}", flush=True)
    return tuple(
        ElementMatrixHistory(paths, matrices[name], element_index, name) for name in names
    )


def mobius_to_origin(z: np.ndarray | complex, center: complex) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    denominator = 1 - np.conjugate(center) * z
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Degenerate Mobius-centering denominator.")
    return (z - center) / denominator


def mobius_from_origin(w: np.ndarray | complex, center: complex) -> np.ndarray:
    w = np.asarray(w, dtype=complex)
    denominator = 1 + np.conjugate(center) * w
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("Degenerate inverse Mobius denominator.")
    return (w + center) / denominator


def poincare_geodesic(
    start: complex,
    end: complex,
    progress: np.ndarray | float,
) -> np.ndarray:
    """Interpolate at constant hyperbolic arclength between two disk points."""
    progress = np.asarray(progress, dtype=float)
    relative_end = complex(mobius_to_origin(end, start))
    radius = abs(relative_end)
    if radius >= 1:
        raise ValueError("The geodesic endpoint lies outside the Poincare disk.")
    if radius < 1e-14:
        return np.full(progress.shape, start, dtype=complex)
    direction = relative_end / radius
    relative = np.tanh(progress * np.arctanh(radius)) * direction
    return mobius_from_origin(relative, start)


def build_gamma_timeline(
    history: ElementMatrixHistory,
    loads: Sequence[float],
    camera_history: ElementMatrixHistory,
    *,
    gamma_per_frame: float = 0.002,
    camera_smoothing_gamma: float = 0.04,
) -> GammaTimeline:
    """Sample time uniformly in load and smooth the camera along ``camera_history``."""
    loads = np.asarray(loads, dtype=float)
    if loads.shape != (len(history.paths),):
        raise ValueError("loads must contain one value for every history state.")
    if not np.all(np.isfinite(loads)) or np.any(np.diff(loads) < 0):
        raise ValueError("loads must be finite and non-decreasing.")
    if camera_history.paths != history.paths:
        raise ValueError("The tracked and camera histories must use identical paths.")
    if camera_history.element_index != history.element_index:
        raise ValueError("The tracked and camera histories must use one element.")
    if gamma_per_frame <= 0:
        raise ValueError("gamma_per_frame must be positive.")
    if camera_smoothing_gamma < 0:
        raise ValueError("camera_smoothing_gamma must be non-negative.")

    change_mask = np.r_[
        True,
        np.any(history.matrices[1:] != history.matrices[:-1], axis=(1, 2)),
    ]
    node_indices = np.flatnonzero(change_mask)
    node_coordinates = history.poincare_coordinates()[node_indices]

    unique_loads, first_indices = np.unique(loads, return_index=True)
    path_indices = np.r_[first_indices[1:] - 1, len(loads) - 1]
    if len(unique_loads) < 2:
        raise ValueError("The history must span at least two distinct loads.")
    span = float(unique_loads[-1] - unique_loads[0])
    frame_count = max(2, int(np.ceil(span / gamma_per_frame)) + 1)
    frame_loads = np.linspace(unique_loads[0], unique_loads[-1], frame_count)

    upper = np.searchsorted(unique_loads, frame_loads, side="right")
    lower = np.clip(upper - 1, 0, len(unique_loads) - 1)
    upper = np.clip(upper, 0, len(unique_loads) - 1)
    denominator = unique_loads[upper] - unique_loads[lower]
    progress = np.divide(
        frame_loads - unique_loads[lower],
        denominator,
        out=np.zeros_like(frame_loads),
        where=denominator > 0,
    )

    camera_coordinates = camera_history.poincare_coordinates()[path_indices]
    camera_nodes = camera_coordinates[:, 0] + 1j * camera_coordinates[:, 1]
    camera_raw = np.asarray(
        [
            complex(poincare_geodesic(camera_nodes[lo], camera_nodes[hi], value))
            for lo, hi, value in zip(lower, upper, progress, strict=True)
        ]
    )
    delta_gamma = float(frame_loads[1] - frame_loads[0])
    camera = _gaussian_smooth(
        camera_raw,
        sigma_frames=camera_smoothing_gamma / delta_gamma,
    )
    if np.any(np.abs(camera) >= 1):
        raise ValueError("The smoothed camera path left the Poincare disk.")
    center_array = np.column_stack((camera.real, camera.imag))
    t_node_counts = np.searchsorted(loads[node_indices], frame_loads, side="right")
    t_node_counts = np.clip(t_node_counts, 1, len(node_indices))

    return GammaTimeline(
        node_indices,
        node_coordinates,
        path_indices,
        frame_loads,
        t_node_counts,
        lower,
        progress,
        center_array,
        path_indices[lower],
    )


def _gaussian_smooth(values: np.ndarray, *, sigma_frames: float) -> np.ndarray:
    if sigma_frames < 0:
        raise ValueError("sigma_frames must be non-negative.")
    if sigma_frames < 0.5:
        return np.asarray(values, dtype=complex).copy()
    radius = int(np.ceil(3 * sigma_frames))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    weights /= weights.sum()
    padded = np.pad(np.asarray(values, dtype=complex), radius, mode="edge")
    return np.convolve(padded, weights, mode="valid")


def extract_poincare_grid_segments(
    *,
    depth: int = 5,
    samples_per_line: int = 50,
    grid_size: int = 800,
    transformation: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Generate a standard grid patch after an optional exact congruence."""
    if depth < 0 or samples_per_line < 2:
        raise ValueError("Invalid grid depth or samples_per_line.")
    fig, ax = plt.subplots()
    prepPoincareFig(
        grid_size=grid_size,
        ax=ax,
        withCircle=False,
        withGrid=False,
        withYieldSurface=False,
        minimalTicks=True,
    )
    drawPoincareGrid(
        ax,
        grid_size=grid_size,
        depth=depth,
        transformation=transformation,
        c="gray",
    )
    half = grid_size / 2
    segments: list[np.ndarray] = []
    for line in ax.lines:
        x, y = line.get_data()
        z = (np.asarray(x) - half) / half + 1j * (np.asarray(y) - half) / half
        finite = np.isfinite(z) & (np.abs(z) <= 1 + 1e-8)
        z = z[finite]
        if len(z) < 2:
            continue
        indices = np.unique(
            np.linspace(0, len(z) - 1, min(samples_per_line, len(z))).astype(int)
        )
        segments.append(z[indices])
    plt.close(fig)
    if not segments:
        raise RuntimeError("The Poincare grid generator produced no line segments.")
    return tuple(segments)


def extract_mesh_neighborhood(
    vtu_path: str | Path,
    element_index: int,
    *,
    ring_depth: int = 2,
) -> MeshNeighborhood:
    """Extract vertex-adjacent element rings and centre them on the target cell."""
    if ring_depth != 2:
        raise ValueError("Only the two-ring classification is currently supported.")
    data = VTUData(vtu_path)
    points = np.asarray(data.points[:, :2], dtype=float)
    triangles = np.asarray(data.triangles, dtype=int)
    if not 0 <= element_index < len(triangles):
        raise IndexError(
            f"Element {element_index} is outside a mesh with {len(triangles)} cells."
        )
    second_ring, first_ring = _neighborhood_rings(triangles, element_index)
    center = points[triangles[element_index]].mean(axis=0)

    def polygons(indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        return tuple(points[triangles[index]] - center for index in sorted(indices))

    result = MeshNeighborhood(
        polygons(second_ring),
        polygons(first_ring),
        points[triangles[element_index]] - center,
        Path(vtu_path),
    )
    lower, upper = result.bounds()
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError(f"Non-finite local mesh coordinates in {vtu_path}")
    return result


def _neighborhood_rings(
    triangles: np.ndarray,
    element_index: int,
) -> tuple[set[int], set[int]]:
    target_nodes = set(triangles[element_index])
    first_with_target = {
        index
        for index, triangle in enumerate(triangles)
        if target_nodes.intersection(triangle)
    }
    first_ring = first_with_target - {element_index}
    first_nodes = set(triangles[list(first_with_target)].reshape(-1))
    through_second = {
        index
        for index, triangle in enumerate(triangles)
        if first_nodes.intersection(triangle)
    }
    second_ring = through_second - first_with_target
    return second_ring, first_ring


def extract_periodic_mesh(
    vtu_path: str | Path,
    element_index: int,
    load: float,
    *,
    box_size: float,
    energy_field: str = "energy_field",
) -> PeriodicMeshSnapshot:
    """Wrap a sheared periodic mesh into a square fractional-coordinate box."""
    if not np.isfinite(load) or box_size <= 0:
        raise ValueError("load must be finite and box_size must be positive.")
    data = VTUData(vtu_path)
    points = np.asarray(data.points[:, :2], dtype=float)
    triangles = np.asarray(data.triangles, dtype=int)
    energies = _cell_energy_values(data, len(triangles), energy_field)
    if not 0 <= element_index < len(triangles):
        raise IndexError(
            f"Element {element_index} is outside a mesh with {len(triangles)} cells."
        )
    if "refIndex" not in data.mesh.point_data:
        raise KeyError(f"refIndex is required to locate the periodic origin in {vtu_path}")
    reference_indices = np.asarray(data.mesh.point_data["refIndex"]).reshape(-1)
    plotted_polygons, plotted_energies, wrapped = wrap_periodic_mesh(
        points,
        triangles,
        energies,
        reference_indices,
        load,
        box_size,
        source_path=vtu_path,
    )

    second_ring, first_ring = _neighborhood_rings(triangles, element_index)
    region_cells = sorted(second_ring | first_ring | {element_index})
    region_nodes = np.unique(triangles[region_cells])
    target = wrapped[triangles[element_index]]
    target = target[:1] + (target - target[:1]) - np.round(target - target[:1])
    target_center = target.mean(axis=0)
    region = wrapped[region_nodes]
    region = target_center + (region - target_center) - np.round(region - target_center)
    padding = 0.008
    lower = region.min(axis=0) - padding
    upper = region.max(axis=0) + padding
    rectangle = np.array(
        [
            [lower[0], lower[1]],
            [upper[0], lower[1]],
            [upper[0], upper[1]],
            [lower[0], upper[1]],
            [lower[0], lower[1]],
        ]
    )
    outlines = _periodic_curve_copies(rectangle)
    return PeriodicMeshSnapshot(
        plotted_polygons,
        plotted_energies,
        outlines,
        Path(vtu_path),
    )


def _cell_energy_values(
    data: VTUData,
    cell_count: int,
    energy_field: str,
) -> np.ndarray:
    energies, location, _ = data.field(energy_field)
    energies = np.asarray(energies, dtype=float)
    if location != "cell" or energies.shape != (cell_count,):
        raise ValueError(f"{energy_field} must contain one value per triangle.")
    if np.any(~np.isfinite(energies)) or np.any(energies < 0):
        raise ValueError(f"{energy_field} must contain finite non-negative values.")
    return energies


def _periodic_curve_copies(curve: np.ndarray) -> tuple[np.ndarray, ...]:
    copies: list[np.ndarray] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = curve + np.array([dx, dy])
            if np.all(shifted.max(axis=0) >= 0) and np.all(shifted.min(axis=0) <= 1):
                copies.append(shifted)
    if not copies:
        raise RuntimeError("Periodic wrapping produced no visible region outline.")
    return tuple(copies)


def load_mesh_neighborhoods(
    vtu_paths: Sequence[str | Path],
    element_index: int,
    *,
    ring_depth: int = 2,
    progress_every: int = 50,
) -> tuple[MeshNeighborhood, ...]:
    snapshots: list[MeshNeighborhood] = []
    for index, path in enumerate(vtu_paths):
        snapshots.append(
            extract_mesh_neighborhood(path, element_index, ring_depth=ring_depth)
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"loaded mesh neighbourhoods: {index + 1}/{len(vtu_paths)}", flush=True)
    return tuple(snapshots)


def load_periodic_meshes(
    vtu_paths: Sequence[str | Path],
    loads: Sequence[float],
    element_index: int,
    *,
    box_size: float,
    progress_every: int = 50,
) -> tuple[PeriodicMeshSnapshot, ...]:
    if len(vtu_paths) != len(loads):
        raise ValueError("One load is required for every periodic mesh snapshot.")
    snapshots: list[PeriodicMeshSnapshot] = []
    for index, (path, load) in enumerate(zip(vtu_paths, loads, strict=True)):
        snapshots.append(
            extract_periodic_mesh(path, element_index, load, box_size=box_size)
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"loaded periodic meshes: {index + 1}/{len(vtu_paths)}", flush=True)
    return tuple(snapshots)


def mesh_half_width(snapshots: Sequence[MeshNeighborhood], margin: float = 1.06) -> float:
    if not snapshots:
        raise ValueError("At least one mesh snapshot is required.")
    extent = 0.0
    for snapshot in snapshots:
        lower, upper = snapshot.bounds()
        extent = max(extent, float(np.max(np.abs([lower, upper]))))
    return margin * extent


def render_poincare_animation(
    history: ElementMatrixHistory,
    timeline: GammaTimeline,
    output_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    reconstructed_history: ElementMatrixHistory | None = None,
    fps: int = 30,
    dpi: int = 120,
    grid_depth: int = 5,
    grid_samples: int = 50,
) -> Path:
    """Render T and the reconstructed path with a smoothed moving camera."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reconstructed_history is not None:
        if reconstructed_history.paths != history.paths:
            raise ValueError("The T and reconstructed histories must use identical paths.")
        if reconstructed_history.element_index != history.element_index:
            raise ValueError("The T and reconstructed histories must track one element.")
    grid_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, ...]] = {}
    node_grids: list[tuple[np.ndarray, ...]] = []
    for history_index in timeline.node_history_indices:
        matrix = history.matrices[int(history_index)]
        rounded = np.rint(matrix).astype(int)
        if not np.allclose(matrix, rounded, atol=1e-12, rtol=0):
            raise ValueError(f"T is not integer-valued at history index {history_index}.")
        determinant = rounded[0, 0] * rounded[1, 1] - rounded[0, 1] * rounded[1, 0]
        if determinant != 1:
            raise ValueError(
                f"T is not in SL(2,Z) at history index {history_index}: det={determinant}."
            )
        key = tuple(int(value) for value in rounded.reshape(-1))
        if key not in grid_cache:
            grid_cache[key] = extract_poincare_grid_segments(
                depth=grid_depth,
                samples_per_line=grid_samples,
                transformation=rounded,
            )
        node_grids.append(grid_cache[key])
    print(
        f"local Poincare grids: {len(grid_cache)} unique wells at depth {grid_depth}",
        flush=True,
    )
    nodes = timeline.node_complex
    t_path = tuple(
        poincare_geodesic(nodes[index], nodes[index + 1], np.linspace(0, 1, 28))
        for index in range(len(nodes) - 1)
    )
    reconstructed_nodes = None
    reconstructed_path = None
    if reconstructed_history is not None:
        coordinates = reconstructed_history.poincare_coordinates()
        coordinates = coordinates[timeline.path_history_indices]
        reconstructed_nodes = coordinates[:, 0] + 1j * coordinates[:, 1]
        reconstructed_path = tuple(
            poincare_geodesic(
                reconstructed_nodes[index],
                reconstructed_nodes[index + 1],
                np.linspace(0, 1, 10),
            )
            for index in range(len(reconstructed_nodes) - 1)
        )

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.add_patch(Circle((0, 0), 1, fill=False, color="black", linewidth=1.3, zorder=20))
    grid_collection = LineCollection([], colors="0.72", linewidths=0.45, zorder=1)
    t_collection = LineCollection(
        [], colors="0.12", linewidths=1.8, linestyles=[(0, (5, 3))], zorder=8
    )
    reconstructed_collection = LineCollection(
        [], colors="#2166ac", linewidths=2.2, zorder=9
    )
    ax.add_collection(grid_collection)
    ax.add_collection(t_collection)
    ax.add_collection(reconstructed_collection)
    visited = ax.scatter([], [], s=18, color="0.12", zorder=10)
    reconstructed_current = ax.scatter(
        [], [], s=58, facecolor="#2166ac", edgecolor="black", zorder=12
    )
    t_current = ax.scatter(
        [], [], s=72, facecolor="#d73027", edgecolor="black", zorder=13
    )
    gamma_label = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        zorder=30,
    )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ticks = np.linspace(-1, 1, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel(r"$\widetilde{x}_p$")
    ax.set_ylabel(r"$\widetilde{y}_p$")
    handles = [
        Line2D([0], [0], color="0.12", linewidth=1.8, linestyle=(0, (5, 3)), label=r"$T$"),
    ]
    if reconstructed_history is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#2166ac",
                linewidth=2.2,
                label=r"$F_E T$",
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=False)

    writer = _writer(ffmpeg_executable, fps)
    with writer.saving(fig, str(output_path), dpi):
        for frame in range(timeline.frame_count):
            center = timeline.center_complex[frame]
            completed_count = int(timeline.t_node_counts[frame])
            grid = node_grids[completed_count - 1]
            transformed_grid = [complex_to_xy(mobius_to_origin(curve, center)) for curve in grid]
            grid_collection.set_segments(transformed_grid)

            traced = list(t_path[: max(0, completed_count - 1)])
            t_collection.set_segments(
                [complex_to_xy(mobius_to_origin(curve, center)) for curve in traced]
            )
            completed = mobius_to_origin(nodes[:completed_count], center)
            visited.set_offsets(complex_to_xy(completed))
            t_current.set_offsets(
                complex_to_xy(mobius_to_origin(nodes[completed_count - 1], center))
            )
            if reconstructed_nodes is not None and reconstructed_path is not None:
                lower = int(timeline.path_lower_indices[frame])
                fraction = float(timeline.path_progress[frame])
                reconstructed_traced = list(reconstructed_path[:lower])
                if lower < len(reconstructed_nodes) - 1 and fraction > 1e-12:
                    reconstructed_traced.append(
                        poincare_geodesic(
                            reconstructed_nodes[lower],
                            reconstructed_nodes[lower + 1],
                            np.linspace(0, fraction, max(2, int(9 * fraction) + 2)),
                        )
                    )
                    current = complex(
                        poincare_geodesic(
                            reconstructed_nodes[lower],
                            reconstructed_nodes[lower + 1],
                            fraction,
                        )
                    )
                else:
                    current = reconstructed_nodes[lower]
                reconstructed_collection.set_segments(
                    [
                        complex_to_xy(mobius_to_origin(curve, center))
                        for curve in reconstructed_traced
                    ]
                )
                reconstructed_current.set_offsets(
                    complex_to_xy(mobius_to_origin(current, center))
                )
            gamma_label.set_text(rf"$\gamma = {timeline.frame_loads[frame]:.2f}$")
            writer.grab_frame(facecolor="white", transparent=False)
            if (frame + 1) % 30 == 0:
                print(f"Poincare frames: {frame + 1}/{timeline.frame_count}", flush=True)
    plt.close(fig)
    return output_path


def render_mesh_animation(
    history: ElementMatrixHistory,
    timeline: GammaTimeline,
    snapshots: Sequence[MeshNeighborhood],
    output_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    fps: int = 30,
    dpi: int = 120,
) -> Path:
    """Render a transparent, fixed-window, target-centred mesh animation."""
    if len(snapshots) != len(history.paths):
        raise ValueError("One mesh snapshot is required for every history state.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    half_width = mesh_half_width(snapshots)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outer = PolyCollection([], facecolors="0.94", edgecolors="0.68", linewidths=0.8)
    neighbors = PolyCollection(
        [], facecolors="#d9e8f5", edgecolors="0.38", linewidths=1.0
    )
    target = PolyCollection(
        [], facecolors="#d73027", edgecolors="black", linewidths=1.4
    )
    ax.add_collection(outer)
    ax.add_collection(neighbors)
    ax.add_collection(target)
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    writer = _alpha_writer(ffmpeg_executable, fps)
    with writer.saving(fig, str(output_path), dpi):
        for frame, mesh_index in enumerate(timeline.mesh_history_indices):
            snapshot = snapshots[int(mesh_index)]
            outer.set_verts(snapshot.outer_polygons)
            neighbors.set_verts(snapshot.neighbor_polygons)
            target.set_verts([snapshot.target_polygon])
            writer.grab_frame(transparent=True, facecolor="none")
            if (frame + 1) % 30 == 0:
                print(f"mesh frames: {frame + 1}/{timeline.frame_count}", flush=True)
    plt.close(fig)
    return output_path


def render_periodic_mesh_animation(
    timeline: GammaTimeline,
    vtu_paths: Sequence[str | Path],
    loads: Sequence[float],
    element_index: int,
    output_path: str | Path,
    *,
    box_size: float,
    ffmpeg_executable: str | Path,
    fps: int = 30,
    dpi: int = 120,
    energy_field: str = "energy_field",
    energy_sample_count: int = 64,
) -> Path:
    """Stream the full energy-coloured mesh inside a square periodic box."""
    # NOTE: When generating another video, experiment with the Cartesian
    # viewport-culling method in pyplotFunctions instead of this square box.
    if len(vtu_paths) != len(loads):
        raise ValueError("One load is required for every periodic mesh state.")
    if len(vtu_paths) <= int(np.max(timeline.mesh_history_indices)):
        raise ValueError("Periodic mesh states do not cover the animation timeline.")
    if energy_sample_count < 1:
        raise ValueError("energy_sample_count must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.unique(
        np.linspace(
            0,
            len(vtu_paths) - 1,
            min(energy_sample_count, len(vtu_paths)),
            dtype=int,
        )
    )
    samples: list[np.ndarray] = []
    for index in sample_indices:
        data = VTUData(vtu_paths[int(index)])
        cell_count = len(data.triangles)
        values = _cell_energy_values(data, cell_count, energy_field)
        stride = max(1, len(values) // 4096)
        samples.append(values[::stride])
    upper_energy = float(np.quantile(np.concatenate(samples), 0.995))
    if upper_energy <= 0:
        raise ValueError("The periodic mesh has no positive energy values.")
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=upper_energy, clip=True)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    energy = PolyCollection(
        [],
        cmap="coolwarm",
        norm=norm,
        edgecolors="face",
        linewidths=0.35,
        antialiaseds=False,
    )
    region = LineCollection(
        [], colors="#00a651", linewidths=4.0, linestyles="--", zorder=5
    )
    ax.add_collection(energy)
    ax.add_collection(region)
    ax.add_patch(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="0.2", linewidth=1.8, zorder=6)
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()

    writer = _writer(ffmpeg_executable, fps)
    previous_index = -1
    snapshot: PeriodicMeshSnapshot | None = None
    with writer.saving(fig, str(output_path), dpi):
        for frame, mesh_index in enumerate(timeline.mesh_history_indices):
            mesh_index = int(mesh_index)
            if mesh_index != previous_index:
                snapshot = extract_periodic_mesh(
                    vtu_paths[mesh_index],
                    element_index,
                    loads[mesh_index],
                    box_size=box_size,
                    energy_field=energy_field,
                )
                previous_index = mesh_index
            if snapshot is None:
                raise RuntimeError("No periodic mesh snapshot was loaded.")
            energy.set_verts(snapshot.polygons)
            energy.set_array(snapshot.energies)
            region.set_segments(snapshot.region_outlines)
            writer.grab_frame(facecolor="white", transparent=False)
            if (frame + 1) % 30 == 0:
                print(
                    f"periodic mesh frames: {frame + 1}/{timeline.frame_count}",
                    flush=True,
                )
    plt.close(fig)
    return output_path


def compose_picture_in_picture(
    poincare_video: str | Path,
    mesh_video: str | Path,
    periodic_mesh_video: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    inset_fraction: float = 0.34,
    periodic_inset_fraction: float = 0.60,
    local_margin: int = 130,
    periodic_margin: int = 130,
) -> Path:
    """Place local and full periodic mesh videos along the right edge."""
    invalid_inset = not 0.1 <= inset_fraction <= 0.6
    invalid_periodic_inset = not 0.1 <= periodic_inset_fraction <= 0.6
    if invalid_inset or invalid_periodic_inset:
        raise ValueError("Inset fractions must lie inside [0.1, 0.6].")
    if local_margin < 0 or periodic_margin < 0:
        raise ValueError("Inset margins must be non-negative.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_expression = (
        f"[1:v]format=rgba,scale=w=trunc(iw*{inset_fraction}/2)*2:"
        f"h=trunc(ih*{inset_fraction}/2)*2[local];"
        f"[2:v]format=rgba,scale=w=trunc(iw*{periodic_inset_fraction}/2)*2:"
        f"h=trunc(ih*{periodic_inset_fraction}/2)*2[periodic];"
        f"[0:v][local]overlay=W-w-{local_margin}:{local_margin}:shortest=1[withlocal];"
        f"[withlocal][periodic]overlay=W-w-{periodic_margin}:"
        f"H-h-{periodic_margin}:shortest=1[out]"
    )
    command = [
        str(ffmpeg_executable),
        "-y",
        "-i",
        str(poincare_video),
        "-i",
        str(mesh_video),
        "-i",
        str(periodic_mesh_video),
        "-filter_complex",
        scale_expression,
        "-map",
        "[out]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def complex_to_xy(values: np.ndarray | complex) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    return np.column_stack((values.real.reshape(-1), values.imag.reshape(-1)))


def _writer(ffmpeg_executable: str | Path, fps: int) -> FFMpegWriter:
    ffmpeg_executable = Path(ffmpeg_executable)
    if not ffmpeg_executable.is_file():
        raise FileNotFoundError(ffmpeg_executable)
    if fps <= 0:
        raise ValueError("fps must be positive.")
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_executable)
    return FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def _alpha_writer(ffmpeg_executable: str | Path, fps: int) -> FFMpegWriter:
    ffmpeg_executable = Path(ffmpeg_executable)
    if not ffmpeg_executable.is_file():
        raise FileNotFoundError(ffmpeg_executable)
    if fps <= 0:
        raise ValueError("fps must be positive.")
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_executable)
    return FFMpegWriter(fps=fps, codec="qtrle", extra_args=["-pix_fmt", "argb"])

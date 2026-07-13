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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from MTMath.poincareEnergy import drawPoincareGrid, prepPoincareFig
from Plotting.element_tracking import ElementMatrixHistory
from Plotting.vtuDataForSylvain import VTUData


@dataclass(frozen=True)
class TransitionTimeline:
    """Shared frame timing for Poincare and mesh animations."""

    node_history_indices: np.ndarray
    node_coordinates: np.ndarray
    segment_indices: np.ndarray
    segment_progress: np.ndarray
    center_coordinates: np.ndarray
    mesh_history_indices: np.ndarray

    def __post_init__(self) -> None:
        frame_count = len(self.segment_indices)
        arrays = (
            self.segment_progress,
            self.center_coordinates,
            self.mesh_history_indices,
        )
        if any(len(array) != frame_count for array in arrays):
            raise ValueError("All frame-wise timeline arrays must have equal length.")
        if self.node_coordinates.shape != (len(self.node_history_indices), 2):
            raise ValueError("node_coordinates must have shape (number of nodes, 2).")
        if len(self.node_history_indices) < 2:
            raise ValueError("At least two distinct matrix states are required.")
        if np.any((self.segment_progress < 0) | (self.segment_progress > 1)):
            raise ValueError("segment_progress must remain inside [0, 1].")

    @property
    def frame_count(self) -> int:
        return len(self.segment_indices)

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


def eased_progress(progress: np.ndarray | float, strength: float = 0.35) -> np.ndarray:
    """Blend linear motion with smoothstep for a slight acceleration/deceleration."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie inside [0, 1].")
    progress = np.asarray(progress, dtype=float)
    if np.any((progress < 0) | (progress > 1)):
        raise ValueError("progress must lie inside [0, 1].")
    smooth = progress * progress * (3 - 2 * progress)
    return (1 - strength) * progress + strength * smooth


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


def build_transition_timeline(
    history: ElementMatrixHistory,
    *,
    frames_per_transition: int = 30,
    hold_frames: int = 6,
    easing_strength: float = 0.35,
) -> TransitionTimeline:
    """Build one fixed-duration segment for every distinct matrix transition."""
    if frames_per_transition < 2:
        raise ValueError("frames_per_transition must be at least two.")
    if hold_frames < 0:
        raise ValueError("hold_frames must be non-negative.")
    change_mask = np.r_[
        True,
        np.any(history.matrices[1:] != history.matrices[:-1], axis=(1, 2)),
    ]
    node_indices = np.flatnonzero(change_mask)
    node_coordinates = history.poincare_coordinates()[node_indices]
    nodes = node_coordinates[:, 0] + 1j * node_coordinates[:, 1]

    segments: list[int] = []
    progress_values: list[float] = []
    centers: list[complex] = []
    mesh_indices: list[int] = []

    for _ in range(hold_frames):
        segments.append(0)
        progress_values.append(0)
        centers.append(nodes[0])
        mesh_indices.append(int(node_indices[0]))

    for segment in range(len(nodes) - 1):
        raw = np.arange(1, frames_per_transition + 1) / frames_per_transition
        eased = eased_progress(raw, easing_strength)
        geodesic = poincare_geodesic(nodes[segment], nodes[segment + 1], eased)
        source_mesh = int(node_indices[segment])
        target_mesh = int(node_indices[segment + 1])
        mesh_for_segment = np.rint(
            source_mesh + eased * (target_mesh - source_mesh)
        ).astype(int)
        segments.extend([segment] * frames_per_transition)
        progress_values.extend(eased.tolist())
        centers.extend(geodesic.tolist())
        mesh_indices.extend(mesh_for_segment.tolist())
        for _ in range(hold_frames):
            segments.append(segment)
            progress_values.append(1)
            centers.append(nodes[segment + 1])
            mesh_indices.append(target_mesh)

    center_array = np.column_stack(
        (np.asarray(centers).real, np.asarray(centers).imag)
    )
    centered_current = mobius_to_origin(
        center_array[:, 0] + 1j * center_array[:, 1],
        center_array[:, 0] + 1j * center_array[:, 1],
    )
    if not np.allclose(centered_current, 0, atol=1e-12):
        raise RuntimeError("The generated Poincare timeline does not remain centered.")
    return TransitionTimeline(
        node_indices,
        node_coordinates,
        np.asarray(segments, dtype=int),
        np.asarray(progress_values),
        center_array,
        np.asarray(mesh_indices, dtype=int),
    )


def extract_poincare_grid_segments(
    *,
    depth: int = 5,
    samples_per_line: int = 50,
    grid_size: int = 800,
) -> tuple[np.ndarray, ...]:
    """Generate the standard SimulationScripts grid as downsampled disk curves."""
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
    drawPoincareGrid(ax, grid_size=grid_size, depth=depth, c="gray")
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
    timeline: TransitionTimeline,
    output_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    reconstructed_history: ElementMatrixHistory | None = None,
    fps: int = 30,
    dpi: int = 120,
    grid_depth: int = 10,
    grid_samples: int = 50,
) -> Path:
    """Render T and an optional reconstructed total path while centring on T."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reconstructed_history is not None:
        if reconstructed_history.paths != history.paths:
            raise ValueError("The T and reconstructed histories must use identical paths.")
        if reconstructed_history.element_index != history.element_index:
            raise ValueError("The T and reconstructed histories must track one element.")
    grid = extract_poincare_grid_segments(
        depth=grid_depth, samples_per_line=grid_samples
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
    ax.scatter([0], [0], s=72, facecolor="#d73027", edgecolor="black", zorder=12)
    status = ax.text(
        0.5,
        0.98,
        "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
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
    ax.legend(handles=handles, loc="upper left", frameon=False)

    writer = _writer(ffmpeg_executable, fps)
    with writer.saving(fig, str(output_path), dpi):
        for frame in range(timeline.frame_count):
            center = timeline.center_complex[frame]
            segment = int(timeline.segment_indices[frame])
            progress = float(timeline.segment_progress[frame])
            transformed_grid = [complex_to_xy(mobius_to_origin(curve, center)) for curve in grid]
            grid_collection.set_segments(transformed_grid)

            traced = [t_path[index] for index in range(segment)]
            partial_progress = np.linspace(0, progress, max(2, int(27 * progress) + 2))
            traced.append(
                poincare_geodesic(nodes[segment], nodes[segment + 1], partial_progress)
            )
            t_collection.set_segments(
                [complex_to_xy(mobius_to_origin(curve, center)) for curve in traced]
            )
            completed_count = segment + 1 + int(np.isclose(progress, 1))
            completed = mobius_to_origin(nodes[:completed_count], center)
            visited.set_offsets(complex_to_xy(completed))
            if reconstructed_nodes is not None and reconstructed_path is not None:
                source_index = int(timeline.node_history_indices[segment])
                target_index = int(timeline.node_history_indices[segment + 1])
                history_position = source_index + progress * (target_index - source_index)
                lower = min(int(np.floor(history_position)), len(reconstructed_nodes) - 1)
                fraction = history_position - lower
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
            status.set_text(f"element {history.element_index}    transition {segment + 1}/{len(nodes) - 1}")
            writer.grab_frame(facecolor="white")
            if (frame + 1) % 30 == 0:
                print(f"Poincare frames: {frame + 1}/{timeline.frame_count}", flush=True)
    plt.close(fig)
    return output_path


def render_mesh_animation(
    history: ElementMatrixHistory,
    timeline: TransitionTimeline,
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


def compose_picture_in_picture(
    poincare_video: str | Path,
    mesh_video: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    inset_fraction: float = 0.34,
    margin: int = 24,
) -> Path:
    """Place the mesh video in the upper-right corner of the Poincare video."""
    if not 0.1 <= inset_fraction <= 0.6:
        raise ValueError("inset_fraction must lie inside [0.1, 0.6].")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_expression = (
        f"[1:v]format=rgba,scale=w=trunc(iw*{inset_fraction}/2)*2:"
        f"h=trunc(ih*{inset_fraction}/2)*2[mesh];"
        f"[0:v][mesh]overlay=W-w-{margin}:{margin}:shortest=1[out]"
    )
    command = [
        str(ffmpeg_executable),
        "-y",
        "-i",
        str(poincare_video),
        "-i",
        str(mesh_video),
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

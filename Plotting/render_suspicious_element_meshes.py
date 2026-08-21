#!/usr/bin/env python3
"""Render full and local energy meshes for the two tracked event elements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting import meshEventPlotting as mesh_plot
from Plotting.pyplotFunctions import draw_periodic_element_outlines, plot_mesh


DEFAULT_ROOT = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0/data/reversibilityData"
)
BOX_SIZE = 200.0
LOAD_INCREMENT = 1e-5
FULL_VIEWPORT = (0.0, 400.0, 0.0, 200.0)


def _single_state(directory: Path) -> Path:
    paths = sorted(directory.glob("state2_relaxed_gamma_plus.*.vtu"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one state2_relaxed_gamma_plus VTU in {directory}, "
            f"found {len(paths)}."
        )
    return paths[0]


def _plot_mesh_view(
    state_path: Path,
    state: mesh_plot.MeshState,
    element_index: int,
    load: float,
    viewport: tuple[float, float, float, float],
    output_path: Path,
    *,
    title: str,
    figsize: tuple[float, float],
) -> Path:
    energy = np.asarray(state.cell_fields["energy_field"], dtype=float)
    if energy.shape != (len(state.triangles),):
        raise ValueError(f"Energy field has the wrong shape in {state_path}.")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError(f"Energy field is invalid in {state_path}.")
    if not 0 <= element_index < len(energy):
        raise IndexError(f"Element slot {element_index} is outside {state_path}.")

    figure, axis = plt.subplots(figsize=figsize)
    plot_mesh(
        state_path,
        mesh_property="energy",
        e_lims=(0.0, float(np.max(energy))),
        ax=axis,
        add_colorbar=True,
        add_rombus=True,
        periodic_load=load,
        periodic_box_size=BOX_SIZE,
        load_increment=LOAD_INCREMENT,
        cartesian_viewport_culling=True,
        cartesian_viewport=viewport,
        unwrap_periodic_triangles=True,
    )
    draw_periodic_element_outlines(
        axis,
        state,
        np.asarray([element_index], dtype=int),
        load=load,
        box_size=BOX_SIZE,
        viewport=viewport,
        color="#d62728",
        linewidth=2.0,
        linestyle="-",
        zorder=30,
    )
    axis.set_xlim(viewport[:2])
    axis.set_ylim(viewport[2:])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{title}; slot {element_index}; E={energy[element_index]:.6g}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return output_path


def _plot_before_after_mesh_view(
    before_path: Path,
    before_state: mesh_plot.MeshState,
    after_path: Path,
    after_state: mesh_plot.MeshState,
    element_index: int,
    before_load: float,
    after_load: float,
    viewport: tuple[float, float, float, float],
    output_path: Path,
    *,
    title: str,
    figsize: tuple[float, float],
) -> Path:
    """Show one serialized element slot before and after the event."""

    before_energy = np.asarray(before_state.cell_fields["energy_field"], dtype=float)
    after_energy = np.asarray(after_state.cell_fields["energy_field"], dtype=float)
    for path, energy, state in (
        (before_path, before_energy, before_state),
        (after_path, after_energy, after_state),
    ):
        if energy.shape != (len(state.triangles),):
            raise ValueError(f"Energy field has the wrong shape in {path}.")
        if not np.all(np.isfinite(energy)) or np.any(energy < 0):
            raise ValueError(f"Energy field is invalid in {path}.")
        if not 0 <= element_index < len(energy):
            raise IndexError(f"Element slot {element_index} is outside {path}.")
    common_max = float(max(np.max(before_energy), np.max(after_energy)))
    if not np.isfinite(common_max) or common_max <= 0:
        raise ValueError("The before/after energy fields have no positive finite maximum.")

    figure, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    for axis, path, state, load, label, energy in (
        (axes[0, 0], before_path, before_state, before_load, "before", before_energy),
        (axes[0, 1], after_path, after_state, after_load, "after", after_energy),
    ):
        plot_mesh(
            path,
            mesh_property="energy",
            e_lims=(0.0, common_max),
            ax=axis,
            add_colorbar=True,
            add_rombus=True,
            periodic_load=load,
            periodic_box_size=BOX_SIZE,
            load_increment=LOAD_INCREMENT,
            cartesian_viewport_culling=True,
            cartesian_viewport=viewport,
            unwrap_periodic_triangles=True,
        )
        draw_periodic_element_outlines(
            axis,
            state,
            np.asarray([element_index], dtype=int),
            load=load,
            box_size=BOX_SIZE,
            viewport=viewport,
            color="#d62728",
            linewidth=2.0,
            linestyle="-",
            zorder=30,
        )
        axis.set_xlim(viewport[:2])
        axis.set_ylim(viewport[2:])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{label}; slot {element_index}; E={energy[element_index]:.6g}")
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return output_path


def render_event_meshes(
    directory: Path,
    *,
    element_index: int,
    load: float,
    output_directory: Path,
    stem: str,
    zoom_half_width: float = 7.5,
) -> tuple[Path, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if not np.isfinite(load) or load <= 0:
        raise ValueError("The event load must be finite and positive.")
    if not np.isfinite(zoom_half_width) or zoom_half_width <= 0:
        raise ValueError("zoom_half_width must be finite and positive.")
    state_path = _single_state(directory)
    state = mesh_plot.load_mesh_state(state_path)
    centers = mesh_plot.periodic_triangle_centres(
        state, load=load, box_size=BOX_SIZE
    )
    if not 0 <= element_index < len(centers):
        raise IndexError(f"Element slot {element_index} is outside {state_path}.")
    center_x, center_y = centers[element_index]
    zoom_viewport = (
        float(center_x - zoom_half_width),
        float(center_x + zoom_half_width),
        float(center_y - zoom_half_width),
        float(center_y + zoom_half_width),
    )
    full = _plot_mesh_view(
        state_path,
        state,
        element_index,
        load,
        FULL_VIEWPORT,
        output_directory / f"{stem}_full_mesh.png",
        title="new equilibrium energy mesh",
        figsize=(14.0, 7.0),
    )
    zoomed = _plot_mesh_view(
        state_path,
        state,
        element_index,
        load,
        zoom_viewport,
        output_directory / f"{stem}_zoomed_mesh.png",
        title="local energy mesh",
        figsize=(8.0, 8.0),
    )
    return full, zoomed


def render_before_after_event_meshes(
    directory: Path,
    *,
    element_index: int,
    before_load: float,
    after_load: float,
    output_directory: Path,
    stem: str,
    zoom_half_width: float = 7.5,
) -> tuple[Path, Path]:
    """Render common-window before/after full and zoomed views."""

    if not directory.is_dir():
        raise FileNotFoundError(directory)
    before_paths = sorted(directory.glob("state0_min_gamma.*.vtu"))
    after_paths = sorted(directory.glob("state2_relaxed_gamma_plus.*.vtu"))
    if len(before_paths) != 1 or len(after_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one before and after VTU in {directory}; "
            f"found {len(before_paths)} and {len(after_paths)}."
        )
    before_path, after_path = before_paths[0], after_paths[0]
    before_state = mesh_plot.load_mesh_state(before_path)
    after_state = mesh_plot.load_mesh_state(after_path)
    after_centers = mesh_plot.periodic_triangle_centres(
        after_state, load=after_load, box_size=BOX_SIZE
    )
    if not 0 <= element_index < len(after_centers):
        raise IndexError(f"Element slot {element_index} is outside {after_path}.")
    center_x, center_y = after_centers[element_index]
    zoom_viewport = (
        float(center_x - zoom_half_width),
        float(center_x + zoom_half_width),
        float(center_y - zoom_half_width),
        float(center_y + zoom_half_width),
    )
    full = _plot_before_after_mesh_view(
        before_path,
        before_state,
        after_path,
        after_state,
        element_index,
        before_load,
        after_load,
        FULL_VIEWPORT,
        output_directory / f"{stem}_full_mesh.png",
        title="rank 408 slot 72927: before and after",
        figsize=(14.0, 7.0),
    )
    zoomed = _plot_before_after_mesh_view(
        before_path,
        before_state,
        after_path,
        after_state,
        element_index,
        before_load,
        after_load,
        zoom_viewport,
        output_directory / f"{stem}_zoomed_mesh.png",
        title="rank 408 slot 72927: common local window",
        figsize=(12.0, 6.0),
    )
    return full, zoomed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("Plots/reconnecting_largest_energy_events_preview"),
    )
    args = parser.parse_args()
    isolated = render_event_meshes(
        args.root / "irrev_drop_l_1.31901",
        element_index=72927,
        load=1.31902,
        output_directory=args.output_directory,
        stem="isolated_rank408_slot72927",
    )
    isolated_before_after = render_before_after_event_meshes(
        args.root / "irrev_drop_l_1.31901",
        element_index=72927,
        before_load=1.31901,
        after_load=1.31902,
        output_directory=args.output_directory,
        stem="isolated_rank408_slot72927_before_after",
    )
    largest = render_event_meshes(
        args.root / "irrev_drop_l_1.43444",
        element_index=6198,
        load=1.43445,
        output_directory=args.output_directory,
        stem="largest_rank006_slot6198",
    )
    for path in isolated + isolated_before_after + largest:
        print(path)


if __name__ == "__main__":
    main()

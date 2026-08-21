#!/usr/bin/env python3
"""Render mesh views around the extreme fixed-element-ID Poincare jumps."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting import meshEventPlotting as mesh_plot
from Plotting.render_suspicious_element_meshes import (
    BOX_SIZE,
    FULL_VIEWPORT,
    LOAD_INCREMENT,
    _plot_before_after_mesh_view,
)


DEFAULT_JOB = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
DEFAULT_ELEMENT_ID = 72927
LOAD_PATTERN = re.compile(r"_load=(?P<load>[0-9.eE+-]+)_")


def _load_from_name(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}.")
    return float(match.group("load"))


def _path_for_load(data_directory: Path, load: float) -> Path:
    paths = sorted(data_directory.glob("*.vtu"), key=_load_from_name)
    matches = [path for path in paths if np.isclose(_load_from_name(path), load, atol=1e-10, rtol=0.0)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one VTU at load {load:.8f}, found {len(matches)} in {data_directory}.")
    return matches[0]


def render_spike_pair(
    data_directory: Path,
    *,
    element_id: int,
    before_load: float,
    after_load: float,
    output_directory: Path,
    stem: str,
    zoom_half_width: float = 7.5,
) -> tuple[Path, Path]:
    before_path = _path_for_load(data_directory, before_load)
    after_path = _path_for_load(data_directory, after_load)
    before_state = mesh_plot.load_mesh_state(before_path)
    after_state = mesh_plot.load_mesh_state(after_path)
    for path, state in ((before_path, before_state), (after_path, after_state)):
        if not 0 <= element_id < len(state.triangles):
            raise IndexError(f"Element ID {element_id} is outside {path}.")
    after_centers = mesh_plot.periodic_triangle_centres(
        after_state, load=after_load, box_size=BOX_SIZE
    )
    center_x, center_y = after_centers[element_id]
    viewport = (
        float(center_x - zoom_half_width),
        float(center_x + zoom_half_width),
        float(center_y - zoom_half_width),
        float(center_y + zoom_half_width),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    full = _plot_before_after_mesh_view(
        before_path,
        before_state,
        after_path,
        after_state,
        element_id,
        before_load,
        after_load,
        FULL_VIEWPORT,
        output_directory / f"{stem}_full_mesh.png",
        title=(
            f"fixed element ID {element_id}: load {before_load:.5f} to "
            f"{after_load:.5f}"
        ),
        figsize=(14.0, 7.0),
    )
    zoomed = _plot_before_after_mesh_view(
        before_path,
        before_state,
        after_path,
        after_state,
        element_id,
        before_load,
        after_load,
        viewport,
        output_directory / f"{stem}_zoomed_mesh.png",
        title=(
            f"fixed element ID {element_id}: common local window; "
            f"{before_load:.5f} to {after_load:.5f}"
        ),
        figsize=(12.0, 6.0),
    )
    return full, zoomed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--element-id", type=int, default=DEFAULT_ELEMENT_ID)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args()
    if args.element_id < 0:
        raise ValueError("element-id must be nonnegative.")
    data_directory = args.job / "data"
    pairs = (
        (1.31231, 1.31296, "element_id72927_jump_into_isolated_state"),
        (1.31296, 1.32189, "element_id72927_jump_out_of_isolated_state"),
    )
    for before_load, after_load, stem in pairs:
        for output in render_spike_pair(
            data_directory,
            element_id=args.element_id,
            before_load=before_load,
            after_load=after_load,
            output_directory=args.output_directory,
            stem=stem,
        ):
            print(output)


if __name__ == "__main__":
    main()

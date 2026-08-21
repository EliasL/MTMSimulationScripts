#!/usr/bin/env python3
"""Render meshes for high-energy states of one fixed serialized element slot."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting import meshEventPlotting as mesh_plot
from Plotting.render_suspicious_element_meshes import (
    BOX_SIZE,
    FULL_VIEWPORT,
    _plot_mesh_view,
)


DEFAULT_JOB = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0"
)
DEFAULT_HISTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview/element_id72927_history.csv"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
DEFAULT_ELEMENT_ID = 72927
LOAD_PATTERN = re.compile(r"_load=(?P<load>[0-9.eE+-]+)_")


def _load_from_name(path: Path) -> float:
    match = LOAD_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse load from {path.name}.")
    return float(match.group("load"))


def _high_energy_rows(history_path: Path, *, element_id: int, threshold: float) -> list[dict[str, str]]:
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if int(row["element_id"]) != element_id:
            raise ValueError("The history contains more than one element ID.")
        if float(row["energy"]) > threshold:
            selected.append(row)
    if not selected:
        raise RuntimeError(f"No history rows have energy above {threshold:g}.")
    return selected


def _path_for_load(data_directory: Path, load: float) -> Path:
    paths = sorted(data_directory.glob("*.vtu"), key=_load_from_name)
    matches = [path for path in paths if np.isclose(_load_from_name(path), load, atol=1e-10, rtol=0.0)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one VTU at load {load:.8f}, found {len(matches)} in {data_directory}.")
    return matches[0]


def render_high_energy_states(
    job: Path,
    history_path: Path,
    *,
    element_id: int,
    threshold: float,
    output_directory: Path,
    zoom_half_width: float,
) -> list[Path]:
    if not job.is_dir():
        raise FileNotFoundError(job)
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    if element_id < 0 or not np.isfinite(threshold):
        raise ValueError("element_id must be nonnegative and threshold must be finite.")
    if not np.isfinite(zoom_half_width) or zoom_half_width <= 0:
        raise ValueError("zoom_half_width must be finite and positive.")

    rows = _high_energy_rows(history_path, element_id=element_id, threshold=threshold)
    data_directory = job / "data"
    states = []
    for row in rows:
        load = float(row["load"])
        path = data_directory / row["file"]
        if not path.is_file():
            path = _path_for_load(data_directory, load)
        state = mesh_plot.load_mesh_state(path)
        if not 0 <= element_id < len(state.triangles):
            raise IndexError(f"Element slot {element_id} is outside {path}.")
        states.append((row, path, state, load))

    # Keep the zoom window fixed across all anomalous states.  It is anchored
    # on the state with the largest tracked-element energy, so the comparison
    # does not recenter on a different element after reconnection.
    anchor_row, anchor_path, anchor_state, anchor_load = max(
        states, key=lambda item: float(item[0]["energy"])
    )
    anchor_centres = mesh_plot.periodic_triangle_centres(
        anchor_state, load=anchor_load, box_size=BOX_SIZE
    )
    center_x, center_y = anchor_centres[element_id]
    zoom_viewport = (
        float(center_x - zoom_half_width),
        float(center_x + zoom_half_width),
        float(center_y - zoom_half_width),
        float(center_y + zoom_half_width),
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for row, path, state, load in states:
        load_label = f"{load:.5f}"
        stem = f"element_id{element_id}_high_energy_load_{load_label}"
        title = (
            f"fixed element ID {element_id}; load {load_label}; "
            f"E={float(row['energy']):.6g}"
        )
        outputs.append(
            _plot_mesh_view(
                path,
                state,
                element_id,
                load,
                FULL_VIEWPORT,
                output_directory / f"{stem}_full_mesh.png",
                title=title,
                figsize=(14.0, 7.0),
            )
        )
        outputs.append(
            _plot_mesh_view(
                path,
                state,
                element_id,
                load,
                zoom_viewport,
                output_directory / f"{stem}_zoomed_mesh.png",
                title=title + "; common high-energy window",
                figsize=(8.0, 8.0),
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--element-id", type=int, default=DEFAULT_ELEMENT_ID)
    parser.add_argument("--energy-threshold", type=float, default=0.1)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--zoom-half-width", type=float, default=7.5)
    args = parser.parse_args()
    for output in render_high_energy_states(
        args.job,
        args.history,
        element_id=args.element_id,
        threshold=args.energy_threshold,
        output_directory=args.output_directory,
        zoom_half_width=args.zoom_half_width,
    ):
        print(output)


if __name__ == "__main__":
    main()

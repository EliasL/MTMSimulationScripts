#!/usr/bin/env python3
"""Link a final elastic Poincare hotspot to energy-coloured mesh elements.

The pooled-density entry point cannot display element correspondence in real
space. This composition reuses its density helpers and the standard mesh
renderer and periodic outline helper, preserving VTU cell order throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np

from Plotting.dataFunctions import VTUData
from Plotting.meshEventPlotting import load_mesh_state
from Plotting.pyplotFunctions import add_padding, draw_periodic_element_outlines, plot_mesh
from Plotting.render_final_size_scaling_poincare_density import _final_elastic_map
from Plotting.render_largest_reconnecting_events import (
    DENSITY_GRID_SIZE, DISK_GRID_SIZE, _accumulate_poincare_density_histogram,
    _disk_coordinates, _draw_poincare_density_histogram, _metric_from_total_T,
    _prepare_poincare_axis,
)


def render_hotspot(vtu: Path, output: Path, *, bounds=(-0.22, -0.18, -0.02, 0.02),
                   zoom=3.0, dpi=250):
    xmin, xmax, ymin, ymax = map(float, bounds)
    if xmax <= xmin or ymax <= ymin or not np.isclose(xmax-xmin, ymax-ymin):
        raise ValueError("Hotspot bounds must define a square.")
    data = VTUData(vtu, load_increment=1e-5)
    seed = int(re.search(r"s(\d+)_load=", vtu.name).group(1))
    state = load_mesh_state(vtu)
    assert np.array_equal(data.get_connectivity(), state.triangles)
    elastic = _final_elastic_map(vtu, load_increment=1e-5)
    x, y = _disk_coordinates(_metric_from_total_T(elastic, source=vtu))
    selected = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    indices = np.flatnonzero(selected)
    if not indices.size:
        raise ValueError("No elements in the specified hotspot.")
    energy = data.get_energy_field()
    assert len(energy) == len(x) == len(state.triangles)
    # Freeze limits from the original untiled mesh before rendering copies.
    lower = state.points.min(axis=0)
    upper = state.points.max(axis=0)
    viewport = add_padding((lower[0], upper[0], lower[1], upper[1]), 0.03)
    pink = "#ff1493"
    fig = plt.figure(figsize=(16, 7.2))
    mesh_ax = fig.add_axes((0.035, 0.22, 0.53, 0.65))
    disk_ax = fig.add_axes((0.625, 0.22, 0.34, 0.65))
    energy_bar_ax = fig.add_axes((0.16, 0.11, 0.28, 0.025))
    disk_bar_ax = fig.add_axes((0.67, 0.11, 0.25, 0.025))
    _, cmap, norm = plot_mesh(
        vtu, mesh_property="energy", ax=mesh_ax, add_colorbar=False,
        periodic_load=float(data.load), periodic_box_size=float(data.size[0]),
        load_increment=1e-5, cartesian_viewport_culling=True,
        cartesian_viewport=viewport, unwrap_periodic_triangles=True,
    )
    draw_periodic_element_outlines(
        mesh_ax, state, indices, load=float(data.load), box_size=float(data.size[0]),
        viewport=viewport, color=pink, linewidth=0.30, linestyle="solid",
    )
    mesh_ax.set_xticks([])
    mesh_ax.set_yticks([])
    mesh_ax.set_title(r"(a) Energy; pink outlines select the hotspot")
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(energy)
    fig.colorbar(mappable, cax=energy_bar_ax, orientation="horizontal").set_label(
        r"Element energy $E^{(e)}$"
    )
    histogram = np.zeros((DENSITY_GRID_SIZE, DENSITY_GRID_SIZE), dtype=int)
    visible = _accumulate_poincare_density_histogram(histogram, x, y, zoom=zoom)
    _prepare_poincare_axis(disk_ax, zoom=zoom)
    image = _draw_poincare_density_histogram(disk_ax, histogram, zoom=zoom)
    scale = zoom * DISK_GRID_SIZE / 2
    center = DISK_GRID_SIZE / 2
    disk_ax.add_patch(Rectangle(
        (center+scale*xmin, center+scale*ymin), scale*(xmax-xmin), scale*(ymax-ymin),
        fill=False, edgecolor=pink, linewidth=1.6, zorder=25,
    ))
    disk_ax.set_title(r"(b) Elastic $F_e$; $3\times$ zoom")
    fig.colorbar(image, cax=disk_bar_ax, orientation="horizontal").set_label(
        f"Bin counts (N={len(x)})"
    )
    fig.suptitle(
        f"Seed {seed}, " + rf"$\gamma={data.load:.5f}$" +
        f"; {len(indices):,} / {len(x):,} elements selected", y=0.97,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["element_index_zero_based", "x_p", "y_p", "energy"])
        writer.writerows((int(i), float(x[i]), float(y[i]), float(energy[i])) for i in indices)
    summary = dict(source=str(vtu.resolve()), seed=seed, load=float(data.load),
                   bounds=list(bounds), zoom=zoom, selected_count=len(indices),
                   total_count=len(x), visible_disk_count=visible,
                   viewport=list(map(float, viewport)), energy_colormap="coolwarm",
                   energy_limits=[float(norm.vmin), float(norm.vmax)])
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bounds", type=float, nargs=4, default=(-0.22, -0.18, -0.02, 0.02))
    args = parser.parse_args()
    render_hotspot(args.vtu, args.output, bounds=args.bounds)


if __name__ == "__main__":
    main()

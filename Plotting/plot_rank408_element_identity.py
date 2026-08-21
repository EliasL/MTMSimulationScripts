#!/usr/bin/env python3
"""Mark the suspected pooled-cloud outlier for an element-identity check."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Plotting.render_largest_reconnecting_events import (
    DENSITY_GRID_SIZE,
    DISK_GRID_SIZE,
    _disk_coordinates,
    _metric_from_total_T,
    _prepare_poincare_axis,
    _single_element_poincare_transition,
    _total_T,
)
from MTMath.poincareEnergy import drawCScatter


EVENT_DIRECTORY = Path(
    "/Volumes/data/MTS2D_output/"
    "reversibilityProtocolTest,s200x200l1.0,1e-05,5.1PBCedgeFlipt2epsR0.0"
    "LBFGSEpsg0.0LBFGSEpsx1e-06s0/data/reversibilityData/irrev_drop_l_1.31901"
)
OUTPUT_PATH = Path(
    "Plots/reconnecting_largest_energy_events_preview/"
    "rank408_slot72927_poincare_identity.png"
)
ELEMENT_INDEX = 72927


def _state(directory: Path, prefix: str) -> Path:
    paths = sorted(directory.glob(f"{prefix}.*.vtu"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {prefix} VTU in {directory}, found {len(paths)}.")
    return paths[0]


def _plot_mark(axis: plt.Axes, x: float, y: float, *, zoom: float, color: str, label: str) -> None:
    scale = zoom * DISK_GRID_SIZE / 2.0
    center = DISK_GRID_SIZE / 2.0
    px = center + scale * x
    py = center + scale * y
    axis.scatter(
        [px], [py], s=360, facecolors="none", edgecolors=color,
        linewidths=2.2, zorder=30, label=label,
    )
    axis.scatter([px], [py], s=34, color=color, marker="x", linewidths=1.8, zorder=31)


def main() -> None:
    before_path = _state(EVENT_DIRECTORY, "state0_min_gamma")
    after_path = _state(EVENT_DIRECTORY, "state2_relaxed_gamma_plus")
    transition = _single_element_poincare_transition(
        before_path, after_path, load_increment=1e-5
    )
    after_T = _total_T(after_path, load_increment=1e-5)
    after_metric = _metric_from_total_T(after_T, source=after_path)
    before_x = float(transition.before_x[ELEMENT_INDEX])
    before_y = float(transition.before_y[ELEMENT_INDEX])
    after_x = float(transition.after_x[ELEMENT_INDEX])
    after_y = float(transition.after_y[ELEMENT_INDEX])

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 7.0))
    for axis, zoom, title in zip(
        axes,
        (1.0, 2.0),
        ("full disk", "zoomed around suspected point"),
        strict=True,
    ):
        _prepare_poincare_axis(axis, zoom=zoom)
        density = drawCScatter(
            axis,
            after_metric,
            DISK_GRID_SIZE,
            zoom=zoom,
            density_method="hist",
            density_grid_size=200,
            show_colorbar=False,
            alpha=0.70,
            zorder=4,
        )
        if density is None:
            raise RuntimeError("The event Poincare cloud is empty.")
        _plot_mark(axis, after_x, after_y, zoom=zoom, color="#d62728", label="after; slot 72927")
        _plot_mark(axis, before_x, before_y, zoom=zoom, color="#1f77b4", label="before; slot 72927")
        axis.set_title(title)
        axis.legend(loc="upper left", fontsize=8, framealpha=0.9)

    figure.suptitle(
        "rank 408, event load 1.31901→1.31902; "
        f"slot {ELEMENT_INDEX}; after=({after_x:.4f}, {after_y:.4f})"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    print(f"before=({before_x:.12g}, {before_y:.12g})")
    print(f"after=({after_x:.12g}, {after_y:.12g})")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot angular distributions of pooled Poincare transitions.

Each panel is one spatial averaging cell in the reduced-before Poincare disk.
The plotted angle is the direction of the individual transition
``(after-before)``.  Cells are restricted to the fundamental elastic domain,
and nine populated cells are selected to cover that domain rather than simply
choosing nine neighbouring high-density cells.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MTMath.poincareEnergy import poincareDisk2C
from Plotting.render_largest_reconnecting_events import (
    DEFAULT_JOB,
    _single_element_poincare_transition,
    select_events,
)


DEFAULT_OUTPUT_DIRECTORY = (
    ROOT / "Plots" / "reconnecting_largest_energy_events_preview"
)
SPATIAL_BINS = 30
ANGLE_BINS = 36
MIN_CELL_COUNT = 50
PANEL_COUNT = 9


def _fundamental_domain_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the strict fundamental elastic-domain mask for disk coordinates."""

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Poincare coordinates must have matching shapes.")
    inside = np.isfinite(x) & np.isfinite(y) & (x * x + y * y < 1.0)
    mask = np.zeros(x.shape, dtype=bool)
    if not np.any(inside):
        return mask
    C = poincareDisk2C(x[inside], y[inside])
    C11 = C[:, 0, 0]
    C12 = C[:, 0, 1]
    C22 = C[:, 1, 1]
    mask[inside] = (
        (C12 > 0.0)
        & (2.0 * C12 < C11)
        & (C11 <= C22)
    )
    return mask


def _collect_angle_histograms(
    events: list,
    *,
    spatial_bins: int,
    angle_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pool transition angles into spatial-cell histograms without retaining VTUs."""

    if not events:
        raise ValueError("At least one event is required.")
    if spatial_bins <= 0 or angle_bins <= 0:
        raise ValueError("Both bin counts must be positive.")
    edges = np.linspace(-1.0, 1.0, spatial_bins + 1)
    angle_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
    histogram = np.zeros((spatial_bins, spatial_bins, angle_bins), dtype=np.int64)
    cell_counts = np.zeros((spatial_bins, spatial_bins), dtype=np.int64)
    total_elements = 0

    for event in events:
        transition = _single_element_poincare_transition(
            event.state_paths.state0_min_gamma,
            event.state_paths.state2_relaxed_gamma_plus,
            load_increment=event.load_increment,
        )
        x = np.asarray(transition.before_x, dtype=float)
        y = np.asarray(transition.before_y, dtype=float)
        dx = np.asarray(transition.after_x, dtype=float) - x
        dy = np.asarray(transition.after_y, dtype=float) - y
        if not (x.shape == y.shape == dx.shape == dy.shape):
            raise ValueError("Poincare transition arrays have inconsistent shapes.")
        total_elements += x.size

        fundamental = _fundamental_domain_mask(x, y)
        ix = np.searchsorted(edges, x, side="right") - 1
        iy = np.searchsorted(edges, y, side="right") - 1
        theta = np.arctan2(dy, dx)
        ia = np.searchsorted(angle_edges, theta, side="right") - 1
        # Include the exact right endpoint in the final angular bin.
        ia[ia == angle_bins] = angle_bins - 1
        valid = (
            fundamental
            & (ix >= 0)
            & (ix < spatial_bins)
            & (iy >= 0)
            & (iy < spatial_bins)
            & (ia >= 0)
            & (ia < angle_bins)
            & np.isfinite(theta)
        )
        np.add.at(cell_counts, (iy[valid], ix[valid]), 1)
        np.add.at(histogram, (iy[valid], ix[valid], ia[valid]), 1)
    return histogram, cell_counts, edges, total_elements


def _cell_is_fundamental(x: float, y: float) -> bool:
    return bool(_fundamental_domain_mask(np.array([x]), np.array([y]))[0])


def _select_cells(
    cell_counts: np.ndarray,
    edges: np.ndarray,
    *,
    count_threshold: int,
    number: int,
) -> list[tuple[int, int]]:
    """Choose populated cells that are spatially spread across the domain."""

    centers = 0.5 * (edges[:-1] + edges[1:])
    candidates = []
    for iy, ix in np.argwhere(cell_counts >= count_threshold):
        x, y = float(centers[ix]), float(centers[iy])
        if _cell_is_fundamental(x, y):
            candidates.append((int(ix), int(iy), int(cell_counts[iy, ix])))
    if len(candidates) < number:
        raise RuntimeError(
            f"Only {len(candidates)} fundamental-domain cells have at least "
            f"{count_threshold} elements; cannot select {number}."
        )

    # Start at the most populated cell.  Subsequent cells maximize their
    # distance from the already selected set, with count as a deterministic
    # tie-breaker.  This gives nine genuinely different local distributions.
    candidates.sort(key=lambda item: (-item[2], item[1], item[0]))
    selected = [candidates[0]]
    while len(selected) < number:
        selected_xy = np.array(
            [[centers[ix], centers[iy]] for ix, iy, _ in selected], dtype=float
        )
        remaining = [item for item in candidates if item not in selected]
        scored = []
        for item in remaining:
            ix, iy, count = item
            point = np.array([centers[ix], centers[iy]], dtype=float)
            separation = float(np.min(np.linalg.norm(selected_xy - point, axis=1)))
            scored.append((separation, count, -iy, -ix, item))
        scored.sort(reverse=True)
        selected.append(scored[0][-1])
    return [(ix, iy) for ix, iy, _ in selected]


def _write_cell_manifest(path: Path, selected: list[tuple[int, int]], counts: np.ndarray, edges: np.ndarray) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("panel", "cell_ix", "cell_iy", "x_center", "y_center", "count"))
        for panel, (ix, iy) in enumerate(selected, start=1):
            writer.writerow((panel, ix, iy, centers[ix], centers[iy], counts[iy, ix]))


def render_theta_distributions(
    events: list,
    output_path: Path,
    *,
    dpi: int,
    spatial_bins: int = SPATIAL_BINS,
    angle_bins: int = ANGLE_BINS,
    count_threshold: int = MIN_CELL_COUNT,
) -> tuple[Path, Path]:
    """Create the 3x3 theta-distribution figure and its cell manifest."""

    histogram, counts, edges, total_elements = _collect_angle_histograms(
        events, spatial_bins=spatial_bins, angle_bins=angle_bins
    )
    selected = _select_cells(
        counts,
        edges,
        count_threshold=count_threshold,
        number=PANEL_COUNT,
    )
    angle_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    angle_width = angle_edges[1] - angle_edges[0]
    centers = 0.5 * (edges[:-1] + edges[1:])

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(12.0, 9.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(3, 3)
    for panel, (ix, iy) in enumerate(selected):
        axis = axes.flat[panel]
        values = histogram[iy, ix].astype(float)
        if values.sum() <= 0:
            raise RuntimeError(f"Selected cell ({ix}, {iy}) has no angle data.")
        probability = values / values.sum() / angle_width
        axis.bar(
            angle_centers,
            probability,
            width=angle_width,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.9,
        )
        axis.set_title(
            f"cell ({ix}, {iy}),  "
            rf"$(x,y)=({centers[ix]:.3f},{centers[iy]:.3f})$,  "
            f"N={int(values.sum())}",
            fontsize=9,
        )
        axis.grid(axis="y", linewidth=0.35, alpha=0.35)
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks((-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi))
        axis.set_xticklabels((r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"))
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\theta = \mathrm{atan2}(\Delta y,\Delta x)$")
    for axis in axes[:, 0]:
        axis.set_ylabel("fraction per radian")
    figure.suptitle(
        "Poincare transition directions in nine fundamental-domain cells\n"
        f"{len(events)} saved events, {total_elements:,} element transitions; "
        f"{spatial_bins}x{spatial_bins} spatial bins",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    manifest_path = output_path.with_name(output_path.stem + "_cells.csv")
    _write_cell_manifest(manifest_path, selected, counts, edges)
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--top", type=int, default=None, help="Use the largest saved events only.")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dpi", type=int, default=234)
    parser.add_argument("--spatial-bins", type=int, default=SPATIAL_BINS)
    parser.add_argument("--angle-bins", type=int, default=ANGLE_BINS)
    parser.add_argument("--min-cell-count", type=int, default=MIN_CELL_COUNT)
    args = parser.parse_args()
    if args.dpi <= 0 or args.spatial_bins <= 0 or args.angle_bins <= 0 or args.min_cell_count <= 0:
        raise ValueError("dpi and all bin/count settings must be positive.")
    if args.top is not None and args.top <= 0:
        raise ValueError("--top must be positive.")
    events = select_events(args.job, number=args.top, saved_only=True)
    output_path = args.output_directory / "aggregate_theta_distributions.png"
    output, manifest = render_theta_distributions(
        events,
        output_path,
        dpi=args.dpi,
        spatial_bins=args.spatial_bins,
        angle_bins=args.angle_bins,
        count_threshold=args.min_cell_count,
    )
    print(output)
    print(manifest)


if __name__ == "__main__":
    main()

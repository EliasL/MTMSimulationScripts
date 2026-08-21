#!/usr/bin/env python3
"""Render maximal-flow angle, magnitude, and dispersion heatmaps."""

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
from matplotlib.colors import LogNorm, Normalize

from Plotting.render_largest_reconnecting_events import (
    DEFAULT_JOB,
    _disk_coordinates,
    _draw_poincare_grid_image,
    _draw_poincare_loss_of_ellipticity_limit,
    _loss_of_ellipticity_boundary,
    _prepare_poincare_axis,
    _single_element_poincare_transition,
    select_events,
)
from MTMath.reduction import plastic_reduction


DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
# Match the 200x200 Poincare density histogram used by the aggregate event
# plots.  The zoomed view below covers the central radius 1/2 of the disk.
SPATIAL_BINS = 200
MIN_CELL_COUNT = 1
ZOOMED_DISK_ZOOM = 2.0
PREVIOUS_MESH_ZOOM = 3.5


def _collect_vectors(
    events: list,
    *,
    bins: int,
    zoom: float,
    coordinate_source: str = "after",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Pool finite transitions and retain their spatial-bin membership."""

    if not events:
        raise ValueError("At least one event is required.")
    if bins <= 0 or not np.isfinite(zoom) or zoom < 1:
        raise ValueError("bins must be positive and zoom must be at least one.")
    if coordinate_source not in {"before", "after", "after_reduced"}:
        raise ValueError(
            "coordinate_source must be 'before', 'after', or 'after_reduced'."
        )
    radius = 1.0 / zoom
    edges = np.linspace(-radius, radius, bins + 1)
    u_parts, v_parts, ix_parts, iy_parts = [], [], [], []
    count = np.zeros((bins, bins), dtype=np.int64)
    total_elements = 0
    for event in events:
        transition = _single_element_poincare_transition(
            event.state_paths.state0_min_gamma,
            event.state_paths.state2_relaxed_gamma_plus,
            load_increment=event.load_increment,
        )
        before_x = np.asarray(transition.before_x, dtype=float)
        before_y = np.asarray(transition.before_y, dtype=float)
        after_x = np.asarray(transition.after_x, dtype=float)
        after_y = np.asarray(transition.after_y, dtype=float)
        if coordinate_source == "before":
            x, y = before_x, before_y
        elif coordinate_source == "after":
            x, y = after_x, after_y
        else:
            reduced_after_metric, _ = plastic_reduction(
                transition.after_metric, compute_M=True
            )
            x, y = _disk_coordinates(reduced_after_metric)
            # The plotted vector ends at the finally reduced after-state.
            after_x, after_y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        # The before view keeps the raw Delta-T endpoint; the after-reduced view
        # shows the endpoint after applying the final plastic reduction.
        u = after_x - before_x
        v = after_y - before_y
        if not (x.shape == y.shape == u.shape == v.shape):
            raise ValueError("Poincare coordinates and vector arrays have inconsistent shapes.")
        total_elements += x.size
        ix = np.searchsorted(edges, x, side="right") - 1
        iy = np.searchsorted(edges, y, side="right") - 1
        valid = (
            (ix >= 0)
            & (ix < bins)
            & (iy >= 0)
            & (iy < bins)
            & (x * x + y * y <= radius * radius)
            & np.isfinite(u)
            & np.isfinite(v)
        )
        if np.any(valid):
            ix_valid = ix[valid].astype(np.int16, copy=False)
            iy_valid = iy[valid].astype(np.int16, copy=False)
            u_parts.append(u[valid].astype(np.float32, copy=False))
            v_parts.append(v[valid].astype(np.float32, copy=False))
            ix_parts.append(ix_valid)
            iy_parts.append(iy_valid)
            np.add.at(count, (iy_valid, ix_valid), 1)
    if not u_parts:
        raise RuntimeError("No finite Poincare transitions lie inside the requested disk.")
    return (
        np.concatenate(u_parts),
        np.concatenate(v_parts),
        np.concatenate(ix_parts),
        np.concatenate(iy_parts),
        count,
        total_elements,
    )


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-15:
        return None
    return np.asarray(vector, dtype=float) / norm


def _max_abs_projection_direction(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the axis maximizing ``sum(abs(u*d_x + v*d_y))``.

    For fixed projection signs, the maximizer is the normalized signed vector
    sum.  Iterating that update from several deterministic starting axes gives
    the global maximal-flow axis in the 2-D cells here while avoiding a dense
    angular grid, including for the very populous central cell.
    """

    if u.size == 0 or u.shape != v.shape:
        raise ValueError("A non-empty pair of matching vector arrays is required.")
    vectors = np.column_stack((u.astype(float), v.astype(float)))
    nonzero = np.linalg.norm(vectors, axis=1) > 1e-15
    vectors = vectors[nonzero]
    if vectors.size == 0:
        raise ValueError("A maximal-flow direction is undefined for zero vectors.")
    starts: list[np.ndarray] = []
    mean_direction = _unit(np.sum(vectors, axis=0))
    if mean_direction is not None:
        starts.append(mean_direction)
    covariance = vectors.T @ vectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = _unit(eigenvectors[:, int(np.argmax(eigenvalues))])
    if principal is not None:
        starts.append(principal)
    starts.extend((np.array((1.0, 0.0)), np.array((0.0, 1.0))))
    largest_vector = _unit(vectors[int(np.argmax(np.sum(vectors * vectors, axis=1)))])
    if largest_vector is not None:
        starts.append(largest_vector)

    best_direction = None
    best_value = -np.inf
    for start in starts:
        direction = start
        for _ in range(50):
            projection = vectors @ direction
            signs = np.where(projection >= 0.0, 1.0, -1.0)
            updated = _unit(np.sum(vectors * signs[:, None], axis=0))
            if updated is None:
                break
            if float(np.dot(updated, direction)) < 0.0:
                updated = -updated
            if 1.0 - abs(float(np.dot(updated, direction))) < 1e-12:
                direction = updated
                break
            direction = updated
        value = float(np.sum(np.abs(vectors @ direction)))
        if value > best_value:
            best_value = value
            best_direction = direction
    if best_direction is None or not np.isfinite(best_value):
        raise RuntimeError("Maximal-flow optimization failed.")
    # The axis is sign-degenerate.  Align it with the mean when possible so
    # the displayed representative is deterministic before reducing modulo pi.
    if mean_direction is not None and float(np.dot(best_direction, mean_direction)) < 0:
        best_direction = -best_direction
    return best_direction, best_value


def _cell_metrics(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    direction, total_magnitude = _max_abs_projection_direction(u, v)
    vectors = np.column_stack((u.astype(float), v.astype(float)))
    nonzero = np.linalg.norm(vectors, axis=1) > 1e-15
    vectors = vectors[nonzero]
    projection_magnitudes = np.abs(vectors @ direction)
    angle = float(np.mod(np.arctan2(direction[1], direction[0]), np.pi))
    # Axial circular standard deviation: directions separated by pi are
    # equivalent, so the doubled-angle resultant is the appropriate measure.
    vector_angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    resultant = float(np.abs(np.mean(np.exp(2j * vector_angles))))
    resultant = float(np.clip(resultant, 1e-15, 1.0))
    angle_std = float(np.sqrt(-0.5 * np.log(resultant)))
    magnitude_std = float(np.std(projection_magnitudes))
    return angle, total_magnitude, angle_std, magnitude_std


def _compute_heatmaps(
    u: np.ndarray,
    v: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    count: np.ndarray,
    *,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if min_count <= 0:
        raise ValueError("min_count must be positive.")
    bins = count.shape[0]
    angle = np.full((bins, bins), np.nan, dtype=float)
    magnitude = np.full((bins, bins), np.nan, dtype=float)
    angle_std = np.full((bins, bins), np.nan, dtype=float)
    magnitude_std = np.full((bins, bins), np.nan, dtype=float)
    # Sort once by cell rather than scanning all pooled transitions for every
    # populated bin.  This matters for the 200x200 grid, where many more cells
    # can pass the count threshold than in the original 30x30 view.
    flat = iy.astype(np.int64) * bins + ix.astype(np.int64)
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_flat)) + 1]
    stops = np.r_[starts[1:], sorted_flat.size]
    for start, stop in zip(starts, stops, strict=True):
        flat_bin = int(sorted_flat[start])
        iy_bin, ix_bin = divmod(flat_bin, bins)
        if stop - start < min_count:
            continue
        members = order[start:stop]
        metrics = _cell_metrics(u[members], v[members])
        angle[iy_bin, ix_bin], magnitude[iy_bin, ix_bin], angle_std[iy_bin, ix_bin], magnitude_std[iy_bin, ix_bin] = metrics
    if not np.any(np.isfinite(angle)):
        raise RuntimeError(f"No spatial cells contain at least {min_count} transitions.")
    return angle, magnitude, angle_std, magnitude_std


def _heatmap_image(axis, values: np.ndarray, *, zoom: float, cmap: str, norm: Normalize):
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("white", alpha=0.0)
    return _draw_poincare_grid_image(
        axis,
        np.ma.masked_invalid(values),
        zoom=zoom,
        cmap=color_map,
        norm=norm,
        alpha=1.0,
        zorder=5,
    )


def _write_manifest(path: Path, count: np.ndarray, metrics: tuple[np.ndarray, ...], edges: np.ndarray) -> None:
    centers = 0.5 * (edges[:-1] + edges[1:])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cell_ix", "cell_iy", "x_center", "y_center", "count", "flow_angle_mod_pi", "total_projected_magnitude", "angle_std_rad", "projected_magnitude_std"))
        for iy_bin, ix_bin in np.argwhere(np.isfinite(metrics[0])):
            writer.writerow((ix_bin, iy_bin, centers[ix_bin], centers[iy_bin], count[iy_bin, ix_bin], *(metric[iy_bin, ix_bin] for metric in metrics)))


def render_maxflow_heatmaps(
    events: list,
    output_path: Path,
    *,
    dpi: int,
    bins: int = SPATIAL_BINS,
    min_count: int = MIN_CELL_COUNT,
    zoom: float = 1.0,
    coordinate_source: str = "after",
) -> tuple[Path, Path]:
    u, v, ix, iy, count, total_elements = _collect_vectors(
        events,
        bins=bins,
        zoom=zoom,
        coordinate_source=coordinate_source,
    )
    metrics = _compute_heatmaps(u, v, ix, iy, count, min_count=min_count)
    angle, magnitude, angle_std, magnitude_std = metrics
    edges = np.linspace(-1.0 / zoom, 1.0 / zoom, bins + 1)
    boundary = _loss_of_ellipticity_boundary()

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 12.0), constrained_layout=True)
    positive_magnitude = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if positive_magnitude.size == 0:
        raise RuntimeError("The projected-flow heatmap has no positive values.")
    panels = (
        (angle, "twilight", Normalize(0.0, np.pi), "flow angle (mod $\\pi$)", (0.0, np.pi / 2.0, np.pi), (r"$0$", r"$\\pi/2$", r"$\\pi$")),
        (magnitude, "viridis", LogNorm(float(np.min(positive_magnitude)), float(np.max(positive_magnitude))), "total projected magnitude", None, None),
        (angle_std, "magma", Normalize(0.0, float(np.nanmax(angle_std))), "axial angle std (rad)", None, None),
        (magnitude_std, "plasma", Normalize(0.0, float(np.nanmax(magnitude_std))), "projected-magnitude std", None, None),
    )
    for axis, (values, cmap, norm, label, ticks, ticklabels) in zip(axes.flat, panels, strict=True):
        _prepare_poincare_axis(axis, zoom=zoom)
        image = _heatmap_image(axis, values, zoom=zoom, cmap=cmap, norm=norm)
        _draw_poincare_loss_of_ellipticity_limit(axis, boundary, zoom=zoom)
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
        colorbar.set_label(label)
        if ticks is not None:
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(ticklabels)
    axes[0, 0].set_title("(a) maximal-flow axis")
    axes[0, 1].set_title("(b) absolute projected flow")
    axes[1, 0].set_title("(c) directional dispersion")
    axes[1, 1].set_title("(d) magnitude dispersion")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    manifest = output_path.with_name(output_path.stem + "_cells.csv")
    _write_manifest(manifest, count, metrics, edges)
    return output_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dpi", type=int, default=234)
    parser.add_argument("--bins", type=int, default=SPATIAL_BINS)
    parser.add_argument("--min-count", type=int, default=MIN_CELL_COUNT)
    parser.add_argument("--zoom", type=float, default=ZOOMED_DISK_ZOOM)
    parser.add_argument("--previous-zoom", type=float, default=PREVIOUS_MESH_ZOOM)
    args = parser.parse_args()
    if (
        args.dpi <= 0
        or args.bins <= 0
        or args.min_count <= 0
        or not np.isfinite(args.zoom)
        or args.zoom < 1
        or not np.isfinite(args.previous_zoom)
        or args.previous_zoom < 1
    ):
        raise ValueError("dpi, bins, min-count must be positive and zooms must be at least one.")
    if args.top is not None and args.top <= 0:
        raise ValueError("--top must be positive.")
    events = select_events(args.job, number=args.top, saved_only=True)
    output, manifest = render_maxflow_heatmaps(
        events,
        args.output_directory / "aggregate_maxflow_heatmaps.png",
        dpi=args.dpi,
        bins=args.bins,
        min_count=args.min_count,
        zoom=args.zoom,
        coordinate_source="after",
    )
    previous_output, previous_manifest = render_maxflow_heatmaps(
        events,
        args.output_directory / "aggregate_maxflow_heatmaps_previous.png",
        dpi=args.dpi,
        bins=args.bins,
        min_count=args.min_count,
        zoom=args.previous_zoom,
        coordinate_source="before",
    )
    reduced_after_output, reduced_after_manifest = render_maxflow_heatmaps(
        events,
        args.output_directory / "aggregate_maxflow_heatmaps_after_reduced.png",
        dpi=args.dpi,
        bins=args.bins,
        min_count=args.min_count,
        zoom=args.previous_zoom,
        coordinate_source="after_reduced",
    )
    print(output)
    print(manifest)
    print(previous_output)
    print(previous_manifest)
    print(reduced_after_output)
    print(reduced_after_manifest)


if __name__ == "__main__":
    main()

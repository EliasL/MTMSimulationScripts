#!/usr/bin/env python3
"""Render maximal-flow heatmaps in Cauchy-stress coordinates."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from dataclasses import dataclass
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
import cmocean

from Plotting.render_largest_reconnecting_events import (
    DEFAULT_JOB,
    _loss_of_ellipticity_boundary,
    _stress_coordinates_from_T,
    _stress_coordinates_from_vtu,
    _stress_loss_of_ellipticity_limit,
    _total_T_transition,
    select_events,
)
from Plotting.render_poincare_maxflow_heatmaps import (
    MIN_CELL_COUNT,
    SPATIAL_BINS,
    _compute_heatmaps,
    _write_manifest,
)
from Plotting.heatmap_cache import (
    heatmap_cache_path,
    load_heatmap_cache,
    save_heatmap_cache,
)


DEFAULT_OUTPUT_DIRECTORY = ROOT / "Plots/reconnecting_largest_energy_events_preview"
STRESS_Y_LIMIT: float | None = None
STRESS_EXTENT = 1.2
# The serialized total-T components are rounded independently of the exported
# stress.  Regular slots agree at this tolerance; the known ill-conditioned
# branch is many orders of magnitude farther away.
TOTAL_T_STRESS_TOLERANCE = 1e-2


@dataclass(frozen=True)
class TotalTStressRejection:
    """One total-``T`` endpoint pair that cannot reproduce its VTU stress."""

    event_rank: int
    load_step: int
    element_index: int
    failed_states: str
    maximum_coordinate_error: float
    maximum_T_frobenius_norm: float


def _stress_coordinates_from_usable_T(
    T: np.ndarray, *, source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate total-``T`` stress only for finite, orientation-preserving slots."""

    T = np.asarray(T, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (2, 2):
        raise ValueError(f"Expected (..., 2, 2) total T matrices in {source}, got {T.shape}.")
    determinant = np.linalg.det(T)
    usable = np.all(np.isfinite(T), axis=(-2, -1)) & np.isfinite(determinant) & (
        determinant > 0.0
    )
    sigma_xy = np.full(len(T), np.nan, dtype=float)
    normal_difference = np.full(len(T), np.nan, dtype=float)
    if np.any(usable):
        sigma_xy[usable], normal_difference[usable] = _stress_coordinates_from_T(
            T[usable], source=source
        )
    return sigma_xy, normal_difference, usable


def _total_T_stress_validity(
    event,
    *,
    before_sigma_xy: np.ndarray,
    before_normal_difference: np.ndarray,
    after_sigma_xy: np.ndarray,
    after_normal_difference: np.ndarray,
    before_T: np.ndarray,
    after_T: np.ndarray,
    before_usable: np.ndarray,
    after_usable: np.ndarray,
) -> tuple[np.ndarray, list[TotalTStressRejection]]:
    """Validate total-``T`` stress against exported VTU stress componentwise."""

    before_vtu_xy, before_vtu_normal = _stress_coordinates_from_vtu(
        event.state_paths.state0_min_gamma, load_increment=event.load_increment
    )
    after_vtu_xy, after_vtu_normal = _stress_coordinates_from_vtu(
        event.state_paths.state2_relaxed_gamma_plus, load_increment=event.load_increment
    )
    before_error = np.full(len(before_T), np.inf, dtype=float)
    after_error = np.full(len(after_T), np.inf, dtype=float)
    before_error[before_usable] = np.maximum(
        np.abs(before_sigma_xy[before_usable] - before_vtu_xy[before_usable]),
        np.abs(before_normal_difference[before_usable] - before_vtu_normal[before_usable]),
    )
    after_error[after_usable] = np.maximum(
        np.abs(after_sigma_xy[after_usable] - after_vtu_xy[after_usable]),
        np.abs(after_normal_difference[after_usable] - after_vtu_normal[after_usable]),
    )
    if not (before_error.shape == after_error.shape == (len(before_T),)):
        raise ValueError("Total-T stress validation produced inconsistent array shapes.")
    valid = (before_error <= TOTAL_T_STRESS_TOLERANCE) & (
        after_error <= TOTAL_T_STRESS_TOLERANCE
    )
    rejected: list[TotalTStressRejection] = []
    for element_index in np.flatnonzero(~valid):
        failed_states = ";".join(
            state
            for state, usable, error in (
                ("state0", before_usable[element_index], before_error[element_index]),
                ("state2", after_usable[element_index], after_error[element_index]),
            )
            if not usable or error > TOTAL_T_STRESS_TOLERANCE
        )
        rejected.append(
            TotalTStressRejection(
                event_rank=event.rank,
                load_step=event.load_step,
                element_index=int(element_index),
                failed_states=failed_states,
                maximum_coordinate_error=float(
                    max(before_error[element_index], after_error[element_index])
                ),
                maximum_T_frobenius_norm=float(
                    max(
                        np.linalg.norm(before_T[element_index]),
                        np.linalg.norm(after_T[element_index]),
                    )
                ),
            )
        )
    return valid, rejected


def _collect_stress_vectors(
    events: list,
    *,
    bins: int,
    min_count: int,
    vector_source: str = "delta_T",
    exclude_unreconstructable_T: bool = False,
    stress_extent: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[TotalTStressRejection],
]:
    """Pool stress transitions and assign spatial bins.

    ``delta_T`` uses the full reconnection-invariant total ``T`` at both
    endpoints.  Its flow is therefore ``sigma(T_after) - sigma(T_before)``,
    with ``T_after = Delta_T @ T_before``.  ``local_stress`` independently
    reads the Cauchy-stress fields directly from the before/after VTUs.
    """

    if not events:
        raise ValueError("At least one event is required.")
    if bins <= 0 or min_count <= 0:
        raise ValueError("bins and min_count must be positive.")
    if vector_source not in {"delta_T", "local_stress"}:
        raise ValueError("vector_source must be 'delta_T' or 'local_stress'.")
    before_x_parts: list[np.ndarray] = []
    before_y_parts: list[np.ndarray] = []
    after_x_parts: list[np.ndarray] = []
    after_y_parts: list[np.ndarray] = []
    rejected_total_T: list[TotalTStressRejection] = []

    for event in events:
        if vector_source == "delta_T":
            transition = _total_T_transition(
                event.state_paths.state0_min_gamma,
                event.state_paths.state2_relaxed_gamma_plus,
                load_increment=event.load_increment,
            )
            before_sigma_xy, before_normal_difference, before_usable = _stress_coordinates_from_usable_T(
                transition.before_T,
                source=f"{event.state_directory} state0 total T",
            )
            after_sigma_xy, after_normal_difference, after_usable = _stress_coordinates_from_usable_T(
                transition.after_T,
                source=f"{event.state_directory} state2 total T",
            )
            valid, rejected = _total_T_stress_validity(
                event,
                before_sigma_xy=before_sigma_xy,
                before_normal_difference=before_normal_difference,
                after_sigma_xy=after_sigma_xy,
                after_normal_difference=after_normal_difference,
                before_T=transition.before_T,
                after_T=transition.after_T,
                before_usable=before_usable,
                after_usable=after_usable,
            )
            if rejected and not exclude_unreconstructable_T:
                details = ", ".join(
                    f"rank {item.event_rank}, slot {item.element_index}" for item in rejected
                )
                raise ValueError(
                    "Full T does not reproduce the exported VTU stress for "
                    f"{details}. Re-run with exclude_unreconstructable_T=True "
                    "only if an explicit rejection audit is desired."
                )
            rejected_total_T.extend(rejected)
            before_sigma_xy = before_sigma_xy[valid]
            before_normal_difference = before_normal_difference[valid]
            after_sigma_xy = after_sigma_xy[valid]
            after_normal_difference = after_normal_difference[valid]
        else:
            before_sigma_xy, before_normal_difference = _stress_coordinates_from_vtu(
                event.state_paths.state0_min_gamma,
                load_increment=event.load_increment,
            )
            after_sigma_xy, after_normal_difference = _stress_coordinates_from_vtu(
                event.state_paths.state2_relaxed_gamma_plus,
                load_increment=event.load_increment,
            )
        if not (
            before_sigma_xy.shape
            == before_normal_difference.shape
            == after_sigma_xy.shape
            == after_normal_difference.shape
        ):
            raise ValueError(f"Stress transition arrays have inconsistent shapes in {event.state_directory}.")
        # x is (sigma22-sigma11)/2; y is sigma_xy.
        before_x_parts.append(before_normal_difference.astype(np.float32, copy=False))
        before_y_parts.append(before_sigma_xy.astype(np.float32, copy=False))
        after_x_parts.append(after_normal_difference.astype(np.float32, copy=False))
        after_y_parts.append(after_sigma_xy.astype(np.float32, copy=False))
        # VTUData owns large meshio/VTK objects.  Explicit collection keeps a
        # 66-event aggregate bounded instead of retaining native mesh buffers
        # until Python's cyclic collector happens to run.
        del before_sigma_xy, before_normal_difference
        del after_sigma_xy, after_normal_difference
        gc.collect()

    before_x = np.concatenate(before_x_parts)
    before_y = np.concatenate(before_y_parts)
    after_x = np.concatenate(after_x_parts)
    after_y = np.concatenate(after_y_parts)
    u = after_x - before_x
    v = after_y - before_y
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("Stress-space transition vectors contain non-finite values.")

    # The heatmap location is always the after-state stress coordinate.  The
    # vector attached to each occupied cell is the transition from the
    # corresponding before-state stress to that after-state stress.
    if stress_extent is None:
        extent = max(
            float(np.max(np.abs(after_x))),
            float(np.max(np.abs(after_y))),
        )
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("The final-reduced stress cloud has no finite nonzero extent.")
        extent *= 1.02
    else:
        extent = float(stress_extent)
        if not np.isfinite(extent) or extent <= 0.0:
            raise ValueError("stress_extent must be finite and positive.")
    edges = np.linspace(-extent, extent, bins + 1)
    ix = np.searchsorted(edges, after_x, side="right") - 1
    iy = np.searchsorted(edges, after_y, side="right") - 1
    valid = (
        (ix >= 0)
        & (ix < bins)
        & (iy >= 0)
        & (iy < bins)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    if not np.any(valid):
        raise RuntimeError("No finite final-reduced stress transitions fit the plot extent.")
    ix_valid = ix[valid].astype(np.int32, copy=False)
    iy_valid = iy[valid].astype(np.int32, copy=False)
    u_valid = u[valid].astype(np.float32, copy=False)
    v_valid = v[valid].astype(np.float32, copy=False)
    count = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(count, (iy_valid, ix_valid), 1)
    if not np.any(count >= min_count):
        raise RuntimeError(f"No stress cells contain at least {min_count} transitions.")
    return u_valid, v_valid, ix_valid, iy_valid, count, edges, rejected_total_T


def _stress_heatmap_image(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    cmap,
    norm: Normalize,
):
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("white", alpha=0.0)
    return axis.imshow(
        np.ma.masked_invalid(values),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=color_map,
        norm=norm,
        alpha=1.0,
        zorder=5,
    )


def _write_stress_manifest(
    path: Path,
    count: np.ndarray,
    metrics: tuple[np.ndarray, ...],
    edges: np.ndarray,
) -> None:
    _write_manifest(path, count, metrics, edges)


def _write_total_T_rejections(
    path: Path, rejected: list[TotalTStressRejection]
) -> None:
    """Write every explicitly excluded total-``T`` slot for later inspection."""

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "event_rank",
                "load_step",
                "element_index",
                "failed_states",
                "maximum_coordinate_error",
                "maximum_T_frobenius_norm",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "event_rank": item.event_rank,
                "load_step": item.load_step,
                "element_index": item.element_index,
                "failed_states": item.failed_states,
                "maximum_coordinate_error": item.maximum_coordinate_error,
                "maximum_T_frobenius_norm": item.maximum_T_frobenius_norm,
            }
            for item in rejected
        )


def _write_stress_vector_chunk(
    path: Path,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[TotalTStressRejection],
    ],
) -> None:
    """Persist a bounded event batch for a later exact aggregate render."""

    u, v, ix, iy, count, edges, rejected = data
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=u,
        v=v,
        ix=ix,
        iy=iy,
        count=count,
        edges=edges,
        rejection_rank=np.asarray([item.event_rank for item in rejected], dtype=np.int64),
        rejection_load_step=np.asarray([item.load_step for item in rejected], dtype=np.int64),
        rejection_element_index=np.asarray([item.element_index for item in rejected], dtype=np.int64),
        rejection_failed_states=np.asarray([item.failed_states for item in rejected], dtype=str),
        rejection_error=np.asarray([item.maximum_coordinate_error for item in rejected], dtype=float),
        rejection_T_norm=np.asarray([item.maximum_T_frobenius_norm for item in rejected], dtype=float),
    )


def _load_stress_vector_chunks(
    paths: list[Path],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[TotalTStressRejection],
]:
    """Load compatible event chunks without changing any per-transition data."""

    if not paths:
        raise ValueError("At least one stress-vector chunk is required.")
    u_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    ix_parts: list[np.ndarray] = []
    iy_parts: list[np.ndarray] = []
    count: np.ndarray | None = None
    edges: np.ndarray | None = None
    rejected: list[TotalTStressRejection] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as chunk:
            chunk_edges = np.asarray(chunk["edges"], dtype=float)
            chunk_count = np.asarray(chunk["count"], dtype=np.int64)
            if edges is None:
                edges = chunk_edges
                count = np.zeros_like(chunk_count)
            elif not np.array_equal(chunk_edges, edges) or chunk_count.shape != count.shape:
                raise ValueError(f"Stress-vector chunk {path} has incompatible bin edges.")
            u_parts.append(np.asarray(chunk["u"], dtype=np.float32))
            v_parts.append(np.asarray(chunk["v"], dtype=np.float32))
            ix_parts.append(np.asarray(chunk["ix"], dtype=np.int32))
            iy_parts.append(np.asarray(chunk["iy"], dtype=np.int32))
            count += chunk_count
            for values in zip(
                chunk["rejection_rank"],
                chunk["rejection_load_step"],
                chunk["rejection_element_index"],
                chunk["rejection_failed_states"],
                chunk["rejection_error"],
                chunk["rejection_T_norm"],
                strict=True,
            ):
                rank, load_step, element_index, failed_states, error, T_norm = values
                rejected.append(
                    TotalTStressRejection(
                        event_rank=int(rank),
                        load_step=int(load_step),
                        element_index=int(element_index),
                        failed_states=str(failed_states),
                        maximum_coordinate_error=float(error),
                        maximum_T_frobenius_norm=float(T_norm),
                    )
                )
    if count is None or edges is None or not u_parts:
        raise ValueError("The stress-vector chunks contain no transitions.")
    return (
        np.concatenate(u_parts),
        np.concatenate(v_parts),
        np.concatenate(ix_parts),
        np.concatenate(iy_parts),
        count,
        edges,
        rejected,
    )


def render_stress_maxflow_heatmaps(
    events: list | None,
    output_path: Path,
    *,
    dpi: int,
    bins: int = SPATIAL_BINS,
    min_count: int = MIN_CELL_COUNT,
    y_limit: float | None = STRESS_Y_LIMIT,
    vector_source: str = "delta_T",
    exclude_unreconstructable_T: bool = False,
    stress_extent: float | None = STRESS_EXTENT,
    force_recompute: bool = False,
    vector_data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[TotalTStressRejection],
    ] | None = None,
) -> tuple[Path, Path]:
    if y_limit is not None and (not np.isfinite(y_limit) or y_limit <= 0):
        raise ValueError("y_limit must be finite and positive when supplied.")
    if vector_source not in {"delta_T", "local_stress"}:
        raise ValueError("vector_source must be 'delta_T' or 'local_stress'.")
    if events is None:
        raise ValueError("events are required for a content-addressed heatmap cache.")
    cache_path = heatmap_cache_path(
        output_path.parent / "cache" / "heatmaps",
        "stress",
        events,
        parameters={
            "bins": bins,
            "min_count": min_count,
            "vector_source": vector_source,
            "stress_extent": stress_extent,
            "metric_algorithm": "max_abs_projection_v1",
        },
    )
    cached = None if force_recompute else load_heatmap_cache(cache_path)
    if cached is not None:
        *metrics, count, edges = cached
        metrics = tuple(metrics)
        rejected_total_T = []
        print(f"Loaded stress heatmap cache: {cache_path}")
    else:
        if vector_data is None:
            if events is None:
                raise ValueError("events are required when no precomputed vectors are supplied.")
            u, v, ix, iy, count, edges, rejected_total_T = _collect_stress_vectors(
                events,
                bins=bins,
                min_count=min_count,
                vector_source=vector_source,
                exclude_unreconstructable_T=exclude_unreconstructable_T,
                stress_extent=stress_extent,
            )
        else:
            u, v, ix, iy, count, edges, rejected_total_T = vector_data
        metrics = _compute_heatmaps(u, v, ix, iy, count, min_count=min_count)
        save_heatmap_cache(cache_path, metrics, count, edges)
        print(f"Saved stress heatmap cache: {cache_path}")
    angle, magnitude, angle_std, magnitude_std = metrics
    extent = (float(edges[0]), float(edges[-1]), float(edges[0]), float(edges[-1]))
    plot_y_limit = float(extent[3]) if y_limit is None else float(y_limit)
    boundary = _loss_of_ellipticity_boundary()
    stress_limit_x, stress_limit_y = _stress_loss_of_ellipticity_limit(boundary)

    positive_magnitude = magnitude[np.isfinite(magnitude) & (magnitude > 0.0)]
    if positive_magnitude.size == 0:
        raise RuntimeError("The stress projected-flow heatmap has no positive values.")
    if not np.any(np.isfinite(angle_std)) or not np.any(np.isfinite(magnitude_std)):
        raise RuntimeError("The stress dispersion heatmaps have no finite values.")
    panels = (
        (
            angle,
            cmocean.cm.phase,
            Normalize(0.0, np.pi),
            "flow angle (mod $\\pi$)",
            (0.0, np.pi / 2.0, np.pi),
            (r"$0$", r"$\\pi/2$", r"$\\pi$"),
        ),
        (
            magnitude,
            "viridis",
            LogNorm(float(np.min(positive_magnitude)), float(np.max(positive_magnitude))),
            "average projected magnitude",
            None,
            None,
        ),
        (
            angle_std,
            "magma",
            Normalize(0.0, float(np.nanmax(angle_std))),
            "axial angle std (rad)",
            None,
            None,
        ),
        (
            magnitude_std,
            "plasma",
            Normalize(0.0, float(np.nanmax(magnitude_std))),
            "projected-magnitude std",
            None,
            None,
        ),
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), constrained_layout=False)
    for axis, (values, cmap, norm, label, ticks, ticklabels) in zip(
        axes.flat, panels, strict=True
    ):
        image = _stress_heatmap_image(
            axis, values, extent=extent, cmap=cmap, norm=norm
        )
        axis.plot(
            stress_limit_x,
            stress_limit_y,
            color="#d62728",
            linewidth=1.3,
            zorder=6,
            label="loss of ellipticity",
        )
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(-plot_y_limit, plot_y_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$(\sigma_{22}-\sigma_{11})/2$")
        axis.set_ylabel(r"$\sigma_{12}$")
        colorbar = figure.colorbar(
            image,
            ax=axis,
            orientation="vertical",
            pad=0.02,
            fraction=0.046,
        )
        source_suffix = (
            r" (full-$T$, $\Delta T$-derived)" if vector_source == "delta_T" else ""
        )
        colorbar.set_label(f"{label}{source_suffix}")
        if ticks is not None:
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(ticklabels)
    axes[0, 0].set_title("(a) maximal-flow axis")
    axes[0, 1].set_title("(b) absolute projected flow")
    axes[1, 0].set_title("(c) directional dispersion")
    axes[1, 1].set_title("(d) magnitude dispersion")
    if vector_source == "delta_T":
        legend_title = (
            r"full-$T$ stress coordinates:" "\n"
            r"$T_\mathrm{state2}=\Delta T\,T_\mathrm{state0}$"
        )
    else:
        legend_title = None
    axes[0, 0].legend(
        loc="upper left",
        frameon=True,
        title=legend_title,
        title_fontsize=8 if legend_title else None,
    )
    figure.tight_layout(pad=0.5, w_pad=0.15, h_pad=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    manifest = output_path.with_name(output_path.stem + "_cells.csv")
    _write_stress_manifest(manifest, count, metrics, edges)
    if rejected_total_T:
        rejection_path = output_path.with_name(
            output_path.stem + "_unreconstructable_total_T.csv"
        )
        _write_total_T_rejections(rejection_path, rejected_total_T)
        print(
            f"Excluded {len(rejected_total_T)} total-T endpoint pairs; "
            f"see {rejection_path}."
        )
    return output_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dpi", type=int, default=234)
    parser.add_argument("--bins", type=int, default=SPATIAL_BINS)
    parser.add_argument("--min-count", type=int, default=MIN_CELL_COUNT)
    parser.add_argument(
        "--y-limit",
        type=float,
        default=STRESS_Y_LIMIT,
        help="Stress y-limit; omit to fit the data cloud and ignore the boundary.",
    )
    parser.add_argument(
        "--stress-extent",
        type=float,
        default=STRESS_EXTENT,
        help="Use fixed centred stress-coordinate limits, required for chunks.",
    )
    parser.add_argument(
        "--event-start",
        type=int,
        default=None,
        help="Zero-based start index in the saved-event list for one chunk.",
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=None,
        help="Number of saved events to extract into one chunk.",
    )
    parser.add_argument(
        "--save-vector-chunk",
        type=Path,
        default=None,
        help="Save raw binned transitions instead of rendering a figure.",
    )
    parser.add_argument(
        "--load-vector-chunks",
        type=Path,
        nargs="+",
        default=None,
        help="Combine compatible raw-transition chunks into one exact figure.",
    )
    parser.add_argument(
        "--exclude-unreconstructable-T",
        action="store_true",
        help=(
            "Exclude and audit total-T endpoint pairs that do not reproduce "
            "their exported VTU Cauchy stress."
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore a matching heatmap cache and rebuild it.",
    )
    parser.add_argument(
        "--vector-source",
        choices=("both", "delta_T", "local_stress"),
        default="local_stress",
        help="Generate the direct-VTU sigma plot, the Delta-T plot, or both.",
    )
    args = parser.parse_args()
    if (
        args.dpi <= 0
        or args.bins <= 0
        or args.min_count <= 0
        or (
            args.y_limit is not None
            and (not np.isfinite(args.y_limit) or args.y_limit <= 0)
        )
        or (args.stress_extent is not None and (not np.isfinite(args.stress_extent) or args.stress_extent <= 0))
    ):
        raise ValueError("dpi, bins, y-limit, and stress-extent must be positive.")
    if args.top is not None and args.top <= 0:
        raise ValueError("--top must be positive.")
    if args.event_start is not None and args.event_start < 0:
        raise ValueError("--event-start must be non-negative.")
    if args.event_count is not None and args.event_count <= 0:
        raise ValueError("--event-count must be positive.")
    if args.save_vector_chunk is not None and args.load_vector_chunks is not None:
        raise ValueError("Cannot save and load stress-vector chunks in one invocation.")
    if (args.save_vector_chunk is not None or args.load_vector_chunks is not None) and args.vector_source == "both":
        raise ValueError("Chunk input/output requires exactly one --vector-source.")
    if args.save_vector_chunk is not None:
        if args.stress_extent is None:
            raise ValueError("--save-vector-chunk requires --stress-extent.")
        events = select_events(args.job, number=args.top, saved_only=True)
        start = 0 if args.event_start is None else args.event_start
        stop = len(events) if args.event_count is None else start + args.event_count
        events = events[start:stop]
        if not events:
            raise ValueError("The requested saved-event chunk is empty.")
        _write_stress_vector_chunk(
            args.save_vector_chunk,
            _collect_stress_vectors(
                events,
                bins=args.bins,
                min_count=args.min_count,
                vector_source=args.vector_source,
                exclude_unreconstructable_T=args.exclude_unreconstructable_T,
                stress_extent=args.stress_extent,
            ),
        )
        print(args.save_vector_chunk)
        return
    precomputed_vectors = (
        _load_stress_vector_chunks(args.load_vector_chunks)
        if args.load_vector_chunks is not None
        else None
    )
    events = select_events(args.job, number=args.top, saved_only=True)
    sources = ("delta_T", "local_stress") if args.vector_source == "both" else (args.vector_source,)
    for source in sources:
        filename = (
            "aggregate_stress_maxflow_heatmaps.png"
            if source == "local_stress"
            else "aggregate_stress_maxflow_heatmaps_delta_T.png"
        )
        output, manifest = render_stress_maxflow_heatmaps(
            events,
            args.output_directory / filename,
            dpi=args.dpi,
            bins=args.bins,
            min_count=args.min_count,
            y_limit=args.y_limit,
            vector_source=source,
            exclude_unreconstructable_T=args.exclude_unreconstructable_T,
            stress_extent=args.stress_extent,
            vector_data=precomputed_vectors,
            force_recompute=args.force_recompute,
        )
        print(output)
        print(manifest)


if __name__ == "__main__":
    main()

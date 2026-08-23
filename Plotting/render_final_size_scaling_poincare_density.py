#!/usr/bin/env python3
"""Plot pooled Poincare densities from final size-scaling VTU frames.

Each input is one final saved VTU frame from one simulation.  The plot pools
the element states from those frames directly; it does not construct event
transitions or average events.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Plotting.dataFunctions import VTUData
from Plotting.render_largest_reconnecting_events import (
    DENSITY_GRID_SIZE,
    _accumulate_poincare_density_histogram,
    _disk_coordinates,
    _draw_poincare_density_histogram,
    _draw_poincare_loss_of_ellipticity_limit,
    _loss_of_ellipticity_boundary,
    _metric_from_total_T,
    _prepare_poincare_axis,
    _render_pooled_poincare_flow,
)


def _final_deformation_map(path: Path, *, load_increment: float) -> np.ndarray:
    """Read the appropriate full deformation map from a final VTU.

    Reconnecting VTUs contain the rotation-safe ``F_E`` and legacy ``T_p``
    fields, so use ``VTUData.get_T()``.  The older non-reconnecting VTUs do
    not export those fields; there, their stored ``F`` is the full map.
    """

    data = VTUData(path, load_increment=load_increment)
    fields = set(data.mesh.cell_data)
    total_fields = {"F_E11", "F_E12", "F_E21", "F_E22", "T11", "T12", "T21", "T22"}
    deformation_fields = {"F11", "F12", "F21", "F22"}
    if total_fields <= fields:
        return np.asarray(data.get_T(), dtype=float)
    if deformation_fields <= fields:
        return np.asarray(data.get_F(), dtype=float)
    raise KeyError(
        f"Could not identify a full deformation map in {path}; "
        f"available cell fields are {sorted(fields)}"
    )


def _collect_final_density(
    vtu_paths: list[Path], *, load_increment: float
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    """Collect one fixed-grid density histogram from final VTU frames."""

    if not vtu_paths:
        raise ValueError("At least one final VTU path is required.")
    histogram = np.zeros((DENSITY_GRID_SIZE, DENSITY_GRID_SIZE), dtype=int)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total_element_count = 0
    visible_density_count = 0

    for path in vtu_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Final VTU file does not exist: {path}")

        total_T = _final_deformation_map(path, load_increment=load_increment)
        metric = _metric_from_total_T(total_T, source=path)
        x, y = _disk_coordinates(metric)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.shape != y.shape or x.size == 0:
            raise ValueError(f"Final VTU produced an empty/inconsistent Poincare cloud: {path}")

        visible_density_count += _accumulate_poincare_density_histogram(
            histogram, x, y, zoom=1.0
        )
        total_element_count += int(x.size)
        x_parts.append(x)
        y_parts.append(y)

    return (
        histogram,
        total_element_count,
        visible_density_count,
        np.concatenate(x_parts),
        np.concatenate(y_parts),
    )


def render_final_density(
    vtu_paths: list[Path],
    output_path: Path,
    *,
    dpi: int = 234,
    load_increment: float = 1e-5,
) -> Path:
    """Render one density plot from one final VTU per simulation."""

    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if not np.isfinite(load_increment) or load_increment <= 0.0:
        raise ValueError("load_increment must be finite and positive.")

    (
        histogram,
        total_element_count,
        visible_density_count,
        x,
        y,
    ) = _collect_final_density(vtu_paths, load_increment=load_increment)
    boundary = _loss_of_ellipticity_boundary()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The pooled renderer is shared with the event plot.  Passing the same
    # final-frame cloud as both endpoints disables transition arrows while
    # preserving its established density, grid, boundary, and colourbar style.
    return _render_pooled_poincare_flow(
        output_path,
        zoom=1.0,
        density_histogram=histogram,
        visible_density_count=visible_density_count,
        total_element_count=total_element_count,
        before_x=x,
        before_y=y,
        after_x=x,
        after_y=y,
        delta_T_frobenius=np.zeros_like(x),
        boundary=boundary,
        dpi=dpi,
        direction_split_otsu=False,
        show_arrows=False,
    )


def render_combined_density(
    reconnecting_paths: list[Path],
    nonreconnecting_paths: list[Path],
    output_path: Path,
    *,
    dpi: int = 234,
    load_increment: float = 1e-5,
) -> Path:
    """Render both densities with one shared colour scale and colourbar."""

    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if not np.isfinite(load_increment) or load_increment <= 0.0:
        raise ValueError("load_increment must be finite and positive.")

    group_data = [
        (
            "(a) Reconnecting",
            _collect_final_density(reconnecting_paths, load_increment=load_increment),
        ),
        (
            "(b) Non-reconnecting",
            _collect_final_density(nonreconnecting_paths, load_increment=load_increment),
        ),
    ]
    shared_norm = LogNorm(
        vmin=1,
        vmax=max(2, max(int(np.max(data[0])) for _, data in group_data)),
    )
    boundary = _loss_of_ellipticity_boundary()
    figure = plt.figure(figsize=(18.0, 9.6))
    axes = [
        figure.add_axes((0.04, 0.21, 0.42, 0.70)),
        figure.add_axes((0.54, 0.21, 0.42, 0.70)),
    ]
    colorbar_axis = figure.add_axes((0.31, 0.10, 0.38, 0.022))
    images = []
    for axis, (title, data) in zip(axes, group_data):
        histogram, _, _, _, _ = data
        _prepare_poincare_axis(axis, zoom=1.0)
        images.append(
            _draw_poincare_density_histogram(
                axis, histogram, zoom=1.0, norm=shared_norm
            )
        )
        _draw_poincare_loss_of_ellipticity_limit(axis, boundary, zoom=1.0)
        axis.set_title(title, pad=12.0)
        axis.legend(loc="upper left", frameon=True)

    colorbar = figure.colorbar(images[0], cax=colorbar_axis, orientation="horizontal")
    totals = {data[1] for _, data in group_data}
    all_visible = all(data[1] == data[2] for _, data in group_data)
    if len(totals) == 1 and all_visible:
        colorbar.set_label(f"Bin counts (N={next(iter(totals))})")
    else:
        colorbar.set_label("Bin counts")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vtu", type=Path, nargs="+")
    parser.add_argument("--reconnecting-vtu", type=Path, nargs="+")
    parser.add_argument("--non-reconnecting-vtu", type=Path, nargs="+")
    parser.add_argument("--dpi", type=int, default=234)
    parser.add_argument("--load-increment", type=float, default=1e-5)
    args = parser.parse_args()

    if args.reconnecting_vtu is not None or args.non_reconnecting_vtu is not None:
        if args.reconnecting_vtu is None or args.non_reconnecting_vtu is None:
            parser.error("Both grouped VTU arguments are required for a combined plot.")
        if args.vtu is not None:
            parser.error("Use either --vtu or the two grouped VTU arguments, not both.")
        output = render_combined_density(
            args.reconnecting_vtu,
            args.non_reconnecting_vtu,
            args.output,
            dpi=args.dpi,
            load_increment=args.load_increment,
        )
    else:
        if args.vtu is None:
            parser.error("Provide --vtu or both grouped VTU arguments.")
        output = render_final_density(
            args.vtu,
            args.output,
            dpi=args.dpi,
            load_increment=args.load_increment,
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

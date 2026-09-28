#!/usr/bin/env python3
"""Colour initial meshes by the geometric quadrant of total T in the disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

from MTMath.poincareEnergy import prepPoincareFig
from Plotting.plasticReductionDeterminantQuadrantsIllustration import quadrant_palette
from Plotting.pyplotFunctions import plot_mesh
from Plotting.render_final_size_scaling_poincare_density import _final_deformation_map
from Plotting.render_largest_reconnecting_events import (
    _disk_coordinates,
    _metric_from_total_T,
)


DISK_SIZE = 500
# Geometric quadrants I--IV are displayed as 3, 1, 2, 4.  Values are
# zero-based here so each displayed number retains its palette colour.
GEOMETRIC_TO_DISPLAY = np.array([2, 0, 1, 3], dtype=int)
QUADRANT_NAMES = ("3", "1", "2", "4")
INSET_TITLE_BBOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.78,
    "pad": 1.5,
}


def geometric_disk_quadrants(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return remapped geometric Poincare-disk quadrant labels 0--3."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x) & np.isfinite(y)):
        raise ValueError("Poincare coordinates must be finite and shape-matched.")

    quadrants = np.full(x.shape, -1, dtype=int)
    quadrants[(x >= 0.0) & (y >= 0.0)] = 0
    quadrants[(x < 0.0) & (y >= 0.0)] = 1
    quadrants[(x < 0.0) & (y < 0.0)] = 2
    quadrants[(x >= 0.0) & (y < 0.0)] = 3
    if np.any(quadrants < 0):
        raise ValueError("Some Poincare points could not be assigned a quadrant.")
    return GEOMETRIC_TO_DISPLAY[quadrants]


def total_T_disk_quadrants(
    vtu_path: Path, *, load_increment: float
) -> np.ndarray:
    """Map unreduced total T to the full disk and classify by x/y signs."""

    total_T = _final_deformation_map(vtu_path, load_increment=load_increment)
    metric = _metric_from_total_T(total_T, source=vtu_path)
    x, y = _disk_coordinates(metric)
    return geometric_disk_quadrants(x, y)


def _draw_disk_key(axis: plt.Axes, cmap: ListedColormap) -> None:
    """Draw the unreduced full Poincare disk with the mesh's four colours."""

    coordinate = np.linspace(-1.0, 1.0, DISK_SIZE)
    x, y = np.meshgrid(coordinate, coordinate)
    labels = geometric_disk_quadrants(x, y).astype(float)
    labels[x * x + y * y > 1.0] = np.nan

    prepPoincareFig(
        grid_size=DISK_SIZE,
        ax=axis,
        withGrid=True,
        grid_depth=4,
        grid_color="#30343B",
        withYieldSurface=False,
        minimalTicks=True,
    )
    disk_cmap = cmap.copy()
    disk_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    axis.imshow(
        labels,
        origin="lower",
        extent=(0, DISK_SIZE, 0, DISK_SIZE),
        interpolation="nearest",
        cmap=disk_cmap,
        norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
        alpha=0.82,
        zorder=0,
    )
    center = DISK_SIZE / 2
    axis.axhline(center, color="black", linewidth=0.7, zorder=20)
    axis.axvline(center, color="black", linewidth=0.7, zorder=20)
    positions = (
        (0.73, 0.73),
        (0.27, 0.73),
        (0.27, 0.27),
        (0.73, 0.27),
    )
    for name, (x_fraction, y_fraction) in zip(QUADRANT_NAMES, positions):
        label = axis.text(
            x_fraction * DISK_SIZE,
            y_fraction * DISK_SIZE,
            name,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=30,
        )
        label.set_path_effects(
            [patheffects.withStroke(linewidth=1.7, foreground="black")]
        )
    axis.set_title(
        r"$T^{\mathsf{T}}T$",
        fontsize=8,
        pad=1.5,
        bbox=INSET_TITLE_BBOX,
    )


def _draw_quadrant_fractions(
    axis: plt.Axes,
    quadrants: np.ndarray,
    cmap: ListedColormap,
) -> None:
    """Draw the fraction of mesh elements assigned to each quadrant."""

    counts = np.bincount(np.asarray(quadrants, dtype=int), minlength=4)
    if counts.shape != (4,) or counts.sum() == 0:
        raise ValueError("Expected at least one element in quadrants 1--4.")
    fractions = counts / counts.sum()
    labels = [
        f"{number}\n" + rf"${100.0 * fraction:.0f}\%$"
        for number, fraction in zip(range(1, 5), fractions)
    ]
    axis.pie(
        counts,
        labels=labels,
        colors=cmap(np.arange(4)),
        startangle=90,
        counterclock=True,
        labeldistance=0.62,
        textprops={
            "fontsize": 7,
            "color": "black",
            "ha": "center",
            "va": "center",
        },
        wedgeprops={"edgecolor": "black", "linewidth": 0.65, "alpha": 0.86},
    )
    # Matplotlib's pie helper leaves a 25% margin by default.  Removing it
    # makes this circle fill the same-sized inset as the Poincare disk.
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    axis.set_title(
        "Element fractions",
        fontsize=8,
        pad=1.5,
        bbox=INSET_TITLE_BBOX,
    )


def render_quadrant_mesh(
    vtu_path: Path,
    output_path: Path,
    *,
    title: str,
    load_increment: float = 1e-5,
    dpi: int = 300,
) -> Path:
    """Render one mesh and its full-disk geometric-quadrant key."""

    vtu_path = Path(vtu_path)
    if not vtu_path.is_file():
        raise FileNotFoundError(vtu_path)
    quadrants = total_T_disk_quadrants(
        vtu_path, load_increment=load_increment
    )
    palette = quadrant_palette(1)
    cmap = ListedColormap(palette, name="total_T_disk_quadrants")
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)

    figure, axis = plt.subplots(figsize=(8.0, 7.4))
    plot_mesh(
        vtu_path,
        ax=axis,
        mesh_property="total_T_disk_quadrant",
        field_override=quadrants,
        field_cmap=cmap,
        field_norm=norm,
        field_background_color=(0.12, 0.12, 0.12, 0.58),
        add_colorbar=False,
        add_rombus=True,
        unwrap_periodic_triangles=True,
        cartesian_viewport_culling=True,
        load_increment=load_increment,
    )
    axis.set_title(title, fontsize=13, pad=9)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    inset = axis.inset_axes((0.705, 0.035, 0.275, 0.275), zorder=100)
    inset.set_facecolor((1.0, 1.0, 1.0, 0.70))
    _draw_disk_key(inset, cmap)

    pie_axis = axis.inset_axes((0.02, 0.035, 0.275, 0.275), zorder=100)
    pie_axis.set_facecolor((1.0, 1.0, 1.0, 0.70))
    _draw_quadrant_fractions(pie_axis, quadrants, cmap)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--load-increment", type=float, default=1e-5)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    print(
        render_quadrant_mesh(
            args.vtu,
            args.output,
            title=args.title,
            load_increment=args.load_increment,
            dpi=args.dpi,
        )
    )


if __name__ == "__main__":
    main()

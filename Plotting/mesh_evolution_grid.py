"""Render eight native VTUs in portrait and landscape grids, in row order.

Reuse plot_mesh for energy colours, periodic unwrapping, and viewport tiling.
The viewport stays fixed in physical coordinates while the cell shears.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np

from Plotting.dataFunctions import VTUData, infer_strain_from_vtu, resolve_vtu_files
from Plotting.pyplotFunctions import plot_mesh


DEFAULT_RUN = Path(
    "/Volumes/data/MTS2D_output/"
    "simpleShear,s200x200l0.15,1e-05,5.1PBCedgeFlipt5epsR1e-05"
    "LBFGSEpsg1e-08LBFGSEpsx1e-06s0"
)
DEFAULT_STRAINS = (0.15, 0.5, 1, 1.5, 2.5, 3, 4, 4.5)


def render_grid(source, output, *, strains=DEFAULT_STRAINS, margin=0.05, dpi=200,
                horizontal=False):
    strains = np.asarray(strains, dtype=float)
    if strains.shape != (8,) or not np.all(np.isfinite(strains)):
        raise ValueError("Specify eight finite strains, in row reading order.")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("The viewport margin must be finite and nonnegative.")
    files = resolve_vtu_files(source)
    loads = np.asarray([infer_strain_from_vtu(p) for p in files], dtype=float)
    if not np.all(np.isfinite(loads)):
        raise ValueError("All VTUs must have finite strain metadata.")
    indices = [int(np.argmin(np.abs(loads - target))) for target in strains]
    if len(set(indices)) != 8 or np.any(np.abs(loads[indices] - strains) > 0.02):
        raise ValueError("Need eight distinct VTUs within 0.02 of the requested strains.")
    first = VTUData(files[indices[0]], load_increment=1e-5)
    size = float(first.size[0])
    if first.BC != "PBC" or first.size[1] != size:
        raise ValueError("Expected a square periodic simulation cell.")
    viewport = (-margin * size, (1 + margin) * size) * 2
    del first
    if horizontal:
        columns = 4
        fig = plt.figure(figsize=(16.7 * 0.7, 8 * 0.7))
        grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, 0.055),
                               hspace=0.18, wspace=0.08,
                               left=0.015, right=0.94, bottom=0.025, top=0.94)
        colorbar_slot, orientation = grid[:, 4], "vertical"
    else:
        columns = 2
        fig = plt.figure(figsize=(8 * 0.7, 16.7 * 0.7))
        grid = fig.add_gridspec(5, 2, height_ratios=(1, 1, 1, 1, 0.055),
                               hspace=0.12, wspace=0.035,
                               left=0.025, right=0.975, bottom=0.045, top=0.98)
        colorbar_slot, orientation = grid[4, :], "horizontal"
    records = []
    for panel, index in enumerate(indices):
        row, column = divmod(panel, columns)
        ax = fig.add_subplot(grid[row, column])
        _, cmap, norm = plot_mesh(
            files[index], ax=ax, e_lims=(0, 0.12), add_colorbar=False,
            load_increment=1e-5, cartesian_viewport=viewport,
            cartesian_viewport_culling=True, unwrap_periodic_triangles=True,
        )
        for collection in ax.collections:
            # Edge strokes can obscure narrow energetic triangles in small panels.
            collection.set_edgecolor("none")
            collection.set_rasterized(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"({chr(ord('a') + panel)}) $\gamma={loads[index]:.5f}$",
                     fontsize=12, pad=4)
        records.append(dict(row=row + 1, column=column + 1,
                            requested_strain=float(strains[panel]),
                            strain=float(loads[index]), source=str(Path(files[index]).resolve())))
        print(f"Rendered panel {panel + 1}/8: gamma={loads[index]:.5f}", flush=True)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                       cax=fig.add_subplot(colorbar_slot), orientation=orientation,
                       ticks=np.linspace(0, 0.12, 7))
    bar.set_label(r"$E_i$")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(output.with_suffix(suffix), dpi=dpi)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(dict(
        viewport=viewport, energy_limits=[0, 0.12], panels=records,
        figure_inches=list(fig.get_size_inches()), colorbar_orientation=orientation,
    ), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path,
                        default=Path("Plots/mesh_evolution_grid/mesh_evolution_grid"))
    parser.add_argument("--strains", type=float, nargs=8, default=DEFAULT_STRAINS)
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Margin on each side as a fraction of the original cell size.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--layout", choices=("portrait", "landscape", "both"), default="both")
    args = parser.parse_args()
    for layout in ("portrait", "landscape"):
        if args.layout not in (layout, "both"):
            continue
        horizontal = layout == "landscape"
        output = (args.output.with_name(args.output.stem + "_horizontal")
                  if horizontal else args.output)
        render_grid(args.source, output, strains=args.strains,
                    margin=args.margin, dpi=args.dpi, horizontal=horizontal)


if __name__ == "__main__":
    main()

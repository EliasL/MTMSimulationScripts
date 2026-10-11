"""Draw the sequential-shear energy figure from its original Grace datasets.

The legacy Grace input retains orientation II (G0.S1) and reconnection
(G0.S2). Explicit run paths use that pair's actual energies and endpoint
mesh, with matching physical and loading parameters checked before plotting.
The horizontal-first protocol ends at (u_x, u_y) = (1, 3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

from Plotting.dataFunctions import VTUData, infer_strain_from_vtu
from Plotting.pyplotFunctions import plot_mesh

from Plotting.meshLockingFigure import (
    DEFAULT_LOADS,
    ROOT,
    _configure_matplotlib,
    _nice_energy_max,
    classify_protocol_nodes,
    select_vtu_at_load,
    validate_horizontal_first_runs,
    loading_displacements,
)


def read_grace_datasets(path: Path) -> dict[str, np.ndarray]:
    """Read the embedded XY datasets without altering their sampling."""
    datasets = {}
    target = None
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("@target "):
            target = line.split()[1]
            rows = []
        elif line.strip() == "&" and target is not None:
            datasets[target] = np.asarray(rows, dtype=float)
            target = None
        elif target is not None and line.strip() and not line.startswith(("@", "#")):
            rows.append([float(value) for value in line.split()])
    return datasets


def make_energy_sequential_shear_figure(
    source: Path,
    *,
    no_reconnection_run: Path | None = None,
    edge_flipping_run: Path | None = None,
    inset_vtu: Path | None = None,
    reconnected_inset_vtu: Path | None = None,
    alternating: bool | None = None,
):
    """Show the full energy protocol and optionally the actual endpoint mesh."""
    if alternating is None:
        alternating = no_reconnection_run is None
    if (no_reconnection_run is None) != (edge_flipping_run is None):
        raise ValueError("Supply both explicit run paths, or neither.")
    if no_reconnection_run is None:
        data = read_grace_datasets(source)
    else:
        validate_horizontal_first_runs(no_reconnection_run, edge_flipping_run,
                                       alternating=alternating)
        data = {
            key: pd.read_csv(Path(run) / "macroData.csv")
            [["load", "total_energy"]].to_numpy(dtype=float)
            for key, run in (("G0.S1", no_reconnection_run),
                             ("G0.S2", edge_flipping_run))
        }
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.subplots_adjust(left=0.12, right=0.965, bottom=0.19, top=0.91)
    curves = (
        ("G0.S1", "Fixed connectivity", "#222222", (0, (5, 3))),
        ("G0.S2", "Reconnection", "#0072B2", "-"),
    )
    for key, label, color, linestyle in curves:
        values = data[key]
        if values[0, 0] > 0 or values[-1, 0] < 4:
            raise ValueError(f"{key} does not cover the full protocol from load 0 to 4.")
        mask = (values[:, 0] >= 0) & (values[:, 0] <= 4 + 1e-9)
        ax.plot(values[mask, 0], values[mask, 1], label=label,
                color=color, linestyle=linestyle, linewidth=2)

    for letter, load in zip("ABCD", DEFAULT_LOADS):
        ax.vlines(load, 0, 3, color="#6B7077", linestyle=(0, (2, 3)),
                  linewidth=1, alpha=0.65, zorder=0)
        ax.text(load, 3, letter, ha="center", va="center", fontsize=17,
                fontweight="bold", zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        for key, _, color, _ in curves:
            values = data[key]
            index = int(np.argmin(np.abs(values[:, 0] - load)))
            if not np.isclose(values[index, 0], load, atol=1e-7, rtol=0):
                raise ValueError(f"{key} has no observation at snapshot {letter}.")
            ax.plot(load, values[index, 1], "o", markersize=5.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.2, clip_on=False, zorder=5)

    ax.set_xlim(0, 4)
    maximum = max(
        float(np.max(data[key][(data[key][:, 0] >= 0) & (data[key][:, 0] <= 4), 1]))
        for key, _, _, _ in curves
    )
    if inset_vtu is None and no_reconnection_run is not None:
        inset_vtu = select_vtu_at_load(Path(no_reconnection_run), 4.0)
    energy_max = max(4.0 if inset_vtu is not None else 3.5,
                     _nice_energy_max(maximum))
    ax.set_ylim(0, energy_max)
    ax.set_xticks(range(5))
    pairs = [loading_displacements(t, alternating=alternating) for t in range(5)]
    ax.set_xticklabels([rf"$u_x={ux:g}$" + "\n" + rf"$u_y={uy:g}$"
                        for ux, uy in pairs], fontsize=15)
    ax.set_yticks(np.arange(0, energy_max + 1e-9, 1))
    ax.set_ylabel("Total energy", fontsize=19)
    ax.tick_params(direction="in", top=False, right=False, length=6,
                   axis="y", labelsize=15)
    ax.tick_params(axis="x", direction="in", top=False, length=6, pad=9)
    ax.legend(loc="upper left", fontsize=16)
    if inset_vtu is not None:
        snapshot = VTUData(inset_vtu)
        reference = snapshot.get_reference_nodes()[:, :2]
        groups = classify_protocol_nodes(reference, snapshot.get_fixed_status())
        displacement = snapshot.get_nodes()[:, :2] - reference
        ux = float(np.median(displacement[groups["x_loaded"], 0]))
        uy = float(np.median(displacement[groups["y_loaded"], 1]))
        inset_bounds = ([0.52, 0.43, 0.28, 0.34] if energy_max <= 4
                        else [0.45, 0.61, 0.28, 0.34])
        # Double the displayed area, preserving aspect ratio and the top centre.
        x, y, width, height = inset_bounds
        scale = np.sqrt(2.0)
        inset_bounds = [x - (scale - 1) * width / 2 - 0.04,
                        y - (scale - 1) * height - 0.04,
                        width * scale, height * scale]
        inset = ax.inset_axes(inset_bounds, zorder=7)
        plot_mesh(str(inset_vtu), e_lims=(0, 0.12), mesh_property="energy",
                  ax=inset, add_colorbar=False, add_rombus=False)
        inset.set_axis_on()
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(1.1)
        inset.set_title("Fixed connectivity", fontsize=13, pad=5)
        fixed = data["G0.S1"]
        load = float(infer_strain_from_vtu(inset_vtu))
        index = int(np.argmin(np.abs(fixed[:, 0] - load)))
        if not np.isclose(fixed[index, 0], load, atol=1e-7, rtol=0):
            raise ValueError("Inset snapshot has no matching fixed-connectivity energy.")
        anchor = (float(fixed[index, 0]), float(fixed[index, 1]))
        ax.plot(*anchor, "o", color="#222222", markersize=5, clip_on=False, zorder=8)
        for corner in ((1, 0), (1, 1)):
            connector = ConnectionPatch(
                xyA=corner, coordsA=inset.transAxes, xyB=anchor, coordsB=ax.transData,
                axesA=inset, axesB=ax, color="#555555", linewidth=0.9,
                clip_on=False, zorder=6,
            )
            ax.add_artist(connector)
    if reconnected_inset_vtu is None and edge_flipping_run is not None:
        inset_load = float(infer_strain_from_vtu(inset_vtu)) if inset_vtu is not None else 4.0
        reconnected_inset_vtu = select_vtu_at_load(Path(edge_flipping_run), inset_load)
    if reconnected_inset_vtu is not None:
        load = float(infer_strain_from_vtu(reconnected_inset_vtu))
        if inset_vtu is not None and not np.isclose(load, float(infer_strain_from_vtu(inset_vtu)), atol=1e-7, rtol=0):
            raise ValueError("The two mesh insets must have the same load.")
        snapshot = VTUData(reconnected_inset_vtu)
        nodes = snapshot.get_nodes()[:, :2]
        reconnect_inset = ax.inset_axes([0.71, 1.0 / energy_max, 0.27, 0.44], zorder=7)
        plot_mesh(str(reconnected_inset_vtu), e_lims=(0, 0.12), mesh_property="energy",
                  ax=reconnect_inset, add_colorbar=False, add_rombus=False)
        # No lower padding: the mesh itself starts at energy-axis height y=1.
        reconnect_inset.set_xlim(float(nodes[:, 0].min()), float(nodes[:, 0].max()))
        reconnect_inset.set_ylim(float(nodes[:, 1].min()), float(nodes[:, 1].max()))
        reconnect_inset.set_aspect("equal", adjustable="box", anchor="SE")
        reconnect_inset.set_axis_on()
        reconnect_inset.set_xticks([])
        reconnect_inset.set_yticks([])
        reconnect_inset.set_facecolor("white")
        for spine in reconnect_inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#0072B2")
            spine.set_linewidth(1.1)
        reconnect_inset.set_title("Reconnection", loc="right", fontsize=13,
                                  color="black", pad=5)
        values = data["G0.S2"]
        index = int(np.argmin(np.abs(values[:, 0] - load)))
        if not np.isclose(values[index, 0], load, atol=1e-7, rtol=0):
            raise ValueError("Reconnected inset has no matching energy observation.")
        anchor = (float(values[index, 0]), float(values[index, 1]))
        ax.plot(*anchor, "o", color="#0072B2", markersize=5, clip_on=False, zorder=8)
        for corner in ((1, 0), (1, 1)):
            ax.add_artist(ConnectionPatch(
                xyA=corner, coordsA=reconnect_inset.transAxes,
                xyB=anchor, coordsB=ax.transData, axesA=reconnect_inset, axesB=ax,
                color="#0072B2", linewidth=0.9, clip_on=False, zorder=6,
            ))
    return fig, ax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "Plots" / "energy_sequential_shear.agr")
    parser.add_argument("--output-stem", type=Path,
                        default=ROOT / "Plots" / "energy_sequential_shear")
    parser.add_argument("--no-reconnection-run", type=Path,
                        help="Use this run's energy instead of the Grace dataset.")
    parser.add_argument("--edge-flipping-run", type=Path,
                        help="Use this run's energy instead of the Grace dataset.")
    parser.add_argument("--inset-vtu", type=Path,
                        help="Fixed-connectivity mesh snapshot for a boxed inset.")
    parser.add_argument("--reconnected-inset-vtu", type=Path,
                        help="Reconnected mesh at the same load as the fixed inset.")
    parser.add_argument("--alternating", action="store_true", default=None,
                        help="Use the original alternating path ending at (2,2).")
    args = parser.parse_args()
    fig, _ = make_energy_sequential_shear_figure(
        args.source, no_reconnection_run=args.no_reconnection_run,
        edge_flipping_run=args.edge_flipping_run,
        inset_vtu=args.inset_vtu,
        reconnected_inset_vtu=args.reconnected_inset_vtu,
        alternating=args.alternating,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

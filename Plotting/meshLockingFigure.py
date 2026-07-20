"""Paper-style mesh-locking figure for the double-dislocation test.

The left panel summarizes the piecewise boundary loading.  The right panel
compares the same minor-diagonal 20x20 mesh without reconnection and with edge
flipping at four values of the protocol load ``gamma*``.

Run this file directly to write PNG and PDF copies to ``Plots/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from Plotting.dataFunctions import VTUData, infer_strain_from_vtu, resolve_vtu_files
from Plotting.mesh_plotting import (
    MeshFigure,
    MeshStyle,
    draw_mesh_nodes,
)
from Plotting.pyplotFunctions import plot_mesh


DEFAULT_DATA_ROOT = ROOT / "GeneratedData" / "MTS2D_output"
DEFAULT_LOADS = (0.4, 1.0, 1.4, 2.0)
DEFAULT_SIZE = 10
DEFAULT_SWITCH_LOAD = 1.0

FIXED_COLOR = "#2F343B"
X_LOAD_COLOR = "#0072B2"
Y_LOAD_COLOR = "#D55E00"
FREE_COLOR = "#009E73"
MESH_COLOR = "#7F858D"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_double_dislocation_run(
    data_root: Path,
    *,
    edge_flip: bool,
    size: int = 20,
    switch_load: float = 0.8,
) -> Path:
    """Find the requested minor-mesh double-dislocation output folder."""

    data_root = Path(data_root)
    prefix = f"doubleDislocationTest,s{size}x{size}"
    switch_token = f"GP3{switch_load:g}"
    matches = [
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and "meshDiagonalminor" in path.name
        and switch_token in path.name
        and ("edgeFlip" in path.name) == edge_flip
        and (path / "collection.pvd").exists()
    ]
    if len(matches) != 1:
        method = "edge flipping" if edge_flip else "no reconnection"
        raise ValueError(
            f"Expected one {size}x{size} minor-mesh run with {method} and "
            f"GP3={switch_load:g}; found {len(matches)} in {data_root}."
        )
    return matches[0]


def select_vtu_at_load(run_path: Path, target_load: float) -> Path:
    """Return the VTU snapshot whose exported load matches ``target_load``."""

    candidates = []
    for vtu_file in resolve_vtu_files(run_path):
        load = infer_strain_from_vtu(vtu_file)
        if load is not None and np.isfinite(load):
            candidates.append((abs(float(load) - target_load), float(load), Path(vtu_file)))
    if not candidates:
        raise FileNotFoundError(f"No load-labelled VTU files found in {run_path}.")
    error, load, vtu_file = min(candidates, key=lambda item: item[0])
    if error > 5e-7:
        raise ValueError(
            f"Closest snapshot to gamma*={target_load:g} is gamma*={load:g} "
            f"in {run_path}."
        )
    return vtu_file


def classify_protocol_nodes(
    reference_nodes: np.ndarray,
    fixed_status: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split constrained boundary nodes by their role in the loading protocol."""

    reference_nodes = np.asarray(reference_nodes, dtype=float)[:, :2]
    fixed_status = np.asarray(fixed_status).ravel().astype(bool)
    if len(reference_nodes) != len(fixed_status):
        raise ValueError("reference_nodes and fixed_status must have matching lengths.")

    x = reference_nodes[:, 0]
    y = reference_nodes[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    atol = 1e-7 * max(1.0, xmax - xmin, ymax - ymin)

    on_left = np.isclose(x, xmin, atol=atol)
    on_bottom = np.isclose(y, ymin, atol=atol)
    x_loaded = fixed_status & on_left & (y > ymid)
    y_loaded = fixed_status & on_bottom & (x > xmid)
    fixed = fixed_status & ~x_loaded & ~y_loaded
    top = np.isclose(y, ymax, atol=atol)
    right = np.isclose(x, xmax, atol=atol)

    return {
        "fixed": np.flatnonzero(fixed),
        "x_loaded": np.flatnonzero(x_loaded),
        "y_loaded": np.flatnonzero(y_loaded),
        "top": np.flatnonzero(top),
        "right": np.flatnonzero(right),
    }


def _chain_edges(nodes: np.ndarray, node_ids: np.ndarray, coordinate: int) -> np.ndarray:
    """Connect a boundary-node subset in coordinate order."""

    node_ids = np.asarray(node_ids, dtype=int)
    order = np.argsort(nodes[node_ids, coordinate])
    ordered = node_ids[order]
    if len(ordered) < 2:
        return np.empty((0, 2), dtype=int)
    return np.column_stack([ordered[:-1], ordered[1:]])


def _draw_loading_protocol(
    ax: plt.Axes,
    reference_nodes: np.ndarray,
    connectivity: np.ndarray,
    fixed_status: np.ndarray,
    *,
    switch_load: float,
    final_load: float,
) -> None:
    reference_nodes = np.asarray(reference_nodes, dtype=float)[:, :2]
    groups = classify_protocol_nodes(reference_nodes, fixed_status)
    xmin, ymin = np.min(reference_nodes, axis=0)
    xmax, ymax = np.max(reference_nodes, axis=0)
    span = max(float(xmax - xmin), float(ymax - ymin))

    mesh = MeshFigure(ax)
    mesh.draw_mesh(
        reference_nodes,
        connectivity,
        style=MeshStyle(
            color=MESH_COLOR,
            face_alpha=0.0,
            edge_alpha=0.52,
            linewidth=0.36,
            draw_faces=False,
            draw_nodes=False,
            zorder=0,
        ),
    )

    fixed_edges = np.vstack(
        [
            _chain_edges(
                reference_nodes,
                groups["fixed"][
                    np.isclose(reference_nodes[groups["fixed"], 1], ymin)
                ],
                0,
            ),
            _chain_edges(
                reference_nodes,
                groups["fixed"][
                    np.isclose(reference_nodes[groups["fixed"], 0], xmin)
                ],
                1,
            ),
        ]
    )
    mesh.draw_edges(
        reference_nodes,
        fixed_edges,
        color=FIXED_COLOR,
        linewidth=2.2,
        zorder=4,
    )
    mesh.draw_edges(
        reference_nodes,
        _chain_edges(reference_nodes, groups["top"], 0),
        color=FREE_COLOR,
        linewidth=2.0,
        linestyle=(0, (4.0, 2.5)),
        zorder=3,
    )
    mesh.draw_edges(
        reference_nodes,
        _chain_edges(reference_nodes, groups["right"], 1),
        color=FREE_COLOR,
        linewidth=2.0,
        linestyle=(0, (4.0, 2.5)),
        zorder=3,
    )

    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["fixed"],
        facecolor=FIXED_COLOR,
        edgecolor="white",
        size=17,
        linewidth=0.45,
        zorder=6,
    )
    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["x_loaded"],
        facecolor=X_LOAD_COLOR,
        edgecolor="white",
        size=21,
        linewidth=0.55,
        zorder=7,
    )
    draw_mesh_nodes(
        ax,
        reference_nodes,
        node_ids=groups["y_loaded"],
        facecolor=Y_LOAD_COLOR,
        edgecolor="white",
        size=21,
        linewidth=0.55,
        zorder=7,
    )

    for node_id in groups["x_loaded"]:
        y = reference_nodes[node_id, 1]
        ax.annotate(
            "",
            xy=(xmin - 0.015 * span, y),
            xytext=(xmin - 0.115 * span, y),
            arrowprops={"arrowstyle": "-|>", "color": X_LOAD_COLOR, "lw": 1.5},
            zorder=8,
        )
    for node_id in groups["y_loaded"]:
        x = reference_nodes[node_id, 0]
        ax.annotate(
            "",
            xy=(x, ymin - 0.015 * span),
            xytext=(x, ymin - 0.115 * span),
            arrowprops={"arrowstyle": "-|>", "color": Y_LOAD_COLOR, "lw": 1.5},
            zorder=8,
        )

    ax.text(
        xmin + 0.25 * (xmax - xmin),
        ymin + 0.88 * (ymax - ymin),
        "1. horizontal  " + rf"$0\leq\gamma^*\leq {switch_load:g}$",
        color=X_LOAD_COLOR,
        ha="center",
        va="center",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.75 * (xmax - xmin),
        ymin + 0.12 * (ymax - ymin),
        "2. vertical  " + rf"${switch_load:g}<\gamma^*\leq {final_load:g}$",
        color=Y_LOAD_COLOR,
        ha="center",
        va="center",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.20 * (xmax - xmin),
        ymin + 0.12 * (ymax - ymin),
        "fixed nodes",
        color=FIXED_COLOR,
        ha="center",
        va="center",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(
        xmin + 0.76 * (xmax - xmin),
        ymin + 0.88 * (ymax - ymin),
        "free edges",
        color=FREE_COLOR,
        ha="center",
        va="center",
        fontsize=13,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )

    ax.text(-0.04, 1.045, r"\textbf{(a)}", transform=ax.transAxes, fontsize=12)
    mesh.configure_axis(
        xlim=(xmin - 0.16 * span, xmax + 0.035 * span),
        ylim=(ymin - 0.16 * span, ymax + 0.035 * span),
        equal_aspect=True,
        hide_axes=True,
    )


def _nice_energy_max(maximum: float) -> float:
    if not np.isfinite(maximum) or maximum <= 0:
        return 1.0
    scale = 10.0 ** np.floor(np.log10(maximum))
    return float(np.ceil(2.0 * maximum / scale) * 0.5 * scale)


def make_mesh_locking_figure(
    data_root: Path = DEFAULT_DATA_ROOT,
    *,
    loads: tuple[float, ...] = DEFAULT_LOADS,
    size: int = DEFAULT_SIZE,
    switch_load: float = DEFAULT_SWITCH_LOAD,
    energy_limits: tuple[float, float] | None = None,
) -> tuple[plt.Figure, dict[str, object]]:
    """Build the complete loading-protocol and mesh-evolution figure."""

    if len(loads) != 4:
        raise ValueError(f"Expected four load snapshots, got {len(loads)}.")
    _configure_matplotlib()

    run_paths = {
        "no_reconnection": find_double_dislocation_run(
            data_root, edge_flip=False, size=size, switch_load=switch_load
        ),
        "edge_flipping": find_double_dislocation_run(
            data_root, edge_flip=True, size=size, switch_load=switch_load
        ),
    }
    snapshots = {
        method: [select_vtu_at_load(path, load) for load in loads]
        for method, path in run_paths.items()
    }

    all_energies = np.concatenate(
        [
            VTUData(vtu_file).get_energy_field().ravel()
            for method_snapshots in snapshots.values()
            for vtu_file in method_snapshots
        ]
    )
    if energy_limits is None:
        energy_limits = (0.0, _nice_energy_max(float(np.nanmax(all_energies))))

    all_nodes = np.concatenate(
        [
            VTUData(vtu_file).get_nodes()[:, :2]
            for method_snapshots in snapshots.values()
            for vtu_file in method_snapshots
        ],
        axis=0,
    )
    panel_min = float(np.min(all_nodes))
    panel_max = float(np.max(all_nodes))
    panel_padding = 0.035 * (panel_max - panel_min)
    panel_limits = (panel_min - panel_padding, panel_max + panel_padding)

    initial_vtu = select_vtu_at_load(run_paths["no_reconnection"], 0.0)
    initial_data = VTUData(initial_vtu)
    final_load = max(loads)

    figure = plt.figure(figsize=(15.8, 6.15))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 2.0),
        left=0.025,
        right=0.992,
        bottom=0.105,
        top=0.94,
        wspace=0.055,
    )
    protocol_ax = figure.add_subplot(outer[0, 0])
    evolution_grid = outer[0, 1].subgridspec(
        2,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.065),
        hspace=0.075,
        wspace=0.05,
    )
    evolution_axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for column in range(4):
            evolution_axes[row, column] = figure.add_subplot(
                evolution_grid[row, column]
            )
    colorbar_ax = figure.add_subplot(evolution_grid[:, 4])

    _draw_loading_protocol(
        protocol_ax,
        initial_data.get_reference_nodes(),
        initial_data.get_connectivity(),
        initial_data.get_fixed_status(),
        switch_load=switch_load,
        final_load=final_load,
    )

    method_rows = (
        ("no_reconnection", "No reconnection"),
        ("edge_flipping", "With reconnecting"),
    )
    cmap = None
    norm = None
    for row, (method, row_label) in enumerate(method_rows):
        for column, (load, vtu_file) in enumerate(zip(loads, snapshots[method])):
            ax = evolution_axes[row, column]
            _, cmap, norm = plot_mesh(
                str(vtu_file),
                e_lims=energy_limits,
                mesh_property="energy",
                ax=ax,
                add_colorbar=False,
                add_rombus=False,
            )
            ax.set_xlim(*panel_limits)
            ax.set_ylim(*panel_limits)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(rf"$\gamma^*={load:g}$", pad=5)
            if column == 0:
                ax.set_ylabel(row_label, labelpad=5, fontsize=11.5)

    evolution_axes[0, 0].text(
        -0.13,
        1.08,
        r"\textbf{(b)}",
        transform=evolution_axes[0, 0].transAxes,
        fontsize=12,
    )
    if cmap is None or norm is None:
        raise RuntimeError("No mesh panels were drawn.")
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(*energy_limits), cmap=cmap),
        cax=colorbar_ax,
        orientation="vertical",
    )
    colorbar.ax.set_title(r"$E_i$", pad=6)
    colorbar.outline.set_linewidth(0.6)

    metadata = {
        "run_paths": run_paths,
        "snapshots": snapshots,
        "energy_limits": energy_limits,
        "panel_limits": panel_limits,
        "protocol_axis": protocol_ax,
        "evolution_axes": evolution_axes,
    }
    return figure, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the double-dislocation mesh-locking comparison."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing MTS2D output folders.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "Plots" / "mesh_locking_double_dislocation",
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Number of nodes along each side of the square mesh.",
    )
    parser.add_argument(
        "--switch-load",
        type=float,
        default=DEFAULT_SWITCH_LOAD,
        help="Protocol load at which horizontal loading switches to vertical.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    figure, _ = make_mesh_locking_figure(
        args.data_root,
        size=args.size,
        switch_load=args.switch_load,
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

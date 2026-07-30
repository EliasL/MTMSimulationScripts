import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import os
import cv2
from multiprocessing import Pool
from pathlib import Path

import threading
from MTMath.poincareEnergy import (
    plotEnergyField,
    generate_energy_grid,
    drawCScatter,
    C2Plane,
)
from MTMath.reduction import plastic_reduction
from Management.jobs import propperJob

from MTMath.energyFunction import ContiEnergy
from MTMath.meshUtils import (
    triangle_shape_grads_and_area as _triangle_shape_grads_and_area,
    shape_grads_and_area_from_F,
    _element_subset_indices,
    cell_energy_to_node_energy,
    _center_node_index,
    _assemble_nodal_forces,
    perfect_grid_nodes,
    grid_index,
)
from .makePlots import makePlot, energy_drop_label
from .remotePlotting import get_csv_files
from .dataFunctions import (
    get_data_from_name,
    VTUData,
    get_previous_data,
    resolve_vtu_files,
    infer_strain_from_vtu,
    match_vtu_to_macro_row,
)
from .plotPowerLaw import plot_plastic_counts, get_energy_drops
from Management.updateCSV import update_df_header, read_macrodata_csv
# matplotlib.use("Agg")  # Use a non-interactive backend

# We get almost all variables dynamically, but we choose to set the force scale
minForce = 0
maxForce = 0.3


def get_energy_range(vtu_files, cvs_file):
    df = pd.read_csv(cvs_file, usecols=["max_energy"])
    max_energy = df["max_energy"].max()
    # Sometimes, the energy is too high because of a crash
    if max_energy > 100:
        # Then we take the second last
        max_energy = df["max_energy"][:-1].max()
    # We assume that the minimum energy throughout the whole run is the minimum
    # of the initial state
    energy_field = VTUData(vtu_files[0]).get_energy_field()
    min_energy = energy_field.min()

    return [min_energy, max_energy]


def get_matrix_range(vtu_files, matrix_name):
    max_abs = 0.0
    found_finite_value = False

    for vtu_file in vtu_files:
        components = VTUData(vtu_file).get_matrix_components(matrix_name)
        for field in components.values():
            values = np.asarray(field, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            found_finite_value = True
            max_abs = max(max_abs, float(np.max(np.abs(finite))))

    if not found_finite_value:
        raise ValueError(f"No finite values found for matrix '{matrix_name}'.")
    if max_abs == 0.0:
        max_abs = 1.0

    return (-max_abs, max_abs)


def canonical_plastic_shear_counts(matrix, loops=1000):
    """Return signed horizontal/vertical counts in a canonical U/V decomposition."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("matrix must have shape (..., 2, 2)")
    C = matrix.swapaxes(-1, -2) @ matrix
    a, b, c = C[..., 0, 0].copy(), C[..., 0, 1].copy(), C[..., 1, 1].copy()
    horizontal = np.zeros(a.shape, dtype=int)
    vertical = np.zeros(a.shape, dtype=int)

    for _ in range(loops):
        denominator = np.minimum(a, c)
        active = np.isfinite(a + b + c) & (
            (denominator <= 0) | (np.abs(b) > 0.5 * denominator)
        )
        if not np.any(active):
            return horizontal, vertical

        use_horizontal = active & (a < c)  # Match the current MTS2D tie-break.
        use_vertical = active & ~use_horizontal
        denom = np.where(use_horizontal, a, c)
        x = np.divide(-b, denom, out=np.zeros_like(b), where=active & (denom != 0))
        n = np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(int)
        if np.any(active & (n == 0)):
            raise RuntimeError("Canonical plastic-shear decomposition made no progress")

        old_b = b.copy()
        if np.any(use_horizontal):
            horizontal[use_horizontal] -= n[use_horizontal]
            b[use_horizontal] = old_b[use_horizontal] + n[use_horizontal] * a[use_horizontal]
            c[use_horizontal] += (
                2 * n[use_horizontal] * old_b[use_horizontal]
                + n[use_horizontal] ** 2 * a[use_horizontal]
            )
        if np.any(use_vertical):
            vertical[use_vertical] -= n[use_vertical]
            b[use_vertical] = old_b[use_vertical] + n[use_vertical] * c[use_vertical]
            a[use_vertical] += (
                2 * n[use_vertical] * old_b[use_vertical]
                + n[use_vertical] ** 2 * c[use_vertical]
            )

    raise RuntimeError(f"Canonical plastic-shear decomposition exceeded {loops} loops")


def get_plastic_shear_counts(vtu_file, reconnecting=None):
    """Read the appropriate plastic branch and return its signed U/V counts."""
    data = VTUData(vtu_file)
    if reconnecting is not False:
        try:
            branch = np.stack(
                [
                    np.stack([data.get_cell_data("T11"), data.get_cell_data("T12")], axis=-1),
                    np.stack([data.get_cell_data("T21"), data.get_cell_data("T22")], axis=-1),
                ],
                axis=-2,
            )
            return (*canonical_plastic_shear_counts(branch), data, "T")
        except KeyError:
            if reconnecting:
                raise KeyError(
                    "Reconnecting plastic-shear plots require T11, T12, T21 and T22"
                )

    try:
        branch = np.stack(
            [
                np.stack(
                    [data.get_cell_data("F_P11"), data.get_cell_data("F_P12")],
                    axis=-1,
                ),
                np.stack(
                    [data.get_cell_data("F_P21"), data.get_cell_data("F_P22")],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        source = "F_P"
    except KeyError:
        branch = data.get_F()
        source = "F"
    return (*canonical_plastic_shear_counts(branch), data, source)


def get_plastic_shear_ranges(vtu_files, reconnecting=None):
    """Return direction-specific symmetric count limits over one simulation."""
    maxima = [0, 0]
    source = None
    for vtu_file in vtu_files:
        horizontal, vertical, _, frame_source = get_plastic_shear_counts(
            vtu_file, reconnecting
        )
        if source is None:
            source = frame_source
        elif source != frame_source:
            raise ValueError(f"Plastic branch source changed from {source} to {frame_source}")
        maxima[0] = max(maxima[0], int(np.max(np.abs(horizontal), initial=0)))
        maxima[1] = max(maxima[1], int(np.max(np.abs(vertical), initial=0)))
    print(
        f"Plastic shear source={source}, "
        f"horizontal range=±{maxima[0]}, vertical range=±{maxima[1]}"
    )
    return tuple(max(1, value) for value in maxima)


def _infer_ref_grid_dims(dims, n_nodes):
    nx, ny = dims
    if nx * ny == n_nodes:
        return nx, ny
    if (nx + 1) * (ny + 1) == n_nodes:
        return nx + 1, ny + 1
    n = int(np.sqrt(n_nodes))
    if n * n == n_nodes:
        return n, n
    return nx, ny


def extract_center_node_force_series(sim_path, pvd_file="collection.pvd"):
    pvd_path = os.path.join(sim_path, pvd_file)
    if not os.path.exists(pvd_path):
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")

    vtu_files = resolve_vtu_files(pvd_path)
    if not vtu_files:
        raise ValueError(f"No VTU files found in {pvd_path}")

    data0 = VTUData(vtu_files[0])
    nodes0 = data0.get_nodes()
    dims = data0.get_size()
    ref0 = None
    try:
        ref0 = data0.get_reference_nodes()
    except Exception:
        ref0 = None
    n_ref = len(ref0) if ref0 is not None else len(nodes0)
    nx, ny = _infer_ref_grid_dims(dims, n_ref)
    nodes_ref = perfect_grid_nodes((nx, ny))
    if nx > 1 and ny > 1:
        center_idx = grid_index(1, 1, nx)
    else:
        center_idx = _center_node_index(nodes0[:, :2])

    loads = []
    forces = []
    for vtu_file in vtu_files:
        load = infer_strain_from_vtu(vtu_file)
        if load is None or not np.isfinite(load):
            meta = get_data_from_name(vtu_file)
            load = meta.get("load", np.nan)
        loads.append(load)

        force_field = VTUData(vtu_file).get_force_field()
        force_field = np.asarray(force_field)
        if force_field.shape[1] < 2:
            raise ValueError(
                f"Force field must have at least 2 components, got shape {force_field.shape}"
            )
        if center_idx >= force_field.shape[0]:
            center_idx = _center_node_index(nodes0[:, :2])
        forces.append(force_field[center_idx, :2])

    return np.asarray(loads, dtype=float), np.asarray(forces, dtype=float), center_idx


def extract_center_node_conti_force_series(sim_path, pvd_file="collection.pvd"):
    pvd_path = os.path.join(sim_path, pvd_file)
    if not os.path.exists(pvd_path):
        raise FileNotFoundError(f"PVD file not found: {pvd_path}")

    vtu_files = resolve_vtu_files(pvd_path)
    if not vtu_files:
        raise ValueError(f"No VTU files found in {pvd_path}")

    data0 = VTUData(vtu_files[0])
    dims = data0.get_size()
    ref0 = data0.get_reference_nodes()
    nx, ny = _infer_ref_grid_dims(dims, len(ref0))
    if nx > 1 and ny > 1:
        center_idx = grid_index(1, 1, nx)
    else:
        center_idx = _center_node_index(ref0[:, :2])

    loads = []
    lag_forces = []
    eul_forces = []
    for vtu_file in vtu_files:
        load = infer_strain_from_vtu(vtu_file)
        if load is None or not np.isfinite(load):
            meta = get_data_from_name(vtu_file)
            load = meta.get("load", np.nan)
        loads.append(load)

        data = VTUData(vtu_file)
        connectivity = data.get_connectivity()
        ref_nodes = data.get_reference_nodes()
        nx_ref, ny_ref = _infer_ref_grid_dims(dims, len(ref_nodes))
        if nx_ref > 1 and ny_ref > 1:
            center_idx = grid_index(1, 1, nx_ref)
        if connectivity.size and connectivity.max() >= len(ref_nodes):
            raise ValueError(
                f"Connectivity max index {connectivity.max()} exceeds reference nodes {len(ref_nodes)}."
            )
        ref_elem_coords = ref_nodes[connectivity][:, :, :2]
        dN_dX, area_ref = _triangle_shape_grads_and_area(ref_elem_coords)

        F = data.get_F()
        dN_dx, area_cur = shape_grads_and_area_from_F(dN_dX, area_ref, F)

        lag_elem = ContiEnergy.lagrangian_forces_from_F(F, dN_dX, area=area_ref)
        eul_elem = ContiEnergy.eulerian_forces_from_F(F, dN_dx, area=area_cur)

        n_nodes = len(ref_nodes)
        lag_nodes = _assemble_nodal_forces(lag_elem, connectivity, n_nodes)
        eul_nodes = _assemble_nodal_forces(eul_elem, connectivity, n_nodes)

        lag_forces.append(lag_nodes[center_idx, :2])
        eul_forces.append(eul_nodes[center_idx, :2])
    return (
        np.asarray(loads, dtype=float),
        np.asarray(lag_forces, dtype=float),
        np.asarray(eul_forces, dtype=float),
        center_idx,
    )


def plot_center_node_forces(
    sim_paths, labels=None, pvd_file="collection.pvd", plot_mode="scatter"
):
    if labels is None:
        labels = [os.path.basename(p.rstrip(os.sep)) for p in sim_paths]
    if len(labels) != len(sim_paths):
        raise ValueError("labels must match sim_paths length")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
    ax_fx, ax_fy = axes

    plot_mode = str(plot_mode or "line").lower()
    if plot_mode not in ("line", "scatter"):
        raise ValueError("plot_mode must be 'line' or 'scatter'")

    markers = ["o", "s", "^"]

    n_sims = len(sim_paths)
    if n_sims == 1:
        sim_widths = [2.0]
    else:
        sim_widths = np.linspace(2.6, 1.2, n_sims)
    order = np.argsort(sim_widths)[::-1]  # widest first (behind)
    zorder_base = {int(i): 1 + int(rank) for rank, i in enumerate(order)}

    for idx, (path, label) in enumerate(zip(sim_paths, labels)):
        base_w = float(sim_widths[idx])
        z_base = zorder_base[idx]
        color = ax_fx._get_lines.get_next_color()
        loads, forces, _ = extract_center_node_force_series(path, pvd_file=pvd_file)
        loads_c, lag_forces, eul_forces, _ = extract_center_node_conti_force_series(
            path, pvd_file=pvd_file
        )

        if plot_mode == "line":
            line_fx = ax_fx.plot(
                loads,
                forces[:, 0],
                label=f"{label} (simulation)",
                linewidth=base_w,
                color=color,
                zorder=z_base,
            )[0]
            ax_fy.plot(
                loads,
                forces[:, 1],
                label=f"{label} (simulation)",
                linewidth=base_w,
                color=color,
                zorder=z_base,
            )

            ax_fx.plot(
                loads_c,
                lag_forces[:, 0],
                label=f"{label} (lagrangian)",
                linestyle="--",
                linewidth=base_w * 0.7,
                color=color,
                zorder=z_base + 0.1,
            )
            ax_fy.plot(
                loads_c,
                lag_forces[:, 1],
                label=f"{label} (lagrangian)",
                linestyle="--",
                linewidth=base_w * 0.7,
                color=color,
                zorder=z_base + 0.1,
            )

            ax_fx.plot(
                loads_c,
                eul_forces[:, 0],
                label=f"{label} (eulerian)",
                linestyle=":",
                linewidth=base_w * 0.55,
                color=color,
                zorder=z_base + 0.2,
            )
            ax_fy.plot(
                loads_c,
                eul_forces[:, 1],
                label=f"{label} (eulerian)",
                linestyle=":",
                linewidth=base_w * 0.55,
                color=color,
                zorder=z_base + 0.2,
            )
        else:
            marker_stress = markers[0]
            marker_lag = markers[1]
            marker_eul = markers[2]

            ax_fx.scatter(
                loads,
                forces[:, 0],
                label=f"{label} (simulation)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_stress,
                linewidths=1.0,
                zorder=z_base,
            )
            ax_fy.scatter(
                loads,
                forces[:, 1],
                label=f"{label} (simulation)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_stress,
                linewidths=1.0,
                zorder=z_base,
            )

            ax_fx.scatter(
                loads_c,
                lag_forces[:, 0],
                label=f"{label} (lagrangian)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_lag,
                linewidths=1.0,
                zorder=z_base + 0.1,
            )
            ax_fy.scatter(
                loads_c,
                lag_forces[:, 1],
                label=f"{label} (lagrangian)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_lag,
                linewidths=1.0,
                zorder=z_base + 0.1,
            )

            ax_fx.scatter(
                loads_c,
                eul_forces[:, 0],
                label=f"{label} (eulerian)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_eul,
                linewidths=1.0,
                zorder=z_base + 0.2,
            )
            ax_fy.scatter(
                loads_c,
                eul_forces[:, 1],
                label=f"{label} (eulerian)",
                s=18,
                facecolors="none",
                edgecolors=color,
                marker=marker_eul,
                linewidths=1.0,
                zorder=z_base + 0.2,
            )

    ax_fx.set_ylabel(r"Center node $f_x$")
    ax_fy.set_ylabel(r"Center node $f_y$")
    ax_fy.set_xlabel(r"$\gamma$")
    ax_fx.legend(loc="center left")
    fig.tight_layout()
    return fig, axes


# Use this function to set axis limits in your plot_frame function
def get_axis_limits(cvs_file, return_plastic=False):
    desired_cols = [
        "maxX",
        "minX",
        "maxY",
        "minY",
        "max_m3_nr",
        "max_positive_plastic_jump",
        "max_negative_plastic_jump",
    ]
    header = pd.read_csv(cvs_file, nrows=0)
    usecols = [c for c in desired_cols if c in header.columns]
    df = pd.read_csv(cvs_file, usecols=usecols)

    x_max = df["maxX"].max()
    x_min = df["minX"].min()
    y_max = df["maxY"].max()
    y_min = df["minY"].min()

    axis_limits = (x_min, x_max, y_min, y_max)
    if not return_plastic:
        return axis_limits

    plastic_limits = {}
    if "max_m3_nr" in df:
        plastic_limits["max_plastic"] = float(df["max_m3_nr"].max())
    if "max_positive_plastic_jump" in df:
        plastic_limits["max_plastic_change"] = float(
            df["max_positive_plastic_jump"].max()
        )
    if "max_negative_plastic_jump" in df:
        plastic_limits["min_plastic_change"] = float(
            df["max_negative_plastic_jump"].min()
        )

    return axis_limits, plastic_limits


def add_padding(axis_limits, padding_ratio):
    # Define your axis limits
    x_min, x_max, y_min, y_max = axis_limits

    # Calculate padding amounts
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_padding = x_range * padding_ratio
    y_padding = y_range * padding_ratio

    # Adjusted axis limits with padding
    adjusted_x_min = x_min - x_padding
    adjusted_x_max = x_max + x_padding
    adjusted_y_min = y_min - y_padding
    adjusted_y_max = y_max + y_padding

    return adjusted_x_min, adjusted_x_max, adjusted_y_min, adjusted_y_max


def _make_integer_bins(vmin, vmax, n_bins=12, gamma=0.5):
    vmin = int(np.floor(vmin))
    vmax = int(np.ceil(vmax))
    if vmax <= vmin:
        return np.array([vmin], dtype=int)
    if (vmax - vmin) <= n_bins:
        return np.arange(vmin, vmax + 1, dtype=int)
    scaled = np.linspace(0, (vmax - vmin) ** gamma, n_bins + 1)
    bins = np.unique(np.round(vmin + scaled ** (1.0 / gamma)).astype(int))
    bins[0] = vmin
    bins[-1] = vmax
    return bins


def _make_discrete_boundaries(min_val, max_val, n_bins=12, gamma=0.5):
    if min_val >= 0:
        bins = _make_integer_bins(0, max_val, n_bins=n_bins, gamma=gamma)
    elif max_val <= 0:
        bins = -_make_integer_bins(0, -min_val, n_bins=n_bins, gamma=gamma)[::-1]
    else:
        pos_bins = _make_integer_bins(0, max_val, n_bins=n_bins, gamma=gamma)
        neg_bins = -_make_integer_bins(0, -min_val, n_bins=n_bins, gamma=gamma)[::-1]
        bins = np.concatenate([neg_bins[:-1], pos_bins])
    boundaries = np.concatenate([bins - 0.5, [bins[-1] + 0.5]])
    return boundaries


def _discrete_ticks_and_labels(boundaries):
    centers = 0.5 * (boundaries[:-1] + boundaries[1:])
    labels = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        low = int(np.ceil(lo + 0.5))
        high = int(np.floor(hi - 0.5))
        if low == high:
            labels.append(str(low))
        else:
            labels.append(f"{low}..{high}")
    return centers, labels


def _add_frame_table(
    ax,
    vtu_file=None,
    previous_frame_vtu_file=None,
    delta_title=False,
    frame_index=None,
    totalEnergy=None,
    avgRSS=None,
    energyDrop=None,
    delAvgRSS=None,
    delx=None,
    stress_label=None,
    energy_drop_label=None,
    **kwargs,
):
    metaData = get_data_from_name(vtu_file)
    load = infer_strain_from_vtu(vtu_file)
    if load is None or not np.isfinite(load):
        load = metaData["load"]
    load_step = metaData["loadIncrement"]
    nrPlasticEvents = metaData["nrM"]
    nr_func_evals = metaData.get("nr_func_evals")
    if previous_frame_vtu_file:
        previous_load = infer_strain_from_vtu(previous_frame_vtu_file)
        if previous_load is None or not np.isfinite(previous_load):
            previous_load = get_data_from_name(previous_frame_vtu_file)["load"]
        steps_since_last_frame = (
            int((load - previous_load) / load_step)
            if np.isfinite(load_step) and not np.isclose(load_step, 0.0)
            else 0
        )
    else:
        steps_since_last_frame = 0

    stress_label = stress_label or r"\sigma"
    energy_drop_label = energy_drop_label or "E"
    if delta_title:
        data_row = [
            rf"$\Delta\gamma$: {delx:.1e}",
            rf"$\Delta {energy_drop_label}$: {energyDrop:.2e}",
            rf"$\Delta\langle {stress_label} \rangle$: {delAvgRSS:.2e}",
        ]
    else:
        data_row = [
            rf"$\gamma$: {load:.5f}",
            rf"$E$: {totalEnergy:.0f}",
            rf"$\langle {stress_label} \rangle$: {avgRSS:.3f}",
        ]
    data_row.append(rf"$N_p$: {nrPlasticEvents}")
    if nr_func_evals is not None:
        data_row.append(rf"$N_f$: {nr_func_evals}")
    data_row.append(f"f: {frame_index}")
    table = ax.table(cellText=[data_row], cellLoc="center", loc="top", edges="open")
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_linewidth(0)


def base_plot(
    vtu_file=None,
    previous_frame_vtu_file=None,
    axis_limits=None,
    add_title=True,
    delta_title=False,
    frame_index=None,
    totalEnergy=None,
    avgRSS=None,
    energyDrop=None,
    delAvgRSS=None,
    delx=None,
    macroData=None,
    macroDataRowIndex=None,
    equalAspect=True,
    remove_ticks=True,
    dpi=250,
    stress_label=None,
    energy_drop_label=None,
    **kwargs,
):
    quality = 1
    width = 1920 * quality
    height = 1080 * quality
    if not remove_ticks:
        height = 512
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    if equalAspect:
        ax.set_aspect("equal")

    # Setting the axis limits
    if axis_limits:
        x_min, x_max, y_min, y_max = add_padding(axis_limits, 0.03)
        if (
            np.isfinite(x_min)
            and np.isfinite(x_max)
            and np.isfinite(y_min)
            and np.isfinite(y_max)
        ):
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    if add_title:
        _add_frame_table(
            ax,
            vtu_file=vtu_file,
            previous_frame_vtu_file=previous_frame_vtu_file,
            delta_title=delta_title,
            frame_index=frame_index,
            totalEnergy=totalEnergy,
            avgRSS=avgRSS,
            energyDrop=energyDrop,
            delAvgRSS=delAvgRSS,
            delx=delx,
            stress_label=stress_label,
            energy_drop_label=energy_drop_label,
        )
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, fig


def calculate_shifts(ax, vtuData):
    """Calculate shifts needed to cover the visible area based on ax limits and mesh periodicity."""
    if vtuData.BC != "PBC":
        return [(0, 0)]  # No shifts needed for non-periodic BC

    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    N = vtuData.size[0]  # Assumes square mesh
    # Calculate required shifts in x direction
    x_shifts = []
    if xlim[0] < 0 or xlim[1] > N:
        # Need multiple shifts in x direction
        min_x_shift = int(np.floor((xlim[0]) / N)) - 1
        max_x_shift = int(np.ceil((xlim[1]) / N)) + 1
        x_shifts = list(range(min_x_shift * N, (max_x_shift + 1) * N, N))
    else:
        x_shifts = [0]

    # Calculate required shifts in y direction
    y_shifts = []
    if ylim[0] < 0 or ylim[1] > N:
        # Need multiple shifts in y direction
        min_y_shift = int(np.floor((ylim[0])) / N) - 1
        max_y_shift = int(np.ceil((ylim[1]) / N)) + 1
        y_shifts = list(range(min_y_shift * N, (max_y_shift + 1) * N, N))
    else:
        y_shifts = [0]

    # Combine shifts (only need combinations where both are needed)
    if not x_shifts and not y_shifts:
        return [(0, 0)]

    # Create all combinations of shifts
    shifts = [(dx, dy) for dx in x_shifts for dy in y_shifts]

    return shifts


def wrap_periodic_mesh(
    points,
    triangles,
    values,
    reference_indices,
    load,
    box_size,
    *,
    source_path="periodic mesh",
):
    """Map a sheared periodic mesh to a unit square and tile only its edges."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    values = np.asarray(values)
    reference_indices = np.asarray(reference_indices).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape (number of points, at least 2).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (number of triangles, 3).")
    if values.shape != (len(triangles),):
        raise ValueError("One value is required for every triangle.")
    if reference_indices.shape != (len(points),):
        raise ValueError("One reference index is required for every point.")
    if not np.isfinite(load) or box_size <= 0:
        raise ValueError("load must be finite and box_size must be positive.")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise IndexError("triangles contains an invalid point index.")

    origin_candidates = points[reference_indices == 0]
    if len(origin_candidates) == 0:
        raise ValueError(f"No refIndex=0 periodic origin found in {source_path}")
    box = np.array([[box_size, load * box_size], [0.0, box_size]])
    fractional = (points[:, :2] - origin_candidates[0, :2]) @ np.linalg.inv(box).T
    if not np.all(np.isfinite(fractional)):
        raise ValueError(f"Non-finite square-periodic coordinates in {source_path}")
    wrapped = fractional - np.floor(fractional)

    polygons = wrapped[triangles]
    delta = polygons - polygons[:, :1]
    polygons = polygons[:, :1] + delta - np.round(delta)
    polygons -= np.floor(polygons.mean(axis=1))[:, None, :]
    plotted, plotted_values = _tile_periodic_triangles(polygons, values)
    return plotted, plotted_values, wrapped


def _tile_periodic_triangles(polygons, values):
    plotted = []
    plotted_values = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = polygons + np.array([dx, dy])
            lower = shifted.min(axis=1)
            upper = shifted.max(axis=1)
            inside = np.all(upper >= 0, axis=1) & np.all(lower <= 1, axis=1)
            if np.any(inside):
                plotted.append(shifted[inside])
                plotted_values.append(values[inside])
    if not plotted:
        raise RuntimeError("Periodic wrapping produced no visible triangles.")
    return np.concatenate(plotted), np.concatenate(plotted_values)


def tile_periodic_mesh(polygons, values, xlim, ylim):
    """Repeat a unit-cell polygon set only over the requested view window."""
    polygons = np.asarray(polygons, dtype=float)
    values = np.asarray(values)
    if polygons.ndim != 3 or polygons.shape[1:] != (3, 2):
        raise ValueError("polygons must have shape (number of triangles, 3, 2).")
    if values.shape != (len(polygons),):
        raise ValueError("One value is required for every polygon.")
    if len(xlim) != 2 or len(ylim) != 2 or xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        raise ValueError("xlim and ylim must be increasing two-item intervals.")

    plotted = []
    plotted_values = []
    x_shifts = range(int(np.floor(xlim[0])) - 1, int(np.ceil(xlim[1])) + 1)
    y_shifts = range(int(np.floor(ylim[0])) - 1, int(np.ceil(ylim[1])) + 1)
    for dx in x_shifts:
        for dy in y_shifts:
            shifted = polygons + np.array([dx, dy])
            lower = shifted.min(axis=1)
            upper = shifted.max(axis=1)
            inside = (
                (upper[:, 0] >= xlim[0])
                & (lower[:, 0] <= xlim[1])
                & (upper[:, 1] >= ylim[0])
                & (lower[:, 1] <= ylim[1])
            )
            if np.any(inside):
                plotted.append(shifted[inside])
                plotted_values.append(values[inside])
    if not plotted:
        raise RuntimeError("Periodic tiling produced no visible polygons.")
    return np.concatenate(plotted), np.concatenate(plotted_values)


def draw_rhombus(ax, vtuData):
    N = vtuData.size[0]
    if vtuData.BC == "PBC":
        rhombus_x = [0, N, N + vtuData.load * N, vtuData.load * N, 0]
        rhombus_y = [0, 0, N, N, 0]
        ax.plot(rhombus_x, rhombus_y, "k--")


def round_to_nearest_16(x):
    return ((x + 8) // 16) * 16


# Function to save the figure with transparent background and close it
def save_and_close_plot(ax, path, transparent=False):
    # Save the figure using matplotlib
    fig = ax.get_figure()
    # Not fixing this can cause jitter
    ax.xaxis.set_label_coords(0.5, -0.08)  # centered, fixed offset below axes
    fig.tight_layout()
    fig.savefig(
        path,
        transparent=transparent,
    )
    plt.close(fig)

    # Load with OpenCV (preserve alpha if needed)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Check if image has alpha channel
    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False

    # Get new size divisible by 16
    height, width = img.shape[:2]
    new_width = round_to_nearest_16(width)
    new_height = round_to_nearest_16(height)

    # Resize using appropriate interpolation
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save again
    cv2.imwrite(path, resized)


def plot_nodes(vtu_file, ax=None, axis_limits=None, show_connections=False, **kwargs):
    if ax is None:
        ax, fig = base_plot(vtu_file=vtu_file, axis_limits=axis_limits, **kwargs)
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    fixed = data.get_fixed_status()
    dims = get_data_from_name(vtu_file)["dims"]

    #                            Fixed color, Free color
    color = np.where(fixed == 1, "#2a3857", "#d24646")
    x, y = nodes[:, 0], nodes[:, 1]

    # Calculate grid size
    # We use the y axis because it will be closest to the real size.
    grid_size = (axis_limits[3] - axis_limits[2]) / float(dims[1])

    # Calculate frame size and DPI
    inches_per_data_unit = (
        0.6 * fig.dpi * (fig.get_size_inches()[1] / (axis_limits[3] - axis_limits[2]))
    )
    # Calculate circle size in points, considering 's' as the area in points squared
    circle_diameter = 0.5 * grid_size
    circle_radius = circle_diameter / 2
    circle_point_size = (circle_radius * inches_per_data_unit) ** 2

    # Grid
    if show_connections:
        connectivity = data.get_connectivity()
    shifts = calculate_shifts(ax, data)

    for dx, dy in shifts:  # Now unpacking dx, dy from tuples
        sheared_x = x + dx + data.load * dy
        if show_connections:
            # Mesh lines between nodes
            triang = mtri.Triangulation(sheared_x, y + dy, connectivity)
            ax.triplot(triang, color="black", linewidth=0.1, alpha=0.3)

        ax.scatter(
            sheared_x,
            y + dy,
            s=circle_point_size,
            c=color,
            marker="o",
            alpha=1,
            linewidth=0,
        )

    draw_rhombus(ax, data)
    return ax


def pretty_mesh_property(mesh_property):
    if mesh_property == "energy":
        return r"$E_i$"
    elif mesh_property == "stress":
        return r"$\sigma$"
    elif mesh_property == "m":
        # p is the number of times m3 was applied during lagrange reduction
        return r"$N_p$"
    elif mesh_property == "m_diff":
        # p is the number of times m3 was applied during lagrange reduction
        return r"$\Delta N_p$"


def _pretty_math_token(token):
    greek = {
        "alpha": r"\alpha",
        "beta": r"\beta",
        "gamma": r"\gamma",
        "delta": r"\delta",
        "epsilon": r"\epsilon",
        "eta": r"\eta",
        "kappa": r"\kappa",
        "lambda": r"\lambda",
        "mu": r"\mu",
        "nu": r"\nu",
        "omega": r"\omega",
        "phi": r"\phi",
        "pi": r"\pi",
        "rho": r"\rho",
        "sigma": r"\sigma",
        "tau": r"\tau",
        "theta": r"\theta",
        "xi": r"\xi",
        "zeta": r"\zeta",
    }
    mapped = greek.get(str(token).lower())
    if mapped is not None:
        return mapped
    return rf"\mathrm{{{token}}}"


def _pretty_matrix_symbol(matrix_name):
    parts = str(matrix_name).split("_")
    if not parts or not parts[0]:
        raise ValueError(f"Invalid matrix name: {matrix_name!r}")

    base = _pretty_math_token(parts[0])
    if base.startswith(r"\mathrm{") and len(parts[0]) == 1:
        base = parts[0]

    if len(parts) == 1:
        return base

    suffix = r"\,".join(_pretty_math_token(part) for part in parts[1:] if part)
    if not suffix:
        return base
    return f"{base}_{{{suffix}}}"


def pretty_matrix_property(matrix_name):
    return f"${_pretty_matrix_symbol(matrix_name)}_{{ij}}$"


def pretty_matrix_component(matrix_name, i, j):
    return f"${_pretty_matrix_symbol(matrix_name)}_{{{i}{j}}}$"


def _validate_cartesian_viewport(viewport):
    values = np.asarray(viewport, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "cartesian_viewport must contain finite (xmin, xmax, ymin, ymax) values."
        )
    xmin, xmax, ymin, ymax = values
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("cartesian_viewport bounds must be strictly increasing.")
    return tuple(float(value) for value in values)


def plot_mesh(
    vtu_file,
    e_lims=None,
    mesh_property="energy",
    ax=None,
    shift=True,
    add_rombus=True,
    add_m12_marks=False,
    add_colorbar=True,
    max_plastic=10,
    max_plastic_change=4,
    min_plastic_change=-2,
    show_force=False,
    square_periodic_mesh=False,
    periodic_box_size=None,
    cartesian_viewport_culling=False,
    cartesian_viewport=None,
    **kwargs,
):
    if square_periodic_mesh and (
        cartesian_viewport_culling or cartesian_viewport is not None
    ):
        raise ValueError(
            "square_periodic_mesh and cartesian viewport rendering are mutually exclusive."
        )
    if cartesian_viewport is not None:
        cartesian_viewport = _validate_cartesian_viewport(cartesian_viewport)
    # Initialize plot and get data
    ax, data = _initialize_plot(
        vtu_file,
        ax,
        square_periodic_mesh=square_periodic_mesh,
        cartesian_viewport=cartesian_viewport,
        **kwargs,
    )
    if cartesian_viewport is not None:
        ax.set_xlim(*cartesian_viewport[:2])
        ax.set_ylim(*cartesian_viewport[2:])
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    # Configure property-specific settings
    (
        field,
        cmap,
        norm,
        boundaries,
        backgroundColor,
        state_indices,
        tick_positions,
        tick_labels,
    ) = _configure_property_settings(
        data,
        mesh_property,
        e_lims,
        max_plastic,
        max_plastic_change,
        min_plastic_change,
    )

    element_indices = _element_subset_indices(len(connectivity), kwargs.get("element_subset"))
    if element_indices is not None:
        connectivity = connectivity[element_indices]
        field = field[element_indices]
        if state_indices is not None:
            state_indices = state_indices[element_indices]
        if show_force:
            force_contributions = data.get_force_contributions()[element_indices]
        else:
            force_contributions = None
    else:
        force_contributions = data.get_force_contributions() if show_force else None
    # Main plotting
    if square_periodic_mesh:
        if getattr(data, "BC", None) != "PBC":
            raise ValueError("square_periodic_mesh requires periodic boundary conditions.")
        if show_force or add_m12_marks:
            raise NotImplementedError(
                "square_periodic_mesh does not support force vectors or m12 markers."
            )
        if "refIndex" not in data.mesh.point_data:
            raise KeyError("square_periodic_mesh requires the VTU refIndex point field.")
        box_size = data.size[0] if periodic_box_size is None else periodic_box_size
        polygons, field, _ = wrap_periodic_mesh(
            nodes[:, :2],
            connectivity,
            field,
            data.mesh.point_data["refIndex"],
            data.load,
            box_size,
            source_path=vtu_file,
        )
        view_aspect = ax.figure.get_figwidth() / ax.figure.get_figheight()
        polygons, field = tile_periodic_mesh(
            polygons,
            field,
            (0, view_aspect),
            (0, 1),
        )
        mappable = PolyCollection(
            polygons,
            array=field,
            cmap=cmap,
            norm=norm,
            edgecolors=backgroundColor,
            linewidths=0.1,
            antialiaseds=False,
        )
        ax.add_collection(mappable)
        ax.set_xlim(0, view_aspect)
        ax.set_ylim(0, 1)
    else:
        mappable = _plot_mesh_elements(
            ax,
            x,
            y,
            connectivity,
            field,
            norm,
            cmap,
            data,
            mesh_property,
            backgroundColor,
            add_m12_marks,
            state_indices,
            show_force,
            force_contributions,
            cull_to_view=cartesian_viewport_culling,
        )

    # Add additional elements
    _add_additional_elements(
        ax,
        mappable,
        mesh_property,
        add_colorbar,
        boundaries,
        tick_positions,
        tick_labels,
        add_rombus and not square_periodic_mesh,
        nodes,
        data,
        show_force,
    )

    return ax, cmap, norm


def plot_matrix_component_grid(
    vtu_file,
    matrix_name,
    matrix_lims=None,
    ax=None,
    add_colorbar=True,
    add_tile_labels=True,
    **kwargs,
):
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    components = data.get_matrix_components(matrix_name)
    element_indices = _element_subset_indices(
        len(connectivity), kwargs.get("element_subset")
    )
    if element_indices is not None:
        connectivity = connectivity[element_indices]
        components = {
            key: np.asarray(field)[element_indices]
            for key, field in components.items()
        }

    if matrix_lims is None:
        frame_max = max(
            float(np.nanmax(np.abs(np.asarray(field, dtype=float))))
            for field in components.values()
        )
        if not np.isfinite(frame_max):
            raise ValueError(
                f"Could not determine a finite color scale for matrix '{matrix_name}'."
            )
        if frame_max == 0.0:
            frame_max = 1.0
        matrix_lims = (-frame_max, frame_max)

    matrix_abs_max = max(abs(float(matrix_lims[0])), abs(float(matrix_lims[1])))
    matrix_abs_max = 5 # Manually set until fixed
    norm = mcolors.Normalize(vmin=-matrix_abs_max, vmax=matrix_abs_max)
    cmap = "coolwarm"
    backgroundColor = plt.get_cmap(cmap)(0.5)

    x_period = float(data.size[0])
    y_period = float(data.size[1])
    base_limits = kwargs.get("axis_limits")
    if base_limits is None:
        base_limits = (
            float(np.min(x)),
            float(np.max(x)),
            float(np.min(y)),
            float(np.max(y)),
        )

    tile_specs = [
        ((1, 1), (0.0, y_period)),
        ((1, 2), (x_period, y_period)),
        ((2, 1), (0.0, 0.0)),
        ((2, 2), (x_period, 0.0)),
    ]
    super_limits = [
        min(base_limits[0] + dx for _, (dx, dy) in tile_specs),
        max(base_limits[1] + dx for _, (dx, dy) in tile_specs),
        min(base_limits[2] + dy for _, (_, dy) in tile_specs),
        max(base_limits[3] + dy for _, (_, dy) in tile_specs),
    ]

    if ax is None:
        plot_kwargs = dict(kwargs)
        plot_kwargs["axis_limits"] = tuple(super_limits)
        ax, fig = base_plot(vtu_file=vtu_file, **plot_kwargs)

    mappable = None
    for (i, j), (dx, dy) in tile_specs:
        field = np.asarray(components[(i, j)], dtype=float).ravel()
        mappable = _plot_mesh_elements(
            ax,
            x,
            y,
            connectivity,
            field,
            norm,
            cmap,
            data,
            "matrix",
            backgroundColor,
            False,
            None,
            False,
            None,
            shifts=[(dx, dy)],
            apply_shear_to_shift=False,
        )

        if add_tile_labels:
            x0 = base_limits[0] + dx
            x1 = base_limits[1] + dx
            y0 = base_limits[2] + dy
            y1 = base_limits[3] + dy
            ax.text(
                x0 + 0.03 * (x1 - x0),
                y1 - 0.04 * (y1 - y0),
                pretty_matrix_component(matrix_name, i, j),
                ha="left",
                va="top",
                fontsize=11,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    _add_additional_elements(
        ax,
        mappable,
        "matrix",
        add_colorbar,
        None,
        None,
        None,
        False,
        nodes,
        data,
        False,
        colorbar_label=pretty_matrix_property(matrix_name),
    )

    return ax, cmap, norm


def plot_plastic_shear_counts(
    vtu_file,
    plastic_shear_lims,
    reconnecting=None,
    axis_limits=None,
    add_colorbar=True,
    **kwargs,
):
    """Plot tiled signed horizontal and vertical plastic-shear counts."""
    horizontal, vertical, data, _ = get_plastic_shear_counts(
        vtu_file, reconnecting
    )
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    fields = (horizontal, vertical)
    labels = (
        (r"Horizontal integer shear", r"Horizontal shear count $n_h$"),
        (r"Vertical integer shear", r"Vertical shear count $n_v$"),
    )

    dpi = 250
    fig, axes = plt.subplots(2, 1, figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
    if axis_limits is None:
        x, y = nodes[:, 0], nodes[:, 1]
        axis_limits = (np.min(x), np.max(x), np.min(y), np.max(y))
    x_min, x_max, y_min, y_max = add_padding(axis_limits, 0.03)
    target_x_span = 3.2 * (y_max - y_min)
    if x_max - x_min < target_x_span:
        x_center = 0.5 * (x_min + x_max)
        x_min, x_max = x_center - 0.5 * target_x_span, x_center + 0.5 * target_x_span

    for ax, field, limit, (panel_label, colorbar_label) in zip(
        axes, fields, plastic_shear_lims, labels
    ):
        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        boundaries = _make_discrete_boundaries(-limit, limit)
        cmap = plt.get_cmap("coolwarm", len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        mappable = _plot_mesh_elements(
            ax,
            nodes[:, 0],
            nodes[:, 1],
            connectivity,
            field,
            norm,
            cmap,
            data,
            "plastic_shear_count",
            cmap(norm(0)),
            False,
            None,
            False,
            None,
        )
        draw_rhombus(ax, data)
        ax.text(
            0.01,
            0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        if add_colorbar:
            colorbar = fig.colorbar(mappable, ax=ax, pad=0.01, label=colorbar_label)
            ticks, tick_labels = _discrete_ticks_and_labels(boundaries)
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels(tick_labels)

    _add_frame_table(axes[0], vtu_file=vtu_file, **kwargs)
    return axes[0]


def _initialize_plot(vtu_file, ax, **kwargs):
    """Initialize the plot and load VTU data."""
    if ax is None:
        if kwargs.pop("square_periodic_mesh", False):
            kwargs["axis_limits"] = (0, 1, 0, 1)
        else:
            viewport = kwargs.pop("cartesian_viewport", None)
            if viewport is not None:
                kwargs["axis_limits"] = viewport
        ax, fig = base_plot(vtu_file=vtu_file, **kwargs)
    data = VTUData(vtu_file)
    return ax, data


def _configure_property_settings(
    data,
    mesh_property,
    e_lims,
    max_plastic,
    max_plastic_change,
    min_plastic_change,
):
    """Configure property-specific settings like colormaps and norms."""
    cmap = "coolwarm"
    norm = None
    boundaries = None
    backgroundColor = None
    state_indices = None
    tick_positions = None
    tick_labels = None

    if mesh_property == "energy":
        field = data.get_energy_field()
        if e_lims is None:
            e_lims = (min(field), max(field))
        norm = mcolors.Normalize(vmin=e_lims[0], vmax=e_lims[1])
        backgroundColor = plt.get_cmap(cmap)(0)

    elif mesh_property == "stress":
        field = data.get_stress_field()
        norm = mcolors.Normalize(vmin=-1.5, vmax=1.5)
        backgroundColor = plt.get_cmap(cmap)(0.5)

    elif mesh_property == "m":
        cmap = "viridis"
        nrm3 = data.get_m_nr_field()
        field = nrm3
        field_max = float(np.nanmax(field)) if field.size else 0.0
        if not np.isfinite(max_plastic):
            max_plastic = field_max
        else:
            max_plastic = max(float(max_plastic), field_max)
        max_plastic = max(1.0, max_plastic)
        boundaries = _make_discrete_boundaries(0.0, max_plastic, n_bins=12, gamma=0.5)
        cmap = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        backgroundColor = plt.get_cmap(cmap)(0)

        #marker_patterns = ["", "_", "|", "+"]
        #colors = ["", (0.7, 0.7, 0.7), (0.8, 0.2, 0.2), (1, 0.5, 0.5)]
        #state_indices = (nrm1 % 2) + (nrm2 % 2) * 2

    elif mesh_property == "m_diff":
        raw_field = data.get_m3_change_field()
        field, tick_labels = _categorize_m_diff_field(raw_field)
        cmap = "coolwarm"
        boundaries = np.arange(-3.5, 4.5, 1.0)
        cmap = plt.get_cmap(cmap, len(boundaries) - 1)
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        backgroundColor = plt.get_cmap(cmap)(0)
        tick_positions = np.arange(-3, 4, 1)

    # Ensure field is a 1D array for matplotlib collections
    field = np.asarray(field).ravel()

    # If data does not have a load property, we set it to 0
    if not hasattr(data, "load"):
        data.load = 0

    return (
        field,
        cmap,
        norm,
        boundaries,
        backgroundColor,
        state_indices,
        tick_positions,
        tick_labels,
    )


def _categorize_m_diff_field(field):
    """Map m_diff values into signed categorical bins for discrete plotting."""
    field = np.asarray(field)
    abs_val = np.abs(field)

    category = np.zeros_like(abs_val, dtype=int)
    category[abs_val == 1] = 1
    category[abs_val == 2] = 2
    category[(abs_val >= 3) & (abs_val <= 9)] = 3

    sign = np.sign(field)
    sign = np.where(np.isfinite(sign), sign, 0)

    tick_labels = [
        "-3+",
        "-2",
        "-1",
        "0",
        "1",
        "2",
        "3+",
    ]

    return category * sign.astype(int), tick_labels


def _plot_mesh_elements(
    ax,
    x,
    y,
    connectivity,
    field,
    norm,
    cmap,
    data,
    mesh_property,
    backgroundColor,
    add_m12_marks,
    state_indices,
    show_force,
    force_contributions,
    shifts=None,
    apply_shear_to_shift=True,
    cull_to_view=False,
):
    """Optimized version of original approach"""
    edgecolors = "none" if len(x) > 2000 else "black"
    mappable = None

    if shifts is None:
        shifts = calculate_shifts(ax, data)

    for dx, dy in shifts:
        shift_x = dx + data.load * dy if apply_shear_to_shift else dx
        sheared_x = x + shift_x
        sheared_y = y + dy

        if cull_to_view:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            triangle_x = sheared_x[connectivity]
            triangle_y = sheared_y[connectivity]
            visible = (
                (triangle_x.max(axis=1) >= xlim[0])
                & (triangle_x.min(axis=1) <= xlim[1])
                & (triangle_y.max(axis=1) >= ylim[0])
                & (triangle_y.min(axis=1) <= ylim[1])
            )
            if not np.any(visible):
                continue
            plotted_connectivity = connectivity[visible]
            plotted_field = field[visible]
            plotted_state_indices = (
                state_indices[visible] if state_indices is not None else None
            )
            plotted_force_contributions = (
                force_contributions[visible]
                if force_contributions is not None
                else None
            )
        else:
            plotted_connectivity = connectivity
            plotted_field = field
            plotted_state_indices = state_indices
            plotted_force_contributions = force_contributions

        # Create triangulation once per shift
        triang = mtri.Triangulation(
            sheared_x,
            sheared_y,
            triangles=plotted_connectivity,
        )

        # Plot base mesh
        ax.triplot(triang, color=backgroundColor, lw=0.1)

        # Plot colored elements and keep last mappable
        mappable = ax.tripcolor(
            triang,
            facecolors=plotted_field,
            norm=norm,
            cmap=cmap,
            edgecolors=edgecolors,
        )

        # Conditional plotting
        if mesh_property == "m" and add_m12_marks:
            _add_markers(
                ax,
                plotted_connectivity,
                sheared_x,
                sheared_y,
                plotted_state_indices,
            )
        if show_force:
            _plot_force_vectors(
                ax,
                data,
                plotted_connectivity,
                sheared_x,
                sheared_y,
                force_contributions=plotted_force_contributions,
            )

    return mappable


def _add_markers(ax, connectivity, sheared_x, sheared_y, state_indices):
    """Add markers for m1/m2 states."""
    centroids_x = np.mean(sheared_x[connectivity], axis=1)
    centroids_y = np.mean(sheared_y[connectivity], axis=1)

    marker_patterns = ["", "_", "|", "+"]
    colors = ["", (0.7, 0.7, 0.7), (0.8, 0.2, 0.2), (1, 0.5, 0.5)]

    for i in range(1, 4):  # Skip 0
        mask = state_indices == i
        ax.scatter(
            centroids_x[mask],
            centroids_y[mask],
            marker=marker_patterns[i],
            color=colors[i],
            s=1,
            linewidths=0.15,
            zorder=10,
        )


def _plot_force_vectors(
    ax, data, connectivity, sheared_x, sheared_y, force_contributions=None
):
    """Plot force vectors on the mesh, with contributions color-coded by magnitude."""
    if force_contributions is None:
        force_contributions = data.get_force_contributions()
    centroids_x = np.mean(sheared_x[connectivity], axis=1)
    centroids_y = np.mean(sheared_y[connectivity], axis=1)

    quiver_x, quiver_y, quiver_u, quiver_v, colors = [], [], [], [], []

    # plot force contribution vectors
    for elem_idx in range(len(connectivity)):
        node_indices = connectivity[elem_idx]
        centroid_x = centroids_x[elem_idx]
        centroid_y = centroids_y[elem_idx]

        for node_index in range(3):
            node_idx = node_indices[node_index]
            node_x = sheared_x[node_idx]
            node_y = sheared_y[node_idx]

            midpoint_x = (centroid_x + node_x) / 2
            midpoint_y = (centroid_y + node_y) / 2

            force_x = force_contributions[elem_idx, 0, node_index]
            force_y = force_contributions[elem_idx, 1, node_index]

            # Normalize and scale
            magnitude = np.linalg.norm([force_x, force_y])
            scale_factor = max(magnitude, 1e-8)
            length = 0.15
            force_x = force_x / scale_factor * length
            force_y = force_y / scale_factor * length

            quiver_x.append(midpoint_x)
            quiver_y.append(midpoint_y)
            quiver_u.append(force_x)
            quiver_v.append(force_y)
            colors.append(magnitude)

    # Normalize colors for colormap
    norm = mcolors.Normalize(vmin=minForce, vmax=maxForce)
    cmap = plt.cm.coolwarm
    mapped_colors = cmap(norm(colors))

    width = 0.007
    headWidth = 3
    outlineScalse = 1.1
    # Black outlines (drawn first, slightly thicker)
    ax.quiver(
        quiver_x,
        quiver_y,
        quiver_u,
        quiver_v,
        angles="xy",
        scale_units="xy",
        scale=1 / outlineScalse,
        color="black",
        width=width * outlineScalse**2,  # thicker for outline
        headwidth=headWidth * outlineScalse,
        zorder=9,
    )
    quiver = ax.quiver(
        quiver_x,
        quiver_y,
        quiver_u,
        quiver_v,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=mapped_colors,
        width=width,
        headwidth=headWidth,
        zorder=10,
    )

    # plot node force vectors
    force_at_nodes = data.get_force_field()
    scale = 0.01 / maxForce
    ax.quiver(
        sheared_x,
        sheared_y,
        force_at_nodes[:, 0] * scale,
        force_at_nodes[:, 1] * scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.006,
        headwidth=4,
        zorder=11,
    )


def _add_additional_elements(
    ax,
    mappable,
    mesh_property,
    add_colorbar,
    boundaries,
    tick_positions,
    tick_labels,
    add_rombus,
    nodes,
    data,
    show_force,
    colorbar_label=None,
):
    """Add colorbar and rhombus if needed."""
    if add_colorbar and mappable is not None:
        label = colorbar_label
        if label is None:
            label = pretty_mesh_property(mesh_property)
        cbar = plt.colorbar(mappable, ax=ax, label=label)
        if boundaries is not None:
            if tick_positions is not None or tick_labels is not None:
                if tick_positions is None:
                    tick_positions, _ = _discrete_ticks_and_labels(boundaries)
                if tick_labels is None:
                    _, tick_labels = _discrete_ticks_and_labels(boundaries)
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
            else:
                ticks, labels = _discrete_ticks_and_labels(boundaries)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(labels)
            # Normalize colors for colormap
        if show_force:
            norm = mcolors.Normalize(vmin=minForce, vmax=maxForce)
            cmap = plt.cm.coolwarm
            # After defining norm and cmap
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # Required for ScalarMappable

            # Add the colorbar
            cbar = plt.colorbar(sm, ax=ax, label=r"$|F_{ei}|$")

    if add_rombus:
        if hasattr(data, "BC"):
            draw_rhombus(ax, data)


def make_static_plot(fileName, **kwargs):
    ax, fig = base_plot(
        add_title=False, equalAspect=False, remove_ticks=False, dpi=150, **kwargs
    )

    macro_data = kwargs["macro_data"]
    if fileName == "energy_plot":
        makePlot(
            macro_data,
            ax=ax,
            fig=fig,
            Y="total_energy",
            save=False,
        )

    elif fileName == "e_drop_plot":
        plot_plastic_counts([macro_data], ax=ax, save=False)

    return fig, ax


def remove_vlines(ax):
    for line in ax.lines[:]:
        xdata = line.get_xdata()
        # Check if it's a vertical line
        if len(xdata) <= 2 and len(set(xdata)) == 1:
            line.remove()  # Remove the line from the plot


def plot_plot(
    vtu_file,
    ax=None,
    fileName=None,
    energyDrop=None,
    **kwargs,
):
    remove_vlines(ax)
    data = VTUData(vtu_file)

    if fileName == "energy_plot":
        x = data.load
    elif fileName == "e_drop_plot":
        x = -energyDrop

    ax.axvline(
        x=x,
        color="red",
        linewidth=1,
    )

    return ax


# Define a lock to make sure only one thread initializes GRID
grid_lock = threading.Lock()
GRID = None  # Start with GRID as None to check later


# This can be an expensive function, so we want to avoid recalculating it all the time
def get_energy_grid(zoom=1):
    # If GRID is not yet defined, generate it
    global GRID
    if GRID is None:
        with grid_lock:  # Ensure thread-safety while initializing
            if GRID is None:  # Double-check inside the lock
                GRID = generate_energy_grid(
                    resolution=1000, energy_lim=[None, 0.37], zoom=zoom
                )
    return GRID


def reset_energy_grid_cache():
    global GRID
    with grid_lock:
        GRID = None


def _bin_poincare_velocity_field(x, y, u, v, bins=40, zoom=1):
    """Bin scattered velocities on the Poincare disk into a regular grid."""
    if bins is None or bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    r = 1.0 / zoom
    edges = np.linspace(-r, r, bins + 1)

    ix = np.searchsorted(edges, x, side="right") - 1
    iy = np.searchsorted(edges, y, side="right") - 1

    valid = (
        (ix >= 0)
        & (ix < bins)
        & (iy >= 0)
        & (iy < bins)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    if not np.any(valid):
        return None

    sum_u = np.zeros((bins, bins), dtype=float)
    sum_v = np.zeros((bins, bins), dtype=float)
    count = np.zeros((bins, bins), dtype=int)

    np.add.at(sum_u, (iy[valid], ix[valid]), u[valid])
    np.add.at(sum_v, (iy[valid], ix[valid]), v[valid])
    np.add.at(count, (iy[valid], ix[valid]), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_u = np.where(count > 0, sum_u / count, np.nan)
        mean_v = np.where(count > 0, sum_v / count, np.nan)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean_u, mean_v, count


def plot_in_poincare_disk(
    vtu_file, ax=None, fig=None, do_plastic_reduction=False, **kwargs
):
    if ax is None:
        poincare_kwargs = dict(kwargs)
        poincare_kwargs["axis_limits"] = None
        ax, fig = base_plot(vtu_file=vtu_file, **poincare_kwargs)
    data = VTUData(vtu_file)

    poincare_matrix = str(kwargs.get("poincare_matrix", "C")).upper()
    if poincare_matrix == "C":
        disk_matrix = data.get_C()
    elif poincare_matrix == "G":
        disk_matrix = data.get_G()
    else:
        raise ValueError(f"Unsupported Poincare disk matrix: {poincare_matrix!r}")

    element_indices = _element_subset_indices(
        len(disk_matrix), kwargs.get("element_subset")
    )
    if element_indices is not None:
        disk_matrix = disk_matrix[element_indices]
    if do_plastic_reduction:
        # Do the plastic reduction
        disk_matrix, _ = plastic_reduction(disk_matrix)
        zoom = 3
    else:
        zoom = 1

    g = get_energy_grid(zoom=zoom)
    with_grid = kwargs.get("withGrid")
    if with_grid is None:
        with_grid = kwargs.get("poincare_with_grid", True)
    plotEnergyField(
        g,
        fig,
        ax,
        save=False,
        add_title=False,
        zoom=zoom,
        remove_max_color=zoom == 1,
        withYieldSurface=kwargs.get("withYieldSurface", True),
        withGrid=with_grid,
        minimalTicks=kwargs.get("poincare_minimal_ticks", False),
        transformation=kwargs.get("poincare_transformation", None),
        yieldSurface_kwargs=kwargs.get("yieldSurface_kwargs", None),
    )

    legend_label = rf"$\mathbf{{{poincare_matrix}}}$"
    scatter = drawCScatter(
        ax,
        disk_matrix,
        len(g),
        zoom=zoom,
        remove_max_color=False,
        label=legend_label,
    )
    if scatter is not None:
        legend_handle = ax.scatter(
            [],
            [],
            s=3,
            c="black",
            linewidth=0,
            label=legend_label,
        )
        ax.legend(handles=[legend_handle], labels=[legend_label], loc="upper left")

    return ax


def plot_velocity_field_in_poincare_disk(
    vtu_file,
    ax=None,
    fig=None,
    do_plastic_reduction=False,
    previous_frame_vtu_file=None,
    next_frame_vtu_file=None,
    delx=None,
    **kwargs,
):
    use_streamplot = bool(kwargs.get("use_streamplot", False))
    velocity_mode = str(kwargs.get("velocity_mode", "step")).lower() # Velocity or step
    show_velocity_colorbar = True
    velocity_color_min = float(kwargs.get("velocity_color_min", 0.0))
    velocity_color_max = kwargs.get("velocity_color_max", None)
    velocity_scale = kwargs.get("velocity_scale", None)
    velocity_cmap = kwargs.get("velocity_cmap", "viridis")
    velocity_color = kwargs.get("velocity_color", True)
    if ax is None:
        poincare_kwargs = dict(kwargs)
        poincare_kwargs["axis_limits"] = None
        ax, fig = base_plot(vtu_file=vtu_file, **poincare_kwargs)

    data = VTUData(vtu_file)
    C = data.get_C()
    element_indices = _element_subset_indices(len(C), kwargs.get("element_subset"))
    if element_indices is not None:
        C = C[element_indices]
    C_raw = C.copy()
    M = None
    if do_plastic_reduction:
        M = data.get_M(elastic_M=True)
        if element_indices is not None:
            M = M[element_indices]
        M_T = np.swapaxes(M, -1, -2)
        C = np.matmul(np.matmul(M_T, C_raw), M)
        zoom = 3
    else:
        zoom = 1

    strain_step = kwargs.get("velocity_strain_step")
    if strain_step is None:
        try:
            vtu_meta = get_data_from_name(vtu_file)
            strain_step = vtu_meta.get("loadIncrement")
        except Exception:
            strain_step = None
    if velocity_color_max is None:
        if velocity_mode == "velocity":
            if strain_step is not None and np.isfinite(strain_step) and strain_step != 0:
                velocity_color_max = 1.0 / float(strain_step)
            else:
                velocity_color_max = 1.0
        else:
            velocity_color_max = 0.5

    if velocity_scale is None:
        if velocity_mode == "velocity":
            if strain_step is not None and np.isfinite(strain_step):
                velocity_scale = float(strain_step)
            else:
                velocity_scale = 1.0
        else:
            velocity_scale = 1.0
    else:
        velocity_scale = float(velocity_scale)

    g = get_energy_grid(zoom=zoom)
    with_grid = kwargs.get("withGrid")
    if with_grid is None:
        with_grid = kwargs.get("poincare_with_grid", True)
    plotEnergyField(
        g,
        fig,
        ax,
        save=False,
        add_title=False,
        zoom=zoom,
        remove_max_color=zoom == 1,
        withYieldSurface=kwargs.get("withYieldSurface", True),
        withGrid=with_grid,
        minimalTicks=kwargs.get("poincare_minimal_ticks", False),
        transformation=kwargs.get("poincare_transformation", None),
        yieldSurface_kwargs=kwargs.get("yieldSurface_kwargs", None),
    )
    if kwargs.get("velocity_show_density_background", True):
        drawCScatter(
            ax,
            C,
            len(g),
            zoom=zoom,
            remove_max_color=False,
            show_colorbar=False,
            alpha=float(kwargs.get("velocity_density_alpha", 0.2)),
            zorder=4,
        )

    def _ensure_colorbar(vmin, vmax):
        if not show_velocity_colorbar or not velocity_color:
            return
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=velocity_cmap)
        sm.set_array([])
        cbar = plt.colorbar(
            sm,
            ax=ax,
            label=r"$\langle |\mathbf{C}_v|\rangle$",
            pad=0.01,
        )
        if hasattr(cbar, "solids") and cbar.solids is not None:
            cbar.solids.set_alpha(1.0)

    ref_frame_vtu_file = (
        next_frame_vtu_file if velocity_mode == "step" else previous_frame_vtu_file
    )
    if ref_frame_vtu_file is None:
        if kwargs.get("velocity_show_points", False):
            drawCScatter(
                ax,
                C,
                len(g),
                zoom=zoom,
                remove_max_color=False,
                show_colorbar=False,
            )
        _ensure_colorbar(velocity_color_min, float(velocity_color_max))
        return ax

    ref_data = VTUData(ref_frame_vtu_file)
    C_prev = ref_data.get_C()
    M_prev = ref_data.get_M(elastic_M=True) if do_plastic_reduction else None
    if element_indices is not None:
        element_indices = element_indices[element_indices < len(C_prev)]
        if element_indices.size == 0:
            return ax
        C_prev = C_prev[element_indices]
        if M_prev is not None:
            M_prev = M_prev[element_indices]
        if len(C) > len(element_indices):
            C = C[: len(element_indices)]
            if M is not None:
                M = M[: len(element_indices)]
    else:
        n = min(len(C), len(C_prev))
        if n == 0:
            return ax
        C = C[:n]
        C_prev = C_prev[:n]
        if M is not None:
            M = M[:n]
        if M_prev is not None:
            M_prev = M_prev[:n]
    C_prev_raw = C_prev.copy()
    if do_plastic_reduction:
        M_prev_T = np.swapaxes(M_prev, -1, -2)
        C_prev = np.matmul(np.matmul(M_prev_T, C_prev_raw), M_prev)
        if M is not None and M_prev is not None:
            try:
                dM = np.matmul(np.linalg.inv(M_prev), M)
                dM_T = np.swapaxes(dM, -1, -2)
                C_prev = np.matmul(np.matmul(dM_T, C_prev), dM)
            except np.linalg.LinAlgError:
                pass

    x, y = C2Plane(C, plane="PoincareDisk")
    x_prev, y_prev = C2Plane(C_prev, plane="PoincareDisk")

    n = min(x.size, x_prev.size)
    if n == 0:
        return ax
    x, y, x_prev, y_prev = x[:n], y[:n], x_prev[:n], y_prev[:n]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_prev) & np.isfinite(y_prev)
    if not np.any(mask):
        return ax

    if velocity_mode == "step":
        dx = x_prev[mask] - x[mask]
        dy = y_prev[mask] - y[mask]
    else:
        dx = x[mask] - x_prev[mask]
        dy = y[mask] - y_prev[mask]

    delx_val = delx
    if velocity_mode == "velocity":
        if delx_val is None or not np.isfinite(delx_val) or delx_val == 0:
            try:
                current_load = infer_strain_from_vtu(vtu_file)
                previous_load = infer_strain_from_vtu(ref_frame_vtu_file)
                if current_load is None or not np.isfinite(current_load):
                    current_load = get_data_from_name(vtu_file)["load"]
                if previous_load is None or not np.isfinite(previous_load):
                    previous_load = get_data_from_name(ref_frame_vtu_file)["load"]
                delx_val = current_load - previous_load
            except Exception:
                delx_val = 1.0
        if not np.isfinite(delx_val) or delx_val == 0:
            delx_val = 1.0
        u = dx / delx_val
        v = dy / delx_val
    else:
        delx_val = 1.0
        u = dx
        v = dy
    velocity_width = float(kwargs.get("velocity_width", 0.002))
    velocity_headwidth = float(kwargs.get("velocity_headwidth", 3.0))

    grid_size = len(g)
    scale_xy = zoom * grid_size / 2
    x_m = x[mask]
    y_m = y[mask]
    x_plot = x_m * scale_xy + grid_size / 2
    y_plot = y_m * scale_xy + grid_size / 2
    u_plot = u * scale_xy * velocity_scale
    v_plot = v * scale_xy * velocity_scale

    if use_streamplot:
        velocity_grid_size = int(kwargs.get("velocity_grid_size", 100))
        velocity_min_count = int(kwargs.get("velocity_min_count", 3))
        binned = _bin_poincare_velocity_field(
            x_m, y_m, u, v, bins=velocity_grid_size, zoom=zoom
        )
        if binned is None:
            _ensure_colorbar(velocity_color_min, velocity_color_max)
            return ax
        centers, mean_u, mean_v, count = binned

        Xc, Yc = np.meshgrid(centers, centers)
        valid_bins = (
            (count >= velocity_min_count) & np.isfinite(mean_u) & np.isfinite(mean_v)
        )
        if not np.any(valid_bins):
            _ensure_colorbar(velocity_color_min, velocity_color_max)
            return ax

        u_grid = mean_u * scale_xy * velocity_scale
        v_grid = mean_v * scale_xy * velocity_scale
        x_grid = centers * scale_xy + grid_size / 2
        y_grid = centers * scale_xy + grid_size / 2

        invalid = ~valid_bins | ~np.isfinite(u_grid) | ~np.isfinite(v_grid)
        u_masked = np.ma.array(u_grid, mask=invalid)
        v_masked = np.ma.array(v_grid, mask=invalid)
        if velocity_color:
            vmin = velocity_color_min
            vmax = velocity_color_max
            velocity_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            speed = np.ma.sqrt(u_masked**2 + v_masked**2)
            stream = ax.streamplot(
                x_grid,
                y_grid,
                u_masked,
                v_masked,
                color=speed,
                cmap=velocity_cmap,
                norm=velocity_norm,
                linewidth=velocity_width * 300,
                density=1.0,
                zorder=8,
            )
            if show_velocity_colorbar:
                cbar = plt.colorbar(
                    stream.lines,
                    ax=ax,
                    label=r"$\langle |\mathbf{C}_v|\rangle$",
                    pad=0.01,
                )
                if hasattr(cbar, "solids") and cbar.solids is not None:
                    cbar.solids.set_alpha(1.0)
        else:
            ax.streamplot(
                x_grid,
                y_grid,
                u_masked,
                v_masked,
                color=kwargs.get("velocity_color_value", "black"),
                linewidth=velocity_width * 300,
                density=1.0,
                zorder=8,
            )
    else:
        quiver_kwargs = dict(
            angles="xy",
            scale_units="xy",
            scale=1,
            width=velocity_width,
            headwidth=velocity_headwidth,
            zorder=8,
        )

        if velocity_color:
            mag = np.sqrt(u**2 + v**2)
            vmin = velocity_color_min
            vmax = velocity_color_max
            velocity_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            quiver = ax.quiver(
                x_plot,
                y_plot,
                u_plot,
                v_plot,
                mag,
                cmap=velocity_cmap,
                norm=velocity_norm,
                **quiver_kwargs,
            )
            if show_velocity_colorbar:
                cbar = plt.colorbar(
                    quiver, ax=ax, label=r"$\langle |\mathbf{C}_v| \rangle$", pad=0.01
                )
                if hasattr(cbar, "solids") and cbar.solids is not None:
                    cbar.solids.set_alpha(1.0)
        else:
            ax.quiver(
                x_plot,
                y_plot,
                u_plot,
                v_plot,
                color=kwargs.get("velocity_color_value", "black"),
                **quiver_kwargs,
            )

    if kwargs.get("velocity_show_points", False):
        drawCScatter(ax, C, len(g), zoom=zoom, remove_max_color=False)

    return ax


def plot_and_save_in_poincare_disk(**kwargs):
    return plot_and_save(
        plot_func=plot_in_poincare_disk,
        **kwargs,
    )


def plot_and_save_g_in_poincare_disk(**kwargs):
    kwargs["poincare_matrix"] = "G"
    return plot_and_save(
        plot_func=plot_in_poincare_disk,
        **kwargs,
    )


def plot_and_save_in_plastic_reduced_poincare_disk(**kwargs):
    return plot_and_save(
        plot_func=plot_in_poincare_disk,
        do_plastic_reduction=True,
        **kwargs,
    )


def plot_and_save_velocity_field_in_plastic_reduced_poincare_disk(**kwargs):
    return plot_and_save(
        plot_func=plot_velocity_field_in_poincare_disk,
        do_plastic_reduction=True,
        **kwargs,
    )


def plot_and_save_mesh(**kwargs):
    return plot_and_save(
        plot_func=plot_mesh,
        mesh_property="energy",
        **kwargs,
    )


def plot_and_save_matrix_component_grid(**kwargs):
    return plot_and_save(
        plot_func=plot_matrix_component_grid,
        **kwargs,
    )


def plot_and_save_plastic_shear_counts(**kwargs):
    return plot_and_save(
        plot_func=plot_plastic_shear_counts,
        **kwargs,
    )


def plot_and_save_mesh_with_force(**kwargs):
    systemSize = get_data_from_name(kwargs["vtu_file"])["L"]
    if systemSize > 50:
        print(
            f"Warning: System size is {systemSize}, plotting forces might be very slow!"
        )
    return plot_and_save(
        plot_func=plot_mesh,
        mesh_property="energy",
        show_force=True,
        **kwargs,
    )


def plot_and_save_m_mesh(**kwargs):
    return plot_and_save(
        plot_func=plot_mesh,
        mesh_property="m",
        **kwargs,
    )


def plot_and_save_plot(**kwargs):
    return plot_and_save(
        plot_func=plot_plot,
        **kwargs,
    )


def plot_and_save_m_diff_mesh(**kwargs):
    return plot_and_save(
        plot_func=plot_mesh,
        mesh_property="m_diff",
        delta_title=True,
        **kwargs,
    )


def plot_and_save_nodes(**kwargs):
    return plot_and_save(
        plot_func=plot_nodes,
        remove_keys=["e_lims"],
        **kwargs,
    )


def plot_and_save(
    plot_func,
    frame_path,
    frame_index,
    transparent,
    return_axes_index=0,
    reuse_images=False,
    **kwargs,
):
    # Join using Path, which allows simpler syntax
    fileName = kwargs["fileName"]
    path = Path(frame_path) / fileName / f"{fileName}_frame_{frame_index:04d}.png"
    os.makedirs(path.parent, exist_ok=True)

    # If we want to resuse and the path already exsists
    if reuse_images and os.path.exists(path):
        return path

    # Call the plot function
    plot_result = plot_func(frame_index=frame_index, **kwargs)
    # Handle functions that return multiple values
    if isinstance(plot_result, tuple):
        ax = plot_result[return_axes_index]
    else:
        ax = plot_result
    # Save and close the plot
    save_and_close_plot(ax, path, transparent)
    return path


def process_frame(kwargs, attemps=0):
    kwargs = kwargs.copy()
    # Unpack frameFunction from kwargs and apply retry logic
    frameFunction = kwargs.pop("frameFunction")

    # Sometimes, we get: Exception has occurred: SyntaxError not a PNG file
    # This is a bit random, so we just try again
    try:
        # Call frameFunction with remaining keyword arguments
        return frameFunction(**kwargs)
    except SyntaxError:
        if attemps < 5:
            kwargs["frameFunction"] = frameFunction
            return process_frame(kwargs, attemps=attemps + 1)
        raise


def _resolve_stress_column(df, macro_data):
    if "avg_sigma12" in df.columns:
        return "avg_sigma12", r"\sigma"
    if "avg_RSS" in df.columns:
        print(
            f"Warning: 'avg_sigmaxy' not found in {macro_data}. Using 'avg_RSS' instead."
        )
        return "avg_RSS", r"\mathrm{RSS}"
    if "avg_P12" in df.columns:
        print(
            f"Warning: 'avg_sigmaxy' not found in {macro_data}. Using 'avg_Pxy' instead."
        )
        return "avg_P12", r"P_{xy}"
    raise KeyError(
        "Missing stress column: expected 'avg_sigmaxy' (or fallback 'avg_RSS'/'avg_Pxy')."
    )


def get_corresponding_energy_and_rss(
    vtu_files,
    macro_data,
    X="load",
    energy_type="e_change_from_init",
    averageEnergy=False,
    stress_corrected=False,
):
    """
    Extracts the corresponding "avg_energy" and stress values for each load in vtu_files,
    along with the line numbers (indices) of the matching rows in the CSV file.

    Parameters:
        vtu_files (List[str]): List of VTU file names.
        macro_data (str): Path to the CSV file containing macro data.

    Returns:
        Tuple[List[float], List[float], List[int]]: Lists of average energy, stress values,
        and line numbers of matching rows. Also returns the stress label used for plotting.
    """
    _, drops_info = get_energy_drops(
        macro_data,
        strainLim="all",
        debug=False,
        label=None,
        energy_type=energy_type,
        averageEnergy=averageEnergy,
        stress_corrected=stress_corrected,
    )
    df = drops_info["df"]
    energy_key = drops_info["key"]
    stress_col, stress_label = _resolve_stress_column(df, macro_data)
    diffs = df[energy_key].copy()
    if np.all(diffs >= 0):
        # Keep the sign convention consistent with get_energy_drops
        diffs = -diffs
    total_energy_list = []
    change_energy_list = []
    avg_stress_list = []
    line_numbers = []
    x_list = []

    for vtu_file in vtu_files:
        matching_row, matching_row_index, x = match_vtu_to_macro_row(
            df, vtu_file, X=X
        )
        x_list.append(x)
        line_numbers.append(matching_row_index)

        # Append the extracted values to the respective lists
        total_energy_list.append(matching_row["total_energy"])
        avg_stress_list.append(matching_row[stress_col])
        change_energy_list.append(diffs.iloc[matching_row_index])

    # Find previous data and get change data as well
    px, pTotalEnergy, pAvgRSS = get_previous_energy_and_rss(
        macro_data, line_numbers, X, stress_col=stress_col
    )
    avg_stress_arr = np.array(avg_stress_list)
    change_avg_stress_list = avg_stress_arr - pAvgRSS
    del_x = np.array(x_list) - px

    # Return the lists of values and line numbers
    return (
        total_energy_list,
        avg_stress_list,
        change_energy_list,
        change_avg_stress_list,
        del_x,
        line_numbers,
        stress_label,
    )


def get_previous_energy_and_rss(
    macro_data,
    current_line,
    X="load",
    energy_col="total_energy",
    stress_col="avg_sigma12",
):
    df = read_macrodata_csv(macro_data)
    # Check if current_line is an integer
    if isinstance(current_line, int):
        # Select the previous row relative to current_line
        p_row = df.iloc[current_line - 1]
        return p_row[X], p_row[energy_col], p_row[stress_col]
    else:
        # Handle the case where current_line is an iterable (e.g., list or array)
        # Create empty lists to store previous values
        prev_x, prev_energies, prev_rss = [], [], []

        for line in current_line:
            # Ensure line index is valid (i.e., not the first row)
            line = max(1, line)
            p_row = df.iloc[line - 1]
            prev_x.append(p_row[X])
            prev_energies.append(p_row[energy_col])
            prev_rss.append(p_row[stress_col])

        # Return lists of previous values
        return np.array(prev_x), np.array(prev_energies), np.array(prev_rss)


def make_images(vtu_files, num_processes=-2, use_tqdm=True, X="load", **kwargs):
    print(f"Processing {kwargs['fileName']} video meta data...")
    if not vtu_files:
        raise ValueError("No VTU files provided to make_images().")
    # Calculate global axis limits and energy range
    macro_data = kwargs["macro_data"]
    if macro_data:
        axis_limits, plastic_limits = get_axis_limits(macro_data, return_plastic=True)
        e_lims = get_energy_range(vtu_files, macro_data)
        e_lims[1] = min(e_lims[1], 0.3)  # optional custom limit
        energy_type = kwargs.get("energy_type", "e_change_from_init")
        averageEnergy = kwargs.get("averageEnergy", False)
        stress_corrected = kwargs.get("stress_corrected", False)
        (
            totalEnergy,
            avgRSS,
            energyDrop,
            delAvgRSS,
            delx,
            macroDataRowIndex,
            stress_label,
        ) = get_corresponding_energy_and_rss(
            vtu_files,
            macro_data,
            X,
            energy_type=energy_type,
            averageEnergy=averageEnergy,
            stress_corrected=stress_corrected,
        )
        kwargs["energy_drop_label"] = energy_drop_label(
            energy_type=energy_type,
            stress_corrected=stress_corrected,
            use_avg=averageEnergy,
        )
        if isinstance(plastic_limits, dict):
            if "max_plastic" in plastic_limits and "max_plastic" not in kwargs:
                kwargs["max_plastic"] = plastic_limits["max_plastic"]
            if (
                "max_plastic_change" in plastic_limits
                and "max_plastic_change" not in kwargs
            ):
                kwargs["max_plastic_change"] = plastic_limits["max_plastic_change"]
            if (
                "min_plastic_change" in plastic_limits
                and "min_plastic_change" not in kwargs
            ):
                kwargs["min_plastic_change"] = plastic_limits["min_plastic_change"]

    else:
        # set default values
        axis_limits = None
        e_lims = [0, 0.03]
        totalEnergy = [0] * len(vtu_files)
        avgRSS = [0] * len(vtu_files)
        energyDrop = [0] * len(vtu_files)
        delAvgRSS = [0] * len(vtu_files)
        delx = [0] * len(vtu_files)
        macroDataRowIndex = [0] * len(vtu_files)
        stress_label = r"\sigma"
        # make default macro data
        macro_data = {X: 0, "loadIncrement": 0, "nrM": 0}
        kwargs["macro_data"] = macro_data
    kwargs["stress_label"] = stress_label

    if kwargs.get("matrix_name") is not None and "matrix_lims" not in kwargs:
        kwargs["matrix_lims"] = get_matrix_range(vtu_files, kwargs["matrix_name"])

    # Some ploting functions cannot handle multithreading
    # in particular, if we want to reuse a plot many times
    if "plot" in kwargs["fileName"]:
        multithread = False
        fig, ax = make_static_plot(**kwargs)
        kwargs["ax"] = ax
        kwargs["fig"] = fig

    else:
        multithread = True

    # Create a list of dictionaries for keyword arguments
    kwargs_list = [
        {
            "vtu_file": vtu_files[i],
            "previous_frame_vtu_file": vtu_files[i - 1] if i != 0 else None,
            "next_frame_vtu_file": vtu_files[i + 1] if i + 1 < len(vtu_files) else None,
            "frame_index": i,
            "e_lims": e_lims,
            "axis_limits": axis_limits,
            "totalEnergy": totalEnergy[i],
            "avgRSS": avgRSS[i],
            "energyDrop": energyDrop[i],
            "delAvgRSS": delAvgRSS[i],
            "delx": delx[i],
            "macroDataRowIndex": macroDataRowIndex[i],
            **kwargs,
        }
        for i in range(len(vtu_files))
    ]

    print(f"Processing {kwargs['fileName']} video frames...")
    # Limited memory for large systems
    L = get_data_from_name(vtu_files[0])["L"]
    if L > 300:
        num_processes = 1
    elif L > 200:
        num_processes = 2

    if "disk" in kwargs["fileName"]:
        reset_energy_grid_cache()
    if multithread and num_processes != 1:
        if num_processes < 0:
            import multiprocessing

            # Negative means "max + num_processes"
            num_processes = multiprocessing.cpu_count() + num_processes
        image_paths = []

        image_paths.append(process_frame(kwargs_list[0]))
        remaining = kwargs_list[1:]
        with Pool(processes=num_processes) as pool:
            image_paths.extend(
                list(
                    tqdm(
                        pool.imap(process_frame, remaining),
                        total=len(remaining),
                        disable=not use_tqdm,
                    )
                )
            )
    else:
        image_paths = [
            process_frame(kwargs) for kwargs in tqdm(kwargs_list, disable=not use_tqdm)
        ]

    return image_paths

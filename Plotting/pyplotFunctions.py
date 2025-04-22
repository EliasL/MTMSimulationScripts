import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import os
import cv2
from multiprocessing import Pool
from pathlib import Path

import threading
from MTMath.plotEnergy import (
    plotEnergyField,
    generate_energy_grid,
    drawCScatter,
    lagrange_reduction,
    # elastic_reduction,
)
from Management.jobs import propperJob

from MTMath.contiPotential import ContiEnergy
from .makePlots import makePlot, makeLogPlotComparison
from .remotePlotting import get_csv_files
from .dataFunctions import get_data_from_name, VTUData, CArrsToMat, get_previous_data

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


# This is a conceptual approach and might need adjustments to fit your specific data structure
def cell_energy_to_node_energy(nodes, energy_field, connectivity):
    node_energy = np.zeros(len(nodes))
    node_count = np.zeros(len(nodes))

    for cell_index, cell in enumerate(connectivity):
        for node_index in cell:
            node_energy[node_index] += energy_field[cell_index]
            node_count[node_index] += 1

    # Avoid division by zero for isolated nodes if any (shouldn't happen in a well-defined mesh)
    if (node_count == 0).any():
        raise (Exception("Invalid Mesh"))
    node_energy /= node_count

    return node_energy


# Use this function to set axis limits in your plot_frame function
def get_axis_limits(cvs_file):
    df = pd.read_csv(cvs_file, usecols=["maxX", "minX", "maxY", "minY"])

    x_max = df["maxX"].max()
    x_min = df["minX"].min()
    y_max = df["maxY"].max()
    y_min = df["minY"].min()

    return x_min, x_max, y_min, y_max


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


def base_plot(
    vtu_file=None,
    previous_frame_vtu_file=None,
    axis_limits=None,
    add_title=True,
    delta_title=False,
    frame_index=None,
    avgEnergy=None,
    avgRSS=None,
    delAvgEnergy=None,
    delAvgRSS=None,
    delLoad=None,
    macroData=None,
    macroDataRowIndex=None,
    equalAspect=True,
    remove_ticks=True,
    dpi=250,
    **kwargs,
):
    quality = 1
    width = 1920 * quality
    height = 1080 * quality
    if not remove_ticks:
        height = 500
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
        metaData = get_data_from_name(vtu_file)
        load = metaData["load"]
        load_step = metaData["loadIncrement"]
        nrPlasticEvents = metaData["nrM"]
        if "nr_func_evals" in metaData:
            nr_func_evals = metaData["nr_func_evals"]
        else:
            nr_func_evals = None
        if previous_frame_vtu_file:
            previous_load = get_data_from_name(previous_frame_vtu_file)["load"]
            steps_since_last_frame = int((load - previous_load) / load_step)
        else:
            steps_since_last_frame = 0

        if delta_title:
            data_row = [
                rf"$\Delta\gamma$: {delLoad:.1e}",
                rf"$\Delta\langle E \rangle$: {delAvgEnergy:.2e}",
                rf"$\Delta\langle \sigma \rangle$: {delAvgRSS:.2e}",
            ]
        else:
            data_row = [
                rf"$\gamma$: {load:.5f}",
                rf"$\langle E \rangle$: {avgEnergy:.3f}",
                rf"$\langle \sigma \rangle$: {avgRSS:.3f}",
            ]

        data_row.append(rf"$N_p$: {nrPlasticEvents}")
        if nr_func_evals is not None:
            data_row.append(rf"$N_f$: {nr_func_evals}")
        data_row.append(f"f: {frame_index}")

        data = [data_row]
        # Create the table with invisible borders and gridlines
        table = ax.table(cellText=data, cellLoc="center", loc="top", edges="open")

        table.set_fontsize(10)

        # Remove the cell edges to keep it invisible
        for cell in table.get_celld().values():
            cell.set_linewidth(0)
    if remove_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, fig


def calculate_shifts(nodes, BC, load):
    N = np.sqrt(len(nodes[:, 0])) - 1
    if BC == "PBC":
        return [-N, 0, N]
    return [0]


def draw_rhombus(ax, N, load, BC):
    if BC == "PBC":
        rhombus_x = [0, N, N + load * N, load * N, 0]
        rhombus_y = [0, 0, N, N, 0]
        ax.plot(rhombus_x, rhombus_y, "k--")


def round_to_nearest_16(x):
    return ((x + 8) // 16) * 16


# Function to save the figure with transparent background and close it
def save_and_close_plot(ax, path, transparent=False):
    # Save the figure using matplotlib
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
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
    shifts = calculate_shifts(nodes, data.BC, data.load)
    for dx in shifts:
        for dy in shifts:
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

    draw_rhombus(ax, np.sqrt(len(nodes[:, 0])) - 1, data.load, data.BC)
    return ax


def pretty_mesh_property(mesh_property):
    if mesh_property == "energy":
        return r"$E$"
    elif mesh_property == "stress":
        return r"$\sigma$"
    elif mesh_property == "m":
        # p is the number of times m3 was applied during lagrange reduction
        return r"$n_p$"
    elif mesh_property == "m_diff":
        # p is the number of times m3 was applied during lagrange reduction
        return r"$\Delta n_p$"


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
    **kwargs,
):
    # Initialize plot and get data
    ax, data = _initialize_plot(vtu_file, ax, **kwargs)
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    # Configure property-specific settings
    field, cmap, norm, boundaries, backgroundColor, state_indices = (
        _configure_property_settings(
            data,
            mesh_property,
            e_lims,
            max_plastic,
            max_plastic_change,
            min_plastic_change,
        )
    )

    # Calculate shifts if needed
    shifts = _calculate_shifts_if_needed(shift, nodes, data)

    # Main plotting
    mappable = _plot_mesh_elements(
        ax,
        x,
        y,
        connectivity,
        field,
        norm,
        cmap,
        shifts,
        data,
        mesh_property,
        backgroundColor,
        add_m12_marks,
        state_indices,
        show_force,
    )

    # Add additional elements
    _add_additional_elements(
        ax,
        mappable,
        mesh_property,
        add_colorbar,
        boundaries,
        add_rombus,
        nodes,
        data,
        show_force,
    )

    return ax, cmap, norm


def _initialize_plot(vtu_file, ax, **kwargs):
    """Initialize the plot and load VTU data."""
    if ax is None:
        ax, fig = base_plot(vtu_file=vtu_file, **kwargs)
    data = VTUData(vtu_file)
    return ax, data


def _configure_property_settings(
    data, mesh_property, e_lims, max_plastic, max_plastic_change, min_plastic_change
):
    """Configure property-specific settings like colormaps and norms."""
    cmap = "coolwarm"
    norm = None
    boundaries = None
    backgroundColor = None
    state_indices = None

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
        nrm1, nrm2, nrm3 = data.get_m_nr_field()
        field = nrm3
        if max(field) > max_plastic:
            raise RuntimeError(
                f"Huge plastic deformation! Extend max_plastic to {max(field)}."
            )

        boundaries = np.arange(0, max_plastic + 1) - 0.5
        norm = mcolors.BoundaryNorm(boundaries, plt.get_cmap(cmap).N)
        backgroundColor = plt.get_cmap(cmap)(0)

        marker_patterns = ["", "_", "|", "+"]
        colors = ["", (0.7, 0.7, 0.7), (0.8, 0.2, 0.2), (1, 0.5, 0.5)]
        state_indices = (nrm1 % 2) + (nrm2 % 2) * 2

    elif mesh_property == "m_diff":
        field = data.get_m3_change_field()
        if max(field) > max_plastic_change:
            raise RuntimeError(
                f"Huge plastic jump! Extend max_plastic_change to {max(field)}."
            )
        if min(field) < min_plastic_change:
            raise RuntimeError(
                f"Huge negative plastic jump! Extend min_plastic_change to {min(field)}."
            )

        color_list = ["blue", "lightblue", "white", "lightcoral", "red", "darkred"]
        cmap = mcolors.ListedColormap(color_list)
        boundaries = np.arange(min_plastic_change, max_plastic_change + 1) - 0.5
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        backgroundColor = plt.get_cmap(cmap)(0)

    # If data does not have a load property, we set it to 0
    if not hasattr(data, "load"):
        data.load = 0

    return field, cmap, norm, boundaries, backgroundColor, state_indices


def _calculate_shifts_if_needed(shift, nodes, data):
    """Calculate shifts if shift is True, otherwise return [0]."""
    # handle Exception has occurred: AttributeError 'VTUData' object has no attribute 'BC'
    if not hasattr(data, "BC"):
        return [0]
    else:
        return calculate_shifts(nodes, data.BC, data.load) if shift else [0]


def _plot_mesh_elements(
    ax,
    x,
    y,
    connectivity,
    field,
    norm,
    cmap,
    shifts,
    data,
    mesh_property,
    backgroundColor,
    add_m12_marks,
    state_indices,
    show_force,
):
    """Plot the main mesh elements with appropriate coloring."""
    edgecolors = "none" if len(x) > 2000 else "black"
    mappable = None

    for dx in shifts:
        for dy in shifts:
            sheared_x = x + dx + data.load * dy
            sheared_y = y + dy
            triang = mtri.Triangulation(sheared_x, sheared_y, connectivity)

            # Plot the base mesh
            ax.triplot(triang, color=backgroundColor, lw=0.1)

            # Plot the colored elements
            mappable = ax.tripcolor(
                triang,
                facecolors=field,
                norm=norm,
                cmap=cmap,
                edgecolors=edgecolors,
            )

            # Add markers if needed
            if mesh_property == "m" and add_m12_marks:
                _add_markers(ax, connectivity, sheared_x, sheared_y, state_indices)

            # Add force vectors if needed
            if show_force:
                _plot_force_vectors(ax, data, connectivity, sheared_x, sheared_y)

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


def _plot_force_vectors(ax, data, connectivity, sheared_x, sheared_y):
    """Plot force vectors on the mesh, with contributions color-coded by magnitude."""
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
    scale = 0.1 / maxForce
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
    add_rombus,
    nodes,
    data,
    show_force,
):
    """Add colorbar and rhombus if needed."""
    if add_colorbar and mappable is not None:
        cbar = plt.colorbar(mappable, ax=ax, label=pretty_mesh_property(mesh_property))
        if boundaries is not None:
            ticks = boundaries[:-1] + 0.5
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(int(t)) for t in ticks])
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
            draw_rhombus(ax, np.sqrt(len(nodes[:, 0])) - 1, data.load, data.BC)


def make_static_plot(fileName, **kwargs):
    ax, fig = base_plot(
        add_title=False, equalAspect=False, remove_ticks=False, dpi=150, **kwargs
    )
    if fileName == "energy_plot":
        fig, ax = makePlot(
            kwargs["macro_data"],
            ax=ax,
            fig=fig,
            Y="avg_energy",
            save=False,
            legend=True,
        )

    elif fileName == "e_drop_plot":
        nrSeeds = 40
        configs, labels = propperJob(3, nrSeeds, group_by_seeds=True)
        paths, labels = get_csv_files(configs, labels=labels)
        makeLogPlotComparison(
            [paths[0]],  # We choose only the LBFGS
            innerStrainLims=(1, np.inf),
            outerStrainLims=(0.31, 1),
            plot_post_yield=False,
            save=False,
            use_y_axis_name=True,
            Y="avg_energy",
            ax=ax,
            fig=fig,
            labels=labels,
            legend_loc="lower left",
            show=False,
            add_fit=False,
            **kwargs,
        )
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
    fig=None,
    fileName=None,
    macro_data=None,
    macroDataRowIndex=None,
    avgEnergy=None,
    avgRSS=None,
    delAvgEnergy=None,
    delAvgRSS=None,
    delLoad=None,
    **kwargs,
):
    remove_vlines(ax)
    data = VTUData(vtu_file)

    if fileName == "energy_plot":
        x = data.load
    elif fileName == "e_drop_plot":
        x = -delAvgEnergy

    ax.axvline(
        x=x,
        color="red",
        linewidth=0.5,
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


def plot_in_poincare_disk(
    vtu_file, ax=None, fig=None, do_elastic_reduction=False, **kwargs
):
    if ax is None:
        ax, fig = base_plot(vtu_file=vtu_file, **kwargs)
    data = VTUData(vtu_file)

    C = data.get_C()
    # if do_elastic_reduction:
    #     # Do the elastic reduction
    #     C = arrsToMat(*elastic_reduction(C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]))
    #     zoom = 3
    # else:
    zoom = 1

    g = get_energy_grid(zoom=zoom)
    plotEnergyField(
        g, fig, ax, save=False, add_title=False, zoom=zoom, remove_max_color=zoom == 1
    )

    vmax = 2000 if do_elastic_reduction else 1700
    drawCScatter(ax, C, len(g), vmax=vmax, zoom=zoom, remove_max_color=False)

    return ax


def plot_and_save_in_poincare_disk(**kwargs):
    return plot_and_save(
        plot_func=plot_in_poincare_disk,
        **kwargs,
    )


def plot_and_save_in_e_reduced_poincare_disk(**kwargs):
    return plot_and_save(
        plot_func=plot_in_poincare_disk,
        do_elastic_reduction=True,
        **kwargs,
    )


def plot_and_save_mesh(**kwargs):
    return plot_and_save(
        plot_func=plot_mesh,
        mesh_property="energy",
        **kwargs,
    )


def plot_and_save_mesh_with_force(**kwargs):
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
            process_frame(kwargs, attemps=attemps + 1)


def get_corresponding_energy_and_rss(vtu_files, macro_data, X="load"):
    """
    Extracts the corresponding "avg_energy" and "avg_RSS" values for each load in vtu_files,
    along with the line numbers (indices) of the matching rows in the CSV file.

    Parameters:
        vtu_files (List[str]): List of VTU file names.
        macro_data (str): Path to the CSV file containing macro data.

    Returns:
        Tuple[List[float], List[float], List[int]]: Lists of average energy, RSS values,
        and line numbers of matching rows.
    """
    df = pd.read_csv(
        macro_data,
        usecols=[
            X,
            "avg_energy",
            "avg_RSS",
            "avg_energy_change",
            "nr_plastic_deformations",
        ],
    )
    avg_energy_list = []
    change_avg_energy_list = []
    avg_RSS_list = []
    line_numbers = []
    x_list = []

    for vtu_file in vtu_files:
        x = get_data_from_name(vtu_file)[X]
        x_list.append(x)
        # Filter rows where "load" matches the value
        matching_rows = df[abs(df[X] - x) < 1e-10]

        # Check if there is exactly one matching row
        if len(matching_rows) != 1:
            raise ValueError(
                f"'{X}' value '{x}' is not unique or not found. Found {len(matching_rows)} matches."
            )

        # Get the index (line number) of the matching row
        matching_row_index = matching_rows.index[0]
        line_numbers.append(matching_row_index)

        # Extract the matching row as a Series
        matching_row = matching_rows.iloc[0]

        # Append the extracted values to the respective lists
        avg_energy_list.append(matching_row["avg_energy"])
        avg_RSS_list.append(matching_row["avg_RSS"])
        change_avg_energy_list.append(matching_row["avg_energy_change"])
        if (
            matching_row["avg_energy_change"] < 0
            and matching_row["nr_plastic_deformations"] == 0
        ):
            # print(f"No deformation energy drop: {matching_row_index}, load={load}")
            # This can happen in the beginning in simulations
            pass

        if (
            matching_row["avg_energy_change"] < 0
            and -matching_row["avg_energy_change"] < 5e-8
        ):
            print(f"Super small energy drop: {matching_row_index}, {X}={x}")

    # Find previous data and get change data as well
    px, pAvgEnergy, pAvgRSS = get_previous_energy_and_rss(macro_data, line_numbers, X)
    change_avg_RSS_list = avg_RSS_list - pAvgRSS
    del_x = x_list - px

    # Return the lists of values and line numbers
    return (
        avg_energy_list,
        avg_RSS_list,
        change_avg_energy_list,
        change_avg_RSS_list,
        del_x,
        line_numbers,
    )


def get_previous_energy_and_rss(macro_data, current_line, X="load"):
    # Check if current_line is an integer
    if isinstance(current_line, int):
        df = pd.read_csv(macro_data, usecols=[X, "avg_energy", "avg_RSS"])
        # Select the previous row relative to current_line
        p_row = df.iloc[current_line - 1]
        return p_row[X], p_row["avg_energy"], p_row["avg_RSS"]
    else:
        # Handle the case where current_line is an iterable (e.g., list or array)
        df = pd.read_csv(macro_data, usecols=[X, "avg_energy", "avg_RSS"])
        # Create empty lists to store previous values
        prev_x, prev_energies, prev_rss = [], [], []

        for line in current_line:
            # Ensure line index is valid (i.e., not the first row)
            line = max(1, line)
            p_row = df.iloc[line - 1]
            prev_x.append(p_row[X])
            prev_energies.append(p_row["avg_energy"])
            prev_rss.append(p_row["avg_RSS"])

        # Return lists of previous values
        return np.array(prev_x), np.array(prev_energies), np.array(prev_rss)


def make_images(vtu_files, num_processes=10, use_tqdm=True, X="load", **kwargs):
    print(f"Processing {kwargs['fileName']} video.")
    # Calculate global axis limits and energy range
    macro_data = kwargs["macro_data"]
    if macro_data:
        axis_limits = get_axis_limits(macro_data)
        e_lims = get_energy_range(vtu_files, macro_data)
        e_lims[1] = min(e_lims[1], 0.2)
        avgEnergy, avgRSS, delAvgEnergy, delAvgRSS, delx, macroDataRowIndex = (
            get_corresponding_energy_and_rss(vtu_files, macro_data, X)
        )
    else:
        # set default values
        axis_limits = None
        e_lims = [0, 0.03]
        avgEnergy = [0] * len(vtu_files)
        avgRSS = [0] * len(vtu_files)
        delAvgEnergy = [0] * len(vtu_files)
        delAvgRSS = [0] * len(vtu_files)
        delx = [0] * len(vtu_files)
        macroDataRowIndex = [0] * len(vtu_files)
        # make default macro data
        macro_data = {X: 0, "loadIncrement": 0, "nrM": 0}
        kwargs["macro_data"] = macro_data

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
            "frame_index": i,
            "e_lims": e_lims,
            "axis_limits": axis_limits,
            "avgEnergy": avgEnergy[i],
            "avgRSS": avgRSS[i],
            "delAvgEnergy": delAvgEnergy[i],
            "delAvgRSS": delAvgRSS[i],
            "delx": delx[i],
            "macroDataRowIndex": macroDataRowIndex[i],
            **kwargs,
        }
        for i in range(len(vtu_files))
    ]

    # Use line below to debug with first item in kwargs_list
    image_paths = process_frame(kwargs_list[0])

    if multithread:
        with Pool(processes=num_processes) as pool:
            image_paths = list(
                tqdm(
                    pool.imap(process_frame, kwargs_list),
                    total=len(vtu_files),
                    disable=not use_tqdm,
                )
            )
    else:
        image_paths = [
            process_frame(kwargs) for kwargs in tqdm(kwargs_list, disable=not use_tqdm)
        ]

    return image_paths

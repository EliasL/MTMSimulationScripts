import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from tqdm import tqdm
import time
import random
from multiprocessing import Pool


import threading
from MTMath.plotEnergy import (
    plotEnergyField,
    generate_energy_grid,
    drawCScatter,
    lagrange_reduction,
    elastic_reduction,
)
from MTMath.contiPotential import ground_state_energy

from .dataFunctions import get_data_from_name, VTUData, arrsToMat

# matplotlib.use("Agg")  # Use a non-interactive backend


def get_energy_range(vtu_files, cvs_file):
    df = pd.read_csv(cvs_file, usecols=["Max energy"])
    max_energy = df["Max energy"].max()
    # Sometimes, the energy is too high because of a crash
    if max_energy > 100:
        # Then we take the second last
        max_energy = df["Max energy"][:-1].max()
    # We assume that the minimum energy throughout the whole run is the minimum
    # of the initial state
    energy_field = VTUData(vtu_files[0]).get_energy_field()
    min_energy = energy_field.min()
    return min_energy, max_energy


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
    previous_vtu_file=None,
    axis_limits=None,
    add_title=True,
    frame_index=None,
    AvgEnergy=None,
    AvgRSS=None,
):
    dpi = 250
    width = 2000
    height = 1000
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_aspect("equal")

    # Setting the axis limits
    if axis_limits:
        x_min, x_max, y_min, y_max = add_padding(axis_limits, 0.03)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if add_title:
        metaData = get_data_from_name(vtu_file)
        load = float(metaData["load"])
        load_step = float(metaData["loadIncrement"])
        if previous_vtu_file:
            previous_load = float(get_data_from_name(previous_vtu_file)["load"])
            steps_since_last_frame = int((load - previous_load) / load_step)
        else:
            steps_since_last_frame = 0
        nrPlasticEvents = metaData["nrM"]

        # Data for the table
        data = [
            [
                rf"$\gamma$: {load:.5f}",
                rf"$\langle E \rangle$: {AvgEnergy:.3f}",
                rf"$\langle \sigma \rangle$: {AvgRSS:.3f}",
                rf"$p$: {nrPlasticEvents}",
                # rf"$\Delta_k$: {steps_since_last_frame}",
                f"f: {frame_index}",
            ],
        ]
        # Create the table with invisible borders and gridlines
        table = ax.table(cellText=data, cellLoc="center", loc="top", edges="open")

        # Customize table appearance
        table.scale(1, 1.5)  # Adjust table scale (e.g., to fit text size)
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        # Remove the cell edges to keep it invisible
        for cell in table.get_celld().values():
            cell.set_linewidth(0)

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


# Function to save the figure with transparent background and close it
def save_and_close_plot(ax, path, transparent=False):
    # Set anti-aliasing for lines, text, patches, etc.
    for line in ax.get_lines():
        line.set_antialiased(True)  # Anti-alias lines

    for text in ax.texts:
        text.set_fontproperties(
            text.get_fontproperties()
        )  # Ensure text rendering uses proper anti-aliasing

    # You can also anti-alias patches or other graphical elements
    for patch in ax.patches:
        patch.set_antialiased(True)  # Anti-alias shapes like rectangles, circles

    # Save the plot with transparent background
    dpi = 600
    plt.tight_layout()
    plt.savefig(
        path, bbox_inches="tight", pad_inches=0, transparent=transparent, dpi=dpi
    )
    plt.close()


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


def plot_mesh(
    vtu_file,
    global_min=0,
    global_max=0.37,
    mesh_property="energy",
    ax=None,
    shift=True,
    add_rombus=True,
    **kwargs,
):
    if ax is None:
        ax, fig = base_plot(vtu_file=vtu_file, **kwargs)
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    cmap = "coolwarm"

    hatch = None
    norm = None

    if mesh_property == "energy":
        # The energy field is normalized to have energy=0 in the ground state
        field = data.get_energy_field() - ground_state_energy()
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        backgroundColor = plt.get_cmap(cmap)(0)
    elif mesh_property == "stress":
        field = data.get_stress_field()
        norm = mcolors.Normalize(vmin=-1.5, vmax=1.5)
        backgroundColor = plt.get_cmap(cmap)(0.5)
    elif mesh_property == "m":
        cmap = "viridis"
        nrm1, nrm2, nrm3 = data.get_m_nr_field()
        field = nrm3
        backgroundColor = plt.get_cmap(cmap)(0)
        hatch_patterns = {
            0: "",
            1: "\\",  # Backslash for nrm1 % 2 == 1
            2: "/",  # Forward slash for nrm2 % 2 == 1
            3: "x",  # 'x' for both conditions being true
        }

        # Assuming nrm1 and nrm2 are defined
        # Calculate the hatch value
        hatch_value = (nrm1 % 2) + (nrm2 % 2) * 2

        # Retrieve the hatch pattern string
        hatch = hatch_patterns.get(hatch_value, "")

    if shift:
        shifts = calculate_shifts(nodes, data.BC, data.load)
    else:
        shifts = [0]
    for dx in shifts:
        for dy in shifts:
            sheared_x = x + dx + data.load * dy
            triang = mtri.Triangulation(sheared_x, y + dy, connectivity)
            ax.triplot(triang, color=backgroundColor, lw=0.1)
            ax.tripcolor(
                triang,
                facecolors=field,
                norm=norm,
                cmap=cmap,
                edgecolors="none",
                hatch=hatch,
            )

    if add_rombus:
        draw_rhombus(ax, np.sqrt(len(nodes[:, 0])) - 1, data.load, data.BC)
    return ax, cmap, norm


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
    if do_elastic_reduction:
        # Do the elastic reduction
        C = arrsToMat(*elastic_reduction(C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]))
        zoom = 3
    else:
        zoom = 1

    g = get_energy_grid(zoom=zoom)
    plotEnergyField(
        g, fig, ax, save=False, add_title=False, zoom=zoom, remove_max_color=zoom == 1
    )

    vmax = 2000 if do_elastic_reduction else 1700
    drawCScatter(ax, C, len(g), vmax=vmax, zoom=zoom, remove_max_color=False)

    return ax


def plot_and_save_in_poincare_disk(frame_path, frame_index, transparent, **kwargs):
    # Remove unwanted keywords
    for key in ["global_min", "global_max", "axis_limits"]:
        kwargs.pop(key)
    ax = plot_in_poincare_disk(frame_index=frame_index, **kwargs)

    path = f"{frame_path}/disk_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)

    return path


def plot_and_save_in_e_reduced_poincare_disk(
    frame_path, frame_index, transparent, **kwargs
):
    # Remove unwanted keywords
    for key in ["global_min", "global_max", "axis_limits"]:
        kwargs.pop(key)
    ax = plot_in_poincare_disk(
        frame_index=frame_index, do_elastic_reduction=True, **kwargs
    )

    path = f"{frame_path}/eReduced_disk_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)

    return path


def plot_and_save_mesh(frame_path, frame_index, transparent, **kwargs):
    ax, _, _ = plot_mesh(frame_index=frame_index, add_title=True, **kwargs)

    path = f"{frame_path}/mesh_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)
    return path


def plot_and_save_m_mesh(frame_path, frame_index, transparent, **kwargs):
    ax, _, _ = plot_mesh(
        frame_index=frame_index, add_title=True, mesh_property="m", **kwargs
    )

    path = f"{frame_path}/mesh_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)
    return path


def plot_and_save_nodes(frame_path, frame_index, transparent, **kwargs):
    # Remove unwanted keywords
    for key in ["global_min", "global_max"]:
        kwargs.pop(key)
    ax = plot_nodes(frame_index=frame_index, add_title=True, **kwargs)

    path = f"{frame_path}/node_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)
    return path


def process_frame(kwargs):
    kwargs = kwargs.copy()
    # Unpack frameFunction from kwargs and apply retry logic
    frameFunction = kwargs.pop("frameFunction")
    # Call frameFunction with remaining keyword arguments
    return frameFunction(**kwargs)


def getCorespondingEnergyAndRSS(vtu_files, macro_data):
    df = pd.read_csv(macro_data, usecols=["Load", "Avg energy", "Avg RSS"])
    AvgEnergy = []
    AvgRSS = []
    for vtu_file in vtu_files:
        load = float(get_data_from_name(vtu_file)["load"])

        # Assuming "Load" is a unique identifier, we can filter directly
        matching_row = df[df["Load"] == load]

        # Check if we found a match
        if not matching_row.empty:
            # Extract the corresponding "Avg energy" and "Avg RSS" values
            avg_energy = matching_row["Avg energy"].values[0]
            avg_rss = matching_row["Avg RSS"].values[0]

            # Append the extracted values to the respective lists
            AvgEnergy.append(avg_energy)
            AvgRSS.append(avg_rss)
        else:
            raise (EOFError("We would expect to have an exact match for the load!"))

    # Return the lists or use them as needed
    return AvgEnergy, AvgRSS


def make_images(vtu_files, macro_data, num_processes=10, use_tqdm=True, **kwargs):
    # Calculate global axis limits and energy range
    axis_limits = get_axis_limits(macro_data)
    global_min, global_max = get_energy_range(vtu_files, macro_data)
    AvgEnergy, AvgRSS = getCorespondingEnergyAndRSS(vtu_files, macro_data)
    # Create a list of dictionaries for keyword arguments
    kwargs_list = [
        {
            "vtu_file": vtu_files[i],
            "previous_vtu_file": vtu_files[i - 1] if i != 0 else None,
            "frame_index": i,
            "global_min": global_min,
            "global_max": global_max,
            "axis_limits": axis_limits,
            "AvgEnergy": AvgEnergy[i],
            "AvgRSS": AvgRSS[i],
            **kwargs,
        }
        for i in range(len(vtu_files))
    ]

    # Use line below to debug with first item in kwargs_list
    image_paths = process_frame(kwargs_list[0])

    with Pool(processes=num_processes) as pool:
        image_paths = list(
            tqdm(
                pool.imap(process_frame, kwargs_list),
                total=len(vtu_files),
                disable=not use_tqdm,
            )
        )

    return image_paths

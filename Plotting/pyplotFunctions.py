import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
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

from .dataFunctions import get_data_from_name
import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend


class VTUData:
    def __init__(self, vtu_file_path):
        self.vtu_file_path = vtu_file_path
        self.mesh = self._read_vtu_file()
        result = get_data_from_name(vtu_file_path)
        self.BC = result["BC"]
        self.load = float(result["load"])

    def _read_vtu_file(self):
        # Create a reader for the VTU file
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.vtu_file_path)
        reader.Update()

        # Get the 'vtkUnstructuredGrid' object from the reader
        return reader.GetOutput()

    def get_nodes(self):
        return vtk_to_numpy(self.mesh.GetPoints().GetData())

    def get_force_field(self):
        # NB this is "force". Check the C++ code, might not be what you think
        return vtk_to_numpy(self.mesh.GetPointData().GetArray("stress_field"))

    def get_stress_field(self):
        return vtk_to_numpy(self.mesh.GetCellData().GetArray("P12"))

    def get_energy_field(self):
        return vtk_to_numpy(self.mesh.GetCellData().GetArray("energy_field"))

    def get_fixed_status(self):
        return vtk_to_numpy(self.mesh.GetPointData().GetArray("fixed"))

    def get_C(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [C11, C22, C12] components.
        """
        # Get the C11, C22, and C12 arrays from the VTK object
        C11, C22, C12 = [
            vtk_to_numpy(self.mesh.GetCellData().GetArray(C))
            for C in ["C11", "C22", "C12"]
        ]
        return arrsToMat(C11, C22, C12)

    def get_connectivity(self):
        # Extract Connectivity
        _connectivity = vtk_to_numpy(self.mesh.GetCells().GetData())
        # _connectivity is in a special format: 3 a b c 3 d e f 3 ...
        # We reshape into 4 long arrays, and then drop the column of 3s
        connectivity = _connectivity.reshape(-1, 4)[:, 1:]
        return connectivity


def arrsToMat(C11, C22, C12):
    # Initialize the 3D array to store the 2x2 matrices
    C = np.zeros((C11.shape[0], 2, 2))

    # Fill the matrix with the corresponding values
    C[:, 0, 0] = C11  # (1,1) entry
    C[:, 1, 1] = C22  # (2,2) entry
    C[:, 0, 1] = C12  # (1,2) entry
    C[:, 1, 0] = C12  # (2,1) entry, ensuring symmetry

    return C


def get_energy_range(vtu_files, cvs_file):
    df = pd.read_csv(cvs_file, usecols=["Max energy"])
    max_energy = df["Max energy"].max()
    if max_energy > 100:
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


def base_plot(vtu_file=None, axis_limits=None, add_title=True, frame_index=None):
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
        metadata = get_data_from_name(vtu_file)
        lines = [
            f"{metadata['name']}",
            f"Frame: {frame_index}, " + f"Load: {float(metadata['load']):.3f}",
        ]
        ax.set_title("\n".join(lines), fontsize=8)
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


def calculate_valid_indices(n, m):
    # Create a 2D grid of indices
    indices = np.arange(n * m).reshape(n, m)
    valid_indices = indices[: n - 1, : m - 1].flatten()
    return valid_indices


def trim_connections(nr_nodes, connections):
    # Connections is an Nx3 array where the three numbers indicate
    # the indexes of the nodes that should form elements
    # This function removes all the elements that connect nodes that
    # have an index larger than nr_nodes

    mask = np.all(connections < nr_nodes, axis=1)

    # Use the mask to filter the connections
    trimmed_connections = connections[mask]

    return trimmed_connections


def plot_nodes(args):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args
    ax, fig = base_plot(
        vtu_file=vtu_file, axis_limits=axis_limits, frame_index=frame_index
    )
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    fixed = data.get_fixed_status()
    dims = get_data_from_name(vtu_file)["dims"]
    n, m = dims
    if data.BC == "PBC":
        # When using PBC, we want to hide the last row and column
        # Calculate valid indices (excluding last row and column) using the NumPy function
        valid_indices = calculate_valid_indices(n, m)

        # Filter nodes and fixed status using the computed indices
        nodes = nodes[valid_indices]
        fixed = fixed[valid_indices]
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
    circle_diameter = (
        0.3 * grid_size
    )  # Adjust the 0.25 factor as necessary to prevent overlap
    circle_radius = circle_diameter / 2
    circle_point_size = (circle_radius * inches_per_data_unit) ** 2

    # Grid
    connectivity = data.get_connectivity()
    connectivity = trim_connections(len(x), connectivity)
    shifts = calculate_shifts(nodes, data.BC, data.load)
    for dx in shifts:
        for dy in shifts:
            sheared_x = x + dx + data.load * dy
            # Mesh lines between nodes
            triang = mtri.Triangulation(sheared_x, y + dy, connectivity)
            ax.triplot(triang, color="black", linewidth=0.5, alpha=0.3)

            ax.scatter(
                sheared_x,
                y + dy,
                s=circle_point_size,
                c=color,
                marker="o",
                alpha=1,
                edgecolors="none",
            )

    draw_rhombus(ax, np.sqrt(len(nodes[:, 0])) - 1, data.load, data.BC)
    path = f"{framePath}/node_frame_{frame_index:04d}.png"
    save_and_close_plot(fig, ax, path, transparent)
    return path


def plot_mesh(
    vtu_file,
    global_min=0,
    global_max=0.37,
    useStress=True,
    axis_limits=None,
    frame_index=None,
    add_title=False,
    ax=None,
    shift=True,
    add_rombus=True,
):
    if ax is None:
        ax, fig = base_plot(
            vtu_file=vtu_file,
            axis_limits=axis_limits,
            frame_index=frame_index,
            add_title=add_title,
        )
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    cmap = "coolwarm"

    if useStress:
        field = data.get_stress_field()
        norm = mcolors.Normalize(vmin=-1.5, vmax=1.5)
        backgroundColor = plt.get_cmap(cmap)(0.5)
    else:
        # The energy field is normalized to have energy=0 in the ground state
        field = data.get_energy_field() - ground_state_energy()
        norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        backgroundColor = plt.get_cmap(cmap)(0)

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
                triang, facecolors=field, norm=norm, cmap=cmap, edgecolors="none"
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
    vtu_file,
    frame_index=None,
    add_title=False,
    ax=None,
    fig=None,
    do_elastic_reduction=False,
):
    if ax is None:
        ax, fig = base_plot(
            vtu_file=vtu_file,
            frame_index=frame_index,
            add_title=add_title,
        )
    data = VTUData(vtu_file)
    C = data.get_C()
    if do_elastic_reduction:
        # Do the elastic reduction
        C = arrsToMat(*elastic_reduction(C[:, 0, 0], C[:, 1, 1], C[:, 0, 1]))
        zoom = 3
    else:
        zoom = 1

    g = get_energy_grid(zoom=zoom)
    plotEnergyField(g, fig, ax, save=False, zoom=zoom, remove_max_color=zoom == 1)

    vmax = 2000 if do_elastic_reduction else 1700
    drawCScatter(ax, C, len(g), vmax=vmax, zoom=zoom, remove_max_color=False)

    ax.set_title("Elements in Poincaré disk with point density")
    return ax


def plot_and_save_in_poincare_disk(args):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args

    ax = plot_in_poincare_disk(
        vtu_file=vtu_file,
        frame_index=frame_index,
        add_title=True,
    )

    path = f"{framePath}/disk_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)

    return path


def plot_and_save_in_elastically_reduced_poincare_disk(args):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args

    ax = plot_in_poincare_disk(
        vtu_file=vtu_file,
        frame_index=frame_index,
        add_title=True,
        do_elastic_reduction=True,
    )

    path = f"{framePath}/eReduced_disk_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)

    return path


def plot_and_save_mesh(args, useStress=True):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args

    ax, _, _ = plot_mesh(
        vtu_file=vtu_file,
        global_min=global_min,
        global_max=global_max,
        useStress=useStress,
        axis_limits=axis_limits,
        frame_index=frame_index,
        add_title=True,
    )

    path = f"{framePath}/mesh_frame_{frame_index:04d}.png"
    save_and_close_plot(ax, path, transparent)
    return path


def retry_frame_function(frameFunction, args, max_retries=3):
    # Maybe not needed any more?
    return frameFunction(args)
    for attempt in range(max_retries):
        try:
            return frameFunction(args)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error for file {args[1]}: {str(e)}")
            time.sleep(random.uniform(0.1, 1))  # Random delay between 0.1 and 1 second
    print(f"Failed to process file {args[1]} after {max_retries} attempts.")
    return None


def process_frame(args):
    # Unpack the frameFunction from args and apply retry logic
    frameFunction, other_args = args[0], args[1:]
    return retry_frame_function(frameFunction, other_args)


def make_images(
    frameFunction, framePath, vtu_files, macro_data, transparent=False, num_processes=10
):
    # Assuming vtu_files is defined, calculate global axis limits
    axis_limits = get_axis_limits(macro_data)
    # global_min, global_max = precalculate_global_stress range(vtu_files)
    global_min, global_max = get_energy_range(vtu_files, macro_data)
    args_list = [
        (
            frameFunction,
            framePath,
            vtu_file,
            frame_index,
            global_min,
            global_max,
            axis_limits,
            transparent,
        )
        for frame_index, vtu_file in enumerate(vtu_files)
    ]
    # Use line below to debug
    image_paths = process_frame(args_list[0])

    with Pool(processes=num_processes) as pool:
        image_paths = list(
            tqdm(pool.imap(process_frame, args_list), total=len(vtu_files))
        )

    return image_paths

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

from dataFunctions import get_data_from_name
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend


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

    def get_stress_field(self):
        return vtk_to_numpy(self.mesh.GetPointData().GetArray("stress_field"))

    def get_energy_field(self):
        return vtk_to_numpy(self.mesh.GetCellData().GetArray("energy_field"))

    def get_fixed_status(self):
        return vtk_to_numpy(self.mesh.GetPointData().GetArray("fixed"))

    def get_connectivity(self):
        # Extract Connectivity
        _connectivity = vtk_to_numpy(self.mesh.GetCells().GetData())
        # _connectivity is in a special format: 3 a b c 3 d e f 3 ...
        # We reshape into 4 long arrays, and then drop the column of 3s
        connectivity = _connectivity.reshape(-1, 4)[:, 1:]
        return connectivity


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


def base_plot(args):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args

    dpi = 250
    width = 2000
    height = 1000
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_aspect("equal")

    # Setting the axis limits
    x_min, x_max, y_min, y_max = add_padding(axis_limits, 0.03)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    metadata = get_data_from_name(vtu_file)
    lines = [
        f"State: {metadata['name']}",
        f"Frame: {frame_index}, " + f"Load: {float(metadata['load']):.3f}",
    ]
    ax.set_title("\n".join(lines))
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
def save_and_close_plot(fig, ax, path, transparent=False):
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
    dpi = 300
    plt.savefig(
        path, bbox_inches="tight", pad_inches=0, transparent=transparent, dpi=dpi
    )

    # Close the figure to free memory
    plt.close(fig)
    plt.close()


def calculate_valid_indices(n, m):
    # Create a 2D grid of indices
    indices = np.arange(n * m).reshape(n, m)
    valid_indices = indices[: n - 1, : m - 1].flatten()
    return valid_indices


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
    ax, fig = base_plot(args)
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
    energy_field = data.get_energy_field()
    connectivity = data.get_connectivity()

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


def plot_mesh(args):
    (
        framePath,
        vtu_file,
        frame_index,
        global_min,
        global_max,
        axis_limits,
        transparent,
    ) = args
    ax, fig = base_plot(args)

    cmap_colors = [
        (0.0, (0.29, 0.074, 0.38)),
        (0.07, "#0052cc"),
        (0.3, "#ff6f61"),
        (0.9, "orange"),
        (1.0, "red"),
    ]

    # Create a color map from the list of colors and positions
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", cmap_colors, N=512
    )

    # Define a normalization that highlights small energy changes
    gamma = 1  # Adjust this parameter as needed to highlight small energy changes
    norm = mcolors.PowerNorm(gamma=gamma, vmin=global_min, vmax=global_max)

    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    energy_field = data.get_energy_field()
    connectivity = data.get_connectivity()
    x, y = nodes[:, 0], nodes[:, 1]

    shifts = calculate_shifts(nodes, data.BC, data.load)
    for dx in shifts:
        for dy in shifts:
            sheared_x = x + dx + data.load * dy
            triang = mtri.Triangulation(sheared_x, y + dy, connectivity)
            ax.tripcolor(triang, facecolors=energy_field, norm=norm, cmap=custom_cmap)

    draw_rhombus(ax, np.sqrt(len(nodes[:, 0])) - 1, data.load, data.BC)
    path = f"{framePath}/mesh_frame_{frame_index:04d}.png"
    save_and_close_plot(fig, ax, path, transparent)
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

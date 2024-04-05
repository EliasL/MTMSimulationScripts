import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from multiprocessing import Pool

from dataFunctions import getDataFromName

class VTUData:
    def __init__(self, vtu_file_path):
        self.vtu_file_path = vtu_file_path
        self.mesh = self._read_vtu_file()

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

def precalculate_global_stress_range(vtu_files):
    global_min, global_max = np.inf, -np.inf
    for vtu_file in vtu_files:
        energy_field = VTUData(vtu_file).get_energy_field()
        min_energy, max_energy = energy_field.min(), energy_field.max()
        global_min, global_max = min(global_min, min_energy), max(global_max, max_energy)
    return global_min, global_max

def get_energy_and_stress_range(vtu_file):
    energy_field = VTUData(vtu_file).get_energy_field()
    min_energy, max_energy = energy_field.min(), energy_field.max()
    return min_energy,max_energy

def precalculate_global_stress_range_parallel(vtu_files):
    global_min, global_max = np.inf, -np.inf
    with ProcessPoolExecutor() as executor:
        results = executor.map(get_energy_and_stress_range, vtu_files)
        
        for min_val, max_val in results:
            global_min, global_max = min(global_min, min_val), max(global_max, max_val)

    return global_min, global_max

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
        raise(Exception("Invalid Mesh"))
    node_energy /= node_count

    return node_energy

# Use this function to set axis limits in your plot_frame function
def get_axis_limits(vtu_files):
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    for vtu_file in [vtu_files[0],vtu_files[-1]]: #Remove [[-1]] to search through everything if desired
        nodes = VTUData(vtu_file).get_nodes()
        x_min = min(x_min, nodes[:, 0].min())
        x_max = max(x_max, nodes[:, 0].max())
        y_min = min(y_min, nodes[:, 1].min())
        y_max = max(y_max, nodes[:, 1].max())
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
    framePath, vtu_file, frame_index, global_min, global_max, axis_limits = args
    
    dpi = 200
    width = 2000
    height = 1000
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_aspect('equal')

    # Setting the axis limits
    x_min, x_max, y_min, y_max = add_padding(axis_limits, 0.03)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    metadata = getDataFromName(vtu_file)
    lines = [
        f"State: {metadata['name']}",
        f"Frame: {frame_index}, " +
        f"Load: {float(metadata['load']):.3f}",
    ]
    ax.set_title("\n".join(lines))
    ax.set_xticks([])
    ax.set_yticks([])

    return ax, fig


def plot_nodes(args):
    framePath, vtu_file, frame_index, global_min, global_max, axis_limits = args

    ax, fig = base_plot(args)
    
    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    fixed = data.get_fixed_status()
    color = np.where(fixed==1, 'red', 'blue')
    x, y = nodes[:,0], nodes[:,1]
    
    ax.scatter(x, y, s=20, c=color, marker='o', alpha=1)  # 's' is the size, 'c' is the color
 
    path = f"{framePath}/node_frame_{frame_index:04d}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return path

def plot_mesh(args):
    framePath, vtu_file, frame_index, global_min, global_max, axis_limits = args
    
    ax, fig = base_plot(args)

    data = VTUData(vtu_file)
    nodes = data.get_nodes()
    energy_field = data.get_energy_field()
    connectivity = data.get_connectivity()

    x, y = nodes[:,0], nodes[:,1]
    
    triang = mtri.Triangulation(x, y, connectivity)
    cmap_colors = [
                (0.0, (0.29, 0.074, 0.38)),
                (0.07, "#0052cc"),
                (0.3, "#ff6f61"),
                (0.9, "orange"),
                (1.0, "red")]

    # Create a color map from the list of colors and positions
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", cmap_colors,N=512)

    # Define a normalization that highlights small energy changes
    gamma = 1  # Adjust this parameter as needed to highlight small energy changes
    norm = mcolors.PowerNorm(gamma=gamma, vmin=global_min, vmax=global_max)

    # Apply the custom colormap and normalization to the tripcolor plot
    contour = ax.tripcolor(triang, facecolors=energy_field, norm=norm, cmap=custom_cmap)

    # Create a color bar
    cbar = fig.colorbar(contour, shrink=0.7)
    cbar.set_label('Cell Energy')
    cbar.ax.tick_params(labelsize=8)

    # wire mesh
    #ax.triplot(triang, 'w-', linewidth=0.2, alpha=0.3)
    
    path = f"{framePath}/mesh_frame_{frame_index:04d}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return path

def makeImages(frameFunction, framePath, vtu_files, num_processes=10):
    # Assuming vtu_files is defined, calculate global axis limits
    axis_limits = get_axis_limits(vtu_files)
    # global_min, global_max = precalculate_global_stress_range(vtu_files)
    global_min, global_max = precalculate_global_stress_range_parallel(vtu_files)
    args_list = [(framePath, vtu_file, frame_index, global_min, global_max, axis_limits) for frame_index, vtu_file in enumerate(vtu_files)]
    
    with Pool(processes=num_processes) as pool:
        image_paths = list(tqdm(pool.imap(frameFunction, args_list), total=len(vtu_files)))
    
    return image_paths
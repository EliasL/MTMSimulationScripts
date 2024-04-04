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

def read_vtu_data(vtu_file_path):
    # Create a reader for the VTU file
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()

    # Get the 'vtkUnstructuredGrid' object from the reader
    mesh = reader.GetOutput()

    # Extract Nodes
    nodes = vtk_to_numpy(mesh.GetPoints().GetData())

    # Extract Stress Field
    stress_field = vtk_to_numpy(mesh.GetPointData().GetArray("stress_field"))

    # Extract Energy Field
    energy_field = vtk_to_numpy(mesh.GetCellData().GetArray("energy_field"))

    # Extract Connectivity
    _connectivity = vtk_to_numpy(mesh.GetCells().GetData())
    # _connectivity is in a special format: 3 a b c 3 d e f 3 ...
    # We reshape into 4 long arrays, and then drop the column of 3s
    connectivity = _connectivity.reshape(-1, 4)[:, 1:]
    return nodes, stress_field, energy_field, connectivity

def read_vtu_property(vtu_file_path, property_name, pointProperty=False, cellProperty=False):
    # Create a reader for the VTU file
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_file_path)
    reader.Update()

    # Get the 'vtkUnstructuredGrid' object from the reader
    mesh = reader.GetOutput()

    if(property_name=="nodePos"):
        return vtk_to_numpy(mesh.GetPoints().GetData())
    
    if(property_name=="connectivity"):
        # Extract Connectivity
        _connectivity = vtk_to_numpy(mesh.GetCells().GetData())
        # _connectivity is in a special format: 3 a b c 3 d e f 3 ...
        # We reshape into 4 long arrays, and then drop the column of 3s
        return _connectivity.reshape(-1, 4)[:, 1:]

    if pointProperty:
        return vtk_to_numpy(mesh.GetPointData().GetArray(property_name))
    if cellProperty:
        return vtk_to_numpy(mesh.GetCellData().GetArray(property_name))
    raise(Exception("No property type chosen. Set either point- or cell-Property to True"))


def precalculate_global_stress_range(vtu_files):
    global_min, global_max = np.inf, -np.inf
    for vtu_file in vtu_files:
        _, stress_field, energy_field, _ = read_vtu_data(vtu_file)
        min_energy, max_energy = energy_field.min(), energy_field.max()
        global_min, global_max = min(global_min, min_energy), max(global_max, max_energy)
    return global_min, global_max

def get_energy_and_stress_range(vtu_file):
    _, stress_field, energy_field, _ = read_vtu_data(vtu_file)
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
        nodes, _, _, _ = read_vtu_data(vtu_file)
        x_min = min(x_min, nodes[:, 0].min())
        x_max = max(x_max, nodes[:, 0].max())
        y_min = min(y_min, nodes[:, 1].min())
        y_max = max(y_max, nodes[:, 1].max())
    return x_min, x_max, y_min, y_max

def base_plot(args):
    framePath, vtu_file, frame_index, global_min, global_max, axis_limits = args
    
    dpi = 200
    width = 2000
    height = 1000
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_aspect('equal')

    # Setting the axis limits
    x_min, x_max, y_min, y_max = axis_limits
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
    
    nodes, stress_field, energy_field, connectivity = read_vtu_data(vtu_file)


    x, y = nodes[:,0], nodes[:,1]
    
    ax.scatter(x, y, s=20, c='blue', marker='o', alpha=1)  # 's' is the size, 'c' is the color
 

    path = f"{framePath}/frame_{frame_index:04d}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return path

def plot_mesh(args):
    framePath, vtu_file, frame_index, global_min, global_max, axis_limits = args
    
    ax, fig = base_plot(args)

    nodes, stress_field, energy_field, connectivity = read_vtu_data(vtu_file)

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
    
    path = f"{framePath}/frame_{frame_index:04d}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return path

def makeImages(frameFunction, framePath, vtu_files, num_processes=10):
    print("Creating frames...")
    # Assuming vtu_files is defined, calculate global axis limits
    axis_limits = get_axis_limits(vtu_files)
    # global_min, global_max = precalculate_global_stress_range(vtu_files)
    global_min, global_max = precalculate_global_stress_range_parallel(vtu_files)
    args_list = [(framePath, vtu_file, frame_index, global_min, global_max, axis_limits) for frame_index, vtu_file in enumerate(vtu_files)]
    
    with Pool(processes=num_processes) as pool:
        image_paths = list(tqdm(pool.imap(frameFunction, args_list), total=len(vtu_files)))
    
    return image_paths
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy
from tqdm import tqdm

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

def makeImages(framePath, vtu_files):
    for frame_index, vtu_file in enumerate(tqdm(vtu_files)):
        nodes, stress_field, energy_field, connectivity = read_vtu_data(vtu_file)
        magnitude = np.linalg.norm(stress_field, axis=1)
        # Extract x and y coordinates from nodes
        x, y = nodes[:,0], nodes[:,1]

        # Use connectivity for triangles
        triangles = connectivity

        # Create triangulation
        triang = mtri.Triangulation(x, y, triangles)

        # Set up the figure
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Plot the triangulation.
        ax.tricontourf(triang, magnitude)
        ax.triplot(triang, 'ko-')
        ax.set_title('Triangular grid')

        fig.tight_layout()
        path = f"{framePath}/frame_{frame_index:04d}.png"
        plt.show()

if __name__ == "__main__":
    makeImages(None, '/Volumes/data/MTS2D_output/s10x10l0.15,0.001,1t1s0/data/s10x10l0.15,0.001,1t1s0_load=0.375_.45.vtu')
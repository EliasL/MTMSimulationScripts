import xml.etree.ElementTree as ET
import os
from vtk import vtkXMLUnstructuredGridReader
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
import numpy as np


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

    def get_m_nr_field(self):
        nrm1 = vtk_to_numpy(self.mesh.getCellData().GetArray("nrm1"))
        nrm2 = vtk_to_numpy(self.mesh.getCellData().GetArray("nrm2"))
        nrm3 = vtk_to_numpy(self.mesh.getCellData().GetArray("nrm3"))
        return nrm1, nrm2, nrm3

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


def parse_pvd_file(path, pvd_file):
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter("DataSet"):
        vtu_files.append(os.path.join(path, dataset.attrib["file"]))

    return vtu_files


def get_data_from_name(nameOrPath):
    # Split the filename by underscores
    if not isinstance(nameOrPath, str):
        nameOrPath = str(nameOrPath)
    fileName = nameOrPath.split("/")[-1]
    parts = fileName.split("_")

    # Initialize an empty dictionary
    result = {}

    # We skipp the first and last part. The first part is the 'name', the last
    # part is the type, ie .N.vtu:
    # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0_load=3.79001_nrM=0_.364001.vtu

    result["name"] = parts[0]
    for part in parts[1:-1]:
        key, value = part.split("=")
        # Add the key-value pair to the dictionary
        result[key] = value

    # We can now extract some extra stuff from the name
    # It will for example have the form:
    # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0
    result["dims"] = tuple(
        map(int, result["name"].split(",")[1].split("s")[1].split("l")[0].split("x"))
    )

    # Extract start load, load increment, and max load
    load_parts = result["name"].split(",")[1:]
    result["startLoad"] = load_parts[0].split("l")[1]
    result["loadIncrement"] = load_parts[1]
    if "NPBC" in load_parts[2]:
        result["maxLoad"] = load_parts[2].split("NPBC")[0]
        result["BC"] = "NPBC"
    else:
        result["maxLoad"] = load_parts[2].split("PBC")[0]
        result["BC"] = "PBC"

    return result

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
        if "BC" in result:
            self.BC = result["BC"]
        if "load" in result:
            self.load = float(result["load"])

    def _read_vtu_file(self):
        # Create a reader for the VTU file
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.vtu_file_path)
        reader.Update()

        # Get the 'vtkUnstructuredGrid' object from the reader
        return reader.GetOutput()

    def get_point_data(self, field):
        return vtk_to_numpy(self.mesh.GetPointData().GetArray(field))

    def get_cell_data(self, field):
        return vtk_to_numpy(self.mesh.GetCellData().GetArray(field))

    def get_nodes(self):
        return vtk_to_numpy(self.mesh.GetPoints().GetData())

    def get_force_field(self):
        return self.get_point_data("stress_field")

    def get_stress_field(self):
        return self.get_cell_data("P12")

    def get_energy_field(self):
        return self.get_cell_data("energy_field")

    def get_fixed_status(self):
        return self.get_point_data("fixed")

    def get_m_nr_field(self):
        nrm1 = self.get_cell_data("nrm1").astype(int)
        nrm2 = self.get_cell_data("nrm2").astype(int)
        nrm3 = self.get_cell_data("nrm3").astype(int)
        return nrm1, nrm2, nrm3

    def get_m3_nr_field(self):
        return self.get_cell_data("nrm3").astype(int)

    def get_m3_change_field(self):
        return self.get_cell_data("deltaNrm3").astype(int)

    def get_C(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [C11, C22, C12] components.
        """
        # Get the C11, C22, and C12 arrays from the VTK object
        C11, C12, C22 = [self.get_cell_data(C) for C in ["C11", "C12", "C22"]]
        return CArrsToMat(C11, C12, C22)

    def get_P(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [P11,P12, P21, P22] components.
        """
        # Get the C11, C22, and C12 arrays from the VTK object
        P11, P12, P21, P22 = [
            self.get_cell_data(P) for P in ["P11", "P12", "P21", "P22"]
        ]
        return arrsToMat(P11, P12, P21, P22)

    def get_force_contributions(self):
        """Returns three force vectors for each element in the mesh"""
        P = self.get_P()
        assert len(P.shape) == 3, "Should be an array of 2x2 matrices"
        assert P.shape[-2:] == (2, 2), "Should be an array of 2x2 matrices"

        # dN_dX matrix from C++ code
        dN_dX = np.array(
            [
                [-1, -1],
                [1, 0],
                [0, 1],
            ]
        )  # shape: (3, 2)
        # initialArea assumption
        initArea = 0.5

        # Compute force contributions for each element
        # Result: (n_elements, 2, 3)
        # Note we transpose dN_dX
        eGradientDensity = np.einsum("nij,kj->nik", P, dN_dX)
        # Force is the negative of the gradient
        return -eGradientDensity * initArea

    def get_connectivity(self):
        # Extract Connectivity
        _connectivity = vtk_to_numpy(self.mesh.GetCells().GetData())
        # _connectivity is in a special format: 3 a b c 3 d e f 3 ...
        # We reshape into 4 long arrays, and then drop the column of 3s
        connectivity = _connectivity.reshape(-1, 4)[:, 1:]
        return connectivity


def arrsToMat(A11, A12, A21, A22):
    assert A11.shape == A12.shape == A21.shape == A22.shape, (
        "All components should have the same shape"
    )
    A = np.zeros((A11.shape[0], 2, 2))

    # Fill the matrix with the corresponding values
    A[:, 0, 0] = A11  # (1,1) entry
    A[:, 0, 1] = A12  # (1,2) entry
    A[:, 1, 0] = A21  # (2,1) entry
    A[:, 1, 1] = A22  # (2,2) entry
    return A


def CArrsToMat(C11, C12, C22):
    return arrsToMat(C11, C12, C12, C22)


def parse_pvd_file(path, pvd_file):
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter("DataSet"):
        if "_." not in dataset.attrib["file"]:
            print("Skipping file: ", dataset.attrib["file"])
        else:
            vtu_files.append(os.path.join(path, dataset.attrib["file"]))

    return vtu_files


def get_data_from_name(nameOrPath):
    # Split the filename by underscores
    if not isinstance(nameOrPath, str):
        nameOrPath = str(nameOrPath)
    fileName = nameOrPath.split("/")[-1]
    if fileName == "macroData.csv":
        fileName = nameOrPath.split("/")[-2]
    parts = fileName.split("_")

    # Initialize an empty dictionary
    result = {}

    # We skipp the first and last part. The first part is the 'name', the last
    # part is the type, ie .N.vtu:
    # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0_load=3.79001_nrM=0_.364001.vtu

    # We can also handle .csv values, in which case the last part will be .csv
    if parts[0][-4:] == ".csv":
        parts[0] = parts[0][:-4]

    result["name"] = parts[0]
    for part in parts[1:-1]:
        key, value = part.split("=")
        # Add the key-value pair to the dictionary

        # minStep is special. It has the format iterations.func_evals
        if key == "minStep":
            result["nr_iterations"], result["nr_func_evals"] = map(
                int, value.split(".")
            )
            result["minStep"] = value
            continue

        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value

    # We can now extract some extra stuff from the name
    try:
        # It will for example have the form:
        # resettingSimpleShearPeriodicBoundary,s60x60l0.15,1e-05,10PBCt4s0
        result["dims"] = tuple(
            map(
                int, result["name"].split(",")[1].split("s")[1].split("l")[0].split("x")
            )
        )

        # get seed
        result["seed"] = int(result["name"].split("s")[-1])

        # Extract start load, load increment, and max load
        load_parts = result["name"].split(",")[1:]
        result["startLoad"] = float(load_parts[0].split("l")[1])
        result["loadIncrement"] = float(load_parts[1])
        if "NPBC" in load_parts[2]:
            result["maxLoad"] = float(load_parts[2].split("NPBC")[0])
            result["BC"] = "NPBC"
        else:
            result["maxLoad"] = float(load_parts[2].split("PBC")[0])
            result["BC"] = "PBC"
    except IndexError:
        # Sometimes we load a vtu file that doesn't have the name format we expect
        pass

    # There are some values that we want to have even if they are there, but we
    # give a warning
    if "load" not in result:
        print("Warning: load not found in file name")
        result["load"] = 0.0
    if "BC" not in result:
        print("Warning: BC not found in file name")
        result["BC"] = "Unknown"
    if "seed" not in result:
        print("Warning: seed not found in file name")
        result["seed"] = 0
    if "loadIncrement" not in result:
        print("Warning: loadIncrement not found in file name")
        result["loadIncrement"] = 1e-5
    if "nrM" not in result:
        print("Warning: nrM not found in file name")
        result["nrM"] = 0.0

    return result


def get_file_number(vtu_file):
    return int(vtu_file.split(".")[-2])


def get_previous_data(vtu_file):
    """
    Given the path to a vtu file, it attempts to find the vtu file that comes
    before it by using the .number.vtu in the file name
    """
    # Get the directory and filename
    directory = os.path.dirname(vtu_file)
    # Get all .vtu files in the same directory
    files = [f for f in os.listdir(directory) if f.endswith(".vtu")]

    # Extract numbers and create a list of tuples (number, filename)
    file_numbers = []
    for f in files:
        num = get_file_number(f)
        file_numbers.append((num, f))

    # Sort the list based on the extracted numbers
    file_numbers.sort()

    # Extract the number from the given vtu_file
    given_num = get_file_number(vtu_file)

    # Find the previous file with a smaller number
    previous_file = None
    for num, f in file_numbers:
        if num < given_num:
            previous_file = f
        elif num >= given_num:
            break  # Since the list is sorted, we can break early

    # Return the full path to the previous file
    if previous_file:
        return os.path.join(directory, previous_file)
    else:
        return None  # No previous file found


if __name__ == "__main__":
    vtu_file = (
        "/Users/eliaslundheim/work/PhD/MTS2D/build/defaultName/data/remeshTest.0.vtu"
    )
    # get and print force components in a clear manner
    vtu = VTUData(vtu_file)
    force_components = vtu.get_force_contributions()
    # Print the force components
    print("Force components:")
    for i, force in enumerate(force_components):
        print(f"Element {i}:\n {force}")
    # Print the connectivity
    connectivity = vtu.get_connectivity()
    print("Connectivity:")
    print(connectivity)

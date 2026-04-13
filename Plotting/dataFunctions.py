import xml.etree.ElementTree as ET
import os
import meshio
import numpy as np
import re

from MTMath.meshUtils import arrsToMat, CArrsToMat


class VTUData:
    def __init__(self, vtu_file_path):
        self.vtu_file_path = vtu_file_path
        self.mesh = self._read_vtu_file()
        result = get_data_from_name(vtu_file_path)
        if "BC" in result:
            self.BC = result["BC"]
        if "load" in result:
            self.load = float(result["load"])
        self.size = self.get_size()

    def get_size(self):
        # Find the size in the file name
        # We assume the size is in the format s100x100
        regex = r"s(\d+)x(\d+)"
        match = re.search(regex, str(self.vtu_file_path))
        if match:
            self.size = (int(match.group(1)), int(match.group(2)))
        else:
            # We can guess the size by counting the number of elements
            # The number of nodes is not reliable.
            nrCells = self.get_cell_data("nrm3").shape[0]
            # We assume the mesh is square
            self.size = (int(np.sqrt(nrCells)), int(np.sqrt(nrCells)))
        return self.size

    def _read_vtu_file(self):
        """Read the VTU file using meshio and return a mesh object."""
        return meshio.read(self.vtu_file_path)

    def get_point_data(self, field):
        """Return a NumPy array with point data for the given field name."""
        try:
            return self.mesh.point_data[field]
        except KeyError:
            raise KeyError(f"Point data field '{field}' not found in VTU file")

    def get_cell_data(self, field):
        """Return a NumPy array with cell data for the given field name.

        meshio stores cell data as a dict mapping names to lists aligned with
        self.mesh.cells. We assume a single cell block and return its array.
        """
        if field not in self.mesh.cell_data:
            raise KeyError(f"Cell data field '{field}' not found in VTU file")

        data_list = self.mesh.cell_data[field]
        if len(data_list) != 1:
            raise ValueError(
                f"Cell data field '{field}' has {len(data_list)} blocks; "
                "VTUData assumes a single cell block."
            )
        arr = data_list[0]
        # meshio typically returns shape (n_cells, n_components) for cell data.
        # For scalar fields n_components == 1, which gives shape (N, 1).
        # To make scalar cell data 1D, squeeze a trailing singleton component axis.
        if arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def get_nodes(self):
        """Return node coordinates as a NumPy array of shape (n_points, 3)."""
        return self.mesh.points

    def get_reference_nodes(self):
        """Return reference nodes using a displacement field."""
        nodes = self.get_nodes()
        disp = self.get_point_data("displacement")
        ref = nodes.copy()
        ref[:, : disp.shape[1]] -= disp
        return ref

    def get_force_field(self):
        return self.get_point_data("stress_field")

    def get_stress_field(self):
        return self.get_cell_data("P12")

    def get_energy_field(self):
        return self.get_cell_data("energy_field")

    def get_fixed_status(self):
        return self.get_point_data("fixed")

    def get_m_nr_field(self):
        #nrm1 = self.get_cell_data("nrm1").astype(int)
        #nrm2 = self.get_cell_data("nrm2").astype(int)
        nrm3 = self.get_cell_data("nrm3").astype(int)
        return nrm3

    def get_m3_nr_field(self):
        return self.get_cell_data("nrm3").astype(int)

    def get_m3_change_field(self):
        return self.get_cell_data("deltaNrm3").astype(int)
    
    def get_M(self, elastic_M=False):
        """
        Returns a 3D array of 2x2 reduction matrices (one per element).
        If elastic_M=True, undo the final elastic_to_fundamental step using
        red_quadrant to recover the elastic-domain M.
        """
        M11, M12, M21, M22 = [
            self.get_cell_data(M) for M in ["m11", "m12", "m21", "m22"]
        ]
        M = arrsToMat(M11, M12, M21, M22)
        if not elastic_M:
            return M

        try:
            red_quadrant = self.get_cell_data("red_quadrant").astype(int)
        except KeyError as exc:
            raise KeyError(
                "Missing 'red_quadrant' cell data needed for elastic_M."
            ) from exc

        if red_quadrant.min() == 0:
            raise ValueError("Should be 1..4!")

        m1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
        m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        q2 = red_quadrant == 2
        q3 = red_quadrant == 3
        q4 = red_quadrant == 4

        # Undo elastic_to_fundamental:
        # q2: apply m1, q3: apply m2, q4: apply m2 then m1 (reverse order)
        if np.any(q2):
            M[q2] = M[q2] @ m1
        if np.any(q3):
            M[q3] = M[q3] @ m2
        if np.any(q4):
            M[q4] = M[q4] @ m2 @ m1

        return M

    def get_C(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [C11, C22, C12] components.
        """
        # Get the C11, C22, and C12 arrays from the VTK object
        C11, C12, C22 = [self.get_cell_data(C) for C in ["C11", "C12", "C22"]]
        return CArrsToMat(C11, C12, C22)

    def get_C_fix(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [C11, C22, C12] components.
        """
        # Get the C11, C22, and C12 arrays from the VTK object
        C11, C12, C22 = [
            self.get_cell_data(C) for C in ["C_Fix11", "C_Fix12", "C_Fix22"]
        ]
        return CArrsToMat(C11, C12, C22)

    def get_F(self):
        """
        Returns a 3D array where each slice (2x2 matrix) corresponds to the
        [F11, F12, F21, F22] components.
        """
        F11, F12, F21, F22 = [
            self.get_cell_data(F) for F in ["F11", "F12", "F21", "F22"]
        ]
        return arrsToMat(F11, F12, F21, F22)


    def get_F_fix(self):
        """
        Returns a 3D array for fixed F using stored F_Fix components.
        """
        F11, F12, F21, F22 = [
            self.get_cell_data(F) for F in ["F_Fix11", "F_Fix12", "F_Fix21", "F_Fix22"]
        ]
        return arrsToMat(F11, F12, F21, F22)

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
        """Return connectivity as an array of shape (n_cells, n_vertices_per_cell).

        For typical triangle meshes this will be (n_cells, 3). We prefer the
        "triangle" or "quad" blocks if available, otherwise we fall back to
        the first cell block.
        """
        # Prefer the dictionary interface if available (newer meshio)
        cells_dict = getattr(self.mesh, "cells_dict", None)
        if cells_dict:
            if "triangle" in cells_dict:
                connectivity = cells_dict["triangle"]
                return self._normalize_connectivity(connectivity)
            if "quad" in cells_dict:
                connectivity = cells_dict["quad"]
                return self._normalize_connectivity(connectivity)
            # Fall back to an arbitrary first entry from the dict
            first_key = next(iter(cells_dict))
            connectivity = cells_dict[first_key]
            return self._normalize_connectivity(connectivity)

        # Fallback for older meshio versions: use the first cell block
        cell_type, data = self.mesh.cells[0]
        return self._normalize_connectivity(data)

    def _normalize_connectivity(self, connectivity):
        connectivity = np.asarray(connectivity, dtype=int)
        if connectivity.size == 0:
            return connectivity
        n_nodes = self.get_nodes().shape[0]
        max_idx = int(connectivity.max())
        if max_idx == n_nodes:
            return connectivity - 1
        if max_idx > n_nodes:
            raise ValueError(
                f"Connectivity index {max_idx} exceeds node count {n_nodes}."
            )
        return connectivity


def parse_pvd_file(path, pvd_file):
    tree = ET.parse(pvd_file)
    root = tree.getroot()
    vtu_files = []

    for dataset in root.iter("DataSet"):
        if "_." not in dataset.attrib["file"] and False:
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
    # Initialize an empty dictionary
    result = {}

    # Strip extensions and optional load-step suffix so parsing is robust to
    # flags like "_minimal" or minStep values that contain dots.
    base_name = fileName
    if base_name.endswith(".vtu"):
        base_name = base_name[:-4]
        if "." in base_name:
            maybe_step = base_name.rsplit(".", 1)[1]
            if maybe_step.isdigit():
                result["load_step"] = int(maybe_step)
                base_name = base_name.rsplit(".", 1)[0]
    elif base_name.endswith(".csv"):
        base_name = base_name[:-4]

    parts = base_name.split("_")

    result["name"] = parts[0]
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            # Treat underscore-separated tokens without '=' as flags
            result[part] = True
            continue
        key, value = part.split("=", 1)
        # Add the key-value pair to the dictionary

        # minStep is special. It has the format iterations.func_evals
        if key == "minStep":
            try:
                result["nr_iterations"], result["nr_func_evals"] = map(
                    int, value.split(".")
                )
            except ValueError:
                # Fall back to a plain value if the expected format isn't met
                pass
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

    # Extract size information in the form sLxL
    size_match = re.search(r"s(\d+)x(\d+)", nameOrPath)
    if size_match:
        N1, N2 = int(size_match.group(1)), int(size_match.group(2))
        result["N"] = (N1, N2)
        if N1 == N2:
            result["L"] = N1
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

import xml.etree.ElementTree as ET
import os
import meshio
import numpy as np
import re
from pathlib import Path

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

    def get_matrix_component(self, matrix_name, i, j, symmetric_fallback=True):
        """Return one matrix component stored as cell data.

        Accepts both compact names like ``T11`` and underscored variants like
        ``T_11``. For symmetric tensors, missing off-diagonal components fall
        back to the transposed entry.
        """
        candidates = [
            f"{matrix_name}{i}{j}",
            f"{matrix_name}_{i}{j}",
            f"{matrix_name}{i}_{j}",
            f"{matrix_name}_{i}_{j}",
        ]
        for field in candidates:
            if field in self.mesh.cell_data:
                return self.get_cell_data(field)

        if symmetric_fallback and i != j:
            return self.get_matrix_component(
                matrix_name,
                j,
                i,
                symmetric_fallback=False,
            )

        available = sorted(
            key for key in self.mesh.cell_data.keys() if key.startswith(matrix_name)
        )
        raise KeyError(
            f"Missing matrix component for {matrix_name}[{i},{j}]. "
            f"Tried {candidates}. Available matching fields: {available}"
        )

    def get_matrix_components(self, matrix_name):
        """Return all four 2x2 matrix components as a dict keyed by ``(i, j)``."""
        return {
            (1, 1): self.get_matrix_component(matrix_name, 1, 1),
            (1, 2): self.get_matrix_component(matrix_name, 1, 2),
            (2, 1): self.get_matrix_component(matrix_name, 2, 1),
            (2, 2): self.get_matrix_component(matrix_name, 2, 2),
        }

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


def _vtu_step_from_name(vtu_file):
    match = re.search(r"\.(\d+)\.vtu$", Path(vtu_file).name)
    return int(match.group(1)) if match else None


def _vtu_sort_key(vtu_file):
    step = _vtu_step_from_name(vtu_file)
    if step is None:
        return (np.inf, Path(vtu_file).name)
    return (step, Path(vtu_file).name)


def resolve_vtu_files(vtu_source, pvd_name="collection.pvd"):
    if isinstance(vtu_source, (list, tuple, np.ndarray)):
        vtu_files = [str(p) for p in vtu_source if str(p).endswith(".vtu")]
        if not vtu_files:
            raise ValueError("No VTU files found in provided list.")
        return sorted(vtu_files, key=_vtu_sort_key)

    path = Path(vtu_source)
    if path.suffix.lower() == ".vtu":
        return [str(path)]

    if path.suffix.lower() == ".pvd":
        return sorted(parse_pvd_file(str(path.parent), str(path)), key=_vtu_sort_key)

    if path.suffix.lower() == ".csv":
        path = path.parent

    if path.is_dir():
        pvd_path = path / pvd_name
        if pvd_path.exists():
            return sorted(
                parse_pvd_file(str(path), str(pvd_path)),
                key=_vtu_sort_key,
            )

        data_dir = path / "data"
        vtu_files = sorted(data_dir.glob("*.vtu"), key=_vtu_sort_key)
        if not vtu_files:
            vtu_files = sorted(path.glob("*.vtu"), key=_vtu_sort_key)
        if not vtu_files:
            raise FileNotFoundError(f"No VTU files found in {path}.")
        return [str(p) for p in vtu_files]

    raise ValueError(f"Unsupported VTU source: {vtu_source!r}")

def infer_strain_from_vtu(vtu_file):
    """
    Infer strain directly from VTU naming metadata.
    Priority:
    1) explicit `_load=<value>` token
    2) reconstructed from `startLoad + (load_step-1)*loadIncrement`
       using simulation folder naming convention.
    """
    path_str = str(vtu_file)
    load_match = re.search(r"(?:^|[_/,])load=([-+0-9.eE]+)", path_str)
    if load_match:
        try:
            return float(load_match.group(1))
        except ValueError:
            pass

    step = _vtu_step_from_name(vtu_file)
    if step is None:
        return None

    candidates = [
        Path(vtu_file).parent.parent.name,
        Path(vtu_file).parent.name,
        path_str,
    ]
    for candidate in candidates:
        sim_match = re.search(
            r"s\d+x\d+l([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)(?:NPBC|PBC)",
            candidate,
        )
        if not sim_match:
            continue
        try:
            start_load = float(sim_match.group(1))
            load_increment = float(sim_match.group(2))
            return start_load + (step - 1) * load_increment
        except ValueError:
            continue
    return None


def match_vtu_to_macro_row(df, vtu_file, X="load", fallback_first_on_missing_load=True):
    """
    Match a VTU frame to one row in a macro-data dataframe.

    Matching priority:
    1) `load_step` if available in both VTU metadata and dataframe
    2) `load` otherwise

    Returns:
        (matching_row, matching_row_index, x_value)
    """
    meta = get_data_from_name(vtu_file)

    if "load_step" in df.columns and "load_step" in meta:
        n = meta["load_step"]
        matching_rows = df[df["load_step"] == n]
        if len(matching_rows) != 1:
            print(
                f"Warning: in file {vtu_file}:\n"
                f"load_step value '{n}' is not unique or not found. "
                f"Found {len(matching_rows)} matches."
            )
            print(
                "Try moving/deleting vtu files that are further ahead than the csv file."
            )
    elif "load" in df.columns:
        load = infer_strain_from_vtu(vtu_file)
        if load is None or not np.isfinite(load):
            load = meta.get("load", np.nan)
        matching_rows = df[df["load"] == load]
        if len(matching_rows) == 0:
            print(f"Warning: load {load} not found!")
            if fallback_first_on_missing_load:
                matching_rows = df.iloc[[0]]
    else:
        raise ValueError("Neither 'load_step' nor 'load' columns found in DataFrame.")

    if len(matching_rows) == 0:
        raise ValueError(f"No matching macro-data row found for {vtu_file}.")

    matching_row_index = matching_rows.index[0]
    matching_row = matching_rows.iloc[0]

    if X == "load":
        x_value = infer_strain_from_vtu(vtu_file)
        if x_value is None or not np.isfinite(x_value):
            x_value = meta.get("load", np.nan)
    else:
        x_value = meta.get(X, np.nan)
    return matching_row, matching_row_index, x_value


def _flatten_force_contribution_magnitudes(force_contributions):
    force_contributions = np.asarray(force_contributions, dtype=float)
    if force_contributions.ndim != 3:
        raise ValueError(
            f"Expected 3D force contribution array, got {force_contributions.shape}."
        )

    # Accept both conventions:
    # - (n_elements, 2, 3)
    # - (n_elements, 3, 2)
    if force_contributions.shape[-1] == 2:
        magnitudes = np.linalg.norm(force_contributions, axis=-1).reshape(-1)
    elif force_contributions.shape[1] == 2:
        magnitudes = np.linalg.norm(force_contributions, axis=1).reshape(-1)
    else:
        raise ValueError(
            f"Unexpected force contribution shape {force_contributions.shape}."
        )

    magnitudes = magnitudes[np.isfinite(magnitudes)]
    if magnitudes.size == 0:
        raise ValueError("No finite force-contribution magnitudes found.")
    return magnitudes


def _extract_force_contribution_magnitude_series(vtu_source, contribution_getter, *, use_tqdm=False, series_name="force contributions"):
    vtu_files = resolve_vtu_files(vtu_source)
    strains, means, stds = [], [], []
    skipped_missing_strain = 0
    skipped_bad_vtu = 0

    if use_tqdm:
        from tqdm import tqdm

        iterator = tqdm(vtu_files, desc=f"{series_name} series")
    else:
        iterator = vtu_files

    for vtu_file in iterator:
        strain = infer_strain_from_vtu(vtu_file)
        if strain is None or not np.isfinite(strain):
            skipped_missing_strain += 1
            continue

        try:
            force_contributions = contribution_getter(vtu_file)
            magnitudes = _flatten_force_contribution_magnitudes(force_contributions)
        except Exception as exc:
            print(f"Warning: failed {series_name} extraction for {vtu_file}: {exc}")
            skipped_bad_vtu += 1
            continue

        strains.append(float(strain))
        means.append(float(np.mean(magnitudes)))
        stds.append(float(np.std(magnitudes)))

    if not strains:
        raise ValueError(f"No valid VTU frames with {series_name} found.")

    order = np.argsort(np.asarray(strains))
    strains = np.asarray(strains, dtype=float)[order]
    means = np.asarray(means, dtype=float)[order]
    stds = np.asarray(stds, dtype=float)[order]

    if skipped_missing_strain > 0:
        print(f"Skipped {skipped_missing_strain} VTU frames without inferable strain.")
    if skipped_bad_vtu > 0:
        print(f"Skipped {skipped_bad_vtu} VTU frames due to read/shape issues.")

    return strains, means, stds


def extract_force_contribution_magnitude_series(vtu_source, use_tqdm=False):
    """
    Compute mean/std of |F_ei| from VTU force contributions ("Umut F").
    Returns arrays (strain, mean_magnitude, std_magnitude).
    """
    return _extract_force_contribution_magnitude_series(
        vtu_source,
        lambda vtu_file: VTUData(vtu_file).get_force_contributions(),
        use_tqdm=use_tqdm,
        series_name="Umut F contributions",
    )


def _true_force_contributions_from_F(vtu_file):
    from MTMath.energyFunction import ContiEnergy
    from MTMath.meshUtils import triangle_shape_grads_and_area

    data = VTUData(vtu_file)
    connectivity = data.get_connectivity()
    ref_nodes = data.get_reference_nodes()

    if connectivity.size and connectivity.max() >= len(ref_nodes):
        raise ValueError(
            f"Connectivity max index {connectivity.max()} exceeds reference node count {len(ref_nodes)}."
        )

    ref_elem_coords = ref_nodes[connectivity][:, :, :2]
    dN_dX, area_ref = triangle_shape_grads_and_area(ref_elem_coords)
    F = data.get_F()
    return ContiEnergy.lagrangian_forces_from_F(F, dN_dX, area=area_ref)


def extract_true_force_contribution_magnitude_series(vtu_source, use_tqdm=False):
    """
    Compute mean/std of |F_ei| by recomputing Lagrangian element-node forces from F ("True F").
    Returns arrays (strain, mean_magnitude, std_magnitude).
    """
    return _extract_force_contribution_magnitude_series(
        vtu_source,
        _true_force_contributions_from_F,
        use_tqdm=use_tqdm,
        series_name="True F contributions",
    )


def _resolve_macrodata_csv_path(nameOrPath):
    path_str = str(nameOrPath)
    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        return path
    if path.exists():
        if path.is_dir():
            return path / "macroData.csv"
        return path.parent / "macroData.csv"
    if os.sep in path_str:
        if path.suffix:
            return path.parent / "macroData.csv"
        return path / "macroData.csv"
    return None


def _infer_constant_load_increment(nameOrPath):
    csv_path = _resolve_macrodata_csv_path(nameOrPath)
    if csv_path is None:
        raise ValueError(
            f"loadIncrement not found in file name, and no CSV path could be "
            f"resolved from {nameOrPath!r}."
        )

    from Management.updateCSV import read_macrodata_csv

    df = read_macrodata_csv(csv_path)
    if "load" not in df:
        raise ValueError(
            f"loadIncrement not found in file name, and {csv_path} has no 'load' column."
        )

    load = df["load"].to_numpy(dtype=float)
    if load.size < 2:
        raise ValueError(
            f"loadIncrement not found in file name, and {csv_path} has fewer than "
            "two load entries."
        )

    diffs = np.diff(load)
    if not np.all(np.isfinite(diffs)):
        raise ValueError(
            f"loadIncrement not found in file name, and non-finite load differences "
            f"were found in {csv_path}."
        )

    inferred_increment = float(diffs[0])
    if np.isclose(inferred_increment, 0.0, rtol=0.0, atol=1e-15):
        if np.allclose(diffs, 0.0, rtol=0.0, atol=1e-15):
            print(
                "Warning: loadIncrement not found in file name, and the inferred "
                f"load increment from {csv_path} is zero. Using 0.0."
            )
            return 0.0

    if not np.allclose(diffs, inferred_increment, rtol=1e-9, atol=1e-12):
        return diffs

    return inferred_increment


def _parse_name_metadata(raw_name):
    result = {}
    if not raw_name:
        return result

    base_name = str(raw_name)
    known_suffixes = (".csv", ".conf", ".pvd", ".xml", ".gz", ".mtsb")

    # Strip extensions and optional load-step suffix so parsing is robust to
    # flags like "_minimal" or minStep values that contain dots.
    if base_name.endswith(".vtu"):
        base_name = base_name[:-4]
        if "." in base_name:
            maybe_step = base_name.rsplit(".", 1)[1]
            if maybe_step.isdigit():
                result["load_step"] = int(maybe_step)
                base_name = base_name.rsplit(".", 1)[0]
    else:
        for suffix in known_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

    parts = base_name.split("_")
    if not parts or not parts[0]:
        return result

    result["name"] = parts[0]
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            # Treat underscore-separated tokens without '=' as flags
            result[part] = True
            continue
        key, value = part.split("=", 1)

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
        # Sometimes we load a file that doesn't have the name format we expect
        pass
    except ValueError:
        pass

    return result


def _metadata_score(metadata):
    informative_keys = ("dims", "seed", "BC", "startLoad", "loadIncrement", "maxLoad")
    return sum(key in metadata for key in informative_keys)


def get_data_from_name(nameOrPath):
    if not isinstance(nameOrPath, str):
        nameOrPath = str(nameOrPath)
    path = Path(nameOrPath)
    fileName = path.name
    is_csv = fileName.endswith(".csv")

    file_result = _parse_name_metadata(fileName)
    parent_result = {}
    if path.parent.name:
        parent_result = _parse_name_metadata(path.parent.name)

    if _metadata_score(parent_result) > _metadata_score(file_result):
        result = dict(parent_result)
        fallback = file_result
    else:
        result = dict(file_result)
        fallback = parent_result

    for key, value in fallback.items():
        if key not in result:
            result[key] = value

    # There are some values that we want to have even if they are there, but we
    # give a warning
    if "load" not in result:
        if not is_csv:
            print("Warning: load not found in file name")
        result["load"] = 0.0
    if "BC" not in result:
        print("Warning: BC not found in file name")
        result["BC"] = "Unknown"
    if "seed" not in result:
        print("Warning: seed not found in file name")
        result["seed"] = 0
    if "loadIncrement" not in result:
        inferred_load_increment = _infer_constant_load_increment(nameOrPath)
        if (
            not is_csv
            and isinstance(inferred_load_increment, np.ndarray)
            and "load_step" in result
            and inferred_load_increment.size > 0
        ):
            step_idx = max(int(result["load_step"]) - 1, 0)
            step_idx = min(step_idx, inferred_load_increment.size - 1)
            result["loadIncrement"] = float(inferred_load_increment[step_idx])
        else:
            result["loadIncrement"] = inferred_load_increment
    if "nrM" not in result:
        if not is_csv:
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


def get_metadata(nameOrPath):
    return get_data_from_name(nameOrPath)


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

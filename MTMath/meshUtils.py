import numpy as np


def arrsToMat(A11, A12, A21, A22):
    """Assemble (N,2,2) from component arrays."""
    assert A11.shape == A12.shape == A21.shape == A22.shape, (
        "All components should have the same shape"
    )
    A = np.zeros((A11.shape[0], 2, 2))
    A[:, 0, 0] = A11
    A[:, 0, 1] = A12
    A[:, 1, 0] = A21
    A[:, 1, 1] = A22
    return A


def CArrsToMat(C11, C12, C22):
    """Assemble (N,2,2) from C11,C12,C22."""
    return arrsToMat(C11, C12, C12, C22)


def triangle_shape_grads_and_area(coords):
    """Triangle dN/dx and area."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape[-2:] != (3, 2):
        raise ValueError(f"coords must have shape (..., 3, 2), got {coords.shape}")

    dN_dxi = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    J = coords.swapaxes(-1, -2) @ dN_dxi
    J_inv = np.linalg.inv(J)
    dN_dx = dN_dxi @ J_inv
    area = 0.5 * np.abs(np.linalg.det(J))
    return dN_dx, area


def shape_grads_and_area_from_F(dN_dX, area_ref, F):
    """Compute dN/dx and area from reference gradients and F."""
    dN_dX = np.asarray(dN_dX, dtype=float)
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (2, 2):
        raise ValueError(f"F must have shape (..., 2, 2), got {F.shape}")
    if dN_dX.shape[-2:] != (3, 2):
        raise ValueError(f"dN_dX must have shape (..., 3, 2), got {dN_dX.shape}")

    F_inv = np.linalg.inv(F)
    dN_dx = np.einsum("...ai,...ij->...aj", dN_dX, F_inv)
    detF = np.linalg.det(F)
    area_cur = area_ref * detF
    return dN_dx, area_cur


def _build_triangular_elements(shape):
    nx, ny = shape
    element_indices: list[list[int]] = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            bl = i * nx + j
            br = i * nx + (j + 1)
            tl = (i + 1) * nx + j
            tr = (i + 1) * nx + (j + 1)
            element_indices.append([bl, br, tr])
            element_indices.append([bl, tr, tl])
    return np.asarray(element_indices, dtype=int)


def _compute_dN_dX(ref_positions: np.ndarray, element_indices: np.ndarray) -> np.ndarray:
    dN_dxi = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    elem_X = ref_positions[element_indices]
    X1 = elem_X[:, 0, :]
    X2 = elem_X[:, 1, :]
    X3 = elem_X[:, 2, :]
    v1 = X2 - X1
    v2 = X3 - X1
    J = np.stack([v1, v2], axis=-1)
    J_inv = np.linalg.inv(J)
    return np.einsum("ai,eij->eaj", dN_dxi, J_inv)


def _compute_F(
    ref_positions: np.ndarray,
    u: np.ndarray,
    shear: float,
    element_indices: np.ndarray,
    dN_dX_values: np.ndarray,
) -> np.ndarray:
    u_shear = np.zeros_like(ref_positions)
    u_shear[:, 0] = shear * ref_positions[:, 1]
    x_nodes = ref_positions + u_shear + u
    x_elem = x_nodes[element_indices]
    return np.einsum("eai,eaj->eij", x_elem, dN_dX_values)


def _element_subset_indices(n_elements, element_subset):
    if element_subset is None:
        return None
    if isinstance(element_subset, str):
        subset = element_subset.strip().lower()
    else:
        subset = element_subset
    if not subset or subset == "none":
        return None
    if subset not in ("odd", "even"):
        return None
    start = 1 if subset == "odd" else 0
    return np.arange(start, n_elements, 2, dtype=int)


def cell_energy_to_node_energy(nodes, energy_field, connectivity):
    node_energy = np.zeros(len(nodes))
    node_count = np.zeros(len(nodes))
    for cell_index, cell in enumerate(connectivity):
        for node_index in cell:
            node_energy[node_index] += energy_field[cell_index]
            node_count[node_index] += 1
    if (node_count == 0).any():
        raise Exception("Invalid Mesh")
    node_energy /= node_count
    return node_energy


def _center_node_index(nodes_xy):
    center = 0.5 * (nodes_xy.min(axis=0) + nodes_xy.max(axis=0))
    d2 = np.sum((nodes_xy - center) ** 2, axis=1)
    return int(np.argmin(d2))


def _assemble_nodal_forces(elem_forces, connectivity, n_nodes):
    forces = np.zeros((n_nodes, 2), dtype=elem_forces.dtype)
    for local_idx in range(connectivity.shape[1]):
        np.add.at(forces, connectivity[:, local_idx], elem_forces[:, local_idx, :])
    return forces


def perfect_grid_nodes(shape, dx=1.0, dy=1.0):
    nx, ny = shape
    xs = np.arange(nx, dtype=float) * dx
    ys = np.arange(ny, dtype=float) * dy
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def grid_index(i, j, nx):
    """Row-major index for (i,j)."""
    return int(j * nx + i)

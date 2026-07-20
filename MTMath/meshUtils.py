from typing import Literal

import numpy as np


TriangleDiagonal = Literal["major", "minor"]


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


def structured_triangle_connectivity(
    shape: tuple[int, int],
    *,
    diagonal: TriangleDiagonal = "minor",
) -> np.ndarray:
    """Build counter-clockwise triangle connectivity for a structured node grid.

    ``shape`` is ``(nx, ny)`` in nodes. ``major`` uses the diagonal from the
    upper-left to the lower-right corner of each cell; ``minor`` uses the
    diagonal from the lower-left to the upper-right corner.
    """
    nx, ny = shape
    if nx < 2 or ny < 2:
        raise ValueError(f"shape must contain at least 2 nodes per axis, got {shape}.")
    if diagonal not in ("major", "minor"):
        raise ValueError(
            f"diagonal must be 'major' or 'minor', got {diagonal!r}."
        )

    element_indices: list[list[int]] = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            bl = row * nx + column
            br = bl + 1
            tl = (row + 1) * nx + column
            tr = tl + 1
            if diagonal == "major":
                element_indices.append([bl, br, tl])
                element_indices.append([br, tr, tl])
            else:
                element_indices.append([bl, br, tr])
                element_indices.append([bl, tr, tl])
    return np.asarray(element_indices, dtype=int)


def structured_triangular_mesh(
    shape: tuple[int, int],
    *,
    diagonal: TriangleDiagonal = "minor",
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and connectivity for a structured triangular mesh."""
    nodes = perfect_grid_nodes(shape, dx=dx, dy=dy)
    connectivity = structured_triangle_connectivity(shape, diagonal=diagonal)
    return nodes, connectivity


def unique_mesh_edges(connectivity: np.ndarray) -> np.ndarray:
    """Return the sorted unique node pairs used by a triangular mesh."""
    connectivity = np.asarray(connectivity, dtype=int)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            "connectivity must have shape (n_elements, 3), "
            f"got {connectivity.shape}."
        )
    if connectivity.size == 0:
        return np.empty((0, 2), dtype=int)

    edges = np.concatenate(
        [
            connectivity[:, [0, 1]],
            connectivity[:, [1, 2]],
            connectivity[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def mesh_edge_segments(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Convert node-index pairs into line segments with shape ``(n, 2, 2)``."""
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"nodes must have shape (n_nodes, 2), got {nodes.shape}.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (n_edges, 2), got {edges.shape}.")
    if edges.size and (edges.min() < 0 or edges.max() >= len(nodes)):
        raise ValueError("edges contain a node index outside the nodes array.")
    return nodes[edges]


def element_deformation_gradients(
    reference_nodes: np.ndarray,
    current_nodes: np.ndarray,
    connectivity: np.ndarray,
) -> np.ndarray:
    """Calculate one deformation gradient per linear triangular element.

    ``current_nodes`` may include leading batch dimensions, for example a load
    history with shape ``(n_steps, n_nodes, 2)``.
    """
    reference_nodes = np.asarray(reference_nodes, dtype=float)
    current_nodes = np.asarray(current_nodes, dtype=float)
    connectivity = np.asarray(connectivity, dtype=int)
    if reference_nodes.ndim != 2 or reference_nodes.shape[1] != 2:
        raise ValueError(
            "reference_nodes must have shape (n_nodes, 2), "
            f"got {reference_nodes.shape}."
        )
    if current_nodes.shape[-2:] != reference_nodes.shape:
        raise ValueError(
            "current_nodes must end with the reference node shape "
            f"{reference_nodes.shape}, got {current_nodes.shape}."
        )
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            "connectivity must have shape (n_elements, 3), "
            f"got {connectivity.shape}."
        )
    if connectivity.size and (
        connectivity.min() < 0 or connectivity.max() >= len(reference_nodes)
    ):
        raise ValueError("connectivity contains a node index outside the nodes array.")

    reference_elements = reference_nodes[connectivity]
    current_elements = current_nodes[..., connectivity, :]
    dX = np.stack(
        [
            reference_elements[:, 1] - reference_elements[:, 0],
            reference_elements[:, 2] - reference_elements[:, 0],
        ],
        axis=-1,
    )
    dx = np.stack(
        [
            current_elements[..., 1, :] - current_elements[..., 0, :],
            current_elements[..., 2, :] - current_elements[..., 0, :],
        ],
        axis=-1,
    )
    return dx @ np.linalg.inv(dX)


def _build_triangular_elements(shape):
    """Backward-compatible alias for the historical minor-diagonal mesh."""
    return structured_triangle_connectivity(shape, diagonal="minor")


def _compute_dN_dX(ref_positions: np.ndarray, element_indices: np.ndarray) -> np.ndarray:
    """Backward-compatible indexed wrapper around the public triangle helper."""
    dN_dX, _ = triangle_shape_grads_and_area(ref_positions[element_indices])
    return dN_dX


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
    return element_deformation_gradients(ref_positions, x_nodes, element_indices)


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

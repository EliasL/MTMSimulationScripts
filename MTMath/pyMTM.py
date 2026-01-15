import numpy as np
from scipy.optimize import minimize

from MTMath.energyFunction import ContiEnergy
from tqdm import tqdm
import multiprocessing as mp


def _build_triangular_elements(shape):
    """
    Build a list of triangular elements for an (nx, ny) grid of nodes.

    Each quadrilateral cell is split into two triangles:
    [bl, br, tr] and [bl, tr, tl].
    """
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


def _compute_dN_dX(
    ref_positions: np.ndarray, element_indices: np.ndarray
) -> np.ndarray:
    """
    Numerically compute dN/dX for each linear triangular element.

    Parameters
    ----------
    ref_positions : (N_nodes, 2) array
        Reference coordinates X of all nodes.
    element_indices : (n_elements, 3) int array
        Node indices for each element.

    Returns
    -------
    dN_dX_values : (n_elements, 3, 2) array
        Gradients of the three shape functions w.r.t. X for each element.
    """
    # Natural-coordinate gradients for a linear triangle with nodes
    # (0,0), (1,0), (0,1)
    dN_dxi = np.array(
        [
            [-1.0, -1.0],  # grad N1
            [1.0, 0.0],  # grad N2
            [0.0, 1.0],  # grad N3
        ],
        dtype=float,
    )  # (3,2)

    # Element-wise reference coordinates (n_elements, 3, 2)
    elem_X = ref_positions[element_indices]  # X1, X2, X3
    X1 = elem_X[:, 0, :]  # (E,2)
    X2 = elem_X[:, 1, :]
    X3 = elem_X[:, 2, :]

    # Jacobian J = [X2 - X1, X3 - X1] (E, 2, 2)
    v1 = X2 - X1
    v2 = X3 - X1
    J = np.stack([v1, v2], axis=-1)  # (E,2,2)

    J_inv = np.linalg.inv(J)  # (E,2,2)

    # dN/dX = dN/dxi @ J^{-1}  -> (E,3,2)
    dN_dX_values = np.einsum("ai,eij->eaj", dN_dxi, J_inv)
    return dN_dX_values


def _compute_F(
    ref_positions: np.ndarray,
    u: np.ndarray,
    shear: float,
    element_indices: np.ndarray,
    dN_dX_values: np.ndarray,
) -> np.ndarray:
    """
    Compute deformation gradients F for all elements, numerically.

    Parameters
    ----------
    ref_positions : (N_nodes, 2)
    u             : (N_nodes, 2) displacement field
    shear         : scalar shear parameter
    element_indices : (n_elements, 3)
    dN_dX_values  : (n_elements, 3, 2)

    Returns
    -------
    F_vals : (n_elements, 2, 2)
    """
    n_nodes = ref_positions.shape[0]
    # Macroscopic simple shear displacement: u_shear = [gamma * Y, 0]
    u_shear = np.zeros_like(ref_positions)
    u_shear[:, 0] = shear * ref_positions[:, 1]

    x_nodes = ref_positions + u_shear + u  # (N_nodes, 2)
    x_elem = x_nodes[element_indices]  # (E,3,2)

    # F_e(i,J) = sum_a x_a(i) * dN_dX_a(J)
    F_vals = np.einsum("eai,eaj->eij", x_elem, dN_dX_values)  # (E,2,2)
    return F_vals


def _compute_energies_parallel(F_vals: np.ndarray, n_procs: int | None):
    """
    Optionally compute energies in parallel over elements.

    Parameters
    ----------
    F_vals : (n_elements, 2, 2)
        Deformation gradients for all elements.
    n_procs : int or None
        Number of worker processes to use. If None or <= 1, runs serially.

    Returns
    -------
    energies : (n_elements,) array
        Per-element energies.
    """
    # Fallback to serial evaluation if no real parallelism is requested or if the
    # problem is too small to amortise multiprocessing overhead.
    if n_procs is None or n_procs <= 1 or F_vals.shape[0] < 4 * (n_procs or 1):
        return ContiEnergy.energy_from_F(F_vals)

    # Split elements into chunks and map them to worker processes.
    chunks = np.array_split(F_vals, n_procs, axis=0)
    with mp.Pool(processes=n_procs) as pool:
        results = pool.map(ContiEnergy.energy_from_F, chunks)
    # Concatenate back to a single (n_elements,) array.
    return np.concatenate(results, axis=0)


def _compute_forces_parallel(
    F_vals: np.ndarray, dN_dX_values: np.ndarray, n_procs: int | None
) -> np.ndarray:
    """
    Optionally compute per-element forces in parallel over elements.

    Parameters
    ----------
    F_vals : (n_elements, 2, 2)
        Deformation gradients for all elements.
    dN_dX_values : (n_elements, 3, 2)
        Shape function gradients for each element.
    n_procs : int or None
        Number of worker processes to use. If None or <= 1, runs serially.

    Returns
    -------
    forces : (n_elements, 3, 2)
        Lagrangian forces for each element.
    """
    if n_procs is None or n_procs <= 1 or F_vals.shape[0] < 4 * (n_procs or 1):
        return ContiEnergy.lagrangian_forces_from_F(F_vals, dN_dX_values)

    # Split both F and dN/dX consistently along the element axis.
    F_chunks = np.array_split(F_vals, n_procs, axis=0)
    dN_chunks = np.array_split(dN_dX_values, n_procs, axis=0)

    with mp.Pool(processes=n_procs) as pool:
        results = pool.starmap(
            ContiEnergy.lagrangian_forces_from_F,
            zip(F_chunks, dN_chunks),
        )

    return np.concatenate(results, axis=0)


def pyMTM(
    shape=(2, 2),
    shearValues=np.linspace(0, 3, 100),
    n_procs: int | None = None,
):
    """
    Perform a shear simulation on an arbitrary rectangular grid of nodes.

    Parameters
    ----------
    shape : tuple of ints
        Tuple ``(nx, ny)`` giving the number of nodes in the x‐ and y‐directions.
        For example ``(2, 2)`` reproduces the 2×2 case from ``miniMTM``.
    shearValues : array_like
        One‐dimensional array of shear magnitudes at which to compute the deformation.
    n_procs : int or None, optional
        Number of worker processes to use for parallel element-wise energy and
        force evaluations. If ``None`` or ``<= 1``, computations are performed
        serially. Note that starting worker processes inside the optimiser can
        incur overhead; for small meshes it is usually best to leave this as
        ``None``.

    Returns
    -------
    pos_history : list of ndarray
        A list with length equal to ``len(shearValues)``.  Each entry is an array of shape
        ``(N_nodes, 2)`` giving the deformed positions of every node at the corresponding
        shear value after energy minimisation.
    energies : list of float
        The average element energy at each shear step.

    Notes
    -----
    This function generalises the ``simpleShearSystem2`` helper from ``miniMTM``.  It
    constructs a triangular finite element mesh over a regular ``nx × ny`` grid and
    applies a simple shear transformation.  At each shear value a constrained energy
    minimisation problem is solved using the L‑BFGS‑B algorithm from ``scipy.optimize``.
    Nodes on the boundary of the domain are fixed (zero displacement) while interior
    nodes are free to move.  The per‐element energy and its gradient are computed via
    ``ContiEnergy.energy_from_F`` and ``ContiEnergy.lagrangian_forces_from_F``.
    """
    # Unpack grid dimensions
    nx, ny = shape
    n_nodes = nx * ny

    # Build the list of triangular elements and reference positions
    element_indices = _build_triangular_elements(shape)  # (n_elements, 3)
    n_elements = element_indices.shape[0]

    # Reference positions in (x, y) order; x varies fastest.
    ref_positions = np.array(
        [[j, i] for i in range(ny) for j in range(nx)], dtype=float
    )

    # Precompute dN/dX for each element (independent of shear and displacement)
    dN_dX_values = _compute_dN_dX(ref_positions, element_indices)

    # Identify boundary and interior node indices for fixed boundaries
    interior_indices = []
    boundary_indices = []
    for idx in range(n_nodes):
        i = idx // nx  # row
        j = idx % nx  # column
        if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
            boundary_indices.append(idx)
        else:
            interior_indices.append(idx)
    interior_indices = np.array(interior_indices, dtype=int)

    # Initialise displacement field u for all nodes (zeros)
    u_full = np.zeros((n_nodes, 2), dtype=float)

    # Reproducible random perturbation for first minimization
    rng = np.random.default_rng(12345)  # fixed seed
    first_step = True

    # Storage for positions and average energies
    pos_history = []
    avg_energies = []

    # Loop over shear values
    for s in tqdm(shearValues):
        if first_step:
            # Add small random perturbation to initial node positions (displacements)
            u_full += rng.normal(scale=1e-3, size=u_full.shape)
            first_step = False
        # Current u flattened for interior nodes as initial guess
        x0 = u_full[interior_indices].reshape(-1)

        # Define the energy functional
        def energy_func(u_flat):
            # Reconstruct full displacement array from flattened interior values
            u_tmp = np.zeros_like(u_full)
            u_tmp[interior_indices] = u_flat.reshape(len(interior_indices), 2)
            # Evaluate F at current shear and displacement (purely numerical)
            F_vals = _compute_F(ref_positions, u_tmp, s, element_indices, dN_dX_values)
            # Compute per-element energies (optionally in parallel) and return the average
            energies = _compute_energies_parallel(F_vals, n_procs)
            return float(np.mean(energies))

        # Define the gradient of the energy functional
        def grad_func(u_flat):
            # Reconstruct full displacement array
            u_tmp = np.zeros_like(u_full)
            u_tmp[interior_indices] = u_flat.reshape(len(interior_indices), 2)
            # Evaluate F and compute forces (optionally in parallel)
            F_vals = _compute_F(ref_positions, u_tmp, s, element_indices, dN_dX_values)
            forces = _compute_forces_parallel(F_vals, dN_dX_values, n_procs)  # (E,3,2)
            # Accumulate forces on global nodes using scatter-add
            total_force = np.zeros((n_nodes, 2), dtype=float)
            np.add.at(total_force, element_indices, forces)
            # The gradient of energy w.r.t. displacements is minus the nodal force
            grad = -total_force[interior_indices].reshape(-1)
            return grad

        # Minimise the energy using L-BFGS-B on interior node displacements
        res = minimize(energy_func, x0, jac=grad_func, method="L-BFGS-B")

        # Update the full displacement field with the optimised values
        u_full[interior_indices] = res.x.reshape(len(interior_indices), 2)

        # Compute final F and energies for this shear step
        F_vals = _compute_F(ref_positions, u_full, s, element_indices, dN_dX_values)
        energies = _compute_energies_parallel(F_vals, n_procs)
        avg_energies.append(float(np.mean(energies)))

        # Deformed positions: reference + macroscopic shear + optimised displacements
        u_shear = np.zeros_like(ref_positions)
        u_shear[:, 0] = s * ref_positions[:, 1]
        pos_s = ref_positions + u_shear + u_full
        pos_history.append(pos_s)

    return pos_history, avg_energies


if __name__ == "__main__":
    strain = np.arange(0.15, 1, 1e-5)
    # Slow
    pos, e = pyMTM((60, 60), shearValues=strain)
    from matplotlib import pyplot as plt

    plt.plot(range(len(e)), e)
    plt.show()

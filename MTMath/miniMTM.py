import numpy as np

try:
    # Package context (e.g. python -m MTMath.miniMTM)
    from .SymbolicFEM import FEM
    from .energyFunction import ContiEnergy
except ImportError:  # Direct script execution (e.g. python MTMath/miniMTM.py)
    from SymbolicFEM import FEM
    from energyFunction import ContiEnergy
from matplotlib import pyplot as plt
import sympy as sp


def simpleShearSystem(L=2, shearValues=np.linspace(0, 3, 100)):
    """
    Make a LxL system of nodes connected in a triangular mesh.
    Apply shear and calculate F
    """

    # Create nodes (L**2)
    nodes = FEM.make_N_nodes(L**2)

    # Create elements (list of three nodes)
    # We cheat for now, and force L=2
    elements = [
        [nodes[0], nodes[1], nodes[2]],
        [nodes[1], nodes[2], nodes[3]],
        [nodes[1], nodes[3], nodes[2]],
        [nodes[3], nodes[2], nodes[1]],
    ]

    # Use .tolist() so Sympy keeps the (elements, 2, 2) / (elements, 3, 2) shapes
    F = sp.Array([FEM.F(e).tolist() for e in elements])
    dN_dX = sp.Array([FEM.dN_dX(e).tolist() for e in elements])

    # Create symbolic shear variable
    shear = sp.symbols("shear")
    sheared_F = FEM.apply_shear(F, shear)

    # Set reference positions.
    ref_positions = np.array([[i % L, i // L] for i in range(L**2)])

    F_func = sp.lambdify([FEM.X, shear], sheared_F, "numpy")
    dN_dX_func = sp.lambdify(FEM.X, dN_dX, "numpy")
    dN_dX_values = dN_dX_func(ref_positions)

    sheared_positions = FEM.apply_shear(FEM.x, shear)
    pos_func = sp.lambdify([FEM.X, shear], sheared_positions, "numpy")
    pos = np.array([pos_func(ref_positions, s) for s in shearValues])

    # Evaluate F for each shear value and stack results
    F_values = np.array([F_func(ref_positions, s) for s in shearValues])

    # Reshape into (shear_steps, elements, 2, 2)
    n_shear = shearValues.shape[0]
    n_elements = len(elements)
    F_values = F_values.reshape(n_shear, n_elements, 2, 2)
    dN_dX_values = dN_dX_values.reshape(n_elements, 3, 2)
    return pos, elements, F_values, dN_dX_values


# New function using the updated FEM.Element abstraction
def simpleShearSystem2(L=2, shearValues=np.linspace(0, 3, 100), periodic=False):
    """
    Make a LxL system of nodes connected in a triangular mesh.
    Apply shear and calculate F using new FEM.Element abstraction.
    Note: L is the number of nodes per side; this yields (L-1)^2 * 2 triangular elements.
    """
    if periodic:
        raise NotImplementedError(
            "Periodic boundary conditions are not implemented yet for simpleShearSystem2."
        )
    N = L**2
    FEM.make_N_nodes(N)

    # Explicit triangular elements using node indices
    element_indices = []
    for j in range(L - 1):
        for i in range(L - 1):
            n0 = j * L + i
            n1 = n0 + 1
            n2 = n0 + L
            n3 = n2 + 1
            # Two triangles per cell (n0, n1, n2) and (n1, n3, n2)
            element_indices.append([n0, n1, n2])
            element_indices.append([n1, n3, n2])
    elements = [FEM.Element(ids) for ids in element_indices]

    # Use .tolist() so Sympy keeps the (elements, 2, 2) / (elements, 3, 2) shapes
    F = sp.Array([FEM.F(e).tolist() for e in elements])
    dN_dX = sp.Array([FEM.dN_dX(e).tolist() for e in elements])

    shear = sp.symbols("shear")

    # Apply shear to deformation gradient and interpolated x field
    sheared_F = FEM.apply_shear(F, shear)

    # Apply shear to interpolated x field for each node, gather into matrix
    sheared_positions = sp.Matrix(
        [FEM.apply_shear(node["x"], shear) for node in FEM.nodes]
    )

    # Lambdify evaluation functions
    ref_positions = np.array([[i % L, i // L] for i in range(N)])
    F_func = sp.lambdify([FEM.X, FEM.u, shear], sheared_F, "numpy")
    zero_u = np.zeros_like(ref_positions)
    dN_dX_func = sp.lambdify(FEM.X, dN_dX, "numpy")
    pos_func = sp.lambdify([FEM.X, FEM.u, shear], sheared_positions, "numpy")

    dN_dX_values = dN_dX_func(ref_positions)
    raw_pos = np.array([pos_func(ref_positions, zero_u, s) for s in shearValues])
    # Ensure pos shape is (n_shear, n_nodes, 2)
    pos = raw_pos.reshape(len(shearValues), -1, 2)
    F_values = np.array([F_func(ref_positions, zero_u, s) for s in shearValues])

    # Reshape
    n_shear = shearValues.shape[0]
    n_elements = len(elements)
    F_values = F_values.reshape(n_shear, n_elements, 2, 2)
    dN_dX_values = dN_dX_values.reshape(n_elements, 3, 2)

    return pos, elements, F_values, dN_dX_values


def calculateForcesAndEnergy(F_values, dN_dX):
    energies = ContiEnergy.energy_from_F(F_values)
    # Strain steps are in F_values, so we should tile dN_dX to match
    n_shear = F_values.shape[0]
    # Shape: (strainSteps, elements, nodes, 2)
    # dN_dX = np.tile(dN_dX, (n_shear, 1, 1, 1))
    force = ContiEnergy.lagrangian_forces_from_F(F_values, dN_dX)
    return energies, force


def plotEnergyAndForces(shearValues, energies, forces):
    # plt.plot(
    #     shearValues,
    #     energies,
    #     label=["element 1", "element 2", "element 3", "element 4"],
    # )
    # plt.legend()
    # plt.xlabel("Shear")
    # plt.ylabel("Energy")
    # plt.title("Energy vs Shear in Simple Shear Test")
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    # Forces shape: (strainSteps, elements, nodes, 2)
    n_shear, n_elements, n_nodes, _ = forces.shape
    for i in range(1):
        element_forces = forces[:, i, :, :]  # shape (n_shear, n_nodes, 2)
        fig, axes = plt.subplots(1, n_nodes, figsize=(15, 5))
        for j in range(n_nodes):
            axes[j].plot(shearValues, element_forces[:, j, 0], label="x-component")
            axes[j].plot(shearValues, element_forces[:, j, 1], label="y-component")
            axes[j].set_title(f"Node {j + 1} Forces in Element {i + 1}")
            axes[j].set_xlabel("Shear")
            axes[j].set_ylabel("Force")
            axes[j].grid()
            axes[j].legend()
    plt.tight_layout()
    plt.show()


def plotFValues(F):
    # F shape: (shear_steps, elements, 2, 2)
    n_shear = F.shape[0]
    n_elements = F.shape[1]
    fig, axes = plt.subplots(n_elements, 1, figsize=(10, 5 * n_elements))
    for i in range(n_elements):
        axes[i].plot(F[:, i, 0, 0], label="Fxx")
        axes[i].plot(F[:, i, 0, 1], label="Fxy")
        axes[i].plot(F[:, i, 1, 0], label="Fyx")
        axes[i].plot(F[:, i, 1, 1], label="Fyy")
        axes[i].set_title(f"Element {i + 1} Force Components vs Shear")
        axes[i].set_xlabel("Shear Step")
        axes[i].set_ylabel("Force Component")
        axes[i].grid()
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def _assert_elements_equal(stress, name, rtol=1e-9, atol=1e-12):
    if stress.shape[1] <= 1:
        return
    ref = stress[:, :1, :, :]
    if not np.allclose(stress, ref, rtol=rtol, atol=atol):
        diff = np.max(np.abs(stress - ref))
        raise AssertionError(f"{name} differs across elements; max |diff|={diff:.3e}")


def plotStressComponents(shearValues, F_values, element=0, assert_equal_elements=True):
    """
    Plot P_21, P_12 and sigma_21, sigma_12 vs load (shear).
    element: integer element index, or "mean" to average across elements.
    """
    # F_values shape: (n_shear, n_elements, 2, 2)
    P = ContiEnergy.P_from_F(F_values)
    sigma = ContiEnergy.cauchy_from_F(F_values)

    if assert_equal_elements:
        _assert_elements_equal(P, "P")
        _assert_elements_equal(sigma, "sigma")

    if element == "mean":
        P_use = P.mean(axis=1)
        sigma_use = sigma.mean(axis=1)
        title = "Stress components (mean over elements)"
    else:
        P_use = P[:, element, :, :]
        sigma_use = sigma[:, element, :, :]
        title = f"Stress components (element {element})"

    plt.figure(figsize=(10, 6))
    plt.plot(shearValues, P_use[:, 1, 0], label="P_21 (1st PK)")
    plt.plot(shearValues, P_use[:, 0, 1], label="P_12 (1st PK)", linestyle="--")
    plt.plot(shearValues, sigma_use[:, 1, 0], label="sigma_21 (Cauchy)", linestyle="--")
    plt.plot(shearValues, sigma_use[:, 0, 1], label="sigma_12 (Cauchy)", linestyle="--")
    plt.xlabel("Load (shear)")
    plt.ylabel("Stress")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def printStressComponentsAtLoads(loads, L=2, element=0, assert_equal_elements=True):
    """
    Recompute simpleShearSystem2 at exact loads and print P_21, P_12,
    sigma_21, sigma_12 for each load. Also prints the element-averaged
    off-diagonal values at each load.
    """
    shear = np.asarray(loads, dtype=float)
    _, _, F_values, _ = simpleShearSystem2(L=L, shearValues=shear)

    P = ContiEnergy.P_from_F(F_values)
    sigma = ContiEnergy.cauchy_from_F(F_values)

    if assert_equal_elements:
        _assert_elements_equal(P, "P")
        _assert_elements_equal(sigma, "sigma")

    if element == "mean":
        P_use = P.mean(axis=1)
        sigma_use = sigma.mean(axis=1)
        label = "mean over elements"
    else:
        P_use = P[:, element, :, :]
        sigma_use = sigma[:, element, :, :]
        label = f"element {element}"

    print(f"Stress components at exact loads ({label}):")
    def _zero_small(x, tol=1e-10):
        return 0.0 if abs(x) < tol else x

    # Element-averaged off-diagonal values for each load
    p_offdiag_mean = 0.5 * (P[:, :, 1, 0] + P[:, :, 0, 1]).mean(axis=1)
    s_offdiag_mean = 0.5 * (sigma[:, :, 1, 0] + sigma[:, :, 0, 1]).mean(axis=1)

    for i, load in enumerate(shear):
        p21 = _zero_small(P_use[i, 1, 0])
        p12 = _zero_small(P_use[i, 0, 1])
        s21 = _zero_small(sigma_use[i, 1, 0])
        s12 = _zero_small(sigma_use[i, 0, 1])
        p_avg = _zero_small(p_offdiag_mean[i])
        s_avg = _zero_small(s_offdiag_mean[i])
        print(
            f"load={load}: "
            f"P21={p21:.6g}, P12={p12:.6g}, "
            f"sigma21={s21:.6g}, sigma12={s12:.6g}, "
            f"P_avg={p_avg:.6g}, sigma_avg={s_avg:.6g}"
        )


def makeVideo(pos, forces, element_indices):
    # Use FEM.x to get the node positions
    n_shear, n_elements, n_local_nodes, _ = forces.shape
    import matplotlib.animation as animation

    # Pos is frames, nodes, 2
    # Forces is frames, elements, nodes, 2
    fig, ax = plt.subplots(figsize=(10, 6))
    print("max |force| =", np.max(np.linalg.norm(forces, axis=-1)))
    print("min |force| =", np.min(np.linalg.norm(forces, axis=-1)))

    def update(frame):
        ax.clear()

        ax.set_title(f"Shear Step: {frame + 1}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect("equal")
        ax.grid()
        # Plot node positions for the current shear step
        ax.scatter(pos[frame][:, 0], pos[frame][:, 1], color="blue")

        # Draw force vectors as arrows on top of node positions
        # Optional: scale forces for better visualization
        arrow_scale = 1  # Increase for shorter arrows, decrease for longer arrows
        for e_idx in range(n_elements):
            for local_idx in range(n_local_nodes):
                global_idx = element_indices[e_idx][local_idx]
                node_pos = pos[frame][global_idx]
                fx, fy = forces[frame, e_idx, local_idx]
                ax.quiver(
                    node_pos[0],
                    node_pos[1],
                    fx,
                    fy,
                    angles="xy",
                    scale_units="xy",
                    scale=arrow_scale,
                    color="red",
                    width=0.005,
                )

    ani = animation.FuncAnimation(fig, update, frames=n_shear, repeat=True)
    plt.show()


if __name__ == "__main__":
    shear = np.linspace(0, 3, 300)
    pos, elements, F, dN_dX = simpleShearSystem2(L=3, shearValues=shear)
    energies, forces = calculateForcesAndEnergy(F, dN_dX)
    # plotEnergyAndForces(shear, energies, forces)
    # plotFValues(F)
    # plotStressComponents(shear, F, element=0, assert_equal_elements=True)
    printStressComponentsAtLoads(
        [0.3, 0.5, 1.3, 1.5, 2.3, 2.5],
        L=2,
        element=0,
        assert_equal_elements=True,
    )
    element_indices = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [3, 0, 1],
    ]
    # makeVideo(pos, forces, element_indices)

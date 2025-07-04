import numpy as np
from SymbolicFEM import FEM
from contiPotential import ContiEnergy
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

    F = sp.Array([FEM.F(e) for e in elements])
    dN_dX = sp.Array([FEM.dN_dX(e) for e in elements])

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
def simpleShearSystem2(L=2, shearValues=np.linspace(0, 3, 100)):
    """
    Make a LxL system of nodes connected in a triangular mesh.
    Apply shear and calculate F using new FEM.Element abstraction.
    """
    N = L**2
    FEM.make_N_nodes(N)

    # Explicit triangular elements using node indices
    element_indices = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [3, 0, 1],
    ]
    elements = [FEM.Element(ids) for ids in element_indices]

    F = sp.Array([FEM.F(e) for e in elements])
    dN_dX = sp.Array([FEM.dN_dX(e) for e in elements])

    shear = sp.symbols("shear")

    # Apply shear to interpolated x field for each node, gather into matrix
    sheared_positions = sp.Matrix(
        [FEM.apply_shear(node["x"], shear) for node in FEM.nodes]
    )

    # Lambdify evaluation functions
    ref_positions = np.array([[i % L, i // L] for i in range(N)])
    F_func = sp.lambdify([FEM.X, FEM.u, shear], F, "numpy")
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
    force = ContiEnergy.forces_from_F(F_values, dN_dX)
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


def makeVideo(pos, forces):
    # Use FEM.x to get the node positions
    n_shear, n_elements, n_nodes, _ = forces.shape
    import matplotlib.animation as animation

    # Pos is frames, nodes, 2
    # Forces is frames, elements, nodes, 2
    fig, ax = plt.subplots(figsize=(10, 6))

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
        arrow_scale = 0.00001  # Increase for shorter arrows, decrease for longer arrows
        for e_idx in range(n_elements):
            for n_idx in range(n_nodes):
                node_pos = pos[frame][n_idx]
                fx, fy = forces[frame][e_idx][n_idx]
                ax.quiver(
                    node_pos[0],
                    node_pos[1],
                    fx,
                    fy,
                    angles="xy",
                    scale_units="xy",
                    scale=arrow_scale,
                    color="red",
                    width=0.01,
                )

    ani = animation.FuncAnimation(fig, update, frames=n_shear, repeat=True)
    plt.show()


if __name__ == "__main__":
    shear = np.linspace(0, 1, 30)
    pos, elements, F, dN_dX = simpleShearSystem2(L=2, shearValues=shear)
    energies, forces = calculateForcesAndEnergy(F, dN_dX)
    # plotEnergyAndForces(shear, energies, forces)
    # plotFValues(F)
    makeVideo(pos, forces)

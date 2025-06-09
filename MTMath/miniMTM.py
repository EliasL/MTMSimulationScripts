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
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    ani = animation.FuncAnimation(fig, update, frames=n_shear, repeat=True)
    plt.show()


if __name__ == "__main__":
    shear = np.linspace(0, 3, 100)
    pos, elements, F, dN_dX = simpleShearSystem(L=2, shearValues=shear)
    energies, forces = calculateForcesAndEnergy(F, dN_dX)
    # plotEnergyAndForces(shear, energies, forces)
    # plotFValues(F)
    makeVideo(pos, forces)

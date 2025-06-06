import numpy as np
from SymbolicFEM import makeNode, set_reference_positions, compute_F
from contiPotential import ContiEnergy
from matplotlib import pyplot as plt
import sympy as sp


def apply_shear(nodes, shear):
    for node in nodes:
        x, y = node["X"]
        node["u"] = [shear * y, 0]


def simpleShear(L=2, shearValues=np.linspace(0, 1, 10)):
    """
    Make a LxL system of nodes connected in a triangular mesh.
    Apply shear and calculate F
    """

    # Create nodes (L**2)
    nodes = [makeNode(f"{i}") for i in range(L * L)]

    # Create elements (list of three nodes)
    # We cheat for now, and force L=2
    elements = [[nodes[0], nodes[1], nodes[2]], [nodes[1], nodes[2], nodes[3]]]

    # Set reference positions.
    positions = np.array([[i % L, i // L] for i in range(L * L)])
    # Create symbolic shear variable
    shear = sp.symbols("shear")
    apply_shear(nodes, shear)
    F = [compute_F(e) for e in elements]

    F = sp.Matrix(F)
    set_reference_positions(F, nodes, positions)
    # Create a numeric function of F for a given shear value
    f_F = sp.lambdify(shear, F, "numpy")
    # Compute energy for each shear value
    energies = []
    for sv in shearValues:
        F_val = f_F(sv)
        energy_val = ContiEnergy.energy_from_F(F_val)
        energies.append(energy_val)
    energies = np.array(energies)

    plt.plot(shearValues, energies, label="Energy")
    plt.xlabel("Shear")
    plt.ylabel("Energy")
    plt.title("Energy vs Shear in Simple Shear Test")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simpleShear()
    # plt.show()  # Uncomment if you want to display the plot interactively

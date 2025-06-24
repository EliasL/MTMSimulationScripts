from contiPotential import ContiEnergy
from matplotlib import pyplot as plt
import numpy as np


def plot_energy():
    # Generate strain arrays
    strain = np.linspace(0.0, 1, 100)

    # Compute energies
    e = ContiEnergy.energy_from_simpleShear(strain)

    # Create figure and axis for energy
    fig, ax = plt.subplots()
    ax.plot(strain, e, label="Energy")
    ax.set_xlabel(r"$\gamma$ (Strain)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy in Simple Shear")
    ax.legend()

    # Display or save as needed
    fig.tight_layout()


def plot_forces():
    # Generate strain array
    strain = np.linspace(0.0, 5, 1000)

    # Define dN_dX matrix for three nodes, tiled over strain length
    dN_dX = np.array([[-1, -1], [1, 0], [0, 1]])
    dN_dX = np.tile(dN_dX, (len(strain), 1, 1))

    # Compute forces: assume output shape is (len(strain), 3, 2)
    forces = ContiEnergy.forces_from_simpleShear(strain, dN_dX)

    # Create figure and axes for forces: a row of 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for i, ax in enumerate(axes):
        # Plot x and y force components for node i
        ax.plot(strain, forces[:, i, 0], label="$f_x$")
        ax.plot(strain, forces[:, i, 1], label="$f_y$", linestyle="--")
        ax.set_xlabel(r"$\gamma$ (Strain)")
        if i == 0:
            ax.set_ylabel("Force")
        ax.set_title(f"Node {i + 1}")
        ax.legend()

    fig.suptitle("Forces on Nodes in Simple Shear")
    fig.tight_layout()
    # print("Sum of forces:", np.sum(forces, axis=1))


if __name__ == "__main__":
    plot_energy()
    # plot_forces()
    plt.show()

from MTMath.plotEnergy import generate_energy_grid, OneDPotential
from Plotting.makeEnergyField import plotEnergyField

# g = generate_energy_grid()
# plotEnergyField(g)
OneDPotential()
# temp
import matplotlib.pyplot as plt

# Create the figure and subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Define the range of lattice points
lattice_range = range(-1, 2)

# Define vectors e1 and three versions of e2
e1 = (1, 0)
e2_list = [(-1, 1), (0, 1), (1, 1)]  # Different cases for e2

# Labels for each plot
labels = [r"$\mathbf{e_1}$", r"$\mathbf{e_2}$"]

# Loop through subplots and vectors
for i, e2 in enumerate(e2_list):
    ax = axes[i]

    # Add lattice grid with points
    for x in lattice_range:
        for y in lattice_range:
            ax.plot(x, y, "ko")  # Black dots for lattice points

    # Add vectors e1 and e2
    ax.quiver(
        0,
        0,
        e1[0],
        e1[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        headwidth=5,
    )  # e1
    ax.quiver(
        0,
        0,
        e2[0],
        e2[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        headwidth=5,
    )  # e2

    # Label the vectors
    ax.text(0.5, 0.1, r"$\mathbf{e_1}$", fontsize=40)
    ax.text(
        e2[0] / 2 + 0.1 + (0.1 if e2[0] == 1 else 0),
        e2[1] / 2,
        r"$\mathbf{e_2}$",
        fontsize=40,
    )

    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)

    # Add gridlines aligned with lattice points
    ax.set_xticks(lattice_range)
    ax.set_yticks(lattice_range)
    ax.grid(True, which="both", linestyle="--", color="gray")

    # Hide the spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Hide the tick labels (but keep the ticks)
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels

    # Set aspect ratio to be equal
    ax.set_aspect("equal")

# Show the plot
plt.tight_layout()
plt.show()

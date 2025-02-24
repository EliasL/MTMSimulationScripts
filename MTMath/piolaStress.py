import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Define the reference triangle (nodes in reference configuration)
nodes = np.array(
    [[0, 0], [1, 0], [0, 1]]
)  # Triangle with vertices at (0,0), (1,0), (0,1)
element_area = 0.5  # Area of reference triangle

# Shape function gradients in reference configuration (constant for linear triangle)
grad_N = np.array([[-1, -1], [1, 0], [0, 1]])


# Define a more interesting energy density function (e.g., nonlinear function of strain)
def energy_density(F):
    C = np.dot(F.T, F)  # Right Cauchy-Green tensor
    return 0.5 * (np.trace(C) + np.linalg.det(C))  # Nonlinear energy function


# Define a deformation gradient F (assumed varying for visualization)
F_base = np.array([[1.2, 0.5], [30.1, 0.3]])  # Base deformation gradient

# Compute the Piola stress tensor (simplified derivative of energy function)
P = np.dot(F_base, np.eye(2))  # Assuming dW/dF = F for a simple material model

# Compute nodal forces
nodal_forces = np.array([element_area * np.dot(grad.T, P) for grad in grad_N])
print(np.sum(nodal_forces, 0))
# Generate a grid for visualization
x = np.linspace(-0.2, 1.2, 30)
y = np.linspace(-0.2, 1.2, 30)
X, Y = np.meshgrid(x, y)
energy_map = np.zeros_like(X)

# Evaluate energy density function over the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        F_var = F_base + np.array(
            [[0.2 * X[i, j], 0.0], [0.0, 0.1 * Y[i, j]]]
        )  # Varying deformation gradient based on grid location
        energy_map[i, j] = energy_density(F_var)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Plot energy density as a heatmap
ax.contourf(X, Y, energy_map, levels=20, cmap="coolwarm", alpha=0.5)

# Plot the triangle
triang = tri.Triangulation(nodes[:, 0], nodes[:, 1])
ax.triplot(triang, "k-", linewidth=1)

# Plot the nodes
ax.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=3, label="Nodes")

# Plot forces acting on reference-aligned surfaces
reference_normals = [np.array([1, 0]), np.array([0, 1])]  # Reference X and Y directions
for i, normal in enumerate(reference_normals):
    force_vector = np.dot(P, normal)  # Force per unit reference area
    ax.arrow(
        0.5,
        0.5,
        force_vector[0] * 0.1,
        force_vector[1] * 0.1,
        color="blue",
        head_width=0.02,
        label="Force on Ref. Surface" if i == 0 else "",
    )

# Plot the nodal forces as red arrows
for i, node in enumerate(nodes):
    ax.arrow(
        node[0],
        node[1],
        nodal_forces[i, 0] * 0.1,
        nodal_forces[i, 1] * 0.1,
        color="red",
        head_width=0.03,
        label="Nodal Force" if i == 0 else "",
    )

ax.set_xlim([-0.2, 1.2])
ax.set_ylim([-0.2, 1.2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Energy Density, Piola Stress (Ref. Surfaces), and Nodal Forces")
ax.legend()
plt.show()

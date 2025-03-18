import matplotlib.pyplot as plt
import numpy as np
from contiPotential import ContiEnergy


class ElementGrid:
    def __init__(self, ref_nodes, displacements, displacementGrid):
        """Initialize the element with reference node positions."""
        assert ref_nodes.shape == (3, 2), "Reference nodes must be (3,2)"
        assert displacements.shape == (3, 2), "Displacements must be (3,2)"

        X, Y = displacementGrid.shape[:2]
        assert displacementGrid.shape == (X, Y, 2), "DisplacementGrid must be (X,Y,2)"

        self.ref_nodes = ref_nodes
        self.disp = displacements
        self.current_nodes = ref_nodes + displacements

        # Grid of displaced node positions
        # Grid X, grid Y, node, u/v
        self.disp_grid = (
            displacements[np.newaxis, np.newaxis, :, :]
            + displacementGrid[:, :, np.newaxis, :]
        )

        # Grid X, grid Y, node, x/y
        self.pos_grid = ref_nodes[np.newaxis, np.newaxis, :, :] + self.disp_grid

    def get_deformation_gradient(self, referenceNode=0, selectedNodes=[0]):
        """Compute deformation gradient tensor for a grid of node positions."""
        otherNodes = np.delete(np.arange(3), referenceNode)

        # Reference shape function gradients (constant)
        dX_dxi = np.zeros((2, 2))
        dX_dxi[:, 0] = self.ref_nodes[otherNodes[0]] - self.ref_nodes[referenceNode]
        dX_dxi[:, 1] = self.ref_nodes[otherNodes[1]] - self.ref_nodes[referenceNode]

        dX_dxi = self.ref_nodes[otherNodes] - self.ref_nodes[referenceNode]

        # We calculate du_dxi by looking at the difference in displacements,
        # but we also purturbe the reference node by the displacement grid
        # to see many F by how it would be

        # Compute displacement differences, ensuring that they follow the single-point logic
        du_dxi = np.zeros((*self.disp_grid.shape[:2], 2, 2))  # (X, Y, 2, 2)
        # Compute per-column displacement differences similar to single-point version
        for i in range(2):
            if otherNodes[i] in selectedNodes:
                du_dxi[..., i] = (
                    self.disp_grid[:, :, otherNodes[i], :]
                    - self.disp_grid[:, :, referenceNode, :]
                )
            else:
                du_dxi[..., i] = (
                    self.disp[np.newaxis, np.newaxis, otherNodes[i], :]
                    - self.disp_grid[:, :, referenceNode, :]
                )
        assert du_dxi.shape == (*self.disp_grid.shape[:2], 2, 2)

        # Compute inverse only once since dX_dxi is constant
        # Don't ask me why we need the transpose. I think it is because of the
        # way we do the matrix multiplication with einsum, but feel like it should
        # not be needed. I think it has something to do with how numpy stores
        # the rows and columns.
        dX_dxi_inv = np.linalg.inv(dX_dxi).T
        # Compute deformation gradient for each grid point
        F = np.eye(2)[np.newaxis, np.newaxis, :, :] + np.einsum(
            "...ij,jk->...ik", du_dxi, dX_dxi_inv
        )
        return F


def plot_triangle(ax, element, showReferenceState=False):
    """Draws the reference and current state of the triangle."""
    curr_points = np.vstack([element.current_nodes, element.current_nodes[0]])
    ax.plot(
        curr_points[:, 0], curr_points[:, 1], "bo-", markersize=8, label="Current State"
    )

    if showReferenceState:
        ref_points = np.vstack([element.ref_nodes, element.ref_nodes[0]])
        ax.plot(
            ref_points[:, 0],
            ref_points[:, 1],
            "r--",
            alpha=0.5,
            label="Reference State",
        )


def highlight_selected_nodes(ax, element, referenceNode=0, selectedNodes=[0]):
    """Draws a circle around the selected reference node."""

    label = "Selected Node"
    if len(selectedNodes) > 1:
        label += "s"
    ax.scatter(
        element.current_nodes[selectedNodes, 0],
        element.current_nodes[selectedNodes, 1],
        s=100,
        color="red",
        zorder=3,
        label=label,
    )
    ax.scatter(
        element.current_nodes[referenceNode, 0],
        element.current_nodes[referenceNode, 1],
        s=100,
        marker="+",
        color="green",  # Green edge color
        zorder=4,
        label="Reference Node",
    )


def plot_2x2mat_quiver(ax, pos_grid, mat_grid, referenceNode=0, name="M"):
    """Plot the deformation gradient as vectors using quiver."""
    if name == "C_":
        name = r"\tilde{C}"

    X = pos_grid[:, :, referenceNode, 0]  # X-coordinates of reference node
    Y = pos_grid[:, :, referenceNode, 1]  # Y-coordinates of reference node

    if name == "F":
        # Get first columns
        v1 = mat_grid[..., :, 0]  # Deformation in x-direction
        # Get second columns
        v2 = mat_grid[..., :, 1]  # Deformation in y-direction
        v1Label = rf"$\mathbf{{{name}}}_x$"
        v2Label = rf"$\mathbf{{{name}}}_y$"
    elif "C" in name:
        # Get length 1
        length1 = mat_grid[..., 0, 0]
        # Get length 2
        length2 = mat_grid[..., 1, 1]
        # Get angle
        dot = mat_grid[..., 0, 1]  # Dot product between vectors

        # Now we try to reconstruct the dx and dy vectors
        theta = np.arccos(np.clip(dot / (length1 * length2), -1, 1))  # Angle in radians

        # dx_vec is always along the x-axis
        v1 = np.stack([length1, np.zeros_like(length1)], axis=-1)

        # dy_vec computed using angle
        v2 = np.stack([length2 * np.cos(theta), length2 * np.sin(theta)], axis=-1)
        v1Label = rf"$\mathbf{{{name}}}_{{11}}$"
        v2Label = rf"$\mathbf{{{name}}}_{{22}}$"

    ax.quiver(
        X,
        Y,
        v1[..., 0],
        v1[..., 1],
        scale=10,
        color="g",
        alpha=0.6,
        label=v1Label,
    )
    ax.quiver(
        X,
        Y,
        v2[..., 0],
        v2[..., 1],
        scale=10,
        color="m",
        alpha=0.6,
        label=v2Label,
    )

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(rf"$\mathbf{{{name}}}$ Vector Field")


def matshow_energy_grid(
    ax,
    element,
    energy_grid,
    referenceNode=0,
    clim=(0, 0.5),
    title="",
    addTriangle=True,
    showReferenceState=False,
):
    """
    Plots the energy associated with each grid point as a background color map.
    """
    X = element.pos_grid[:, :, referenceNode, 0]  # X-coordinates of grid
    Y = element.pos_grid[:, :, referenceNode, 1]  # Y-coordinates of grid

    c = ax.pcolormesh(X, Y, energy_grid, shading="auto", cmap="coolwarm")
    plt.colorbar(c, ax=ax, label="Energy")
    # set the max and min of the colorbar to the max and min of the energy grid
    c.set_clim(*clim)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)

    if addTriangle:
        plot_triangle(ax, element, showReferenceState)


def show_2x2mat_grid(ax, referenceNode, element, mat_grid, name="M"):
    # Make sparce F_grid
    num_samples = 5  # Number of evenly spaced points (including first and last)
    indices = np.linspace(0, len(mat_grid) - 1, num_samples, dtype=int)

    sparce_mat_grid = mat_grid[np.ix_(indices, indices)]
    sparce_pos_grid = element.pos_grid[np.ix_(indices, indices)]
    plot_2x2mat_quiver(ax, sparce_pos_grid, sparce_mat_grid, referenceNode, name)
    plot_triangle(ax, element, showReferenceState=True)


def plot_combined(element, F_grid, beta=-1 / 4, referenceNode=0, selectedNodes=[0]):
    """Plot the triangle along with the deformation gradient field."""

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    C = np.einsum("...ji,...jk->...ik", F_grid, F_grid)

    gridWithoutVolumetricPart, C_ = ContiEnergy.energy_from_F(
        F_grid, beta, K=0, zeroReference=True, returnReducedC=True
    )
    energyGrid = ContiEnergy.energy_from_reduced_C(C_, beta, K=4, zeroReference=True)

    onlyVolumetricPart = energyGrid - gridWithoutVolumetricPart

    matshow_energy_grid(
        axs[0, 0],
        element,
        gridWithoutVolumetricPart,
        referenceNode,
        clim=(0, 0.7),
        title="Without Volumetric Part",
    )
    matshow_energy_grid(
        axs[0, 1],
        element,
        onlyVolumetricPart,
        referenceNode,
        clim=(0, 5),
        title="Volumetric Part",
    )

    matshow_energy_grid(
        axs[0, 2],
        element,
        energyGrid,
        referenceNode,
        clim=(0, 0.7),
        title="Combined",
    )

    show_2x2mat_grid(axs[1, 0], referenceNode, element, F_grid, name="F")
    show_2x2mat_grid(axs[1, 1], referenceNode, element, C, name="C")
    show_2x2mat_grid(axs[1, 2], referenceNode, element, C_, name="C_")

    for ax in axs.flat:
        highlight_selected_nodes(ax, element, referenceNode, selectedNodes)
        ax.set_aspect("equal")
        # ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    nodes = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    displacements = np.array(
        [
            [0, 0],
            [0, 0],
            [-1, 0],
        ]
    )

    beta = -0.25  # Square
    # beta = 4  # Triangle
    xr = 2.1
    yr = 2.1
    xResolution = 200
    yResolution = max(int(xResolution * yr / xr), 2)
    x_range = (-xr, xr)
    y_range = (-yr, yr)
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_range[0], x_range[1], xResolution),
        np.linspace(y_range[0], y_range[1], yResolution),
    )
    displacementGrid = np.stack([x_vals, y_vals], axis=-1)
    element = ElementGrid(nodes, displacements, displacementGrid)

    refNode = 0
    selectedNodes = [refNode, 1]
    # Generate deformation gradients for node 0 over a grid
    F_grid = element.get_deformation_gradient(
        referenceNode=refNode, selectedNodes=selectedNodes
    )

    # Plot combined visualization
    plot_combined(
        element, F_grid, beta=beta, referenceNode=refNode, selectedNodes=selectedNodes
    )

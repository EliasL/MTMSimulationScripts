import matplotlib.pyplot as plt
import numpy as np


class Element:
    def __init__(self, ref_nodes):
        """Initialize the element with reference node positions."""
        self.ref_nodes = np.array(ref_nodes)
        self.curr_nodes = np.array(ref_nodes)

    def set_current_state(self, curr_nodes):
        """Update the current state of the element."""
        self.curr_nodes = np.array(curr_nodes)

    def set_displacements(self, displacements):
        """Update the current state of the element."""
        self.curr_nodes = self.ref_nodes + np.array(displacements)

    def get_displacements(self):
        """Return the displacement vectors of the nodes."""
        return self.curr_nodes - self.ref_nodes

    def get_deformation_gradient(self, referenceNode=0):
        """Return the deformation gradient tensor."""
        disp = self.get_displacements()
        print("disp\n", disp)
        otherNodes = np.delete(range(3), referenceNode)
        du_dxi = np.zeros((2, 2))
        du_dxi[:, 0] = disp[otherNodes[0]] - disp[referenceNode]
        du_dxi[:, 1] = disp[otherNodes[1]] - disp[referenceNode]
        print("du_dxi\n", du_dxi)
        dX_dxi = np.zeros((2, 2))
        dX_dxi[:, 0] = self.ref_nodes[otherNodes[0]] - self.ref_nodes[referenceNode]
        dX_dxi[:, 1] = self.ref_nodes[otherNodes[1]] - self.ref_nodes[referenceNode]
        print("dX_dxi\n", dX_dxi)
        print("dX_dxi_inv\n", np.linalg.inv(dX_dxi))
        F = np.eye(2) + du_dxi @ np.linalg.inv(dX_dxi)
        return F


def plot_triangle(ax, element):
    """Draws a triangle onto the provided axis, showing both reference and current states."""
    ref_points = np.vstack([element.ref_nodes, element.ref_nodes[0]])
    curr_points = np.vstack([element.curr_nodes, element.curr_nodes[0]])

    ax.plot(
        ref_points[:, 0], ref_points[:, 1], "r--", alpha=0.5, label="Reference State"
    )
    ax.plot(
        curr_points[:, 0], curr_points[:, 1], "bo-", markersize=8, label="Current State"
    )

    # Annotate current state points
    for i, (x, y) in enumerate(element.curr_nodes):
        ax.text(
            x + 0.15,
            y - 0.1,
            rf"$P_{i + 1}$",
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
        )

    ax.set_xlim(
        min(ref_points[:, 0].min(), curr_points[:, 0].min()) - 1,
        max(ref_points[:, 0].max(), curr_points[:, 0].max()) + 1,
    )
    ax.set_ylim(
        min(ref_points[:, 1].min(), curr_points[:, 1].min()) - 1,
        max(ref_points[:, 1].max(), curr_points[:, 1].max()) + 1,
    )

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def draw_F(ax, F, node_zero):
    """Draws the deformation gradient vectors (columns of F) from the specified reference point."""
    dx_vec = F[:, 0]
    dy_vec = F[:, 1]

    ax.quiver(
        node_zero[0],
        node_zero[1],
        dx_vec[0] / 2,
        dx_vec[1] / 2,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="g",
        label=r"$\mathbf{d}_x$",
    )
    ax.quiver(
        node_zero[0],
        node_zero[1],
        dy_vec[0],
        dy_vec[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="m",
        label=r"$\mathbf{d}_y$",
    )


# Example usage
# element = Element([(0, 0), (1, 0), (0, 1)])
# element.set_current_state([(0, 0), (1, 0.2), (1, 1)])

element = Element([(0, 0), (1, 0), (0, 1)])
element.set_displacements([(0, 0), (0, 0), (0, 0)])

fig, ax = plt.subplots(figsize=(6, 6))
plot_triangle(ax, element)

for i in range(2, 3):
    F = element.get_deformation_gradient(referenceNode=i)
    print(F)
    draw_F(ax, F, element.curr_nodes[i])

plt.show()

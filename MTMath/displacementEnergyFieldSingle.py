import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


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
        # print("disp\n", disp)
        otherNodes = np.delete(range(3), referenceNode)
        du_dxi = np.zeros((2, 2))
        du_dxi[:, 0] = disp[otherNodes[0]] - disp[referenceNode]
        du_dxi[:, 1] = disp[otherNodes[1]] - disp[referenceNode]
        # print("du_dxi\n", du_dxi)
        dX_dxi = np.zeros((2, 2))
        dX_dxi[:, 0] = self.ref_nodes[otherNodes[0]] - self.ref_nodes[referenceNode]
        dX_dxi[:, 1] = self.ref_nodes[otherNodes[1]] - self.ref_nodes[referenceNode]
        # print("dX_dxi\n", dX_dxi)
        # print("dX_dxi_inv\n", np.linalg.inv(dX_dxi))
        F = np.eye(2) + du_dxi @ np.linalg.inv(dX_dxi)
        return F


def plot_triangle(ax, element):
    """Draws a triangle onto the provided axis, showing both reference and current states."""
    ref_points = np.vstack([element.ref_nodes, element.ref_nodes[0]])
    curr_points = np.vstack([element.curr_nodes, element.curr_nodes[0]])

    ax.plot(
        ref_points[:, 0],
        ref_points[:, 1],
        "r--",
        alpha=0.5,
        label=r"Reference State $\mathbf{X}$",
    )
    ax.plot(
        curr_points[:, 0],
        curr_points[:, 1],
        "b-",
        markersize=8,
        label=r"Current State $\mathbf{x}$",
    )

    # # Annotate current state points
    # for i, (x, y) in enumerate(element.curr_nodes):
    #     ax.text(
    #         x + 0.15,
    #         y - 0.1,
    #         rf"$P_{i + 1}$",
    #         fontsize=12,
    #         verticalalignment="top",
    #         horizontalalignment="right",
    #     )

    padding = 0.4

    ax.set_xlim(
        min(ref_points[:, 0].min(), curr_points[:, 0].min()) - padding,
        max(ref_points[:, 0].max(), curr_points[:, 0].max()) + padding,
    )
    ax.set_ylim(
        min(ref_points[:, 1].min(), curr_points[:, 1].min()) - padding,
        max(ref_points[:, 1].max(), curr_points[:, 1].max()) + padding,
    )

    ax.set_aspect("equal")


def draw_F(ax, F, nodes):
    """Draws the deformation gradient vectors (columns of F) from the specified reference point."""
    dx_vec = F[:, 0]
    dy_vec = F[:, 1]

    # Get center of all nodes
    cetner = np.mean(nodes, axis=0)

    ax.quiver(
        cetner[0],
        cetner[1],
        dx_vec[0],
        dx_vec[1],
        angles="xy",
        scale_units="xy",
        scale=3,
        color="g",
        label=r"$\mathbf{f}_1$",
    )
    ax.quiver(
        cetner[0],
        cetner[1],
        dy_vec[0],
        dy_vec[1],
        angles="xy",
        scale_units="xy",
        scale=3,
        color="m",
        label=r"$\mathbf{f}_2$",
    )
    # ax.legend()  # Removed legend from here; will be handled elsewhere


def create_simple_triangle_F_animation(
    output_path="triangle_F_animation.mp4", n_frames=100, fps=20
):
    """Create an MP4 animation of the triangle deformation and deformation gradient."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    element = Element([(0, 0), (1, 0), (0, 1)])

    def update(frame):
        ax.clear()
        t = frame / (n_frames - 1)
        displacements = [(0, 0), (0, 0), (t, 0)]
        element.set_displacements(displacements)
        plot_triangle(ax, element)
        F = element.get_deformation_gradient()
        draw_F(ax, F, element.curr_nodes)
        ax.legend()

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    writer = FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)


def create_strange_triangle_F_animation(
    output_path="triangle_F_animation.mp4", n_frames_pr_move=100, fps=20
):
    """Create an MP4 animation of the triangle deformation and deformation gradient."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    element = Element([(0, 0), (1, 0), (1, -1)])

    # We create a list of displacements that will be applied to the nodes

    # We first move the third node to the left, then we move the second node down
    currentDisp = np.array([(0, 0), (0, 0), (0, 0)], dtype=float)
    move1 = np.array([(0, 0), (0, 0), (-1, 0)], dtype=float)
    move2 = np.array([(0, 0), (0, -1), (0, 0)], dtype=float)
    moves = [move1, move2]
    displacements = [currentDisp.copy()]
    for move in moves:
        for i in range(n_frames_pr_move):
            # Now we interpolate between these displacements
            t = np.linalg.norm(move) / (n_frames_pr_move)
            currentDisp += move * t
            displacements.append(currentDisp.copy())

    def update(frame, displacements):
        ax.clear()
        element.set_displacements(displacements[frame])
        plot_triangle(ax, element)
        F = element.get_deformation_gradient()
        draw_F(ax, F, element.curr_nodes)
        ax.legend()

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames_pr_move * len(moves) + 1,
        fargs=(displacements,),
        interval=1000 / fps,
    )
    writer = FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer)


if __name__ == "__main__":
    # Example usage of the animation function
    create_strange_triangle_F_animation(
        output_path="Plots/strange_triangle_F_animation.mp4",
        n_frames_pr_move=60,
        fps=30,
    )
    create_simple_triangle_F_animation(
        output_path="Plots/triangle_F_animation.mp4",
        n_frames=60,
        fps=30,
    )
    # # Example usage
    # # element = Element([(0, 0), (1, 0), (0, 1)])
    # # element.set_current_state([(0, 0), (1, 0.2), (1, 1)])

    # element = Element([(0, 0), (1, 0), (0, 1)])
    # element.set_displacements([(0, 0), (0, 0), (1, 0)])

    # fig, ax = plt.subplots(figsize=(6, 6))
    # plot_triangle(ax, element)

    # F = element.get_deformation_gradient()
    # print(F)
    # draw_F(ax, F, element.curr_nodes)

    # plt.show()

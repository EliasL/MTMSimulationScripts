import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Default camera settings
DEFAULT_ELEV = 10
DEFAULT_AZIM = 30


# --- MetricGeometry class for managing surfaces, lines, points ---
class MetricGeometry:
    def __init__(self):
        # Each surface is a (..., 3) array representing [C11, C22, C12]
        self.surfaces = []
        # Each line is a (N, 3) array [C11, C22, C12]
        self.lines = []
        # Each point is a (3,) array [C11, C22, C12]
        self.points = []
        self.styles = []

    def add_surface(self, C):
        """C should be (..., 3) array with [C11, C22, C12] ordering"""
        self.surfaces.append(C)

    def add_line(self, C):
        """C should be (N, 3) array with [C11, C22, C12] ordering"""
        self.lines.append(C)

    def add_point(self, C):
        """C should be (3,) array with [C11, C22, C12] ordering"""
        self.points.append(C)

    def drawC(self, C, C12=None, C22=None, **kwargs):
        # Accepts either a (N, 3) array or (N, 2, 2) array, or C, C12, C22 as separate vectors
        import numpy as np

        if C12 is not None and C22 is not None:
            # Assuming C is C11 here (vector), stack to full matrix in [C11, C22, C12] order
            C = np.array([C, C22, C12])
            C = np.stack([C[0], C[1], C[2]], axis=-1)
            C = C.transpose(1, 0)
            # Now shape (N, 3) as [C11, C22, C12]
        # If C is (N, 3), convert to (N, 2, 2)
        if C.shape[-1] == 3:
            C11 = C[..., 0]
            C22 = C[..., 1]
            C12 = C[..., 2]
            Cmat = np.zeros(C.shape[:-1] + (2, 2))
            Cmat[..., 0, 0] = C11
            Cmat[..., 0, 1] = C12
            Cmat[..., 1, 0] = C12
            Cmat[..., 1, 1] = C22
        elif C.shape[-2:] == (2, 2):
            Cmat = C
        else:
            raise ValueError("drawC: C must be (N,3) or (N,2,2)")

        # Convert back to [C11, C22, C12] order for line
        C11 = Cmat[..., 0, 0]
        C22 = Cmat[..., 1, 1]
        C12 = Cmat[..., 0, 1]
        path = np.stack([C11, C22, C12], axis=-1)
        self.add_line(path)
        self.styles.append(kwargs)

    def drawAllVariations(self, C, depth=0, **kwargs):
        nr = len(C)
        one = np.array([1] * nr)
        zero = np.array([0] * nr)

        m1 = np.array([[one, zero], [zero, -one]]).transpose(2, 0, 1)
        m2 = np.array([[zero, one], [one, zero]]).transpose(2, 0, 1)
        m3 = np.array([[one, -one], [zero, one]]).transpose(2, 0, 1)

        def up(C):
            return conTrans(C, m3)

        def right(C):
            return conTrans(C, m3.transpose(0, 2, 1))

        self.drawC(C, **kwargs)
        self.drawC(conTrans(C, m1), **kwargs)
        self.drawC(conTrans(C, m2), **kwargs)
        self.drawC(conTrans(conTrans(C, m1), m2), **kwargs)

        if depth > 0:
            self.drawAllVariations(up(C), depth - 1, **kwargs)
            self.drawAllVariations(right(C), depth - 1, **kwargs)

    def drawCircle(self, r=1):
        # Draw a circle in metric space: C11 = C22 = r, C12 = 0
        nr = 200
        C11 = r * np.ones(nr)
        C22 = r * np.ones(nr)
        C12 = np.zeros(nr)
        C = np.stack([C11, C22, C12], axis=-1)
        self.add_line(C)

    def drawF(self, F11, F12, F21, F22, width=1, color="#222", zValue=-2):
        # Draw a line for the deformation gradient F (F11, F12; F21, F22)
        # Fij: arrays of shape (N,)
        F = np.stack(
            [
                np.stack([F11, F12], axis=-1),
                np.stack([F21, F22], axis=-1),
            ],
            axis=-2,
        )  # shape (N, 2, 2)
        # Compute C = F^T F
        C = np.einsum("...ji,...jk->...ik", F, F)
        # Convert to (N, 3) in [C11, C22, C12] order
        C_flat = np.stack([C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]], axis=-1)
        self.add_line(C_flat)

    def drawFundamentalDomain(self):
        nr = 100
        zero = np.zeros(nr)
        # VERTICAL LINE
        t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
        sqrt_term = np.sqrt(t**2 - 1)
        C_V_P = np.array([[t, sqrt_term], [sqrt_term, t]]).transpose(2, 0, 1)
        self.drawC(C_V_P)

        # HORIZONTAL LINE
        t = np.sinh(np.linspace(np.arcsinh(1e-1), np.arcsinh(1), nr))
        C_H = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
        self.drawC(C_H)

        # FUNDAMENTAL DOMAIN
        t = np.sinh(np.linspace(np.arcsinh(1e-1), np.arcsinh(2 / np.sqrt(3)), nr))
        C_F = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)
        self.drawC(C_F)

    def drawMetricSpaceBackground(self, depth=5):
        # Draw lines
        self.drawCircle(1)
        nr = 1000
        one = np.ones(nr)
        zero = np.zeros(nr)

        # Shearing circles, now t extends further in the negative direction
        t = np.sinh(np.linspace(np.arcsinh(-300), np.arcsinh(300), nr))

        # # Shearing circles
        # self.drawF(one, zero, t, one, width=1, color="#222", zValue=-2)
        # self.drawF(one, t, zero, one, width=1, color="#222", zValue=-2)

        # VERTICAL LINE
        t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
        sqrt_term = np.sqrt(t**2 - 1)
        C_V_P = np.array([[t, sqrt_term], [sqrt_term, t]]).transpose(2, 0, 1)
        C_V_N = np.array([[t, -sqrt_term], [-sqrt_term, t]]).transpose(2, 0, 1)
        self.drawAllVariations(C_V_P, depth, linestyle="--")
        self.drawAllVariations(C_V_N, depth, linestyle="--")

        # HORIZONTAL LINE
        t = np.sinh(np.linspace(np.arcsinh(1e-7), np.arcsinh(1), nr))
        C_H = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
        self.drawAllVariations(C_H, depth, linestyle="--")

        # FUNDAMENTAL DOMAIN
        t = np.sinh(np.linspace(np.arcsinh(1e-7), np.arcsinh(2 / np.sqrt(3)), nr))
        C_F = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)
        self.drawAllVariations(C_F, depth)


def conTrans(A, m):
    """
    Apply congruence transform: m^T @ A @ m for each matrix in A, m.
    A: (..., 2, 2)
    m: (..., 2, 2)
    Returns: (..., 2, 2)
    """
    return np.einsum("...ji,...jk,...kl->...il", m, A, m)


def plot_surface_3d(C11, C22, C12, xlim, ylim, zlim, color=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Ensure C12 is a masked array for smooth edge masking (z axis)
    C12_masked = np.ma.masked_invalid(C12)
    if color is None:
        # Compute distance from reference point (1, 1, 0)
        dist = np.sqrt((C11 - 1) ** 2 + (C22 - 1) ** 2 + (C12 - 0) ** 2)
        norm = plt.Normalize(np.nanmin(dist), np.nanmax(dist))
        facecolors = plt.cm.coolwarm(norm(dist))
    else:
        facecolors = color

    ax.plot_surface(
        C11,
        C22,
        C12_masked,
        facecolors=facecolors,
    )

    ax.set_xlabel(r"$\mathbf{C}_{11}$")
    ax.set_ylabel(r"$\mathbf{C}_{22}$")
    ax.set_zlabel(r"$\mathbf{C}_{12}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # Never auto-show, always return fig, ax
    return fig, ax


# --- Consistent plotting interface using MetricGeometry and ax argument ---
def plot_surfaces(ax, geometry):
    """Plot all surfaces in the MetricGeometry on the provided ax."""
    for C in geometry.surfaces:
        C11 = C[..., 0]
        C22 = C[..., 1]
        C12 = C[..., 2]
        C12_masked = np.ma.masked_invalid(C12)
        dist = np.sqrt((C11 - 1) ** 2 + (C22 - 1) ** 2 + (C12_masked - 0) ** 2)
        norm = plt.Normalize(np.nanmin(dist), np.nanmax(dist))
        facecolors = plt.cm.coolwarm(norm(dist))
        ax.plot_surface(C11, C22, C12_masked, facecolors=facecolors)


def plot_lines(ax, geometry):
    """Plot all lines in the MetricGeometry on the provided ax with dynamic clipping."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    for L in geometry.lines:
        x = L[:, 0]  # C11
        y = L[:, 1]  # C22
        z = L[:, 2]  # C12

        mask = (
            (x >= xlim[0])
            & (x <= xlim[1])
            & (y >= ylim[0])
            & (y <= ylim[1])
            & (z >= zlim[0])
            & (z <= zlim[1])
        )

        if not np.any(mask):
            continue

        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) >= 2:
            ax.plot(x, y, z, color="k", linewidth=1, zorder=10)


def spinPlot(
    geometry, ax=None, fig=None, filename="Plots/spin.mp4", duration=10, fps=5
):
    """Spin and save the 3D plot of the given MetricGeometry."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Add axis labels to match the other plot styles
        ax.set_xlabel(r"$\mathbf{C}_{11}$")
        ax.set_ylabel(r"$\mathbf{C}_{22}$")
        ax.set_zlabel(r"$\mathbf{C}_{12}$")

    # Set axis limits based on surface bounding box
    all_C11, all_C22, all_C12 = [], [], []
    for C in geometry.surfaces:
        all_C11.append(C[..., 0])
        all_C22.append(C[..., 1])
        all_C12.append(C[..., 2])
    if all_C11:
        all_C11 = np.concatenate([c.ravel() for c in all_C11])
        all_C22 = np.concatenate([c.ravel() for c in all_C22])
        all_C12 = np.concatenate([c.ravel() for c in all_C12])
        ax.set_xlim(np.min(all_C11), np.max(all_C11))
        ax.set_ylim(np.min(all_C22), np.max(all_C22))
        ax.set_zlim(np.min(all_C12), np.max(all_C12))
    # plot_surfaces(ax, geometry)
    plot_lines(ax, geometry)

    initial_azim = DEFAULT_AZIM
    ax.view_init(elev=DEFAULT_ELEV, azim=initial_azim)

    def update(frame):
        ax.view_init(elev=DEFAULT_ELEV, azim=initial_azim + frame)
        return (fig,)

    n_frames = duration * fps
    ani = animation.FuncAnimation(
        fig, update, frames=np.linspace(0, 360, n_frames), blit=False
    )
    ani.save(filename, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved spinning animation to {filename}")


def plot_C11_plus_C22_eq_1(geometry=None, n=100):
    C11_vals = np.linspace(0, 1, n)
    C12_vals = np.linspace(-0.5, 0.5, n)
    C11, C12 = np.meshgrid(C11_vals, C12_vals)
    C22 = 1 - C11

    fig, ax = plot_surface_3d(
        C11,
        C22,
        C12,
        xlim=(0, 1),
        ylim=(0, 1),
        zlim=(-0.5, 0.5),
        color=None,
    )
    if geometry is not None:
        # Stack into (..., 3) array [C11, C22, C12]
        C = np.stack([C11, C22, C12], axis=-1)
        geometry.add_surface(C)
    # Set the camera perspective

    ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)

    return fig, ax


def plot_det_eq_1(geometry=None, n=100):
    # Adaptive domain per C12 row; valid only where C22 in bounds
    CXY = (-0.7, 0.7)
    C11_list, C22_list, C12_list = [], [], []

    C12_vals = np.linspace(*CXY, n)
    for c12 in C12_vals:
        c11_min = (1 + c12**2) / 1.5
        c11_vals = np.linspace(c11_min, 1.5, 150)
        c11_grid, _ = np.meshgrid(c11_vals, [c12])
        c12_grid = np.full_like(c11_grid, c12)
        c22_grid = (1 + c12**2) / c11_grid

        C11_list.append(c11_grid)
        C22_list.append(c22_grid)
        C12_list.append(c12_grid)

    C11 = np.vstack(C11_list)
    C22 = np.vstack(C22_list)
    C12 = np.vstack(C12_list)

    fig, ax = plot_surface_3d(
        C11,
        C22,
        C12,
        xlim=(0.6, 1.5),
        ylim=(0.6, 1.5),
        zlim=CXY,
        color=None,
    )
    if geometry is not None:
        # Stack into (..., 3) array [C11, C22, C12]
        C = np.stack([C11, C22, C12], axis=-1)
        geometry.add_surface(C)

    ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
    return fig, ax


def addShearLines(geometry):
    # Add lines representing horizontal and vertical simple shear
    shear = np.linspace(-1, 1, 100)

    # Horizontal shear: F = [[1, γ], [0, 1]] for γ in shear
    # Shape: (100, 2, 2)
    F_h = np.stack(
        [
            np.stack([np.ones_like(shear), shear], axis=-1),  # First row: [1, γ]
            np.stack(
                [np.zeros_like(shear), np.ones_like(shear)], axis=-1
            ),  # Second row: [0, 1]
        ],
        axis=-2,
    )

    # Vertical shear: F = [[1, 0], [γ, 1]] for γ in shear
    F_v = np.stack(
        [
            np.stack(
                [np.ones_like(shear), np.zeros_like(shear)], axis=-1
            ),  # First row: [1, 0]
            np.stack([shear, np.ones_like(shear)], axis=-1),  # Second row: [γ, 1]
        ],
        axis=-2,
    )

    # Compute C = F^T F for both cases (shape: (100, 2, 2))
    C_h = np.einsum("...ji,...jk->...ik", F_h, F_h)
    C_v = np.einsum("...ji,...jk->...ik", F_v, F_v)

    # Extract components and convert to (N, 3) in [C11, C22, C12] order
    # C11 = [0,0], C22 = [1,1], C12 = [0,1]
    C_h_flat = np.stack([C_h[..., 0, 0], C_h[..., 1, 1], C_h[..., 0, 1]], axis=-1)
    C_v_flat = np.stack([C_v[..., 0, 0], C_v[..., 1, 1], C_v[..., 0, 1]], axis=-1)

    # Add to geometry as lines
    geometry.add_line(C_h_flat)
    geometry.add_line(C_v_flat)


def animate_surface_to_poincare(
    arg, filename="surface_to_poincare.mp4", duration=5, fps=5
):
    """
    Animate the transformation of one or more surfaces and lines to the Poincare disk.
    Accepts either a MetricGeometry object or a single (..., 3) array.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define start and end view configurations
    # Swap ylim and zlim for start_view, and set end_view to x-z projection
    start_view = {
        "elev": 0,
        "azim": -130,
        "xlim": (0, 1),  # C11
        "ylim": (0, 1),  # C22
        "zlim": (-0.5, 0.5),  # C12
    }
    end_view = {
        "elev": 0,
        "azim": -90,
        "xlim": (-1.05, 1.05),
        "ylim": (-0.1, 0.1),
        "zlim": (-1.05, 1.05),
    }

    # Determine input type
    if isinstance(arg, MetricGeometry):
        surfaces = arg.surfaces
        lines = arg.lines
    else:
        surfaces = [arg]
        lines = []

    # Precompute all surface animations
    surface_data = []
    for C in surfaces:
        C11 = C[..., 0]
        C22 = C[..., 1]
        C12 = C[..., 2]
        C12_masked = np.ma.masked_invalid(C12)
        dist = np.sqrt((C11 - 1) ** 2 + (C22 - 1) ** 2 + (C12_masked - 0) ** 2)
        norm = plt.Normalize(np.nanmin(dist), np.nanmax(dist))
        facecolors = plt.cm.coolwarm(norm(dist))
        dets = C11 * C22 - C12**2
        dets = np.clip(dets, 0, np.inf)
        x_ = C12 / C22
        y_ = np.sqrt(dets) / C22
        x2D = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
        y2D = 2 * x_ / (x_**2 + (y_ + 1) ** 2)
        Z = np.zeros_like(x2D)
        surface_data.append((C11, C22, C12, x2D, y2D, Z, facecolors))

    # Precompute all line animations
    line_data = []
    for L in lines:
        C11 = L[:, 0]
        C22 = L[:, 1]
        C12 = L[:, 2]
        dets = C11 * C22 - C12**2
        dets = np.clip(dets, 0, np.inf)
        x_ = C12 / C22
        y_ = np.sqrt(dets) / C22
        x2D = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
        y2D = 2 * x_ / (x_**2 + (y_ + 1) ** 2)
        line_data.append((C11, C22, C12, x2D, y2D))

    n_frames = int(duration * fps)

    def lerp(a, b, t):
        return (1 - t) * np.array(a) + t * np.array(b)

    def update(frame):
        t = frame / (n_frames - 1)
        ax.clear()

        # Camera and axes limits interpolation
        elev = lerp(start_view["elev"], end_view["elev"], t)
        azim = lerp(start_view["azim"], end_view["azim"], t)

        ax.view_init(elev=elev, azim=azim)

        ax.set_xlim(*lerp(start_view["xlim"], end_view["xlim"], t))
        ax.set_ylim(*lerp(start_view["ylim"], end_view["ylim"], t))
        ax.set_zlim(*lerp(start_view["zlim"], end_view["zlim"], t))

        ax.set_xlabel(r"$\mathbf{C}_{11} \rightarrow x_P$")
        ax.set_ylabel(r"$\mathbf{C}_{22} \rightarrow 0$")
        ax.set_zlabel(r"$\mathbf{C}_{12} \rightarrow y_P$")

        for C11, C22, C12, x2D, y2D, Z, facecolors in surface_data:
            X = (1 - t) * C11 + t * x2D
            Z_frame = (1 - t) * C12 + t * y2D
            Y = (1 - t) * C22
            ax.plot_surface(X, Y, Z_frame, facecolors=facecolors)
        for C11, C22, C12, x2D, y2D in line_data:
            X = (1 - t) * C11 + t * x2D
            Z = (1 - t) * C12 + t * y2D
            Y = (1 - t) * C22
            ax.plot(X, Y, Z, color="k", linewidth=1, zorder=10)

        # Fade out the y-axis major and minor ticks
        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            tick.label1.set_alpha(1 - t)
        for line in ax.yaxis.get_ticklines():
            line.set_alpha(1 - t)

        fig.tight_layout()
        return (fig,)

    # Save first and last frames as PDF in the Plots folder
    update(0)
    fig.savefig("Plots/frame_start.pdf", bbox_inches="tight")
    update(n_frames - 1)
    fig.savefig("Plots/frame_end.pdf", bbox_inches="tight")

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(filename, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved transformation animation to {filename}")


# --- New function: animate_surface_to_poincare_steriographic ---
def animate_surface_to_poincare_steriographic(
    arg, filename="surface_to_poincare_steriographic.mp4", duration=5, fps=5
):
    """
    Animate the transformation of one or more surfaces and lines to the stereographic Poincaré disk.
    This version flips the surface so the projection target is at the bottom.
    Accepts either a MetricGeometry object or a single (..., 3) array.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define start and end view configurations
    start_view = {
        "elev": 0,
        "azim": -90,
        "xlim": (-1.05, 1.05),
        "ylim": (-0.1, 0.1),
        "zlim": (-1.05, 1.05),
    }
    end_view = {
        "elev": 0,
        "azim": -90,
        "xlim": (-1.05, 1.05),
        "ylim": (-0.1, 0.1),
        "zlim": (-1.05, 1.05),
    }

    # Determine input type
    if isinstance(arg, MetricGeometry):
        surfaces = arg.surfaces
        lines = arg.lines
    else:
        surfaces = [arg]
        lines = []

    surface_data = []
    for C in surfaces:
        C11 = C[..., 0]
        C22 = C[..., 1]
        C12 = C[..., 2]
        C12_masked = np.ma.masked_invalid(C12)
        dist = np.sqrt((C11 - 1) ** 2 + (C22 - 1) ** 2 + (C12_masked - 0) ** 2)
        norm = plt.Normalize(np.nanmin(dist), np.nanmax(dist))
        facecolors = plt.cm.coolwarm(norm(dist))
        dets = C11 * C22 - C12**2
        dets = np.clip(dets, 0, np.inf)
        x_ = C12 / C22
        y_ = np.sqrt(dets) / C22
        x2D = 2 * x_ / (1 + x_**2 + y_**2)
        y2D = 2 * y_ / (1 + x_**2 + y_**2)
        Z = np.zeros_like(x2D)
        surface_data.append((C11, C22, C12, x2D, y2D, Z, facecolors))

    line_data = []
    for L in lines:
        C11 = L[:, 0]
        C22 = L[:, 1]
        C12 = L[:, 2]
        dets = C11 * C22 - C12**2
        dets = np.clip(dets, 0, np.inf)
        x_ = C12 / C22
        y_ = np.sqrt(dets) / C22
        x2D = 2 * x_ / (1 + x_**2 + y_**2)
        y2D = 2 * y_ / (1 + x_**2 + y_**2)
        line_data.append((C11, C22, C12, x2D, y2D))

    n_frames = int(duration * fps)

    def lerp(a, b, t):
        return (1 - t) * np.array(a) + t * np.array(b)

    def update(frame):
        t = frame / (n_frames - 1)
        ax.clear()

        elev = lerp(start_view["elev"], end_view["elev"], t)
        azim = lerp(start_view["azim"], end_view["azim"], t)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*lerp(start_view["xlim"], end_view["xlim"], t))
        ax.set_ylim(*lerp(start_view["ylim"], end_view["ylim"], t))
        ax.set_zlim(*lerp(start_view["zlim"], end_view["zlim"], t))
        ax.set_xlabel(r"$\mathbf{C}_{11} \rightarrow x_P$")
        ax.set_ylabel(r"$\mathbf{C}_{22} \rightarrow 0$")
        ax.set_zlabel(r"$\mathbf{C}_{12} \rightarrow y_P$")

        for C11, C22, C12, x2D, y2D, Z, facecolors in surface_data:
            X = (1 - t) * C11 + t * x2D
            Z_frame = (1 - t) * C12 + t * y2D
            Y = (1 - t) * C22
            ax.plot_surface(X, Y, Z_frame, facecolors=facecolors)

        for C11, C22, C12, x2D, y2D in line_data:
            X = (1 - t) * C11 + t * x2D
            Z = (1 - t) * C12 + t * y2D
            Y = (1 - t) * C22
            ax.plot(X, Y, Z, color="k", linewidth=1, zorder=10)

        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            tick.label1.set_alpha(1 - t)
        for line in ax.yaxis.get_ticklines():
            line.set_alpha(1 - t)

        fig.tight_layout()
        return (fig,)

    update(0)
    fig.savefig("Plots/frame_stereo_start.pdf", bbox_inches="tight")
    update(n_frames - 1)
    fig.savefig("Plots/frame_stereo_end.pdf", bbox_inches="tight")

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(filename, writer="ffmpeg", fps=fps, dpi=300)
    print(f"Saved stereographic animation to {filename}")


def makeCleanPoincare(addEnergyBackground=False):
    """Create a clean 2D Poincare disk plot without energy surfaces."""
    geometry = MetricGeometry()
    geometry.drawMetricSpaceBackground(6)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw unit disk boundary
    circle = plt.Circle((0, 0), 1, color="black", fill=False)
    ax.add_artist(circle)

    for L, style in zip(geometry.lines, geometry.styles):
        C11 = L[:, 0]
        C22 = L[:, 1]
        C12 = L[:, 2]

        dets = C11 * C22 - C12**2
        dets = np.clip(dets, 0, np.inf)
        x_ = C12 / C22
        y_ = np.sqrt(dets) / C22
        x2D = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
        y2D = 2 * x_ / (x_**2 + (y_ + 1) ** 2)

        # Default styles
        color = style.get("color", "black")
        linewidth = style.get("linewidth", 0.7)
        linestyle = style.get("linestyle", "-")

        ax.plot(x2D, y2D, color=color, linewidth=linewidth, linestyle=linestyle)

    if addEnergyBackground:
        from MTMath.poincareEnergy import generate_energy_grid

        energy_grid = generate_energy_grid(resolution=400, zeroReference=True)
        extent = (-1, 1, -1, 1)
        # Comment out or remove the existing imshow
        # ax.imshow(
        #     energy_grid,
        #     extent=extent,
        #     origin="lower",
        #     cmap="coolwarm",  # or 'viridis', 'hot', etc.
        # )
        im = ax.imshow(
            energy_grid,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Energy density")

    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"$x_P$")
    ax.set_ylabel(r"$y_P$")
    ax.xaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.yaxis.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(False)

    if addEnergyBackground:
        figName = "Plots/poincareDisk.pdf"
    else:
        figName = "Plots/poincareDiskNoEnergy.pdf"

    fig.savefig(figName, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    fps = 6
    # makeCleanPoincare()
    # makeCleanPoincare(True)
    # geometry = MetricGeometry()
    # plot_C11_plus_C22_eq_1(geometry, n=100)
    # plt.savefig("Plots/1+2=1.pdf")

    # geometry = MetricGeometry()
    # plot_C11_plus_C22_eq_1(geometry, n=1)
    # plt.savefig("Plots/1+2=1Empty.pdf")

    geometry = MetricGeometry()
    fig, ax = plot_det_eq_1(geometry=geometry)
    # plt.savefig("Plots/det=1.pdf")

    # addShearLines(geometry)
    # geometry.drawFundamentalDomain()
    # spinPlot(geometry, ax, fig, filename="Plots/noFundamentalDomainSpin.mp4", fps=fps)
    # geometry.drawMetricSpaceBackground()
    animate_surface_to_poincare_steriographic(
        geometry, filename="Plots/all_to_poincare_noFundamentalDomain.mp4", fps=fps
    )

import numpy as np
from .contiPotential import numericContiPotential, ground_state_energy
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import scipy.interpolate as interpolate
from matplotlib import colors
from matplotlib import cm
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm


def oneDPotential():
    # Load the potential and its derivatives
    phi, divPhi, divDivPhi = numericContiPotential()

    # Define size and variables
    distance = (-1.7, 1.7)
    size = 100 * (distance[1] - distance[0])
    shear = np.linspace(distance[0], distance[1], int(size))

    # Create the deformation gradient tensor F for each value of shear
    F = np.array([[[1, s % 1], [0, 1]] for s in shear])

    # Compute the right Cauchy-Green tensor C as F^T * F for each deformation gradient
    # Use matrix multiplication (@) for F.T @ F
    C = np.array([f.T @ f for f in F])

    # Extract the components from C for input into the potential function
    C_00 = C[:, 0, 0]  # First row, first column (C[0, 0])
    C_11 = C[:, 1, 1]  # Second row, second column (C[1, 1])
    C_01 = C[:, 0, 1]  # First row, second column (C[1, 0])

    # Pass the computed components to the phi function (assume constant extra arguments)
    # You may need to adjust these arguments to match the correct inputs for phi
    potential_values = phi(C_00, C_11, C_01, 1.0, 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    # Plot the extended vectors
    ax.plot(shear, potential_values)

    ax.set_xlabel(r"$\gamma$", fontsize=34)
    ax.set_ylabel(r"$\Phi$", fontsize=34)

    plt.tight_layout()

    plt.show()


def oneDPotentialDissordered():
    # Load the potential and its derivatives
    phi, divPhi, divDivPhi = numericContiPotential()

    # Define size and variables
    distance = (-1.7, 1.7)
    size = 100 * (distance[1] - distance[0])
    shear = np.linspace(distance[0], distance[1], int(size))

    # Create the deformation gradient tensor F for each value of shear
    F = np.array([[[1, s % 1], [0, 1]] for s in shear])

    # Compute the right Cauchy-Green tensor C as F^T * F for each deformation gradient
    # Use matrix multiplication (@) for F.T @ F
    C = np.array([f.T @ f for f in F])

    # Extract the components from C for input into the potential function
    C_00 = C[:, 0, 0]  # First row, first column (C[0, 0])
    C_11 = C[:, 1, 1]  # Second row, second column (C[1, 1])
    C_01 = C[:, 0, 1]  # First row, second column (C[1, 0])

    # Pass the computed components to the phi function (assume constant extra arguments)
    # You may need to adjust these arguments to match the correct inputs for phi
    potential_values = phi(C_00, C_11, C_01, 1.0, 1.0, 1.0)

    # Add sinusoidal waves to the potential
    sinusoidal_wave = (
        0.1 * np.sin(10 * shear) + 0.05 * np.sin(15 * shear) + 0.02 * np.sin(5 * shear)
    )
    potential_values += sinusoidal_wave

    fig, ax = plt.subplots(figsize=(7, 4))
    # Plot the extended vectors
    ax.plot(shear, potential_values)

    ax.set_xlabel(r"$\gamma$", fontsize=34)
    ax.set_ylabel(r"$\Phi$", fontsize=34)
    plt.tight_layout()
    plt.show()


def lagrange_reduction(C11, C22, C12, loops=1000):
    for i in range(loops):
        mask1 = C12 < 0
        # m1 (flip) operation
        C12[mask1] *= -1

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break
        # m3 operation
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]

        if i + 1 == loops:
            raise (RuntimeError("Not enough loops"))

    return C11, C22, C12


def elastic_reduction(C11, C22, C12, loops=1000):
    """
    We transform the reduced C an extra time with m1 or m2 such that the number
    of m1 and m2 transformations is even. We also make sure to transform first
    """
    # We create a mask of false everywhere
    odd_swaps_C11 = C11 != C11
    odd_flips_C12 = C12 != C12
    for i in range(loops):
        mask1 = C12 < 0
        C12[mask1] *= -1

        # Stores the last change made to C12
        odd_flips_C12 = np.logical_xor(odd_flips_C12, mask1)

        mask2 = C22 < C11
        # Swap operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        # Stores the last change made to C11 and C22
        odd_swaps_C11 = np.logical_xor(odd_swaps_C11, mask2)

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break
        else:
            C22[mask3] += C11[mask3] - 2 * C12[mask3]
            C12[mask3] -= C11[mask3]

        if i + 1 == loops:
            raise (RuntimeError("Not enough loops"))

    # Now we want to undo the m1 and m2 transformations (Which is the same as
    # doing them again)

    C12[odd_flips_C12] *= -1
    C11[odd_swaps_C11], C22[odd_swaps_C11] = (
        C22[odd_swaps_C11].copy(),
        C11[odd_swaps_C11].copy(),
    )

    return C11, C22, C12


def generate_energy_grid(
    resolution=500,
    beta=-0.25,
    K=4,
    energy_lim=[None, 0.37],
    return_XY=False,
    zoom=1,
    poincareDisk=True,
):
    # Load the potential and its derivatives
    phi, divPhi, divDivPhi = numericContiPotential()

    # Poicare disk
    if poincareDisk:
        # Define the range for x and y based on the unit circle
        radius = 1.0 / zoom

        x_min, x_max = -radius, radius
        y_min, y_max = -radius, radius

        # Create the meshgrid for the x and y coordinates
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )

        # Calculate the mask for points inside the unit circle
        # (We don't need to use radius or zoom here because its only to avoid infinities anyway)
        mask = (X**2 + Y**2) >= (1 - 1e-9)
        X[mask] = np.nan
        Y[mask] = np.nan
        # Precompute some common terms used in a, b, c12, c22, and c11 calculations
        denominator = X**2 - 2 * X + Y**2 + 1
        a = (2 * Y) / denominator
        b = -(X**2 + Y**2 - 1) / denominator

        # Avoid division by zero or near-zero by masking those values in b
        safe_b = np.where(b == 0, np.nan, b)

        # Calculate c12, c22, and c11
        C12 = a / safe_b
        C22 = 1 / safe_b
        C11 = (1 + C12**2) / C22
    else:
        # Define the range for x and y based on the unit circle
        radius = (0.5) / zoom

        x_min, x_max = 0, 1
        y_min, y_max = -0.5, 0.5

        # Create the meshgrid for the x and y coordinates
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        # Calculate the mask for points inside the unit circle
        # (We don't need to use radius or zoom here because its only to avoid infinities anyway)
        mask = ((X - 0.5) ** 2 + Y**2) >= (0.5 - 1e-9) ** 2
        X[mask] = np.nan
        Y[mask] = np.nan
        C12 = Y
        C11 = X
        C22 = 1 - C11

    C11, C22, C12 = lagrange_reduction(C11, C22, C12)

    # Initialize the energy grid with NaNs for points outside the unit circle
    energy_grid = np.full_like(X, np.nan)

    # Apply the phi function only to the points inside the unit circle
    energy_grid = phi(C11, C22, C12, beta, K, 1)

    energy_grid -= ground_state_energy()

    if energy_lim is None:
        energy_lim = (np.nanmin(energy_grid), np.nanmax(energy_grid))
    elif energy_lim[0] is None:
        energy_lim[0] = np.nanmin(energy_grid)
    elif energy_lim[1] is None:
        energy_lim[1] = np.nanmax(energy_grid)

    energy_grid = np.clip(energy_grid, *energy_lim)
    if return_XY:
        # We don't need to have nan in X and Y, only in the energy grid
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        return energy_grid, X, Y
    else:
        return energy_grid


def C2PoincareDisk(C):
    if C.ndim == 2:
        x_, y_ = (C[0, 1] / C[1, 1], np.sqrt(np.linalg.det(C)) / C[1, 1])
    else:
        dets = np.linalg.det(C)
        x_, y_ = (C[:, 0, 1] / C[:, 1, 1], np.sqrt(dets) / C[:, 1, 1])

    x = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
    y = 2 * x_ / (x_**2 + (y_ + 1) ** 2)

    return x, y


def drawC(ax, C, grid_size, zoom=1, c="w", linestyle="--", linewidth=0.6, **kwargs):
    x, y = C2PoincareDisk(C)
    ax.plot(
        x * zoom * grid_size / 2 + grid_size / 2,
        y * zoom * grid_size / 2 + grid_size / 2,
        c=c,
        linewidth=linewidth,
        linestyle=linestyle,
        **kwargs,
    )


def drawCScatter(
    ax, C, grid_size, remove_max_color=True, vmax=None, log_scale=True, zoom=1
):
    x, y = C2PoincareDisk(C)
    # Create a density estimate
    xy = np.vstack([x, y])

    # Scott rule
    bandwidth = len(x) ** (-1 / 6)
    try:
        kde = gaussian_kde(xy, bw_method=bandwidth)
        density1 = kde(xy)
    except np.linalg.LinAlgError:
        # Assign a uniform value to make all points appear red
        density1 = np.ones_like(x) * 1e10  # High value to map to red

    cmap = "inferno"
    if remove_max_color:
        coolwarm = cm.get_cmap(cmap, 256)  # 256 colors
        newcolors = coolwarm(np.linspace(0, 1, 256))
        n = 2
        newcolors[-n:, -1] = np.linspace(1, 0, n) ** (1 / 2)
        cmap = colors.ListedColormap(newcolors)

    # Check if log scale is to be applied
    norm = None
    if log_scale:
        # Use LogNorm for logarithmic scale normalization
        norm = LogNorm(vmin=1, vmax=vmax)
        # We set it to None so that it is not given to the scatter function
        vmax = None

    # Plot with scatter, adjusting color based on density
    scatter = ax.scatter(
        x * zoom * grid_size / 2 + grid_size / 2,
        y * zoom * grid_size / 2 + grid_size / 2,
        c=density1,
        s=0.2,
        linewidth=0,
        cmap=cmap,
        norm=norm,
        vmax=vmax,
    )
    plt.colorbar(scatter, ax=ax, label="Kernel density estimate", pad=-0.0005)


def drawFundamentalDomain(ax, **kwargs):
    nr = 1000
    zero = np.array([0] * nr)
    # VERTICAL LINE
    t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
    # Values from -1<t<1 give complex solutions
    # det=1, C12=C21, C11=C22
    C = np.array([[t, np.sqrt(t**2 - 1)], [np.sqrt(t**2 - 1), t]]).transpose(2, 0, 1)

    drawC(ax, C, **kwargs)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
    drawC(ax, C, **kwargs)

    # FUNDAMENTAL DOMAIN (0.01 to avoid div by 0)
    # https://www.wolframalpha.com/input?i=0%3Ca%3Cd%2C+b%3Da%2F2%2C+++a*d-b*c%3D1%2C+b%3Dc
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2 / np.sqrt(3)), nr))
    # Negative values are outside of the circle
    # det=1, C12=C21,
    C = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)
    drawC(ax, C, **kwargs)


def drawShearPath(ax, **kwargs):
    nr = 1000
    one = np.array([1] * nr)

    t = np.sinh(np.linspace(np.arcsinh(0.001), np.arcsinh(300), nr))
    C = np.array([[one, t], [t, one + t**2]]).transpose(2, 0, 1)
    drawC(ax, C, **kwargs)


def plotEnergyField(
    energy_grid,
    fig=None,
    ax=None,
    save=True,
    add_title=True,
    zoom=1,
    remove_max_color=True,
):
    # Define the range for x and y based on the unit circle
    radius = 1.0
    x_min, x_max = -radius / zoom, radius / zoom
    y_min, y_max = -radius / zoom, radius / zoom
    grid_size = len(energy_grid)

    # Create the plot
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

    max_energy = np.nanmax(energy_grid)

    cmap = "coolwarm"
    if remove_max_color:
        coolwarm = cm.get_cmap(cmap, 256)  # 256 colors
        newcolors = coolwarm(np.linspace(0, 1, 256))
        n = 2
        newcolors[-n:, -1] = np.linspace(1, 0, n) ** (1 / 2)
        cmap = colors.ListedColormap(newcolors)
    img = ax.imshow(energy_grid, cmap=cmap, origin="lower")

    # Add a thin black circle
    circleSize = (grid_size / 2) * zoom
    circle_center_x = grid_size / 2
    circle_center_y = grid_size / 2
    circle = Circle(
        (circle_center_x, circle_center_y),
        circleSize,
        color="black",
        fill=False,
        linewidth=1,
    )
    fig.gca().add_patch(circle)

    # Draw fundamental domain
    drawFundamentalDomain(ax, grid_size=grid_size, zoom=zoom)

    # Draw shear path
    drawShearPath(ax, grid_size=grid_size, zoom=zoom, linestyle="-")

    # Draw elastic domain
    # TODO

    # Adjusting ticks
    ax.set_xticks(
        np.linspace(0, grid_size - 1, 5),
        np.linspace(x_min, x_max, 5).round(2),
    )
    ax.set_yticks(
        np.linspace(0, grid_size - 1, 5),
        np.linspace(y_min, y_max, 5).round(2),
    )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Add colorbar
    cbar = fig.colorbar(img, label="Energy", pad=-0.01)
    default_font_size = plt.rcParams["font.size"]  # Fetch default font size
    cbar.ax.set_title(f"Capped at ${max_energy}$", fontsize=default_font_size)
    nbs = "\u00a0"  # non-breaking-space
    # $P_x$(Length ratio)
    ax.set_xlabel(f"← Tall {nbs*6} Wide →")
    # $P_y$(Length ratio and $\\theta - \\pi/2$)
    ax.set_ylabel(f"← Large angle {nbs*6} Small angle →")
    if add_title:
        ax.set_title("Energy field in a Poincaré disk")

    if save:
        output_pdf_path = "energy_field.pdf"
        fig.savefig(
            output_pdf_path,
            format="pdf",
            dpi=600,
            bbox_inches="tight",
        )


def add_arrow_3d(xdata, ydata, zdata, ax, start_ind, end_ind, size=15, color="red"):
    """
    NOT WORKING. It removes other lines for some reason. Very annoying.
    Add an arrow to a 3D line by specifying start and end indices along the data points.

    xdata, ydata, zdata: Coordinates of the 3D line.
    ax: The 3D axes object.
    start_ind: Starting index for the arrow.
    end_ind: Ending index for the arrow.
    size: Size of the arrow in fontsize points.
    color: Color of the arrow.
    """
    # Annotate with an arrow
    ax.quiver(
        xdata[start_ind],
        ydata[start_ind],
        zdata[start_ind],  # Starting point
        xdata[end_ind] - xdata[start_ind],  # Arrow vector in x direction
        ydata[end_ind] - ydata[start_ind],  # Arrow vector in y direction
        zdata[end_ind] - zdata[start_ind],  # Arrow vector in z direction
        arrow_length_ratio=0.3,  # Control the size of the arrow head
        color=color,
        linewidth=1.5,
    )


def plot_arch(
    energy_grid,
    X,
    Y,
    ax,
    radius=0.5,
    start_angle=np.pi,
    end_angle=0,
    center_x=0,
    center_y=0,
    num_points=200,
    arrow_interval=10,
    label="path",
):
    """
    Generates x, y coordinates and interpolates z values for a semi-circle.

    Parameters:
    - center_x, center_y: center of the circle
    - radius: radius of the circle
    - start_angle, end_angle: range of angles (in radians) for the arch
    - num_points: number of points along the arch

    Returns:
    - x_circle, y_circle: coordinates of the arch
    - z_circle: interpolated z values along the arch
    """
    theta = np.linspace(start_angle, end_angle, num_points)  # Parametrize the angles
    x_circle = center_x + radius * np.cos(theta)  # X coordinates of the arch
    y_circle = center_y + radius * np.sin(theta)  # Y coordinates of the arch

    # Interpolate z values along the arch
    X_mesh, Y_mesh = np.meshgrid(X[0], Y[:, 0])
    X_flat = X_mesh.flatten()
    Y_flat = Y_mesh.flatten()
    energy_flat = np.nan_to_num(energy_grid, nan=0).flatten()

    z_circle = interpolate.griddata(
        (X_flat, Y_flat),
        energy_flat,
        (x_circle, y_circle),
        method="linear",  # Linear interpolation
    )
    # Plot the line along the arch
    ax.plot(
        x_circle,
        y_circle,
        z_circle,
        color="black",
        linewidth=1,
        label=label,
        zorder=10,
    )
    # Add arrows along the arch at specified intervals
    # for i in range(arrow_interval, len(x_circle), arrow_interval):
    # add_arrow_3d(x_circle, y_circle, z_circle, ax, i - 1, i, size=15, color="red")


def make3DEnergyField(
    energy_grid,
    X,
    Y,
    energy_lim=None,
    zScale=0.3,
    data_radius=0.8,
    zoom=1,
    add_front_hole=True,
    remove_max_color=True,
    left_arch=False,
    right_arch=True,
):
    print("Plotting energy field...")

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if energy_lim is None:
        energy_lim = (np.nanmin(energy_grid), np.nanmax(energy_grid))
    elif energy_lim[0] is None:
        energy_lim[0] = np.nanmin(energy_grid)
    elif energy_lim[1] is None:
        energy_lim[1] = np.nanmax(energy_grid)

    # Calculate the radii from the meshgrid (X, Y)
    radii = np.sqrt(X**2 + Y**2)
    # Create the first mask for points outside the main circle
    center_mask = radii > data_radius

    if add_front_hole:
        # For a better view of the landscape, we also want to hide a small portion
        # (0, 0.4)
        # Calculate the radii from the meshgrid (X, Y)
        radii2 = np.sqrt((X) ** 2 + (Y - 1.4) ** 2)
        # Create the second mask to exclude the small portion
        front_hole = radii2 < 1
        mask = center_mask | front_hole
    else:
        mask = center_mask

    # Apply the mask to the energy grid to set values outside the unit circle to NaN
    energy_grid[mask] = np.nan

    base_cmap_name = "coolwarm"
    if remove_max_color:
        coolwarm = cm.get_cmap(base_cmap_name, 256)  # 256 colors
        newcolors = coolwarm(np.linspace(0, 1, 256))
        n = 2
        newcolors[-n:, -1] = np.linspace(1, 0, n) ** (1 / 2)
        cmap = colors.ListedColormap(newcolors)
    else:
        cmap = base_cmap_name

    # Plot the surface with the masked energy grid
    surf = ax.plot_surface(
        X,
        Y,
        energy_grid,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
        rstride=1,  # Increase the number of rows used for plotting
        cstride=1,  # Increase the number of columns used for plotting
        vmin=energy_lim[0],
        vmax=energy_lim[1],
    )
    # plot semi-circles
    if right_arch:
        plot_arch(energy_grid, X, Y, ax, start_angle=-1.2, end_angle=0.9, center_x=-0.5)
    if left_arch:
        plot_arch(energy_grid, X, Y, ax, start_angle=3.9, end_angle=2.2, center_x=0.5)
    # Add a color bar
    cbar = fig.colorbar(surf, location="right")

    zLabel = r"Energy density $\Phi$"
    cbar.set_label(zLabel)

    default_font_size = plt.rcParams["font.size"]  # Fetch default font size
    cbar.ax.set_title(f"Capped at ${energy_lim[1]}$", fontsize=default_font_size)
    # Set labels for the axes
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")
    ax.set_zlabel(zLabel)
    # ax.set_title("Energy Surface Plot")

    # Set x and y limits based on the valid data (non-NaN values in the energy grid)
    x_min = np.nanmin(X[~mask])
    x_max = np.nanmax(X[~mask])
    y_min = np.nanmin(Y[~mask])
    y_max = np.nanmax(Y[~mask])

    def lim(zoom, lim):
        width = np.diff(lim)
        center = lim[0] + width / 2
        shift = (width / 2) / zoom
        return center - shift, center + shift

    ax.set_xlim(*lim(zoom, [x_min, x_max]))
    ax.set_ylim(*lim(zoom, [y_min, y_max]))

    # Adjust limits and view angle for better visualization
    ax.set_zlim(*lim(zScale, energy_lim))
    ax.view_init(elev=35, azim=80)  # Set the view angle (elevation and azimuth)
    fig.tight_layout()
    plt.savefig("Plots/3DEnergy.png", dpi=500)
    plt.show()

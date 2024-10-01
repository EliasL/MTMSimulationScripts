import numpy as np
from .contiPotential import numericContiPotential
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import scipy.interpolate as interpolate
from mpl_toolkits.mplot3d import Axes3D


def OneDPotential():
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

    fig, ax = plt.subplots(figsize=(12, 4))
    # Plot the extended vectors
    ax.plot(shear, potential_values)

    ax.set_xlabel(r"$\gamma$", fontsize=34)
    ax.set_ylabel(r"$\Phi$", fontsize=34)

    plt.show()


def lagrange_reduction(C11, C22, C12, loops=600):
    for i in range(loops):
        mask1 = C12 < 0
        C12[mask1] *= -1

        mask2 = C22 < C11
        # Swap operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        mask3 = 2 * C12 > C11
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]
    return C11, C22, C12


def generate_energy_grid(
    resolution=500, beta=-0.25, K=4, min_energy=3.9, max_energy=4.16, return_XY=False
):
    # Load the potential and its derivatives (assuming this function is defined elsewhere)
    phi, divPhi, divDivPhi = numericContiPotential()

    # Define the range for x and y based on the unit circle
    radius = 1.0
    x_min, x_max = -radius, radius
    y_min, y_max = -radius, radius

    # Create the meshgrid for the x and y coordinates
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # Calculate the mask for points inside the unit circle
    mask = X**2 + Y**2 <= 1 - 1e-9

    # Precompute some common terms used in a, b, c12, c22, and c11 calculations
    denominator = X**2 - 2 * X + Y**2 + 1
    a = (2 * Y) / denominator
    b = -(X**2 + Y**2 - 1) / denominator

    # Avoid division by zero or near-zero by masking those values in b
    safe_b = np.where(b == 0, np.nan, b)

    # Calculate c12, c22, and c11 using vectorized operations
    C12 = a / safe_b
    C22 = 1 / safe_b
    C11 = (1 + C12**2) / C22
    C11, C22, C12 = lagrange_reduction(C11, C22, C12)

    # Initialize the energy grid with NaNs for points outside the unit circle
    energy_grid = np.full_like(X, np.nan)

    # Apply the phi function only to the points inside the unit circle
    energy_grid[mask] = phi(C11[mask], C22[mask], C12[mask], beta, K, 1)

    # Cap energies to avoid extreme values
    if min_energy is None:
        min_energy = energy_grid.min()
    if max_energy is None:
        max_energy = energy_grid.max()

    energy_grid = np.clip(energy_grid, min_energy, max_energy)
    if return_XY:
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


def drawC(ax, C, scale):
    pos = C2PoincareDisk(C)
    ax.plot(
        pos[0] * scale / 2 + scale / 2,
        pos[1] * scale / 2 + scale / 2,
        c="black",
        linewidth=0.6,
        linestyle="--",
    )


def drawFundamentalDomain(ax, scale):
    nr = 1000
    zero = np.array([0] * nr)
    # VERTICAL LINE
    t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
    # Values from -1<t<1 give complex solutions
    # det=1, C12=C21, C11=C22
    C = np.array([[t, np.sqrt(t**2 - 1)], [np.sqrt(t**2 - 1), t]]).transpose(2, 0, 1)

    drawC(ax, C, scale)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
    drawC(ax, C, scale)

    # FUNDAMENTAL DOMAIN (0.01 to avoid div by 0)
    # https://www.wolframalpha.com/input?i=0%3Ca%3Cd%2C+b%3Da%2F2%2C+++a*d-b*c%3D1%2C+b%3Dc
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2 / np.sqrt(3)), nr))
    # Negative values are outside of the circle
    # det=1, C12=C21,
    C = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)
    drawC(ax, C, scale)


def plotEnergyField(energy_grid):
    print("Plotting energy field...")
    # Define the range for x and y based on the unit circle
    radius = 1.0
    x_min, x_max = -radius, radius
    y_min, y_max = -radius, radius
    grid_size = len(energy_grid)

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    max_energy = np.nanmax(energy_grid)

    img = ax.imshow(energy_grid, cmap="viridis", origin="lower")

    # Add a thin black circle
    circleSize = grid_size / 2
    circle_center_x = circleSize
    circle_center_y = circleSize
    circle = Circle(
        (circle_center_x, circle_center_y),
        circleSize,
        color="black",
        fill=False,
        linewidth=1,
    )
    fig.gca().add_patch(circle)

    # Draw fundamental domain
    # drawFundamentalDomain(ax, grid_size)

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
    cbar = fig.colorbar(img, label="Energy")
    default_font_size = plt.rcParams["font.size"]  # Fetch default font size
    cbar.ax.set_title(f"Capped at ${max_energy}$", fontsize=default_font_size)
    nbs = "\u00a0"  # non-breaking-space
    ax.set_xlabel(f"← Tall {nbs*7} $P_x$(Length ratio) {nbs*7} Wide →")
    ax.set_ylabel(
        f"← Large angle {nbs*7} $P_y$(Length ratio and $\\theta - \\pi/2$) {nbs*7} Small angle →"
    )
    ax.set_title("Energy field in a Poincaré disk")

    output_pdf_path = "energy_field.pdf"
    fig.savefig(
        output_pdf_path, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0
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
    energy_flat = energy_grid.flatten()

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


def make3DEnergyField(energy_grid, X, Y, energy_lim=None, zScale=0.3, zoom=0):
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
    # For a better view of the landscape, we also want to hide a small portion
    # (0, 0.4)
    # Calculate the radii from the meshgrid (X, Y)
    radii2 = np.sqrt((X) ** 2 + (Y - 1.4) ** 2)

    # Create the first mask for points outside the main circle
    mask1 = radii > 0.8

    # Create the second mask to exclude the small portion
    mask2 = radii2 < 1

    # Remove large max plates
    d = 0.6
    r = 0.35
    radii3 = np.sqrt((X - d) ** 2 + (Y) ** 2)
    radii4 = np.sqrt((X + d) ** 2 + (Y) ** 2)
    mask3 = (radii3 < r) | (radii4 < r)
    # Combine the two masks using element-wise logical AND
    mask = mask1 | mask2 | mask3

    # Apply the mask to the energy grid to set values outside the unit circle to NaN
    energy_grid[mask] = np.nan
    # Plot the surface with the masked energy grid
    surf = ax.plot_surface(
        X,
        Y,
        energy_grid,
        cmap="coolwarm",
        linewidth=0,
        antialiased=False,
        rstride=1,  # Increase the number of rows used for plotting
        cstride=1,  # Increase the number of columns used for plotting
        vmin=energy_lim[0],
        vmax=energy_lim[1],
    )
    # plot semi-circles
    plot_arch(energy_grid, X, Y, ax, start_angle=-1.3, end_angle=0.9, center_x=-0.5)
    # plot_arch(energy_grid, X, Y, ax, start_angle=3.8, end_angle=2, center_x=0.5)
    # Add a color bar
    cbar = fig.colorbar(surf)
    zLabel = r"Energy density ($\Phi$)"
    cbar.set_label(zLabel)

    default_font_size = plt.rcParams["font.size"]  # Fetch default font size
    cbar.ax.set_title(f"Capped at ${energy_lim[1]}$", fontsize=default_font_size)
    # Set labels for the axes
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")
    ax.set_zlabel(zLabel)
    # ax.set_title("Energy Surface Plot")

    # Set x and y limits based on the valid data (non-NaN values in the energy grid)
    valid_x_min = np.nanmin(X[~mask])
    valid_x_max = np.nanmax(X[~mask])
    valid_y_min = np.nanmin(Y[~mask])
    valid_y_max = np.nanmax(Y[~mask])

    xDiff = valid_x_max - valid_x_min
    yDiff = valid_y_max - valid_y_min
    ax.set_xlim(valid_x_min + xDiff * zoom / 2, valid_x_max - xDiff * zoom / 2)
    ax.set_ylim(valid_y_min + yDiff * zoom / 2, valid_y_max - yDiff * zoom / 2)

    # Adjust limits and view angle for better visualization
    ax.set_zlim(
        energy_lim[0] - 0.5 / zScale * np.diff(energy_lim),
        energy_lim[1] + 0.5 / zScale * np.diff(energy_lim),
    )
    ax.view_init(elev=30, azim=70)  # Set the view angle (elevation and azimuth)

    plt.savefig("Plots/3DEnergy.png", dpi=500)
    # plt.show()

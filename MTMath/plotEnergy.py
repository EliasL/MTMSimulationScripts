import numpy as np


try:
    from .contiPotential import ContiEnergy, lagrange_reduction
except ModuleNotFoundError:
    from contiPotential import ContiEnergy, lagrange_reduction
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import scipy.interpolate as interpolate
from matplotlib import colors
from matplotlib import cm
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm


def oneDPotential():
    # Load the potential and its derivatives
    phi, divPhi, divDivPhi = ContiEnergy.numeric_potential()

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
    phi, divPhi, divDivPhi = ContiEnergy.numeric_potential()

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


def generate_poincare_disk(
    resolution=500, zoom=1, returnMask=False, transformation=None
):
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

    C = poincareDisk2C(X, Y, transformation=transformation)

    if returnMask:
        return C, mask
    return C


def generate_energy_grid(
    resolution=500,
    zoom=1,
    beta=-0.25,
    K=4,
    energy_lim=[None, 0.37],
    return_XY=False,
    poincareDisk=True,
    zeroReference=True,
):
    x_min, x_max = 0, 1
    y_min, y_max = -0.5, 0.5
    # Poicare disk
    if poincareDisk:
        C = generate_poincare_disk(resolution, zoom)
    else:
        # Create the meshgrid for the x and y coordinates
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        C12 = Y
        C11 = X
        C22 = 1 - C11
        C = np.stack(
            [
                np.stack([C11, C12], axis=-1),  #
                np.stack([C12, C22], axis=-1),
            ],
            axis=-2,
        )

    energy_grid = ContiEnergy.energy_from_C_in_place(C, beta, K, 1, zeroReference)

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


def generate_angle_region(resolution=500, zoom=1):
    C, r_mask = generate_poincare_disk(resolution, zoom, returnMask=True)
    # Create a boolean mask for the region that should be transparent
    mask = (C[..., 0, 1] > 1) | (C[..., 0, 1] < 0)
    # Create a float array (e.g., filled with ones) with the same shape as the mask
    region = np.ones_like(mask, dtype=float)
    # Set the parts where the mask is True to np.nan
    region[mask | r_mask] = np.nan
    return region


def C_to_xy(C, eps=1e-12, transformation=None):
    """
    Map a symmetric 2x2 matrix C to (x, y) on the Poincaré disk by:
      (i)  normalizing C so det(C)=1 (if det>0),
      (ii) projecting the normalized matrix to (x,y).

    Supports a single 2x2 or a batch of shape (..., 2, 2).
    Returns x, y, and the normalized matrix C_hat (det=1 where valid, else NaNs).
    """
    C = np.asarray(C, dtype=float)

    C = transformC(C, transformation)

    a = C[..., 0, 0]
    b = C[..., 0, 1]
    c = C[..., 1, 1]
    # Determinant and validity
    # Dont use np.linalg.det. It is less numerically stable
    det = a * c - b * b
    valid = det > eps

    # Scale to det=1 without taking sqrt on invalid entries
    scale = np.empty_like(det, dtype=float)
    scale[valid] = np.sqrt(det[valid])
    scale[~valid] = np.nan
    C_hat = C / scale[..., None, None]  # det(C_hat) = 1 where valid

    # Projection to (x,y) from the det=1 surface (stereographic-style inverse)
    c11 = C_hat[..., 0, 0]
    c12 = C_hat[..., 0, 1]
    c22 = C_hat[..., 1, 1]

    t = 1.0 / (2.0 + c11 + c22)
    x = t * (c11 - c22)
    y = 2 * t * c12

    return x, y


def C2PoincareDisk(C, transformation=None):
    return C_to_xy(C, transformation=transformation)

    """Map a metric C to Poincaré disk coordinates (x, y).

    By default, the identity metric maps to the center. To center another
    target metric C0 at the origin, first apply a congruence transform that
    sends C0 to the identity: choose M = inv(cholesky(C0)) and map
    C -> M^T C M before computing (x, y).

    Supported values for `transformation`:
      - "triangular": centers the triangular metric [[1, 1/2], [1/2, 1]].
      - np.ndarray (2x2 matrix): this matrix M is used directly as a
        congruence transform C -> M^T C M.
    """

    C = transformC(C, transformation)

    with np.errstate(divide="ignore", invalid="ignore"):
        if C.ndim == 2:
            # t = 2.0 / (2.0 + C[0, 0] + C[1, 1])
            # x = t * (C[0, 0] - C[1, 1]) / 2.0
            # y = t * C[0, 1]
            # return x, y

            det = np.linalg.det(C)
            y_ = np.sqrt(det) / C[1, 1] if det >= 0 else np.nan
            x_ = C[0, 1] / C[1, 1]
            x = (x_**2 + y_**2 - 1) / (x_**2 + (y_ + 1) ** 2)
            y = 2 * x_ / (x_**2 + (y_ + 1) ** 2)
            if not np.isfinite(y_):
                x, y = np.nan, np.nan

        else:
            # t = 2.0 / (2.0 + C[:, 0, 0] + C[:, 1, 1])

            # x = t * (C[:, 0, 0] - C[:, 1, 1]) / 2.0
            # y = t * C[:, 0, 1]
            # return x, y
            dets = np.linalg.det(C)
            valid = dets >= 0
            y_ = np.full_like(dets, np.nan)
            y_[valid] = np.sqrt(dets[valid]) / C[valid, 1, 1]
            x_ = C[:, 0, 1] / C[:, 1, 1]

            denom = x_**2 + (y_ + 1) ** 2
            x = (x_**2 + y_**2 - 1) / denom
            y = 2 * x_ / denom

            # Leave invalid points as NaN so plotting functions can ignore them
            x[~valid] = np.nan
            y[~valid] = np.nan

        return x, y


def poincareDisk2C(X, Y, transformation=None):
    if True:
        r = 1.0 - X**2 - Y**2
        safe_r = np.where(r == 0, np.nan, r)
        t = 2.0 / safe_r
        C11 = t * (1.0 + X) - 1.0
        C22 = t * (1.0 - X) - 1.0
        C12 = t * Y
    else:
        # Old code. Should be equivalent
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

    C = np.stack(
        [
            np.stack([C11, C12], axis=-1),  #
            np.stack([C12, C22], axis=-1),
        ],
        axis=-2,
    )

    C = transformC(C, transformation)
    return C


def transformC(C, transformation):
    if transformation is not None:
        if isinstance(transformation, np.ndarray):
            # Use the provided matrix directly as a congruence transform
            # (broadcasts over C if C has a leading dimension)
            C = conTrans(C, transformation)
        elif transformation == "triangular":
            M = np.array([[-1.0, 0.0], [0.5, -np.sqrt(3) / 2]])
            gamma = (4 / 3) ** (1 / 4)
            M = np.array(
                [
                    [gamma, 0],
                    [gamma / 2, gamma * np.sqrt(3) / 2],
                ]
            )
            C = conTrans(C, M)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
    return C


def drawC(
    ax,
    C,
    grid_size,
    zoom=1,
    c="black",
    linestyle="-",
    linewidth=0.6,
    transformation=None,
    **kwargs,
):
    x, y = C2PoincareDisk(C, transformation=transformation)
    # Mask invalid points; matplotlib will break lines at NaNs
    if np.ndim(x) == 0:
        return
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    ax.plot(
        x * zoom * grid_size / 2 + grid_size / 2,
        y * zoom * grid_size / 2 + grid_size / 2,
        c=c,
        linewidth=linewidth,
        linestyle=linestyle,
        **kwargs,
    )


def drawRegion(ax, region, grid_size, zoom=1, cmap=None, **kwargs):
    # Doesn't work with transformations

    # Mask invalid values (NaNs) so they become transparent
    data = np.ma.masked_invalid(region)

    # Use a copy of the colormap so we can set NaNs to be transparent
    if cmap is None:
        cmap = kwargs.pop("cmap", cm.get_cmap("Greens").copy())

    # Keep the same pixel coordinate system as the other plots
    extent = [
        (grid_size / 2) * (1 - 1 / zoom),
        (grid_size / 2) * (1 + 1 / zoom),
        (grid_size / 2) * (1 - 1 / zoom),
        (grid_size / 2) * (1 + 1 / zoom),
    ]

    img = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cmap,
        **kwargs,
    )
    return img


def drawCScatter(
    ax,
    C,
    grid_size,
    remove_max_color=True,
    vmax=None,
    log_scale=True,
    zoom=1,
    transformation=None,
):
    x, y = C2PoincareDisk(C, transformation)
    # Filter out invalid points
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    # Create a density estimate
    xy = np.vstack([x, y])

    # Scott rule
    bandwidth = len(x) ** (-1 / 6)
    try:
        if x.size < 2:
            raise np.linalg.LinAlgError("insufficient points for KDE")
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

    # Avoid log(0) by setting a small floor
    density_safe = np.clip(density1, 1e-10, None)

    # Take log and normalize to [0, 1]
    log_density = np.log(density_safe)
    log_density_norm = (log_density - np.min(log_density)) / (
        np.max(log_density) - np.min(log_density) + 1e-12
    )

    # High density → small size (0.5), low density → large size (3)
    min_size = 0.5
    max_size = 3
    sizes = min_size + (1.0 - log_density_norm) * (max_size - min_size)

    # Plot with scatter, adjusting color based on density
    scatter = ax.scatter(
        x * zoom * grid_size / 2 + grid_size / 2,
        y * zoom * grid_size / 2 + grid_size / 2,
        c=density1,
        s=sizes,
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


def drawPoincareGrid(ax, grid_size, zoom=1, depth=6, **kwargs):
    nr = 1000
    zero = np.array([0] * nr)
    # VERTICAL LINE
    t = np.sinh(np.linspace(np.arcsinh(1), np.arcsinh(2 / np.sqrt(3)), nr))
    # Values from -1<t<1 give complex solutions
    # det=1, C12=C21, C11=C22

    # Vertical Positive
    C = np.array([[t, np.sqrt(t**2 - 1)], [np.sqrt(t**2 - 1), t]]).transpose(2, 0, 1)
    drawAllVariations(ax, C, grid_size, depth=depth, zoom=zoom, **kwargs)

    # Vertical Negative
    C = np.array([[t, -np.sqrt(t**2 - 1)], [-np.sqrt(t**2 - 1), t]]).transpose(2, 0, 1)
    drawAllVariations(ax, C, grid_size, depth=depth, zoom=zoom, **kwargs)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
    drawAllVariations(
        ax, C, grid_size, depth=depth, zoom=zoom, **kwargs, linestyle="--"
    )

    # FUNDAMENTAL DOMAIN (0.01 to avoid div by 0)
    # https://www.wolframalpha.com/input?i=0%3Ca%3Cd%2C+b%3Da%2F2%2C+++a*d-b*c%3D1%2C+b%3Dc
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2 / np.sqrt(3)), nr))
    # Negative values are outside of the circle
    # det=1, C12=C21,
    C = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)

    drawAllVariations(ax, C, grid_size, depth=depth, zoom=zoom, **kwargs)


def conTrans(C, M):
    """Apply a congruence transform: return M^T @ C @ M.

    Works for C with shape (2, 2) or (N, 2, 2).
    M can be (2, 2) for a single transform broadcast over all slices, or (N, 2, 2)
    to use a different matrix per slice (matching the leading dimension of C).
    """
    C_arr = np.asarray(C)
    M_arr = np.asarray(M, dtype=C_arr.dtype)

    if M_arr.ndim == 2:
        # Broadcast a single 2x2 over all slices of C
        return M_arr.T @ C_arr @ M_arr
    if M_arr.ndim == 3:
        # Per-slice transform; relies on batch matmul broadcasting
        return np.swapaxes(M_arr, -1, -2) @ C_arr @ M_arr

    raise ValueError("M must have shape (2,2) or (N,2,2)")


def _m3_const(dtype_str):
    dt = np.dtype(dtype_str)
    # Shear matrix [[1, -1], [0, 1]] used in up/right moves
    return np.array([[1, -1], [0, 1]], dtype=dt)


def _m3_for(C):
    """Return the canonical m3 with the proper dtype for C."""
    return _m3_const(np.asarray(C).dtype.str)


def up(C):
    M = _m3_for(C)  # shape (2,2), broadcasts over slices of C
    return conTrans(C, M)


def down(C):
    M = np.linalg.inv(_m3_for(C))
    return conTrans(C, M)


def right(C):
    M = _m3_for(C).T
    return conTrans(C, M)


def left(C):
    M = np.linalg.inv(_m3_for(C).T)
    return conTrans(C, M)


def upInv(C):
    return np.linalg.inv(up(C))


def rightInv(C):
    return np.linalg.inv(right(C))


def drawSquareElasticDomain(ax, **kwargs):
    nr = 1000

    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(2 / np.sqrt(3)), nr))
    # Negative values are outside of the circle
    # det=1, C12=C21,
    C = np.array([[t, t / 2], [t / 2, (t**2 + 4) / (4 * t)]]).transpose(2, 0, 1)

    drawC(ax, C, **kwargs)
    drawC(ax, upInv(C), **kwargs)
    drawC(ax, up(C), **kwargs)
    drawC(ax, right(upInv(C)), **kwargs)


def drawTriangularElasticDomain(ax, shade=False, **kwargs):
    nr = 1000

    zero = np.array([0] * nr)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)

    drawC(ax, C, **kwargs)
    drawC(ax, down(C), **kwargs)
    drawC(ax, left(C), **kwargs)
    drawC(ax, left(up(C)), **kwargs)
    drawC(ax, left(up(left(C))), **kwargs)
    drawC(ax, left(left(up(left(C)))), **kwargs)

    # drawC(ax, C, **kwargs)
    # drawC(ax, right(upInv(C)), **kwargs)
    # drawC(ax, upInv(C), **kwargs)
    # drawC(ax, rightInv(C), **kwargs)
    # drawC(ax, upInv(up(rightInv(C))), **kwargs)
    # drawC(ax, rightInv(up(rightInv(C))), **kwargs)

    # Shade the region defined by 0 <= C12 <= min(C11, C22)
    # Shading does not work with transformations
    transformation = kwargs.get("transformation", None)
    if shade and transformation is None:
        grid_size = kwargs.get("grid_size", 200)
        zoom_val = kwargs.get("zoom", 1)

        # C has shape (grid, grid, 2, 2); r_mask is True on/outside the rim
        C, r_mask = generate_poincare_disk(
            grid_size, zoom_val, returnMask=True, transformation=transformation
        )

        C11 = C[..., 0, 0]
        C12 = C[..., 0, 1]
        C22 = C[..., 1, 1]

        # Region: 0 <= C12 <= min(C11, C22)
        region_mask = np.logical_and(0 <= C12, C12 <= np.minimum(C11, C22))
        drawRegion(
            ax,
            region=region_mask.astype(float),
            grid_size=grid_size,
            zoom=zoom_val,
            alpha=0.3,
        )


def drawAllVariations(
    ax,
    C,
    grid_size,
    depth=0,
    zoom=1,
    transformation=None,
    drawn=None,
    **kwargs,
):
    # Initialize the dedup set once at the top-level call
    if drawn is None:
        drawn = set()

    def _hash_metric(M):
        """Make a stable hashable key for a metric or batch of metrics.
        We round to 12 decimals to avoid tiny numeric noise duplications."""
        A = np.asarray(M)
        return tuple(np.round(A.reshape(-1), 12))

    def _maybe_draw(MC):
        key = _hash_metric(MC)
        if key in drawn:
            return False
        drawn.add(key)
        drawC(
            ax,
            MC,
            grid_size=grid_size,
            zoom=zoom,
            transformation=transformation,
            **kwargs,
        )
        return True

    # Ensure dtype consistency for symmetry generators
    dt = np.asarray(C).dtype
    m1 = np.array([[1, 0], [0, -1]], dtype=dt)
    m2 = np.array([[0, 1], [1, 0]], dtype=dt)

    # Draw the base and a few simple symmetries
    _maybe_draw(C)
    _maybe_draw(conTrans(C, m1))
    _maybe_draw(conTrans(C, m2))
    _maybe_draw(conTrans(conTrans(C, m1), m2))

    # Recurse via generators corresponding to up/right moves
    if depth > 0:
        drawAllVariations(
            ax,
            up(C),
            grid_size,
            depth=depth - 1,
            zoom=zoom,
            transformation=transformation,
            drawn=drawn,
            **kwargs,
        )
        drawAllVariations(
            ax,
            right(C),
            grid_size,
            depth=depth - 1,
            zoom=zoom,
            transformation=transformation,
            drawn=drawn,
            **kwargs,
        )
    return drawn


def drawShearPath(ax, **kwargs):
    nr = 1000
    one = np.array([1] * nr)

    t = np.sinh(np.linspace(np.arcsinh(0.001), np.arcsinh(300), nr))

    C = np.array([[one, -t], [-t, one + t**2]]).transpose(2, 0, 1)
    drawC(ax, C, linestyle="--", **kwargs)

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
    # drawFundamentalDomain(ax, grid_size=grid_size, zoom=zoom)

    # Draw shear path
    # drawShearPath(ax, grid_size=grid_size, zoom=zoom, linestyle="-")

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
    ax.set_xlabel(f"← Tall {nbs * 6} Wide →")
    # $P_y$(Length ratio and $\\theta - \\pi/2$)
    ax.set_ylabel(f"← Large angle {nbs * 6} Small angle →")
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


def plotPoincareDisk(ax=None, fig=None, save=True, grid_size=200, depth=5):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    zoom = 1
    transformation = "triangular"

    drawPoincareGrid(
        ax,
        grid_size=grid_size,
        zoom=zoom,
        depth=depth,
        c="gray",
        transformation=transformation,
    )

    # drawSquareElasticDomain(
    #     ax=ax,
    #     grid_size=grid_size,
    #     zoom=zoom,
    #     c="red",
    #     transformation=transformation,
    #     linewidth=1,
    # )
    drawTriangularElasticDomain(
        ax=ax,
        grid_size=grid_size,
        zoom=zoom,
        c="green",
        transformation=transformation,
        linewidth=1.5,
        shade=True,
    )

    drawFundamentalDomain(
        ax,
        grid_size=grid_size,
        zoom=zoom,
        linewidth=1.5,
        c="black",
        # More spacing between dashes
        linestyle=(0, (3, 6)),
        transformation=transformation,
    )

    # drawShearPath(
    #     ax,
    #     grid_size=grid_size,
    #     zoom=zoom,
    #     linewidth=1.5,
    #     c="blue",
    #     transformation=transformation,
    # )

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

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1, 1, 5).round(2),
    )
    ax.set_yticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1, 1, 5).round(2),
    )
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")

    plt.tight_layout()
    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        plt.savefig("Plots/poincareDisk.pdf", dpi=500)
        print("Saved plot to Plots/poincareDisk.pdf")
        plt.show()


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

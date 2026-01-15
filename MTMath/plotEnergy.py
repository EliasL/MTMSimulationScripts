import numpy as np


from .energyFunction import EnergyFunction, ContiEnergy, lagrange_reduction, SShear, F_from_C
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
    resolution=500,
    zoom=1,
    returnMask=False,
    transformation=None,
    eps=1e-9,
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
    mask = (X**2 + Y**2) >= (1 - eps)
    X[mask] = np.nan
    Y[mask] = np.nan

    C = poincareDisk2C(X, Y, transformation=transformation)

    if returnMask:
        return C, mask
    return C


def generate_energy_grid(
    E_func=EnergyFunction, beta=-0.25, K=4, energy_lim=[None, 0.37], **kwargs
):
    return generate_grid(
        E_func.energy_from_C_in_place, beta=beta, K=K, lim=energy_lim, **kwargs
    )


def generate_cauchy_stress_grid(E_func=EnergyFunction, beta=-0.25, K=4, **kwargs):
    return generate_grid(E_func.cauchy_from_C, beta=beta, K=K, **kwargs)


def generate_piola_stress_grid(
    E_func=EnergyFunction, beta=-0.25, K=4, second_PK=True, **kwargs
):
    if second_PK:
        return generate_grid(E_func.S_from_C, beta=beta, K=K, **kwargs)
    else:
        return generate_grid(E_func.P_from_C, beta=beta, K=K, **kwargs)


def generate_grid(
    function,
    resolution=500,
    zoom=1,
    lim=None,
    return_XY=False,
    poincareDisk=True,
    eps=1e-9,
    **kwargs,
):
    x_min, x_max = 0, 1
    y_min, y_max = -0.5, 0.5
    # Poicare disk
    if poincareDisk:
        C = generate_poincare_disk(resolution, zoom, eps=eps)
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

    grid = function(C, **kwargs)

    if lim is not None:
        if lim[0] is None:
            lim[0] = np.nanmin(grid)
        elif lim[1] is None:
            lim[1] = np.nanmax(grid)

        grid = np.clip(grid, *lim)

    if return_XY:
        # We don't need to have nan in X and Y, only in the energy grid
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        return grid, X, Y
    else:
        return grid


def generate_angle_region(resolution=500, zoom=1):
    C, r_mask = generate_poincare_disk(resolution, zoom, returnMask=True)
    # Create a boolean mask for the region that should be transparent
    mask = (C[..., 0, 1] > 1) | (C[..., 0, 1] < 0)
    # Create a float array (e.g., filled with ones) with the same shape as the mask
    region = np.ones_like(mask, dtype=float)
    # Set the parts where the mask is True to np.nan
    region[mask | r_mask] = np.nan
    return region


def C2PoincareDisk(C, transformation=None, eps=1e-12):
    """
    Map a symmetric 2x2 matrix C to (x, y) on the Poincaré disk by:
      (i)  normalizing C so det(C)=1 (if det>0),
      (ii) projecting the normalized matrix to (x,y).

    Supports a single 2x2 or a batch of shape (..., 2, 2).
    Returns x, y
    """
    C = np.asarray(C)
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


def poincareDisk2C(X, Y, transformation=None, eps=1e-12):
    r = 1.0 - X**2 - Y**2
    if np.any(r < 0):
        raise ValueError("Point outside of disk!")
    safe_r = np.where(np.abs(r) <= eps, np.nan, r)
    t = 2.0 / safe_r
    C11 = t * (1.0 + X) - 1.0
    C22 = t * (1.0 - X) - 1.0
    C12 = t * Y

    C = np.stack(
        [
            np.stack([C11, C12], axis=-1),  #
            np.stack([C12, C22], axis=-1),
        ],
        axis=-2,
    )

    C = transformC(C, transformation, inverse=True)
    return C


def transformC(C, transformation, inverse=False):
    if transformation is None:
        return C
    elif isinstance(transformation, np.ndarray):
        # Use the provided matrix directly as a congruence transform
        # (broadcasts over C if C has a leading dimension)
        M = transformation
    elif transformation.lower() == "none":
        return C
    elif transformation == "triangular":
        gamma = (4 / 3) ** (1 / 4)
        # pre_M = gamma*np.array([[-1.0, 0.0], [0.5, -np.sqrt(3) / 2]])
        pre_M = gamma * np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
        pre_M = np.linalg.inv(pre_M)
        # Optional: go the other direction
        # pre_M = -pre_M
        M = pre_M
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    if inverse:
        Minv = np.linalg.inv(M)
        return conTrans(C, Minv)
    else:
        return conTrans(C, M)


def drawF(ax=None, F=None, transformation=None, leftApplied=False, **kwargs):
    if transformation is not None:
        if leftApplied:
            F = transformation @ F
        else:
            F = F @ transformation
    C = F.swapaxes(-1, -2) @ F
    # Pass F through so drawC can use it for debugging/inspection
    return drawC(ax=ax, C=C, F=F, **kwargs)


def drawC(
    ax=None,
    C=None,
    grid_size=200,
    zoom=1,
    c: (str | None) = "black",
    linestyle="-",
    linewidth=0.6,
    transformation=None,
    scatter=False,
    arrow=False,
    label=None,
    shade=False,
    shadeColor=None,
    shade_values=None,  # scalar field, same indexing as C
    cmap="coolwarm",  # colormap
    cbarLims=None,  # colorbar limits when using shade_values
    agg="mean",  # aggregation method when using shade_values
    F=None,  # optional: underlying deformation gradients for debugging
    **kwargs,
):
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size, zoom=zoom)

    x, y = C2PoincareDisk(C, transformation=transformation)

    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return ax

    x_plot = x * zoom * grid_size / 2 + grid_size / 2
    y_plot = y * zoom * grid_size / 2 + grid_size / 2

    plt_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("label_")}

    if scatter:
        ax.scatter(x_plot, y_plot, c=c, **plt_kwargs)
        xm = x_plot
        ym = y_plot
    elif arrow:
        # Expect two endpoints
        assert len(x_plot) == len(y_plot) == 2, "Arrow mode requires two points (x,y)."
        ax.annotate(
            "",
            xy=(x_plot[1], y_plot[1]),
            xytext=(x_plot[0], y_plot[0]),
            arrowprops=dict(
                arrowstyle="-|>", mutation_scale=20, color=c, linewidth=linewidth
            ),
        )
        xm = [(x_plot[0] + x_plot[1]) / 2]
        ym = [(y_plot[0] + y_plot[1]) / 2]
    elif shade:
        # --- convert all valid points to pixel indices ---
        xv = x_plot[valid]
        yv = y_plot[valid]

        ix = np.rint(xv).astype(int)
        iy = np.rint(yv).astype(int)

        mask = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
        ix = ix[mask]
        iy = iy[mask]

        if shade_values is None:
            # old binary mask behaviour
            pixels = np.zeros((grid_size, grid_size), dtype=float)
            np.add.at(pixels, (iy, ix), 1.0)
            pixels = (pixels > 0).astype(float)
            drawRegion(
                ax,
                region=pixels,
                grid_size=grid_size,
                zoom=zoom,
                label=label,
                c=shadeColor if shadeColor is not None else c,
                **kwargs,
            )
        else:
            # --- scalar field case: build a float grid ---
            # align shade_values with valid & mask
            vals = shade_values[valid]
            vals = vals[mask]

            pixels = np.full((grid_size, grid_size), np.nan, dtype=float)

            # Temporary check of values. Remove later.
            # Do the overlapping values have similar magnitudes?
            # nan-safe max
            if False:
                tmp = np.full_like(pixels, -np.inf)
                np.maximum.at(tmp, (iy, ix), vals)
                tmp[tmp == -np.inf] = np.nan
                max_vals = tmp
                sum_ = np.zeros_like(pixels)
                cnt_ = np.zeros_like(pixels)
                np.add.at(sum_, (iy, ix), vals)
                np.add.at(cnt_, (iy, ix), 1.0)
                average_vals = np.where(cnt_ > 0, sum_ / cnt_, np.nan)
                print(
                    "Shade values aggregation check: max vs mean difference statistics:"
                )
                diff = np.abs(max_vals - average_vals)
                print(
                    "  max - mean (min, max, mean):  ",
                    np.nanmin(diff),
                    np.nanmax(diff),
                    np.nanmean(diff),
                )
                print("max nr overlapping values per pixel:", np.nanmax(cnt_))

                # --- Debug information: locate pixels with largest max-vs-mean difference ---
                # Only consider pixels where we truly have overlap (more than one value)
                overlap_mask = (cnt_ > 1) & np.isfinite(diff)
                if np.any(overlap_mask):
                    # Indices (iy, ix) of pixels with overlap
                    overlap_indices = np.argwhere(overlap_mask)
                    diff_flat = diff[overlap_mask]

                    # How many pixels to inspect (up to 3)
                    k = min(3, diff_flat.size)
                    # Indices of top-k differences (sorted descending)
                    top_k_idx = np.argsort(diff_flat)[-k:][::-1]
                    top_pixels = overlap_indices[top_k_idx]

                    print(
                        "Top pixels by |max-mean| difference (only where overlap occurs):"
                    )

                    # If F is available, align it with the same valid/mask selection as vals
                    F_vals = None
                    if F is not None:
                        try:
                            F_vals = F[valid]
                            F_vals = F_vals[mask]
                        except Exception as e:
                            print("Warning: could not align F with shade_values:", e)

                    for rank, (py, px) in enumerate(top_pixels, start=1):
                        d_val = diff[py, px]
                        count_here = int(cnt_[py, px])
                        print(
                            f"  #{rank}: pixel (ix={px}, iy={py}), diff={d_val}, count={count_here}"
                        )

                        # Collect all samples that landed in this pixel
                        same_pixel = (ix == px) & (iy == py)
                        pixel_vals = vals[same_pixel]
                        print("    shade_values at this pixel:", pixel_vals)

                        if F_vals is not None:
                            pixel_F = F_vals[same_pixel]
                            print(
                                "    Associated F values (one per overlapping point):"
                            )
                            for i_F, Fmat in enumerate(pixel_F):
                                # Fmat expected shape (2,2)
                                print(f"      F[{i_F}]:\n{Fmat}")

                        # Also mark these pixels visually with circles on the plot
                        try:
                            circ = Circle(
                                (px + 0.5, py + 0.5),
                                radius=3.0,
                                edgecolor="black",
                                facecolor="none",
                                linewidth=1.5,
                                zorder=10,
                            )
                            ax.add_patch(circ)
                        except Exception as e:
                            print("Warning: could not draw debug circle:", e)

            if agg == "max":
                # nan-safe max
                tmp = np.full_like(pixels, -np.inf)
                np.maximum.at(tmp, (iy, ix), vals)
                tmp[tmp == -np.inf] = np.nan
                pixels = tmp
            elif agg == "mean":
                sum_ = np.zeros_like(pixels)
                cnt_ = np.zeros_like(pixels)
                np.add.at(sum_, (iy, ix), vals)
                np.add.at(cnt_, (iy, ix), 1.0)
                pixels = np.full_like(sum_, np.nan, dtype=float)
                np.divide(sum_, cnt_, out=pixels, where=cnt_ > 0)
            else:
                raise ValueError(f"Unknown agg mode: {agg}")

            # You can either implement color support in drawRegion,
            # or plot directly with imshow here
            im = ax.imshow(
                pixels,
                origin="lower",
                extent=(0, grid_size, 0, grid_size),
                cmap=cmap,
                interpolation="nearest",
            )
            # Optional: add a colorbar
            if cbarLims is not None:
                im.set_clim(*cbarLims)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    else:
        ax.plot(
            x_plot,
            y_plot,
            c=c,
            linewidth=linewidth,
            linestyle=linestyle,
            **plt_kwargs,
        )
        return ax  # No label in line mode

    if label is not None and not shade:
        # If we shade, the label is taken care of in drawRegion
        # ensure x and y are numbers (not arrays) and scatter is True
        if isinstance(label, str):
            label = [label]
            xm = np.asarray(xm)
            ym = np.asarray(ym)

        assert len(label) == xm.size, "Number of labels does not match number of points"

        for x, y, lab in zip(xm, ym, label):
            addLabel(ax, x, y, lab, **kwargs)
    return ax


def drawRegion(ax, region, grid_size=200, zoom=1, label=None, **kwargs):
    # Support two modes:
    # (A) Solid-color mode if `c` or `color` is provided → render an RGBA image with that color and given alpha
    # (B) Colormap mode (fallback) → same as before, but respect `alpha` and keep NaNs fully transparent

    # Pull optional styling from kwargs
    color_kw = kwargs.pop("c", None)
    if color_kw is None:
        color_kw = kwargs.pop("color", None)
    if color_kw is None:
        color_kw = "green"
    alpha_kw = float(kwargs.pop("alpha", 1.0))

    arr = np.asarray(region, dtype=float)
    # Mask of pixels to show (True where region is present)
    mask = np.isfinite(arr) & (arr > 0)

    # Compute extent in pixel coordinates so everything aligns with other layers
    extent = [
        (grid_size / 2) * (1 - 1 / zoom),
        (grid_size / 2) * (1 + 1 / zoom),
        (grid_size / 2) * (1 - 1 / zoom),
        (grid_size / 2) * (1 + 1 / zoom),
    ]
    xmin, xmax, ymin, ymax = extent

    # Build an RGBA image where only mask==True pixels carry the given color+alpha; others are fully transparent
    from matplotlib import colors as _mcolors

    rgba = np.zeros(mask.shape + (4,), dtype=float)
    r, g, b, _ = _mcolors.to_rgba(color_kw, alpha=1.0)
    rgba[mask] = (r, g, b, alpha_kw)

    img = ax.imshow(
        rgba,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        zorder=0,
    )

    # ---- Label at area-weighted (equal per pixel) centroid ----
    if label is not None:
        if np.any(mask):
            ny, nx = mask.shape
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            # pixel centers
            xs = xmin + (np.arange(nx) + 0.5) * dx
            ys = ymin + (np.arange(ny) + 0.5) * dy

            iy, ix = np.where(mask)
            xc = xs[ix].mean()
            yc = ys[iy].mean()

            addLabel(ax, xc, yc, label, **kwargs)

    return img


def addLabel(
    ax,
    x,
    y,
    label,
    label_x=0,
    label_y=0,
    label_ha="left",  # left, right, center
    label_va="bottom",  # top, bottom, center
    label_color="black",
    label_fontsize=16,
    label_bbox=None,
    label_zorder=3,
    **kwargs,
):
    ax.text(
        x + label_x,
        y + label_y,
        label,
        ha=label_ha,
        va=label_va,
        fontsize=label_fontsize,
        color=label_color,
        bbox=label_bbox,
        zorder=label_zorder,
    )


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


def getCFundamental(grid_size=200, zoom_val=1, transformation=None, returnMask=False):
    C = generate_poincare_disk(grid_size, zoom_val, transformation=transformation)
    # Get the grid from C (NxMx2x2)
    r_mask = np.ones(C.shape[:2], dtype=bool)  # NxM

    C11 = C[..., 0, 0]
    C12 = C[..., 0, 1]
    C22 = C[..., 1, 1]

    # Region: 0 <= C12 <= min(C11, C22)
    r_mask = np.logical_and(r_mask, 0 < C12)
    r_mask = np.logical_and(r_mask, C12 < C11 / 2)
    r_mask = np.logical_and(r_mask, C11 <= C22)
    # Set C outside of fundamental domain to nan
    C[np.logical_not(r_mask), :, :] = np.nan
    if returnMask:
        return C, r_mask
    return C


def getFFundamental(grid_size=200, zoom_val=1, transformation=None, returnMask=False):
    C, r_mask = getCFundamental(grid_size, zoom_val, transformation, returnMask=True)
    F = F_from_C(C)
    if returnMask:
        return F, r_mask
    return F


def drawFundamentalDomain(ax, shade=False, **kwargs):
    #  0<=C12<=C11/2, C11<=C22
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

    if shade:
        if "transformation" in kwargs:
            transformation = kwargs["transformation"]

        grid_size = kwargs.get("grid_size", 200)
        zoom_val = kwargs.get("zoom", 1)
        C, r_mask = getCFundamental(
            grid_size, zoom_val, transformation, returnMask=True
        )
        drawC(ax, C, shade=True, **kwargs)

        # drawRegion(
        #     ax,
        #     region=r_mask.astype(float),
        #     **kwargs,
        # )


def drawPoincareGrid(ax=None, grid_size=200, zoom=1, depth=6, **kwargs):
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size, zoom=zoom)
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
    return ax


def drawFVectors(ax=None, F=None, scale=0.2, margin=0.05, **kwargs):
    """
    Draw the two column vectors of F as arrows in a small inset
    in the top-right corner of `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Axes to draw into. If None, a new figure and axes are created.
    F : array_like, shape (2, 2)
        Linear map whose columns are drawn as vectors.
    scale : float
        Fraction of the main axes width/height occupied by the inset
        (in axes coordinates).
    margin : float
        Offset from the top and right edges, in axes coordinates.
    **kwargs :
        Passed on to `Axes.annotate` for the arrows (e.g. zorder, linewidth,
        arrowprops, etc.). If `arrowprops` is not provided, a default is used.
    """
    import numpy as _np

    # Ensure we have an axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Default F is identity
    if F is None:
        F = _np.eye(2)
    F = _np.asarray(F, dtype=float)
    if F.shape != (2, 2):
        raise ValueError(f"drawFVectors expects a (2,2) matrix, got shape {F.shape}")

    # Create an inset in axes coordinates in the top-right corner
    # [left, bottom, width, height] all in axes fraction
    left = 1.0 - margin - scale
    bottom = 1.0 - margin - scale
    width = scale
    height = scale
    inset = ax.inset_axes([left, bottom, width, height])

    # Make inset clean: no frame, ticks, or labels
    inset.set_axis_off()
    inset.patch.set_alpha(0.0)
    inset.set_aspect("equal", adjustable="box")

    cols = [F[:, 0], F[:, 1]]

    # Norms of the two column vectors
    norms = _np.linalg.norm(F, axis=0)  # shape (2,)
    max_norm = float(_np.max(norms)) if norms.size > 0 else 0.0

    # Enforce a minimum "visual" scale so tiny F is still visible
    base_scale = max(1.0, max_norm)

    L = base_scale

    inset.set_xlim(-L, L)
    inset.set_ylim(-L, L)

    # Prepare arrow properties
    user_arrowprops = kwargs.pop("arrowprops", None)
    default_arrowprops = dict(arrowstyle="-|>", lw=1.5)
    if user_arrowprops is not None:
        arrowprops = {**default_arrowprops, **user_arrowprops}
    else:
        arrowprops = default_arrowprops

    # Colors for the two basis vectors if user did not specify one
    base_colors = kwargs.pop("colors", None)
    if base_colors is None:
        base_colors = ["C0", "C1"]

    # Draw each column as an arrow from the origin
    origin = (0.0, 0.0)
    for idx, v in enumerate(cols):
        c = base_colors[idx % len(base_colors)]
        # Allow per-vector override while keeping user kwargs
        this_arrowprops = dict(arrowprops)
        if "color" not in this_arrowprops and "edgecolor" not in this_arrowprops:
            this_arrowprops["color"] = c
        inset.annotate(
            "",
            xy=(v[0], v[1]),
            xytext=origin,
            arrowprops=this_arrowprops,
            **kwargs,
        )

    return ax


def drawCircles(ax=None, F=None, applyFromLeft=True, grid_size=200, zoom=1, dot=False):
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size, zoom=zoom)
    h = 500  # nr of energy well jumps
    q = 200  # quality of curve
    # vals has many values close to 0, and fewer larger values
    vals = np.sinh(np.linspace(np.arcsinh(-h), np.arcsinh(h), q))

    def Sx(g):
        return SShear(g, 0)

    def Sy(g):
        return SShear(g, np.pi / 2)

    def Sxy(g):
        return SShear(g, np.pi / 4)

    def Sxy2(g):
        return SShear(g, 3 * np.pi / 4)

    moves = (Sx, Sy, Sxy, Sxy2)
    c1 = "#06923E"
    c2 = "#F4991A"
    colors = (c1, c1, c2, c2)
    for i, M, c in zip(range(len(moves)), moves, colors):
        S = M(vals)
        if applyFromLeft:
            pert_F = S @ F
        else:
            pert_F = F @ S
        C = np.einsum("...ji,...jk->...ik", pert_F, pert_F)
        dash = i % 2 == 1
        drawC(
            ax,
            C,
            grid_size=grid_size,
            zoom=zoom,
            c=c,
            linestyle="--" if dash else "-",
            linewidth=5,
        )

    if dot:
        drawC(ax, F.T @ F, scatter=True, c="blue", zorder=10)


def conTrans(C, M):
    """Apply a congruence transform: return M^T @ C @ M.

    Works for C with shape (2, 2) or (N, 2, 2).
    M can be (2, 2) for a single transform broadcast over all slices, or (N, 2, 2)
    to use a different matrix per slice (matching the leading dimension of C).
    """
    C_arr = np.asarray(C)
    M_arr = np.asarray(M, dtype=C_arr.dtype)

    return np.swapaxes(M_arr, -1, -2) @ C_arr @ M_arr


def _m3_const(dtype_str):
    dt = np.dtype(dtype_str)
    # Shear matrix [[1, -1], [0, 1]] used in up/right moves
    return np.array([[1, -1], [0, 1]], dtype=dt)


def _m2_const(dtype_str):
    dt = np.dtype(dtype_str)
    # Shear matrix [[1, -1], [0, 1]] used in up/right moves
    return np.array([[0, -1], [1, 0]], dtype=dt)


# See SL2(Z)  KEITH CONRAD
def get_T(C):
    """Return the canonical m3 with the proper dtype for C."""
    return _m3_const(np.asarray(C).dtype.str)


def T(C):
    M = get_T(C)  # shape (2,2), broadcasts over slices of C
    return conTrans(C, M)


def T_inv(C):
    M = np.linalg.inv(get_T(C))
    return conTrans(C, M)


def get_S(C):
    """Return the canonical m3 with the proper dtype for C."""
    return _m2_const(np.asarray(C).dtype.str)


def S(C):
    M = get_S(C)
    return conTrans(C, M)


def up(C):
    M = get_T(C)  # shape (2,2), broadcasts over slices of C
    return conTrans(C, M)


def down(C):
    M = np.linalg.inv(get_T(C))
    return conTrans(C, M)


def right(C):
    M = get_T(C).T
    return conTrans(C, M)


def left(C):
    M = np.linalg.inv(get_T(C).T)
    return conTrans(C, M)


def upInv(C):
    return np.linalg.inv(up(C))


def rightInv(C):
    return np.linalg.inv(right(C))


def applyTransformations(F, transformations):
    right = np.array([[1, 1], [0, 1]], dtype=int)
    left = np.array([[1, -1], [0, 1]], dtype=int)
    up = np.array([[1, 0], [1, 1]], dtype=int)
    down = np.array([[1, 0], [-1, 1]], dtype=int)

    # Transformations is a string like "RULDD"
    for char in transformations.lower():
        if char == "r":
            F = F @ right
        elif char == "l":
            F = F @ left
        elif char == "u":
            F = F @ up
        elif char == "d":
            F = F @ down
    return F


def applyCongruenceTransformations(C_, transformations):
    C = C_.copy()
    right = np.array([[1, 1], [0, 1]], dtype=int)
    left = np.array([[1, -1], [0, 1]], dtype=int)
    up = np.array([[1, 0], [1, 1]], dtype=int)
    down = np.array([[1, 0], [-1, 1]], dtype=int)

    # Transformations is a string like "RULDD"
    for char in transformations.lower():
        if char == "r":
            C = right.T @ C @ right
        elif char == "l":
            C = left.T @ C @ left
        elif char == "u":
            C = up.T @ C @ up
        elif char == "d":
            C = down.T @ C @ down
    return C


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


def drawReconnectionDomain(ax, shade=False, **kwargs):
    nr = 1000

    zero = np.array([0] * nr)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)

    def draw(C):
        drawC(ax, C, **kwargs)

    draw(C)  # I
    draw(S(C))
    draw(S(T(C)))
    draw(S(T(S(C))))
    draw(T_inv(S(C)))
    draw(T_inv(C))

    # Shade the region defined by 0 <= C12 <= min(C11, C22)
    # Shading does not work with transformations yet
    transformation = kwargs.get("transformation", None)
    if shade:  # shade and transformation is None:
        grid_size = kwargs.get("grid_size", 200)
        zoom_val = kwargs.get("zoom", 1)

        # C has shape (grid, grid, 2, 2); r_mask is True on/outside the rim
        G, r_mask = generate_poincare_disk(
            grid_size, zoom_val, returnMask=True, transformation=transformation
        )
        r_mask = np.zeros_like(r_mask, dtype=bool)

        a = G[..., 0, 0]
        b = G[..., 0, 1]
        c = G[..., 1, 1]

        case_ac = a <= c

        mask_ac = (b >= -3 * a / 2) & (b <= a / 2) & (b >= -(3 * a + c) / 4)

        mask_ca = (b >= -3 * c / 2) & (b <= c / 2) & (b >= -(a + 3 * c) / 4)

        r_mask = np.where(case_ac, mask_ac, mask_ca)

        drawRegion(
            ax,
            region=r_mask.astype(float),
            grid_size=grid_size,
            zoom=zoom_val,
            alpha=0.3,
        )


def drawTriangularElasticDomain(ax, shade=False, **kwargs):
    nr = 1000

    zero = np.array([0] * nr)

    # HORIZONTAL LINE
    # Values from -1<t<1 are outside of the circle
    t = np.sinh(np.linspace(np.arcsinh(0.0000001), np.arcsinh(1), nr))
    # det=1, C12=C21, C12=0
    G = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)

    def draw(G):
        drawC(ax, G, **kwargs)

    # See SL2(Z)  KEITH CONRAD page 3
    # draw(C)  # I
    # draw(S(C))
    # draw(S(T_inv(C)))
    # draw(T(S(T(C))))
    # draw(T(S(C)))
    # draw(T(C))

    draw(G)  # I
    draw(S(G))
    draw(S(T(G)))
    draw(S(T(S(G))))
    draw(T_inv(S(G)))
    draw(T_inv(G))

    # Shade the region defined by 0 <= C12 <= min(C11, C22)
    # Shading does not work with transformations yet
    transformation = kwargs.get("transformation", None)
    if True:  # shade and transformation is None:
        grid_size = kwargs.get("grid_size", 200)
        zoom_val = kwargs.get("zoom", 1)

        # C has shape (grid, grid, 2, 2); r_mask is True on/outside the rim
        G, r_mask = generate_poincare_disk(
            grid_size, zoom_val, returnMask=True, transformation=transformation
        )
        r_mask = np.ones_like(r_mask, dtype=bool)

        G11 = G[..., 0, 0]
        G12 = G[..., 0, 1]
        G22 = G[..., 1, 1]

        # Region: 0 <= C12 <= min(C11, C22)
        r_mask = np.logical_and(r_mask, 0 <= G12)
        r_mask = np.logical_and(r_mask, G12 <= G11)
        r_mask = np.logical_and(r_mask, G12 <= G22)

        drawRegion(
            ax,
            region=r_mask.astype(float),
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


def generateShearTransformations(depth, startingPoint=None, leftApplied=True):
    # Generate all unique combinations of up, down, left, right moves up to the given depth
    # Remove duplicate transformations using a set
    transformation_keys = set()
    transformations = []
    labels = []

    def recurse(F, current_depth, current_label=""):
        key = tuple(np.round(F.reshape(-1), 12))
        if key not in transformation_keys:
            transformation_keys.add(key)
            transformations.append(F)
            if current_label == "":
                labels.append(r"$\mathbf{I}$")
            else:
                labels.append(current_label)
        if current_depth == 0:
            return

        for label in ["r", "l", "u", "d"]:
            # SShear can take these string directions as an argument. Sorry if
            # that's a bit confusing
            if leftApplied:
                recurse(
                    SShear(label) @ F,
                    current_depth - 1,
                    current_label=current_label + rf"${label}$",
                )
            else:
                print("Warning, using right applied shears!")
                recurse(
                    F @ SShear(label),
                    current_depth - 1,
                    current_label=current_label + label,
                )

    if startingPoint is None:
        # Start with the identity matrix
        startingPoint = np.array([[1, 0], [0, 1]], dtype=float)

    recurse(startingPoint, depth)
    return transformations, labels


def drawShearPath(ax, **kwargs):
    nr = 1000
    one = np.array([1] * nr)

    t = np.sinh(np.linspace(np.arcsinh(0.001), np.arcsinh(300), nr))

    C = np.array([[one, -t], [-t, one + t**2]]).transpose(2, 0, 1)
    drawC(ax, C, linestyle="--", **kwargs)

    C = np.array([[one, t], [t, one + t**2]]).transpose(2, 0, 1)
    drawC(ax, C, **kwargs)


def drawUnitCircle(ax, grid_size, zoom=1):
    circle_size = (grid_size / 2) * zoom
    circle_center_x = grid_size / 2
    circle_center_y = grid_size / 2

    circle = Circle(
        (circle_center_x, circle_center_y),
        circle_size,
        color="black",
        fill=False,
        linewidth=1,
    )
    ax.add_patch(circle)


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
    drawUnitCircle(ax, grid_size=grid_size, zoom=zoom)

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


def prepPoincareFig(grid_size=200, zoom=1, ax=None, withGrid=True):
    # Zoom does not always work properly. Be careful
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
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
    ax.add_patch(circle)

    if withGrid:
        drawPoincareGrid(
            ax,
            grid_size=grid_size,
            zoom=zoom,
            c="gray",
        )

    ax.set_xticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1 / zoom, 1 / zoom, 5).round(2),
    )
    ax.set_yticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1 / zoom, 1 / zoom, 5).round(2),
    )
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")
    ax.set_aspect("equal")
    return fig, ax


def plotPoincareDisk(ax=None, save=True, grid_size=200, depth=5, transformation="none"):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size, withGrid=False)
    zoom = 1

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

    plt.tight_layout()
    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        plt.savefig("Plots/poincareDisk.pdf", dpi=500)
        print("Saved plot to Plots/poincareDisk.pdf")
        plt.show()


def plotPoincareCTiling(
    ax=None,
    save=True,
    grid_size=1000,
    depth=2,
    quadrants="abcd",
    show=False,
    use_labels=True,
    leftApplied=False,
    arrows=False,
):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size)
    zoom = 1

    Fs, labels = generateShearTransformations(depth=depth, leftApplied=leftApplied)

    swap = np.array([[0, 1], [1, 0]], dtype=int)  # col1 <- col1 + col2
    flip = np.array([[-1, 0], [0, 1]], dtype=int)  # col1 <- col1 - col2

    # Draw fundamental domain
    def df(c, trans, label=None, label_va="center"):
        drawFundamentalDomain(
            ax,
            grid_size=grid_size,
            zoom=zoom,
            c=c,
            shade=True,
            linewidth=0,
            transformation=trans,
            alpha=0.3,
            label=label if use_labels else None,
            label_va=label_va if quadrants == "abcd" else "center",
        )

    for t, L in zip(Fs, labels):
        if "a" in quadrants:
            df("green", t, label=rf"$1{L}$", label_va="bottom")
        if L == r"\mathbf{I}":
            L = ""
        if "b" in quadrants:
            df("blue", t @ swap, label=rf"$2{L}$", label_va="bottom")
        if "c" in quadrants:
            df("red", t @ flip, label=rf"$3{L}$", label_va="top")
        if "d" in quadrants:
            df("purple", t @ swap @ flip, label=rf"$4{L}$", label_va="top")

    if arrows and False:
        center_point = np.array([[0.86, 0.20], [0.20, 1.21]])  # Random central point
        points_to_transform = [center_point]
        nr_points = len(points_to_transform)
        for i in range(depth):
            for t in generateShearTransformations(i, leftApplied=leftApplied):
                for startC in points_to_transform:
                    drawC(
                        ax,
                        np.stack(
                            [
                                startC,
                            ]
                        ),
                    )

    # plt.tight_layout()
    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        path = f"Plots/poincareDisk_{'left' if leftApplied else 'right'}_{quadrants}_{depth}{'_lab' if use_labels else ''}.pdf"
        plt.savefig(path, dpi=500, bbox_inches="tight")

        print(f"Saved plot to {path}")
        if show:
            plt.show()

    return ax


def plotPoincareFTiling(
    ax=None,
    save=True,
    grid_size=1000,
    extra_grid=1,
    depth=2,
    quadrants="abcd",
    show=False,
    use_labels=True,
    leftApplied=False,
    arrows=False,
):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size)


    Fs, labels = generateShearTransformations(depth=depth, leftApplied=leftApplied)

    F = getFFundamental(grid_size=int(grid_size * extra_grid))
    for t, lab in zip(Fs, labels):
        drawF(
            ax,
            F,
            transformation=t,
            leftApplied=leftApplied,
            grid_size=grid_size,
            shade=True,
            label=lab if use_labels else None,
            alpha=0.3,
        )

    # plt.tight_layout()
    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        path = f"Plots/poincareDisk_{'left' if leftApplied else 'right'}_{quadrants}_{depth}{'_lab' if use_labels else ''}.pdf"
        plt.savefig(path, dpi=500, bbox_inches="tight")

        print(f"Saved plot to {path}")
        if show:
            plt.show()
    return ax


def plotPoincarePointMapping(
    ax=None,
    fig=None,
    save=True,
    grid_size=500,
    show=False,
):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size)

    swap = np.array([[0, 1], [1, 0]], dtype=int)  # col1 <- col1 + col2
    flip = np.array([[-1, 0], [0, 1]], dtype=int)  # col1 <- col1 - col2

    p1 = np.array([[1, 0.3], [0, 1]])
    p2 = np.array([[1, 0], [-0.3, 1]])
    p1 = p1.T @ p1
    p2 = p2.T @ p2

    def drawPoint(point, c, size=60, shape="o", label=None, **kwargs):
        drawC(
            ax,
            point,
            grid_size=grid_size,
            scatter=True,
            c=c,
            s=size,
            marker=shape,
            label=label,
            **kwargs,
        )

    def drawArrow(pa, pb, label=None, c="black", **kwargs):
        points = np.array([pa, pb])
        drawC(
            ax,
            points,
            grid_size=grid_size,
            arrow=True,
            c=c,
            label=label,
            linewidth=1,
            **kwargs,
        )

    # redPoints
    r1 = p1
    r2 = applyCongruenceTransformations(p1, "r")
    drawArrow(r1, r2, label="r", c="red")
    # bluePoints
    b1 = p2
    blu = applyCongruenceTransformations(p2, "lu")
    # intermediate points
    bu = applyCongruenceTransformations(p2, "u")
    bl = applyCongruenceTransformations(p2, "l")
    bul = applyCongruenceTransformations(p2, "ul")
    # Show arrows
    drawArrow(b1, bu, label="u", c="gray", linestyle="--", label_ha="right")
    drawArrow(bu, bul, label="l", c="gray", linestyle="--")
    drawArrow(b1, bl, label="l", c="gray", linestyle="--")
    drawArrow(bl, blu, label="u", c="gray", linestyle="--")

    drawPoint(p1, "red", shape="s", label=r"$\mathrm{A}_0$")
    drawPoint(p2, "blue", shape="s", label=r"$\mathrm{B}_0$")
    drawPoint(
        applyCongruenceTransformations(p1, "r"), "red", label="A", label_ha="right"
    )
    drawPoint(applyCongruenceTransformations(p2, "lu"), "blue", size=20, label="B")
    drawPoint(applyCongruenceTransformations(p2, "u"), "gray")
    drawPoint(applyCongruenceTransformations(p2, "l"), "gray")
    drawPoint(applyCongruenceTransformations(p2, "ul"), "green", label="C")

    # plt.tight_layout()
    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        path = "Plots/poincareDiskPointMapping.pdf"
        plt.savefig(path, dpi=500, bbox_inches="tight")

        print(f"Saved plot to {path}")
        if show:
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

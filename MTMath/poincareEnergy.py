import numpy as np
import hashlib
import os
import pickle
from pathlib import Path

from .energyFunction import (
    EnergyFunction,
    ContiEnergy,
    PieceWiseQuadratic,
    SShear,
    F_from_C,
)
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Circle
import scipy.interpolate as interpolate
from matplotlib import colors
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm, PowerNorm

from .reduction import (
    elastic_domain_quadrant,
    plastic_reduction,
    plastic_reduction_history,
    lagrange_reduction_history,
)


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
    E_func: type[EnergyFunction] = ContiEnergy,
    beta=-0.25,
    K=4,
    energy_lim=[None, 0.37],
    **kwargs,
):
    return generate_grid(
        E_func.energy_from_C_in_place, beta=beta, K=K, lim=energy_lim, **kwargs
    )


def generate_stability_min_angle_grid(
    E_func: type[EnergyFunction] = ContiEnergy,
    beta=-0.25,
    K=4,
    energy_lim=[0, 1],
    boolStability=False,
    **kwargs,
):
    def minAngle(C, **kwargs):
        F = F_from_C(C)
        t = np.linspace(0, np.pi, 100, endpoint=False)
        n = np.stack([np.cos(t), np.sin(t)], axis=-1)
        angle, det = E_func.min_det_angle(F, n, **kwargs)
        if boolStability:
            return det
        else:
            return angle

    return generate_grid(minAngle, beta=beta, K=K, lim=energy_lim, **kwargs)


def approximate_ellipticity_boundary(
    E_func: type[EnergyFunction] = ContiEnergy,
    beta=-0.25,
    K=4,
    resolution=200,
    n_angles=60,
    zoom=1,
    transformation=None,
    loops=1000,
    eulerian=True,
    return_all=False,
    eps=1e-9,
):
    """
    Approximate the ellipticity-loss boundary by contouring the continuous
    minimum acoustic determinant on the Poincare disk. Returns a list of
    (x, y) points along the boundary curve (longest contour by default).

    If return_all=True, returns a list of curves, each a list of (x, y) points.
    """

    def _hashable(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, np.ndarray):
            return ("array", value.shape, value.dtype.str, value.tobytes())
        if isinstance(value, (list, tuple)):
            return ("seq", tuple(_hashable(v) for v in value))
        if isinstance(value, dict):
            return ("dict", tuple(sorted((k, _hashable(v)) for k, v in value.items())))
        return repr(value)

    key_payload = {
        "E_func": f"{E_func.__module__}.{E_func.__name__}",
        "beta": float(beta),
        "K": float(K),
        "resolution": int(resolution),
        "n_angles": int(n_angles),
        "zoom": float(zoom),
        "transformation": _hashable(transformation),
        "loops": int(loops),
        "eulerian": bool(eulerian),
        "eps": float(eps),
        "boundary_mode": "min_det_contour_v1",
    }

    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "ellipticity_boundary"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        pickle.dumps(key_payload, protocol=5)
    ).hexdigest()
    cache_path = cache_dir / f"ellipticity_boundary_{cache_key}.pkl"

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            curves = cached.get("curves", cached)
            if curves is None:
                curves = []
            if return_all:
                return curves
            return max(curves, key=len) if curves else []
        except Exception:
            pass

    radius = 1.0 / zoom
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    mask_outside = (X**2 + Y**2) >= (1 - eps)

    X_safe = X.copy()
    Y_safe = Y.copy()
    X_safe[mask_outside] = 0.0
    Y_safe[mask_outside] = 0.0

    C_grid = poincareDisk2C(X_safe, Y_safe, transformation=transformation, eps=eps)
    C_grid[mask_outside] = np.nan

    F = F_from_C(C_grid)

    theta = np.linspace(0, np.pi, n_angles, endpoint=False)
    n = np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    _, min_det = E_func.min_det_angle(
        F, n, beta=beta, K=K, loops=loops, eulerian=eulerian
    )
    stability_field = np.where(mask_outside, np.nan, min_det)
    stability_field = np.ma.masked_invalid(stability_field)

    fig_tmp, ax_tmp = plt.subplots()
    print(
        "Computing ellipticity boundary (may take several minutes). "
        "Result will be cached for future runs."
    )
    contour = ax_tmp.contour(X, Y, stability_field, levels=[0.0])
    paths_vertices = (
        [np.asarray(seg) for seg in contour.allsegs[0] if seg is not None]
        if contour.allsegs and contour.allsegs[0]
        else []
    )
    plt.close(fig_tmp)
    if not paths_vertices:
        return [] if not return_all else []

    def _clip_to_disk(xv, yv, eps_clip):
        r2 = xv * xv + yv * yv
        r = np.sqrt(np.maximum(r2, 0.0))
        mask = r >= (1 - eps_clip)
        if np.any(mask):
            scale = (1 - eps_clip) / np.where(r == 0, 1.0, r)
            xv = xv.copy()
            yv = yv.copy()
            xv[mask] = xv[mask] * scale[mask]
            yv[mask] = yv[mask] * scale[mask]
        return xv, yv

    curves = []
    for verts in paths_vertices:
        if verts.shape[0] < 2:
            continue
        xv, yv = _clip_to_disk(verts[:, 0], verts[:, 1], eps_clip=1e-6)
        curves.append(np.stack([xv, yv], axis=-1))

    if not curves:
        return [] if not return_all else []

    try:
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump({"curves": curves, "meta": key_payload}, f, protocol=5)
        os.replace(tmp_path, cache_path)
    except Exception:
        pass

    if return_all:
        return curves
    # Return the longest curve
    longest = max(curves, key=len)
    return longest


def generate_cauchy_stress_grid(
    E_func: type[EnergyFunction] = ContiEnergy, beta=-0.25, K=4, **kwargs
):
    return generate_grid(E_func.cauchy_from_C, beta=beta, K=K, **kwargs)


def generate_piola_stress_grid(
    E_func: type[EnergyFunction] = ContiEnergy,
    beta=-0.25,
    K=4,
    second_PK=True,
    **kwargs,
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
    transformation=None,
    eps=1e-9,
    **kwargs,
):
    x_min, x_max = 0, 1
    y_min, y_max = -0.5, 0.5
    # Poicare disk
    if poincareDisk:
        C = generate_poincare_disk(
            resolution,
            zoom,
            transformation=transformation,
            eps=eps,
        )
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


def generate_elastic_quadrant_grid(
    resolution=500,
    zoom=1,
    transformation=None,
    loops=1000,
):
    """Return well-quadrant identifiers over the Poincare disk.

    Values inside the disk are the integer labels 0--3 returned after plastic
    reduction.  Points outside the disk, and any unclassified points, are NaN
    so plotting code can leave the surrounding canvas transparent or white.
    """
    C, outside = generate_poincare_disk(
        resolution=resolution,
        zoom=zoom,
        returnMask=True,
        transformation=transformation,
    )
    C_reduced, _ = plastic_reduction(C, loops=loops)
    quadrants = elastic_domain_quadrant(C_reduced).astype(float)
    quadrants[outside | (quadrants < 0)] = np.nan
    return quadrants


def generate_angle_region(resolution=500, zoom=1):
    C, r_mask = generate_poincare_disk(resolution, zoom, returnMask=True)
    # Create a boolean mask for the region that should be transparent
    mask = (C[..., 0, 1] > 1) | (C[..., 0, 1] < 0)
    # Create a float array (e.g., filled with ones) with the same shape as the mask
    region = np.ones_like(mask, dtype=float)
    # Set the parts where the mask is True to np.nan
    region[mask | r_mask] = np.nan
    return region


def C2Plane(C, plane="PoincareDisk", transformation=None, eps=1e-12):
    match plane:
        case "PoincareDisk":
            return C2PoincareDisk(C, transformation=transformation, eps=eps)
        case "LogEuclideanPlane":
            return C2LogEuclideanPlane(C, transformation=transformation, eps=eps)
        case _:
            raise ValueError(f"No such transformation: {plane}")


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


def C2LogEuclideanPlane(C, transformation=None, eps=1e-12):
    """
    Map symmetric 2x2 C to a *flat* (Euclidean) plane via the matrix logarithm:
      (i)  normalize C so det(C)=1 (if det>0),
      (ii) compute L = log(C_hat) (symmetric, trace 0),
      (iii) return planar coordinates from L.

    Supports a single 2x2 or a batch of shape (..., 2, 2).
    Returns x, y
    """
    C = np.asarray(C)
    C = transformC(C, transformation)

    a = C[..., 0, 0]
    b = C[..., 0, 1]
    c = C[..., 1, 1]

    det = a * c - b * b
    valid = det > eps

    scale = np.empty_like(det, dtype=float)
    scale[valid] = np.sqrt(det[valid])
    scale[~valid] = np.nan
    C_hat = C / scale[..., None, None]  # det(C_hat)=1 where valid

    # --- Flat-plane projection via log(C_hat) using symmetric eigendecomp ---
    w, V = np.linalg.eigh(C_hat)  # w: (..., 2), V: (..., 2, 2)
    # Guard: log requires positive eigenvalues (SPD); invalid entries become nan anyway
    logw = np.log(w)
    L = V @ (logw[..., None] * np.swapaxes(V, -2, -1))  # V diag(logw) V^T

    # L is symmetric, trace ~ 0 when det=1
    x = L[..., 0, 0]
    y = L[..., 0, 1]  # off-diagonal

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
    c: str | None = "black",
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
    F=None,  # retained for compatibility with drawF callers
    **kwargs,
):
    """Draw one or more metric tensors in the configured coordinate plane.

    ``C`` may be a single ``(2, 2)`` tensor or a batch with shape
    ``(..., 2, 2)``.  Matplotlib treats the coordinates of a single tensor as
    scalars, so point-like modes normalize them to flat arrays before plotting
    and labeling.  This keeps scalar, list, and NumPy-array labels consistent.
    """
    if ax is None:
        _, ax = prepPoincareFig(grid_size=grid_size, zoom=zoom)

    x, y = C2Plane(C, transformation=transformation)

    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return ax

    x_plot = x * zoom * grid_size / 2 + grid_size / 2
    y_plot = y * zoom * grid_size / 2 + grid_size / 2

    # Text styling belongs to addLabel, not to Matplotlib's line/scatter
    # artists.  Accept the existing label_* convention as well as the common
    # unprefixed text aliases used by older call sites.
    text_aliases = {
        "fontsize": "label_fontsize",
        "ha": "label_ha",
        "va": "label_va",
        "bbox": "label_bbox",
    }
    label_kwargs = {k: v for k, v in kwargs.items() if k.startswith("label_")}
    for source, destination in text_aliases.items():
        if source in kwargs and destination not in label_kwargs:
            label_kwargs[destination] = kwargs[source]

    plt_kwargs = {
        k: v
        for k, v in kwargs.items()
        if not k.startswith("label_") and k not in text_aliases
    }

    # Scatter points and arrow midpoints are conceptually one-dimensional,
    # even when C has several leading batch dimensions.
    x_points = np.asarray(x_plot).reshape(-1)
    y_points = np.asarray(y_plot).reshape(-1)

    if scatter:
        ax.scatter(x_points, y_points, c=c, **plt_kwargs)
        label_x = x_points
        label_y = y_points
    elif arrow:
        if x_points.size != 2 or y_points.size != 2:
            raise ValueError("Arrow mode requires exactly two tensor endpoints")
        if not np.all(np.isfinite(x_points)) or not np.all(np.isfinite(y_points)):
            raise ValueError("Arrow endpoints must map to finite coordinates")

        arrowprops = {
            "arrowstyle": "-|>",
            "mutation_scale": 20,
            "color": "black" if c is None else c,
            "linewidth": linewidth,
            "linestyle": linestyle,
        }
        if "alpha" in plt_kwargs:
            arrowprops["alpha"] = plt_kwargs["alpha"]

        annotation_kwargs = {}
        if "zorder" in plt_kwargs:
            annotation_kwargs["zorder"] = plt_kwargs["zorder"]
        ax.annotate(
            "",
            xy=(x_points[1], y_points[1]),
            xytext=(x_points[0], y_points[0]),
            arrowprops=arrowprops,
            **annotation_kwargs,
        )
        label_x = np.array([(x_points[0] + x_points[1]) / 2])
        label_y = np.array([(y_points[0] + y_points[1]) / 2])
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
                **(plt_kwargs | label_kwargs),
            )
        else:
            # --- scalar field case: build a float grid ---
            # align shade_values with valid & mask
            try:
                values = np.broadcast_to(np.asarray(shade_values), np.shape(x_plot))
            except ValueError as exc:
                raise ValueError(
                    "shade_values must be scalar or match the leading shape of C"
                ) from exc
            vals = values[valid]
            vals = vals[mask]

            finite_values = np.isfinite(vals)
            ix = ix[finite_values]
            iy = iy[finite_values]
            vals = vals[finite_values]

            pixels = np.full((grid_size, grid_size), np.nan, dtype=float)

            if agg == "max":
                # nan-safe max
                tmp = np.full_like(pixels, -np.inf)
                np.maximum.at(tmp, (iy, ix), vals)
                tmp[tmp == -np.inf] = np.nan
                pixels = tmp
            elif agg == "min":
                tmp = np.full_like(pixels, np.inf)
                np.minimum.at(tmp, (iy, ix), vals)
                tmp[tmp == np.inf] = np.nan
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
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
        # A scalar label and a one-item iterable are equivalent.  Batched
        # points still require one label per point, as before, but now produce
        # a useful error instead of failing while iterating a NumPy scalar.
        labels = np.asarray(label, dtype=object)
        if labels.ndim == 0:
            labels = labels.reshape(1)
        else:
            labels = labels.reshape(-1)

        if labels.size != label_x.size:
            raise ValueError(
                f"Received {labels.size} label(s) for {label_x.size} plotted point(s)"
            )

        for x_value, y_value, point_label in zip(label_x, label_y, labels):
            if np.isfinite(x_value) and np.isfinite(y_value):
                addLabel(
                    ax,
                    x_value,
                    y_value,
                    point_label,
                    **label_kwargs,
                )
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
    # generate_poincare_disk samples [-1/zoom, 1/zoom], while drawC multiplies
    # projected coordinates by zoom.  The resulting image therefore always
    # spans the full plotting canvas, independently of zoom.
    extent = [0, grid_size, 0, grid_size]
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
    density_method="hist",
    density_grid_size=400,
    show_colorbar=True,
    alpha=None,
    zorder=None,
    **scatter_kwargs,
):
    x, y = C2Plane(C, plane="PoincareDisk", transformation=transformation)
    # Filter out invalid points
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return

    # Create a density estimate (default: histogram counts)
    density_method = (density_method or "hist").lower()
    if density_method == "hist":
        bins = int(density_grid_size) if density_grid_size else 400
        r = 1.0 / zoom
        scale = bins / (2.0 * r)
        xi = ((x + r) * scale).astype(int)
        yi = ((y + r) * scale).astype(int)
        xi = np.clip(xi, 0, bins - 1)
        yi = np.clip(yi, 0, bins - 1)
        hist = np.zeros((bins, bins), dtype=int)
        np.add.at(hist, (xi, yi), 1)
        density1 = hist[xi, yi]
    elif density_method == "kde":
        # KDE is slower but can be enabled explicitly
        xy = np.vstack([x, y])
        bandwidth = len(x) ** (-1 / 6)  # Scott rule
        try:
            if x.size < 2:
                raise np.linalg.LinAlgError("insufficient points for KDE")
            kde = gaussian_kde(xy, bw_method=bandwidth)
            density1 = kde(xy)
        except np.linalg.LinAlgError:
            # Assign a uniform value to make all points appear red
            density1 = np.ones_like(x) * 1e10  # High value to map to red
    else:
        raise ValueError(f"Unknown density_method: {density_method}")

    cmap = "inferno"
    if remove_max_color:
        coolwarm = cm.get_cmap(cmap, 256)  # 256 colors
        newcolors = coolwarm(np.linspace(0, 1, 256))
        n = 2
        newcolors[-n:, -1] = np.linspace(1, 0, n) ** (1 / 2)
        cmap = colors.ListedColormap(newcolors)

    if vmax is None:
        vmax = len(C)

    # Check if log scale is to be applied
    norm = None
    if log_scale:
        # LogNorm requires vmax > vmin (vmin=1)
        vmax = max(vmax, 2)
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
    extra_kwargs = {}
    if zorder is not None:
        extra_kwargs["zorder"] = zorder
    extra_kwargs.update(scatter_kwargs)

    scatter = ax.scatter(
        x * zoom * grid_size / 2 + grid_size / 2,
        y * zoom * grid_size / 2 + grid_size / 2,
        c=density1,
        s=sizes,
        linewidth=0,
        cmap=cmap,
        norm=norm,
        vmax=vmax,
        **extra_kwargs,
    )
    if density_method == "kde":
        cbar_label = "Kernel density estimate"
    else:
        cbar_label = "Bin counts"
    scatter.set_alpha(alpha)
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, label=cbar_label, pad=0.01)
        if hasattr(cbar, "solids") and cbar.solids is not None:
            cbar.solids.set_alpha(1.0)
    return scatter


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
        r_mask = np.logical_and(r_mask, G12 <= G11/2)
        r_mask = np.logical_and(r_mask, G12 <= G22/2)

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


def drawPoincareSymmetryPoints(
    ax,
    grid_size=200,
    depth=5,
    zoom=1,
    transformation=None,
    color="gray",
    alpha=0.7,
    square_size=42,
    triangular_size=46,
    linewidth=0.9,
    zorder=14,
):
    """Draw square- and triangular-lattice symmetry points on the disk.

    The square point is the identity metric ``C = I``. The triangular point
    is the determinant-one metric with ``C11 = C22 = 2/sqrt(3)`` and
    ``C12 = 1/sqrt(3)``. ``drawAllVariations`` tiles both points using the
    same symmetry moves as the Lagrange-reduction grid.
    """
    square_C = np.eye(2, dtype=float)
    triangular_C = np.array(
        [
            [2.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
            [1.0 / np.sqrt(3.0), 2.0 / np.sqrt(3.0)],
        ]
    )

    common_kwargs = {
        "grid_size": grid_size,
        "depth": depth,
        "zoom": zoom,
        "transformation": transformation,
        "scatter": True,
        "c": None,
        "facecolors": "none",
        "edgecolors": color,
        "alpha": alpha,
        "linewidths": linewidth,
        "zorder": zorder,
    }
    drawAllVariations(
        ax,
        square_C,
        marker="s",
        s=square_size,
        **common_kwargs,
    )
    drawAllVariations(
        ax,
        triangular_C,
        marker="^",
        s=triangular_size,
        **common_kwargs,
    )


def generateShearTransformations(depth, startingPoint=None, leftApplied=True):
    """Generate unique shear transforms and their raw path labels.

    Labels are returned as unformatted path strings. The empty string is the
    identity transform; non-empty values contain only move letters, such as
    ``"rld"``. Immediate inverse moves are not followed, so paths do not
    backtrack. Plotting functions apply any mathtext formatting they need.
    """
    transformation_keys = set()
    transformations = []
    labels = []

    def recurse(F, current_depth, current_label=""):
        key = tuple(np.round(F.reshape(-1), 12))
        if key not in transformation_keys:
            transformation_keys.add(key)
            transformations.append(F)
            labels.append(current_label)
        if current_depth == 0:
            return

        inverse = {"r": "l", "l": "r", "u": "d", "d": "u"}
        previous_label = current_label[-1:] if current_label else None
        for label in ["r", "l", "u", "d"]:
            if label == inverse.get(previous_label):
                continue
            # SShear can take these string directions as an argument. Sorry if
            # that's a bit confusing
            if leftApplied:
                recurse(
                    SShear(label) @ F,
                    current_depth - 1,
                    current_label=current_label + label,
                )
            else:
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

def plotPoincareLines(
    ax=None,
    save=True,
    grid_size=200,
    depth=5,
    transformation="none",
    show=False,
):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(
            grid_size=grid_size,
            withGrid=True,
            withYieldSurface=False,
        )

    nr = 1000
    zero = np.zeros(nr)

    # Base parameter range
    t = np.sinh(
        np.linspace(
            np.arcsinh(0.0000001),
            np.arcsinh(2 / np.sqrt(3)),
            nr,
        )
    )

    # This is T(C0), where C0 = diag(t, 1/t)
    C = np.array(
        [
            [t, t],
            [t, t + 1 / t],
        ]
    ).transpose(2, 0, 1)
    drawC(ax, C, grid_size, c="red")

    # This is S(T(C0))
    C = np.array(
        [
            [t + 1 / t, t],
            [t, t],
        ]
    ).transpose(2, 0, 1)
    drawC(ax, C, grid_size, c="blue")


    t = np.sinh(
        np.linspace(
            np.arcsinh(0.0000001),
            np.arcsinh(1),
            nr,
        )
    )

    C = np.array([[t, zero], [zero, 1 / t]]).transpose(2, 0, 1)
    drawC(ax, C, grid_size, c="green")
    C = np.array([[1 / t, zero], [zero, t]]).transpose(2, 0, 1)
    drawC(ax, C, grid_size, c="yellow")



    return ax

def plotEnergyField(
    energy_grid,
    fig=None,
    ax=None,
    save=True,
    add_title=True,
    zoom=1,
    remove_max_color=True,
    scale=1.0,
    withYieldSurface=True,
    withGrid=True,
    grid_depth=6,
    withFundamentalDomain=False,
    fundamentalDomain_kwargs=None,
    fundamentalDomain_label=None,
    fundamentalDomain_label_xy=(-0.35, 0.12),
    fundamentalDomain_label_kwargs=None,
    withSymmetryPoints=False,
    symmetryPoints_kwargs=None,
    minimalTicks=False,
    transformation=None,
    yieldSurface_kwargs=None,
    show_colorbar_cap=True,
    colorbar_label="Energy",
    output_path="energy_field.pdf",
):
    grid_size = len(energy_grid)

    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

    prepPoincareFig(
        grid_size=grid_size,
        zoom=zoom,
        ax=ax,
        withCircle=True,
        withGrid=withGrid,
        grid_depth=grid_depth,
        minimalTicks=minimalTicks,
        withYieldSurface=withYieldSurface,
        transformation=transformation,
        yieldSurface_kwargs=yieldSurface_kwargs,
    )

    max_energy = np.nanmax(energy_grid)

    cmap = "coolwarm"
    if remove_max_color:
        coolwarm = cm.get_cmap(cmap, 256)  # 256 colors
        newcolors = coolwarm(np.linspace(0, 1, 256))
        n = 2
        newcolors[-n:, -1] = np.linspace(1, 0, n) ** (1 / 2)
        cmap = colors.ListedColormap(newcolors)
    if scale is None:
        scale = 1.0
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Scale must be positive. Got {scale}.")
    norm = PowerNorm(gamma=scale)

    img = ax.imshow(
        energy_grid,
        cmap=cmap,
        origin="lower",
        norm=norm,
        extent=(0, grid_size , 0, grid_size),
        zorder=0,
    )

    cbar = fig.colorbar(img, label=colorbar_label, pad=-0.01)
    vmin, vmax = img.get_clim()
    if np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax:
        t = np.linspace(0.0, 1.0, 7)
        # Evenly spaced in color space; invert PowerNorm to data space.
        ticks = vmin + (vmax - vmin) * (t ** (1.0 / scale))
        cbar.set_ticks(ticks)
        cbar.formatter = ticker.FormatStrFormatter("%.2f")
        cbar.update_ticks()
    if show_colorbar_cap:
        default_font_size = plt.rcParams["font.size"]  # Fetch default font size
        cbar.ax.set_title(
            f"Capped at ${max_energy}$", fontsize=default_font_size, loc="left"
        )

    if withFundamentalDomain:
        domain_kwargs = {
            "grid_size": grid_size,
            "zoom": zoom,
            "transformation": transformation,
            "c": "black",
            "linewidth": 1.8,
            "zorder": 12,
        }
        if fundamentalDomain_kwargs:
            domain_kwargs.update(fundamentalDomain_kwargs)
        drawFundamentalDomain(ax, **domain_kwargs)

        if fundamentalDomain_label is not None:
            label_kwargs = {
                "color": "black",
                "fontsize": 20,
                "ha": "center",
                "va": "center",
                "zorder": 16,
            }
            if fundamentalDomain_label_kwargs:
                label_kwargs.update(fundamentalDomain_label_kwargs)
            label_x, label_y = fundamentalDomain_label_xy
            center = grid_size / 2
            half = grid_size / 2
            ax.text(
                center + zoom * half * label_x,
                center + zoom * half * label_y,
                fundamentalDomain_label,
                **label_kwargs,
            )

    if withSymmetryPoints:
        point_kwargs = {
            "grid_size": grid_size,
            "zoom": zoom,
            "transformation": transformation,
            "depth": grid_depth,
            "color": "gray",
            "alpha": 0.7,
            "square_size": 42,
            "triangular_size": 46,
            "linewidth": 0.9,
            "zorder": 14,
        }
        if symmetryPoints_kwargs:
            point_kwargs.update(symmetryPoints_kwargs)
        drawPoincareSymmetryPoints(ax, **point_kwargs)

    if add_title:
        ax.set_title("Energy field in a Poincaré disk")

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
        )
        print(f"Saved plot to {output_path.resolve()}")
    return ax


def prepPoincareFig(
    grid_size=200,
    zoom=1,
    ax=None,
    withCircle=True,
    withGrid=True,
    grid_color="gray",
    grid_depth=6,
    minimalTicks=False,
    withYieldSurface=True,
    transformation=None,
    yieldSurface_kwargs=None,
):
    # Zoom does not always work properly. Be careful
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    if withCircle:
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
            zorder=100,
        )
        ax.add_patch(circle)

    if withGrid:
        drawPoincareGrid(
            ax,
            grid_size=grid_size,
            zoom=zoom,
            depth=grid_depth,
            c=grid_color,
            alpha=0.7,
            zorder=1,
            transformation=transformation,
        )
    if withYieldSurface:
        ys_kwargs = {
            "E_func": ContiEnergy,
            "beta": -0.25,
            "K": 4,
            "resolution": max(80, grid_size),
            "n_angles": 50,
            "zoom": zoom,
            "transformation": transformation,
        }
        if yieldSurface_kwargs:
            ys_kwargs.update(yieldSurface_kwargs)
        boundary_xy = approximate_ellipticity_boundary(**ys_kwargs)

        if boundary_xy is not None and len(boundary_xy) > 0:
            xy = np.asarray(boundary_xy)
            if xy.ndim == 2 and xy.shape[1] == 2:
                C_boundary = poincareDisk2C(
                    xy[:, 0],
                    xy[:, 1],
                    transformation=transformation,
                )
                tile_depth = 4
                if yieldSurface_kwargs and "tile_depth" in yieldSurface_kwargs:
                    tile_depth = int(yieldSurface_kwargs["tile_depth"])
                drawAllVariations(
                    ax,
                    C_boundary,
                    grid_size,
                    depth=tile_depth,
                    zoom=zoom,
                    transformation=transformation,
                    c="white",
                    linewidth=0.7,
                    alpha=0.3,
                    zorder=10,
                )
            else:
                x_plot = xy[:, 0] * zoom * grid_size / 2 + grid_size / 2
                y_plot = xy[:, 1] * zoom * grid_size / 2 + grid_size / 2
                ax.plot(
                    x_plot,
                    y_plot,
                    color="white",
                    linewidth=0.8,
                    alpha=0.8,
                    label="Yield surface",
                    zorder=10,
                )
    center = grid_size / 2
    half = grid_size / 2
    tick_pos = np.linspace(center - half, center + half, 5)
    # Map tick positions back to Poincare coordinates using the same scaling as the data layer
    tick_lab = ((tick_pos - center) / (zoom * half)).round(2)

    if minimalTicks:
        ax.set_xticks(tick_pos, [""] * len(tick_pos))
        ax.set_yticks(tick_pos, [""] * len(tick_pos))
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_frame_on(False)
    else:
        ax.set_xticks(tick_pos, tick_lab)
        ax.set_yticks(tick_pos, tick_lab)
        ax.set_xlabel(r"$x_p$")
        ax.set_ylabel(r"$y_p$")

    ax.set_xlim(center - half, center + half)
    ax.set_ylim(center - half, center + half)
    ax.set_aspect("equal")
    return fig, ax


def plot_reduction_history(
    F,
    ax=None,
    histories=None,
    resolution=500,
    grid_depth=6,
    transformation=None,
    show_grid=True,
    show_colorbar=False,
    show_legend=True,
    show_axes=True,
    colorbar_label=None,
    lagrange_color="#9DFA9B",
    plastic_color="#00940F",
    grid_color="#555555",
    linewidth=2.0,
    white_background=True,
):
    """Plot reduction histories in configuration space.

    Parameters are deliberately Matplotlib-oriented so this function is useful
    both for static figures and for callers such as the interactive reduction
    visualizer.  ``F`` is a single 2x2 deformation gradient.  By default the
    Lagrange and unit-step plastic-reduction histories are shown.  Pass
    ``histories`` as ``(history, color, label)`` triples to compare other paths.
    """
    F = np.asarray(F, dtype=float)
    if F.shape != (2, 2):
        raise ValueError("F must have shape (2, 2)")
    if not np.all(np.isfinite(F)) or abs(np.linalg.det(F)) <= 1e-12:
        raise ValueError("F must be finite and invertible")

    if ax is None:
        fig, ax = prepPoincareFig(
            grid_size=resolution,
            withCircle=False,
            withGrid=False,
            withYieldSurface=False,
        )
    else:
        fig = ax.get_figure()

    if white_background:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

    C = F.T @ F
    quadrant_grid = generate_elastic_quadrant_grid(
        resolution=resolution,
        transformation=transformation,
    )
    quadrant_cmap = colors.ListedColormap(
        plt.colormaps["coolwarm"](np.linspace(0, 1, 4)),
        name="well_quadrants",
    )
    quadrant_cmap.set_bad("white" if white_background else (0, 0, 0, 0))
    quadrant_norm = colors.BoundaryNorm(np.arange(-0.5, 4.5), 4)
    image = ax.imshow(
        quadrant_grid,
        origin="lower",
        extent=(0, resolution, 0, resolution),
        interpolation="nearest",
        cmap=quadrant_cmap,
        norm=quadrant_norm,
        zorder=0,
    )

    if show_grid:
        drawPoincareGrid(
            ax=ax,
            grid_size=resolution,
            depth=grid_depth,
            c=grid_color,
            alpha=0.55,
            linewidth=0.45,
            transformation=transformation,
            zorder=1,
        )

    if histories is None:
        histories = (
            (lagrange_reduction_history(C), lagrange_color, "Lagrange reduction"),
            (plastic_reduction_history(C), plastic_color, "Plastic reduction"),
        )
    else:
        histories = tuple(histories)

    for history, color, _ in histories:
        x, y = C2PoincareDisk(history, transformation=transformation)
        points = np.column_stack(
            (
                x * resolution / 2 + resolution / 2,
                y * resolution / 2 + resolution / 2,
            )
        )
        for start, end in zip(points[:-1], points[1:]):
            annotation = ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": linewidth,
                    "mutation_scale": 14,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=4,
            )


    start_x, start_y = C2PoincareDisk(C, transformation=transformation)
    ax.scatter(
        start_x * resolution / 2 + resolution / 2,
        start_y * resolution / 2 + resolution / 2,
        s=50,
        c=plastic_color,
        edgecolors=lagrange_color,
        linewidths=1.5,
        zorder=6,
    )

    circle = Circle(
        (resolution / 2, resolution / 2),
        resolution / 2,
        fill=False,
        edgecolor=grid_color,
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(circle)

    if show_colorbar:
        colorbar = fig.colorbar(
            image,
            ax=ax,
            ticks=np.arange(4),
            fraction=0.02,
            pad=-0.05,
            shrink=0.20,
            anchor=(0, 0.03),
        )
        if colorbar_label:
            colorbar.set_label(colorbar_label)

    if show_legend:
        handles = [
            Line2D([0], [0], color=color, linewidth=linewidth, label=label)
            for _, color, label in histories
        ]
        ax.legend(handles=handles, loc="upper left")

    ax.set_xlim(0, resolution)
    ax.set_ylim(0, resolution)
    ax.set_aspect("equal")
    if not show_axes:
        ax.set_axis_off()
    return fig, ax


def plotPoincareDisk(ax=None, save=True, grid_size=200, depth=5, transformation="none", show=False):
    # Make plot of fundamental domain
    if ax is None:
        fig, ax = prepPoincareFig(grid_size=grid_size, withGrid=False, withYieldSurface=False)
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
    if show:
        plt.show()


def transformed_fundamental_domain_inequalities(transformation):
    """Return the three linear inequalities for a transformed C-domain.

    The fundamental domain is defined by ``C12 > 0``,
    ``C11 / 2 - C12 > 0``, and ``C22 - C11 >= 0``.  For the tile
    ``M.T @ D @ M``, a metric ``C`` lies in the tile when
    ``M^{-T} @ C @ M^{-1}`` satisfies those inequalities.  Each returned row
    contains the coefficients of one inequality in the vector
    ``(C11, C12, C22)``.
    """
    M = np.asarray(transformation, dtype=float)
    if M.shape != (2, 2):
        raise ValueError("transformation must have shape (2, 2)")
    if abs(np.linalg.det(M)) <= 1e-12:
        raise ValueError("transformation must be invertible")

    inverse = np.linalg.inv(M)
    first_column = inverse[:, 0]
    second_column = inverse[:, 1]

    def bilinear_coefficients(left, right):
        return np.array(
            [
                left[0] * right[0],
                left[0] * right[1] + left[1] * right[0],
                left[1] * right[1],
            ],
            dtype=float,
        )

    C11 = bilinear_coefficients(first_column, first_column)
    C12 = bilinear_coefficients(first_column, second_column)
    C22 = bilinear_coefficients(second_column, second_column)
    return np.stack((C12, C11 / 2 - C12, C22 - C11))


def _fundamental_domain_inequality_mask(C, coefficients, tolerance=1e-12):
    """Evaluate transformed fundamental-domain inequalities on ``C``."""
    C11 = C[..., 0, 0]
    C12 = C[..., 0, 1]
    C22 = C[..., 1, 1]
    first = (
        coefficients[0, 0] * C11
        + coefficients[0, 1] * C12
        + coefficients[0, 2] * C22
    )
    second = (
        coefficients[1, 0] * C11
        + coefficients[1, 1] * C12
        + coefficients[1, 2] * C22
    )
    third = (
        coefficients[2, 0] * C11
        + coefficients[2, 1] * C12
        + coefficients[2, 2] * C22
    )
    return (first > tolerance) & (second > tolerance) & (third >= -tolerance)


def _canonical_congruence_transform(transformation, decimals=12):
    """Canonicalize ``M`` and ``-M``, which induce the same C-domain."""
    canonical = np.round(np.asarray(transformation, dtype=float), decimals=decimals)
    significant = np.flatnonzero(np.abs(canonical).ravel() > 10 ** (-decimals))
    if significant.size == 0:
        raise ValueError("transformation must be non-zero")
    if canonical.ravel()[significant[0]] < 0:
        canonical = -canonical
    return tuple(canonical.ravel())


def _generate_shear_transformation_path_counts(
    depth,
    leftApplied,
    collect_labels=False,
):
    """Aggregate non-backtracking path words up to ``depth`` by their matrix."""
    if depth < 0:
        raise ValueError("depth must be non-negative")

    entries = {}

    def recurse(transform, remaining_depth, path):
        key = tuple(np.round(transform.reshape(-1), 12))
        if key not in entries:
            entries[key] = {
                "transform": transform.copy(),
                "path_count": 0,
                "path_labels": [path],
            }
        entry = entries[key]
        if collect_labels and entry["path_count"] > 0:
            entry["path_labels"].append(path)
        entry["path_count"] += 1
        if remaining_depth == 0:
            return

        inverse = {"r": "l", "l": "r", "u": "d", "d": "u"}
        previous_move = path[-1:] if path else None
        for move in "rlud":
            if move == inverse.get(previous_move):
                continue
            shear = SShear(move)
            next_transform = (
                shear @ transform if leftApplied else transform @ shear
            )
            recurse(next_transform, remaining_depth - 1, path + move)

    recurse(np.eye(2), depth, "")
    return list(entries.values())


def _poincare_tiling_specifications(
    depth,
    quadrants,
    leftApplied,
    deduplicate_domains=True,
    count_paths=False,
    collect_labels=False,
):
    valid_quadrants = set("abcd")
    requested_quadrants = set(quadrants)
    if not requested_quadrants or not requested_quadrants <= valid_quadrants:
        raise ValueError("quadrants must be a non-empty subset of 'abcd'")

    swap = np.array([[0, 1], [1, 0]], dtype=float)
    flip = np.array([[-1, 0], [0, 1]], dtype=float)
    quadrant_transforms = {
        "a": np.eye(2),
        "b": swap,
        "c": flip,
        "d": swap @ flip,
    }
    if count_paths:
        path_entries = _generate_shear_transformation_path_counts(
            depth=depth,
            leftApplied=leftApplied,
            collect_labels=collect_labels,
        )
    else:
        transformations, labels = generateShearTransformations(
            depth=depth,
            leftApplied=leftApplied,
        )
        path_entries = [
            {
                "transform": transform,
                "path_count": 1,
                "path_labels": [label],
            }
            for transform, label in zip(transformations, labels)
        ]

    specifications = []
    seen_domains = set()
    for path_entry in path_entries:
        transform = path_entry["transform"]
        path_label = path_entry["path_labels"][0]
        for quadrant in "abcd":
            if quadrant not in requested_quadrants:
                continue
            tile_transform = transform @ quadrant_transforms[quadrant]
            key = _canonical_congruence_transform(tile_transform)
            if deduplicate_domains and key in seen_domains:
                continue
            seen_domains.add(key)
            specifications.append(
                {
                    "quadrant": quadrant,
                    "label": path_label,
                    "path_count": path_entry["path_count"],
                    "path_labels": path_entry["path_labels"],
                    "transform": tile_transform,
                    "inequalities": transformed_fundamental_domain_inequalities(
                        tile_transform
                    ),
                }
            )
    return specifications


def generatePoincareCTilingRegions(
    depth=2,
    quadrants="abcd",
    leftApplied=False,
    grid_size=500,
    zoom=1,
    tolerance=1e-12,
):
    """Classify Poincare-disk pixels into transformed fundamental domains.

    Unlike :func:`plotPoincareCTiling`, this creates the disk grid only once.
    It evaluates precomputed linear inequalities for each candidate tile and
    returns an integer tile-id image together with tile metadata.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    specifications = _poincare_tiling_specifications(
        depth=depth,
        quadrants=quadrants,
        leftApplied=leftApplied,
        deduplicate_domains=True,
    )
    C, outside_disk = generate_poincare_disk(
        resolution=grid_size,
        zoom=zoom,
        returnMask=True,
    )
    C11 = C[..., 0, 0]
    C12 = C[..., 0, 1]
    C22 = C[..., 1, 1]
    tile_ids = np.full(C11.shape, -1, dtype=int)
    unassigned = (
        ~outside_disk
        & np.isfinite(C11)
        & np.isfinite(C12)
        & np.isfinite(C22)
    )

    for tile_id, specification in enumerate(specifications):
        inside = _fundamental_domain_inequality_mask(
            C,
            specification["inequalities"],
            tolerance=tolerance,
        ) & unassigned
        tile_ids[inside] = tile_id
        unassigned[inside] = False

    return tile_ids, specifications


def generatePoincareCTilingMultiplicity(
    depth=2,
    quadrants="abcd",
    leftApplied=False,
    grid_size=500,
    zoom=1,
    tolerance=1e-12,
    collect_labels=False,
):
    """Count how many generated labeled domains contain each disk pixel.

    Every generated transformation is counted, including transformations that
    induce the same C-domain (such as ``M`` and ``-M``), because each such
    transformation corresponds to a separate path label in the tiling plot.
    The returned centroids allow the caller to place those labels without
    recomputing the domain masks.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    specifications = _poincare_tiling_specifications(
        depth=depth,
        quadrants=quadrants,
        leftApplied=leftApplied,
        deduplicate_domains=False,
        count_paths=True,
        collect_labels=collect_labels,
    )
    C, outside_disk = generate_poincare_disk(
        resolution=grid_size,
        zoom=zoom,
        returnMask=True,
    )
    valid_pixels = (
        ~outside_disk
        & np.isfinite(C[..., 0, 0])
        & np.isfinite(C[..., 0, 1])
        & np.isfinite(C[..., 1, 1])
    )
    total_counts = np.zeros(C.shape[:2], dtype=np.int32)
    counts_by_quadrant = {
        quadrant: np.zeros(C.shape[:2], dtype=np.int32)
        for quadrant in "abcd"
        if quadrant in set(quadrants)
    }
    centroids = []

    for specification in specifications:
        inside = _fundamental_domain_inequality_mask(
            C,
            specification["inequalities"],
            tolerance=tolerance,
        ) & valid_pixels
        total_counts += specification["path_count"] * inside
        counts_by_quadrant[specification["quadrant"]] += (
            specification["path_count"] * inside
        )
        row_indices, column_indices = np.where(inside)
        if row_indices.size:
            centroids.append(
                (
                    column_indices.mean() + 0.5,
                    row_indices.mean() + 0.5,
                )
            )
        else:
            centroids.append(None)

    return total_counts, counts_by_quadrant, specifications, centroids


def _draw_poincare_multiplicity(ax, counts, grid_size, color, base_alpha=0.3):
    """Draw repeated-domain opacity equivalent to stacked alpha overlays."""
    rgba = np.zeros(counts.shape + (4,), dtype=float)
    red, green, blue, _ = colors.to_rgba(color)
    positive = counts > 0
    rgba[positive, :3] = (red, green, blue)
    rgba[positive, 3] = 1.0 - (1.0 - base_alpha) ** counts[positive]
    ax.imshow(
        rgba,
        origin="lower",
        extent=(0, grid_size, 0, grid_size),
        interpolation="nearest",
        zorder=0,
    )


def plotPoincareCTilingInequalities(
    ax=None,
    save=True,
    grid_size=1000,
    depth=2,
    quadrants="abcd",
    show=False,
    use_labels=True,
    leftApplied=False,
    withGrid=True,
    grid_depth=6,
    withYieldSurface=False,
):
    """Plot a C-space tiling using one final disk rasterization pass."""
    if ax is None:
        _, ax = prepPoincareFig(
            grid_size=grid_size,
            withGrid=withGrid,
            grid_depth=grid_depth,
            withYieldSurface=withYieldSurface,
        )

    _, counts_by_quadrant, specifications, centroids = generatePoincareCTilingMultiplicity(
        depth=depth,
        quadrants=quadrants,
        leftApplied=leftApplied,
        grid_size=grid_size,
        collect_labels=use_labels,
    )
    quadrant_colors = {
        "a": "green",
        "b": "blue",
        "c": "red",
        "d": "purple",
    }

    for quadrant in "abcd":
        if quadrant not in counts_by_quadrant:
            continue
        _draw_poincare_multiplicity(
            ax,
            counts=counts_by_quadrant[quadrant],
            grid_size=grid_size,
            color=quadrant_colors[quadrant],
        )

    if use_labels:
        for specification, centroid in zip(specifications, centroids):
            if centroid is None:
                continue
            quadrant = specification["quadrant"]
            label_va = "center"
            if quadrants == "abcd":
                label_va = "bottom" if quadrant in "ab" else "top"
            for path_label in specification["path_labels"]:
                addLabel(
                    ax,
                    centroid[0],
                    centroid[1],
                    rf"${'abcd'.index(quadrant) + 1}{path_label}$",
                    label_va=label_va,
                )

    if save:
        import os

        if not os.path.exists("Plots"):
            os.makedirs("Plots")
        path = (
            f"Plots/poincareDisk_{'left' if leftApplied else 'right'}_"
            f"{quadrants}_{depth}{'_lab' if use_labels else ''}.pdf"
        )
        plt.savefig(path, dpi=500, bbox_inches="tight")
        print(f"Saved plot to {path}")
    if show:
        plt.show()
    return ax


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
        label_body = L
        if "a" in quadrants:
            df("green", t, label=rf"$1{label_body}$", label_va="bottom")
        if "b" in quadrants:
            df("blue", t @ swap, label=rf"$2{label_body}$", label_va="bottom")
        if "c" in quadrants:
            df("red", t @ flip, label=rf"$3{label_body}$", label_va="top")
        if "d" in quadrants:
            df("purple", t @ swap @ flip, label=rf"$4{label_body}$", label_va="top")

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
        if lab == "":
            lab = r"$\mathbf{I}$"
        else:
            lab = rf"${lab}$"
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
    output_path="Plots/3DEnergy.png",
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=500)
    print(f"Saved plot to {output_path.resolve()}")
    plt.show()


def crop_white_border(
    image_path,
    output_path=None,
    white_threshold=255,
):
    """Crop only the fully white outer border from a raster image.

    White rows and columns inside the image are preserved. The image is
    replaced in place when ``output_path`` is omitted.

    Returns
    -------
    pathlib.Path
        The path of the cropped image.
    """
    from PIL import Image

    image_path = Path(image_path)
    output_path = image_path if output_path is None else Path(output_path)
    white_threshold = int(white_threshold)
    if not 0 <= white_threshold <= 255:
        raise ValueError(
            f"white_threshold must be between 0 and 255, got {white_threshold}."
        )

    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGBA"))

    is_white = np.all(pixels[..., :3] >= white_threshold, axis=-1)
    nonwhite_rows = np.flatnonzero(~np.all(is_white, axis=1))
    nonwhite_columns = np.flatnonzero(~np.all(is_white, axis=0))
    if not nonwhite_rows.size or not nonwhite_columns.size:
        raise ValueError(f"Image contains no non-white pixels: {image_path}")

    row_start, row_end = nonwhite_rows[[0, -1]]
    column_start, column_end = nonwhite_columns[[0, -1]]
    cropped = pixels[row_start : row_end + 1, column_start : column_end + 1]
    Image.fromarray(cropped, mode="RGBA").save(output_path)
    return output_path


def remove_blank_rows_and_columns(
    image_path,
    output_path=None,
    white_threshold=255,
):
    """Backward-compatible alias for :func:`crop_white_border`."""
    return crop_white_border(
        image_path,
        output_path=output_path,
        white_threshold=white_threshold,
    )


def plot3DEnergyDensityComparison(
    resolution=250,
    zoom=1.0,
    transformation=None,
    loops=1000,
    conti_beta=-0.25,
    conti_K=4.0,
    conti_noise=1.0,
    quadratic_energy_function=PieceWiseQuadratic,
    quadratic_kappa=1.0,
    quadratic_xi=1.0,
    quadratic_eta=1.0,
    energy_lim=(0.0, 0.37),
    data_radius=0.8,
    add_front_hole=True,
    z_scale=0.6,
    z_tick_count=4,
    elev=35.0,
    azim=80.0,
    roll=0.0,
    figsize=(10, 5),
    cmap="coolwarm",
    remove_max_color=True,
    surface_alpha=1.0,
    surface_kwargs=None,
    show_colorbars=True,
    show_grid=True,
    titles=(r"(a) $\Phi$", r"(b) $\Phi_{\mathrm{quad}}$"),
    z_axis_labels=(r"$\Phi$", r"$\Phi_{\mathrm{quad}}$"),
    title_y=0.95,
    title_pad=-12.0,
    output_path=None,
    dpi=300,
    save_pad_inches=0.02,
    autocrop_png=True,
    white_threshold=255,
    show=False,
):
    """Plot the square Conti and quadratic energy densities side by side.

    The energy surfaces use the same Poincare-disk coordinates and shared
    color/z limits, making the two landscapes directly comparable.  The
    default quadratic model is :class:`PieceWiseQuadratic`; pass
    ``SuperSimple`` as ``quadratic_energy_function`` for the simple quadratic
    model instead.

    Parameters most likely to be adjusted for a publication figure are
    ``resolution``, ``zoom``, ``energy_lim``, ``z_scale``, ``elev``, ``azim``,
    ``roll``, ``data_radius``, ``z_tick_count``, ``show_grid``, and
    ``surface_kwargs``. For PNG output, ``autocrop_png`` removes fully white
    the fully white outer border after saving. ``quadratic_kappa``,
    ``quadratic_xi``, and ``quadratic_eta`` are passed to
    ``PieceWiseQuadratic`` as its three coefficients.

    Returns
    -------
    fig, axes
        The Matplotlib figure and the two 3D axes.
    """
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError(f"resolution must be at least 2, got {resolution}.")
    zoom = float(zoom)
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}.")
    z_scale = float(z_scale)
    if z_scale <= 0:
        raise ValueError(f"z_scale must be positive, got {z_scale}.")
    z_tick_count = int(z_tick_count)
    if z_tick_count < 2:
        raise ValueError(
            f"z_tick_count must be at least 2, got {z_tick_count}."
        )
    if data_radius is not None and float(data_radius) <= 0:
        raise ValueError(
            f"data_radius must be positive or None, got {data_radius}."
        )
    if len(titles) != 2:
        raise ValueError("titles must contain exactly two labels.")
    if len(z_axis_labels) != 2:
        raise ValueError("z_axis_labels must contain exactly two labels.")

    # generate_grid currently returns plotting coordinates for its flat-plane
    # branch even when the energy is generated on a Poincare disk.  Build the
    # matching coordinates here so the surface is not shifted or stretched.
    radius = 1.0 / zoom
    disk_axis = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(disk_axis, disk_axis)

    conti_grid = generate_energy_grid(
        E_func=ContiEnergy,
        beta=conti_beta,
        K=conti_K,
        energy_lim=None,
        resolution=resolution,
        zoom=zoom,
        transformation=transformation,
        poincareDisk=True,
        loops=loops,
    )
    quadratic_grid = generate_energy_grid(
        E_func=quadratic_energy_function,
        beta=quadratic_kappa,
        K=quadratic_xi,
        noise=quadratic_eta,
        energy_lim=None,
        resolution=resolution,
        zoom=zoom,
        transformation=transformation,
        poincareDisk=True,
        loops=loops,
    )
    grids = [
        np.array(conti_grid, dtype=float, copy=True),
        np.array(quadratic_grid, dtype=float, copy=True),
    ]

    geometric_mask = np.zeros_like(X, dtype=bool)
    if data_radius is not None:
        geometric_mask |= np.hypot(X, Y) > float(data_radius)
    if add_front_hole:
        geometric_mask |= np.hypot(X, Y - 1.4) < 1.0

    finite_values = [
        grid[np.isfinite(grid) & ~geometric_mask]
        for grid in grids
    ]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        raise ValueError("Neither energy grid contains any finite values.")
    all_values = np.concatenate(finite_values)

    if energy_lim is None:
        z_min, z_max = np.nanmin(all_values), np.nanmax(all_values)
    else:
        if len(energy_lim) != 2:
            raise ValueError("energy_lim must be None or a (min, max) pair.")
        z_min = (
            np.nanmin(all_values)
            if energy_lim[0] is None
            else float(energy_lim[0])
        )
        z_max = (
            np.nanmax(all_values)
            if energy_lim[1] is None
            else float(energy_lim[1])
        )
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        raise ValueError(
            "energy_lim must define a finite increasing range, "
            f"got {(z_min, z_max)}."
        )

    # Clip before plotting so the shared z/color limits are also the visible
    # limits, while preserving NaNs from the edge of the Poincare disk.
    for grid in grids:
        finite = np.isfinite(grid)
        grid[finite] = np.clip(grid[finite], z_min, z_max)
        grid[geometric_mask] = np.nan

    if remove_max_color:
        base_cmap = cm.get_cmap(cmap, 256)
        cmap_values = base_cmap(np.linspace(0, 1, 256))
        cmap_values[-2:, -1] = np.linspace(1, 0, 2) ** 0.5
        surface_cmap = colors.ListedColormap(cmap_values)
    else:
        surface_cmap = cmap

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    surface_options = {
        "cmap": surface_cmap,
        "linewidth": 0,
        "antialiased": False,
        "rstride": 1,
        "cstride": 1,
        "vmin": z_min,
        "vmax": z_max,
        "alpha": surface_alpha,
    }
    if surface_kwargs:
        surface_options.update(surface_kwargs)

    surfaces = []
    for ax, grid, title, z_axis_label in zip(
        axes, grids, titles, z_axis_labels
    ):
        surface = ax.plot_surface(X, Y, grid, **surface_options)
        surfaces.append(surface)

        ax.set_title(title, y=title_y, pad=title_pad)
        ax.set_xlabel(r"$x_p$")
        ax.set_ylabel(r"$y_p$")
        ax.set_zlabel(z_axis_label)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(z_min, z_max)
        ax.zaxis.set_major_locator(ticker.LinearLocator(numticks=z_tick_count))
        ax.set_box_aspect((1.0, 1.0, z_scale))
        ax.grid(show_grid)
        try:
            ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:  # Matplotlib versions before the roll argument.
            ax.view_init(elev=elev, azim=azim)

    if show_colorbars:
        cbar = fig.colorbar(surfaces[0], ax=axes, shrink=0.5, pad=-0.0)
        cbar.set_label("Energy density")
        cbar.locator = ticker.LinearLocator(numticks=z_tick_count)
        cbar.update_ticks()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=save_pad_inches,
            facecolor="white",
        )
        if autocrop_png and output_path.suffix.lower() == ".png":
            crop_white_border(
                output_path,
                white_threshold=white_threshold,
            )
        print(f"Saved plot to {output_path.resolve()}")
    if show:
        plt.show()
    return fig, axes

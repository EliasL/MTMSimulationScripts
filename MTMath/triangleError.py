import numpy as np
from typing import Dict, Optional, Tuple
from MTMath.plotEnergy import (
    generate_poincare_disk,
    drawTriangularElasticDomain,
    drawShearPath,
    drawFundamentalDomain,
)
import numpy.typing as npt

# -----------------------------------------------------------------------------
# Waldron (1998), "The Error in Linear Interpolation" — triangle (2D) case.
#
# References (equation/section numbers are from Waldron 1998):
# - (3.1)  : definition of |D^2 f|(x) = sup_{||ξ||=1} |D^2_ξ f(x)|
# - Thm 3.1 (3.4): |f(x) - L_Θ f(x)| ≤ 1/2 * (R^2 - ||x - c||^2) * || |D^2 f| ||_{L∞(T)}
# - (3.6)  : ‖f - L_Θ f‖_{L∞(T)} ≤ 1/2 * (R^2 - d^2) * || |D^2 f| ||_{L∞(T)},
#            where d = dist(c, T)
# - 2D special cases (discussion around Fig. 3.1):
#            If c ∈ T (triangle acute): d = 0  → α(T) = 1/2 * R^2
#            If c ∉ T (triangle obtuse): α(T) = 1/8 * h^2  (Pythagoras argument)
# - Also see (3.15): universal bound ‖f - L_Θ f‖_{L∞(T)} ≤ (1/6) h^2 * || |D^2 f| ||_{L∞(T)}
#   (sharp iff equilateral), which our piecewise formula refines case-by-case.
#
# Geometry from Gram matrix:
# Let e1, e2 be triangle edges from a reference vertex: C = [[e1·e1, e1·e2],[e2·e1, e2·e2]].
# Side lengths:  c = ||e1||, b = ||e2||, a = ||e2 - e1||.  Area A = 0.5 * sqrt(det(C)).
# Circumradius:  R = (a*b*c) / (4A).  Diameter h = max(a,b,c).
# Triangle is acute  ⇔  for the longest side s_max^2, we have  2*s_max^2 < a^2+b^2+c^2.
# -----------------------------------------------------------------------------


def _side_lengths_from_gram(C: np.ndarray) -> Tuple[float, float, float]:
    """
    Given 2x2 Gram matrix C of the two edge vectors e1, e2 (from a common vertex),
    return the three side lengths (a, b, c) where:
      c = ||e1||, b = ||e2||, a = ||e2 - e1||.

    Checks:
    - C must be 2x2 symmetric positive definite for a nondegenerate triangle.
    """
    C = np.asarray(C, dtype=float)
    if np.isnan(C).any():
        return (np.nan, np.nan, np.nan)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if not np.allclose(C, C.T, atol=1e-12):
        raise ValueError("C must be symmetric.")
    detC = np.linalg.det(C)
    if detC <= 0:
        raise ValueError(
            "C must be positive definite (det(C) > 0) for a valid triangle."
        )

    # c = ||e1||, b = ||e2||, a = ||e2 - e1||
    c = np.sqrt(C[0, 0])
    b = np.sqrt(C[1, 1])
    a2 = C[0, 0] + C[1, 1] - 2.0 * C[0, 1]
    if a2 <= 0:
        # Degenerate: e1 and e2 are collinear with zero separation for the third side
        raise ValueError(
            "Triangle is degenerate (third side length computed non-positive)."
        )
    a = np.sqrt(a2)
    return a, b, c


def _area_from_gram(C: np.ndarray) -> float:
    """
    Triangle area A from Gram matrix C:  A = 0.5 * sqrt(det(C)).
    (Because det(C) = ||e1||^2 ||e2||^2 - (e1·e2)^2 = (2A)^2 in 2D.)
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any():
        return np.nan
    detC = np.linalg.det(C)
    return 0.5 * np.sqrt(detC)


def _circumradius_from_gram(C: np.ndarray) -> float:
    """
    Circumradius R = (a*b*c) / (4A),
    with (a,b,c) as in _side_lengths_from_gram and A = area.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any():
        return np.nan
    a, b, c = _side_lengths_from_gram(C)
    A = _area_from_gram(C)
    return (a * b * c) / (4.0 * A)


def _diameter_from_gram(C: np.ndarray) -> float:
    """
    Diameter h = max pairwise vertex distance = max(a,b,c).
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any():
        return np.nan
    a, b, c = _side_lengths_from_gram(C)
    return max(a, b, c)


def _is_acute_from_gram(C: np.ndarray) -> bool:
    """
    Acute test via sides (Law of Cosines):
    triangle is acute  ⇔  for the longest side s_max^2, we have 2*s_max^2 < a^2 + b^2 + c^2.
    Equivalently, no angle ≥ 90°.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any():
        return False  # will be ignored upstream because we short-circuit on NaN
    a, b, c = _side_lengths_from_gram(C)
    if np.isnan(a) or np.isnan(b) or np.isnan(c):
        return False
    s2 = np.array([a * a, b * b, c * c])
    s2_max = float(np.max(s2))
    return (2.0 * s2_max) < float(np.sum(s2))


def waldron_Linf_shape_constant(C: np.ndarray) -> Dict[str, float]:
    """
    Compute the *sharp* shape-only coefficient α(T) for the L∞ error bound
      ‖f - L_Θ f‖_{L∞(T)} ≤ α(T) * || |D^2 f| ||_{L∞(T)}.
    Per Waldron Thm 3.1: α(T) = 1/2 * (R^2 - d^2).  (Eq. (3.6))
    In 2D:
      - if circumcenter c ∈ T (triangle acute):   α(T) = 1/2 * R^2      (take d=0)
      - if c ∉ T (triangle obtuse):               α(T) = 1/8 * h^2      (Fig. 3.1 argument)

    Returns a dict with:
      - R : circumradius
      - h : diameter
      - acute : 1.0 if acute else 0.0
      - alpha : sharp α(T)

    Notes:
      - The universal bound (3.15) α_universal = (1/6)*h^2 (sharp iff equilateral)
        is weaker than the piecewise sharp α(T) above; we report it as well for reference.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any():
        return dict(
            R=np.nan, h=np.nan, acute=np.nan, alpha=np.nan, alpha_universal=np.nan
        )
    R = _circumradius_from_gram(C)
    h = _diameter_from_gram(C)
    acute = _is_acute_from_gram(C)

    if acute:
        alpha = 0.5 * (R**2)  # Eq. (3.6) with d=0; sharp when c ∈ T.
    else:
        alpha = 0.125 * (
            h**2
        )  # 1/8 h^2 for obtuse triangle (2D specialization of Thm 3.1).

    alpha_universal = (1.0 / 6.0) * (
        h**2
    )  # Eq. (3.15): universal (suboptimal) constant.

    return dict(
        R=R, h=h, acute=float(acute), alpha=alpha, alpha_universal=alpha_universal
    )


def waldron_Linf_bound_from_C_and_Hnorm(C: np.ndarray, H_opnorm_Linf: float) -> float:
    """
    Final L∞ error bound from C and a supplied seminorm of the Hessian:
      ‖f - L_Θ f‖_{L∞(T)} ≤ α(T) * H_opnorm_Linf,
    where H_opnorm_Linf := || |D^2 f| ||_{L∞(T)}  (Eq. (3.1) and (3.6)).

    If you don't have H_opnorm_Linf, use `waldron_operator_seminorm_from_H(H)` below
    for a *pointwise* spectral radius of a symmetric Hessian H (max |eigenvalue|),
    then take a supremum over T by sampling.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (2, 2):
        raise ValueError("C must be a 2x2 Gram matrix for a triangle.")
    if np.isnan(C).any() or np.isnan(H_opnorm_Linf):
        return np.nan
    alpha = waldron_Linf_shape_constant(C)["alpha"]
    return alpha * float(H_opnorm_Linf)


def waldron_operator_seminorm_from_components(
    a: float, b: float, c: float, *, use_frobenius: bool = False
) -> float:
    """
    Parameterized 2D Hessian H = [[a, b], [b, c]].
    If use_frobenius=True, return the Frobenius norm (conservative upper bound):
        ||H||_F = sqrt(a^2 + 2 b^2 + c^2).
    Otherwise return the spectral radius (exact operator seminorm):
        max(|λ_min(H)|, |λ_max(H)|).
    This matches Waldron's |D^2 f| in (3.1).
    """
    if use_frobenius:
        return float(np.sqrt(a * a + 2.0 * b * b + c * c))
    # spectral radius of symmetric 2x2
    tr = a + c
    disc = np.sqrt((a - c) ** 2 + 4.0 * b * b)
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    return float(max(abs(lam1), abs(lam2)))


def _vectorized_geom_from_gram(
    Cgrid: npt.NDArray[np.floating],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized geometry from a grid of 2x2 Gram matrices.

    Parameters
    ----------
    Cgrid : array, shape (Ny, Nx, 2, 2)
        Grid of Gram matrices.

    Returns
    -------
    a, b, c, A, detC : arrays of shape (Ny, Nx)
        Side lengths (a=||e2-e1||, b=||e2||, c=||e1||), area A, and determinant detC.

    Notes
    -----
    - NaNs in C propagate to outputs (NaN-safe via numpy broadcasting).
    - Symmetry of C is assumed; off-diagonal entries C12 and C21 are both used in det.
    """
    C = np.asarray(Cgrid, dtype=float)
    if C.ndim != 4 or C.shape[-2:] != (2, 2):
        raise ValueError("Cgrid must have shape (Ny, Nx, 2, 2)")
    C11 = C[..., 0, 0]
    C22 = C[..., 1, 1]
    C12 = C[..., 0, 1]
    C21 = C[..., 1, 0]
    # Determinant of 2x2 Gram
    detC = C11 * C22 - C12 * C21
    # Side lengths
    c = np.sqrt(C11)
    b = np.sqrt(C22)
    a2 = C11 + C22 - 2.0 * C12
    a = np.sqrt(a2)
    # Area: A = 0.5 * sqrt(detC)
    A = 0.5 * np.sqrt(detC)
    return a, b, c, A, detC


def waldron_Linf_shape_constant_grid(
    Cgrid: npt.NDArray[np.floating],
) -> Dict[str, np.ndarray]:
    """
    Vectorized *shape-only* coefficient α(T) over a grid of Gram matrices.

    Returns dict of arrays (Ny, Nx):
      - 'R' : circumradius
      - 'h' : diameter
      - 'acute' : boolean mask (True if acute)
      - 'alpha' : sharp α(T)
      - 'alpha_universal' : (1/6) h^2 (Waldron (3.15))

    References: Waldron Thm 3.1 / eqs. (3.4), (3.6). 2D cases: c∈T → α=1/2 R^2; c∉T → α=1/8 h^2.
    """
    a, b, c, A, detC = _vectorized_geom_from_gram(Cgrid)

    # Circumradius R = (a*b*c)/(4A), handle division by zero via numpy rules (→ NaN/inf)
    R = (a * b * c) / (4.0 * A)

    # Diameter h = max(a,b,c) per-cell
    h = np.maximum(a, np.maximum(b, c))

    # Acute test: 2*s_max^2 < a^2 + b^2 + c^2
    s2 = np.stack([a * a, b * b, c * c], axis=-1)  # (Ny, Nx, 3)
    s2_sum = np.sum(s2, axis=-1)
    s2_max = np.max(s2, axis=-1)
    acute = (2.0 * s2_max) < s2_sum

    # Piecewise sharp α(T)
    alpha = np.where(acute, 0.5 * (R**2), 0.125 * (h**2))
    alpha_universal = (1.0 / 6.0) * (h**2)

    # Propagate NaNs from invalid Gram matrices: if any of C entries is NaN, result stays NaN.
    return dict(
        R=R,
        h=h,
        acute=acute,
        alpha=alpha,
        alpha_universal=alpha_universal,
        a=a,
        b=b,
        c=c,
        A=A,
    )


def waldron_Linf_bound_from_Cgrid_and_Hnorm(
    Cgrid: npt.NDArray[np.floating], H_opnorm_Linf: float
) -> np.ndarray:
    """
    Vectorized L∞ error bound over a grid of Gram matrices and a scalar Hessian seminorm.

    ‖f - L_Θ f‖_{L∞(T)} ≤ α(T) * H_opnorm_Linf, elementwise.
    """
    if np.isnan(H_opnorm_Linf):
        # Produce a matching NaN array
        return np.full(Cgrid.shape[:2], np.nan, dtype=float)
    shape_info = waldron_Linf_shape_constant_grid(Cgrid)
    return shape_info["alpha"] * float(H_opnorm_Linf)


def Shewchuk_error_Size_and_shape_f_g(
    Cgrid: npt.NDArray[np.floating], H_opnorm_Linf: float
) -> np.ndarray:
    shape_info = waldron_Linf_shape_constant_grid(Cgrid)
    R = shape_info["R"]
    h = shape_info["h"]
    acute = shape_info["acute"]
    ct = 1.0 if np.isnan(H_opnorm_Linf) else float(H_opnorm_Linf)
    rmc = np.where(acute, R, 0.5 * h)
    return 1.0 / (ct * (rmc**2))


def Shewchuk_error_Size_and_shape_grad_f_grad_g(
    Cgrid: npt.NDArray[np.floating], H_opnorm_Linf: float
) -> np.ndarray:
    # Vectorized implementation of Shewchuk Table 3 (gradient error, stronger bound)
    a, b, c, A, detC = _vectorized_geom_from_gram(Cgrid)
    # Stack edge lengths and sort along last axis (per cell)
    edges = np.stack([a, b, c], axis=-1)  # shape (..., 3)
    edges_sorted = np.sort(edges, axis=-1)
    lmin = edges_sorted[..., 0]
    lmed = edges_sorted[..., 1]
    lmax = edges_sorted[..., 2]
    # Inradius magnitude
    rin = 2.0 * A / (a + b + c)
    abs_rin = np.abs(rin)
    ct = 1.0 if np.isnan(H_opnorm_Linf) else float(H_opnorm_Linf)
    # Q = A / (ct * (lmax * lmed * (lmin + 4 * abs_rin)))
    denom = ct * (lmax * lmed * (lmin + 4.0 * abs_rin))
    Q = A / denom
    # Numerical safety: set Q = nan where any of A, lmin, lmed, lmax <= 0, or any input nan
    mask_invalid = (
        (A <= 0)
        | (lmin <= 0)
        | (lmed <= 0)
        | (lmax <= 0)
        | np.isnan(A)
        | np.isnan(lmin)
        | np.isnan(lmed)
        | np.isnan(lmax)
        | np.isnan(denom)
    )
    Q = np.where(mask_invalid, np.nan, Q)
    return Q


# ---------------------------------------------------------------------
# Helper functions for field-dependent Hessian seminorm
def Shewchuk_error_Size_and_shape_f_g_field(
    Cgrid: npt.NDArray[np.floating], Hfield: np.ndarray
) -> np.ndarray:
    """
    Like Shewchuk_error_Size_and_shape_f_g, but with spatially varying Hessian seminorm field Hfield.
    """
    shape_info = waldron_Linf_shape_constant_grid(Cgrid)
    R = shape_info["R"]
    h = shape_info["h"]
    acute = shape_info["acute"]
    # ct is now the field Hfield
    rmc = np.where(acute, R, 0.5 * h)
    # Avoid division by zero or invalid Hfield
    Q = 1.0 / (Hfield * (rmc**2))
    # Propagate invalids in Hfield
    Q = np.where(~np.isfinite(Hfield), np.nan, Q)
    return Q


def Shewchuk_error_Size_and_shape_grad_f_grad_g_field(
    Cgrid: npt.NDArray[np.floating], Hfield: np.ndarray
) -> np.ndarray:
    """
    Like Shewchuk_error_Size_and_shape_grad_f_grad_g, but with spatially varying Hessian seminorm field Hfield.
    """
    a, b, c, A, detC = _vectorized_geom_from_gram(Cgrid)
    edges = np.stack([a, b, c], axis=-1)
    edges_sorted = np.sort(edges, axis=-1)
    lmin = edges_sorted[..., 0]
    lmed = edges_sorted[..., 1]
    lmax = edges_sorted[..., 2]
    rin = 2.0 * A / (a + b + c)
    abs_rin = np.abs(rin)
    denom = Hfield * (lmax * lmed * (lmin + 4.0 * abs_rin))
    Q = A / denom
    # Guard against invalids and zero denominators, as well as invalid Hfield
    mask_invalid = (
        (A <= 0)
        | (lmin <= 0)
        | (lmed <= 0)
        | (lmax <= 0)
        | np.isnan(A)
        | np.isnan(lmin)
        | np.isnan(lmed)
        | np.isnan(lmax)
        | np.isnan(denom)
        | ~np.isfinite(Hfield)
        | (denom == 0)
    )
    Q = np.where(mask_invalid, np.nan, Q)
    return Q


def Shewchuk_error_Conditioning(
    Cgrid: npt.NDArray[np.floating], H_opnorm_Linf: float
) -> np.ndarray:
    """
    Vectorized implementation of Shewchuk's scale-invariant conditioning quality measure (Table 3, second-to-last row).
    Q = A / ( 3*l_rms^2 + sqrt( (3*l_rms^2)^2 - 48*A^2 ) )
    where l_rms^2 = (a^2 + b^2 + c^2)/3, a,b,c are side lengths, A is area.
    """
    # Ignore H_opnorm_Linf (not used in conditioning)
    a, b, c, A, detC = _vectorized_geom_from_gram(Cgrid)
    # S = a^2 + b^2 + c^2 = 3 * l_rms^2
    S = a * a + b * b + c * c
    disc = S * S - 48.0 * (A * A)
    # Clamp negative discriminant to zero for numerical safety
    disc = np.where(disc < 0, 0.0, disc)
    root = np.sqrt(disc)
    den = S + root
    Q = A / den
    # Set Q = nan where detC <= 0, A <= 0, den <= 0, or any input is nan
    mask_invalid = (
        (detC <= 0)
        | (A <= 0)
        | (den <= 0)
        | np.isnan(a)
        | np.isnan(b)
        | np.isnan(c)
        | np.isnan(A)
        | np.isnan(detC)
        | np.isnan(den)
    )
    Q = np.where(mask_invalid, np.nan, Q)
    return Q


def waldron_operator_seminorm_from_H(H: Optional[np.ndarray]) -> float:
    """
    Compute |D^2 f|(x) operator seminorm.

    If H is None, return 1.0 to visualize *pure shape* (unit-curvature convention),
    which corresponds to plotting the geometry factor α(T) alone (Waldron (3.6)).
    Otherwise, for a *symmetric* Hessian H(x),
      |D^2 f|(x) = sup_{||ξ||=1} |ξᵀ H ξ| = max(|λ_min(H)|, |λ_max(H)|).
    This is the pointwise quantity from Eq. (3.1). For the L∞ bound you need its
    supremum over the triangle; obtain that by sampling H(x) if H varies spatially.
    """
    if H is None:
        return 1.0
    H = np.asarray(H, dtype=float)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square (symmetric) Hessian matrix.")
    # We assume symmetry; if needed, symmetrize numerically:
    Hs = 0.5 * (H + H.T)
    eigvals = np.linalg.eigvalsh(Hs)
    return float(np.max(np.abs(eigvals)))


# ---------------------------------------------------------------------
# Option A: Hessian seminorm field generator
def Hnorm_field_optionA(X: np.ndarray, Y: np.ndarray, k: float = 8.0) -> np.ndarray:
    """Diagonal Hessian with spatially varying eigenvalues; returns operator seminorm field.
    lam1 = 1 + 0.6*sin(k*X); lam2 = 0.3 + 0.7*cos(k*Y); Hnorm = max(|lam1|, |lam2|).
    """
    lam1 = 1.0 + 0.6 * np.sin(k * X)
    lam2 = 0.3 + 0.7 * np.cos(k * Y)
    return np.maximum(np.abs(lam1), np.abs(lam2))


def test():
    N = 2000
    transformation = "triangular"

    C = generate_poincare_disk(resolution=N, transformation=transformation)

    Hnorm = waldron_operator_seminorm_from_H(None)

    # error = waldron_Linf_bound_from_Cgrid_and_Hnorm(C, Hnorm)
    error = Shewchuk_error_Size_and_shape_f_g(C, Hnorm)
    # error = Shewchuk_error_Size_and_shape_grad_f_grad_g(C, Hnorm)
    # error = Shewchuk_error_Conditioning(C, Hnorm)

    x_vals = np.linspace(0, 1 * N, C.shape[1])
    y_vals = np.linspace(0, 1 * N, C.shape[0])

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    cf = ax.contourf(
        x_vals,
        y_vals,
        error,
        cmap="coolwarm",
    )

    drawTriangularElasticDomain(ax, grid_size=N, transformation=transformation)
    drawShearPath(ax, grid_size=N, transformation=transformation)
    drawFundamentalDomain(ax, grid_size=N, transformation=transformation)
    # Find and plot the global minimum
    imin, jmin = np.unravel_index(np.nanargmax(error), error.shape)
    x_min = x_vals[jmin]
    y_min = y_vals[imin]
    C_min = C[imin, jmin]
    print("argmin indices (i,j):", imin, jmin)
    print("x_min, y_min:", x_min, y_min)
    print("error_min:", error[imin, jmin])
    print("C at global minimum (Gram matrix):\n", C_min)
    ax.plot(
        x_min,
        y_min,
        "o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="k",
    )

    cbar = fig.colorbar(cf, ax=ax, label=r"$r_{mc}^{-2}$")
    ax.set_title("Waldron L error factor (unit curvature)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")
    # ax.legend(loc="upper right")
    plt.show()


def test2():
    # Choose which measure to visualize: 'f_g', 'grad', or 'cond'
    measure = "f_g"
    N = 2000
    transformation = "triangular"

    C = generate_poincare_disk(resolution=N, transformation=transformation)

    # Constant curvature bound (unit curvature convention)
    Hnorm_scalar = waldron_operator_seminorm_from_H(None)

    # Coordinates for field construction
    x_vals = np.linspace(0, 1 * N, C.shape[1])
    y_vals = np.linspace(0, 1 * N, C.shape[0])
    X, Y = np.meshgrid(x_vals / np.max(x_vals), y_vals / np.max(y_vals))

    # Option A: spatially varying Hessian seminorm field
    Hgrid = Hnorm_field_optionA(X, Y, k=8.0)

    # Compute error/quality for both cases
    if measure == "f_g":
        error_const = Shewchuk_error_Size_and_shape_f_g(C, Hnorm_scalar)
        error_field = Shewchuk_error_Size_and_shape_f_g_field(C, Hgrid)
    elif measure == "grad":
        error_const = Shewchuk_error_Size_and_shape_grad_f_grad_g(C, Hnorm_scalar)
        error_field = Shewchuk_error_Size_and_shape_grad_f_grad_g_field(C, Hgrid)
    elif measure == "cond":
        # Conditioning is geometry-only; show same plot twice for completeness
        error_const = Shewchuk_error_Conditioning(C, Hnorm_scalar)
        error_field = error_const.copy()
    else:
        raise ValueError("Unknown measure: choose 'f_g', 'grad', or 'cond'")

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Common color scale
    finite_vals = np.hstack(
        [
            np.ravel(error_const[np.isfinite(error_const)]),
            np.ravel(error_field[np.isfinite(error_field)]),
        ]
    )
    vmin = np.min(finite_vals) if finite_vals.size else None
    vmax = np.max(finite_vals) if finite_vals.size else None

    # Left: constant curvature bound
    cf0 = axs[0].contourf(
        x_vals, y_vals, error_const, levels=64, vmin=vmin, vmax=vmax, cmap="coolwarm"
    )
    drawTriangularElasticDomain(axs[0], grid_size=N, transformation=transformation)
    drawShearPath(axs[0], grid_size=N, transformation=transformation)
    i0, j0 = np.unravel_index(np.nanargmax(error_const), error_const.shape)
    axs[0].plot(
        x_vals[j0],
        y_vals[i0],
        "o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="k",
    )
    axs[0].set_title(f"{measure} — constant |D^2f| (unit)")
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_xlabel(r"$x_p$")
    axs[0].set_ylabel(r"$y_p$")

    # Right: Option A field
    cf1 = axs[1].contourf(
        x_vals, y_vals, error_field, levels=64, vmin=vmin, vmax=vmax, cmap="coolwarm"
    )
    drawTriangularElasticDomain(axs[1], grid_size=N, transformation=transformation)
    drawShearPath(axs[1], grid_size=N, transformation=transformation)
    i1, j1 = np.unravel_index(np.nanargmax(error_field), error_field.shape)
    axs[1].plot(
        x_vals[j1],
        y_vals[i1],
        "o",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="k",
    )
    axs[1].set_title(f"{measure} — Option A |D^2f|(x,y)")
    axs[1].set_aspect("equal", adjustable="box")
    axs[1].set_xlabel(r"$x_p$")

    # Single shared colorbar
    cbar = fig.colorbar(
        cf1, ax=axs, location="right", shrink=0.9, label="quality measure Q"
    )

    plt.show()

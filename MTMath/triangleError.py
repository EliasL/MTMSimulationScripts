import numpy as np
from typing import Dict, Optional, Tuple
from MTMath.poincareEnergy import (
    generate_poincare_disk,
    drawTriangularElasticDomain,
    drawShearPath,
    drawFundamentalDomain,
    drawUnitCircle,
    poincareDisk2C,
)
import numpy.typing as npt
from scipy import optimize as opt
from pathlib import Path

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


# ---------------------------------------------------------------------
# Singular-value–based metrics (via Gram matrix C = J^T J)
# For T3, J = [e1 e2], so C = [[e1·e1, e1·e2],[e2·e1, e2·e2]] = J^T J.
# Singular values of J are sqrt of eigenvalues of C.
# The element stiffness can be formed directly from C:
#   K_elem = A * Ĝ^T * C^{-1} * Ĝ,  with A = 0.5*sqrt(det C), |Ω̂|=1/2.
#
# We report:
#   - κ(J) = σ1/σ2
#   - κ*(K) = ratio of the two positive eigenvalues of K_elem (ignoring the constant mode)
#   - ρ := κ*(K) / κ(J)^2      (should be ~constant across shape, e.g. ≈ 1/3 for equilateral)
#
# Vectorized grid versions are provided as well.

# Reference gradient matrix on the unit right reference triangle
# N̂1 = 1 - x̂ - ŷ, N̂2 = x̂, N̂3 = ŷ  ->  ∇x̂N̂1=[-1,-1], ∇x̂N̂2=[1,0], ∇x̂N̂3=[0,1]
G_HAT = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=float)


def _inv2x2_grid(Cgrid: npt.NDArray[np.floating]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert a grid of 2x2 matrices with broadcasting.
    Returns (Cinv, detC). NaNs propagate; singular detC -> NaNs in Cinv.
    """
    C = np.asarray(Cgrid, dtype=float)
    if C.ndim != 4 or C.shape[-2:] != (2, 2):
        raise ValueError("Cgrid must have shape (Ny, Nx, 2, 2)")
    a = C[..., 0, 0]
    b = C[..., 0, 1]
    c = C[..., 1, 0]
    d = C[..., 1, 1]
    det = a * d - b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.empty_like(C)
        inv[..., 0, 0] = d / det
        inv[..., 0, 1] = -b / det
        inv[..., 1, 0] = -c / det
        inv[..., 1, 1] = a / det
        inv = np.where(np.isfinite(det)[..., None, None], inv, np.nan)
    return inv, det


def svd_metrics_from_gram_grid(
    Cgrid: npt.NDArray[np.floating],
) -> Dict[str, np.ndarray]:
    """
    Grid of singular-value metrics: σ1, σ2, κ(J).
    Robust to NaNs/degenerate cells: invalid cells → NaN outputs without crashing.
    """
    C = np.asarray(Cgrid, dtype=float)
    if C.ndim != 4 or C.shape[-2:] != (2, 2):
        raise ValueError("Cgrid must have shape (Ny, Nx, 2, 2)")
    C11 = C[..., 0, 0]
    C22 = C[..., 1, 1]
    C12 = C[..., 0, 1]
    C21 = C[..., 1, 0]
    detC = C11 * C22 - C12 * C21
    mask_bad = (
        np.isnan(C11) | np.isnan(C22) | np.isnan(C12) | np.isnan(C21) | (detC <= 0)
    )

    # Build a clean SPD surrogate for bad cells to keep eigvalsh from failing
    Cclean = C.copy()
    I2 = np.eye(2, dtype=float)
    if np.any(mask_bad):
        Cclean[mask_bad] = I2

    evals = np.linalg.eigvalsh(Cclean)  # shape (..., 2)
    sigma2 = np.sqrt(evals[..., 0])
    sigma1 = np.sqrt(evals[..., 1])
    kappaJ = sigma1 / sigma2

    # Invalidate bad cells and those with nonpositive smallest eigenvalue
    bad2 = mask_bad | ~(evals[..., 0] > 0)
    sigma1 = np.where(bad2, np.nan, sigma1)
    sigma2 = np.where(bad2, np.nan, sigma2)
    kappaJ = np.where(bad2, np.nan, kappaJ)
    return dict(sigma1=sigma1, sigma2=sigma2, kappaJ=kappaJ)


def kappa_star_K_from_gram_grid(
    Cgrid: npt.NDArray[np.floating], *, tol: float = 1e-12
) -> np.ndarray:
    """
    Grid of κ*(K) using K_elem = A * G_HAT^T * C^{-1} * G_HAT.
    Robust to NaNs/degenerate cells: invalid cells → NaN outputs without crashing.
    """
    C = np.asarray(Cgrid, dtype=float)
    if C.ndim != 4 or C.shape[-2:] != (2, 2):
        raise ValueError("Cgrid must have shape (Ny, Nx, 2, 2)")
    C11 = C[..., 0, 0]
    C22 = C[..., 1, 1]
    C12 = C[..., 0, 1]
    C21 = C[..., 1, 0]
    detC = C11 * C22 - C12 * C21
    mask_badC = (
        np.isnan(C11) | np.isnan(C22) | np.isnan(C12) | np.isnan(C21) | (detC <= 0)
    )

    # Inverse and area
    Cinv, detC_inv = _inv2x2_grid(C)
    A = 0.5 * np.sqrt(detC)

    # S = G^T Cinv G  (broadcasted)
    tmp = np.einsum("ia,...ab->...ib", G_HAT.T, Cinv)
    S = np.einsum("...ib, bj->...ij", tmp, G_HAT)
    K = A[..., None, None] * S

    # For invalid cells, drop in a harmless SPD to keep eigvalsh stable
    if np.any(mask_badC):
        K = K.copy()
        K[mask_badC] = np.eye(3, dtype=float)

    evals = np.linalg.eigvalsh(K)  # (..., 3), sorted ascending
    pos_min = evals[..., 1]
    pos_max = evals[..., 2]

    # Combine bad conditions
    bad = (
        mask_badC
        | (A <= 0)
        | (pos_max <= tol)
        | (pos_min <= tol)
        | ~np.isfinite(pos_max)
        | ~np.isfinite(pos_min)
    )
    kappaK = pos_max / pos_min
    kappaK = np.where(bad, np.nan, kappaK)
    return kappaK


def ratio_kappaK_over_kappaJ2_grid(Cgrid: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Grid of ρ = κ*(K) / κ(J)^2.
    """
    sv = svd_metrics_from_gram_grid(Cgrid)
    kJ2 = sv["kappaJ"] ** 2
    kK = kappa_star_K_from_gram_grid(Cgrid)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = kK / kJ2
    rho = np.where((kJ2 <= 0) | ~np.isfinite(kJ2) | ~np.isfinite(kK), np.nan, rho)
    return rho


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


# ---------------------------------------------------------------------
# Axis utilities


def _set_axis_ticks_unit(ax, N: int, hide_y: bool = False) -> None:
    """Set x and y axis ticks to [-1, -0.5, 0, 0.5, 1] mapped onto [0, N].
    If hide_y is True, keep the y tick *positions* aligned but hide their labels
    to avoid repetition on adjacent subplots.
    """
    positions = np.linspace(0, N, 5)
    labels = ["-1", "-0.5", "0", "0.5", "1"]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(positions)
    if hide_y:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(labels)


def _save_plot(fig, filename_base: str) -> None:
    """Save figure as PDF under Plots/elementError/ with a space-free name.
    Prints: "Saved to <path>".
    """
    outdir = Path("Plots") / "elementError"
    outdir.mkdir(parents=True, exist_ok=True)
    safe = filename_base.replace(" ", "_")
    outpath = outdir / f"{safe}.pdf"
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    print("Saved to", outpath)


# ---------------------------------------------------------------------
# Plot utilities to reduce duplication


def _disk_to_plot(x: float, y: float, N: int) -> Tuple[float, float]:
    """Map Poincaré disk coords in [-1,1]x[-1,1] to plot coords [0,N]x[0,N]."""
    return 0.5 * (x + 1.0) * N, 0.5 * (y + 1.0) * N


def _decorate_disk_axes(
    ax,
    *,
    N: int,
    transformation: str,
    hide_y: bool,
    title: Optional[str] = None,
    xlabel: str = r"$x_p$",
    ylabel: Optional[str] = r"$y_p$",
) -> None:
    """Common overlays + ticks + labels used on all disk plots."""
    drawTriangularElasticDomain(ax, grid_size=N, transformation=transformation)
    drawShearPath(ax, grid_size=N, transformation=transformation)
    drawUnitCircle(ax, grid_size=N)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    _set_axis_ticks_unit(ax, N, hide_y=hide_y)


def _contourf_with_min(
    ax,
    fig,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    Z: np.ndarray,
    *,
    levels: int = 50,
    cmap: str = "coolwarm",
    label: Optional[str] = None,
) -> Tuple[any, any, float]:
    """Draw contourf with vmin at data minimum and ensure colorbar shows it.
    Returns (cf, cbar, vmin).
    """
    vmin = float(np.nanmin(Z))
    cf = ax.contourf(x_vals, y_vals, Z, levels=levels, cmap=cmap, vmin=vmin)
    cf.set_edgecolor("face")
    cbar = None
    if label is not None:
        cbar = fig.colorbar(cf, ax=ax, location="right", shrink=0.9, label=label)
        try:
            ticks = cbar.get_ticks()
            if ticks is None:
                ticks = np.array([])
            if (ticks.size == 0) or (abs(float(ticks[0]) - vmin) > 1e-9):
                cbar.set_ticks(np.concatenate(([vmin], ticks[1:])))
                cbar.update_ticks()
        except Exception:
            pass
    return cf, cbar, vmin


def _plot_point(
    ax,
    x: float,
    y: float,
    marker: str = "o",
    ms: float = 5.0,
    label: Optional[str] = None,
):
    h = ax.plot(
        x,
        y,
        marker,
        markersize=ms,
        label=label,
        markerfacecolor="none",
        markeredgecolor="k",
    )
    return h[0]


def find_minimum_ratio(
    method: str = "SLSQP",
    tol: float = 1e-12,
    maxiter: int = 200,
    transformation="none",
) -> Dict[str, float]:
    """
    Constrained minimization of ρ = κ*(K)/κ(J)^2 over the Poincaré disk.

    The optimizer is constrained to the *unit circle*: x^2 + y^2 ≤ 1 and
    coordinate bounds x ∈ [-1, 1], y ∈ [-1, 1].

    Returns a dict with the optimal (y, x) and the minimized value.
    Notes:
      - The internal variable order is (y, x) for consistency with the codebase.
      - We call poincareDisk2C(x, y), so we swap when evaluating.
    """

    def objective(ij: np.ndarray) -> float:
        x, y = ij  # fractional (row, col)
        # Hard guard: if outside unit disk, give a large penalty.
        if x * x + y * y > 1.0:
            return 1e6
        C = poincareDisk2C(x, y, transformation=transformation)
        C_grid = np.zeros((1, 1, 2, 2))
        C_grid[0, 0] = C
        ratio = ratio_kappaK_over_kappaJ2_grid(C_grid)[0, 0]
        # Be robust to nans/infs coming from degenerate C
        if not np.isfinite(ratio):
            return 1e6
        return float(ratio)

    # Bounds |x|, |y| <= 1 (remember variable order is (y, x))
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    # Nonlinear inequality constraint: 1 - (x^2 + y^2) >= 0 (i.e., x^2 + y^2 <= 1)
    # SLSQP supports dict-style constraints.
    constraints = [
        {
            "type": "ineq",
            "fun": lambda ij: 1.0 - (ij[0] * ij[0] + ij[1] * ij[1]),
        }
    ]

    res = opt.minimize(
        objective,
        x0=np.array([0.0, 0.0], dtype=float),  # start at the origin (inside the disk)
        method=method,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": maxiter},
    )

    i_opt, j_opt = float(res.x[0]), float(res.x[1])
    val_opt = float(objective(res.x))
    print(i_opt, j_opt, val_opt)
    return dict(i=i_opt, j=j_opt, value=val_opt)


def test_Kappa():
    N = 2000
    transformation = "none"
    C = generate_poincare_disk(resolution=N, transformation=transformation)

    # Coordinates for field construction
    x_vals = np.linspace(0, 1 * N, C.shape[1])
    y_vals = np.linspace(0, 1 * N, C.shape[0])

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    # Errors
    error_J = svd_metrics_from_gram_grid(C)["kappaJ"]
    error_J = error_J.clip(0, 10)
    error_K = kappa_star_K_from_gram_grid(C)
    error_K = error_K.clip(0, 10)
    error_ratio = ratio_kappaK_over_kappaJ2_grid(C)

    # Compute center indices and values for each field
    ic, jc = error_J.shape[0] // 2, error_J.shape[1] // 2
    center_vals = {
        "kappaJ_center": float(error_J[ic, jc]),
        "kappaK_center": float(error_K[ic, jc]),
        "ratio_center": float(error_ratio[ic, jc]),
    }

    # --- κ(J)
    cf0, cbar0, _ = _contourf_with_min(
        axs[0],
        fig,
        x_vals,
        y_vals,
        error_J,
        cmap="coolwarm",
        label=r"$\kappa(J)$",
    )
    _decorate_disk_axes(
        axs[0], N=N, transformation=transformation, hide_y=False, title=r"$\kappa(J)$"
    )
    iJmin, jJmin = np.unravel_index(np.nanargmin(error_J), error_J.shape)
    _plot_point(
        axs[0],
        x_vals[jJmin],
        y_vals[iJmin],
    )

    # --- κ(K)
    cf1, cbar1, _ = _contourf_with_min(
        axs[1],
        fig,
        x_vals,
        y_vals,
        error_K,
        cmap="coolwarm",
        label=r"$\kappa(K)$",
    )
    _decorate_disk_axes(
        axs[1], N=N, transformation=transformation, hide_y=True, title=r"$\kappa(K)$"
    )
    iKmin, jKmin = np.unravel_index(np.nanargmin(error_K), error_K.shape)
    _plot_point(
        axs[1],
        x_vals[jKmin],
        y_vals[iKmin],
    )

    # --- ratio κ*(K)/κ(J)^2
    cf2, cbar2, _ = _contourf_with_min(
        axs[2],
        fig,
        x_vals,
        y_vals,
        error_ratio,
        cmap="coolwarm",
        label=r"$\kappa(K)/\kappa(J)^2$",
    )
    _decorate_disk_axes(
        axs[2],
        N=N,
        transformation=transformation,
        hide_y=True,
        title=r"$\kappa(K)/\kappa(J)^2$",
    )
    # grid minimum
    # iRmin, jRmin = np.unravel_index(np.nanargmin(error_ratio), error_ratio.shape)
    # _plot_point(
    #     axs[2],
    #     x_vals[jRmin],
    #     y_vals[iRmin],
    #     marker="o",
    #     ms=5.0,
    #     label=rf"grid min: {error_ratio[iRmin, jRmin]:.4g}",
    # )
    # optimized minimum
    opt_min = find_minimum_ratio(transformation=transformation)
    x_plot, y_plot = _disk_to_plot(opt_min["i"], opt_min["j"], N)
    _plot_point(
        axs[2],
        x_plot,
        y_plot,
    )

    # Values at the minimum of κ(J) evaluated across all fields
    minJ_vals = {
        "kappaJ_at_minJ": float(error_J[iJmin, jJmin]),
        "kappaK_at_minJ": float(error_K[iJmin, jJmin]),
        "ratio_at_minJ": float(error_ratio[iJmin, jJmin]),
    }

    # Print summary for each subplot: center and min-κ(J) values
    print(
        "— κ(J) plot — center:",
        center_vals["kappaJ_center"],
        " | at min κ(J):",
        minJ_vals["kappaJ_at_minJ"],
    )
    print(
        "— κ(K) plot — center:",
        center_vals["kappaK_center"],
        " | at min κ(J):",
        minJ_vals["kappaK_at_minJ"],
    )
    print(
        "— κ(K)/κ(J)^2 plot — center:",
        center_vals["ratio_center"],
        " | at min κ(J):",
        minJ_vals["ratio_at_minJ"],
    )
    if transformation == "none":
        _save_plot(fig, "square_kappa_fields")
    else:
        _save_plot(fig, f"{transformation}_kappa_fields")
    plt.show()


def test():
    N = 2000
    transformation = "triangular"

    C = generate_poincare_disk(resolution=N, transformation=transformation)

    Hnorm = waldron_operator_seminorm_from_H(None)

    # error = waldron_Linf_bound_from_Cgrid_and_Hnorm(C, Hnorm)
    error = Shewchuk_error_Size_and_shape_f_g(C, Hnorm)
    # error = ratio_kappaK_over_kappaJ2_grid(C)
    # error = Shewchuk_error_Size_and_shape_grad_f_grad_g(C, Hnorm)
    # error = Shewchuk_error_Conditioning(C, Hnorm)
    error = np.clip(error, 0, 10)
    x_vals = np.linspace(0, 1 * N, C.shape[1])
    y_vals = np.linspace(0, 1 * N, C.shape[0])

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    cf, cbar, vmin_t = _contourf_with_min(
        ax, fig, x_vals, y_vals, error, cmap="coolwarm", label=r"$r_{mc}^{-2}$"
    )
    _decorate_disk_axes(ax, N=N, transformation=transformation, hide_y=False)

    # Find and plot the global minimum
    imin, jmin = np.unravel_index(np.nanargmin(error), error.shape)
    x_min = x_vals[jmin]
    y_min = y_vals[imin]
    C_min = C[imin, jmin]
    print("argmin indices (i,j):", imin, jmin)
    print("x_min, y_min:", x_min, y_min)
    print("error_min:", error[imin, jmin])
    print("C at global minimum (Gram matrix):\n", C_min)
    _plot_point(ax, x_min, y_min, marker="o", ms=6.0)

    cbar = fig.colorbar(cf, ax=ax, location="right", label=r"$r_{mc}^{-2}$")
    try:
        ticks = cbar.get_ticks()
        if ticks is None:
            ticks = np.array([])
        if (ticks.size == 0) or (abs(float(ticks[0]) - float(vmin_t)) > 1e-9):
            cbar.set_ticks(np.concatenate(([vmin_t], ticks)))
            cbar.update_ticks()
    except Exception:
        pass
    ax.set_title("Waldron L error factor (unit curvature)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")
    _set_axis_ticks_unit(ax, N)
    # ax.legend(loc="upper right")
    _save_plot(fig, "waldron_unit_curvature")
    plt.show()


def test2():
    # Choose which measure to visualize: 'f_g', 'grad', 'cond', 'kappaJ', 'kappaK', or 'sv_ratio'
    measure = "sv_ratio"
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
    elif measure == "kappaJ":
        sv = svd_metrics_from_gram_grid(C)
        error_const = sv["kappaJ"]
        error_field = error_const.copy()
    elif measure == "kappaK":
        error_const = kappa_star_K_from_gram_grid(C)
        error_field = error_const.copy()
    elif measure == "sv_ratio":
        error_const = ratio_kappaK_over_kappaJ2_grid(C)
        error_field = error_const.copy()
    else:
        raise ValueError(
            "Unknown measure: choose 'f_g', 'grad', 'cond', 'kappaJ', 'kappaK', or 'sv_ratio'"
        )

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
    cf0.set_edgecolor("face")
    _decorate_disk_axes(axs[0], N=N, transformation=transformation, hide_y=False)
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

    # Right: Option A field
    cf1 = axs[1].contourf(
        x_vals, y_vals, error_field, levels=64, vmin=vmin, vmax=vmax, cmap="coolwarm"
    )
    cf1.set_edgecolor("face")
    _decorate_disk_axes(axs[1], N=N, transformation=transformation, hide_y=True)
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

    # Left: set axis ticks and colorbar
    _set_axis_ticks_unit(axs[0], N, hide_y=False)
    cbarL = fig.colorbar(
        cf0, ax=axs[0], location="right", shrink=0.9, label="quality measure Q"
    )
    try:
        if vmin is not None and np.isfinite(vmin):
            ticksL = cbarL.get_ticks()
            if ticksL is None:
                ticksL = np.array([])
            if (ticksL.size == 0) or (abs(float(ticksL[0]) - float(vmin)) > 1e-9):
                cbarL.set_ticks(np.concatenate(([vmin], ticksL)))
                cbarL.update_ticks()
    except Exception:
        pass

    # Right: set axis ticks and colorbar
    _set_axis_ticks_unit(axs[1], N, hide_y=True)
    cbarR = fig.colorbar(
        cf1, ax=axs[1], location="right", shrink=0.9, label="quality measure Q"
    )
    try:
        if vmin is not None and np.isfinite(vmin):
            ticksR = cbarR.get_ticks()
            if ticksR is None:
                ticksR = np.array([])
            if (ticksR.size == 0) or (abs(float(ticksR[0]) - float(vmin)) > 1e-9):
                cbarR.set_ticks(np.concatenate(([vmin], ticksR)))
                cbarR.update_ticks()
    except Exception:
        pass

    _save_plot(fig, f"comparison_{measure}")
    plt.show()

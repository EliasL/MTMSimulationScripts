import numpy as np
from typing import Dict, Optional, Tuple
from MTMath.plotEnergy import generate_poincare_disk


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


def test():
    C = generate_poincare_disk(resolution=100)

    Hnorm = waldron_operator_seminorm_from_H(None)

    error = np.zeros(C.shape[:2])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            error[i, j] = waldron_Linf_bound_from_C_and_Hnorm(C[i, j], Hnorm)

    from matplotlib import pyplot as plt

    extent = (-1, 1, -1, 1)
    fig, ax = plt.subplots()
    im = ax.imshow(
        error.clip(0, 1),
        extent=extent,
        origin="lower",
        # cmap="coolwarm",
    )
    cbar = fig.colorbar(im, ax=ax, label=r"shape-only $\alpha (T)$")
    ax.set_title("Waldron L error factor (unit curvature)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

from sympy import symbols, diff, sqrt, log, lambdify, ccode, cse, simplify, Rational
import numpy as np


# The following class is a bit messy, with all the static methods and class methods.
# The reason is to avoid using global variables and to cache the symbolic computations.
class EnergyFunction:
    # Define symbols as class constants (immutables)
    _C11, _C22, _C12, _BETA, _K, _NOISE = symbols("C_{11} C_{22} C_{12} beta K noise")
    _PHI_ARGS = (_C11, _C22, _C12, _BETA, _K, _NOISE)

    # Cache for symbolic and numeric computations
    _PHI = None
    _DIV_PHI = None
    _DIV_DIV_PHI = None

    _PHI_SYMBOLIC = None
    _DIV_PHI_SYMBOLIC = None
    _DIV_DIV_PHI_SYMBOLIC = None

    @classmethod
    def _initialize_phi(cls):
        """Compute and cache the potential function _PHI."""
        if cls._PHI is None:
            cls._PHI_SYMBOLIC = cls.phi(
                cls._C11, cls._C22, cls._C12, cls._BETA, cls._K, cls._NOISE
            )
            cls._PHI = lambdify(cls._PHI_ARGS, cls._PHI_SYMBOLIC)

    @classmethod
    def _initialize_div_phi(cls):
        """Compute and cache the first derivatives of _PHI."""
        if cls._DIV_PHI is None:
            cls._initialize_phi()  # Ensure _PHI is initialized
            first_derivatives = {
                "dPhi_dC11": diff(cls._PHI_SYMBOLIC, cls._C11),
                "dPhi_dC22": diff(cls._PHI_SYMBOLIC, cls._C22),
                "dPhi_dC12": diff(cls._PHI_SYMBOLIC, cls._C12),
            }
            cls._DIV_PHI_SYMBOLIC = first_derivatives
            cls._DIV_PHI = {
                k: lambdify(cls._PHI_ARGS, v) for k, v in first_derivatives.items()
            }

    @classmethod
    def _initialize_div_div_phi(cls):
        """Compute and cache the second derivatives of _PHI."""
        if cls._DIV_DIV_PHI is None:
            cls._initialize_div_phi()  # Ensure first derivatives are initialized
            first_derivatives = cls._DIV_PHI_SYMBOLIC
            second_derivatives = {
                "dPhi_dC11_dC11": diff(first_derivatives["dPhi_dC11"], cls._C11),
                "dPhi_dC22_dC22": diff(first_derivatives["dPhi_dC22"], cls._C22),
                "dPhi_dC12_dC12": diff(first_derivatives["dPhi_dC12"], cls._C12),
                "dPhi_dC11_dC22": diff(first_derivatives["dPhi_dC11"], cls._C22),
                "dPhi_dC11_dC12": diff(first_derivatives["dPhi_dC11"], cls._C12),
                "dPhi_dC22_dC12": diff(first_derivatives["dPhi_dC22"], cls._C12),
            }
            cls.DIV_DIV_PHI_SYMBOLIC = second_derivatives
            cls._DIV_DIV_PHI = {
                k: lambdify(cls._PHI_ARGS, v) for k, v in second_derivatives.items()
            }

    @classmethod
    def _initialize_all(cls):
        """Compute and cache all potential functions and derivatives."""
        cls._initialize_phi()
        cls._initialize_div_phi()
        cls._initialize_div_div_phi()

    @classmethod
    def symbolic_potential(cls):
        cls._initialize_all()
        return cls._PHI_SYMBOLIC, cls._DIV_PHI_SYMBOLIC, cls.DIV_DIV_PHI_SYMBOLIC

    @classmethod
    def numeric_potential(cls):
        cls._initialize_all()
        return cls._PHI, cls._DIV_PHI, cls._DIV_DIV_PHI

    @classmethod
    def ground_state_energy(cls, beta=-1 / 4, K=4, noise=1):
        """Caches and returns the ground state energy."""
        if cls._PHI is None:
            cls._initialize_phi()
        return cls._PHI(1, 1, 0, beta, K, noise)

    @classmethod
    def energy_from_simple_shear(cls, shear, beta=-1 / 4, K=4, noise=1):
        """Caches and returns the ground state energy."""
        if cls._PHI is None:
            cls._initialize_phi()
        return cls._PHI(1, 1 + shear**2, shear, beta, K, noise)

    @classmethod
    def energy_from_reduced_C_components(
        cls, C11, C22, C12, beta=-1 / 4, K=4, noise=1, zeroReference=True
    ):
        if cls._PHI is None:
            cls._initialize_phi()
        energy = cls._PHI(C11, C22, C12, beta, K, noise)

        # Subtract ground state energy
        if zeroReference:
            energy -= cls.ground_state_energy(beta=beta, K=K)
        return energy

    @classmethod
    def energy_from_reduced_C(cls, C_, beta=-1 / 4, K=4, noise=1, zeroReference=True):
        assert C_.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
        C11, C22, C12 = C_[..., 0, 0], C_[..., 1, 1], C_[..., 0, 1]
        return cls.energy_from_reduced_C_components(
            C11, C22, C12, beta, K, noise, zeroReference
        )

    # Warning: This method modifies C in-place
    @classmethod
    def energy_from_C_in_place(
        cls, C, beta=-1 / 4, K=4, noise=1, zeroReference=True, loops=1000
    ):
        # Reduce using Lagrange reduction
        # C is reduced!
        lagrange_reduction(C, loops=loops)
        return cls.energy_from_reduced_C(C, beta, K, noise, zeroReference)

    @classmethod
    def energy_from_C_components_in_place(
        cls,
        C11,
        C22,
        C12,
        beta=-1 / 4,
        K=4,
        noise=1,
        zeroReference=True,
        loops=1000,
    ):
        lagrange_reduction_components(C11, C22, C12, loops=loops)
        return cls.energy_from_reduced_C_components(
            C11, C22, C12, beta, K, noise, zeroReference
        )

    @staticmethod
    def F_from_C(C, tol=1e-15, upper=True):
        """
        C = F^T F
        If upper=True  -> enforce F21 = 0   (upper-triangular F)
        If upper=False -> enforce F12 = 0   (lower-triangular F)
        """
        C11 = C[..., 0, 0]
        C12 = C[..., 0, 1]
        C22 = C[..., 1, 1]

        if upper:
            # F = [[F11, F12],
            #      [0,   F22]]
            F11 = np.sqrt(np.maximum(C11, tol))
            F12 = C12 / F11
            r_sq = C22 - F12**2
            F22 = np.sqrt(np.maximum(r_sq, 0.0))

            zeros = np.zeros_like(F11)
            F = np.stack(
                [np.stack([F11, F12], axis=-1), np.stack([zeros, F22], axis=-1)],
                axis=-2,
            )

        else:
            # F = [[F11, 0],
            #      [F21, F22]]
            F22 = np.sqrt(np.maximum(C22, tol))
            F21 = C12 / F22
            r_sq = C11 - F21**2
            F11 = np.sqrt(np.maximum(r_sq, 0.0))

            zeros = np.zeros_like(F11)
            F = np.stack(
                [np.stack([F11, zeros], axis=-1), np.stack([F21, F22], axis=-1)],
                axis=-2,
            )

        return F

    # F is a deformation gradient tensor of shape (..., 2, 2)
    # For example, it could be a X, Y grid of 2x2 matrixes.
    @classmethod
    def energy_from_F(
        cls,
        F,
        beta=-1 / 4,
        K=4,
        noise=1,
        zeroReference=True,
        returnReducedC=False,
        accuracy=1,
        loops=None,
    ):
        assert F.shape[-2:] == (2, 2), (
            f"F must have shape (..., 2, 2), but got shape {F.shape}"
        )

        # C = F^T F
        C = np.einsum("...ji,...jk->...ik", F, F)

        if accuracy < 1:
            if loops is None:
                loops = 200

        elif loops is None:
            loops = 1000

        # C is reduced in place
        energy = cls.energy_from_C_in_place(
            C, beta, K, noise, zeroReference, loops=loops
        )
        if returnReducedC:
            return energy, C
        return energy

    @classmethod
    def sigma_from_C_R(cls, C_R, beta=-1 / 4, K=4, noise=1):
        assert C_R.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
        if cls._DIV_PHI is None:
            cls._initialize_div_phi()
        C_11, C_22, C_12 = C_R[..., 0, 0], C_R[..., 1, 1], C_R[..., 0, 1]
        dPhi_dC11 = cls._DIV_PHI["dPhi_dC11"](C_11, C_22, C_12, beta, K, noise)
        dPhi_dC22 = cls._DIV_PHI["dPhi_dC22"](C_11, C_22, C_12, beta, K, noise)
        dPhi_dC12 = cls._DIV_PHI["dPhi_dC12"](C_11, C_22, C_12, beta, K, noise)
        # sigma = 1/2 (∂Φ/∂C_R + (∂Φ/∂C_R)^T)
        # Preallocate an array of shape (..., 2, 2):
        sigma = np.empty(C_R.shape, dtype=dPhi_dC11.dtype)
        sigma[..., 0, 0] = dPhi_dC11
        sigma[..., 0, 1] = dPhi_dC12 / 2
        sigma[..., 1, 0] = sigma[..., 0, 1]  # dPhi_dC12 / 2
        sigma[..., 1, 1] = dPhi_dC22
        return sigma

    @classmethod
    def P_from_F(cls, F, beta=-1 / 4, K=4, noise=1):
        """
        Compute the first Piola-Kirchhoff stress tensor P from the deformation gradient F.
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
        C = np.einsum("...ji,...jk->...ik", F, F)
        C_R = C.copy()  # to be modified in place
        M_R = lagrange_reduction(C_R, returnM=True)

        # C_E, C_R, M_E, M_R = lagrange_reduction_shears_vectorized(
        #     C, fundamental=True, returnM=True
        # )
        # print(np.allclose(C_R, C_R2, equal_nan=True))
        # print(np.allclose(M_R, M_2, equal_nan=True))

        # Find indices where they differ (excluding equal NaNs)
        # mask_diff = ~(np.isclose(M_R, M_2, equal_nan=True))

        # Get first differing matrix index
        # idx = np.argwhere(mask_diff.any(axis=(-1, -2)))
        # if idx.size > 0:
        #     i = tuple(idx[0])
        #     print("First differing 2x2 matrix:")
        #     print("M_R:")
        #     print(M_R[i])
        #     print("M_2:")
        #     print(M_2[i])
        #     print("C:", C[i])
        # else:
        #     print("All 2x2 matrices are equal (accounting for NaNs).")

        def mapBack(M, C_R):  # returns M C M^T
            Minv = np.linalg.inv(M)
            return np.swapaxes(Minv, -1, -2) @ C_R @ Minv

        # Which M maps C_R -> C ?
        # print("M_2 back to C:", np.allclose(C, congruence(M_2, C_R), equal_nan=True))
        print("M_R back to C:", np.allclose(C, mapBack(M_R, C_R), equal_nan=True))

        sigma = cls.sigma_from_C_R(C_R, beta=beta, K=K, noise=noise)

        # Swapaxes is equivalent to transpose for 2D matrices, but works for ND-arrays
        P = 2 * F @ M_R @ sigma @ M_R.swapaxes(-1, -2)
        return P

    @classmethod
    def cauchy_from_F(cls, F, M=None, beta=-1 / 4, K=4, noise=1):
        """
        Compute the Cauchy stress tensor σ from the deformation gradient F.
        Handles NaNs in F safely when computing det(F).
        Uses σ = (1/J) * P * F^T with J = det(F), where P is the first
        Piola–Kirchhoff stress from P_from_F.
        """
        # First Piola–Kirchhoff
        P = cls.P_from_F(F, beta=beta, K=K, noise=noise)

        # Compute det(F) robustly, ignoring NaNs
        F_det_safe = np.where(np.isnan(F), 0.0, F)
        J = np.linalg.det(F_det_safe)

        # Where all elements in F are NaN, det should be NaN as well
        all_nan_mask = np.all(np.isnan(F), axis=(-2, -1))
        J = np.where(all_nan_mask, np.nan, J)

        # Check for invalid determinants
        if np.any((J < 0) & ~np.isnan(J)):
            raise ValueError(
                "Non-positive J encountered in Cauchy_from_F (det(F) < 0)."
            )

        # σ = (1/J) * P * F^T
        FT = F.swapaxes(-1, -2)
        sigma_cauchy = np.einsum("...ij,...jk->...ik", P, FT)
        sigma_cauchy /= J[..., None, None]
        return sigma_cauchy

    @classmethod
    def cauchy_from_C(cls, C, M=None, beta=-1 / 4, K=4, noise=1):
        return cls.cauchy_from_F(cls.F_from_C(C), M=M, beta=beta, K=K, noise=noise)

    @classmethod
    def lagrangian_forces_from_F(cls, F, dN_dX, beta=-1 / 4, K=4, noise=1, area=0.5):
        """
        Compute the forces from the first Piola-Kirchhoff stress tensor P.
        dN_dX is the partial derivative of the shape functions with respect to the reference coordinates.
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
        assert dN_dX.shape[-2:] == (3, 2), "dN_dX must have shape (..., n_nodes(3), 2)"

        P = cls.P_from_F(F, beta=beta, K=K, noise=noise)
        # f = -P * dN_dX^T
        forces = -area * np.einsum("...ij,...jk->...ik", P, dN_dX.swapaxes(-1, -2))

        # forces is now an ND-array with shape (..., 2, n_nodes)
        # so we swap the last two axes to get forces with shape (..., n_nodes, 2)
        return forces.swapaxes(-1, -2)

    @classmethod
    def eulerian_forces_from_F(cls, F, dN_dx, beta=-1 / 4, K=4, noise=1, area=0.5):
        """
        Eulerian internal nodal forces using a single matmul, mirroring the Lagrangian form.
        Inputs:
        F      : (...,2,2)
        dN_dx  : (...,3,2)  # current gradients (row i = [∂N_i/∂x, ∂N_i/∂y])
        area   : (...,)     # current element area A
        Returns:
        forces : (...,3,2)
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (...,2,2)"
        assert dN_dx.shape[-2:] == (3, 2), "dN_dx must have shape (...,3,2)"

        sigma = cls.cauchy_from_F(F, beta=beta, K=K, noise=noise)  # (...,2,2)
        # Mirror your Lagrangian pattern: -σ @ (dN_dx)^T, then swap last two axes
        forces = -area * np.einsum("...ij,...jk->...ik", sigma, dN_dx.swapaxes(-1, -2))
        return forces.swapaxes(-1, -2)  # (...,3,2)

    # Strain is an ND-array of strain values with shape (..., 1)
    @classmethod
    def energy_from_simpleShear(
        cls,
        strain,
        beta=-1 / 4,
        K=4,
        noise=1,
        zeroReference=True,
        returnReducedC=False,
        accuracy=1,
        loops=None,
    ):
        # Create deformation gradient matrix array with strain values as F12.
        # The rest is identity
        # For each element in strain, replace it with a eye matrix, but with the
        # strain value in the F12 position and pass the rest to the
        # cls.energy_from_F function
        strain = np.atleast_1d(strain)
        F = np.tile(np.eye(2), (*strain.shape, 1, 1)).astype(float)
        F[..., 0, 1] = strain

        return cls.energy_from_F(
            F,
            beta=beta,
            K=K,
            noise=noise,
            zeroReference=zeroReference,
            returnReducedC=returnReducedC,
            accuracy=accuracy,
            loops=loops,
        )

    @classmethod
    def lagrangian_forces_from_simpleShear(
        cls, strain, dN_dX, beta=-1 / 4, K=4, noise=1, area=0.5
    ):
        """
        Compute the forces from the first Piola-Kirchhoff stress tensor P for simple shear.
        """
        strain = np.atleast_1d(strain)
        F = np.tile(np.eye(2), (*strain.shape, 1, 1)).astype(float)
        F[..., 0, 1] = strain
        forces = cls.lagrangian_forces_from_F(F, dN_dX, beta, K, noise, area)
        return forces

    @classmethod
    def eulerian_forces_from_simpleShear(
        cls, strain, dN_dx, beta=-1 / 4, K=4, noise=1, area=0.5
    ):
        """
        Compute the forces from the first Piola-Kirchhoff stress tensor P for simple shear.
        """
        strain = np.atleast_1d(strain)
        F = np.tile(np.eye(2), (*strain.shape, 1, 1)).astype(float)
        F[..., 0, 1] = strain
        forces = cls.eulerian_forces_from_F(F, dN_dx, beta, K, noise, area)
        return forces


class ContiEnergy(EnergyFunction):
    @staticmethod
    def I1(C11, C22, C12):
        return (1.0 / 3.0) * (C11 + C22 - C12)

    @staticmethod
    def I2(C11, C22, C12):
        return (1.0 / 4.0) * ((C11 - C22) ** 2) + (1.0 / 12.0) * (
            (C11 + C22 - 4 * C12) ** 2
        )

    @staticmethod
    def I3(C11, C22, C12):
        return ((C11 - C22) ** 2) * (C11 + C22 - 4 * C12) - (1.0 / 9.0) * (
            (C11 + C22 - 4 * C12) ** 3
        )

    @staticmethod
    def psi1(I1, I2, I3):
        return (
            (I1**4 * I2)
            - (41.0 * I2**3 / 99.0)
            + (7 * I1 * I2 * I3 / 66.0)
            + (I3**2 / 1056.0)
        )

    @staticmethod
    def psi2(I1, I2, I3):
        return (
            (4.0 * I2**3 / 11.0)
            + (I1**3 * I3)
            - (8.0 * I1 * I2 * I3 / 11.0)
            + (17.0 * I3**2 / 528.0)
        )

    @classmethod
    def phi_d(cls, C11, C22, C12, beta):
        sqrtDet = sqrt(C11 * C22 - C12 * C12)
        C11_norm = C11 / sqrtDet
        C22_norm = C22 / sqrtDet
        C12_norm = C12 / sqrtDet

        _I1 = cls.I1(C11_norm, C22_norm, C12_norm)
        _I2 = cls.I2(C11_norm, C22_norm, C12_norm)
        _I3 = cls.I3(C11_norm, C22_norm, C12_norm)

        return beta * cls.psi1(_I1, _I2, _I3) + cls.psi2(_I1, _I2, _I3)

    @staticmethod
    def phi_v(detC, K, noise):
        return K * (detC * noise - log(detC * noise))

    @classmethod
    def phi(cls, C11, C22, C12, beta, K, noise):
        detC = C11 * C22 - C12 * C12
        return cls.phi_d(C11, C22, C12, beta) + cls.phi_v(detC, K, noise)


class SuperSimple(EnergyFunction):
    @classmethod
    def phi(cls, C11, C22, C12, beta, K, noise):
        return Rational(1, 2) * ((C11 - 1) ** 2 + (C22 - 1) ** 2 + C12**2)


class ZeroEnergy(EnergyFunction):
    @classmethod
    def phi(cls, C11, C22, C12, beta, K, noise):
        return 0


def apply_right_trans(t, A):
    # We apply a matrix multiplication from the right: A@t
    # This is because we are using the right cauchy green stress tensor C
    # So C=F^TF, so when we apply something to F, we need to do it from
    # the right, so that F'=Ft => C' = t^TF^TFt = t^TCt.
    # If we were applying changes from the left, we need to have access to F:
    # F_=tF => C_=F^Tt^TtF. Here, unlike when applying changes from the right,
    # the changes appear "inside" C_, so we can't apply them.
    return np.einsum("...ij,...jk->...ik", A, t)


def lagrange_reduction_components(C11, C22, C12, loops=1000, returnMs=False):
    """
    Lagrange reduction of (C11, C22, C12) in place.
    If returnMs is True, returns an array M of right-multiplication matrices such that
    the reduced C = M.T @ C @ M for each entry.
    """
    if returnMs:
        # Initialize M as identity matrices everywhere, shape (...,2,2)
        M = np.zeros(C11.shape + (2, 2), dtype=float)
        M[..., 0, 0] = 1.0
        M[..., 1, 1] = 1.0

        # Define the three right-multiplication matrices
        m1 = np.array([[1, 0], [0, -1]], dtype=float)
        m2 = np.array([[0, 1], [1, 0]], dtype=float)
        m3 = np.array([[1, -1], [0, 1]], dtype=float)

    for i in range(loops):
        mask1 = C12 < 0
        # m1 (flip) operation
        C12[mask1] *= -1
        if returnMs:
            M_mask = M[mask1]
            if M_mask.shape[0] > 0:
                # Right-multiply: M <- M @ m1
                M[mask1] = apply_right_trans(m1, M_mask)

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()
        if returnMs:
            M_mask = M[mask2]
            if M_mask.shape[0] > 0:
                M[mask2] = apply_right_trans(m2, M_mask)

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break

        # m3 operation
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]
        if returnMs:
            M_mask = M[mask3]
            if M_mask.shape[0] > 0:
                M[mask3] = apply_right_trans(m3, M_mask)

        if i + 1 == loops and loops > 200:
            print("Warning: Not enough loops")
            raise (RuntimeError("Not reduced"))
    # Modifies in place
    if returnMs:
        return M


def lagrange_reduction_shears_vectorized(
    C, loops=64, eps=1e-12, fundamental=False, return_ops=False, returnM=False
):
    """
    Vectorized Lagrange/Gauss reduction using only shears:
      H(n): S = [[1, n],[0,1]]  -> C' = S^T C S,  F' = F S
      V(n): S = [[1, 0],[n,1]]  -> C' = S^T C S,  F' = F S

    Parameters
    ----------
    C : ndarray, shape (..., 2, 2), symmetric blocks [[a,b],[b,c]]
    loops : int, max iterations
    eps : float, tolerance for near-zero denominators and convergence
    fundamental : bool, if True snap to fundamental domain (b>=0, a<=c)
    return_ops : bool, if True return op codes per iter: 0 none, odd=H( n ), even=V( n )
    returnM : bool, if True also return right-multiplication matrices:
              m_E for C_E; and if fundamental, m_R for C_R

    Returns
    -------
    C_E : ndarray (...,2,2)  # after shear-only reduction
    [C_R] : if fundamental True
    [ops] : if return_ops True, int8 array (..., loops)
    [m_E] : if returnM True
    [m_R] : if returnM and fundamental True
    """
    C = np.asarray(C)
    if C.shape[-2:] != (2, 2):
        raise ValueError("C must have shape (..., 2, 2)")

    # Work on copies (preserve input)
    a = C[..., 0, 0].astype(float).copy()
    b = C[..., 0, 1].astype(float).copy()
    c = C[..., 1, 1].astype(float).copy()

    # Track ops if requested
    ops = None
    if return_ops:
        ops = np.zeros(a.shape + (loops,), dtype=np.int8)

    # Track right-multiplication matrices m (for F' = F m)
    m = None
    if returnM:
        # identity grid matching the batch shape
        m = np.zeros(a.shape + (2, 2), dtype=float)
        m[..., 0, 0] = 1.0
        m[..., 1, 1] = 1.0

    # --- helpers that update (a,b,c) and, if requested, m ---
    def apply_H(a, b, c, n_full, mask, nan_mask=None):
        # Only update indices not in nan_mask
        if not np.any(mask):
            return
        if nan_mask is not None:
            mask = mask & (~nan_mask)
        n = n_full[mask]
        if n.size == 0:
            return
        b_old = b[mask].copy()
        a_m = a[mask]
        # C update
        b[mask] = b_old + n * a_m
        c[mask] = c[mask] + (2.0 * n) * b_old + (n * n) * a_m  # a unchanged
        # m update: m <- m @ H(n) = [[1,n],[0,1]]
        if m is not None:
            pm = m[mask, 0, 0]
            qm = m[mask, 0, 1]
            rm = m[mask, 1, 0]
            sm = m[mask, 1, 1]
            m[mask, 0, 0] = pm + qm * n
            m[mask, 0, 1] = qm
            m[mask, 1, 0] = rm + sm * n
            m[mask, 1, 1] = sm

    def apply_V(a, b, c, n_full, mask, nan_mask=None):
        # Only update indices not in nan_mask
        if not np.any(mask):
            return
        if nan_mask is not None:
            mask = mask & (~nan_mask)
        n = n_full[mask]
        if n.size == 0:
            return
        b_old = b[mask].copy()
        c_m = c[mask]
        # C update
        b[mask] = b_old + n * c_m
        a[mask] = a[mask] + (2.0 * n) * b_old + (n * n) * c_m  # c unchanged
        # m update: m <- m @ V(n) = [[1,0],[n,1]]
        if m is not None:
            pm = m[mask, 0, 0]
            qm = m[mask, 0, 1]
            rm = m[mask, 1, 0]
            sm = m[mask, 1, 1]
            m[mask, 0, 0] = pm
            m[mask, 0, 1] = qm + pm * n
            m[mask, 1, 0] = rm
            m[mask, 1, 1] = sm + rm * n

    # --- main loop ---
    for t in range(loops):
        # Ignore entries with any NaN component
        nan_mask = np.isnan(a) | np.isnan(b) | np.isnan(c)
        reduced = np.abs(2.0 * b) <= np.minimum(a, c) + eps
        active = (~reduced) & (~nan_mask)
        if not np.any(active):
            break

        use_H = active & (a <= c + eps)
        use_V = active & ~use_H

        # H step (full-shaped nH to avoid flattening)
        if np.any(use_H):
            # compute rounded integers only where use_H, keep full shape
            ratioH = np.zeros_like(b, dtype=float)
            np.divide(b, a, out=ratioH, where=use_H & np.isfinite(a))
            nH = -np.rint(ratioH).astype(int)
            mask = use_H & (nH != 0)
            apply_H(a, b, c, nH, mask, nan_mask=nan_mask)
            if return_ops:
                ops[..., t][mask] = nH[mask] * 2 + 1  # odd for H

        # V step (full-shaped nV to avoid flattening)
        if np.any(use_V):
            ratioV = np.zeros_like(b, dtype=float)
            np.divide(b, c, out=ratioV, where=use_V & np.isfinite(c))
            nV = -np.rint(ratioV).astype(int)
            mask = use_V & (nV != 0)
            apply_V(a, b, c, nV, mask, nan_mask=nan_mask)
            if return_ops:
                ops[..., t][mask] = nV[mask] * 2  # even for V

    # Assemble shear-reduced (elastic) result
    C_E = np.empty_like(C, dtype=float)
    C_E[..., 0, 0] = a
    C_E[..., 0, 1] = b
    C_E[..., 1, 0] = b
    C_E[..., 1, 1] = c
    # Restore NaN in all positions where any input was NaN
    nan_mask = np.isnan(C[..., 0, 0]) | np.isnan(C[..., 0, 1]) | np.isnan(C[..., 1, 1])
    if np.any(nan_mask):
        C_E[nan_mask, :, :] = np.nan
        if returnM:
            m[nan_mask, :, :] = np.nan

    # early returns if no fundamental snapping needed
    if not fundamental:
        if return_ops and returnM:
            return C_E, ops, m  # m_E
        if return_ops:
            return C_E, ops
        if returnM:
            return C_E, m
        return C_E

    # --- snap to fundamental domain (non-shear) ---
    # b>=0 via D = diag(1,-1) when b<0; a<=c via swap P
    b1 = np.where(b < 0, -b, b)
    a1 = a
    c1 = c
    swap = c1 < a1
    a2 = np.where(swap, c1, a1)
    c2 = np.where(swap, a1, c1)
    b2 = b1

    C_R = np.empty_like(C, dtype=float)
    C_R[..., 0, 0] = a2
    C_R[..., 0, 1] = b2
    C_R[..., 1, 0] = b2
    C_R[..., 1, 1] = c2
    # Restore NaN in all positions where any input was NaN
    if np.any(nan_mask):
        C_R[nan_mask, :, :] = np.nan

    if returnM:
        # Build m_R by applying the same orthogonal post-ops on the right
        m_R = m.copy()
        # D on entries where b<0
        neg_mask = b < 0
        if np.any(neg_mask):
            # m <- m @ D, D = diag(1,-1)
            m_R[neg_mask, 0, 1] *= -1.0
            m_R[neg_mask, 1, 1] *= -1.0
        # P (swap) on entries where a and c were swapped
        if np.any(swap):
            # m <- m @ P, P swaps columns: [*,*] * [[0,1],[1,0]] swaps col0<->col1
            m_tmp0 = m_R[swap, :, 0].copy()
            m_R[swap, :, 0] = m_R[swap, :, 1]
            m_R[swap, :, 1] = m_tmp0
        # Restore NaN in all positions where any input was NaN
        if np.any(nan_mask):
            m_R[nan_mask, :, :] = np.nan
        if return_ops:
            return C_E, C_R, ops, m, m_R
        return C_E, C_R, m, m_R

    if return_ops:
        return C_E, C_R, ops
    return C_E, C_R


def lagrange_reduction(C, loops=1000, returnM=False):
    """
    Modifies C in place to its Lagrange-reduced form.
    """
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

    squeeze_back = False
    if C.ndim == 2:
        C_view = C[np.newaxis, ...]  # view, not a copy
        squeeze_back = True
    else:
        C_view = C

    # Extract views (no copy) from the promoted array
    C11, C22, C12 = C_view[..., 0, 0], C_view[..., 1, 1], C_view[..., 0, 1]

    # Call function (which modifies arrays in-place and returns M if requested)
    M = lagrange_reduction_components(C11, C22, C12, loops=loops, returnMs=returnM)

    # Explicitly enforce symmetry
    C_view[..., 1, 0] = C_view[..., 0, 1]

    if returnM:
        # Set M to NaN where C has any NaN
        mask_nan = np.isnan(C_view).any(axis=(-2, -1))
        if np.any(mask_nan):
            M[mask_nan] = np.nan
        return M[0] if squeeze_back else M
    # When returnM is False we just modify C in place and return None


def flip(matrix, row, col):
    matrix[..., row, col] *= -1


def lag_m1(matrix):
    flip(matrix, 0, 1)
    flip(matrix, 1, 1)


def lag_m2(matrix):
    # swap columns only
    c0 = matrix[..., :, 0].copy()
    matrix[..., :, 0] = matrix[..., :, 1]
    matrix[..., :, 1] = c0


# applies m3 n times
def lag_m3(matrix, n=1):
    # https://www.wolframalpha.com/input?i=%7B%7B1%2C+-1%7D%2C+%7B0%2C+1%7D%7D%5En
    multiplier_matrix = np.array([[1, -n], [0, 1]])
    # Perform matrix multiplication on the last two axes
    np.einsum("...ij,...jk->...ik", matrix, multiplier_matrix, out=matrix)


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


def generate_cpp_code(expressions_dict):
    expressions = list(expressions_dict.values())
    var_names = list(expressions_dict.keys())

    # Apply common subexpression elimination
    replacements, reduced_exprs = cse(expressions)

    # Simplify the reduced expressions
    simplified_exprs = []
    for expr in reduced_exprs:
        simplified_exprs.append(simplify(expr))
    reduced_exprs = simplified_exprs

    # Generate C++ code
    ccode_replacements = []
    for var, expr in replacements:
        try:
            ccode_replacements.append(f"double {var} = {ccode(expr)};")
        except Exception as e:
            ccode_replacements.append(f"// Error processing {var}: {str(e)}")

    ccode_expressions = []
    for i, (name, expr) in enumerate(zip(var_names, reduced_exprs)):
        try:
            ccode_expressions.append(f"double {name} = {ccode(expr)};")
        except Exception as e:
            ccode_expressions.append(f"// Error processing {name}: {str(e)}")

    # Combine with a blank line separator
    return "\n".join(ccode_replacements + [""] + ccode_expressions)


def compute_energy_and_derivatives(
    phi_func, div_phi_dict, div_div_phi_dict=None, include_second_derivatives=False
):
    # Handle energy function - wrap it in a dictionary with a single key
    energy_dict = {"phi": phi_func}
    energy_code = generate_cpp_code(energy_dict)

    # Generate combined code if second derivatives are requested
    if include_second_derivatives:
        assert div_div_phi_dict is not None, "Second derivatives must be provided."
        # Combine first and second derivatives
        combined_dict = {**div_phi_dict, **div_div_phi_dict}

        first_and_second_derivative_code = generate_cpp_code(combined_dict)

        return energy_code, first_and_second_derivative_code
    else:
        first_derivative_code = generate_cpp_code(div_phi_dict)
        return energy_code, first_derivative_code


def sanityCheck_Piola(verbose=True):
    """Quick regression test for the first Piola–Kirchhoff stress.

    Uses a simple-shear deformation with strain = 0.15 and compares
    the computed P against a stored reference matrix `true_P`.

    Returns
    -------
    ok : bool
        Whether the computed tensor matches the reference within tolerances.
    P : ndarray, shape (2, 2)
        The computed first Piola–Kirchhoff stress.
    """

    def check(true_P, gamma):
        F = np.eye(2, dtype=float)
        F[0, 1] = gamma

        # Compute P using the model (ContiEnergy inherits the implementation)
        P = ContiEnergy.P_from_F(F)

        # Compare with reference
        rtol = 1e-3
        atol = 1e-5
        ok = np.allclose(P, true_P, rtol=rtol, atol=atol, equal_nan=False)

        if verbose:
            print("Sanity check (Piola)")
            print("Reference P:\n", true_P)
            print("Computed  P:\n", P)
            err = np.abs(P - true_P)
            print("Max abs error:", np.max(err))
            print("Result:", "PASS" if ok else "FAIL")

    # Reference Piola tensor for simple shear gamma = 0.15
    true_P = np.array([[-0.017859, 0.31451], [0.3189, -0.029317]])

    # Build deformation gradient for simple shear: F = [[1, gamma],[0,1]]
    gamma = 0.15

    check(true_P, gamma)

    true_P = np.array([[0.25499, -0.28435], [-0.26254, -0.02723]])
    gamma = 0.801
    check(true_P, gamma)


if __name__ == "__main__":
    sanityCheck_Piola()
    # # Get symbolic expressions from ContiEnergy
    # phi_func, div_phi_dict, div_div_phi_dict = ContiEnergy.symbolic_potential()

    # # Choose whether to include second derivatives
    # include_second_derivatives = False  # Set to True when needed

    # # Generate the code
    # energy_code, stress_code = compute_energy_and_derivatives(
    #     phi_func, div_phi_dict, div_div_phi_dict, include_second_derivatives
    # )

    # # Output results
    # print("Energy function:\n", energy_code)
    # print("\n")

    # print("Stress function:\n", stress_code)
    # print(ContiEnergy.ground_state_energy())

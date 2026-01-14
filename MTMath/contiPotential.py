from sympy import symbols, diff, sqrt, log, lambdify, ccode, cse, simplify, Rational
import numpy as np
from typing import TypeAlias
from numpy.typing import ArrayLike, NDArray

Array: TypeAlias = ArrayLike | int | float
Array_Str: TypeAlias = Array | str


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
    def phi(cls, *args, **kwargs):
        raise RuntimeError("Phi must be implemented")
        return None

    @classmethod
    def _initialize_phi(cls):
        """Compute and cache the potential function _PHI."""
        if cls._PHI is None:
            cls._PHI_SYMBOLIC = cls.phi(
                cls._C11, cls._C22, cls._C12, cls._BETA, cls._K, cls._NOISE
            )
            cls._PHI = lambdify(cls._PHI_ARGS, cls._PHI_SYMBOLIC)
        assert cls._PHI is not None

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
            assert cls._DIV_PHI_SYMBOLIC is not None
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
        cls._initialize_phi()
        assert cls._PHI is not None
        return cls._PHI(1, 1, 0, beta, K, noise)

    @classmethod
    def energy_from_simple_shear(cls, shear, beta=-1 / 4, K=4, noise=1):
        """Caches and returns the ground state energy."""
        cls._initialize_phi()
        assert cls._PHI is not None
        return cls._PHI(1, 1 + shear**2, shear, beta, K, noise)

    @classmethod
    def energy_from_reduced_C_components(
        cls, C11, C22, C12, beta=-1 / 4, K=4, noise=1, zeroReference=True
    ):
        cls._initialize_phi()
        assert cls._PHI is not None
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
        cls._initialize_div_phi()
        assert cls._DIV_PHI is not None
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
    def S_from_C(cls, C, beta=-1 / 4, K=4, noise=1):
        """
        Compute the second Piola-Kirchhoff stress tensor P from the metric stress tensor C.
        """
        assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

        C_R = C.copy()  # to be modified in place
        M_R = lagrange_reduction(C_R, returnM=True)

        sigma = cls.sigma_from_C_R(C_R, beta=beta, K=K, noise=noise)

        # Swapaxes is equivalent to transpose for 2D matrices, but works for ND-arrays
        S = 2 * M_R @ sigma @ M_R.swapaxes(-1, -2)
        return S

    @classmethod
    def S_from_F(cls, F, beta=-1 / 4, K=4, noise=1):
        """
        Compute the second Piola-Kirchhoff stress tensor P from the deformation gradient F.
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
        C = np.einsum("...ji,...jk->...ik", F, F)
        return cls.S_from_C(C, beta=beta, K=K, noise=noise)

    @classmethod
    def P_from_F(cls, F, beta=-1 / 4, K=4, noise=1):
        """
        Compute the first Piola-Kirchhoff stress tensor P from the deformation gradient F.
        """
        S = cls.S_from_F(F, beta=beta, K=K, noise=noise)
        P = F @ S
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

        # Compute det(F) robustly and check physicality using the shared helper
        _, J = _assert_physical_det(F, "cauchy_from_F", return_J=True)

        # σ = (1/J) * P * F^T
        FT = F.swapaxes(-1, -2)
        sigma_cauchy = np.einsum("...ij,...jk->...ik", P, FT)
        sigma_cauchy /= J[..., None, None]
        return sigma_cauchy

    @classmethod
    def cauchy_from_C(cls, C, M=None, beta=-1 / 4, K=4, noise=1):
        return cls.cauchy_from_F(F_from_C(C), M=M, beta=beta, K=K, noise=noise)

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


class PieceWiseQuadratic(EnergyFunction):
    @classmethod
    # N. Perchikov & L. Truskinovsky, Journal of the Mechanics and Physics of Solids, (2024)
    def phi(cls, C11, C22, C12, kappa, xi, eta):
        J = sqrt(C11 * C22 - C12 * C12)
        return (
            0.5 * xi * (((C11 - C22) / 2) ** 2)
            + 0.5 * eta * C12**2
            + 0.5 * kappa * (J - 1) ** 2
        )


def apply_right_trans(t, A):
    # We apply a matrix multiplication from the right: A@t
    # This is because we are using the right cauchy green stress tensor C
    # So C=F^TF, so when we apply something to F, we need to do it from
    # the right, so that F'=Ft => C' = t^TF^TFt = t^TCt.
    # If we were applying changes from the left, we need to have access to F:
    # F_=tF => C_=F^Tt^TtF. Here, unlike when applying changes from the right,
    # the changes appear "inside" C_, so we can't apply them.
    # This explination is not accurate...
    return np.einsum("...ij,...jk->...ik", A, t)


def lagrange_reduction_components(C11, C22, C12, loops=1000, returnM=False):
    """
    Lagrange reduction of (C11, C22, C12) in place.
    If returnMs is True, returns an array M of right-multiplication matrices such that
    the reduced C = M.T @ C @ M for each entry.
    """
    if returnM:
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
        if returnM:
            M_mask = M[mask1]
            if M_mask.shape[0] > 0:
                # Right-multiply: M <- M @ m1
                M[mask1] = apply_right_trans(m1, M_mask)

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()
        if returnM:
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
        if returnM:
            M_mask = M[mask3]
            if M_mask.shape[0] > 0:
                M[mask3] = apply_right_trans(m3, M_mask)

        if i + 1 == loops and loops > 200:
            print("Warning: Not enough lagrange reduction loops")

            # print example of non-reduced C
            index = np.where(mask1 | mask2 | mask3)
            print("Indices of non-reduced C:", index)
            print(
                f"Example of non-reduced C: C11={C11[index][0]}, C22={C22[index][0]}, C12={C12[index][0]}"
            )

            # raise (RuntimeError("Not reduced"))
    # Modifies in place
    if returnM:
        return M


def lagrange_reduction(C, loops=1000, returnM=False):
    """
    Modifies C in place to its Lagrange-reduced form.
    """
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), "C must be symetric"

    # Extract views (no copy) from the promoted array
    C11, C22, C12 = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]

    # Call function (which modifies arrays in-place and returns M if requested)
    M = lagrange_reduction_components(C11, C22, C12, loops=loops, returnM=returnM)

    # Explicitly enforce symmetry
    C[..., 1, 0] = C[..., 0, 1]

    if returnM:
        return M
    # When returnM is False we just modify C in place and return None


def lagrange_reduction_F(F, loops=1000, returnM=False):
    """
    Lagrange reduction acting on F in place.
    We use lagrange reduction on C=F^T F to get the unimodular matrices M,
    then apply F <- F M.
    """
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (2, 2):
        raise ValueError("F must have shape (..., 2, 2)")

    # Columns of F are e1, e2
    e1 = F[..., :, 0]
    e2 = F[..., :, 1]

    # Components of C = F^T F
    C11 = np.einsum("...i,...i->...", e1, e1)
    C22 = np.einsum("...i,...i->...", e2, e2)
    C12 = np.einsum("...i,...i->...", e1, e2)

    if isinstance(C11, float):
        # Scalar case: promote to 1-element arrays for lagrange_reduction_components
        C11 = np.array([C11])
        C22 = np.array([C22])
        C12 = np.array([C12])

    # Run your existing Lagrange reduction on the components, but ask for M
    M = lagrange_reduction_components(C11, C22, C12, loops=loops, returnM=True)

    if M.shape[0] == 1 and F.shape == (2, 2):
        M = M[0, :, :]
    assert M.shape == F.shape, "M must have shape matching F"

    # Apply the accumulated unimodular matrices to F from the right:
    # F_reduced = F @ M  (vectorized over the leading dimensions)
    F[...] = np.einsum("...ij,...jk->...ik", F, M)

    if returnM:
        return M


def get_LR_M(C=None, F=None):
    assert (C is None and F is not None) or (C is not None and F is None), "Only C or F"
    if C is None:
        C_r = F.swapaxes(-1, -2) @ F
    else:
        C_r = C.copy()
    return lagrange_reduction(C_r, returnM=True)


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


def Rotation(theta: (float | np.ndarray) = 0.0) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    # Stack last two dims as 2×2
    return np.stack(
        [
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ],
        axis=-2,
    )


def SShear(
    h: Array_Str = 1.0,
    theta: Array = 0.0,
    s_conponent=(0, 1),
    returnR=False,
    getBodyRotation=False,
) -> np.ndarray:
    # --- string convenience interface, unchanged ---
    if isinstance(h, str):
        d = h.lower()
        if d in ["r", "right"]:
            return SShear(1.0, 0.0)
        elif d in ["l", "left"]:
            return SShear(-1.0, 0.0)
        elif d in ["u", "up"]:
            return SShear(-1.0, np.pi / 2)
        elif d in ["d", "down"]:
            return SShear(1.0, np.pi / 2)
        else:
            raise ValueError(f"Unknown direction: {h}")

    # --- numeric interface: fully vectorized ---
    h = np.asarray(h, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if h.ndim == 1 and theta.ndim == 1:
        h, theta = np.meshgrid(h, theta, indexing="ij")

    h, theta = np.broadcast_arrays(h, theta)

    shear = np.zeros(h.shape + (2, 2), dtype=float)
    shear[..., 0, 0] = 1.0
    shear[..., 1, 1] = 1.0
    shear[..., *s_conponent] += h
    if s_conponent == (1, 1):
        shear[..., 0, 0] = 1 / (1 + h)
    elif s_conponent == (0, 0):
        shear[..., 1, 1] = 1 / (1 + h)

    if getBodyRotation:
        R_body = get_rotation(shear)

    R = Rotation(theta)
    RT = np.swapaxes(R, -1, -2)
    rotShear = R @ shear @ RT

    if returnR:
        if getBodyRotation:
            return rotShear, R, R_body
        return rotShear, R
    if getBodyRotation:
        return rotShear, R_body
    return rotShear


def _nan_mask_and_det2x2(M: np.ndarray):
    """Return (nan_mask, det(M)) for batched 2x2 blocks, NaN-safe."""
    M = np.asarray(M)
    assert M.shape[-2:] == (2, 2), "M must have shape (..., 2, 2)"

    nan_mask = np.isnan(M).any(axis=(-1, -2))
    M_safe = np.where(np.isnan(M), 0.0, M)
    J = np.linalg.det(M_safe)

    all_nan_mask = np.all(np.isnan(M), axis=(-2, -1))
    J = np.where(all_nan_mask, np.nan, J)
    return nan_mask, J


def _assert_physical_det(M: np.ndarray, context: str, return_J: bool = False):
    """Check det(M) > 0 where defined; raise if non-physical."""
    nan_mask, J = _nan_mask_and_det2x2(M)
    if np.any((J < 0) & ~np.isnan(J)):
        print(f"Warning! {context}: encountered det < 0 (non-physical).")
    if return_J:
        return nan_mask, J
    return nan_mask


def get_rotation(F):
    """Extract the orthogonal factor R from F via SVD-based polar decomposition.

    This uses F ≈ R U with F = U Σ V^T from SVD and R = U V^T.
    The function does *not* enforce det(R) = +1; callers are
    responsible for checking that the resulting R is a proper
    rotation for their use case.
    """
    F = np.asarray(F)
    assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"

    # Track NaNs to avoid failures in SVD
    nan_mask = np.isnan(F).any(axis=(-1, -2))

    F_safe = F.copy()
    F_safe[nan_mask] = np.eye(2)

    # SVD-based polar decomposition: F = U Σ V^T, R = U V^T
    U, s, Vt = np.linalg.svd(F_safe)
    R = U @ Vt

    # Optional sanity check on det(R)
    detR = np.linalg.det(np.where(np.isnan(R), 0.0, R))
    bad_R = (detR < -1e-6) & (~nan_mask)
    if np.any(bad_R):
        # give a warning, but don't stop the execution
        print(
            "Warning: get_rotation: rotation part has det(R) < 0 "
            "for a physical F (reflection)."
        )

    # Restore NaNs where original F had NaNs
    if np.any(nan_mask):
        R[nan_mask] = np.nan

    return R


def unRotate_by_F(F, A, reverseDirection=False):
    """Un-rotate a 2x2 tensor A using the rotation part of F.

    Given the polar decomposition F = R U obtained via SVD, this
    function returns A' = R^T A R, i.e. the components of A in the
    co-rotated frame defined by R.
    """
    F = np.asarray(F)
    A = np.asarray(A)

    assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
    assert A.shape[-2:] == (2, 2), "A must have shape (..., 2, 2)"
    assert F.shape == A.shape, "F and A must have the same shape"

    # Physicality + NaN mask in one place
    nan_mask = _assert_physical_det(F, "unRotate_by_F")

    R = get_rotation(F)
    if reverseDirection:
        R = np.swapaxes(R, -1, -2)

    RT = np.swapaxes(R, -1, -2)
    A_unrot = np.einsum("...ij,...jk,...kl->...il", RT, A, R)

    if np.any(nan_mask):
        A_unrot[nan_mask] = np.nan

    return A_unrot


def remove_rotation(F):
    """Remove rotation from a 2x2 tensor A using polar decomposition.

    Interpreting A as a deformation gradient F, with polar
    decomposition F = R U, this returns the stretch-like part
    """
    F = np.asarray(F)
    assert F.shape[-2:] == (2, 2), "A must have shape (..., 2, 2)"

    nan_mask = _assert_physical_det(F, "remove_rotation")

    R = get_rotation(F)
    RT = np.swapaxes(R, -1, -2)
    U = np.einsum("...ij,...jk->...ik", RT, F)

    if np.any(nan_mask):
        U[nan_mask] = np.nan

    return U


def F_from_C(C, theta=np.pi / 3):
    """
    Return the symmetric positive-definite square root of C.

    Given C = F^T F (right Cauchy–Green tensor), this function returns
    the unique symmetric positive-definite tensor U such that

        U^T U = C,

    i.e. the right stretch U in the polar decomposition F = R U.

    Optionally, you can choose a rotation theta

    Note:
        From C alone the rotation R is not identifiable; only U is.
        The `upper` argument is kept only for backward compatibility
        and is ignored.
    """
    C = np.asarray(C)
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

    # Track NaNs to reinsert later
    nan_mask = np.isnan(C).any(axis=(-1, -2))

    # Create a safe copy where NaN blocks are replaced with identity
    C_safe = C.copy()
    C_safe[nan_mask] = np.eye(2)

    R = Rotation(theta)
    # C_safe = R.T @ C_safe @ R

    # Eigen-decomposition of symmetric 2x2 blocks (vectorized)
    # C = Q diag(λ) Q^T
    evals, evecs = np.linalg.eigh(C_safe)

    # # Suppose for 2×2 block we want to swap index 0 and 1
    # idx = np.array([1, 0])  # swap first ↔ second

    # # Permute eigenvalues and eigenvectors accordingly
    # evals = evals[..., idx]
    # evecs = evecs[..., :, idx]

    # Reconstruct C from its eigen-decomposition: C = Q Λ Q^T
    # Multiply columns of Q by the eigenvalues
    C_recon = (evecs * evals[..., None, :]) @ evecs.swapaxes(-1, -2)

    # Check reconstruction (with numerical tolerance)
    assert np.allclose(C_safe, C_recon), "Not able to reconstruct!"

    # Check positive semi-definiteness
    assert np.all(evals >= 0), "Negative Eigen values"
    sqrt_evals = np.sqrt(evals)

    # Build diag(sqrt(λ)) as a matrix with the same broadcast shape as C
    sqrt_diag = np.zeros_like(C, dtype=float)
    sqrt_diag[..., 0, 0] = sqrt_evals[..., 0]
    sqrt_diag[..., 1, 1] = sqrt_evals[..., 1]

    # U = Q diag(sqrt(λ)) Q^T
    F = evecs @ sqrt_diag @ evecs.swapaxes(-1, -2)
    # F = R.T @ F @ R
    # Restore NaNs where original C had NaNs
    if np.any(nan_mask):
        F[nan_mask, :, :] = np.nan

    return F

def in_elastic_domain(C11, C22, C12)->NDArray[np.bool_]:
    """Return True where (C11, C22, C12) lies in the elastic domain.

    Minimal checks:
      - min(C11, C22) > 0
      - |C12| <= 0.5 * min(C11, C22)

    Works with scalars or NumPy arrays (broadcasting). For array inputs, returns
    a boolean array with the broadcasted shape.
    """
    C11a = np.asarray(C11)
    C22a = np.asarray(C22)
    C12a = np.asarray(C12)

    Cmin = np.minimum(C11a, C22a)
    inside = (Cmin > 0) & (np.abs(C12a) <= 0.5 * Cmin)
    return inside

def elastic_domain_quadrant(C) -> NDArray[np.int_]:
    """Return elastic-domain quadrant label (0..3), or -1 if outside.

    Quadrants:
      0: C11>0, C11<=C22,  0 <= C12 <= 0.5*C11
      1: C11>0, C11<=C22, -0.5*C11 <= C12 <  0
      2: C22>0, C22<=C11,  0 <= C12 <= 0.5*C22
      3: C22>0, C22<=C11, -0.5*C22 <= C12 <  0

    Boundary convention: C12 == 0 goes to the non-negative side (0 or 2).
    Works with scalars or NumPy arrays (broadcasting). For array inputs, returns
    an integer array with the broadcasted shape.
    """
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), "C must be symetric"

    C11a = np.asarray(C[..., 0,0])
    C22a = np.asarray(C[..., 1,1])
    C12a = np.asarray(C[..., 0,1])


    q0 = (C11a > 0) & (C11a <= C22a) & (C12a >= 0) & (C12a <= 0.5 * C11a)
    q1 = (C11a > 0) & (C11a <= C22a) & (C12a < 0) & (C12a >= -0.5 * C11a)
    q2 = (C22a > 0) & (C22a <= C11a) & (C12a >= 0) & (C12a <= 0.5 * C22a)
    q3 = (C22a > 0) & (C22a <= C11a) & (C12a < 0) & (C12a >= -0.5 * C22a)

    labels = np.full(C11a.shape, np.nan)
    labels[q0] = 0
    labels[q1] = 1
    labels[q2] = 2
    labels[q3] = 3

    return labels

def elastic_reduction_components(C11, C22, C12, loops=1000):
    """Vectorized elastic reduction of symmetric 2x2 C via component updates.

    Parameters
    ----------
    C11, C22, C12 : array_like
        Components of symmetric C = [[C11, C12],[C12, C22]].
        Can be scalars or NumPy arrays; will be broadcast to a common shape.
    loops : int
        Max iterations.

    Returns
    -------
    C11r, C22r, C12r : ndarray or scalars
        Reduced components. (Scalar in -> scalar out, array in -> array out.)
    """
    C11a = np.asarray(C11, dtype=float)
    C22a = np.asarray(C22, dtype=float)
    C12a = np.asarray(C12, dtype=float)

    C11b, C22b, C12b = np.broadcast_arrays(C11a, C22a, C12a)

    # Work on copies to avoid surprising in-place effects for views
    a = C11b.copy()
    b = C22b.copy()
    c = C12b.copy()

    def _in_elastic(a_, b_, c_):
        Cmin = np.minimum(a_, b_)
        return (Cmin >= 0) & (np.abs(c_) <= 0.5 * Cmin)
    done= False
    for _ in range(loops):
        inside = _in_elastic(a, b, c)
        active = ~inside
        if not np.any(active):
            done=True
            break

        use_U = active & (a < b)
        use_V = active & ~use_U  # includes a >= b

        # m = sign(-c/a) or sign(-c/b), but avoid divide-by-zero / NaNs
        mU = np.zeros_like(a)
        mV = np.zeros_like(a)

        if np.any(use_U):
            denom = np.where(a == 0, 1.0, a)
            mU = np.sign(-c / denom)
            mU = np.where(np.isfinite(mU), mU, 0.0)

        if np.any(use_V):
            denom = np.where(b == 0, 1.0, b)
            mV = np.sign(-c / denom)
            mV = np.where(np.isfinite(mV), mV, 0.0)

        # --- Apply U_m where selected: W = [[1,m],[0,1]] ---
        # a' = a
        # c' = c + m*a
        # b' = b + 2*m*c + m^2*a
        if np.any(use_U):
            m = mU
            c_new = c + m * a
            b_new = b + 2.0 * m * c + (m * m) * a

            c = np.where(use_U, c_new, c)
            b = np.where(use_U, b_new, b)
            # a unchanged for U

        # --- Apply V_m where selected: W = [[1,0],[m,1]] ---
        # b' = b
        # c' = c + m*b
        # a' = a + 2*m*c + m^2*b
        if np.any(use_V):
            m = mV
            c_new = c + m * b
            a_new = a + 2.0 * m * c + (m * m) * b

            c = np.where(use_V, c_new, c)
            a = np.where(use_V, a_new, a)
            # b unchanged for V
    
    if not done:
        print("Warning! Not enough loops in elastic reduction!")
    # Preserve scalar return type when inputs are scalars
    if a.shape == ():
        return float(a), float(b), float(c)
    return a, b, c

def elastic_reduction(C, loops=1000):
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"
    assert np.allclose(C[..., 1, 0], C[..., 0, 1], equal_nan=True), "C must be symetric"

    # Extract views (no copy) from the promoted array
    C11, C22, C12 = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]

    # Call function (which modifies arrays in-place and returns M if requested)
    C11, C22, C12 =elastic_reduction_components(C11, C22, C12, loops=loops)

    C[..., 0, 0]=C11
    C[..., 1, 1]=C22
    C[..., 0, 1]=C12
    C[..., 1, 0]=C12

    return C

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


def debug_symbolic_cauchy_trace():
    """
    Use SymPy to compute the symbolic trace of the Cauchy stress
    for the ContiEnergy model in 2D under a diagonal deformation
    F = diag(a, b).

    It prints a closed-form expression for tr(σ) and shows that for
    det(F) = 1 and noise = 1 the trace is exactly zero, i.e. the
    isochoric part is deviatoric and all trace is volumetric.
    """
    from sympy import symbols, simplify

    # Symbols for principal stretches
    a, b = symbols("a b", positive=True)

    # Use the canonical C, beta, K, noise symbols from EnergyFunction
    C11, C22, C12 = EnergyFunction._C11, EnergyFunction._C22, EnergyFunction._C12
    beta, K, noise = (
        EnergyFunction._BETA,
        EnergyFunction._K,
        EnergyFunction._NOISE,
    )

    # Symbolic energy density for ContiEnergy
    phi_sym = ContiEnergy.phi(C11, C22, C12, beta, K, noise)

    # 2nd Piola–Kirchhoff stress S = 2 ∂φ/∂C
    dPhi_dC11 = diff(phi_sym, C11)
    dPhi_dC22 = diff(phi_sym, C22)
    # dPhi_dC12 is not required for the trace in the diagonal case

    S11 = 2 * dPhi_dC11
    S22 = 2 * dPhi_dC22

    # Specialize to C = diag(a^2, b^2), C12 = 0
    subs_diag = {C11: a**2, C22: b**2, C12: 0}
    S11_ab = S11.subs(subs_diag)
    S22_ab = S22.subs(subs_diag)

    # J = det(F) = a * b
    J = a * b

    # For F = diag(a,b):
    # P = F S, σ = (1/J) P F^T ⇒
    # σ11 = (a^2 / J) * S11,  σ22 = (b^2 / J) * S22
    tr_sigma = simplify((a**2 / J) * S11_ab + (b**2 / J) * S22_ab)

    print("Symbolic trace tr(σ) for F = diag(a, b):")
    print("tr(σ)(a, b, beta, K, noise) =")
    print(tr_sigma)

    # Express in terms of J = det(F) only
    J_sym = symbols("J", positive=True)
    tr_sigma_J = simplify(tr_sigma.subs({a * b: J_sym}))
    print("\nIn terms of J = det(F):")
    print("tr(σ)(J, beta, K, noise) =")
    print(tr_sigma_J)

    # Show that for J = 1 and noise = 1 the trace is exactly zero
    tr_sigma_iso = simplify(tr_sigma_J.subs({J_sym: 1, noise: 1}))
    print("\nFor J = 1 and noise = 1:")
    print("tr(σ) =", tr_sigma_iso)


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


def sanityCheck_LagrangeReduction(C):
    C_reduced = C.copy()
    M = lagrange_reduction(C_reduced, returnM=True)
    test = M.T @ C @ M
    assert np.allclose(test, C_reduced), "Lagrange reduction failed sanity check"



    def mapBack(M, C_R):  # returns M C M^T
        Minv = np.linalg.inv(M)
        return np.swapaxes(Minv, -1, -2) @ C_R @ Minv

    passed = np.allclose(C, mapBack(M_R, C_R), equal_nan=True)
    # Which M maps C_R -> C ?
    print("M_R back to C:", passed)
    return passed

def debug_elastic_reduction():
    F = np.array([[1,1],[0,1]])
    C = F.T@F
    C_R = elastic_reduction(C)
    print(C)
    print(C_R)



if __name__ == "__main__":
    #debug_symbolic_cauchy_trace()
    # sanityCheck_Piola()
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

    debug_elastic_reduction()
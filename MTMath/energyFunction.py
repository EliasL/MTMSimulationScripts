from sympy import (
    Matrix,
    diff,
    hessian,
    lambdify,
    log,
    simplify,
    sqrt,
    symbols,
    Rational,
    S,
    Mul,
)
import math
import numpy as np
from typing import TypeAlias
from numpy.typing import ArrayLike
from .reduction import lagrange_reduction, lagrange_reduction_components

Array: TypeAlias = ArrayLike | int | float
Array_Str: TypeAlias = Array | str


# The following class is a bit messy, with all the static methods and class methods.
# The reason is to avoid using global variables and to cache the symbolic computations.
class EnergyFunction:
    # Define symbols as class constants (immutables)
    _C11, _C22, _C12, _BETA, _K, _NOISE = symbols("C_{11} C_{22} C_{12} beta K noise")
    _PHI_ARGS = (_C11, _C22, _C12, _BETA, _K, _NOISE)
    _C_VARS = Matrix([_C11, _C22, _C12])

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

    @classmethod
    def _initialize_div_phi(cls):
        """Compute and cache the first derivatives of _PHI."""
        if cls._DIV_PHI is None:
            cls._initialize_phi()  # Ensure _PHI is initialized
            grad = Matrix([cls._PHI_SYMBOLIC]).jacobian(cls._C_VARS)
            cls._DIV_PHI_SYMBOLIC = grad
            cls._DIV_PHI = lambdify(cls._PHI_ARGS, grad)

    @classmethod
    def _initialize_div_div_phi(cls):
        """Compute and cache the second derivatives of _PHI."""
        if cls._DIV_DIV_PHI is None:
            cls._initialize_div_phi()  # Ensure first derivatives are initialized
            H = hessian(cls._PHI_SYMBOLIC, cls._C_VARS)
            cls._DIV_DIV_PHI_SYMBOLIC = H
            cls._DIV_DIV_PHI = lambdify(cls._PHI_ARGS, H)

    @classmethod
    def _initialize_phi_divs(cls):
        """Compute and cache all potential functions and derivatives."""
        cls._initialize_phi()
        cls._initialize_div_phi()
        cls._initialize_div_div_phi()

    @classmethod
    def symbolic_potential(cls):
        cls._initialize_phi_divs()
        return cls._PHI_SYMBOLIC, cls._DIV_PHI_SYMBOLIC, cls._DIV_DIV_PHI_SYMBOLIC

    @classmethod
    def numeric_potential(cls):
        cls._initialize_phi_divs()
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
        C, M = lagrange_reduction(C, loops=loops)
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
        C11, C22, C12, M = lagrange_reduction_components(C11, C22, C12, loops=loops)
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
        dPhi = np.asarray(cls._DIV_PHI(C_11, C_22, C_12, beta, K, noise))
        dPhi_dC11 = dPhi[0, 0]
        dPhi_dC22 = dPhi[0, 1]
        dPhi_dC12 = dPhi[0, 2]
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

        C_R, M_R = lagrange_reduction(C)

        sigma = cls.sigma_from_C_R(C_R, beta=beta, K=K, noise=noise)

        # Definition: S = 2 * ∂φ/∂C.
        # The factor 2 comes from the constitutive definition (not from reduction).
        # Swapaxes is equivalent to transpose for 2D matrices, but works for ND-arrays.
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
    def P_from_C(cls, C, beta=-1 / 4, K=4, noise=1):
        """
        Compute the first Piola-Kirchhoff stress tensor P from the metric tensor C, using the polar decomposition to find F.
        """
        F = F_from_C(C)
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
        _, J = _assert_physical_det(F, "cauchy_from_F")

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
        area_arr = np.asarray(area)
        scale = area_arr if area_arr.ndim == 0 else area_arr[..., None, None]
        forces = -scale * np.einsum("...ij,...jk->...ik", P, dN_dX.swapaxes(-1, -2))

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
        # Mirror the Lagrangian pattern: -σ @ (dN_dx)^T, then swap last two axes
        area_arr = np.asarray(area)
        scale = area_arr if area_arr.ndim == 0 else area_arr[..., None, None]
        forces = -scale * np.einsum("...ij,...jk->...ik", sigma, dN_dx.swapaxes(-1, -2))
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

    @classmethod
    def elasticity_tensor(cls, F, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True):
        """
        A_{iJkL} = ∂^2Φ/( ∂F_{iJ}∂F_{kL})
        Formulation in terms of dPhi_dC22_dC12 ('C') from Roberta Baggio's thesis, page 74:

                A_{iJkL} = 2 'C'_{RJSL}F_{kR}F_{iS} + S_{JL}δ_{ik}

        Where S is the second Piola-Kirchhoff stress tensor

        Also called linearized elastic moduli or first elasticity tensor in Robertas thesis.
        Refered to as elastic moduli associated to the pairs (S,F) on Wikipedia: https://en.wikipedia.org/wiki/Incremental_deformations
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"

        cls._initialize_div_div_phi()
        assert cls._DIV_DIV_PHI is not None

        # Compute C and its Lagrange reduction (for the potential defined on reduced C)
        C = np.einsum("...ji,...jk->...ik", F, F)
        C_R, M_R = lagrange_reduction(C, loops=loops)
        C_11, C_22, C_12 = C_R[..., 0, 0], C_R[..., 1, 1], C_R[..., 0, 1]

        # Hessian of Phi with respect to (C11, C22, C12)
        H_raw = cls._DIV_DIV_PHI(C_11, C_22, C_12, beta, K, noise)
        H = np.asarray(H_raw, dtype=float)
        # SymPy lambdify returns a 3x3 matrix with the batch dimension last (3,3,...).
        # Move those two matrix axes to the end so we consistently work with (...,3,3).
        H = np.moveaxis(H, (0, 1), (-2, -1))
        if H.shape[-2:] != (3, 3):
            raise ValueError(f"Hessian has unexpected shape {H.shape}")

        # Map symmetric derivatives into full tensor with shear scaling:
        # C4_red_{ijkl} = E_{ij,a} H_{ab} E_{kl,b}
        E = np.zeros((2, 2, 3), dtype=float)
        E[0, 0, 0] = 1.0
        E[1, 1, 1] = 1.0
        E[0, 1, 2] = 0.5
        E[1, 0, 2] = 0.5
        C4_red = np.einsum("ija,...ab,klb->...ijkl", E, H, E)
        # Definition in Roberta Baggio's thesis: C_{IJKL} = 2 * ∂²φ/∂C_{IJ}∂C_{KL}.
        # Our Hessian H is just ∂²φ/∂C∂C, so multiply by 2 here.
        C4_red *= 2.0

        # Transform reduced C-tensor back to the original basis (ignore dM/dC)
        C4 = np.einsum(
            "...ir,...js,...kt,...lu,...rstu->...ijkl", M_R, M_R, M_R, M_R, C4_red
        )

        # Second Piola-Kirchhoff stress from F
        S = cls.S_from_F(F, beta=beta, K=K, noise=noise)

        # From Roberta's thesis
        #    A_{iJkL} = 2 * C_{R J S L} F_{kR} F_{iS} + S_{JL} δ_{ik}
        # MISTAKE: Thesis should have kS and iR instead of kR and iS.
        term1 = 2 * np.einsum("...rjsl,...ks,...ir->...ijkl", C4, F, F)
        term2 = np.einsum("...jl,ik->...ijkl", S, np.eye(2, dtype=S.dtype))

        # A is (mixed) Lagrangian
        A = term1 + term2

        if eulerian:
            # Push-forward A to a
            a = np.einsum("...jR,...lS,...iRkS->...ijkl", F, F, A)
            return a
        else:
            return A

    @staticmethod
    def print_tangent_elasticity_index_map():
        """
        Print a simple index map for A_{iJkL} = dP_{iJ}/dF_{kL}
        and return it as a (2,2,2,2) tensor of strings.
        """
        tensor = EnergyFunction.tangent_elasticity_index_tensor()
        print("Index map tensor A_{iJkL} = dP_{iJ}/dF_{kL}:")
        print(tensor)
        return tensor

    @staticmethod
    def tangent_elasticity_index_tensor():
        """
        Return a (2,2,2,2) tensor of strings for A_{iJkL} = dP_{iJ}/dF_{kL}.
        """
        tensor = np.empty((2, 2, 2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        tensor[i, j, k, l] = f"dP{i}{j}/dF{k}{l}"
        return tensor

    @classmethod
    def acoustic_tensor(
        cls, F, n, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True
    ):
        """
        Acoustic tensor q_{ik} = n_j n_l a_{ijkl},
        where a is the tangent elasticity tensor computed from reduced C.

        Expected shapes:
        - F: (..., 2, 2)
        - n: (2,) or (N, 2)

        If n is (N, 2), the n-axis is placed after the F batch axes and
        before the final (2,2) of q.
        """
        F = np.asarray(F, dtype=float)
        n = np.asarray(n, dtype=float)
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
        assert n.shape[-1] == 2, "n must have shape (..., 2)"
        if n.ndim not in (1, 2):
            raise ValueError("n must have shape (2,) or (N, 2)")

        a = cls.elasticity_tensor(
            F, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )
        expected_q_shape = F.shape[:-2] + n.shape[:-1] + (2, 2)
        # Estimate memory of q and print a warning if it requires more than 20GB
        q_bytes = math.prod(expected_q_shape) * np.dtype(float).itemsize
        if q_bytes > 20 * 1024**3:
            print(
                "Warning: acoustic_tensor q will be large "
                f"({q_bytes / 1024**3:.2f} GB) with shape {expected_q_shape}."
            )

        if n.ndim == 1:
            n_exp = n.reshape((1,) * (F.ndim - 2) + (2,))
            q = np.einsum("...j,...l,...ijkl->...ik", n_exp, n_exp, a)
        else:
            # Broadcast n across the F batch axes (n-axis comes after F batch axes).
            n_exp = n.reshape((1,) * (F.ndim - 2) + n.shape)
            q = np.einsum("...nj,...nl,...ijkl->...nik", n_exp, n_exp, a)
        assert q.shape == expected_q_shape
        return q

    @classmethod
    def acoustic_tensor_eigenvalues(
        cls, F, n, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True
    ):
        """
        Eigenvalues of the acoustic tensor. If eulerian=True, uses q(n);
        otherwise uses the Lagrangian acoustic tensor Q(N).
        """
        q = cls.acoustic_tensor(
            F, n, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )
        # Enforce symmetry for numerical stability before eigendecomposition.
        q = 0.5 * (q + q.swapaxes(-1, -2))
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            eigs = np.linalg.eigvalsh(q)
        nonfinite_q = ~np.isfinite(q).all(axis=(-1, -2))
        eigs = np.where(nonfinite_q[..., None], np.nan, eigs)
        eigs = np.where(np.isfinite(eigs), eigs, np.nan)
        return eigs

    @classmethod
    def acoustic_determinant(
        cls, F, n, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True
    ):
        """
        Determinant of the acoustic tensor. If eulerian=True, use q(n);
        otherwise use the Lagrangian acoustic tensor Q(N).
        """
        q = cls.acoustic_tensor(
            F, n, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )

        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            det_q = np.linalg.det(q)

        # If q is non-finite anywhere, det should be NaN.
        nonfinite_q = ~np.isfinite(q).all(axis=(-1, -2))
        det_q = np.where(nonfinite_q, np.nan, det_q)
        det_q = np.where(np.isfinite(det_q), det_q, np.nan)
        return det_q

    @classmethod
    def stability(cls, F, n, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True):
        """
        Return True where det(acoustic tensor) >= 0. If eulerian=True, use q(n);
        otherwise use the Lagrangian acoustic tensor Q(N).
        """
        det_q = cls.acoustic_determinant(
            F, n, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )
        if np.asarray(n).ndim == 2:
            finite = np.isfinite(det_q)
            boolField = np.all((det_q > 0) & finite, axis=-1)
            assert np.all(F.shape[:-2] == boolField.shape)
            return boolField
        return (det_q > 0) & np.isfinite(det_q)

    @classmethod
    def min_det_angle(cls, F, n, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True):
        """
        Return (angle, det_min) where angle is the n-direction (rad) that
        minimizes det(q), and det_min is the minimum determinant itself.
        """
        F = np.asarray(F, dtype=float)
        n = np.asarray(n, dtype=float)
        if n.shape[-1] != 2:
            raise ValueError("n must have shape (..., 2)")
        if n.ndim not in (1, 2):
            raise ValueError("n must have shape (2,) or (N, 2)")

        theta = np.arctan2(n[..., 1], n[..., 0])
        det_q = cls.acoustic_determinant(
            F, n, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )

        if n.ndim == 1:
            min_det = det_q
            angle = float(theta)
            if np.asarray(min_det).ndim != 0:
                raise ValueError("Expected scalar det(q) for n shape (2,)")
            return angle, float(min_det)

        # n is (N,2): det_q is (..., N)
        all_nan = ~np.isfinite(det_q).any(axis=-1)
        det_q_safe = np.where(np.isfinite(det_q), det_q, np.inf)
        argmin = np.argmin(det_q_safe, axis=-1)
        angle = np.take(theta, argmin)
        min_det = np.min(det_q_safe, axis=-1)
        if np.ndim(min_det) == 0:
            return float(angle), (float(min_det) if np.isfinite(min_det) else np.nan)
        angle = angle.astype(float, copy=False)
        min_det = min_det.astype(float, copy=False)
        angle = np.where(all_nan, np.nan, angle)
        min_det = np.where(all_nan, np.nan, min_det)
        return angle, min_det

    @staticmethod
    def _voigt_from_A(A):
        """
        Map a 4th-order tensor A_{ijkl} to a 3x3 Voigt-like matrix using symmetric C components.
        Input shape: (..., 2, 2, 2, 2)
        Output shape: (..., 3, 3)

        We use the ordering (11, 22, 12) and the same shear scaling (1/2) as in
        sigma_from_C_R. The mapping is
          A_voigt[a,b] = E_{ij,a} * A_{ijkl} * E_{kl,b},
        where E maps (11,22,12) to (ij) with a 1/2 on shear.
        """
        A = np.asarray(A, dtype=float)
        assert A.shape[-4:] == (2, 2, 2, 2), (
            f"A must have shape (...,2,2,2,2), got {A.shape}"
        )
        # Warn once if A is not symmetric in the minor index pairs (ij) or (kl).
        if not hasattr(EnergyFunction._voigt_from_A, "_warned"):
            EnergyFunction._voigt_from_A._warned = False
        if not EnergyFunction._voigt_from_A._warned:
            sym_ij = np.allclose(A, A.swapaxes(-4, -3), equal_nan=True)
            sym_kl = np.allclose(A, A.swapaxes(-2, -1), equal_nan=True)
            if not (sym_ij and sym_kl):
                print(
                    "Warning: _voigt_from_A assumes minor symmetries in (ij) and (kl); "
                    "A is not symmetric, so Voigt mapping is a symmetrized projection."
                )
                EnergyFunction._voigt_from_A._warned = True
        E = np.zeros((2, 2, 3), dtype=float)
        E[0, 0, 0] = 1.0
        E[1, 1, 1] = 1.0
        E[0, 1, 2] = 0.5
        E[1, 0, 2] = 0.5
        return np.einsum("ija,...ijkl,klb->...ab", E, A, E)

    @classmethod
    def moduli_at_F(cls, F, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True):
        """
        Return mu from the tangent tensor using Voigt component (12,12).
        If eulerian=True, use the pushed-forward tensor a; otherwise use A.
        Note: matches linearized isotropic definition when evaluated at the
        reference configuration.
        """
        A = cls.elasticity_tensor(
            F, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )
        t_voigt = cls._voigt_from_A(A)
        # https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
        mu = t_voigt[..., 2, 2]  # Shear modulus
        Lambda = t_voigt[..., 1, 0]  # First Lame parameter
        return mu, Lambda

    @classmethod
    def moduli_at_C(cls, C, beta=-1 / 4, K=4, noise=1, loops=1000, eulerian=True):
        F = F_from_C(C)
        return cls.moduli_at_F(
            F, beta=beta, K=K, noise=noise, loops=loops, eulerian=eulerian
        )


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
            + 0.5 * kappa * (J - S.One) ** 2
        )


def F_from_C(C):
    """
    Given C = F^T F (right Cauchy–Green tensor), this function returns
    the unique symmetric positive-definite tensor U such that
        U^T U = C,
    """
    C = np.asarray(C)
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

    # Track NaNs to reinsert later
    nan_mask = np.isnan(C).any(axis=(-1, -2))

    # Create a safe copy where NaN blocks are replaced with identity
    C_safe = C.copy()
    C_safe[nan_mask] = np.eye(2)

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


def rotation(theta: (float | np.ndarray) = 0.0) -> np.ndarray:
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


def _sshear_direction_to_params(h: str) -> tuple[float, float]:
    """Map a string direction to (h, theta)."""
    d = h.lower()
    if d in ["r", "right"]:
        return 1.0, 0.0
    elif d in ["l", "left"]:
        return -1.0, 0.0
    elif d in ["u", "up"]:
        return -1.0, np.pi / 2
    elif d in ["d", "down"]:
        return 1.0, np.pi / 2
    else:
        raise ValueError(f"Unknown direction: {h}")


def _SShear_core(
    h: Array_Str = 1.0,
    theta: Array = 0.0,
    s_conponent=(0, 1),
    *,
    compute_R_body: bool = False,
):
    # --- string convenience interface ---
    if isinstance(h, str):
        h_num, theta_num = _sshear_direction_to_params(h)
        h = h_num
        theta = theta_num

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

    R_body = get_rotation(shear) if compute_R_body else None

    R = rotation(theta)
    RT = R.swapaxes(-1, -2)
    rotShear = R @ shear @ RT

    return rotShear, R, R_body


def SShear(
    h: Array_Str = 1.0,
    theta: Array = 0.0,
    s_conponent=(0, 1),
) -> np.ndarray:
    """Return only the rotated shear tensor (backward-compatible default)."""
    rotShear, _, _ = _SShear_core(h=h, theta=theta, s_conponent=s_conponent)
    return rotShear


def SShear_with_R(
    h: Array_Str = 1.0,
    theta: Array = 0.0,
    s_conponent=(0, 1),
):
    """Return (rotShear, R), where R is the imposed rigid rotation Rotation(theta)."""
    rotShear, R, _ = _SShear_core(h=h, theta=theta, s_conponent=s_conponent)
    return rotShear, R


def SShear_with_R_body(
    h: Array_Str = 1.0,
    theta: Array = 0.0,
    s_conponent=(0, 1),
):
    rotShear, R, R_body = _SShear_core(
        h=h, theta=theta, s_conponent=s_conponent, compute_R_body=True
    )
    assert R_body is not None
    return rotShear, R, R_body


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


def _assert_physical_det(M: np.ndarray, context: str="M"):
    """Check det(M) > 0 where defined; raise if non-physical."""
    nan_mask, J = _nan_mask_and_det2x2(M)
    if np.any((J < 0) & ~np.isnan(J)):
        print(f"Warning! {context}: encountered det < 0 (non-physical).")
    return nan_mask, J


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


def debug_symbolic_cauchy_trace():
    """
    Use SymPy to compute the symbolic trace of the Cauchy stress
    for the ContiEnergy model in 2D under a diagonal deformation
    F = diag(a, b).

    It prints a closed-form expression for tr(σ) and shows that for
    det(F) = 1 and noise = 1 the trace is exactly zero, i.e. the
    isochoric part is deviatoric and all trace is volumetric.
    """

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

    S11 = S(2) * dPhi_dC11
    S22 = S(2) * dPhi_dC22

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


if __name__ == "__main__":
    pass

    # # Output results
    # print("Energy function:\n", energy_code)
    # print("\n")

    # print("Stress function:\n", stress_code)
    # print(ContiEnergy.ground_state_energy())

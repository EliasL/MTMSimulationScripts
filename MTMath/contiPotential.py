from sympy import symbols, diff, sqrt, log, lambdify
import numpy as np


# The following class is a bit messy, with all the static methods and class methods.
# The reason is to avoid using global variables and to cache the symbolic computations.
class ContiEnergy:
    # Define symbols as class constants (immutables)
    _C11, _C22, _C12, _BETA, _K, _NOISE = symbols("c11 c22 c12 beta K noise")
    _PHI_ARGS = (_C11, _C22, _C12, _BETA, _K, _NOISE)

    # Cache for symbolic and numeric computations
    _PHI = None
    _DIV_PHI = None
    _DIV_DIV_PHI = None
    _DIV_PHI_SYMBOLIC = None

    @staticmethod
    def I1(c11, c22, c12):
        return (1.0 / 3.0) * (c11 + c22 - c12)

    @staticmethod
    def I2(c11, c22, c12):
        return (1.0 / 4.0) * ((c11 - c22) ** 2) + (1.0 / 12.0) * (
            (c11 + c22 - 4 * c12) ** 2
        )

    @staticmethod
    def I3(c11, c22, c12):
        return ((c11 - c22) ** 2) * (c11 + c22 - 4 * c12) - (1.0 / 9.0) * (
            (c11 + c22 - 4 * c12) ** 3
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
    def phi_d(cls, c11, c22, c12, beta):
        sqrtDet = sqrt(c11 * c22 - c12 * c12)
        c11_norm = c11 / sqrtDet
        c22_norm = c22 / sqrtDet
        c12_norm = c12 / sqrtDet

        _I1 = cls.I1(c11_norm, c22_norm, c12_norm)
        _I2 = cls.I2(c11_norm, c22_norm, c12_norm)
        _I3 = cls.I3(c11_norm, c22_norm, c12_norm)

        return beta * cls.psi1(_I1, _I2, _I3) + cls.psi2(_I1, _I2, _I3)

    @staticmethod
    def phi_v(detC, K, noise):
        return K * (detC * noise - log(detC * noise))

    @classmethod
    def polynomial_energy(cls, c11, c22, c12, beta, K, noise):
        detC = c11 * c22 - c12 * c12
        return cls.phi_d(c11, c22, c12, beta) + cls.phi_v(detC, K, noise)

    @classmethod
    def _initialize_phi(cls):
        """Compute and cache the potential function _PHI."""
        if cls._PHI is None:
            phi_symbolic = cls.polynomial_energy(
                cls._C11, cls._C22, cls._C12, cls._BETA, cls._K, cls._NOISE
            )
            cls._PHI = lambdify(cls._PHI_ARGS, phi_symbolic)

    @classmethod
    def _initialize_div_phi(cls):
        """Compute and cache the first derivatives of _PHI."""
        if cls._DIV_PHI is None:
            cls._initialize_phi()  # Ensure _PHI is initialized
            phi_symbolic = cls.polynomial_energy(
                cls._C11, cls._C22, cls._C12, cls._BETA, cls._K, cls._NOISE
            )
            first_derivatives = {
                "dPhi_dC11": diff(phi_symbolic, cls._C11),
                "dPhi_dC22": diff(phi_symbolic, cls._C22),
                "dPhi_dC12": diff(phi_symbolic, cls._C12),
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
    def numeric_conti_potential(cls):
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
    def energy_from_C_in_place(cls, C, beta=-1 / 4, K=4, noise=1, zeroReference=True):
        # Reduce using Lagrange reduction
        lagrange_reduction(C)
        return cls.energy_from_reduced_C(C, beta, K, noise, zeroReference)

    @classmethod
    def energy_from_C_components_in_place(
        cls, C11, C22, C12, beta=-1 / 4, K=4, noise=1, zeroReference=True
    ):
        C11, C22, C12 = lagrange_reduction_components(C11, C22, C12)
        return cls.energy_from_reduced_C_components(
            C11, C22, C12, beta, K, noise, zeroReference
        )

    # F is a deformation gradient tensor of shape (..., 2, 2)
    # For example, it could be a X, Y grid of 2x2 matrixes.
    @classmethod
    def energy_from_F(
        cls, F, beta=-1 / 4, K=4, noise=1, zeroReference=True, returnReducedC=False
    ):
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"

        # C = F^T F
        C = np.einsum("...ji,...jk->...ik", F, F)
        energy = cls.energy_from_C_in_place(C, beta, K, noise, zeroReference)
        if returnReducedC:
            return energy, C
        return energy


def lagrange_reduction_components(C11, C22, C12, loops=10000):
    for i in range(loops):
        mask1 = C12 < 0
        # m1 (flip) operation
        C12[mask1] *= -1

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break
        # m3 operation
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]

        if i + 1 == loops:
            print("Warning: Not enough loops")

    return C11, C22, C12


def lagrange_reduction(C, loops=10000):
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

    # Extract views (no copy)
    C11, C22, C12 = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]

    # Call original function (which modifies arrays in-place)
    lagrange_reduction_components(C11, C22, C12, loops=loops)

    # Explicitly enforce symmetry
    C[..., 1, 0] = C[..., 0, 1]

    # Return the modified input (same pointer, no new allocation)
    return C


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


if __name__ == "__main__":
    from sympy import expand, collect, latex

    def symbolic_conti_potential_terms():
        # Get the symbolic expression for the potential energy
        phi = ContiEnergy.polynomial_energy(
            ContiEnergy._C11,
            ContiEnergy._C22,
            ContiEnergy._C12,
            ContiEnergy._BETA,
            ContiEnergy._K,
            ContiEnergy._NOISE,
        )
        # Expand the expression to expose all terms involving powers of c11, c22, and c12
        expanded_phi = expand(phi)
        # Collect terms by powers of c11, c22, and c12
        collected_phi = collect(
            expanded_phi, [ContiEnergy._C11, ContiEnergy._C22, ContiEnergy._C12]
        )
        # Break down the expression into individual terms
        terms = collected_phi.as_ordered_terms()
        return terms

    def evaluate_terms_with_input(
        c11_val, c22_val, c12_val, beta_val, K_val, noise_val
    ):
        # Get the individual terms from the symbolic potential
        terms = symbolic_conti_potential_terms()
        # Substitute the provided values into each term
        substituted_terms = [
            term.subs(
                {
                    ContiEnergy._C11: c11_val,
                    ContiEnergy._C22: c22_val,
                    ContiEnergy._C12: c12_val,
                    ContiEnergy._BETA: beta_val,
                    ContiEnergy._K: K_val,
                    ContiEnergy._NOISE: noise_val,
                }
            )
            for term in terms
        ]
        return substituted_terms, terms

    # Example: Provide input values for c11, c22, c12, beta, K, and noise
    s = 10000
    c11_val = s**2
    c22_val = 1 / s**2
    c12_val = 0.0
    beta_val = -0.25
    K_val = 4
    noise_val = 1

    # Get the evaluated terms with the input values
    evaluated_terms, symbolic_terms = evaluate_terms_with_input(
        c11_val, c22_val, c12_val, beta_val, K_val, noise_val
    )

    # Print each term and its evaluated value
    for i, term in enumerate(evaluated_terms, 1):
        # Uncomment to print LaTeX representation
        # print(f"Term {i}: {latex(symbolic_terms[i-1])}")
        print(f"Term {i}: {term}")

from sympy import symbols, diff, sqrt, log, lambdify, ccode, cse, simplify
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
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"

        # C = F^T F
        C = np.einsum("...ji,...jk->...ik", F, F)

        if accuracy < 1:
            if loops is None:
                loops = 200

        elif loops is None:
            loops = 1000

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
        sigma = np.array([[dPhi_dC11, dPhi_dC12 / 2], [dPhi_dC12 / 2, dPhi_dC22]])
        return sigma

    @classmethod
    def P_from_F(cls, F, M=None, beta=-1 / 4, K=4, noise=1):
        """
        Compute the first Piola-Kirchhoff stress tensor P from the deformation gradient F.
        """
        assert F.shape[-2:] == (2, 2), "F must have shape (..., 2, 2)"
        C = np.einsum("...ji,...jk->...ik", F, F)

        M_ = lagrange_reduction(C, returnM=M is None)
        if M is None:
            M = M_
        C_R = C

        sigma = cls.sigma_from_C_R(C_R, beta, K, noise)

        P = 2 * F @ M @ sigma @ M.T
        return P


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
        return (C11 - 1) ** 2 + (C22 - 1) ** 2 + C12**2


def lagrange_reduction_components(C11, C22, C12, loops=1000, returnMs=False):
    # If reurnM is True, we create an array of numbers from 1 to 3 where each number
    # corresponds to the m1, m2 or m3 operation that is applied to the C matrix
    if returnMs:
        ms = np.empty_like(C11, dtype=object)
        # Initialize each element with its own empty list
        it = np.nditer(C11, flags=["multi_index"])
        while not it.finished:
            ms[it.multi_index] = []
            it.iternext()

    for i in range(loops):
        mask1 = C12 < 0
        # m1 (flip) operation
        C12[mask1] *= -1
        if returnMs:
            indices = np.where(mask1)
            for idx in zip(*indices):
                ms[idx].append(1)

        mask2 = C22 < C11
        # m2 (swap) operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()
        if returnMs:
            indices = np.where(mask2)
            for idx in zip(*indices):
                ms[idx].append(2)

        mask3 = 2 * C12 > C11
        # Stop the loop if no changes are made
        if not np.any(mask1 | mask2 | mask3):
            break

        # m3 operation
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]
        if returnMs:
            indices = np.where(mask3)
            for idx in zip(*indices):
                ms[idx].append(3)

        if i + 1 == loops and loops > 200:
            print("Warning: Not enough loops")
    # Modifies in place
    # return C11, C22, C12
    if returnMs:
        return ms


def lagrange_reduction(C, loops=1000, returnM=False):
    assert C.shape[-2:] == (2, 2), "C must have shape (..., 2, 2)"

    # Extract views (no copy)
    C11, C22, C12 = C[..., 0, 0], C[..., 1, 1], C[..., 0, 1]

    # Call original function (which modifies arrays in-place)
    ms = lagrange_reduction_components(C11, C22, C12, loops=loops, returnMs=returnM)

    # Explicitly enforce symmetry
    C[..., 1, 0] = C[..., 0, 1]

    if returnM:
        # Warning: This M calculation is not tested and probably donesn't work yet
        # Now we need to construct the matrix of M matrices
        # M should be like C, but where each 2x2 matrix is the identity matrix
        M = np.zeros_like(C)
        M[..., 0, 0] = 1
        M[..., 1, 1] = 1

        # It is slow, but we will contruct the M matrices one at a time
        for i in range(len(ms)):
            for m in ms[i]:
                if m == 1:
                    lag_m1(M[i])
                elif m == 2:
                    lag_m2(M[i])
                elif m == 3:
                    lag_m3(M[i], n=1)
        return M


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


def flip(matrix, row, col):
    matrix[row, col] *= -1


def swap(matrix, row1, col1, row2, col2):
    temp = matrix[row1, col1]
    matrix[row1, col1] = matrix[row2, col2]
    matrix[row2, col2] = temp


def swap_cols(matrix):
    matrix[:, [0, 1]] = matrix[:, [1, 0]]


def lag_m1(matrix):
    flip(matrix, 0, 1)
    flip(matrix, 1, 1)


def lag_m2(matrix):
    swap_cols(matrix)


# applies m3 n times
def lag_m3(matrix, n=1):
    # https://www.wolframalpha.com/input?i=%7B%7B1%2C+-1%7D%2C+%7B0%2C+1%7D%7D%5En
    multiplier_matrix = np.array([[1, -n], [0, 1]])

    new_matrix = matrix @ multiplier_matrix
    np.copyto(matrix, new_matrix)


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


if __name__ == "__main__":
    # Get symbolic expressions from ContiEnergy
    phi_func, div_phi_dict, div_div_phi_dict = ContiEnergy.symbolic_potential()

    # Choose whether to include second derivatives
    include_second_derivatives = False  # Set to True when needed

    # Generate the code
    energy_code, stress_code = compute_energy_and_derivatives(
        phi_func, div_phi_dict, div_div_phi_dict, include_second_derivatives
    )

    # Output results
    print("Energy function:\n", energy_code)
    print("\n")

    print("Stress function:\n", stress_code)

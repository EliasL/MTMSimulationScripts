from sympy import symbols, diff, sqrt, log, ccode, cse, simplify, lambdify

# Define symbols
c11, c22, c12, beta, K, noise = symbols("c11 c22 c12 beta K noise")


def I1(c11, c22, c12):
    return (1.0 / 3.0) * (c11 + c22 - c12)


def I2(c11, c22, c12):
    return (1.0 / 4.0) * ((c11 - c22) ** 2) + (1.0 / 12.0) * (
        (c11 + c22 - 4 * c12) ** 2
    )


def I3(c11, c22, c12):
    return ((c11 - c22) ** 2) * (c11 + c22 - 4 * c12) - (1.0 / 9.0) * (
        (c11 + c22 - 4 * c12) ** 3
    )


def psi1(I1, I2, I3):
    return (
        (I1**4 * I2)
        - (41.0 * I2**3 / 99.0)
        + (7 * I1 * I2 * I3 / 66.0)
        + (I3**2 / 1056.0)
    )


def psi2(I1, I2, I3):
    return (
        (4.0 * I2**3 / 11.0)
        + (I1**3 * I3)
        - (8.0 * I1 * I2 * I3 / 11.0)
        + (17.0 * I3**2 / 528.0)
    )


def phi_d(c11, c22, c12, beta):
    sqrtDet = sqrt(c11 * c22 - c12 * c12)
    c11_norm = c11 / sqrtDet
    c22_norm = c22 / sqrtDet
    c12_norm = c12 / sqrtDet

    _I1 = I1(c11_norm, c22_norm, c12_norm)
    _I2 = I2(c11_norm, c22_norm, c12_norm)
    _I3 = I3(c11_norm, c22_norm, c12_norm)

    return beta * psi1(_I1, _I2, _I3) + psi2(_I1, _I2, _I3)


def phi_v(detC, K, noise):
    return K * (detC - log(detC)) * noise


def polynomialEnergy(c11, c22, c12, beta, K, noise):
    detC = c11 * c22 - c12 * c12
    return phi_d(c11, c22, c12, beta) + phi_v(detC, K, noise)


def compute_derivatives():
    # Compute first derivatives
    derivatives = {
        "dPhi_dC11": diff(polynomialEnergy(c11, c22, c12, beta, K, noise), c11),
        "dPhi_dC22": diff(polynomialEnergy(c11, c22, c12, beta, K, noise), c22),
        "dPhi_dC12": diff(polynomialEnergy(c11, c22, c12, beta, K, noise), c12),
    }
    return derivatives


def compute_second_derivatives(derivatives):
    # Compute second derivatives using the first derivatives passed in
    second_derivatives = {
        "dPhi_dC11_dC11": diff(derivatives["dPhi_dC11"], c11),
        "dPhi_dC22_dC22": diff(derivatives["dPhi_dC22"], c22),
        "dPhi_dC12_dC12": diff(derivatives["dPhi_dC12"], c12),
        "dPhi_dC11_dC22": diff(derivatives["dPhi_dC11"], c22),
        "dPhi_dC11_dC12": diff(derivatives["dPhi_dC11"], c12),
        "dPhi_dC22_dC12": diff(derivatives["dPhi_dC22"], c12),
    }
    return second_derivatives


def generate_cpp_code():
    derivatives = compute_derivatives()
    second_derivatives = compute_second_derivatives(derivatives)

    # First + Second Derivatives
    all_exprs = list(derivatives.values()) + list(second_derivatives.values())
    replacements_all, reduced_forKlas_all = cse(all_exprs)
    simplified_forKlas_all = [simplify(expr) for expr in reduced_forKlas_all]

    # First Derivatives Only
    first_exprs = list(derivatives.values())
    replacements_first, reduced_forKlas_first = cse(first_exprs)
    simplified_forKlas_first = [simplify(expr) for expr in reduced_forKlas_first]

    # Generate C++ code for all derivatives
    ccode_replacements_all = [
        f"double {var} = {ccode(expr)};" for var, expr in replacements_all
    ]
    ccode_all_derivatives = [
        f"double {key} = {ccode(simplified_forKlas_all[i])};"
        for i, key in enumerate(
            list(derivatives.keys()) + list(second_derivatives.keys())
        )
    ]

    # Generate C++ code for first derivatives only
    ccode_replacements_first = [
        f"double {var} = {ccode(expr)};" for var, expr in replacements_first
    ]
    ccode_first_derivatives = [
        f"double {key} = {ccode(simplified_forKlas_first[i])};"
        for i, key in enumerate(derivatives.keys())
    ]

    # Separate strings for first and all derivatives
    ccode_combined_all = ccode_replacements_all + [""] + ccode_all_derivatives
    ccode_combined_first = ccode_replacements_first + [""] + ccode_first_derivatives

    return "\n".join(ccode_combined_first), "\n".join(ccode_combined_all)


# Print the generated C++ code for both sets
ccode_first_only, ccode_all = generate_cpp_code()
print("First Derivatives Only:\n", ccode_first_only)
print("\nFirst and Second Derivatives:\n", ccode_all)


def compute_numeric_derivatives(c11_val, c22_val, c12_val, beta_val, K_val):
    # Define the polynomial energy using symbolic functions
    phi = polynomialEnergy(c11, c22, c12, beta, K)

    # Define symbolic derivatives
    first_derivatives = {
        "dPhi_dC11": diff(phi, c11),
        "dPhi_dC22": diff(phi, c22),
        "dPhi_dC12": diff(phi, c12),
    }
    second_derivatives = {
        "dPhi_dC11_dC11": diff(first_derivatives["dPhi_dC11"], c11),
        "dPhi_dC22_dC22": diff(first_derivatives["dPhi_dC22"], c22),
        "dPhi_dC12_dC12": diff(first_derivatives["dPhi_dC12"], c12),
        "dPhi_dC11_dC22": diff(first_derivatives["dPhi_dC11"], c22),
        "dPhi_dC11_dC12": diff(first_derivatives["dPhi_dC11"], c12),
        "dPhi_dC22_dC12": diff(first_derivatives["dPhi_dC22"], c12),
    }

    # Create lambdified functions to compute numeric values
    eval_first_derivatives = {
        key: lambdify((c11, c22, c12, beta, K), expr)
        for key, expr in first_derivatives.items()
    }
    eval_second_derivatives = {
        key: lambdify((c11, c22, c12, beta, K), expr)
        for key, expr in second_derivatives.items()
    }

    # Compute numeric values for the given inputs
    numeric_first_derivatives = {
        key: func(c11_val, c22_val, c12_val, beta_val, K_val)
        for key, func in eval_first_derivatives.items()
    }
    numeric_second_derivatives = {
        key: func(c11_val, c22_val, c12_val, beta_val, K_val)
        for key, func in eval_second_derivatives.items()
    }

    return numeric_first_derivatives, numeric_second_derivatives


# Example usage:
# first_derivs, second_derivs = compute_numeric_derivatives(1.0, 1.0, 0.0, 0.3, 0.05)
# print("First Derivatives:", first_derivs)
# print("Second Derivatives:", second_derivs)

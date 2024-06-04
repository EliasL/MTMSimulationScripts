from sympy import symbols, diff, sqrt, log, ccode, cse, simplify

# Define symbols
c11, c22, c12, beta, mu = symbols("c11 c22 c12 beta mu")


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


def phi_v(detC, mu):
    return mu * (detC - log(detC))


def polynomialEnergy(c11, c22, c12, beta, mu):
    detC = c11 * c22 - c12 * c12
    return phi_d(c11, c22, c12, beta) + phi_v(detC, mu)


# Calculate derivatives again with the corrected imports
derivative_c11 = diff(polynomialEnergy(c11, c22, c12, beta, mu), c11)
derivative_c22 = diff(polynomialEnergy(c11, c22, c12, beta, mu), c22)
derivative_c12 = diff(polynomialEnergy(c11, c22, c12, beta, mu), c12)


# Apply Common Subexpression Elimination (CSE)
replacements, reduced_formulas = cse([derivative_c11, derivative_c22, derivative_c12])

# Apply simplification on the reduced formulas
simplified_formulas = [simplify(expr) for expr in reduced_formulas]

# Generate C++ code for the common subexpressions (replacements
ccode_replacements = [f"double {var} = {ccode(expr)};" for var, expr in replacements]
ccode_derivaties = [
    f"double var{i} = {ccode(expr)};" for i, expr in enumerate(simplified_formulas)
]

# Combine the C++ code strings for replacements and derivatives for display
ccode_combined = ccode_replacements + [""] + ccode_derivaties

print("\n".join(ccode_combined))

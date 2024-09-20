from sympy import symbols, diff, sqrt, log, lambdify

# Define symbols
c11, c22, c12, beta, K, noise = symbols("c11 c22 c12 beta K noise")
phiArgs = (c11, c22, c12, beta, K, noise)


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
    return K * (detC * noise - log(detC * noise))


def polynomialEnergy(c11, c22, c12, beta, K, noise):
    detC = c11 * c22 - c12 * c12
    return phi_d(c11, c22, c12, beta) + phi_v(detC, K, noise)


def compute_derivatives(f):
    # Compute first derivatives
    derivatives = {
        "dPhi_dC11": diff(f, c11),
        "dPhi_dC22": diff(f, c22),
        "dPhi_dC12": diff(f, c12),
    }
    return derivatives


def compute_second_derivatives(first_derivatives):
    # Compute second derivatives using the first derivatives passed in
    second_derivatives = {
        "dPhi_dC11_dC11": diff(first_derivatives["dPhi_dC11"], c11),
        "dPhi_dC22_dC22": diff(first_derivatives["dPhi_dC22"], c22),
        "dPhi_dC12_dC12": diff(first_derivatives["dPhi_dC12"], c12),
        "dPhi_dC11_dC22": diff(first_derivatives["dPhi_dC11"], c22),
        "dPhi_dC11_dC12": diff(first_derivatives["dPhi_dC11"], c12),
        "dPhi_dC22_dC12": diff(first_derivatives["dPhi_dC22"], c12),
    }
    return second_derivatives


# Use this function to get the expression for
def symbolicContiPotential():
    return polynomialEnergy(*phiArgs)


def numericContiPotential(compute_derivative=False, compute_second_derivative=False):
    # Define the potential (phi) symbolically
    phi = symbolicContiPotential()  # Assumed to be defined elsewhere

    # Initialize default return values
    fast_first_derivatives = None
    fast_second_derivatives = None

    # Compute first derivatives if requested
    if compute_derivative:
        first_derivatives = compute_derivatives(phi)
        # Create lambdified functions to compute numeric values
        fast_first_derivatives = {
            key: lambdify(phiArgs, expr) for key, expr in first_derivatives.items()
        }

    # Compute second derivatives if requested
    if compute_second_derivative:
        if not compute_derivative:
            # Compute first derivatives if second derivatives are required but not first derivatives
            first_derivatives = compute_derivatives(phi)

        second_derivatives = compute_second_derivatives(first_derivatives)
        # Lambdify the second derivatives
        fast_second_derivatives = {
            key: lambdify(phiArgs, expr) for key, expr in second_derivatives.items()
        }

    # Return the lambdified potential and optionally its derivatives
    return (
        lambdify(phiArgs, phi),  # Lambdified potential
        fast_first_derivatives,  # First derivatives (if any)
        fast_second_derivatives,  # Second derivatives (if any)
    )

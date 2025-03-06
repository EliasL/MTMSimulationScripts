from sympy import symbols, diff, sqrt, log, lambdify


class ContiEnergy:
    # Define symbols as class variables
    c11, c22, c12, beta, K, noise = symbols("c11 c22 c12 beta K noise")
    phi_args = (c11, c22, c12, beta, K, noise)

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
    def log_phi_v(detC, K, noise):
        return K * (detC * noise - log(detC * noise))

    @staticmethod
    def phi_v(detC, K, noise):
        vol = detC * noise
        return K * (vol - 0.02 * (vol - 1) / vol)

    @classmethod
    def polynomial_energy(cls, c11, c22, c12, beta, K, noise):
        detC = c11 * c22 - c12 * c12
        return cls.phi_d(c11, c22, c12, beta) + cls.phi_v(detC, K, noise)

    @classmethod
    def compute_derivatives(cls, f):
        # Compute first derivatives
        derivatives = {
            "dPhi_dC11": diff(f, cls.c11),
            "dPhi_dC22": diff(f, cls.c22),
            "dPhi_dC12": diff(f, cls.c12),
        }
        return derivatives

    @staticmethod
    def compute_second_derivatives(first_derivatives, c11, c22, c12):
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

    @classmethod
    def symbolic_conti_potential(cls):
        return cls.polynomial_energy(
            cls.c11, cls.c22, cls.c12, cls.beta, cls.K, cls.noise
        )

    @classmethod
    def numeric_conti_potential(
        cls, compute_derivative=False, compute_second_derivative=False
    ):
        # Define the potential (phi) symbolically
        phi = cls.symbolic_conti_potential()

        # Initialize default return values
        fast_first_derivatives = None
        fast_second_derivatives = None

        # Compute first derivatives if requested
        if compute_derivative:
            first_derivatives = cls.compute_derivatives(phi)
            # Create lambdified functions to compute numeric values
            fast_first_derivatives = {
                key: lambdify(cls.phi_args, expr)
                for key, expr in first_derivatives.items()
            }

        # Compute second derivatives if requested
        if compute_second_derivative:
            if not compute_derivative:
                # Compute first derivatives if second derivatives are required but not first derivatives
                first_derivatives = cls.compute_derivatives(phi)

            second_derivatives = cls.compute_second_derivatives(
                first_derivatives, cls.c11, cls.c22, cls.c12
            )
            # Lambdify the second derivatives
            fast_second_derivatives = {
                key: lambdify(cls.phi_args, expr)
                for key, expr in second_derivatives.items()
            }

        # Return the lambdified potential and optionally its derivatives
        return (
            lambdify(cls.phi_args, phi),  # Lambdified potential
            fast_first_derivatives,  # First derivatives (if any)
            fast_second_derivatives,  # Second derivatives (if any)
        )

    @classmethod
    def ground_state_energy(cls, beta=-1 / 4, K=4):
        # Load the potential and its derivatives
        phi, _, _ = cls.numeric_conti_potential()
        ground_state_energy = phi(1, 1, 0, beta, K, 1)
        return ground_state_energy


if __name__ == "__main__":
    from sympy import expand, collect, latex

    def symbolic_conti_potential_terms():
        # Get the symbolic expression for the potential energy
        phi = ContiEnergy.polynomial_energy(
            ContiEnergy.c11,
            ContiEnergy.c22,
            ContiEnergy.c12,
            ContiEnergy.beta,
            ContiEnergy.K,
            ContiEnergy.noise,
        )
        # Expand the expression to expose all terms involving powers of c11, c22, and c12
        expanded_phi = expand(phi)
        # Collect terms by powers of c11, c22, and c12
        collected_phi = collect(
            expanded_phi, [ContiEnergy.c11, ContiEnergy.c22, ContiEnergy.c12]
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
                    ContiEnergy.c11: c11_val,
                    ContiEnergy.c22: c22_val,
                    ContiEnergy.c12: c12_val,
                    ContiEnergy.beta: beta_val,
                    ContiEnergy.K: K_val,
                    ContiEnergy.noise: noise_val,
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

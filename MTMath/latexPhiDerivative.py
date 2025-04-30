import sympy as sp
import re
from contiPotential import ContiEnergy, SuperSimple


class SymbolicFEM:
    """Class for handling symbolic finite element computations"""

    def __init__(self):
        # Define all symbols
        self.symbols = self.define_symbols()
        # Initialize attributes that will hold intermediate results
        self.shape_functions = None
        self.X_expr = None
        self.x_expr = None
        self.derivatives = None
        self.F = None
        self.F_d = None
        self.F_factored = None
        self.C = None
        self.C11 = None
        self.C12 = None
        self.C22 = None
        self.C11_clean = None
        self.C12_clean = None
        self.C22_clean = None
        self.phi = None
        self.div_phi = None
        self.div_phi_d = None
        self.div_phi_constrained = []
        self.div_div_phi = None
        self.assumptions = None

    def define_symbols(self):
        """Define all symbolic variables used in the analysis."""
        # Natural coordinates
        xi, eta = sp.symbols("xi eta", real=True)

        # Nodal coordinates for initial configuration
        X1, Y1, X2, Y2, X3, Y3 = sp.symbols("X1 Y1 X2 Y2 X3 Y3", real=True)

        # Nodal coordinates for current configuration
        x1, y1, x2, y2, x3, y3 = sp.symbols("x1 y1 x2 y2 x3 y3", real=True)

        # Vector components
        V1x, V1y, V2x, V2y = sp.symbols("V^1_x V^1_y V^2_x V^2_y", real=True)
        v1x, v1y, v2x, v2y = sp.symbols("v^1_x v^1_y v^2_x v^2_y", real=True)

        return {
            "natural": (xi, eta),
            "initial": (X1, Y1, X2, Y2, X3, Y3),
            "current": (x1, y1, x2, y2, x3, y3),
            "vectors": (V1x, V1y, V2x, V2y, v1x, v1y, v2x, v2y),
        }

    def define_shape_functions(self):
        """Define the linear triangular shape functions."""
        xi, eta = self.symbols["natural"]
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        self.shape_functions = (N1, N2, N3)

    def map_coordinates(self):
        """Map natural coordinates to physical configurations."""
        if self.shape_functions is None:
            self.define_shape_functions()

        N1, N2, N3 = self.shape_functions
        X1, Y1, X2, Y2, X3, Y3 = self.symbols["initial"]
        x1, y1, x2, y2, x3, y3 = self.symbols["current"]

        # Initial configuration mapping
        self.X_expr = (
            N1 * sp.Matrix([X1, Y1])
            + N2 * sp.Matrix([X2, Y2])
            + N3 * sp.Matrix([X3, Y3])
        )

        # Current configuration mapping
        self.x_expr = (
            N1 * sp.Matrix([x1, y1])
            + N2 * sp.Matrix([x2, y2])
            + N3 * sp.Matrix([x3, y3])
        )

    def compute_derivatives(self):
        """Compute derivatives with respect to natural coordinates."""
        if self.X_expr is None or self.x_expr is None:
            self.map_coordinates()

        X_expr = self.X_expr
        x_expr = self.x_expr
        xi, eta = self.symbols["natural"]

        N1, N2, N3 = self.shape_functions

        # Compute the derivatives with respect to the natural coordinates
        dN_dxi = sp.Matrix([sp.diff(N, xi) for N in self.shape_functions])
        dN_deta = sp.Matrix([sp.diff(N, eta) for N in self.shape_functions])
        # Form the 3x2 matrix: each row corresponds to a shape function, columns to xi and eta derivatives
        dN_dxi_eta = sp.Matrix.hstack(dN_dxi, dN_deta)

        # Individual derivatives
        dX_dxi = sp.diff(X_expr, xi)
        dX_deta = sp.diff(X_expr, eta)
        dx_dxi = sp.diff(x_expr, xi)
        dx_deta = sp.diff(x_expr, eta)

        # Complete Jacobian matrices
        JX = X_expr.jacobian([xi, eta])
        Jx = x_expr.jacobian([xi, eta])

        # Use the chain rule: dN/dX = dN/dxi_eta * (JX)^(-1)
        JX_inv = JX.inv()
        dN_dX = dN_dxi_eta * JX_inv

        self.derivatives = {
            "dX_dxi": dX_dxi,
            "dX_deta": dX_deta,
            "dx_dxi": dx_dxi,
            "dx_deta": dx_deta,
            "JX": JX,
            "Jx": Jx,
            "dN_dxi_eta": dN_dxi_eta,
            "dN_dX": dN_dX,
        }

        return self.derivatives

    def compute_deformation_gradient(self):
        """Compute the deformation gradient F = Jx * JX^{-1}."""
        if self.derivatives is None:
            self.compute_derivatives()

        JX = self.derivatives["JX"]
        Jx = self.derivatives["Jx"]
        self.F = sp.simplify(Jx @ JX.inv())

    def factor_common_denominator(self, matrix):
        """Factor out common denominator from a matrix."""
        # Get the denominator and numerator of each element
        numerators = []
        denominators = []
        for element in matrix:
            num, den = sp.fraction(sp.together(element))
            numerators.append(num)
            denominators.append(den)

        # Compute the common denominator
        common_d = denominators[0]
        for d in denominators[1:]:
            common_d = sp.gcd(common_d, d)

        # Create a new matrix with the common denominator factored out
        new_matrix = matrix.applyfunc(lambda x: sp.cancel(x @ common_d))

        return common_d, sp.simplify(new_matrix)

    def compute_cauchy_green(self):
        """Compute the Cauchy-Green deformation tensor C = F^T * F."""
        if self.F is None:
            self.compute_deformation_gradient()

        F = self.F
        self.C = F.T @ F

        # Extract components
        self.C11 = sp.simplify(self.C[0, 0])
        self.C12 = sp.simplify(self.C[0, 1])
        self.C22 = sp.simplify(self.C[1, 1])

    def substitute_vector_components(self):
        """Replace coordinate differences with vector components."""
        if self.C12 is None:
            self.compute_cauchy_green()

        X1, Y1, X2, Y2, X3, Y3 = self.symbols["initial"]
        x1, y1, x2, y2, x3, y3 = self.symbols["current"]
        V1x, V1y, V2x, V2y, v1x, v1y, v2x, v2y = self.symbols["vectors"]

        subs_dict = {
            X1 - X2: V1x,
            Y1 - Y2: V1y,
            X1 - X3: V2x,
            Y1 - Y3: V2y,
            x1 - x2: v1x,
            y1 - y2: v1y,
            x1 - x3: v2x,
            y1 - y3: v2y,
        }

        self.C11_clean = sp.simplify(self.C11.subs(subs_dict))
        self.C12_clean = sp.simplify(self.C12.subs(subs_dict))
        self.C22_clean = sp.simplify(self.C22.subs(subs_dict))

    def evaluate_conti_energy(self):
        """Evaluate Conti energy for the deformation."""
        if self.C11_clean is None:
            self.substitute_vector_components()

        # Get the Conti energy expressions
        self.phi, self.div_phi, self.div_div_phi = SuperSimple.symbolic_potential()

        # Define assumptions using computed C11, C12, C22 values

        C11, C12, C22, noise, beta, K = sp.symbols("C_{11} C_{12} C_{22} noise beta K")

        # This is just a placeholder for demonstration
        # In a real implementation, you might want to use the actual C values
        self.assumptions = {C11: 1, C22: 1 + C12, noise: 1, beta: -1 / 4, K: 4}

    @staticmethod
    def round_floats(expr):
        """Round all float coefficients in an expression to one decimal and remove trailing .0 if applicable."""

        def replacement(x):
            r = round(float(x), 1)
            if r.is_integer():
                return sp.Integer(int(r))
            else:
                return sp.Float(r)

        return expr.replace(lambda x: isinstance(x, sp.Float), replacement)

    @staticmethod
    def format_latex_expression(expr, eq_label=None):
        """Format expression for LaTeX with aligned equations and line breaks."""
        # Round numerical coefficients
        expr_rounded = SymbolicFEM.round_floats(expr)

        # Convert to LaTeX
        latex_expr = sp.latex(expr_rounded)

        # Clean up LaTeX
        # latex_expr = latex_expr.replace(r"\left", "").replace(r"\right", "")
        # latex_expr = latex_expr.replace("=", "&=")

        # # Add line breaks at + and - signs (except in exponents)
        # newString = []
        # for i in range(len(latex_expr)):
        #     if latex_expr[i] in ["+", "-"]:
        #         # Check if we're in an exponent context
        #         if i > 2 and latex_expr[i - 2 : i] != "^{" and latex_expr[i - 2] != "a":
        #             newString.append("\\\\\n&")
        #     newString.append(latex_expr[i])

        # formatted_expr = "".join(newString)

        # # Add label if provided
        if eq_label:
            return f"{eq_label} &= {latex_expr}"
        else:
            return latex_expr

    def generate_equation(self, variable, expr, comment=""):
        """Generate a LaTeX equation environment string."""
        eq_str = ""
        if comment:
            eq_str += comment + "\n"
        eq_str += r"\begin{equation}" + "\n"
        eq_str += variable + " = " + sp.latex(expr) + "\n"
        eq_str += r"\end{equation}" + "\n\n"
        return eq_str

    def generate_latex_fem_document(self):
        """Generate LaTeX equations for FEM quantities."""
        # Ensure necessary computations have been done
        if self.F_factored is None or self.C12_clean is None:
            self.F_d, self.F_factored = self.factor_common_denominator(self.F)
            self.substitute_vector_components()

        N1, N2, N3 = self.shape_functions
        # Generate equations for the shape functions
        eq_N1 = self.generate_equation(r"N_1", N1, "Shape functions")
        eq_N2 = self.generate_equation(r"N_2", N2)
        eq_N3 = self.generate_equation(r"N_3", N3)

        # Generate equations for the mappings
        eq_X = self.generate_equation(
            r"\X(\xi,\eta)",
            self.X_expr,
            "Mapping from natural coordinates to the initial configuration",
        )
        eq_x = self.generate_equation(
            r"\x(\xi,\eta)",
            self.x_expr,
            "Mapping from natural coordinates to the current configuration",
        )

        # Generate equations for the individual derivatives
        eq_dX_dxi = self.generate_equation(
            r"\frac{\partial \X}{\partial \xi}",
            self.derivatives["dX_dxi"],
            r"Derivative of $\X$ with respect to $\xi$",
        )
        eq_dX_deta = self.generate_equation(
            r"\frac{\partial \X}{\partial \eta}",
            self.derivatives["dX_deta"],
            r"Derivative of $\X$ with respect to $\eta$",
        )
        eq_dx_dxi = self.generate_equation(
            r"\frac{\partial \x}{\partial \xi}",
            self.derivatives["dx_dxi"],
            r"Derivative of $\x$ with respect to $\xi$",
        )
        eq_dx_deta = self.generate_equation(
            r"\frac{\partial \x}{\partial \eta}",
            self.derivatives["dx_deta"],
            r"Derivative of $\x$ with respect to $\eta$",
        )

        # Generate intermediate equations for Jacobians
        eq_JX = self.generate_equation(
            r"J_\X",
            sp.Matrix.hstack(self.derivatives["dX_dxi"], self.derivatives["dX_deta"]),
            r"$J_\X = \left[\frac{\partial \X}{\partial \xi}, \frac{\partial \X}{\partial \eta}\right]$",
        )
        eq_Jx = self.generate_equation(
            r"J_\x",
            sp.Matrix.hstack(self.derivatives["dx_dxi"], self.derivatives["dx_deta"]),
            r"$J_\x = \left[\frac{\partial \x}{\partial \xi}, \frac{\partial \x}{\partial \eta}\right]$",
        )

        # Generate equation for common denominator
        eq_d = self.generate_equation(r"d", self.F_d, "")

        # Shape function derivatives
        eq_dN_dxi = self.generate_equation(
            r"\frac{\partial N}{\partial \boldsymbol{\xi}}",
            sp.simplify(self.derivatives["dN_dxi_eta"]),
            r"Derivative of $N$ with respect to $\boldsymbol{\xi}$",
        )
        eq_dN_dX = self.generate_equation(
            r"\frac{\partial N}{\partial X}",
            sp.simplify(self.derivatives["dN_dX"] @ self.F_d),
            r"Derivative of $N$ with respect to $X$",
        )

        # Generate equation for the deformation gradient
        eq_F = self.generate_equation(
            r"F",
            self.F_factored,
            r"Deformation gradient computed as $F = \frac{\partial \x}{\partial \X} = J_\x J_\X^{-1}$",
        )

        # Generate equations for Cauchy-Green components
        eq_C12 = self.generate_equation(
            r"C_{12}",
            self.C12 @ self.F_d**2 * -1,
            r"Off-diagonal component of the Cauchy-Green deformation tensor",
        )
        eq2_C12 = self.generate_equation(
            r"C_{12}",
            self.C12_clean @ self.F_d**2 * -1,
            r"Condensed form of the off-diagonal component",
        )

        # Combine all equations to form the document's body
        document_body = (
            eq_N1
            + eq_N2
            + eq_N3
            + eq_X
            + eq_x
            + eq_dX_dxi
            + eq_dX_deta
            + eq_dx_dxi
            + eq_dx_deta
            + eq_JX
            + eq_Jx
            + eq_d
            + eq_dN_dxi
            + eq_dN_dX
            + eq_F
            + eq_C12
            + eq2_C12
        )

        return document_body

    def generate_latex_conti_energy(self):
        """Generate multiline LaTeX for Conti energy derivatives."""
        if self.div_phi is None:
            self.evaluate_conti_energy()

        latex_output = []
        self.div_phi_constrained = [None for _ in range(len(self.div_phi))]
        # Process each derivative
        for i, (eq_label, expr) in enumerate(self.div_phi.items()):
            # Substitute assumptions and simplify
            self.div_phi_constrained[i] = sp.simplify(expr.subs(self.assumptions))
            # Substitute C12 with C12_clean (Takes too long time. Super long expression)
            # self.expr_assumed_full = sp.simplify(
            #     self.expr_assumed.subs(sp.symbols("C12"), self.C12_clean)
            # )

            # Map derivative labels to LaTeX notation
            if eq_label == "dPhi_dC11":
                label_latex = r"\frac{\partial \Phi}{\partial C_{11}}"
            elif eq_label == "dPhi_dC22":
                label_latex = r"\frac{\partial \Phi}{\partial C_{22}}"
            elif eq_label == "dPhi_dC12":
                label_latex = r"\frac{\partial \Phi}{\partial C_{12}}"
            else:
                label_latex = eq_label

            # Format the expression
            formatted_expr = self.format_latex_expression(
                self.div_phi_constrained[i], label_latex
            )
            # formatted_expr = sp.latex(self.expr_assumed)

            # Create multiline LaTeX
            multiline_latex = f"\\begin{{align*}}\n{formatted_expr}\n\\end{{align*}}"

            latex_output.append(multiline_latex)

        # sigma = 1/2 (∂Φ/∂C_R + (∂Φ/∂C_R)^T)
        sigma = sp.Matrix(
            [
                [self.div_phi_constrained[0], self.div_phi_constrained[2] / 2],
                [self.div_phi_constrained[2] / 2, self.div_phi_constrained[1]],
            ]
        )
        self.div_phi_d, factored_sigma = self.factor_common_denominator(sigma)

        # Make equation for div_phi_d
        latex_sigma = self.format_latex_expression(self.div_phi_d, r"\sigma_d")
        multiline_latex_sigma = f"\\begin{{align*}}\n{latex_sigma}\n\\end{{align*}}"
        latex_output.append(multiline_latex_sigma)

        # We assume M is identity, so we ignore it
        # P = 2 * F @ M @ sigma @ M.T
        # We then assume F is {(1,C12),(0, 1)}
        C12 = sp.symbols("C_{12}")
        F = sp.Matrix([[1, C12], [0, 1]])
        self.P = 2 * F @ factored_sigma
        self.P = sp.simplify(self.P)
        # Generate LaTeX for P
        P_latex = self.format_latex_expression(self.P, r"P")
        multiline_latex_P = f"\\begin{{align*}}\n{P_latex}\n\\end{{align*}}"
        latex_output.append(multiline_latex_P)

        # Substitute C12 with C12_clean (we do some tricks to help it simplify)
        print("substituting C12...")
        d = sp.symbols("d")
        self.full_P = self.P.subs(
            C12, 1 / (d**2) * sp.simplify(self.C12_clean @ self.F_d**2)
        )
        # self.full_P = self.P.subs(C12, self.C12_clean)
        # print("simplifying...")
        # self.full_P = sp.simplify(self.full_P)
        # P_full_latex = self.format_latex_expression(self.full_P, r"P")
        # multiline_latex_P_full = f"\\begin{{align*}}\n{P_full_latex}\n\\end{{align*}}"
        # latex_output.append(multiline_latex_P_full)

        V1x, V1y, V2x, V2y, v1x, v1y, v2x, v2y = self.symbols["vectors"]
        # Only let the current position of the first vector vary
        # The rest are fixed to (0,1)
        # And determinant =1
        self.full_P_constrained = self.full_P.subs(
            {
                V1x: 1,
                V1y: 0,
                V2x: 0,
                V2y: 1,
                v1x: 1,
                v1y: 0,
                d: 1,
            }
        )
        print("simplifying...")
        self.full_P_constrained = sp.simplify(self.full_P_constrained)
        P_full_latex_constrained = self.format_latex_expression(
            self.full_P_constrained, r"P"
        )
        multiline_latex_P_full_constrained = (
            f"\\begin{{align*}}\n{P_full_latex_constrained}\n\\end{{align*}}"
        )
        latex_output.append(multiline_latex_P_full_constrained)

        # Finally, we set v1y=0.2
        # and v1x=1

        self.full_P_constrained = self.full_P_constrained.subs(
            {
                v2x: 0.2,
            }
        )
        print("simplifying...")
        self.full_P_constrained = sp.simplify(self.full_P_constrained)
        P_full_latex_constrained = sp.latex(self.full_P_constrained)
        multiline_latex_P_full_constrained_full = (
            f"\\begin{{align*}}\n{P_full_latex_constrained}\n\\end{{align*}}"
        )
        latex_output.append(multiline_latex_P_full_constrained_full)

        return "\n\n".join(latex_output)

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.define_shape_functions()
        self.map_coordinates()
        self.compute_derivatives()
        self.compute_deformation_gradient()
        self.compute_cauchy_green()
        self.substitute_vector_components()

        # Conti energy analysis
        self.evaluate_conti_renergy()

        # Generate LaTeX output
        fem_latex = self.generate_latex_fem_document()
        energy_latex = self.generate_latex_conti_energy()

        return {
            "fem_latex": fem_latex,
            "energy_latex": energy_latex,
        }


def main():
    """Main function to execute the full computation sequence."""
    analyzer = SymbolicFEM()
    results = analyzer.run_analysis()

    # Print LaTeX outputs
    print("FEM ANALYSIS:")
    print(results["fem_latex"])
    print("\nCONTI ENERGY ANALYSIS:")
    print(results["energy_latex"])

    return results


if __name__ == "__main__":
    results = main()

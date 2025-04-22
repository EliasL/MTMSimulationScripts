import sympy as sp


def define_symbols():
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


def define_shape_functions(xi, eta):
    """Define the linear triangular shape functions."""
    N1 = 1 - xi - eta
    N2 = xi
    N3 = eta
    return N1, N2, N3


def map_coordinates(symbols, shape_functions):
    """Map natural coordinates to physical configurations."""
    N1, N2, N3 = shape_functions
    X1, Y1, X2, Y2, X3, Y3 = symbols["initial"]
    x1, y1, x2, y2, x3, y3 = symbols["current"]

    # Initial configuration mapping
    X_expr = (
        N1 * sp.Matrix([X1, Y1]) + N2 * sp.Matrix([X2, Y2]) + N3 * sp.Matrix([X3, Y3])
    )

    # Current configuration mapping
    x_expr = (
        N1 * sp.Matrix([x1, y1]) + N2 * sp.Matrix([x2, y2]) + N3 * sp.Matrix([x3, y3])
    )

    return X_expr, x_expr


def compute_derivatives(X_expr, x_expr, xi, eta):
    """Compute derivatives with respect to natural coordinates."""
    # Individual derivatives
    dX_dxi = sp.diff(X_expr, xi)
    dX_deta = sp.diff(X_expr, eta)
    dx_dxi = sp.diff(x_expr, xi)
    dx_deta = sp.diff(x_expr, eta)

    # Complete Jacobian matrices
    JX = X_expr.jacobian([xi, eta])
    Jx = x_expr.jacobian([xi, eta])

    return {
        "dX_dxi": dX_dxi,
        "dX_deta": dX_deta,
        "dx_dxi": dx_dxi,
        "dx_deta": dx_deta,
        "JX": JX,
        "Jx": Jx,
    }


def compute_deformation_gradient(derivatives):
    """Compute the deformation gradient F = Jx * JX^{-1}."""
    JX = derivatives["JX"]
    Jx = derivatives["Jx"]
    F = sp.simplify(Jx * JX.inv())
    return F


def factor_common_denominator(matrix):
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
    new_matrix = matrix.applyfunc(lambda x: sp.cancel(x * common_d))
    return common_d, new_matrix


def compute_cauchy_green(F):
    """Compute the Cauchy-Green deformation tensor C = F^T * F."""
    C = F.T * F
    C12 = sp.simplify(C[0, 1])  # Extract off-diagonal component
    return C, C12


def substitute_vector_components(C12, symbols):
    """Replace coordinate differences with vector components."""
    X1, Y1, X2, Y2, X3, Y3 = symbols["initial"]
    x1, y1, x2, y2, x3, y3 = symbols["current"]
    V1x, V1y, V2x, V2y, v1x, v1y, v2x, v2y = symbols["vectors"]

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

    C12_substituted = C12.subs(subs_dict)
    C12_clean = sp.simplify(C12_substituted)

    return C12_clean


def generate_equation(variable, expr, comment=""):
    """Generate a LaTeX equation environment string."""
    eq_str = ""
    if comment:
        eq_str += comment + "\n"
    eq_str += r"\begin{equation}" + "\n"
    eq_str += variable + " = " + sp.latex(expr) + "\n"
    eq_str += r"\end{equation}" + "\n\n"
    return eq_str


def add_comment(comment):
    """Add a LaTeX comment."""
    return comment + "\n\n"


def generate_latex_document(results, symbols):
    """Generate LaTeX equations for all computed quantities."""
    xi, eta = symbols["natural"]
    N1, N2, N3 = results["shape_functions"]
    X_expr = results["X_expr"]
    x_expr = results["x_expr"]
    derivatives = results["derivatives"]
    F_factored = results["F_factored"]
    d = results["d"]
    C12 = results["C12"]
    C12_clean = results["C12_clean"]

    # Generate equations for the shape functions
    eq_N1 = generate_equation(r"N_1", N1, "Shape functions")
    eq_N2 = generate_equation(r"N_2", N2)
    eq_N3 = generate_equation(r"N_3", N3)

    # Generate equations for the mappings
    eq_X = generate_equation(
        r"\X(\xi,\eta)",
        X_expr,
        "Mapping from natural coordinates to the initial configuration",
    )
    eq_x = generate_equation(
        r"\x(\xi,\eta)",
        x_expr,
        "Mapping from natural coordinates to the current configuration",
    )

    # Generate equations for the individual derivatives
    eq_dX_dxi = generate_equation(
        r"\frac{\partial \X}{\partial \xi}",
        derivatives["dX_dxi"],
        r"Derivative of $\X$ with respect to $\xi$",
    )
    eq_dX_deta = generate_equation(
        r"\frac{\partial \X}{\partial \eta}",
        derivatives["dX_deta"],
        r"Derivative of $\X$ with respect to $\eta$",
    )
    eq_dx_dxi = generate_equation(
        r"\frac{\partial \x}{\partial \xi}",
        derivatives["dx_dxi"],
        r"Derivative of $\x$ with respect to $\xi$",
    )
    eq_dx_deta = generate_equation(
        r"\frac{\partial \x}{\partial \eta}",
        derivatives["dx_deta"],
        r"Derivative of $\x$ with respect to $\eta$",
    )

    # Generate intermediate equations for Jacobians
    eq_JX = generate_equation(
        r"J_\X",
        sp.Matrix.hstack(derivatives["dX_dxi"], derivatives["dX_deta"]),
        r"$J_\X = \left[\frac{\partial \X}{\partial \xi}, \frac{\partial \X}{\partial \eta}\right]$",
    )
    eq_Jx = generate_equation(
        r"J_\x",
        sp.Matrix.hstack(derivatives["dx_dxi"], derivatives["dx_deta"]),
        r"$J_\x = \left[\frac{\partial \x}{\partial \xi}, \frac{\partial \x}{\partial \eta}\right]$",
    )

    # Generate equation for common denominator
    eq_d = generate_equation(r"d", d, "")

    # Generate equation for the deformation gradient
    eq_F = generate_equation(
        r"F",
        F_factored,
        r"Deformation gradient computed as $F = \frac{\partial \x}{\partial \X} = J_\x J_\X^{-1}$",
    )

    # Generate equations for Cauchy-Green components
    eq_C12 = generate_equation(
        r"C_{12}",
        C12,
        r"Off-diagonal component of the Cauchy-Green deformation tensor",
    )
    eq2_C12 = generate_equation(
        r"C_{12}",
        C12_clean,
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
        + eq_F
        + eq_C12
        + eq2_C12
    )

    return document_body


def main():
    """Main function to execute the full computation sequence."""
    # Define symbols
    symbols = define_symbols()
    xi, eta = symbols["natural"]

    # Define shape functions
    shape_functions = define_shape_functions(xi, eta)

    # Map coordinates
    X_expr, x_expr = map_coordinates(symbols, shape_functions)

    # Compute derivatives
    derivatives = compute_derivatives(X_expr, x_expr, xi, eta)

    # Compute deformation gradient
    F = compute_deformation_gradient(derivatives)

    # Factor common denominator
    d, F_factored = factor_common_denominator(F)

    # Compute Cauchy-Green deformation tensor
    C, C12 = compute_cauchy_green(F)

    # Substitute vector components
    C12_clean = substitute_vector_components(C12, symbols)

    # Store all results
    results = {
        "shape_functions": shape_functions,
        "X_expr": X_expr,
        "x_expr": x_expr,
        "derivatives": derivatives,
        "F": F,
        "F_factored": F_factored,
        "d": d,
        "C": C,
        "C12": C12,
        "C12_clean": C12_clean,
    }

    # Generate LaTeX document
    latex_output = generate_latex_document(results, symbols)

    return results, latex_output


if __name__ == "__main__":
    results, latex_output = main()
    print(latex_output)

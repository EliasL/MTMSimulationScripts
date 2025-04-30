from contiPotential import SuperSimple
import sympy as sp
import re


def replace_diff(exp):
    """
    Replace occurrences of \\A_i^{a}-\\A_i^{b} with \\A_i^{a-b}}.
    """
    #                         1             2          3          4             5         6
    pattern = (
        r"(\\[A-Za-z])\^\{([A-Za-z])\}_\{(\d)\} - (\\[A-Za-z])\^\{([A-Za-z])\}_\{(\d)\}"
    )

    def repl(match):
        var1 = match.group(1)
        var2 = match.group(4)
        idx1 = match.group(3)
        idx2 = match.group(6)
        # Only replace when variable and subscript match
        if var1 != var2 or idx1 != idx2:
            return match.group(0)
        j = match.group(2)
        k = match.group(5)
        diff = f"{j} - {k}"
        return f"{var1}^{{{diff}}}_{idx1}"

    return re.sub(pattern, repl, exp)


def format_latex_expression(expr, eq_label=None):
    # First, simplify then factor so patterns like u^b-u^c appear explicitly
    fact_expr = sp.factor(sp.simplify(expr))
    # Convert to LaTeX
    latex_expr = sp.latex(fact_expr)
    # Now replace any A_i^a - A_i^b by A_i^{a-b}
    latex_expr = replace_diff(latex_expr)
    if eq_label:
        exp = f"{eq_label} &= {latex_expr}"
        return f"\\begin{{align*}}\n{exp}\n\\end{{align*}}"
    else:
        return latex_expr


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

    return common_d, sp.simplify(new_matrix)


def _condense_entry(expr):
    terms = expr.as_ordered_terms()
    # only handle exactly two terms
    if len(terms) != 2:
        return expr

    # decompose each term into (coeff, other_factors)
    cs = [t.as_coeff_mul() for t in terms]
    # find which is positive, which is negative
    pos_idx = next((i for i, (c, _) in enumerate(cs) if c > 0), None)
    neg_idx = next((i for i, (c, _) in enumerate(cs) if c < 0), None)
    if pos_idx is None or neg_idx is None:
        return expr

    coeff_pos, syms_pos = cs[pos_idx]
    coeff_neg, syms_neg = cs[neg_idx]
    # we only handle the simple case where each has exactly one Symbol factor
    if len(syms_pos) != 1 or len(syms_neg) != 1:
        return expr

    sym_pos = syms_pos[0]
    sym_neg = syms_neg[0]
    if not (sym_pos.is_Symbol and sym_neg.is_Symbol):
        return expr

    # pull apart their .name via regex:
    #   group(1)=base (e.g. "\X"), group(2)=i, group(3)=j
    pat = r"(\\?\\?[A-Za-z]+)\^([A-Za-z])_(\d+)"
    mpos = re.fullmatch(pat, sym_pos.name)
    mneg = re.fullmatch(pat, sym_neg.name)
    if not (mpos and mneg):
        return expr

    base1, a, idx1 = mpos.groups()
    base2, b, idx2 = mneg.groups()
    if base1 != base2 or idx1 != idx2:
        return expr

    diff = f"{a}-{b}"
    new_name = rf"{base1}_{{{idx1}}}^{{{diff}}}"
    return sp.Symbol(new_name)


def condense_matrix(M):
    """
    Given a Sympy Matrix M whose entries may contain things like
        -X_i^{n_j} + X_i^{n_k},
    returns a new Matrix where each such entry has been replaced by
        X_i^{n_{j-k}}.
    """
    return M.applyfunc(_condense_entry)


class FEM:
    # Local variables
    xi1, xi2 = sp.symbols("xi_1 xi_2")
    xi = [xi1, xi2]
    # Shape functions
    N1 = 1 - xi1 - xi2
    N2 = xi1
    N3 = xi2
    shape_functions = [N1, N2, N3]

    @classmethod
    def createShapeFunctionApprox(cls, nodes, key):
        """N1*node1 + N2 * node2 + N3*node3"""
        # grab the first node to see how big your vectors are
        v0 = sp.Matrix(nodes[0][key])
        # make a zero‐matrix of the same shape
        result = sp.zeros(*v0.shape)

        # accumulate N_i * nodal_vector_i
        for N, node in zip(cls.shape_functions, nodes):
            vec = sp.Matrix(node[key])
            result += N * vec

        return result

    @classmethod
    def partialDerivative(cls, nodes, numerator, denominator, condense=True):
        # Compute ∂A_∂B = ∂A_∂xi ∂xi_∂B

        if numerator == "N":
            A = cls.shape_functions
        else:
            A = cls.createShapeFunctionApprox(nodes, numerator)

        B = cls.createShapeFunctionApprox(nodes, denominator)
        dA_dxi = A.jacobian(cls.xi)
        dB_dxi = B.jacobian(cls.xi)
        if condense:
            dA_dxi = condense_matrix(dA_dxi)
            dB_dxi = condense_matrix(dB_dxi)

        xi_dB = sp.simplify(dB_dxi.inv())

        return sp.simplify(dA_dxi @ xi_dB)


def computeF(nodes):
    # n1, n2 and n3 are dictionaries of symbols for three nodes
    # They contain the initial positions (n1["X_1"], n1["X_2"]
    # and displacement (n1["u_1"], n1["u_2"])
    du_dX = FEM.partialDerivative(nodes, "u", "X")

    F = sp.eye(2) + du_dX
    return F


def compute_cauchy_green(F):
    C = F.T @ F
    return C, C[0, 0], C[0, 1], C[1, 1]


def computeP(F):
    # 1) get your symbolic energy
    phi, div_phi, div_div_phi = SuperSimple.symbolic_potential()
    # 2) unpack the symbols used inside phi
    C11_sym, C22_sym, C12_sym, beta, K, noise = SuperSimple._PHI_ARGS

    # 3) get the F‐expressions
    C, C11_expr, C12_expr, C22_expr = compute_cauchy_green(F)

    # 4) build the substitution map
    subs_dict = {
        C11_sym: C11_expr,
        C12_sym: C12_expr,
        C22_sym: C22_expr,
    }

    # 5) apply it (and your numerical assumptions) and simplify
    common_assumps = {noise: 1, beta: -sp.Rational(1, 4), K: 4}
    phi = sp.simplify(phi.subs(subs_dict).subs(common_assumps))

    # Replace C with F variables

    latex_output = []
    div_phi_constrained = {}
    # Process each derivative
    for eq_label, expr in div_phi.items():
        # Substitute assumptions and simplify
        div_phi_constrained[eq_label] = sp.simplify(
            expr.subs(subs_dict).subs(common_assumps)
        )

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
        formatted_expr = format_latex_expression(
            div_phi_constrained[eq_label], label_latex
        )
        # Create multiline LaTeX
        latex_output.append(formatted_expr)

    # sigma = 1/2 (∂Φ/∂C_R + (∂Φ/∂C_R)^T)
    sigma = sp.Matrix(
        [
            [div_phi_constrained["dPhi_dC11"], div_phi_constrained["dPhi_dC12"] / 2],
            [div_phi_constrained["dPhi_dC12"] / 2, div_phi_constrained["dPhi_dC22"]],
        ]
    )

    Sigma_latex = format_latex_expression(sigma, r"\Sigma")
    latex_output.append(Sigma_latex)

    P = 2 * F * sigma
    P = sp.simplify(P)
    # Generate LaTeX for P
    P_latex = format_latex_expression(P, r"\Pb")
    latex_output.append(P_latex)

    # Insert values for F
    sub = {F[0, 0]: 1, F[0, 1]: 0.2, F[1, 0]: 0, F[1, 1]: 1}

    F = F.subs(sub)
    # Generate LaTeX for P
    F_latex = format_latex_expression(F, r"\F")
    latex_output.append(F_latex)

    P = P.subs(sub)
    # Generate LaTeX for P
    P_latex = format_latex_expression(P, r"\Pb")
    latex_output.append(P_latex)

    return "\n\n".join(latex_output)


def computeForces(nodes):
    F = computeF(nodes)
    P = computeP(F)
    dN_dX = FEM.partialDerivative(nodes, "N", "X")
    forces = P * dN_dX.T
    return forces


def makeNode(name):
    X1, X2 = sp.symbols(f"\\X^{name}_1 \\X^{name}_2")
    x1, x2 = sp.symbols(f"\\x^{name}_1 \\x^{name}_2")
    u1, u2 = sp.symbols(f"\\ub^{name}_1 \\ub^{name}_2")
    return {
        "X": sp.Matrix([X1, X2]),
        "x": sp.Matrix([x1, x2]),
        "u": sp.Matrix([u1, u2]),
    }


if __name__ == "__main__":
    a, b, c, d = [makeNode(n) for n in "abcd"]

    A = [a, b, c]
    A_ = [a, b, d]
    B_ = [b, c, d]

    # F - I
    du_dX = FEM.partialDerivative(A, "u", "X")
    J, du_dX = factor_common_denominator(du_dX)

    print(format_latex_expression(du_dX, "du_dX"))

    print(format_latex_expression(J, "\\J"))
    # print(computeP())

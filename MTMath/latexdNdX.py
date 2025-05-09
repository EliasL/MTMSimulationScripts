import sympy as sp
import re


# -------------------------------------------------------------
# Utility: Factor out common denominator from all entries of a matrix
# -------------------------------------------------------------
def factor_denominator(mat: sp.Matrix) -> tuple[sp.Expr, sp.Matrix]:
    """
    For a symbolic matrix `mat`, factor out the least common denominator
    from all its entries. Returns (common_denom, numerator_matrix) such that
      mat = numerator_matrix / common_denom.
    """
    # Extract fractional form of each element
    dens = []
    for entry in mat:
        _, d = sp.fraction(sp.simplify(entry))
        dens.append(d)
    # Compute least common multiple of denominators
    common_d = dens[0]
    for d in dens[1:]:
        common_d = sp.lcm(common_d, d)
    # Multiply each entry by common_d and simplify
    num_mat = mat.applyfunc(lambda x: sp.simplify(x * common_d))
    return common_d, num_mat


def replace_diff(exp):
    """
    Replace occurrences of \\A_i^{n_j}-\\A_i^{n_k} with \\A_i^{n_{j-k}}.
    """
    #                         1            2       3          4             5     6
    pattern = r"\\left\((\\[A-Za-z])\^\{n_(\d)\}_(\d) - (\\[A-Za-z])\^\{n_(\d)\}_(\d)\\right\)"

    def repl(match):
        var1 = match.group(1)
        var2 = match.group(4)
        idx1 = match.group(3)
        idx2 = match.group(6)
        # Only replace when variable and subscript match
        if var1 != var2 or idx1 != idx2:
            return match.group(0)
        j = int(match.group(2))
        k = int(match.group(5))
        diff = f"{j} - {k}"
        return f"{var1}^{{n_{{{diff}}}}}_{idx1}"

    return replace_diff2(re.sub(pattern, repl, exp))


def replace_diff2(exp):
    """
    Replace occurrences of \\A_i^{a}-\\A_i^{b} with \\A_i^{a-b}}.
    """
    #                         1            2       3          4             5     6
    pattern = r"\\left\((\\[A-Za-z])\^\{([A-Za-z])\}_(\d) - (\\[A-Za-z])\^\{([A-Za-z])\}_(\d)\\right\)"

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


# -------------------------------------------------------------
# Symbolic nodal coordinates: reference (capital X) and current (small x)
# -------------------------------------------------------------
# X-coordinates in reference
X1_1, X2_1, X3_1 = sp.symbols("\\X^{n_1}_1 \\X^{n_2}_1 \\X^{n_3}_1")
# Y-coordinates in reference
X1_2, X2_2, X3_2 = sp.symbols("\\X^{n_1}_2 \\X^{n_2}_2 \\X^{n_3}_2")
# X-coordinates in current
x1_1, x2_1, x3_1 = sp.symbols("\\x^{n_1}_1 \\x^{n_2}_1 \\x^{n_3}_1")
# Y-coordinates in current
x1_2, x2_2, x3_2 = sp.symbols("\\x^{n_1}_2 \\x^{n_2}_2 \\x^{n_3}_2")
# P
P11, P12, P21, P22 = sp.symbols("P_{11} P_{12} P_{21} P_{22}")

# -------------------------------------------------------------
# Parent-element shape-function gradients dN/dξ
# -------------------------------------------------------------
dN_dxi = sp.Matrix([[-1, -1], [1, 0], [0, 1]])

# -------------------------------------------------------------
# Jacobians: dX/dξ (ref) and dx/dξ (current)
# -------------------------------------------------------------
dX_dxi = sp.Matrix([[X2_1 - X1_1, X3_1 - X1_1], [X2_2 - X1_2, X3_2 - X1_2]])

dx_dxi = sp.Matrix([[x2_1 - x1_1, x3_1 - x1_1], [x2_2 - x1_2, x3_2 - x1_2]])

# Inverse mapping: dξ/dX
xi_dX = sp.simplify(dX_dxi.inv())

# -------------------------------------------------------------
# Compute: ∇N = dN/dX and deformation gradient F = dx/dX
# -------------------------------------------------------------
dN_dX_full = sp.simplify((dN_dxi * xi_dX).T)
F_full = sp.simplify(dx_dxi * xi_dX)

# Factor denominators
J, dN_num = factor_denominator(dN_dX_full)
J, F_num = factor_denominator(F_full)

# -------------------------------------------------------------
# LaTeX output of results
# -------------------------------------------------------------
print("% Volume Jacobian J = det(dX/dξ)")
print("J =", sp.latex(J))
print()
print("% Shape-gradient numerator (dN/dX * J)")
print("dN_num =", sp.latex(dN_num[:, 0]))
print("dN_num =", sp.latex(dN_num[:, 1]))
print("dN_num =", sp.latex(dN_num[:, 2]))
print("dN/dX = dN_num / J")
print()
print("% Deformation gradient F = dx/dX")
print("F_num =", replace_diff(sp.latex(F_num)))
print("F = F_num / J")

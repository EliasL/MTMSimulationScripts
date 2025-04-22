import sympy as sp
import re
from contiPotential import ContiEnergy

# Retrieve the symbolic expressions from your potential function.
phi, div_phi, div_div_phi = ContiEnergy.symbolic_potential()
# Define your assumptions.
C11, C12, C22, noise = sp.symbols("C11 C12 C22 noise")
assumptions = {C11: 1, C22: 1 + C12 * 2, noise: 1}


# Function to round all float coefficients in an expression to one decimal
def round_floats(expr):
    # Recursively replace all sympy Floats with their rounded value
    return expr.replace(
        lambda x: isinstance(x, sp.Float), lambda x: sp.Float(round(x, 1))
    )


# Loop through the expressions in div_phi, substitute the assumptions,
# simplify the expression, round coefficients, and then generate a multiline LaTeX string.
for eq_label, expr in div_phi.items():
    # Substitute the assumptions and simplify
    expr_assumed = sp.simplify(expr.subs(assumptions))
    # Round numerical coefficients to one decimal
    expr_assumed = round_floats(expr_assumed)
    # Convert the simplified and rounded expression to a LaTeX string.
    latex_expr = sp.latex(expr_assumed)
    # Remove \left and \right from the LaTeX string
    latex_expr = latex_expr.replace(r"\left", "").replace(r"\right", "")
    # Add align to equal sign
    latex_expr = latex_expr.replace("=", "&=")
    # Map the derivative labels to LaTeX notation
    if eq_label == "dPhi_dC11":
        label_latex = r"\frac{\partial \phi}{\partial C{11}}"
    elif eq_label == "dPhi_dC22":
        label_latex = r"\frac{\partial \phi}{\partial C{22}}"
    elif eq_label == "dPhi_dC12":
        label_latex = r"\frac{\partial \phi}{\partial C{12}}"
    else:
        label_latex = eq_label
    # Place new lines such that each exponent has its own line
    newString = []
    for i in range(len(latex_expr)):
        if latex_expr[i] in ["+", "-"]:
            # We notice that we want a new line when we don't have the \beta coefficient, so this i
            if latex_expr[i - 2] != "a":
                newString.append("\\\\\n&")
        newString.append(latex_expr[i])
    latex_expr = "".join(newString)
    # Format the output using the align environment for multiline display.
    multiline_latex = (
        f"\\begin{{align}}\n{label_latex} &= {latex_expr}\n\\end{{align*}}"
    )
    # Print the multiline LaTeX formatted expression.
    print(multiline_latex)

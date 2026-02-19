from sympy import ccode, cse, simplify


def _require_matrix(name, mat, shape):
    if mat is None:
        raise ValueError(f"{name} must be provided.")
    if not hasattr(mat, "shape"):
        raise ValueError(f"{name} must be a matrix with shape {shape}.")
    if mat.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {mat.shape}.")


def _div_phi_pairs(div_phi):
    _require_matrix("div_phi", div_phi, (1, 3))
    return [
        ("dPhi_dC11", div_phi[0, 0]),
        ("dPhi_dC22", div_phi[0, 1]),
        ("dPhi_dC12", div_phi[0, 2]),
    ]


def _div_div_phi_pairs(div_div_phi):
    _require_matrix("div_div_phi", div_div_phi, (3, 3))
    return [
        ("dPhi_dC11_dC11", div_div_phi[0, 0]),
        ("dPhi_dC22_dC22", div_div_phi[1, 1]),
        ("dPhi_dC12_dC12", div_div_phi[2, 2]),
        ("dPhi_dC11_dC22", div_div_phi[0, 1]),
        ("dPhi_dC11_dC12", div_div_phi[0, 2]),
        ("dPhi_dC22_dC12", div_div_phi[1, 2]),
    ]


def generate_cpp_code(expressions):
    expressions = list(expressions)
    var_names = [name for name, _ in expressions]
    expr_values = [expr for _, expr in expressions]

    # Apply common subexpression elimination
    replacements, reduced_exprs = cse(expr_values)

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
    for name, expr in zip(var_names, reduced_exprs):
        try:
            ccode_expressions.append(f"double {name} = {ccode(expr)};")
        except Exception as e:
            ccode_expressions.append(f"// Error processing {name}: {str(e)}")

    # Combine with a blank line separator
    return "\n".join(ccode_replacements + [""] + ccode_expressions)


def _sanitize_cpp_code(code: str) -> str:
    replacements = {
        "C_{11}": "C11",
        "C_11": "C11",
        "C_{22}": "C22",
        "C_22": "C22",
        "C_{12}": "C12",
        "C_12": "C12",
    }
    for old, new in replacements.items():
        code = code.replace(old, new)
    return code


def _indent_cpp_code(code: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join((pad + line) if line else "" for line in code.splitlines())


def compute_energy_and_derivatives(
    phi_func, div_phi, div_div_phi=None, include_second_derivatives=False
):
    # Handle energy function - wrap it in a dictionary with a single key
    energy_code = generate_cpp_code([("phi", phi_func)])

    # Generate combined code if second derivatives are requested
    if include_second_derivatives:
        assert div_div_phi is not None, "Second derivatives must be provided."
        combined = _div_phi_pairs(div_phi) + _div_div_phi_pairs(div_div_phi)
        first_and_second_derivative_code = generate_cpp_code(combined)

        return energy_code, first_and_second_derivative_code
    else:
        first_derivative_code = generate_cpp_code(_div_phi_pairs(div_phi))
        return energy_code, first_derivative_code


def compute_energy_and_derivatives_merged(
    phi_func, div_phi, div_div_phi=None, include_second_derivatives=False
):
    """
    Generate a single C++ code block that computes energy and derivatives
    together, allowing CSE to share subexpressions across all outputs.
    """
    combined = [("phi", phi_func)] + _div_phi_pairs(div_phi)
    if include_second_derivatives:
        assert div_div_phi is not None, "Second derivatives must be provided."
        combined += _div_div_phi_pairs(div_div_phi)
    return generate_cpp_code(combined)


def generate_cpp_energy_stress_code(
    phi_func,
    div_phi,
    div_div_phi=None,
    include_second_derivatives=False,
    include_wrappers=True,
):
    """
    Generate a C++ merged energy+stress kernel and optional wrappers that match
    the energyDensity / stress API style.
    """
    _require_matrix("div_phi", div_phi, (1, 3))

    merged_code = compute_energy_and_derivatives_merged(
        phi_func, div_phi, div_div_phi, include_second_derivatives
    )
    merged_code = _sanitize_cpp_code(merged_code)
    merged_code = _indent_cpp_code(merged_code, spaces=2)

    lines = []
    lines.append(
        "double energyDensityAndStress(double C11, double C22, double C12, "
        "double beta, double K, double noise, Matrix2d* stress_out) {"
    )
    lines.append(merged_code)
    lines.append("")
    lines.append(
        "*stress_out = Matrix2d{{dPhi_dC11, dPhi_dC12 / 2}, "
        "{dPhi_dC12 / 2, dPhi_dC22}};"
    )
    lines.append("  return phi;")
    lines.append("}")

    if include_wrappers:
        lines.append("")
        lines.append(
            "double energyDensity(double C11, double C22, double C12, "
            "double beta, double K, double noise) {"
        )
        lines.append(
            "  return energyDensityAndStress(C11, C22, C12, beta, K, noise, nullptr);"
        )
        lines.append("}")
        lines.append("")
        lines.append(
            "Matrix2d stress(double C11, double C22, double C12, "
            "double beta, double K, double noise) {"
        )
        lines.append("  Matrix2d out;")
        lines.append("  energyDensityAndStress(C11, C22, C12, beta, K, noise, &out);")
        lines.append("  return out;")
        lines.append("}")

    lines.append("")
    return "\n".join(lines)


def generate_cpp_energy_density_code(
    phi_func, function_name="energyDensity", include_signature=True
):
    """
    Generate C++ code for energy density only (no namespace).
    """
    energy_code = generate_cpp_code([("phi", phi_func)])
    energy_code = _sanitize_cpp_code(energy_code)
    energy_code = _indent_cpp_code(energy_code, spaces=2)

    if not include_signature:
        return energy_code

    lines = [
        f"double {function_name}(double C11, double C22, double C12, double beta, "
        "double K, double noise) {",
        energy_code,
        "",
        "  return phi;",
        "}",
    ]
    return "\n".join(lines)


def generate_cpp_stress_function_code(
    div_phi, function_name="stress", include_signature=True
):
    """
    Generate C++ code for stress only (no namespace).
    """
    _require_matrix("div_phi", div_phi, (1, 3))

    stress_code = generate_cpp_code(_div_phi_pairs(div_phi))
    stress_code = _sanitize_cpp_code(stress_code)
    stress_code = _indent_cpp_code(stress_code, spaces=2)

    if not include_signature:
        return stress_code

    lines = [
        f"Matrix2d {function_name}(double C11, double C22, double C12, double beta, "
        "double K, double noise) {",
        stress_code,
        "",
        "  return Matrix2d{{dPhi_dC11, dPhi_dC12 / 2}, "
        "{dPhi_dC12 / 2, dPhi_dC22}};",
        "}",
    ]
    return "\n".join(lines)

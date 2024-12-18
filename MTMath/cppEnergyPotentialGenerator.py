from sympy import ccode, cse, simplify


def generate_cpp_code(derivatives, second_derivatives):
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


if __name__ == "__main__":
    from contiPotential import (
        compute_derivatives,
        compute_second_derivatives,
        symbolicContiPotential,
    )

    # Print the generated C++ code for both sets

    phi = symbolicContiPotential()
    fDiv = compute_derivatives(phi)
    sDiv = compute_second_derivatives(fDiv)
    ccode_first_only, ccode_all = generate_cpp_code(fDiv, sDiv)
    print("First Derivatives Only:\n", ccode_first_only)
    # print("\nFirst and Second Derivatives:\n", ccode_all)

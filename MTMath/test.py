import sympy as sp

# Define symbolic variables
a, b, c, d = sp.symbols("a b c d")

# Define matrices
F = sp.Matrix([[a, b], [c, d]])

H = sp.Matrix([[1, 1], [0, 1]])

# Compute products
HF = H * F
FH = F * H

# Display results
sp.pprint(F)
print("H * F =")
sp.pprint(HF)
print("\nF * H =")
sp.pprint(FH)

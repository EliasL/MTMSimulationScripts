import sympy as sp
import numpy as np

# Define symbolic variables
a, b, c, d = sp.symbols("a b c d")

# Define matrices
F = sp.Matrix([[a, b], [c, d]])

H = sp.Matrix([[1, 1], [0, 1]])

theta = 0.4
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
v = np.array([0.3, 0.6])

print(R @ v)
print(v @ R)
# # Compute products
# HF = H * F
# FH = F * H

# # Display results
# sp.pprint(F)
# print("H * F =")
# sp.pprint(HF)
# print("\nF * H =")
# sp.pprint(FH)

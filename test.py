import sympy as sp

# Symbols
eps, theta = sp.symbols("eps theta", real=True)

c = sp.cos(theta)
s = sp.sin(theta)

# Simple shear
S = sp.Matrix([[1, eps], [0, 1]])

# Rotation matrix
R = sp.Matrix([[c, -s], [s, c]])

# Two different “rotations” of S
S_theta_RSRt = sp.simplify(R * S * R.T)  # R S R^T
S_theta_RtSR = sp.simplify(R.T * S * R)  # R^T S R

print("R S R^T =")
sp.pprint(S_theta_RSRt)

print("\nR^T S R =")
sp.pprint(S_theta_RtSR)

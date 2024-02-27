from sympy import symbols, Matrix

# Define symbols
ξ1, ξ2 = symbols('ξ1 ξ2')
x1, x2, x3, y1, y2, y3 = symbols('x1 x2 x3 y1 y2 y3')

# Define shape functions
N1 = 1 - ξ1 - ξ2
N2 = ξ1
N3 = ξ2

# Expressions for X_1 and X_2
X_1 = N1*x1 + N2*x2 + N3*x3
X_2 = N1*y1 + N2*y2 + N3*y3

# Compute Jacobian matrix
J = Matrix([[X_1.diff(ξ1), X_1.diff(ξ2)],
            [X_2.diff(ξ1), X_2.diff(ξ2)]])

print(J)

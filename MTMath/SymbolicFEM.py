import sympy as sp
import numpy as np


class FEM:
    # Local coordinates
    xi1, xi2 = sp.symbols("xi_1 xi_2")
    xi = [xi1, xi2]

    # Shape functions
    N1 = 1 - xi1 - xi2
    N2 = xi1
    N3 = xi2
    shape_functions = sp.Matrix([N1, N2, N3])

    # Nodes
    i, j = sp.symbols("i j", integer=True)  # index symbols
    X = sp.IndexedBase("X")  # reference‐position array
    u = sp.IndexedBase("u")  # displacement array
    x = X + u

    nodes = []

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
    def partialDerivative(cls, nodes, numerator, denominator):
        # Compute ∂A_∂B = ∂A_∂xi ∂xi_∂B

        if numerator == "N":
            A = cls.shape_functions
        else:
            A = cls.createShapeFunctionApprox(nodes, numerator)

        B = cls.createShapeFunctionApprox(nodes, denominator)
        dA_dxi = A.jacobian(cls.xi)
        dB_dxi = B.jacobian(cls.xi)

        xi_dB = dB_dxi.inv()

        return dA_dxi @ xi_dB

    @classmethod
    def make_N_nodes(cls, N):
        """
        Return a small dict that points at the IndexedBase entries for node k.
        The caller can substitute numeric values later.
        """
        cls.nodes = []
        for k in range(N):
            Xk = sp.Matrix([cls.X[k, 0], cls.X[k, 1]])
            uk = sp.Matrix([cls.u[k, 0], cls.u[k, 1]])
            xk = Xk + uk
            cls.nodes.append({"X": Xk, "u": uk, "x": xk})
        return cls.nodes

    @classmethod
    def F(cls, nodes):
        return cls.partialDerivative(nodes, "x", "X")

    @classmethod
    def dN_dX(cls, nodes):
        return cls.partialDerivative(nodes, "N", "X")

    @classmethod
    def apply_shear(cls, expr, shear_variable):
        subs = {}
        for k in range(len(cls.nodes)):
            subs[cls.u[k, 0]] = shear_variable * cls.X[k, 1]
            subs[cls.u[k, 1]] = 0

        return expr.subs(subs)


if __name__ == "__main__":
    a, b, c, d = FEM.make_N_nodes(4)

    A = [a, b, c]
    A_ = [a, b, d]
    B_ = [b, c, d]

    dN_dX = FEM.dN_dX(A)
    print("dN_dX:\n", dN_dX)
    # Insert positions for nodes
    positions = np.array([[0, 0], [1, 0], [0, 1]])
    dN_dX_num = sp.lambdify(FEM.X, dN_dX, "numpy")

    print(dN_dX_num(positions))

    # F
    F = FEM.F(A)  # same as FEM.partialDerivative(A, "x", "X")
    F_num = sp.lambdify([FEM.X, FEM.u], F, "numpy")
    positions = np.array([[0, 0], [1, 0], [0, 1]])
    displacements = np.zeros_like(positions)  # or real displacements if you have them

    print(F_num(positions, displacements))

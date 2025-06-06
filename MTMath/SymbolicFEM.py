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

    # To refer to node k’s x‐coordinate and y‐coordinate:
    #   X[k,0] and X[k,1],    u[k,0] and u[k,1]
    #
    # You can build the “deformed” coordinate x[k] = [ X[k,0]+u[k,0],  X[k,1]+u[k,1] ].

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
    def make_node_indexed(cls, k):
        """
        Return a small dict that points at the IndexedBase entries for node k.
        The caller can substitute numeric values later.
        """
        Xk = sp.Matrix([cls.X[k, 0], cls.X[k, 1]])
        uk = sp.Matrix([cls.u[k, 0], cls.u[k, 1]])
        xk = Xk + uk
        return {"X": Xk, "u": uk, "x": xk}


def makeNode(name):
    X1, X2 = sp.symbols(f"X^{name}_1 X^{name}_2")
    u1, u2 = sp.symbols(f"u^{name}_1 u^{name}_2")
    x1 = X1 + u1
    x2 = X2 + u2
    return {
        "X": sp.Matrix([X1, X2]),
        "x": sp.Matrix([x1, x2]),
        "u": sp.Matrix([u1, u2]),
    }


def constrain(expr, *nodes, key="u", values=0):
    subs = {}

    if not isinstance(values, (list, np.ndarray)):
        values = [values] * len(nodes)

    for node in nodes:
        for sym, value in zip(node[key], values):
            subs[sym] = value
    return expr.subs(subs)


def compute_F(nodes):
    """
    Compute the deformation gradient F from the nodal positions.
    :param nodes: List of nodes, each a dict with keys 'X' and 'x'.
    :return: Deformation gradient F as a sympy Matrix.
    """
    return FEM.partialDerivative(nodes, "x", "X")


def set_property(prop, expression, nodes, positions):
    for node, pos in zip(nodes, positions):
        constrain(expression, node, key=prop, values=pos)


def set_reference_positions(expression, nodes, positions):
    set_property("X", expression, nodes, positions)


def set_positions(expression, nodes, positions):
    set_property("x", expression, nodes, positions)


def set_displacements(expression, nodes, positions):
    set_property("u", expression, nodes, positions)


if __name__ == "__main__":
    a, b, c, d = [makeNode(n) for n in "abcd"]

    A = [a, b, c]
    A_ = [a, b, d]
    B_ = [b, c, d]

    dN_dX = FEM.partialDerivative(A, "N", "X")
    print("dN_dX:\n", dN_dX)
    # Insert positions for nodes
    dN_dX = constrain(dN_dX, a, key="X", values=[0, 0])
    dN_dX = constrain(dN_dX, b, key="X", values=[1, 0])
    dN_dX = constrain(dN_dX, c, key="X", values=[0, 1])
    print("dN_dX with positions:\n", dN_dX)

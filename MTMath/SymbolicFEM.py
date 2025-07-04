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

    nodes = []

    class Element:
        def __init__(self, node_ids: list[int]):
            assert len(node_ids) == 3, "Only triangular elements are supported"
            self.node_ids: list[int] = node_ids
            self.nodes: list[dict[str, sp.Matrix]] = [FEM.nodes[k] for k in node_ids]

        def interpolate(self, key: str) -> sp.Matrix:
            """
            Returns a symbolic expression for the interpolated field (e.g., 'x', 'u', or 'X')
            inside the triangle using shape functions.

            The result is a vector-valued symbolic expression in terms of local coordinates (xi_1, xi_2),
            representing the field at an arbitrary point inside the element.

            To evaluate this at specific coordinates, you must use sympy.lambdify with (FEM.X, FEM.u, xi_1, xi_2), etc.
            """
            result = sp.zeros(*sp.Matrix(self.nodes[0][key]).shape)
            for N, node in zip(FEM.shape_functions, self.nodes):
                result += N * sp.Matrix(node[key])
            return result

        def deformation_gradient(self) -> sp.Matrix:
            return FEM.partialDerivative(self, "x", "X")

        def dN_dX(self) -> sp.Matrix:
            return FEM.partialDerivative(self, "N", "X")

        def apply_shear_to_interp(self, key: str, shear_variable) -> sp.Matrix:
            interp_expr = self.interpolate(key)
            return FEM.apply_shear(interp_expr, shear_variable)

    @classmethod
    def createShapeFunctionApprox(cls, element: "FEM.Element", key: str) -> sp.Matrix:
        """N1*node1 + N2 * node2 + N3*node3"""
        # grab the first node to see how big your vectors are
        nodes = element.nodes
        v0 = sp.Matrix(nodes[0][key])
        # make a zero‐matrix of the same shape
        result = sp.zeros(*v0.shape)

        # accumulate N_i * nodal_vector_i
        for N, node in zip(cls.shape_functions, nodes):
            vec = sp.Matrix(node[key])
            result += N * vec

        return result

    @classmethod
    def partialDerivative(
        cls, element: "FEM.Element", numerator: str, denominator: str
    ) -> sp.Matrix:
        # Compute ∂A_∂B = ∂A_∂xi ∂xi_∂B
        if numerator == "N":
            A = cls.shape_functions
        else:
            A = cls.createShapeFunctionApprox(element, numerator)

        B = cls.createShapeFunctionApprox(element, denominator)
        dA_dxi = A.jacobian(cls.xi)
        dB_dxi = B.jacobian(cls.xi)

        xi_dB = dB_dxi.inv()

        return dA_dxi @ xi_dB

    @classmethod
    def make_N_nodes(cls, N: int) -> list[dict[str, sp.Matrix]]:
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
    def get_displacement_expr(cls, k: int) -> sp.Matrix:
        return sp.Matrix([cls.u[k, 0], cls.u[k, 1]])

    @classmethod
    def get_position_expr(cls, k: int) -> sp.Matrix:
        return sp.Matrix([cls.X[k, 0], cls.X[k, 1]]) + cls.get_displacement_expr(k)

    @classmethod
    def F(cls, element: "FEM.Element") -> sp.Matrix:
        return cls.partialDerivative(element, "x", "X")

    @classmethod
    def dN_dX(cls, element: "FEM.Element") -> sp.Matrix:
        return cls.partialDerivative(element, "N", "X")

    @classmethod
    def apply_shear(cls, expr, shear_variable):
        subs = {}
        for k in range(len(cls.nodes)):
            subs[cls.u[k, 0]] = shear_variable * cls.X[k, 1]
            subs[cls.u[k, 1]] = 0

        return expr.subs(subs)


if __name__ == "__main__":
    FEM.make_N_nodes(4)

    element = FEM.Element([0, 1, 2])
    dN_dX = FEM.dN_dX(element)
    print("dN_dX:\n", dN_dX)
    # Insert positions for nodes
    positions = np.array([[0, 0], [1, 0], [0, 1]])
    dN_dX_num = sp.lambdify(FEM.X, dN_dX, "numpy")

    print(dN_dX_num(positions))

    # F
    F = FEM.F(element)  # same as FEM.partialDerivative(element, "x", "X")
    F_num = sp.lambdify([FEM.X, FEM.u], F, "numpy")
    positions = np.array([[0, 0], [1, 0], [0, 1]])
    displacements = np.zeros_like(positions)  # or real displacements if you have them

    print(F_num(positions, displacements))

    # Example of using FEM.Element
    x_interp = element.interpolate("x")
    print("Interpolated x field (symbolic):\n", x_interp)

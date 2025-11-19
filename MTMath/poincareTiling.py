import matplotlib.pyplot as plt
from .plotEnergy import (
    plotPoincarePointMapping,
    plotPoincareCTiling,
    plotPoincareFTiling,
    applyCongruenceTransformations,
    drawC,
    drawPoincareGrid,
    generateShearTransformations,
)
import numpy as np
from MTMath.contiPotential import ContiEnergy, SShear


# Energy adapter (expects full 2x2 C)
def energy_from_C(C_: np.ndarray) -> float:
    return ContiEnergy.energy_from_C_in_place(C_.copy())


# Generic central 3-point finite difference for φ along any matrix path C(ε)
def central_diff_phi(path_fn, eps: float = 1e-6) -> float:
    C_plus = path_fn(+eps)
    C_minus = path_fn(-eps)
    return (energy_from_C(C_plus) - energy_from_C(C_minus)) / (2.0 * eps)


def diff(F, eps=1e-6):
    directions = (0, np.pi / 4)
    return [
        central_diff_phi(lambda e: left_apply(F, SShear(e, d)), eps) for d in directions
    ]


# Convenience: congruence update C' = M^T C M
def congruence(C: np.ndarray, M: np.ndarray) -> np.ndarray:
    return M.T @ C @ M


def left_apply(F, M):
    F_ = M @ F
    return F_.T @ F_


def calculateSimpleFiniteDifferenceDerivatives():
    p1 = np.array([[1.0, 0.3], [0, 1.0]])
    p2 = np.array([[1.0, 0], [-0.3, 1.0]])
    # Very strange bug here. But it's fine if i just check some flags first
    np.geterr()
    A0 = p1.T @ p1
    B0 = p2.T @ p2
    A = applyCongruenceTransformations(A0, "r")
    B = applyCongruenceTransformations(B0, "lu")
    assert np.allclose(B, A), "A and B should be equal"

    # Central-diff gradient w.r.t. (C11, C22, C12) using symmetric directions
    def grad(C: np.ndarray) -> np.ndarray:
        D11 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float)
        D22 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=float)
        # d/dC12 changes both off-diagonals equally
        D12 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        dC11 = central_diff_phi(lambda e: C + e * D11)
        dC22 = central_diff_phi(lambda e: C + e * D22)
        dC12 = central_diff_phi(lambda e: C + e * D12)

        return np.array([[dC11, dC12], [dC12, dC22]], dtype=float)

    for C, name in zip([A0, B0, A], ["A0", "B0", "A"]):
        E = energy_from_C(C)
        G = grad(C)
        print(f"{name}:\n {C}")
        print(f"Energy: {E}")
        print(f"Gradient:\n{G}\n")


def calculateShearFiniteDifferenceDerivatives():
    # --- Base test Cs as in your earlier function ---
    p1 = np.array([[1.0, 0.3], [0.0, 1.0]])  # horizontal shear
    p2 = np.array([[1.0, 0.0], [-0.3, 1.0]])  # vertical shear (note sign)
    A0 = p1.T @ p1
    B0 = p2.T @ p2
    A = applyCongruenceTransformations(A0, "r")
    B = applyCongruenceTransformations(B0, "lu")
    assert np.allclose(B, A), "A and B should be equal"

    # --- Shear path generators H_b(ε) and V_b(ε) ---
    def H(eps: float) -> np.ndarray:
        return np.array([[1.0, eps], [0.0, 1.0]], dtype=float)

    def V(eps: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [eps, 1.0]], dtype=float)

    for C, name in zip([A0, B0, A], ["A0", "B0", "A"]):
        # directional derivatives of φ along the shear paths at C
        dphi_dH = central_diff_phi(lambda e: congruence(C, H(e)))
        dphi_dV = central_diff_phi(lambda e: congruence(C, V(e)))
        print(f"{name}:\n {C}")
        print(f"dφ/dH : {dphi_dH:.5f}")
        print(f"dφ/dV : {dphi_dV:.5f}\n")


def plotShearFiniteDifferenceDerivatives():
    # γ grid for base shear states
    gamma = np.linspace(-1, 2, 10000)
    horizontal = False

    # Elementary shear updates used to define the directional paths
    def H(eps: float) -> np.ndarray:
        return np.array([[1.0, eps], [0.0, 1.0]], dtype=float)

    def V(eps: float) -> np.ndarray:
        return np.array([[1.0, 0.0], [eps, 1.0]], dtype=float)

    def P(eps: float) -> np.ndarray:
        # Pure shear
        return np.array([[1.0 + eps, 0.0], [0.0, 1.0 / (1.0 + eps)]], dtype=float)

    # Build right-Cauchy–Green tensor from an upper shear F(γ)
    def C_from_gamma(g: float, horizontal=True) -> np.ndarray:
        A = np.array(
            [[1.0, g if horizontal else 0], [0.0 if horizontal else g, 1.0]],
            dtype=float,
        )
        return A.T @ A

    def C_from_gamma2(g: float) -> np.ndarray:
        A = np.array([[1.0, min(g, 1)], [0.0, 1.0]], dtype=float)
        B = np.array([[1.0, 0], [max(0, g - 1), 1.0]], dtype=float)
        return B.T @ A.T @ A @ B

    def C_from_gamma3(g: float, h) -> np.ndarray:
        A = np.array([[1.0, min(g, 1)], [0.0, 1.0]], dtype=float)
        B = np.array([[1.0, 0], [max(0, g - 1), 1.0]], dtype=float)
        I = np.eye(2)
        F = B @ A @ I
        return F.T @ F

    def F_from_gamma3(g: float, h) -> np.ndarray:
        A = np.array([[1.0, min(g, 1)], [0.0, 1.0]], dtype=float)
        B = np.array([[1.0, 0], [max(0, g - 1), 1.0]], dtype=float)
        I = np.eye(2)
        F = B @ A @ I
        return F

    def F_from_gamma4(g: float, returnC=False) -> np.ndarray:
        F = np.array([[1.0, g if g > 0 else 0], [-g if g < 0 else 0, 1.0]], dtype=float)

        if returnC:
            C = F.T @ F
            return C
        return F

    # Plot both curves vs γ
    plt.figure(figsize=(7, 4))

    # Directional derivatives of φ evaluated at each C(γ)
    energy = np.array([energy_from_C(F_from_gamma4(g, True)) for g in gamma])
    plt.plot(gamma, energy, label=r"$\phi$", color="black", linestyle="--")

    eps = 1e-6
    dphi_dH = np.array(
        [
            central_diff_phi(lambda e, F=F_from_gamma4(g): left_apply(F, H(e)), eps)
            for g in gamma
        ]
    )
    plt.plot(gamma, dphi_dH, label=r"$\partial\phi/\partial \mathbf{C}_\mathbf{H}$")

    dphi_dV = np.array(
        [
            central_diff_phi(lambda e, F=F_from_gamma4(g): left_apply(F, V(e)), eps)
            for g in gamma
        ]
    )
    plt.plot(
        gamma,
        dphi_dV,
        label=r"$\partial\phi/\partial \mathbf{C}_\mathbf{V}$",
        linestyle="--",
    )
    dphi_dS = np.array(
        [
            central_diff_phi(
                lambda e, F=F_from_gamma4(g): left_apply(F, SShear(e, np.pi / 4)), eps
            )
            for g in gamma
        ]
    )
    plt.plot(
        gamma,
        dphi_dS,
        label=r"$\partial\phi/\partial \mathbf{C}_{\mathbf{S}45}$",
        linestyle="-",
    )
    # for eps in [1e-5, 1e-6, 1e-7, 1e-8]:
    #     dphi_dV = np.array(
    #         [
    #             central_diff_phi(lambda e, C=C_from_gamma(g): congruence(C, V(e)), eps)
    #             for g in gamma
    #         ]
    #     )
    #     plt.plot(
    #         gamma,
    #         dphi_dV,
    #         label=r"$\partial\phi/\partial \mathbf{V}, \epsilon=$" + f"{eps:.0e}",
    #     )

    # Mark specific points using the closest gamma value on the grid
    strangePoint = 0.3014201420142014
    points = [
        (-strangePoint, r"C$_0$"),
        (strangePoint, r"A$_0$"),
        (1 + strangePoint, r"A"),
    ]

    for x_target, label in points:
        idx = np.argmin(np.abs(gamma - x_target))
        x = gamma[idx]
        y = dphi_dS[idx]
        print(dphi_dS[idx])
        plt.scatter(x, y, zorder=3)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel(r"$\gamma$ (shear)")
    plt.ylabel(r"Directional derivative of $\phi$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = f"Plots/{'horizontal' if horizontal else 'vertical'}_shear_finite_difference_derivatives_{min(gamma)}_{max(gamma)}.pdf"
    plt.savefig(path, dpi=300)
    print(f"Saved plot to: {path}")
    # plt.show()
    plt.close()
    from MTMath.plotEnergy import drawPoincareGrid, drawC, Circle

    fig, ax = plt.subplots(figsize=(6, 6))
    grid_size = 200

    drawPoincareGrid(
        ax,
        grid_size=grid_size,
        depth=6,
        c="gray",
    )
    C = np.array([F_from_gamma4(g, True) for g in gamma])
    drawC(ax, C, grid_size, c="blue", linewidth=3)

    # Mark B_0, A_0 and A in the Poincaré disk using the closest gamma values
    poincare_points = [
        (-0.3, r"C$_0$"),
        (0.3, r"A$_0$"),
        (1.3, r"A"),
    ]

    for g_target, label in poincare_points:
        idx = np.argmin(np.abs(gamma - g_target))
        g_closest = gamma[idx]
        C_point = np.array([F_from_gamma4(g_closest, True)])
        # drawC with a single C gives a visible point on the path
        if label == r"A$_0$":
            ha = "right"
        else:
            ha = "left"
        drawC(
            ax,
            C_point,
            grid_size,
            c=None,
            linewidth=2,
            label=label,
            scatter=True,
            fontsize=14,
            zorder=3,
            label_ha=ha,
        )

    # Add a thin black circle
    circleSize = grid_size / 2
    circle_center_x = grid_size / 2
    circle_center_y = grid_size / 2
    circle = Circle(
        (circle_center_x, circle_center_y),
        circleSize,
        color="black",
        fill=False,
        linewidth=1,
    )
    fig.gca().add_patch(circle)

    ax.set_xticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1, 1, 5).round(2),
    )
    ax.set_yticks(
        np.linspace(0, grid_size, 5),
        np.linspace(-1, 1, 5).round(2),
    )
    ax.set_xlabel(r"$x_p$")
    ax.set_ylabel(r"$y_p$")

    import os

    if not os.path.exists("Plots"):
        os.makedirs("Plots")
    path = f"Plots/{'horizontal' if horizontal else 'vertical'}_ShearPath.pdf"
    plt.savefig(path, dpi=500, bbox_inches="tight")
    print(f"Saved plot to {path}")


def poincareTiling():
    ax = None
    # plotPoincarePointMapping(ax=ax, fig=fig)
    # ax.clear()
    ax = plotPoincareFTiling(ax=ax, depth=1, quadrants="a", arrows=True)
    ax.clear()
    # plotPoincareTiling(ax=ax, depth=2, quadrants="a")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=3, quadrants="abcd")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=3, quadrants="ab")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=3, quadrants="cd")

    # plotPoincareTiling(ax=ax, depth=4, quadrants="abcd")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=4, quadrants="ab")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=4, quadrants="cd")

    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=1, use_labels=False, quadrants="a")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="a")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="abcd")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="ab")
    # ax.clear()
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="cd")


def quadrantIdentification(F, show=False):
    step = 0.4
    directions = [i * np.pi / 4 for i in range(4)]
    # directions = [-i * np.pi / 2 for i in range(2)]

    shears = [SShear(s, d) for s in (step, -step) for d in directions]
    direction = np.array([s / abs(s) for s in (step, -step) for d in directions])
    labels = np.array(
        [("-" if s == -1 else "") + str(d) for s in (1, -1) for d in range(4)]
    )

    shears = [SShear(s, d) for s in (step, -step) for d in directions]
    # for s, l in zip(shears, labels):
    #     print(l)
    #     print(s)
    direction = np.array([s / abs(s) for s in (step, -step) for d in directions])

    # print("\n".join(map(str, shears)))

    newF = [S @ F for S in shears]
    newC = np.array([F.T @ F for F in newF])
    energies = np.array([energy_from_C(C) for C in newC])
    derivative_like = energies / direction
    derivatives = np.array([diff(F) for F in newF])

    # Show why we should use eps=1e-6
    # exp = [10**e for e in np.linspace(-1, -15, 100)]
    # d = [diff(newF[0], e)[0] for e in exp]
    # plt.plot(exp, abs(np.array(d) - 0.08097078563196192))
    # plt.loglog()
    # plt.show()

    # """
    # Values in elastic domain:
    # """
    # point_1 = [0.08097117, 0.03222707]
    # point_2 = [-0.08097117, 0.03222707]
    # point_3 = [-0.08097117, -0.03222707]
    # point_4 = [0.08097117, -0.03222707]
    # points = np.array([point_1, point_2, point_3, point_4])
    # # We define this arbitrary order, and we now make sure each of the
    # # derivatives fits uniquely will one of these 4 points, and we
    # # return them in that order.
    # # For now, let's try to check if the order is always the same:

    # assert np.allclose(derivatives, points), "Not the same order"

    """
    Energies in elastic domain:
    """
    point0 = 0.0864664948363627
    point1 = 0.1603721122909092
    point2 = 0.0864664948363627
    point3 = 0.1603721122909092
    point_0 = -0.0864664948363627
    point_1 = -0.1603721122909083
    point_2 = -0.0864664948363627
    point_3 = -0.1603721122909083

    points = [
        point0,
        point1,
        point2,
        point3,
        point_0,
        point_1,
        point_2,
        point_3,
    ]
    domains = [
        (point0, point_3),
        (point_0, point1),
        (point_1, point2),
        (point3, point_2),
    ]
    # for d in domains:
    #     print(d)
    if not np.allclose(derivative_like, points, atol=1e-8):
        show = True
        error = True
    else:
        error = False

    if show:
        ax = drawPoincareGrid()
        drawC(
            ax,
            newC,
            scatter=True,
            label=labels,
            label_x=1,
            fontsize=18,
            s=5,
        )
        # [print(f"point {i + 1}: {c}") for i, c in enumerate(derivatives)]
        [
            print(f"point{l} = {c}".replace("-", "_"))
            for l, c in zip(labels, derivative_like)
        ]
        plt.tight_layout()
        path = "Plots/quadrantIdentification.pdf"
        plt.savefig(path)
        print(f"Fig saved to {path}")
        plt.show()
        if error:
            for i, j in zip(derivative_like, points):
                print(i, j)
            raise RuntimeError("Derivatives have changed")


def checkPoincareQuadrants(depth=6):
    # F = np.array([[1, 1], [0, 1]])
    # F = [[1, 0], [1, 1]] @ F
    F = np.array([[1, 0], [0, 1]])
    # # quadrantIdentification(F, show=False)
    # print(F)

    # F = np.array([[1, 0], [1, 1]])
    # F = F @ [[1, 1], [0, 1]]
    # print(F)
    # quadrantIdentification(F, show=True)
    # Generate all F:
    Fs, labels = generateShearTransformations(depth, startingPoint=F, leftApplied=False)

    for F in Fs:
        quadrantIdentification(F, show=False)

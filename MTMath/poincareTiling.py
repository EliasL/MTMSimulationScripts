import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from .poincareEnergy import (
    plotPoincarePointMapping,
    plotPoincareCTiling,
    plotPoincareFTiling,
    applyCongruenceTransformations,
    drawC,
    drawF,
    drawFVectors,
    drawPoincareGrid,
    prepPoincareFig,
    generateShearTransformations,
    drawCircles,
    C2Plane,
)
import numpy as np
from MTMath.energyFunction import (
    EnergyFunction,
    ContiEnergy,
    PieceWiseQuadratic,
    SShear,
    rotation,
    get_rotation,
    lagrange_reduction,
)
from .reduction import (
    elastic_domain_quadrant,
    lagrange_reduction_F,
)
import os
import string


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
    # plotPoincareFTiling(ax=ax, depth=2, quadrants="a", leftApplied=False)
    plotPoincareFTiling(ax=ax, depth=2, quadrants="a", leftApplied=True)
    # plotPoincareTiling(ax=ax, depth=2, quadrants="a")
    # plotPoincareTiling(ax=ax, depth=3, quadrants="abcd")
    # plotPoincareTiling(ax=ax, depth=3, quadrants="ab")
    # plotPoincareTiling(ax=ax, depth=3, quadrants="cd")

    # plotPoincareTiling(ax=ax, depth=4, quadrants="abcd")
    # plotPoincareTiling(ax=ax, depth=4, quadrants="ab")
    # plotPoincareTiling(ax=ax, depth=4, quadrants="cd")

    # plotPoincareTiling(ax=ax, depth=1, use_labels=False, quadrants="a")
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="a")
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="abcd")
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="ab")
    # plotPoincareTiling(ax=ax, depth=2, use_labels=False, quadrants="cd")


def baseValues():
    point0 = 0.0864664948363627  # 0
    point1 = 0.1603721122909092  # 1
    point2 = 0.0864664948363627  # 2
    point3 = 0.1603721122909092  # 3
    point_0 = -0.0864664948363627  # 4
    point_1 = -0.1603721122909083  # 5
    point_2 = -0.0864664948363627  # 6
    point_3 = -0.1603721122909083  # 7

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
        [point0, point_3],
        [point_0, point1],
        [point_1, point2],
        [point3, point_2],
    ]
    d_indexes = [
        [0, 7],
        [4, 1],
        [5, 2],
        [3, 6],
    ]
    return np.array(points), np.array(domains), np.array(d_indexes)


def oldQuadrantIdentification(F, show=False, ax=None, numbers=False):
    step = 0.4
    directions = [i * np.pi / 4 for i in range(4)]
    # directions = [-i * np.pi / 2 for i in range(2)]

    shears = [SShear(s, d) for s in (step, -step) for d in directions]
    direction = np.array([s / abs(s) for s in (step, -step) for d in directions])
    labels = np.array(
        [("-" if s == -1 else "") + str(d) for s in (1, -1) for d in range(4)]
    )

    shears = [SShear(s, d) for s in (step, -step) for d in directions]

    direction = np.array([s / abs(s) for s in (step, -step) for d in directions])

    newF = [S @ F for S in shears]
    newC = np.array([F.T @ F for F in newF])
    energies = np.array([energy_from_C(C) for C in newC])
    derivative_like = energies / direction
    # derivatives = np.array([diff(F) for F in newF])

    points, domains, i = baseValues()

    if not np.allclose(derivative_like, points, atol=1e-8):
        show = True
        error = True
    else:
        error = False

    if ax is None and show:
        ax = drawPoincareGrid()
    if ax is not None or show:
        if numbers:
            drawC(
                ax,
                newC,
                scatter=True,
                label=labels,
                label_x=1,
                label_fontsize=18,
                s=5,
            )
        else:
            # Instead of using numbers, we use scatter points with colors
            colors = ["red", "green", "blue", "black"]
            markerShape = ["o", "^", "x", "+"]
            p, d, i = baseValues()
            for ids, c, m in zip(i, colors, markerShape):
                drawC(
                    ax,
                    newC[ids],
                    scatter=True,
                    c=c if m in "x+" else None,
                    s=20,
                    marker=m,  # marker shape
                    facecolors="none",
                    edgecolors=c,  # outline color
                )
    if show:
        plt.tight_layout()
        path = "Plots/quadrantIdentification.pdf"
        plt.savefig(path)
        print(f"Fig saved to {path}")
        plt.show()

    if error:
        for i, j in zip(derivative_like, points):
            print(i, j)
        raise RuntimeError("Derivatives have changed")

    return ax


def checkPoincareQuadrants(depth=5):
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
    leftAppllied = False
    Fs, labels = generateShearTransformations(
        depth, startingPoint=F, leftApplied=leftAppllied
    )

    ax = drawPoincareGrid()
    for F in Fs:
        oldQuadrantIdentification(F, ax=ax, show=False, numbers=False)

    plt.tight_layout()
    path = (
        f"Plots/quadrantIdentification{depth}{'left' if leftAppllied else 'right'}.pdf"
    )
    plt.savefig(path)
    print(f"Fig saved to {path}")
    plt.show()


def drawLeftRightExplanationFigs():
    alphabet = iter(string.ascii_uppercase)

    # Create a 3x2 grid of subplots: rows correspond to shear=0,1,2 and
    # columns to left/right application
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    shears = (0, 1, 2)
    left_flags = (True, False)

    for i, shear in enumerate(shears):
        for j, left in enumerate(left_flags):
            ax = axes[i, j]
            prepPoincareFig(ax=ax)

            # Draw the Poincaré grid on this subplot
            drawPoincareGrid(ax=ax)

            # Build deformation gradient F
            F = np.eye(2)
            if shear:
                F = np.linalg.matrix_power(SShear("r"), shear) @ F

            # Draw the circles for this configuration
            drawCircles(ax, F, applyFromLeft=left, dot=True)

            # Add a small title for clarity
            side = (
                r"\mathbf{C}_\mathbf{S}"
                if left
                else r"\mathbf{S}_\theta^\mathsf{T}\mathbf{C}\mathbf{S}_\theta"
            )
            fig_name = next(alphabet)
            ax.set_title(rf"{fig_name}: $h={shear}$, ${side}$")

    # Remove x-labels for all but bottom row
    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")

    # Remove y-labels for all but left column
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")

    plt.tight_layout()

    if not os.path.exists("Plots"):
        os.makedirs("Plots")

    path = "Plots/Circles_grid.pdf"
    plt.savefig(path, dpi=300)
    print(f"Fig saved to {path}")


def drawRotationExplanationFigs():
    alphabet = iter(string.ascii_uppercase)

    # Create a 3x2 grid of subplots: rows correspond to shear=0,1,2 and
    # columns to left/right application
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    n = [0, 1, 2, 4]
    rotations = [np.pi * i / 8 for i in n]
    for i, r in enumerate(rotations):
        ax = axes.ravel()[i]
        prepPoincareFig(ax=ax)

        # Draw the Poincaré grid on this subplot
        drawPoincareGrid(ax=ax)

        # Build deformation gradient F
        F = SShear("r")
        # Rotate
        s = np.sin(r)
        c = np.cos(r)
        rot = np.array([[c, -s], [s, c]])
        F = rot @ F
        # Draw the circles for this configuration
        drawCircles(ax, F, applyFromLeft=True, dot=True)

        # Add a small title for clarity
        side = r"\mathbf{C}_\mathbf{S}"
        fig_name = next(alphabet)
        ax.set_title(rf"{fig_name}: $h=1$, $\theta_R={n[i]}\pi/8$, ${side}$")

        drawFVectors(ax, F, scale=0.3, margin=0)

    # Remove x-labels for all but bottom row
    for ax in axes[:-1, :].ravel():
        ax.set_xlabel("")

    # Remove y-labels for all but left column
    for ax in axes[:, 1:].ravel():
        ax.set_ylabel("")

    plt.tight_layout()

    if not os.path.exists("Plots"):
        os.makedirs("Plots")

    path = "Plots/Circles_vectors_grid.pdf"
    plt.savefig(path, dpi=300)
    print(f"Fig saved to {path}")


def drawRotation2ExplanationFigs():
    alphabet = iter(string.ascii_uppercase)

    # Create a 3x2 grid of subplots: rows correspond to shear=0,1,2 and
    # columns to left/right application
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    F1 = SShear("r")
    F2 = SShear(-1, np.pi / 2)
    F3 = SShear(-1, 0) @ SShear(-1, np.pi / 2)
    Fs = [F1, F2, F3]
    F_strings = [
        r"\mathbf{F}=\mathbf{S}|_{\theta=0}^{h=1}",
        r"\mathbf{F}=\mathbf{S}|_{\theta=\pi/2}^{h=-1}",
        r"\mathbf{F}=\mathbf{S}|_{\theta=0}^{h=-1}\mathbf{S}|_{\theta=\pi/2}^{h=-1}",
    ]
    # Build deformation gradient F
    for i, F in enumerate(Fs):
        ax = axes.ravel()[i]
        prepPoincareFig(ax=ax)

        # Draw the Poincaré grid on this subplot
        drawPoincareGrid(ax=ax)
        # Draw the circles for this configuration
        drawCircles(ax, F, applyFromLeft=True, dot=True)

        # Add a small title for clarity
        side = r"\mathbf{C}_\mathbf{S}"
        fig_name = next(alphabet)
        ax.set_title(rf"{fig_name}: $h=1$, ${F_strings[i]}$, ${side}$")

        drawFVectors(ax, F, scale=0.3, margin=0)

    # Remove y-labels for all but left column
    for ax in axes[1:].ravel():
        ax.set_ylabel("")

    if not os.path.exists("Plots"):
        os.makedirs("Plots")

    path = "Plots/Circles_2vectors_grid.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Fig saved to {path}")


def plotStressFromRealF(
    grid_size=100,
    nr_theta=12,
    nr_gamma=1000,
    gamma_lim=3,
    limits=(-0.2, 0.2),
    s_component=(0, 0),
    stress_type="cauchy",
    reduced=False,
    stability_nr_theta=100,
    stability_theta_max=np.pi,
    stability_agg="max",
    stability_cmap=None,
):
    theta = np.linspace(0, 1 * np.pi, nr_theta + 1)[:-1]
    nr_theta = len(theta)
    gamma = np.linspace(0, gamma_lim, nr_gamma)
    eFunc = ContiEnergy  # PieceWiseQuadratic
    F = SShear(h=gamma, theta=theta, s_conponent=s_component)
    if stress_type == "cauchy":
        stress = eFunc.cauchy_from_F(F)
    elif stress_type == "pk2":
        stress = eFunc.S_from_F(F)
    elif stress_type == "quadrant":
        # stress = getQuadrant(F, eFunc=eFunc)
        # stress = getQuadrant2(F, eFunc=eFunc)
        stress = getQuadrantSylvain(F.swapaxes(-1, -2) @ F, eFunc=eFunc)
    elif stress_type == "stability":
        # stability testing angles (return the most unstable direction)
        t = np.linspace(0, stability_theta_max, stability_nr_theta, endpoint=False)
        n = np.stack([np.cos(t), np.sin(t)], axis=-1)
        stress, det_min = eFunc.min_det_angle(F=F, n=n)
    else:
        raise RuntimeError(f"Unknown stress type: {stress_type}")

    if reduced and stress_type in ("cauchy", "pk2"):
        _, M = lagrange_reduction_F(F)
        Minv = np.linalg.inv(M)
        stress = Minv @ stress @ Minv.swapaxes(-1, -2)

    # stress = unRotate_by_F(F, stress)
    # R = R_body
    # RT = np.swapaxes(R, -1, -2)
    # stress = np.einsum("...ij,...jk,...kl->...il", RT, stress, R)

    # N1 is the first normal stress difference
    kwargs = {}
    if stress_type == "quadrant":
        val_q = zip([stress], ["quadrant"])

        limits = (np.min(stress), 3)
        if limits[0] < 0:
            print("Warning: Some pixels have no quadrant assigned")
        # assert min(limits) >= 0, "Quadrant should be non-negative"
        assert np.max(stress) <= 3, "Quadrant should be at most 3"

        from matplotlib.colors import ListedColormap

        nr_Colors = len(range(limits[0], limits[1] + 1))

        base = colormaps["Set1"]
        colors = base(np.linspace(0, 1, nr_Colors + 1))[1:]
        cmap = ListedColormap(colors)
        # To align the integer ticks with the center of the color bars,
        # we add 0.5 to the limits
        kwargs = {
            "cmap": cmap,
            "agg": "max",
            "cbarLims": [limits[0] - 0.5, limits[1] + 0.5],
        }
    elif stress_type == "stability":
        # Angle map for the minimum-determinant direction.
        val_q = zip([stress.astype(float)], ["stability_angle"])
        limits = (0, stability_theta_max)
        if stability_cmap is None:
            stability_cmap = "twilight"
        kwargs = {
            "cmap": stability_cmap,
            "agg": stability_agg,
            "cbarLims": list(limits),
        }
    else:
        assert isinstance(stress, np.ndarray)
        N1 = stress[..., 0, 0] - stress[..., 1, 1]
        shear_stress = stress[..., 0, 1]
        trace = stress[..., 0, 0] + stress[..., 1, 1]
        det = (
            stress[..., 0, 0] * stress[..., 1, 1]
            - stress[..., 0, 1] * stress[..., 1, 0]
        )

        val_q = list(
            zip([N1, shear_stress, trace, det], ["N1", "shear_stress", "trace", "det"])
        )

        # We don't really care about det and trace anymore
        # val_q = val_q[:2]

    def _values_to_pixels(values, *, grid_size, zoom=1, agg="mean"):
        # Replicate drawC pixelization for contours.
        C = F.swapaxes(-1, -2) @ F
        x, y = C2Plane(C)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return np.full((grid_size, grid_size), np.nan, dtype=float)

        x_plot = x * zoom * grid_size / 2 + grid_size / 2
        y_plot = y * zoom * grid_size / 2 + grid_size / 2

        xv = x_plot[valid]
        yv = y_plot[valid]
        ix = np.rint(xv).astype(int)
        iy = np.rint(yv).astype(int)
        mask = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
        ix = ix[mask]
        iy = iy[mask]

        vals = np.asarray(values, dtype=float)[valid]
        vals = vals[mask]
        finite_vals = np.isfinite(vals)
        ix = ix[finite_vals]
        iy = iy[finite_vals]
        vals = vals[finite_vals]

        pixels = np.full((grid_size, grid_size), np.nan, dtype=float)
        if agg == "max":
            tmp = np.full_like(pixels, -np.inf)
            np.maximum.at(tmp, (iy, ix), vals)
            tmp[tmp == -np.inf] = np.nan
            pixels = tmp
        elif agg == "min":
            tmp = np.full_like(pixels, np.inf)
            np.minimum.at(tmp, (iy, ix), vals)
            tmp[tmp == np.inf] = np.nan
            pixels = tmp
        elif agg == "mean":
            sum_ = np.zeros_like(pixels)
            cnt_ = np.zeros_like(pixels)
            np.add.at(sum_, (iy, ix), vals)
            np.add.at(cnt_, (iy, ix), 1.0)
            np.divide(sum_, cnt_, out=pixels, where=cnt_ > 0)
        else:
            raise ValueError(f"Unknown agg mode: {agg}")
        return pixels

    for val, quantity in val_q:
        val = np.clip(val, *limits)
        ax = drawPoincareGrid(grid_size=grid_size)
        drawF(ax, F, shade=True, shade_values=val, grid_size=grid_size, **kwargs)
        if stress_type == "stability":
            # Overlay det(q)=0 contour for the minimum determinant field.
            det_pixels = _values_to_pixels(
                det_min, grid_size=grid_size, agg="min"
            )
            ax.contour(
                det_pixels,
                levels=[0.0],
                colors="k",
                linewidths=0.8,
                origin="lower",
                extent=(0, grid_size, 0, grid_size),
            )
        path = f"Plots/RealF{stress_type}Stress/{quantity}/"
        reducedTag = "reduced" if reduced else ""
        name = f"{reducedTag}{stress_type}Stress_from_realF_{quantity}_t{nr_theta}_g{nr_gamma}_{gamma_lim}_q{grid_size}_S{'_'.join(map(str, s_component))}.pdf"
        os.makedirs(path, exist_ok=True)
        plt.savefig(
            path + name,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Fig saved to {path + name}")
        # plt.show()
    return ax


def plotsLotsOfRealFStress(stress_type="cauchy", reduced=False):
    for s_component in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        grid_size = 1000
        nr_theta = int(grid_size * np.pi)
        plotStressFromRealF(
            grid_size=grid_size,
            nr_theta=nr_theta,
            nr_gamma=int(grid_size * 1.4),
            gamma_lim=3,
            limits=(-0.2, 0.2),
            s_component=s_component,
            stress_type=stress_type,
            reduced=reduced,
        )
        if stress_type == "pk2":
            return


def bug_hunting():
    n_val = 200.0
    dtypes = [np.float32, np.float64]

    print("Testing cauchy_from_F and cauchy_from_C for different dtypes")
    for dt in dtypes:
        n = dt(n_val)
        F = np.array([[dt(1) / n, dt(0)], [dt(0), n]], dtype=dt)

        # Path 1: starting from F
        a = ContiEnergy.cauchy_from_F(F)

        # Path 2: starting from C = F^T F
        C = F.T @ F
        b = ContiEnergy.cauchy_from_C(C)

        tr_a = np.trace(a)
        tr_b = np.trace(b)

        print(f"\n== dtype {dt.__name__} ==")
        print("F =")
        print(F)
        print("trace(cauchy_from_F(F)) =", tr_a)
        print("trace(cauchy_from_C(F.T @ F)) =", tr_b)
        print("difference |tr(a) - tr(b)| =", abs(tr_a - tr_b))


def getQuadrantSimple(F):
    F_r = F.copy()
    lagrange_reduction_F(F_r)
    stress_id = getIdOfF(F)

    m1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    # Generate the 4 candidates via symmetry operations
    F_candidate1 = F_r
    F_candidate2 = m1 @ F_r @ m1
    F_candidate3 = m2 @ F_r @ m2
    F_candidate4 = m1 @ F_candidate3 @ m1

    F_candidates = [F_candidate1, F_candidate2, F_candidate3, F_candidate4]
    for i, f_cand in enumerate(F_candidates):
        if getIdOfF(f_cand) == stress_id:
            return i


def getSState(F, eFunc=ContiEnergy, tol=1e-15):
    S = eFunc.S_from_F(F)
    quadrant_idx = np.full(F.shape[:-2], -1, dtype=int)
    S1 = S[..., 0, 1]
    S2 = (S[..., 0, 0] - S[..., 1, 1]) / 2

    mask1 = (S1 >= 0) & (S2 >= 0)
    mask2 = (S1 < 0) & (S2 >= 0)
    mask3 = (S1 >= 0) & (S2 < 0)
    mask4 = (S1 < 0) & (S2 < 0)

    quadrant_idx[mask1] = 1
    quadrant_idx[mask2] = 2
    quadrant_idx[mask3] = 3
    quadrant_idx[mask4] = 4

    # ---- integer-like F -> state 0 ----
    F_int_like = np.all(np.abs(F - np.round(F)) < tol, axis=(-2, -1))
    quadrant_idx[F_int_like] = 0

    # ---- diagonal-only F (off-diagonals ~ 0) -> state 0 ----
    F_diag_like = (np.abs(F[..., 0, 1]) < tol) & (np.abs(F[..., 1, 0]) < tol)

    # ---- symmetric F -> state 0 ----
    F_sym = (np.abs(F[..., 0, 1] - F[..., 1, 0]) < tol) & (
        np.abs(F[..., 0, 0] - F[..., 1, 1]) < tol
    )

    # Combine the "zero state" conditions
    zero_state = F_int_like | F_diag_like | F_sym
    quadrant_idx[zero_state] = 0

    return quadrant_idx


def getQuadrant(F: np.ndarray, eFunc=ContiEnergy) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (2, 2):
        raise ValueError("F must have shape (..., 2, 2)")

    F_r = F.copy()
    lagrange_reduction_F(F_r)

    ref_state = getSState(F, eFunc=eFunc)  # shape (...,) or (..., a, b, ...)

    m1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    F_candidate1 = F_r
    F_candidate2 = np.einsum("...ij,jk->...ik", F_r, m1)
    F_candidate3 = np.einsum("...ij,jk->...ik", F_r, m2)
    F_candidate4 = np.einsum("...ij,jk->...ik", F_candidate3, m1)
    # F_candidate1 = F_r
    # F_candidate2 = np.einsum("ij,...jk,kl->...il", m1, F_r, m1)
    # F_candidate3 = np.einsum("ij,...jk,kl->...il", m2, F_r, m2)
    # F_candidate4 = np.einsum("ij,...jk,kl->...il", m1, F_candidate3, m1)

    F_candidates = np.stack(
        (F_candidate1, F_candidate2, F_candidate3, F_candidate4), axis=0
    )  # (4, ..., 2, 2)

    cand_states = getSState(
        F_candidates, eFunc=eFunc
    )  # shape (4, ...,) or (4, ..., a, b, ...)

    notZero = ref_state != 0  # (...,)

    eq = cand_states == ref_state[None, ...]  # (4, ..., [state...])

    batch_ndim = F.ndim - 2
    matches = np.all(eq, axis=tuple(range(1 + batch_ndim, eq.ndim)))  # (4, ...)

    # Ignore integer-only reference states: forbid matches there
    matches &= notZero[None, ...]  # (4, ...)

    num_matches = np.sum(matches, axis=0)  # (...,)

    # print(num_matches[57])
    # print(cand_states[:, 57])
    # print(ref_state[57])

    if np.any(num_matches > 1):
        print("Ambiguous quadrant identification for some F")
        ambiguous = np.where(num_matches > 1)  # tuple of index arrays
        n_amb = ambiguous[0].size
        print("Number of ambiguous cases:", n_amb)

        # pick the middle ambiguous case
        amb_list = list(zip(*ambiguous))  # list of index tuples
        idx = amb_list[n_amb // 2]  # one index tuple

        print("Ambiguous index:", idx)
        print("F at idx:")
        print(F[idx])
        print("Ref state:")
        print(ref_state[idx])
        print("Candidates:")
        print(cand_states[:, *idx])

    quadrant_idx = np.argmax(matches, axis=0)

    # Do not treat integer-only / zeroStates as failures or ambiguities.
    # They are explicitly labeled as 0 by getSState.
    quadrant_idx[~notZero] = 0

    # Only consider "no match" among non-zero reference states.
    noMatches = (num_matches == 0) & notZero
    quadrant_idx[noMatches] = -1

    if np.any(noMatches):
        for idx in zip(*np.where(noMatches)):
            print("No match at index:", idx)
            print("Reference state:")
            print(ref_state[idx])
            print("Candidate states:")
            print(cand_states[(slice(None),) + idx])
            print("F")
            print(F[idx])

    return quadrant_idx


def getQuadrant2(F):
    F_r = F.copy()
    lagrange_reduction_F(F_r)

    m1 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=float)
    m2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    F_candidate1 = F_r
    F_candidate2 = np.einsum("ij,...jk,kl->...il", m1, F_r, m1)
    F_candidate3 = np.einsum("ij,...jk,kl->...il", m2, F_r, m2)
    F_candidate4 = np.einsum("ij,...jk,kl->...il", m1, F_candidate3, m1)

    F_candidates = np.stack(
        (F_candidate1, F_candidate2, F_candidate3, F_candidate4), axis=0
    )  # (4, ..., 2, 2)

    oStress = ContiEnergy.cauchy_from_F(F)  # (..., S, S)
    cStress = ContiEnergy.cauchy_from_F(F_candidates)  # (4, ..., S, S)

    # Compare whole SxS tensors, not individual entries
    matches = np.all(
        np.isclose(cStress, oStress[None, ...], rtol=1e-7, atol=1e-9), axis=(-2, -1)
    )  # (4, ...)

    num_matches = np.sum(matches, axis=0)  # (...,)
    if np.any(num_matches > 1):
        print("Ambiguous quadrant identification for some F")
        ambiguous = np.where(num_matches > 1)  # tuple of index arrays
        n_amb = ambiguous[0].size
        print("Number of ambiguous cases:", n_amb)

        # pick the middle ambiguous case
        amb_list = list(zip(*ambiguous))  # list of index tuples
        idx = amb_list[n_amb // 2]  # one index tuple

        print("Ambiguous index:", idx)
        print("F at idx:")
        print(F[idx])
        print("Original stress:")
        print(oStress[idx])
        print("Candidates:")
        print(cStress[:, *idx])

    quadrant_idx = np.argmax(matches, axis=0)
    noMatches = num_matches == 0
    quadrant_idx[noMatches] = -1
    if np.any(noMatches):
        for idx in zip(*np.where(noMatches)):
            print("No match at index:", idx)
            print("Reference state:")
            print(oStress[idx])
            print("Candidate states:")
            print(cStress[(slice(None),) + idx])
            print("F")
            print(F[idx])
    return quadrant_idx


def getQuadrantSylvain(C, eFunc: type[EnergyFunction] = ContiEnergy):
    # Step 1: Gauss/Lagrange reduction to D

    C0, m0 = lagrange_reduction(C)
    m0Inv = np.linalg.inv(m0)

    # Second PK stress from original C
    S = eFunc.S_from_C(C)

    # Step 2: transport stress to reduced basis: S0 = m0^{-1} S m0^{-T}
    S0 = m0Inv @ S @ m0Inv.swapaxes(-1, -2)

    # Initialize Q as identity in every batch entry
    Q = np.zeros_like(C0)
    Q[..., 0, 0] = 1.0
    Q[..., 1, 1] = 1.0

    Qnr = np.zeros(C0.shape[:-2], dtype=int)

    N = np.diag((1, -1))
    P = np.array([[0, 1], [1, 0]])

    # ---- Step 3: Fix the shear sign using N if needed ----
    # Work with S0, as in the note
    S1 = S0.copy()

    mask1 = S1[..., 0, 1] < 0  # (S0)12 < 0
    if np.any(mask1):
        # Q1 = N on those entries
        Q[mask1] = Q[mask1] @ N
        # Update S1 = N^{-1} S0 N^{-T} = N S0 N (since N is its own inverse)
        S1[mask1] = N @ S1[mask1] @ N
        Qnr[mask1] += 1

    # ---- Step 4: Order the normal components using P if needed ----
    # Now we must look at S1, not S0
    mask2 = S1[..., 0, 0] > S1[..., 1, 1]  # (S1)11 > (S1)22
    if np.any(mask2):
        Q[mask2] = Q[mask2] @ P
        # S2 = P^{-1} S1 P^{-T} = P S1 P
        S1[mask2] = P @ S1[mask2] @ P
        Qnr[mask2] += 2

    # For these Q (products of N and P), Q^{-1} = Q^T
    QInv = Q.swapaxes(-1, -2)

    # Central representatives (if/when you want them)
    S_s = QInv @ S0 @ QInv.swapaxes(-1, -2)  # = Q^T S0 Q
    C_s = QInv @ C0 @ Q  # = Q^T C0 Q

    # return C_s, S_s
    return Qnr


# Vectorized version of getIDOfF for batch arrays of F
def getIdOfF(F: np.ndarray, theta: np.ndarray = np.array(0)) -> np.ndarray:
    """Return ``(sigma_12, (sigma_11-sigma_22)/2)`` for Cauchy stress.

    ``theta`` rotates the spatial Cauchy-stress tensor before its two ID
    components are extracted.
    """
    s = ContiEnergy.cauchy_from_F(F)
    # s = ContiEnergy.S_from_F(F)
    R = rotation(theta)
    RT = np.swapaxes(R, -1, -2)
    s = np.einsum("...ij,...jk,...kl->...il", RT, s, R)
    shear = s[..., 0, 1]
    N1 = (s[..., 0, 0] - s[..., 1, 1]) / 2
    return np.stack((shear, N1), axis=-1)


def printRot(A, name=""):
    r = get_rotation(A)
    angle = np.arctan2(r[1, 0], r[0, 0])
    print(name, f"rotation angle (deg): {angle * 180 / np.pi:.5f}")


def tryAllRotations(
    grid_size: int = 200,
    angle_max: float = 0.999 * np.pi,
    n_theta: int = int(1e6),
    first_tolerance: float = 1e-4,
    second_tolerance: float = 1e-3,
    reduction_max_depth: int = 5,
    save: bool = True,
    show: bool = True,
):
    """
    Find the two elastic-domain endpoints with the unit-shear BFS and search
    their lifted deformation gradients for the Cauchy-stress ID of the
    reference F.  The lift is kept as ``F @ M``; reconstructing it from the
    endpoint metric would discard its spatial rotation.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")
    if not np.isfinite(angle_max) or angle_max <= 0:
        raise ValueError("angle_max must be finite and positive")
    if first_tolerance < 0 or second_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if reduction_max_depth < 1:
        raise ValueError("reduction_max_depth must be positive")

    # row of two plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ax = axs[0]
    right_ax = axs[1]

    # Base grid and deformation
    prepPoincareFig(
        ax=ax,
        grid_size=grid_size,
        withGrid=False,
        withYieldSurface=False,
    )
    ax = drawPoincareGrid(ax=ax, grid_size=grid_size)
    F = (
        SShear(1.6, s_conponent=(0, 1))
        # @ SShear(1.3, s_conponent=(1, 0))
        # @ SShear(0.0, s_conponent=(0, 1))
    )
    # Fs = np.array(
    #     [
    #         [[2.03332392, 0.51297622], [0.51297622, 0.62122153]],
    #         [[2.26983373, 0.60578949], [0.60578949, 0.60223834]],
    #         [[2.50725914, 0.69578482], [0.69578482, 0.59192785]],
    #         [[2.74535599, 0.78371367], [0.78371367, 0.58797734]],
    #     ]
    # )
    # F = Fs[1, :, :]
    # #      [[[2.74535599 0.78371367]
    # #    [0.78371367 0.58797734]]]]
    # F = F @ Rotation(-0.055 * np.pi)
    C = F.T @ F
    print("Original F:\n", F)
    printRot(F, "Original F")

    candidate_Cs, candidate_paths = elasticReductionBFS(
        C,
        max_depth=reduction_max_depth,
        eFunc=ContiEnergy,
        stress_measure="cauchy",
        plot=False,
        return_paths=True,
    )

    # Pick the first (therefore shortest) BFS path reaching each distinct
    # endpoint and retain its actual deformation-gradient lift.  The previous
    # F_from_C(candidate_C) reconstruction was the source of the false
    # non-zero rotations.
    representative_paths = []
    F_candidates = []
    for candidate_number, candidate_C in enumerate(candidate_Cs):
        representative = next(
            path
            for path in candidate_paths
            if path["candidate_index"] == candidate_number
        )
        candidate_F = F @ representative["M"]
        if not np.allclose(candidate_F.T @ candidate_F, candidate_C):
            raise RuntimeError("BFS lift is inconsistent with its endpoint metric")
        representative_paths.append(representative)
        F_candidates.append(candidate_F)

    if np.any(elastic_domain_quadrant(candidate_Cs) < 0):
        raise RuntimeError("BFS did not finish in the elastic domain")

    candidate_colors = tuple(
        ("tab:blue", "tab:orange", "tab:green", "tab:red")[: len(F_candidates)]
    )
    candidate_linestyles = tuple(("-", "--", ":", "-.")[: len(F_candidates)])

    print(f"Elastic BFS found {len(F_candidates)} distinct endpoints")
    for i, (candidate_C, candidate_F) in enumerate(
        zip(candidate_Cs, F_candidates)
    ):
        print(f"Candidate {i} C:\n", candidate_C)
        print(f"Candidate {i} lifted F:\n", candidate_F)
        print(f"Candidate {i} path:", " -> ".join(representative_paths[i]["path"]))
        print(f"Candidate {i} unrotated ID:", getIdOfF(candidate_F))

    # ID of the original F
    solution_id = np.array(getIdOfF(F), dtype=float)
    right_ax.plot(
        solution_id[0],
        solution_id[1],
        "ko",
        markersize=5,
        zorder=5,
        label=r"$\mathbf{F}$",
    )
    print(f"Solution IDs: {solution_id}")

    # Angle sampling and precomputed rotation matrices
    theta = np.linspace(0.0, angle_max, n_theta)

    def find_rotation_matches(F_cand, route_name):
        """Return the sampled stress-ID curve and best match per crossing."""
        match_angles = []
        # getIdOfF calculates the Cauchy stress once and evaluates its rotated
        # stress-state ID over all theta values.
        candidate_ids = getIdOfF(F_cand, theta)  # shape (n_theta, 2)
        diff = np.abs(candidate_ids - solution_id)
        min_diff_idx = np.argmin(np.sum(diff, axis=-1))
        print(
            f"{route_name} closest approach at angle "
            f"{theta[min_diff_idx] * 180 / np.pi:.5f} deg\n"
            f"with diff {diff[min_diff_idx]}"
        )
        mask = np.all(diff <= first_tolerance, axis=-1)

        if not np.any(mask):
            return candidate_ids, []

        candidate_indices = np.flatnonzero(mask)
        contiguous_groups = np.split(
            candidate_indices,
            np.flatnonzero(np.diff(candidate_indices) > 1) + 1,
        )
        # A tolerance crossing generally covers several adjacent samples.
        # Represent each crossing by the sample with the smallest total error
        # instead of reporting arbitrary points at both ends of the window.
        best_indices = [
            group[np.argmin(np.sum(diff[group], axis=-1))]
            for group in contiguous_groups
            if group.size
        ]

        route_matches = []
        for idx in best_indices:
            angle = float(theta[idx])

            # Second tolerance: avoid duplicate angles for this candidate.
            if match_angles and np.any(
                np.isclose(angle, match_angles, atol=second_tolerance, rtol=0)
            ):
                continue
            match_angles.append(angle)

            cand_id = candidate_ids[idx]

            print(
                f"Rotation angle: {angle * 180 / np.pi:.5f} deg\n"
                f"IDs: {cand_id}"
            )
            route_matches.append((angle, cand_id))
        return candidate_ids, route_matches

    # Compare the two elastic-domain candidates to the solution.
    matches = []
    for i, F_cand in enumerate(F_candidates):
        candidate_ids, route_matches = find_rotation_matches(
            F_cand,
            f"Candidate {i}",
        )
        sample_step = int(1 + len(candidate_ids) / 1000)
        right_ax.plot(
            candidate_ids[::sample_step, 0],
            candidate_ids[::sample_step, 1],
            color=candidate_colors[i],
            linestyle=candidate_linestyles[i],
            linewidth=1.5,
            zorder=2,
            label=rf"$\tilde{{\mathbf{{F}}}}_{i}$",
        )
        for angle, cand_id in route_matches:
            matches.append((F_cand, angle, cand_id, i))

    # Plot the original metric.
    drawC(
        ax,
        np.array([C]),
        grid_size=grid_size,
        scatter=True,
        s=50,
        label="Original F",
        label_x=4,
        label_y=3,
        label_fontsize=12,
        zorder=5,
    )

    matches_by_candidate = {}
    for _, angle, _, candidate_number in matches:
        matches_by_candidate.setdefault(candidate_number, []).append(angle)

    move_matrices = {
        "U+": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "U-": np.array([[1.0, -1.0], [0.0, 1.0]]),
        "L+": np.array([[1.0, 0.0], [1.0, 1.0]]),
        "L-": np.array([[1.0, 0.0], [-1.0, 1.0]]),
    }

    # Draw one shortest BFS path to each endpoint.  Their shared prefix is
    # drawn once in gray, keeping the small central region readable.
    path_segments = []
    for candidate_number, representative in enumerate(representative_paths):
        current_C = C.copy()
        for move_label in representative["path"]:
            next_C = congruence(current_C, move_matrices[move_label])
            existing = next(
                (
                    segment
                    for segment in path_segments
                    if np.allclose(segment["start"], current_C)
                    and np.allclose(segment["end"], next_C)
                ),
                None,
            )
            if existing is None:
                path_segments.append(
                    {
                        "start": current_C.copy(),
                        "end": next_C.copy(),
                        "candidates": [candidate_number],
                    }
                )
            else:
                existing["candidates"].append(candidate_number)
            current_C = next_C

    for segment in path_segments:
        users = segment["candidates"]
        shared = len(users) > 1
        candidate_number = users[0]
        drawC(
            ax,
            np.array([segment["start"], segment["end"]]),
            grid_size=grid_size,
            arrow=True,
            c="0.45" if shared else candidate_colors[candidate_number],
            linestyle="-" if shared else candidate_linestyles[candidate_number],
            linewidth=1.2,
            alpha=0.8,
            zorder=3,
        )

    for candidate_number, F_cand in enumerate(F_candidates):
        match_angles = matches_by_candidate.get(candidate_number, [])
        rotation_label = ", ".join(
            rf"${angle * 180 / np.pi:.2f}^\circ$" for angle in match_angles
        )
        candidate_label = rf"$\tilde{{\mathbf{{F}}}}_{candidate_number}$"
        if rotation_label:
            candidate_label += "\n" + rotation_label

        drawF(
            ax,
            F_cand,
            grid_size=grid_size,
            scatter=True,
            s=55 if match_angles else 35,
            c="green" if np.linalg.det(F_cand) > 0 else "red",
            edgecolors=candidate_colors[candidate_number],
            linewidths=1.5,
            label=candidate_label,
            label_x=-6 if candidate_number == 0 else 6,
            label_y=8,
            label_ha="center",
            label_va="bottom",
            label_fontsize=11,
            zorder=5,
        )

    ax.set_title("Configuration space")
    right_ax.set_title("Rotated Cauchy-stress IDs")
    right_ax.legend(frameon=False, loc="center")
    # equal aspect
    right_ax.set_aspect("equal")
    right_ax.set_xlabel(r"Cauchy shear stress $\sigma_{12}$")
    right_ax.set_ylabel(r"Cauchy $N_1=(\sigma_{11} - \sigma_{22})/2$")
    plt.tight_layout()

    # Theta space diagnostics
    print(f"Theta range: {theta[0]} to {theta[-1]}")
    print(f"Total matches found: {len(matches)}")
    print("match candidates:", [m[3] for m in matches])
    print("Elastic BFS candidate rotations:")
    for candidate_number in range(len(F_candidates)):
        quadrant = int(elastic_domain_quadrant(candidate_Cs[candidate_number]))
        route_matches = matches_by_candidate.get(candidate_number, [])
        if not route_matches:
            print(
                f"  Candidate {candidate_number}, quadrant {quadrant}: "
                "no stress-ID match"
            )
            continue
        for angle in route_matches:
            matched_id = getIdOfF(F_candidates[candidate_number], angle)
            id_error = np.abs(matched_id - solution_id)
            print(
                f"  Candidate {candidate_number}, quadrant {quadrant}: "
                f"rotation={angle * 180 / np.pi:.5f} deg, "
                f"ID error={id_error}"
            )

    print("Elastic BFS path rotations:")
    for path_result in candidate_paths:
        candidate_number = path_result["candidate_index"]
        path_F = F @ path_result["M"]
        path_id = getIdOfF(path_F)
        representative_id = getIdOfF(F_candidates[candidate_number])
        if np.allclose(path_id, representative_id, atol=1e-12, rtol=1e-12):
            path_matches = [
                (angle, getIdOfF(path_F, angle))
                for angle in matches_by_candidate.get(candidate_number, [])
            ]
        else:
            _, path_matches = find_rotation_matches(
                path_F,
                " -> ".join(path_result["path"]),
            )
        path_result["F"] = path_F
        path_result["matches"] = path_matches
        path_label = " -> ".join(path_result["path"]) or "start"
        for angle, matched_id in path_matches:
            id_error = np.abs(matched_id - solution_id)
            print(
                f"  {path_label} -> candidate {candidate_number}: "
                f"rotation={angle * 180 / np.pi:.5f} deg, "
                f"ID error={id_error}"
            )

    F_str = "_".join(f"{x:.2f}" for x in F.ravel())
    path = f"Plots/tryAllRotationsF_{F_str}.pdf"
    if save:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Fig saved to {path}")
    if show:
        plt.show()
    return fig, axs, matches, candidate_paths


def elasticReductionBFS(
    C0: np.ndarray | None = None,
    *,
    max_depth: int = 5,
    grid_size: int = 250,
    eFunc: type[EnergyFunction] = ContiEnergy,
    stress_measure: str = "cauchy",
    plot: bool = True,
    return_paths: bool = False,
    show: bool = False,
    show_dead_ends=False,
    name="",
):
    """Explore unit-shear paths until they first enter the elastic center.

    The returned array contains distinct terminal metrics, even when several
    BFS paths reach the same metric.  With ``return_paths=True``, a second
    return value records every terminal path and its accumulated right-side
    transformation.  Path signatures use Cauchy stress by default; pass
    ``stress_measure="pk2"`` for the previous second Piola--Kirchhoff behavior.
    ``plot=False`` skips the legacy BFS path figure.
    """
    # -------------------------
    # Input / defaults
    # -------------------------
    if C0 is None:
        F0 = SShear(1.3) @ SShear(0.9, s_conponent=(1, 0))
        C0 = F0.T @ F0

    C0 = np.asarray(C0, dtype=float)
    if C0.shape != (2, 2):
        raise ValueError("C0 must be a (2,2) array")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if stress_measure not in {"cauchy", "pk2"}:
        raise ValueError("stress_measure must be 'cauchy' or 'pk2'")

    def M_upper(k: int) -> np.ndarray:
        return np.array([[1.0, float(k)], [0.0, 1.0]], dtype=float)

    def M_lower(k: int) -> np.ndarray:
        return np.array([[1.0, 0.0], [float(k), 1.0]], dtype=float)

    moves: list[tuple[str, np.ndarray]] = [
        ("U+", M_upper(+1)),
        ("U-", M_upper(-1)),
        ("L+", M_lower(+1)),
        ("L-", M_lower(-1)),
    ]

    def inv_move(move_label):
        match move_label:
            case "U+":
                return "U-"
            case "U-":
                return "U+"
            case "L+":
                return "L-"
            case "L-":
                return "L+"
            case _:
                raise ValueError(f"Invalid move label {move_label}")

    def is_fundamental_domain(C: np.ndarray) -> bool:
        if C[0, 1] < 0:
            return False
        if C[1, 1] < C[0, 0]:
            return False
        if 2 * C[0, 1] > C[0, 0]:
            return False
        return True

    def is_elastic(C: np.ndarray) -> bool:
        m1 = np.array([[1, 0], [0, -1]])
        m2 = np.array([[0, 1], [1, 0]])
        trans = [np.eye(2), m1, m2, m1 @ m2]
        for t in trans:
            if is_fundamental_domain(t.T @ C @ t):
                return True
        return False

    def stress_signature(C: np.ndarray) -> tuple[int, int]:
        C = np.asarray(C, dtype=float)
        if stress_measure == "cauchy":
            stress = eFunc.cauchy_from_C(C)
        else:
            stress = eFunc.S_from_C(C)
        shear = float(stress[0, 1])
        n1 = float((stress[0, 0] - stress[1, 1]) / 2.0)
        # Map near-zero to 0 to avoid noisy sign flips.
        sgn = lambda x: 0 if abs(x) <= 1e-14 else (1 if x > 0 else -1)
        return (sgn(shear), sgn(n1))

    ref_sig = stress_signature(C0)

    std_color, e_color, eMatch_color = "gray", "red", "green"
    # Each node stores its metric, accumulated transformation, and path.
    nodes: list[dict] = []
    nodes.append(
        {
            "C": C0,
            "M": np.eye(2, dtype=int),
            "path": (),
            "parent": None,
            "move": None,
            "depth": 0,
            "color": std_color,
        }
    )

    def backPropogateColor(node, color):
        # Colors have priority std_color<e_color<eMatch_color
        if node["color"] == eMatch_color:
            return
        if node["color"] == e_color and color != eMatch_color:
            return

        node["color"] = color
        if node["parent"] is not None:
            backPropogateColor(nodes[node["parent"]], color)

    from collections import deque

    terminal_indices = []
    if is_elastic(C0):
        terminal_indices.append(0)
        q = deque()
    else:
        q = deque([0])

    while q:
        idx = q.popleft()
        C = nodes[idx]["C"]
        depth = nodes[idx]["depth"]

        if depth >= max_depth:
            continue

        for move_label, M in moves:
            if nodes[idx]["move"] == inv_move(move_label):
                # we don't move directly back from where we came
                continue
            Cn = congruence(C, M)
            M_total = nodes[idx]["M"] @ M
            path = nodes[idx]["path"] + (move_label,)

            nidx = len(nodes)
            inEDomain = is_elastic(Cn)
            nodes.append(
                {
                    "C": Cn,
                    "M": M_total,
                    "path": path,
                    "parent": idx,
                    "move": move_label,
                    "depth": depth + 1,
                    "inEDomain": inEDomain,
                    "color": std_color,
                }
            )
            if not inEDomain:
                q.append(nidx)
            else:
                terminal_indices.append(nidx)
                if stress_signature(Cn) == ref_sig:
                    backPropogateColor(nodes[-1], "green")
                else:
                    backPropogateColor(nodes[-1], "red")

    unique_candidates = []
    terminal_paths = []
    for terminal_index in terminal_indices:
        terminal_node = nodes[terminal_index]
        terminal_C = terminal_node["C"]
        candidate_index = next(
            (
                i
                for i, candidate in enumerate(unique_candidates)
                if np.allclose(terminal_C, candidate, atol=1e-12, rtol=1e-12)
            ),
            None,
        )
        if candidate_index is None:
            candidate_index = len(unique_candidates)
            unique_candidates.append(terminal_C.copy())
        terminal_paths.append(
            {
                "candidate_index": candidate_index,
                "C": terminal_C.copy(),
                "M": terminal_node["M"].copy(),
                "path": terminal_node["path"],
                "depth": terminal_node["depth"],
            }
        )

    if not unique_candidates:
        raise RuntimeError(
            f"Elastic BFS did not reach the center within depth {max_depth}"
        )
    candidate_Cs = np.stack(unique_candidates)

    result = (candidate_Cs, terminal_paths) if return_paths else candidate_Cs
    if not plot:
        return result

    # -------------------------
    # Plot
    # -------------------------
    ax = drawPoincareGrid(grid_size=grid_size)

    # Helper to plot a path as a polyline in the disk
    def plot_line(
        Cs: np.ndarray,
        *,
        linestyle: str = "-",
        linewidth: float = 2.0,
        c=None,
        label="",
    ):
        drawC(
            ax,
            Cs,
            grid_size=grid_size,
            linewidth=linewidth,
            linestyle=linestyle,
            c=c,
            arrow=True,
            label=label,
        )

    def plotPaths(nodes):
        for node in nodes:
            if node["parent"] is None:
                continue
            parent = nodes[node["parent"]]

            color = std_color
            linestyle = "-."

            if node["color"] == parent["color"] or parent["color"] == eMatch_color:
                color = node["color"]

            if color != std_color:
                linestyle = "-"
            if color == std_color and not show_dead_ends:
                continue

            plot_line(
                np.array([parent["C"], node["C"]]),
                c=color,
                linestyle=linestyle,
                label=str(node["depth"]),
            )

    plotPaths(nodes)

    # Mark start
    drawC(
        ax,
        np.array([C0]),
        grid_size=grid_size,
        scatter=True,
        s=40,
        c=None,
        zorder=6,
    )

    plt.tight_layout()
    with_ends = "_with_deads" if show_dead_ends else ""
    out_path = f"Plots/{name}ElasticReduction_d{max_depth}{with_ends}.pdf"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return result


def elasticReductionPlots():
    F0 = SShear(1.3) @ SShear(0.9, s_conponent=(1, 0))
    C0 = F0.T @ F0

    F1 = SShear(1.3) @ SShear(0.4, s_conponent=(1, 0))
    C1 = F1.T @ F1

    for C, name in zip([C0, C1], ["far", "close"]):
        for depth in range(1, 6):
            showDeadEnds = depth < 3
            elasticReductionBFS(
                C, max_depth=depth, show_dead_ends=showDeadEnds, name=name
            )

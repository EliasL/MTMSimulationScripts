import numpy as np
import matplotlib.pyplot as plt

from MTMath.energyFunction import ContiEnergy
from MTMath.sympyToCpp import (
    generate_cpp_energy_density_code,
    generate_cpp_stress_function_code,
)


def generate_cpp_code():
    # Get symbolic expressions from ContiEnergy
    phi_func, div_phi_dict, _ = ContiEnergy.symbolic_potential()

    # Generate the code
    energy_code = generate_cpp_energy_density_code(phi_func)
    stress_code = generate_cpp_stress_function_code(div_phi_dict)

    # Output results
    print(energy_code)
    print("\n")
    print(stress_code)


def plot_moduli_simple_shear(
    gamma_min=-1.0, gamma_max=1.0, num=401, loops=300, theta_deg=0.0
):
    gamma = np.linspace(gamma_min, gamma_max, num)

    # Simple shear F = [[1, γ], [0, 1]]; C = F^T F
    F = np.zeros((gamma.size, 2, 2), dtype=float)
    F[:, 0, 0] = 1.0
    F[:, 0, 1] = gamma
    F[:, 1, 1] = 1.0

    mu, lam = ContiEnergy.moduli_at_F(F, loops=loops, eulerian=True)

    # Compute Eulerian acoustic tensor determinant for a single direction
    # tangent_elasticity_tensor already returns Eulerian a when eulerian=True
    a = ContiEnergy.tangent_elasticity_tensor(F, loops=loops, eulerian=True)
    theta = np.deg2rad(theta_deg)
    n = np.tile(np.array([np.cos(theta), np.sin(theta)]), (gamma.size, 1))
    q = np.einsum("...j,...l,...ijkl->...ik", n, n, a)
    det_q = np.linalg.det(q)

    # Print gamma values where mu crosses zero (linear interpolation).
    zeros = []
    sign_changes = np.where(np.sign(mu[:-1]) * np.sign(mu[1:]) <= 0)[0]
    if sign_changes.size == 0:
        print("No zero crossing for mu in the specified gamma range.")
    else:
        for i in sign_changes:
            g0, g1 = gamma[i], gamma[i + 1]
            m0, m1 = mu[i], mu[i + 1]
            if m0 == 0:
                zeros.append(g0)
            elif m1 == 0:
                zeros.append(g1)
            elif m0 * m1 < 0:
                # Linear interpolation for the root.
                zeros.append(g0 - m0 * (g1 - g0) / (m1 - m0))
        print("mu crosses zero at gamma ≈", ", ".join(f"{z:.6g}" for z in zeros))

    # Zero crossings for det(q)
    det_zeros = []
    det_sign_changes = np.where(np.sign(det_q[:-1]) * np.sign(det_q[1:]) <= 0)[0]
    if det_sign_changes.size == 0:
        print("No zero crossing for det(q) in the specified gamma range.")
    else:
        for i in det_sign_changes:
            g0, g1 = gamma[i], gamma[i + 1]
            d0, d1 = det_q[i], det_q[i + 1]
            if d0 == 0:
                det_zeros.append(g0)
            elif d1 == 0:
                det_zeros.append(g1)
            elif d0 * d1 < 0:
                det_zeros.append(g0 - d0 * (g1 - g0) / (d1 - d0))
        print(
            "det(q) crosses zero at gamma ≈",
            ", ".join(f"{z:.6g}" for z in det_zeros),
        )

    # Values at gamma = 0
    mu0, lam0 = ContiEnergy.moduli_at_F(np.eye(2), loops=loops, eulerian=True)
    mu0 = float(mu0)
    lam0 = float(lam0)
    A0 = ContiEnergy.tangent_elasticity_tensor(np.eye(2), loops=loops, eulerian=False)
    A0_voigt = ContiEnergy._voigt_from_A(A0)
    print("Voigt matrix at gamma=0:\n", A0_voigt)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax2 = ax.twinx()

    (mu_line,) = ax.plot(gamma, mu, label=r"$\mu$ (shear modulus)")
    ax.plot(gamma, lam, label=r"$\lambda$ (Lamé 1st parameter)")

    (det_line,) = ax2.plot(
        gamma,
        det_q,
        linestyle="--",
        color="tab:green",
        label=rf"$\det q(n)$, $\theta$={theta_deg:.0f}°",
    )
    ax.axvline(0.0, color="k", lw=0.8, alpha=0.3)

    # Mark mu zero crossings
    if zeros:
        ax.scatter(
            zeros,
            [0.0] * len(zeros),
            facecolors="none",
            edgecolors=mu_line.get_color(),
            linewidths=1.2,
            s=35,
            zorder=5,
        )
        for z in zeros:
            ax.annotate(
                f"{z:.3g}",
                xy=(z, 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=mu_line.get_color(),
            )
    # Mark det(q) zero crossings
    if det_zeros:
        ax2.scatter(
            det_zeros,
            [0.0] * len(det_zeros),
            facecolors="none",
            edgecolors=det_line.get_color(),
            linewidths=1.2,
            s=35,
            zorder=5,
        )
        for z in det_zeros:
            ax2.annotate(
                f"{z:.3g}",
                xy=(z, 0.0),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=det_line.get_color(),
            )

    # Mark values at gamma = 0
    ax.scatter([0.0], [mu0], color="C0", s=30, zorder=6)
    ax.scatter([0.0], [lam0], color="C1", s=30, zorder=6)
    ax.annotate(
        f"mu(0)={mu0:.3g}",
        xy=(0.0, mu0),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=8,
        color="C0",
    )
    ax.annotate(
        f"lambda(0)={lam0:.3g}",
        xy=(0.0, lam0),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=8,
        color="C1",
    )
    ax.set_xlabel(r"$\gamma$ (simple shear)")
    ax.set_ylabel("modulus")
    ax2.set_ylabel(r"$\det q(n)$")

    # Align zero between twin axes
    y1, y2 = ax.get_ylim()
    if not (y1 <= 0 <= y2):
        y1 = min(y1, 0.0)
        y2 = max(y2, 0.0)
        ax.set_ylim(y1, y2)

    r = (0.0 - y1) / (y2 - y1) if y2 != y1 else 0.5
    dmin = float(np.min(det_q))
    dmax = float(np.max(det_q))
    if r <= 0.0:
        y1b, y2b = 0.0, max(dmax, 0.0)
    elif r >= 1.0:
        y1b, y2b = min(dmin, 0.0), 0.0
    else:
        L1 = dmax / (1.0 - r) if dmax > 0 else 0.0
        L2 = -dmin / r if dmin < 0 else 0.0
        L = max(L1, L2, 1e-12)
        y1b = -r * L
        y2b = (1.0 - r) * L
    ax2.set_ylim(y1b, y2b)
    ax.set_title("Moduli along simple shear")
    ax.legend()
    ax2.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def estimate_mu_from_stress_simple_shear(
    gamma0=0.0, delta=1e-6, beta=-1 / 4, K=4, noise=1, loops=300
):
    """
    Estimate shear-related tangent components from stress differences.
    For small perturbations in F, we have
        dP_{iJ} / dF_{kL} = A_{iJkL}.

    We measure the four shear-related components:
      A_0101 = dP01 / dF01  (upper-right shear -> upper-right Piola)
      A_0110 = dP01 / dF10  (lower-left shear  -> upper-right Piola)
      A_1001 = dP10 / dF01  (upper-right shear -> lower-left Piola)
      A_1010 = dP10 / dF10  (lower-left shear  -> lower-left Piola)

    We also report the Cauchy-stress estimate for dσ01/dF01, which matches A_0101
    at F ≈ I for this constitutive choice.
    """
    # Base F (simple shear with gamma0 on F01).
    F0 = np.array([[1.0, gamma0], [0.0, 1.0]], dtype=float)

    # --- dP01/dF01 (A_0101) ---
    Fp = F0.copy()
    Fm = F0.copy()
    Fp[0, 1] += delta
    Fm[0, 1] -= delta
    P_p = ContiEnergy.P_from_F(Fp, beta=beta, K=K, noise=noise)
    P_m = ContiEnergy.P_from_F(Fm, beta=beta, K=K, noise=noise)
    A_0101 = (P_p[0, 1] - P_m[0, 1]) / (2.0 * delta)

    # Also compute the same perturbation using Cauchy stress as a reference.
    sigma_p = ContiEnergy.cauchy_from_F(Fp, beta=beta, K=K, noise=noise)
    sigma_m = ContiEnergy.cauchy_from_F(Fm, beta=beta, K=K, noise=noise)
    dSigma01_dF01 = (sigma_p[0, 1] - sigma_m[0, 1]) / (2.0 * delta)

    # --- dP01/dF10 (A_0110) ---
    Fp = F0.copy()
    Fm = F0.copy()
    Fp[1, 0] += delta
    Fm[1, 0] -= delta
    P_p = ContiEnergy.P_from_F(Fp, beta=beta, K=K, noise=noise)
    P_m = ContiEnergy.P_from_F(Fm, beta=beta, K=K, noise=noise)
    A_0110 = (P_p[0, 1] - P_m[0, 1]) / (2.0 * delta)

    # --- dP10/dF01 (A_1001) ---
    Fp = F0.copy()
    Fm = F0.copy()
    Fp[0, 1] += delta
    Fm[0, 1] -= delta
    P_p = ContiEnergy.P_from_F(Fp, beta=beta, K=K, noise=noise)
    P_m = ContiEnergy.P_from_F(Fm, beta=beta, K=K, noise=noise)
    A_1001 = (P_p[1, 0] - P_m[1, 0]) / (2.0 * delta)

    # --- dP10/dF10 (A_1010) ---
    Fp = F0.copy()
    Fm = F0.copy()
    Fp[1, 0] += delta
    Fm[1, 0] -= delta
    P_p = ContiEnergy.P_from_F(Fp, beta=beta, K=K, noise=noise)
    P_m = ContiEnergy.P_from_F(Fm, beta=beta, K=K, noise=noise)
    A_1010 = (P_p[1, 0] - P_m[1, 0]) / (2.0 * delta)

    print(f"dSigma01/dF01 (Cauchy): {dSigma01_dF01}")
    print(f"A_0101 = dP01/dF01: {A_0101}")
    print(f"A_0110 = dP01/dF10: {A_0110}")
    print(f"A_1001 = dP10/dF01: {A_1001}")
    print(f"A_1010 = dP10/dF10: {A_1010}")

    # Compare to analytic A from tangent_elasticity_tensor (Lagrangian, mixed indices).
    A_an = ContiEnergy.tangent_elasticity_tensor(F0, loops=loops, eulerian=False)
    A_fd = np.empty((2, 2, 2, 2), dtype=float)
    for k in range(2):
        for L in range(2):
            Fp = F0.copy()
            Fm = F0.copy()
            Fp[k, L] += delta
            Fm[k, L] -= delta
            Pp = ContiEnergy.P_from_F(Fp, beta=beta, K=K, noise=noise)
            Pm = ContiEnergy.P_from_F(Fm, beta=beta, K=K, noise=noise)
            A_fd[:, :, k, L] = (Pp - Pm) / (2.0 * delta)

    print("Analytic A shear components:")
    print("  A_0101:", A_an[0, 1, 0, 1])
    print("  A_0110:", A_an[0, 1, 1, 0])
    print("  A_1001:", A_an[1, 0, 0, 1])
    print("  A_1010:", A_an[1, 0, 1, 0])
    print("Finite-difference A shear components:")
    print("  A_0101:", A_fd[0, 1, 0, 1])
    print("  A_0110:", A_fd[0, 1, 1, 0])
    print("  A_1001:", A_fd[1, 0, 0, 1])
    print("  A_1010:", A_fd[1, 0, 1, 0])
    print("Max |A_an - A_fd|:", np.max(np.abs(A_an - A_fd)))


def print_tangent_elasticity_matrix(F=None, loops=300, eulerian=True, A=None):
    """
    Print A_{ijKL} as a 4x4 matrix where (i,j) select the 2x2 block and
    (K,L) select the entry inside that block.
    Row = 2*i + K, Col = 2*j + L.
    """
    if A is None:
        if F is None:
            F = np.eye(2)
        A = ContiEnergy.tangent_elasticity_tensor(F, loops=loops, eulerian=eulerian)
    if A.shape != (2, 2, 2, 2):
        raise ValueError(f"Expected A shape (2,2,2,2), got {A.shape}")
    is_numeric = np.issubdtype(np.asarray(A).dtype, np.number)
    mat = np.zeros((4, 4), dtype=float if is_numeric else object)
    for i in range(2):
        for j in range(2):
            for K in range(2):
                for L in range(2):
                    row = 2 * i + K
                    col = 2 * j + L
                    mat[row, col] = A[i, j, K, L]
    if is_numeric:
        fmt = {"float_kind": lambda x: f"{x:.2f}"}
    else:
        fmt = {"all": lambda x: str(x)}
    print("Tangent elasticity tensor (4x4), block ordering by (i,j) then (K,L):")
    print(np.array2string(mat, formatter=fmt))
    return A


def plot_det_q_vs_theta(
    gammas=1.36961,
    num=400,
    theta_min=-0.5 * np.pi,
    theta_max=0.5 * np.pi,
    loops=300,
    zero_tol=1e-10,
):
    if np.isscalar(gammas):
        gammas = [float(gammas)]

    thetas = np.linspace(theta_min, theta_max, num)

    fig, ax = plt.subplots(figsize=(7, 4))

    for gamma in gammas:
        det_q = _det_q_vs_theta_for_gamma(gamma, thetas, loops=loops)

        (line,) = ax.plot(thetas, det_q, label=rf"$\gamma$={gamma:.6g}")

        # Zero crossings
        crossings = _zero_crossings(thetas, det_q, tol=zero_tol)

        if crossings:
            ax.scatter(
                crossings,
                [0.0] * len(crossings),
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.0,
                s=35,
                zorder=5,
            )
            for t in crossings:
                deg = np.rad2deg(t)
                ax.annotate(
                    f"{deg:.1f}°",
                    xy=(t, 0.0),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=line.get_color(),
                )

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.3)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\det q(n)$")
    ax.set_title(r"$\det q(n)$ vs $\theta$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _det_q_vs_theta_for_gamma(gamma, thetas, loops=300):
    # Simple shear F = [[1, γ], [0, 1]]
    F = np.array([[1.0, gamma], [0.0, 1.0]], dtype=float)
    a = ContiEnergy.tangent_elasticity_tensor(F, loops=loops, eulerian=True)

    det_q = np.empty_like(thetas)
    for i, theta in enumerate(thetas):
        n = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        q = np.einsum("j,l,ijkl->ik", n, n, a)
        det_q[i] = np.linalg.det(q)
    return det_q


def _zero_crossings(x, y, tol=1e-10):
    y = np.asarray(y, dtype=float).copy()
    y[np.abs(y) < tol] = 0.0

    crossings = []
    for i in range(len(y) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y0 == 0 and y1 == 0:
            continue
        if y0 == 0:
            crossings.append(x0)
            continue
        if y1 == 0:
            crossings.append(x1)
            continue
        if y0 * y1 < 0:
            crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))

    if not crossings:
        return []

    # De-duplicate crossings that are numerically identical.
    crossings.sort()
    unique = [crossings[0]]
    for c in crossings[1:]:
        if abs(c - unique[-1]) > 1e-6:
            unique.append(c)
    return unique


def find_gamma_single_crossing(
    gamma_low=0.132,
    gamma_high=0.133,
    num_theta=400,
    theta_min=-0.5 * np.pi,
    theta_max=0.5 * np.pi,
    loops=300,
    tol=1e-10,
    max_iter=100,
):
    """
    Binary search within [gamma_low, gamma_high] for a gamma with exactly one
    det(q) zero crossing in theta. Assumes the crossing count increases from
    1 to 2 over the interval.
    """
    thetas = np.linspace(theta_min, theta_max, num_theta)

    def count_crossings(gamma):
        det_q = _det_q_vs_theta_for_gamma(gamma, thetas, loops=loops)
        crossings = _zero_crossings(thetas, det_q, tol=tol)
        return len(crossings), float(np.min(det_q)), float(np.max(det_q))

    low = float(gamma_low)
    high = float(gamma_high)

    count_low, dmin_low, dmax_low = count_crossings(low)
    count_high, dmin_high, dmax_high = count_crossings(high)

    if count_low != 1:
        print(
            f"Expected 1 crossing at gamma={low:.6g}, got {count_low}. "
            f"det_q range [{dmin_low:.3g}, {dmax_low:.3g}]"
        )
    if count_high != 2:
        print(
            f"Expected 2 crossings at gamma={high:.6g}, got {count_high}. "
            f"det_q range [{dmin_high:.3g}, {dmax_high:.3g}]"
        )

    if count_low == 1 and count_high == 1:
        return low
    if count_low == 2 and count_high == 2:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        count_mid, _, _ = count_crossings(mid)
        if count_mid <= 1:
            low = mid
        else:
            high = mid

    # Return the highest gamma found that still has exactly one crossing
    count_final, _, _ = count_crossings(low)
    return low if count_final == 1 else None


if __name__ == "__main__":
    print(f"F={np.eye(2)}")
    A = print_tangent_elasticity_matrix(np.eye(2), eulerian=True)
    print("Voigt")
    print(np.round(ContiEnergy._voigt_from_A(A), 2))
    print("")
    print("")
    F = np.array(((1, 0.2), (0.0, 1)))
    print(f"F={F}")
    A = print_tangent_elasticity_matrix(F, eulerian=True)
    print("Voigt")
    print(np.round(ContiEnergy._voigt_from_A(A), 2))
    print("")
    index_tensor = ContiEnergy.tangent_elasticity_index_tensor()
    print_tangent_elasticity_matrix(A=index_tensor)

    plot_moduli_simple_shear()
    gamma_low = 0.132
    gamma_high = 0.133
    gamma_single = find_gamma_single_crossing(
        gamma_low=gamma_low, gamma_high=gamma_high
    )
    if gamma_single is None:
        print("No gamma found with exactly one det(q) zero crossing in the scan range.")
        gammas = [gamma_low, gamma_high, 0.001]
    else:
        print(f"Gamma with exactly one det(q) crossing: {gamma_single:.6g}")
        gammas = [gamma_single, 0.001]
    plot_det_q_vs_theta(gammas=gammas)

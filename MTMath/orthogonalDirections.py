import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# --- basic building blocks ---


def rotation(theta):
    """2D rotation matrix R(θ)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# simple-shear generator A(θ) = R(θ)^T [[0,1],[0,0]] R(θ)
def shear_generator(theta):
    S = np.array([[0.0, 1.0], [0.0, 0.0]])
    R = rotation(theta)
    return R.T @ S @ R


# H(θ) = C A(θ) + A(θ)^T C
def tangent_H(C, theta):
    A = shear_generator(theta)
    return C @ A + A.T @ C


def frobenius_inner(H1, H2):
    """⟨H1, H2⟩_F = tr(H1^T H2)."""
    return float(np.sum(H1 * H2))


# --- symbolic / analytical tools using sympy ---


def _symbolic_inner_product_expression(C, theta0):
    """Return (t, expr(t)) where expr(t) = <H(theta0), H(t)>_F as a sympy expression.

    This function supports both numeric (numpy) and symbolic (sympy) inputs for C and theta0.
    """
    # --- build a sympy matrix Cmat ---
    if isinstance(C, sp.MatrixBase):
        # already symbolic
        Cmat = C
    else:
        # numeric: convert to 2x2 and lift entries to exact rationals
        C_np = np.asarray(C, dtype=float).reshape(2, 2)
        C11, C12, C22 = (
            sp.nsimplify(x, rational=True) for x in (C_np[0, 0], C_np[0, 1], C_np[1, 1])
        )
        Cmat = sp.Matrix([[C11, C12], [C12, C22]])

    # --- handle theta0: numeric or symbolic ---
    if isinstance(theta0, (sp.Expr, sp.Symbol)):
        theta0_sym = theta0
    else:
        theta0_sym = sp.nsimplify(theta0, rational=True)

    t = sp.symbols("t", real=True)
    c, s = sp.cos, sp.sin

    R_t = sp.Matrix([[c(t), -s(t)], [s(t), c(t)]])
    R_0 = sp.Matrix([[c(theta0_sym), -s(theta0_sym)], [s(theta0_sym), c(theta0_sym)]])
    S = sp.Matrix([[0, 1], [0, 0]])

    A_t = R_t.T * S * R_t
    A_0 = R_0.T * S * R_0

    H_t = Cmat * A_t + A_t.T * Cmat
    H_0 = Cmat * A_0 + A_0.T * Cmat

    expr = sum(H_0[i, j] * H_t[i, j] for i in range(2) for j in range(2))
    expr_simplified = sp.simplify(sp.trigsimp(expr))
    return t, expr_simplified


# --- new helper functions for symbolic trigonometric decomposition and LaTeX printing ---


def _inner_product_trig_decomposition(C, theta0):
    """
    Decompose <H(theta0), H(t)>_F into the form
        A*cos(2t) + B*sin(2t) + C0
    and return (t, A, B, C0, expr_trig).
    """
    t, expr = _symbolic_inner_product_expression(C, theta0)
    expr_trig = sp.simplify(sp.expand_trig(expr))

    # extract coefficients of cos(2t) and sin(2t)
    A = sp.simplify(sp.collect(expr_trig, sp.cos(2 * t)).coeff(sp.cos(2 * t)))
    B = sp.simplify(sp.collect(expr_trig, sp.sin(2 * t)).coeff(sp.sin(2 * t)))

    # remaining constant term (should not depend on t anymore)
    C0 = sp.simplify(expr_trig - A * sp.cos(2 * t) - B * sp.sin(2 * t))

    return t, A, B, C0, expr_trig


def theta_orth_symbolic(C, theta0):
    """
    Return the two symbolic solutions (theta_perp_1, theta_perp_2)
    of <H(theta0), H(theta)>_F = 0, using the generic formula for
        A*cos(2 theta) + B*sin(2 theta) + C0 = 0.
    """
    t, A, B, C0, _ = _inner_product_trig_decomposition(C, theta0)

    R = sp.sqrt(A**2 + B**2)
    delta = sp.atan2(B, A)  # phase such that A = R*cos(delta), B = R*sin(delta)

    # cos(2 theta - delta) = -C0/R  =>  2 theta - delta = ± arccos(-C0/R)
    xi = sp.acos(-C0 / R)

    theta1 = sp.simplify((delta + xi) / 2)
    theta2 = sp.simplify((delta - xi) / 2)

    return theta1, theta2


def print_theta_orth_latex(C, theta0, name=r"\theta_{\perp}"):
    """
    Print LaTeX expressions for the two analytical orthogonal angles.
    C and theta0 can be numeric or symbolic; for a nice closed form,
    it is recommended to pass symbolic parameters (e.g. gamma).
    """
    theta1, theta2 = theta_orth_symbolic(C, theta0)

    latex1 = sp.latex(theta1)
    latex2 = sp.latex(theta2)

    print(rf"{name}^{{(1)}} = {latex1}")
    print(rf"{name}^{{(2)}} = {latex2}")


def analytical_theta_orth(C, theta0, interval=(0.0, np.pi)):
    """Compute all analytical solutions theta in the given interval such that
    <H(theta0), H(theta)>_F = 0 using sympy.

    Returns a sorted numpy array of solutions (in radians).
    """
    a, b = float(interval[0]), float(interval[1])
    t, expr = _symbolic_inner_product_expression(C, theta0)

    # Solve analytically in the real interval [a, b]
    sol_set = sp.solveset(sp.Eq(expr, 0), t, domain=sp.Interval(a, b))

    sol_list = sorted(float(sol.evalf()) for sol in sol_set)

    return np.array(sol_list, dtype=float)


def plot_analytical_theta_orth(
    C, theta0, n_samples=2000, ax=None, label="inner product"
):
    """Plot the Frobenius inner product <H(theta0), H(theta)>_F and overlay
    the analytical orthogonal directions where it vanishes.

    Since there are (at least) two solutions in [0, pi), all of them
    are plotted as vertical dashed lines.
    """
    C = np.asarray(C, dtype=float).reshape(2, 2)
    H0 = tangent_H(C, theta0)

    thetas = np.linspace(0.0, np.pi, n_samples)
    inner_vals = np.empty_like(thetas)
    for i, th in enumerate(thetas):
        H_th = tangent_H(C, th)
        inner_vals[i] = frobenius_inner(H0, H_th)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # normalize for nicer plotting, but keep sign
    max_abs = np.max(np.abs(inner_vals))
    if max_abs > 0:
        norm_vals = inner_vals / max_abs
    else:
        norm_vals = inner_vals

    ax.plot(np.degrees(thetas), norm_vals, label=label)
    ax.axhline(0.0, linewidth=0.8)

    # analytical roots
    roots = analytical_theta_orth(C, theta0)
    print(f"nr roots: {len(roots)}")
    for k, th_root in enumerate(roots):
        ax.axvline(
            np.degrees(th_root),
            linestyle="--",
            linewidth=1.0,
            label=rf"analytical $\theta_{{\perp,{k}}}$",
        )

    ax.set_xlabel(r"$\theta$  [deg]")
    ax.set_ylabel(r"$\langle H(\theta_0), H(\theta) \rangle_F$ (normalized)")
    ax.set_title("Frobenius inner product and analytical orthogonal directions")
    ax.legend()

    return ax


# --- main routine ---


def orthogonal_simple_shear_direction(
    C, theta0, n_samples=2000, make_plot=True, ax=None, label=""
):
    """
    Given C (2x2 SPD) and a reference angle theta0 (rad),
    compute H(theta) and the Frobenius inner product
        ⟨H(theta0), H(theta)⟩_F
    as a function of theta. Then find the theta for which
    this inner product is closest to zero (orthogonal direction).
    """
    C = np.asarray(C, dtype=float).reshape(2, 2)

    # reference tangent
    H0 = tangent_H(C, theta0)

    # sample θ in [0, π)
    thetas = np.linspace(0.0, np.pi, n_samples)
    inner_vals = np.empty_like(thetas)

    for i, th in enumerate(thetas):
        H_th = tangent_H(C, th)
        inner_vals[i] = frobenius_inner(H0, H_th)

    abs_inner = np.abs(inner_vals)
    idx_sorted = np.argsort(abs_inner)
    min_index_gap = max(5, n_samples // 200)
    first_idx = idx_sorted[0]
    second_idx = None
    for idx_candidate in idx_sorted[1:]:
        if abs(idx_candidate - first_idx) >= min_index_gap:
            second_idx = idx_candidate
            break
    if second_idx is None:
        second_idx = idx_sorted[1]

    idx1, idx2 = sorted([first_idx, second_idx])
    theta1, theta2 = thetas[idx1], thetas[idx2]
    theta_perp_small = min(theta1, theta2)
    theta_perp_other = max(theta1, theta2)

    # show inner product as a function of θ
    if make_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(np.degrees(thetas), inner_vals / np.max(inner_vals), label=label)
        plt.axhline(0.0, linewidth=0.8)
        plt.xlabel(r"$\theta$  [deg]")
        plt.ylabel(r"$\langle H(\theta_0), H(\theta) \rangle_F$ (normalized)")
        plt.title("Frobenius inner product vs shear direction")
        plt.legend()
        print(f"theta0         = {theta0:.6f} rad  ({np.degrees(theta0):.3f} deg)")
        print(
            f"theta_orth_1   = {theta1:.6f} rad  ({np.degrees(theta1):.3f} deg), inner = {inner_vals[idx1]:.3e}"
        )
        print(
            f"theta_orth_2   = {theta2:.6f} rad  ({np.degrees(theta2):.3f} deg), inner = {inner_vals[idx2]:.3e}"
        )
        print(
            f"theta_chosen   = {theta_perp_small:.6f} rad  ({np.degrees(theta_perp_small):.3f} deg)"
        )
        return ax

    return theta_perp_small, thetas, inner_vals


# --- example usage ---


def demo1():
    theta0 = 0.0  # reference simple-shear direction
    ax = None
    for gamma in [0, 2, 5, 10]:
        s = np.array([[1, gamma], [0, 1]])
        C = np.eye(2)
        C_example = s.T @ C @ s
        ax = orthogonal_simple_shear_direction(
            C_example, theta0, label=rf"$\gamma={gamma}$", ax=ax
        )
    plt.tight_layout()
    plt.show()


def demo2():
    s = np.array([[1, 0.01], [0, 1]])
    C = np.eye(2)
    ortThetas = []
    n = 1000
    theta0 = 0.0
    for i in range(n):
        C = s.T @ C @ s
        theta_perp, thetas, inner_vals = orthogonal_simple_shear_direction(
            C, theta0, make_plot=False
        )
        ortThetas.append(theta_perp)
    # Show theta in degrees
    ortThetas = np.array(ortThetas) * 180.0 / np.pi
    plt.plot(range(n) * s[0, 1], ortThetas)
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"Orthogonal direction $\theta_\perp$ [deg]")
    plt.title("Evolution of orthogonal simple-shear direction under shear")
    plt.tight_layout()
    plt.show()


def demo3():
    # small example of analytical orthogonal directions for a single C
    theta0 = 0.0
    gamma = 0.01
    s = np.array([[1, gamma], [0, 1]])
    C = s.T @ np.eye(2) @ s
    plot_analytical_theta_orth(C, theta0)
    plt.tight_layout()
    plt.show()


def demo4():
    # Equivalent to demo 2, but using the analytical solution

    s = np.array([[1, 1], [0, 1]])
    C = np.eye(2)
    ortThetas = []
    n = 10
    theta0 = 0.0
    for i in range(n):
        C = s.T @ C @ s
        roots = analytical_theta_orth(C, theta0)
        ortThetas.append(roots[0])
    # Show theta in degrees
    ortThetas = np.array(ortThetas) * 180.0 / np.pi
    plt.plot(range(n) * s[0, 1], ortThetas)
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"Orthogonal direction $\theta_\perp$ [deg]")
    plt.title("Evolution of orthogonal simple-shear direction under shear")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # demo1()
    # demo2()
    # demo3()
    # demo4()
    gamma, theta = sp.symbols("gamma, theta", real=True)
    s_sym = sp.Matrix([[1, gamma], [0, 1]])
    C_sym = s_sym.T * sp.eye(2) * s_sym

    print_theta_orth_latex(C_sym, theta)

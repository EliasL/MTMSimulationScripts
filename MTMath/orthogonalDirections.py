import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

_DEFAULT_SOLUTION_PATH = Path(__file__).with_name("orthogonal_shear_solutions.pkl")


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

    # 'theta_ort' is the symbolic variable representing the UNKNOWN angle we want to solve for.
    # This is the θ such that <H(theta0), H(θ)>_F = 0.
    # theta0_sym is the GIVEN reference angle; theta_ort is the angle we search over.
    theta_ort = sp.symbols("theta", real=True)
    c, s = sp.cos, sp.sin

    R_t = sp.Matrix([[c(theta_ort), -s(theta_ort)], [s(theta_ort), c(theta_ort)]])
    R_0 = sp.Matrix([[c(theta0_sym), -s(theta0_sym)], [s(theta0_sym), c(theta0_sym)]])
    S = sp.Matrix([[0, 1], [0, 0]])

    A_t = R_t.T * S * R_t
    A_0 = R_0.T * S * R_0

    H_t = Cmat * A_t + A_t.T * Cmat
    H_0 = Cmat * A_0 + A_0.T * Cmat

    # expr(t) is the Frobenius inner product <H(theta0), H(t)>_F.
    # This is a symbolic function of t. Later we solve expr(t) = 0 for t.
    expr = sum(H_0[i, j] * H_t[i, j] for i in range(2) for j in range(2))
    expr_simplified = sp.simplify(sp.trigsimp(expr))
    # We return both:
    #   - t : the symbolic variable (the unknown angle)
    #   - expr_simplified(t) : the inner product as a symbolic expression of t
    # The caller will solve expr_simplified(t) = 0 for t in a chosen interval.
    return theta_ort, expr_simplified


def analytical_theta_orth(C, theta0):
    """Compute all analytical solutions theta in the given interval such that
    <H(theta0), H(theta)>_F = 0 using sympy.

    Returns a sorted numpy array of solutions (in radians).
    """
    theta_ort, expr = _symbolic_inner_product_expression(C, theta0)
    # Solve analytically in the real interval [a, b]
    print("Solving...")
    sol_set = sp.solve(sp.Eq(expr, 0), theta_ort)
    print("Done")
    return sol_set


def get_gamma_sol(theta0=0, sol_nr=0):
    gamma = sp.symbols("gamma", real=True)

    s_sym = sp.Matrix([[1, gamma], [0, 1]])
    C_sym = s_sym.T * sp.eye(2) * s_sym

    # There are always two solutions
    sols = analytical_theta_orth(C_sym, theta0)
    assert 0 <= sol_nr < 2, "There are only two solutions"
    sol = sols[sol_nr]
    return sol, gamma


def get_gamma_func(theta0=0, sol_nr=0):
    sol, gamma = get_gamma_sol(theta0, sol_nr)
    return sp.lambdify((gamma), sol, "numpy")


def make_numpy_function(theta0=0, sol_nr=0):
    """Print pure Python code for theta(gamma) using NumPy functions.

    This does *not* return a function. It just prints a snippet you can copy-paste.
    """
    sol, gamma = get_gamma_sol(theta0, sol_nr)

    from sympy.printing import pycode

    code = pycode(sol)

    # Use the conventional NumPy alias `np` in the generated code
    code = code.replace("numpy.", "np.").replace("math.", "np.")

    print("def theta_func(gamma):")
    print(f"    return {code}")


def orth_theta_ref0(gamma):
    return -np.atan(
        gamma - np.sqrt(gamma**4 + 3 * gamma**2 + 1) / np.sqrt(gamma**2 + 1)
    )


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
    y = np.linspace(0, 3, 1000)
    f = get_gamma_func(0, sol_nr=1)
    f2 = get_gamma_func(0, sol_nr=0)
    f3 = get_gamma_func(sp.pi / 2, sol_nr=1)
    f4 = get_gamma_func(sp.pi / 2, sol_nr=0)
    plt.plot(y, np.degrees(f(y)))
    plt.plot(y, np.degrees(f2(y)))
    plt.plot(y, np.degrees(f3(y)))
    plt.plot(y, np.degrees(f4(y)))
    plt.show()


def saveGeneralSolution(path: str | Path | None = None):
    if path is None:
        path = _DEFAULT_SOLUTION_PATH
    else:
        path = Path(path)
    # Example 1: print the symbolic inner product expression as a function
    # of theta (and gamma), with theta0 left as a parameter.
    gamma, theta0 = sp.symbols("gamma, theta_0", real=True)
    theta0 = sp.Float(0)  # sp.pi / 2
    s_sym = sp.Matrix([[1, gamma], [0, 1]])
    C_sym = s_sym.T * sp.eye(2) * s_sym

    theta_sym, expr = _symbolic_inner_product_expression(C_sym, theta0)
    print(
        "\nInner product <H(theta0), H(theta)>_F (symbolic, in terms of gamma, theta0, theta):"
    )
    sp.pprint(expr)

    # Solve analytically for theta such that <H(theta0), H(theta)>_F = 0.
    # Here gamma and theta0 are treated as parameters, and theta_sym is the unknown.
    print("\nSolving <H(theta0), H(theta)>_F = 0 for theta...")
    theta_solutions = sp.solve(sp.Eq(expr, 0), theta_sym)

    print("\nSymbolic solutions theta(gamma, theta0) from sp.solve(expr, theta):")
    for k, sol in enumerate(theta_solutions):
        print(f"Solution {k}:")
        sp.pprint(sol)
        print(sp.latex(sol))  # prints LaTeX source
        print()

    # Persist solutions and related data so they can be re-used later without
    # re-running the expensive symbolic solve.
    data_to_save = {
        "expr": expr,
        "theta_solutions": theta_solutions,
        "symbols": {
            "gamma": gamma,
            "theta0": theta0,
            "theta_sym": theta_sym,
        },
    }
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved symbolic data (expr, solutions, symbols) to: {path}")


def loadGeneralSolution(path: str | Path | None = None):
    """Load the previously saved symbolic expression and solutions.

    Returns a dictionary with keys:
        - 'expr'            : sympy expression <H(theta0), H(theta)>_F
        - 'theta_solutions' : list of sympy expressions theta_i(gamma, theta0)
        - 'symbols'         : dict with 'gamma', 'theta0', 'theta_sym' symbols
    """
    if path is None:
        path = _DEFAULT_SOLUTION_PATH
    else:
        path = Path(path)

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_theta_branch_from_file(sol_index: int = 0, path: str | Path | None = None):
    """Return a lambdified branch theta(gamma, theta0) loaded from a saved file.

    Parameters
    ----------
    sol_index : int
        Index of the solution branch in the saved 'theta_solutions' list.
    path : str or Path or None
        Optional path to the pickle file created by saveGeneralSolution().
        If None, uses the default path next to this module.

    Returns
    -------
    callable
        A numpy-compatible function theta(gamma, theta0).
    """
    data = loadGeneralSolution(path)
    theta_solutions = data["theta_solutions"]
    if not (0 <= sol_index < len(theta_solutions)):
        raise IndexError(
            f"sol_index={sol_index} out of range for {len(theta_solutions)} solutions"
        )

    # Recreate the symbols with the same names for clarity, even though the
    # expressions already carry their own Symbol objects.
    gamma = sp.symbols("gamma", real=True)
    theta0 = sp.symbols("theta_0", real=True)

    theta_expr = theta_solutions[sol_index]
    return sp.lambdify((gamma, theta0), theta_expr, "numpy")


if __name__ == "__main__":
    # demo1()
    # demo3()
    saveGeneralSolution()
    # make_numpy_function()

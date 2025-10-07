# Demonstration: reference triangle, deformed triangle, and computing F from Eq. (34)

import numpy as np
import matplotlib.pyplot as plt


# --- Helpers to compare formulations of F ---


def F_edge(X, u):
    """Your edge-based formula (permutation-invariant when permuted consistently)."""
    I = np.eye(2)
    A = np.column_stack([X[1] - X[0], X[2] - X[0]])
    dU = np.column_stack([u[1] - u[0], u[2] - u[0]])
    return I + dU @ np.linalg.inv(A)


def gradN_ccw(X):
    """Gradients of linear shape functions in physical coords for CCW reference order.
    Returns a 2x3 matrix whose columns are [∇N1, ∇N2, ∇N3].
    """
    # Fixed reference gradients for the unit reference triangle (CCW):
    # N1 = 1 - ξ - η, N2 = ξ, N3 = η  =>  ∇^hat N1=[-1,-1], ∇^hat N2=[1,0], ∇^hat N3=[0,1]
    Ghat = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])  # shape (2,3)
    A = np.column_stack([X[1] - X[0], X[2] - X[0]])  # (2,2)
    # A is 2x2; Ghat is 2x3; result is 2x3 (columns are ∇N1, ∇N2, ∇N3)
    return np.linalg.inv(A).T @ Ghat  # (2,3)


def F_ref_with_mismatch_x(X, x, perm):
    """Reference-element F using current positions x with an intentional connectivity/gradient mismatch.

    Uses CCW gradients computed from X, but associates node positions via `perm`.
    This mirrors feeding a CW connectivity while retaining CCW gradients, which
    yields an orientation-reversing mapping (det F < 0) for the unit triangle when
    perm=[0,2,1].
    """
    G = gradN_ccw(X)  # 2x3 grads tied to CCW labels (1,2,3)
    # Since G is calculated assuming labels 1,2,3,
    # it breaks if perm is not the assumed 1,2,3
    x_perm = x[perm]  # mismatch: permute nodal *positions*, not the grads
    # F = sum_a x_a ⊗ ∇N_a (use a simple loop for clarity)
    F = np.zeros((2, 2))
    for a in range(3):
        na = G[:, a]  # gradient column for node a
        xa = x_perm[a]  # position of node a
        F += np.outer(xa, na)  # x_a ⊗ ∇N_a
    return F


def show_det(label, F):
    print(f"{label}:\nF =\n{F}\n det(F) = {np.linalg.det(F): .6f}\n")


def show(X, u):
    x = X + u

    # --- 3) Compute F using the constant-strain triangle formula (Eq. 34) ---
    I = np.eye(2)

    # Build the 2x2 edge matrix in the reference config: [X2 - X1, X3 - X1]
    A = np.column_stack([X[1] - X[0], X[2] - X[0]])  # shape (2,2)

    # Build the 2x2 displacement difference matrix: [u2 - u1, u3 - u1]
    dU = np.column_stack([u[1] - u[0], u[2] - u[0]])  # shape (2,2)

    F_est = I + dU @ np.linalg.inv(A)

    print("\nComputed deformation gradient F (from Eq. 34) =\n", F_est)

    # --- 5) Plot reference and current triangles ---
    fig, ax = plt.subplots(figsize=(5, 5))

    # reference triangle
    ax.plot(
        [X[0, 0], X[1, 0], X[2, 0], X[0, 0]],
        [X[0, 1], X[1, 1], X[2, 1], X[0, 1]],
        marker="o",
        label="Reference",
    )

    # current triangle
    ax.plot(
        [x[0, 0], x[1, 0], x[2, 0], x[0, 0]],
        [x[0, 1], x[1, 1], x[2, 1], x[0, 1]],
        marker="o",
        label="Current",
    )

    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    # # Example 1
    # X1 = np.array([0, 0])
    # X2 = np.array([1, 0])
    # X3 = np.array([0, 1])
    # X = np.stack([X1, X2, X3], axis=0)

    # u1 = np.array([0, 0])
    # u2 = np.array([0, 0.0])
    # u3 = np.array([1, 0])
    # u = np.stack([u1, u2, u3], axis=0)
    # show(X, u)

    # # Example 2
    # X1 = np.array([0, 0])
    # X2 = np.array([0, 1.0])
    # X3 = np.array([1, 1])
    # X = np.stack([X1, X2, X3], axis=0)

    # u1 = np.array([0, 0])
    # u2 = np.array([1, 0.0])
    # u3 = np.array([1, 0])
    # u = np.stack([u1, u2, u3], axis=0)
    # show(X, u)

    # Example 3
    X1 = np.array([0, 0])
    X2 = np.array([1, 0])
    X3 = np.array([0, 1])
    X = np.stack([X3, X2, X1], axis=0)

    u1 = np.array([1, 0])
    u2 = np.array([-1, 1])
    u3 = np.array([0, -1])
    u = np.stack([u3, u2, u1], axis=0)
    # show(X, u)

    # Demo A: Consistent node relabeling (permutation) => same F
    print("DEMO 1")
    F0 = F_edge(X, u)
    show_det("Edge-based F (original order)", F0)

    perm_swap = [0, 2, 1]  # swap nodes 2 and 3 (CW vs CCW)
    X_sw = X[perm_swap]
    u_sw = u[perm_swap]
    F1 = F_edge(X_sw, u_sw)
    show_det("Edge-based F after consistent swap (should match)", F1)

    # Demo B: Mismatched connectivity vs gradients => orientation flip
    print("DEMO 2")
    x_demo = X + u
    F_correct = F_ref_with_mismatch_x(
        X, x_demo, perm=[0, 1, 2]
    )  # no mismatch => identity
    show_det("Reference-element F (correct pairing)", F_correct)

    # Now mismatch: swap nodal positions but keep CCW gradients => orientation reversal
    F_flip = F_ref_with_mismatch_x(X, x_demo, perm=[0, 2, 1])
    show_det("Reference-element F with mismatched connectivity", F_flip)

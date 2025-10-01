# Demonstration: reference triangle, deformed triangle, and computing F from Eq. (34)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show(X, u):
    X1, X2, X3 = X
    x = X + u

    # --- 3) Compute F using the constant-strain triangle formula (Eq. 34) ---
    I = np.eye(2)

    # Build the 2x2 edge matrix in the reference config: [X2 - X1, X3 - X1]
    A = np.column_stack([X2 - X1, X3 - X1])  # shape (2,2)

    # Build the 2x2 displacement difference matrix: [u2 - u1, u3 - u1]
    dU = np.column_stack([u[1] - u[0], u[2] - u[0]])  # shape (2,2)

    F_est = I + dU @ np.linalg.inv(A)

    # --- 4) Display coordinates and F ---
    df = pd.DataFrame(
        {
            "Node": ["1", "2", "3"],
            "X1": X[:, 0],
            "X2": X[:, 1],
            "x1": x[:, 0],
            "x2": x[:, 1],
            "u1": u[:, 0],
            "u2": u[:, 1],
        }
    )

    print("\nComputed deformation gradient F (from Eq. 34) =\n", F_est)

    # --- 5) Plot reference and current triangles ---
    fig, ax = plt.subplots(figsize=(5, 5))

    # reference triangle
    ax.plot(
        [X1[0], X2[0], X3[0], X1[0]],
        [X1[1], X2[1], X3[1], X1[1]],
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
    # Example 1
    X1 = np.array([0, 0])
    X2 = np.array([1, 0])
    X3 = np.array([0, 1])
    X = np.stack([X1, X2, X3], axis=0)

    u1 = np.array([0, 0])
    u2 = np.array([0, 0.0])
    u3 = np.array([1, 0])
    u = np.stack([u1, u2, u3], axis=0)
    show(X, u)

    # Example 2
    X1 = np.array([0, 0])
    X2 = np.array([0, 1.0])
    X3 = np.array([1, 1])
    X = np.stack([X1, X2, X3], axis=0)

    u1 = np.array([0, 0])
    u2 = np.array([1, 0.0])
    u3 = np.array([1, 0])
    u = np.stack([u1, u2, u3], axis=0)
    show(X, u)

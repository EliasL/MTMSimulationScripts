import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Define a symmetric positive definite matrix A and vector b
A = np.array([[4, 1], [1, 2]])
b = np.array([0, 0])  # Minimum at the origin


# Quadratic function: f(x) = 1/2 x^T A x
def f(x):
    return 0.5 * x.T @ A @ x


# Gradient of the function
def grad_f(x):
    return A @ x


def gradient_descent(x0, steps=20, lr=0.1):
    x = x0.copy()
    path = [x.copy()]
    for _ in range(steps):
        grad = grad_f(x)
        x -= lr * grad
        path.append(x.copy())
    return np.array(path)


def conjugate_gradient(x0, steps=20):
    path = []

    def callback(xk):
        path.append(np.copy(xk))

    result = minimize(
        f,
        x0,
        method="CG",
        jac=grad_f,
        callback=callback,
        options={"maxiter": steps, "disp": False},
    )
    return np.array([x0] + path)


def cg2(x0, steps=20):
    x = x0.copy()
    r = -grad_f(x)  # Since b = 0, the residual is -A @ x
    d = r.copy()  # Initial search direction
    path = [x.copy()]

    for _ in range(steps):
        Ad = A @ d
        rTr = r @ r
        alpha = rTr / (d @ Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad

        path.append(x.copy())

        if np.linalg.norm(r_new) < 1e-10:
            break

        beta = (r_new @ r_new) / rTr
        d = r_new + beta * d
        r = r_new

    return np.array(path)


# Plotting
x0 = np.array([0.5, 0.7])
steps = 5
gd_path = gradient_descent(x0, lr=0.3, steps=steps)
# cg_path = conjugate_gradient(x0, steps=steps)
cg_path = cg2(x0, steps=steps)

x_vals = np.linspace(-1, 1, 500)
y_vals = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[f(np.array([x, y])) for x in x_vals] for y in y_vals])

plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=10, colors="blue", linewidths=0.5)
plt.plot(
    cg_path[:, 0],
    cg_path[:, 1],
    "-",
    color="#9456BD",
    linewidth=3,
    label="Conjugate Gradient",
)
plt.plot(
    cg_path[:, 0],
    cg_path[:, 1],
    "x",
    color="#9456BD",
    markersize=8,
)
plt.plot(gd_path[:, 0], gd_path[:, 1], "r--", label="Gradient Descent")
plt.plot(gd_path[:, 0], gd_path[:, 1], "ro", markersize=5, fillstyle="none")
plt.scatter(*x0, color="#80acb4", s=100, zorder=5)
plt.text(x0[0] - 0.1, x0[1] + 0.1, r"$x_0$", fontsize=16)
plt.legend()
plt.tight_layout()
plt.axis("equal")
plt.savefig("MTMath/MinimizationProblem/plots/cgVsGd.pdf")
plt.show()

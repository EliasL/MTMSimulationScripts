import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp
from fire import optimize_fire2
from collections import Counter
import os

# Set larger sizes for all elements
scale = 2
plt.rcParams.update(
    {
        "font.size": scale * 14,  # Adjust font size
        "axes.titlesize": scale * 16,  # Adjust axis title size
        "axes.labelsize": scale * 14,  # Adjust axis label size
        "xtick.labelsize": scale * 12,  # Adjust x-axis tick label size
        "ytick.labelsize": scale * 12,  # Adjust y-axis tick label size
        "lines.linewidth": scale * 2,  # Adjust line width
        "axes.linewidth": scale * 1.5,  # Adjust axes line width
        "lines.markersize": scale * 6,  # Adjust marker size
    }
)


# Symbolic definitions using sympy
def define_hill_with_hole():
    X, Y = sp.symbols("X Y")
    Z = (
        -sp.exp(-0.001 * (X**2 + (Y + 10) ** 2)) * 20
        - sp.exp(-0.1 * ((X - 2) ** 2 + (Y + 1) ** 2))
        - sp.exp(-0.1 * ((X - 3) ** 2 + (Y - 10) ** 2))
    )
    return X, Y, Z


# Gradient calculation
def calculate_gradient(Z, X, Y):
    grad_Z = [sp.diff(Z, var) for var in (X, Y)]
    return grad_Z


# Convert symbolic expressions to numerical functions
def lambdify_functions(X, Y, Z, grad_Z):
    f_func = sp.lambdify((X, Y), Z, "numpy")
    df_func = [sp.lambdify((X, Y), grad, "numpy") for grad in grad_Z]
    return f_func, df_func


def f(x, f_func):
    X, Y = x
    return f_func(X, Y)


def df(x, df_func):
    X, Y = x
    return np.array([grad(X, Y) for grad in df_func])


# Search for local minima
def find_minima(x_range, y_range, f_func, df_func):
    minima = []
    for x in x_range:
        for y in y_range:
            result = minimize(
                lambda x: f(x, f_func),
                [x, y],
                jac=lambda x: df(x, df_func),
                method="BFGS",
            )
            if result.success and is_new_minimum(result.x, minima):
                minima.append(result.x)
    return np.array(minima)


# Check for new minimum
def is_new_minimum(point, minima, tol=1e-2):
    return not any(
        np.linalg.norm(np.array(point) - np.array(min_point)) < tol
        for min_point in minima
    )


# Optimization callbacks and paths
def run_optimizations(x0s, f_func, df_func, tol=1e-5):
    FIRE_paths, LBFGS_paths, CG_paths = [], [], []
    Fire_nit, LBFGS_nit, CG_nit = [], [], []
    LBFGS_nfev, CG_nfev = [], []

    for x0 in x0s:
        # Conjugate Gradient algorithm
        CG_path = []
        result_cg = minimize(
            lambda x: f(x, f_func),
            x0,
            method="CG",
            jac=df_func,
            tol=tol,
            callback=lambda xk: CG_path.append(xk.copy()),
        )
        CG_paths.append(np.array([x0] + CG_path))
        CG_nit.append(result_cg.nit)
        CG_nfev.append(result_cg.nfev)

        # LBFGS algorithm
        LBFGS_path = []
        result_lbfgs = minimize(
            lambda x: f(x, f_func),
            x0,
            method="L-BFGS-B",
            jac=df_func,
            tol=tol,
            callback=lambda xk: LBFGS_path.append(xk.copy()),
        )
        LBFGS_paths.append(np.array([x0] + LBFGS_path))
        LBFGS_nit.append(result_lbfgs.nit)
        LBFGS_nfev.append(result_lbfgs.nfev)

        # FIRE algorithm
        result_fire = optimize_fire2(
            x0,
            lambda x, params=None: f(x, f_func),
            lambda x, params=None: df(x, df_func),
            None,
            atol=tol,
            dt=1,
        )
        x_opt, f_opt, nit, path = result_fire
        FIRE_paths.append(np.array(path))
        Fire_nit.append(nit)

    return {
        "FIRE": {"paths": FIRE_paths, "nit": Fire_nit, "nfev": Fire_nit},
        "CG": {"paths": CG_paths, "nit": CG_nit, "nfev": CG_nfev},
        "LBFGS": {"paths": LBFGS_paths, "nit": LBFGS_nit, "nfev": LBFGS_nfev},
    }


unique_labels = set()


# Function to safely add a label to the legend
def safe_add_label(label):
    if label not in unique_labels:
        unique_labels.add(label)
        return label
    return ""


def scatter(ax, path, onlyStart=False, **kwargs):
    if tuple(np.round(path[-1]).astype(int)) == (2, -3):
        kwargs["alpha"] = 0.8
        kwargs["label"] = safe_add_label(kwargs["label"])
    else:
        kwargs["color"] = (0.1, 0.1, 0.1, 0.1)
        del kwargs["label"]

    if kwargs["marker"] == "o":
        kwargs["facecolors"] = "none"
        kwargs["edgecolors"] = kwargs.pop("color", "grey")
        kwargs["linewidths"] = 3  # Increase the edge line width here
    if onlyStart:
        ax.scatter(path[0, 0], path[0, 1], **kwargs)
    else:
        ax.scatter(path[:, 0], path[:, 1], **kwargs)


def plot(ax, path, **kwargs):
    ax.plot(path[:, 0], path[:, 1], **kwargs)


# Plotting functions
def plot_results(X, Y, Z, minima, results):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")  # noqa: F841
    if len(results["FIRE"]["paths"]) < 10:
        ax.scatter(minima[:, 0], minima[:, 1], c="red", marker="x")

    # Colors and styles for different methods
    colors = {"FIRE": "blue", "LBFGS": "red", "CG": "orange"}
    markers = {"FIRE": "x", "LBFGS": "+", "CG": "o"}
    styles = {"FIRE": "-", "LBFGS": "--", "CG": "-."}

    # Plot the paths for each optimization method
    for method, data in results.items():
        paths = data["paths"]
        for path in paths:
            if len(paths) < 10:
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    styles[method],
                    label=method,
                    color=colors[method],
                    marker="o",
                )
            else:
                scatter(
                    ax,
                    path,
                    onlyStart=True,
                    color=colors[method],
                    marker=markers[method],
                    label=method,
                )

    # Add a color bar for the contour plot
    # fig.colorbar(contour, ax=ax, label="Height")

    # Handle legend and plot aesthetics
    # Assuming you have a plotting object ax
    handles, labels = ax.get_legend_handles_labels()
    order = ["FIRE", "CG", "LBFGS"]  # Define custom order

    # Create a dictionary mapping labels to handles
    by_label = dict(zip(labels, handles))

    # Reorder handles using the predefined order
    sorted_handles = [by_label[label] for label in order if label in by_label]
    # Set the legend with sorted handles
    leg = ax.legend(handles=sorted_handles, loc="lower right")
    # Make markers non-transparent
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_aspect("equal")

    # Layout and save figure
    plt.tight_layout()
    script_dir = os.path.dirname(
        os.path.realpath("__file__")
    )  # This gets the directory of the script
    plt.savefig(os.path.join(script_dir, "optimization_paths.png"))
    plt.show()


def summarize_end_points(results):
    methods = results.keys()
    table = [["Method", "Nr it.", "Nr f-eval", "Minimum 1", "Minimum 2"]]

    for method in methods:
        end_points = [
            tuple(np.round(path[-1]).astype(int)) for path in results[method]["paths"]
        ]
        counter = Counter(end_points)
        total_nit = sum(results[method]["nit"])
        total_nfev = sum(results[method]["nfev"])

        table.append(
            [
                method,
                total_nit,
                total_nfev,
                counter.most_common(2)[0][1],
                counter.most_common(2)[1][1] if len(counter) > 1 else 0,
            ]
        )

    print(tabulate(table, headers="firstrow", tablefmt="grid"))


# Main function to organize code execution
def main():
    X, Y, Z = define_hill_with_hole()
    grad_Z = calculate_gradient(Z, X, Y)
    f_func, df_func = lambdify_functions(X, Y, Z, grad_Z)
    x_range = np.linspace(-10, 10, 30)
    y_range = np.linspace(-15, 15, 30)
    minima = find_minima(x_range, y_range, f_func, df_func)
    X, Y = np.meshgrid(x_range, y_range)
    # Flatten the mesh grid arrays and pair them into starting points
    initial_points = np.column_stack((X.ravel(), Y.ravel()))
    initial_points = np.array([[0, y] for y in [10]])  # , 8, 5, 2]])
    results = run_optimizations(initial_points, f_func, df_func)
    summarize_end_points(results)
    x = np.linspace(-16, 16, 100)
    y = np.linspace(-16, 16, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)
    plot_results(X, Y, Z, minima, results)


if __name__ == "__main__":
    main()

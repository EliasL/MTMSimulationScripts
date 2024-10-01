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


def define_six_hump_camelback():
    # Define symbolic variables
    X, Y = sp.symbols("X Y")

    # Define the Six-Hump Camelback function
    Z = (4 - 2.1 * X**2 + (X**4) / 3) * X**2 + X * Y + (-4 + 4 * Y**2) * Y**2

    # Return the symbolic variables and the function
    return X, Y, Z


def define_three_hump_camelback():
    # Define symbolic variables
    X, Y = sp.symbols("X Y")

    # Define the Six-Hump Camelback function
    Z = 2 * X**2 - 1.05 * X**4 + X**6 / 6 + X * Y + Y**2

    # Return the symbolic variables and the function
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
            dt=0.2,
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
def plot_results(X, Y, Z, minima, results, name, loc="lower left"):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")  # noqa: F841

    # Colors and styles for different methods
    colors = {"FIRE": "blue", "LBFGS": "red", "CG": "orange"}
    markers = {"FIRE": "x", "LBFGS": "+", "CG": "o"}
    styles = {"FIRE": "-", "LBFGS": "--", "CG": "-."}

    # Plot the paths for each optimization method
    for method, data in results.items() if results else []:
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

    if results is not None:
        # Set the legend with sorted handles
        leg = ax.legend(handles=sorted_handles, loc=loc)
        # Make markers non-transparent
        for lh in leg.legend_handles:
            lh.set_alpha(1)

    ax.set_aspect("equal")

    # Save the current axis limits before adding the scatter plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot the scatter points
    if results is None or len(results["FIRE"]["paths"]) < 10:
        ax.scatter(minima[:, 0], minima[:, 1], c="#228B22", marker="x", zorder=10)

    # Restore the original axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Layout and save figure
    plt.tight_layout()
    script_dir = os.path.dirname(
        os.path.realpath(__file__)
    )  # This gets the directory of the script
    path = os.path.join(script_dir, "Plots", name)
    print(f"Saving to {path}")
    plt.savefig(path)
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


def explore_hill_with_hole():
    X, Y, Z = define_hill_with_hole()
    grad_Z = calculate_gradient(Z, X, Y)
    f_func, df_func = lambdify_functions(X, Y, Z, grad_Z)
    x_range = np.linspace(-10, 10, 30)
    y_range = np.linspace(-15, 15, 30)
    minima = find_minima(x_range, y_range, f_func, df_func)
    init_X, init_Y = np.meshgrid(x_range, y_range)
    x = np.linspace(-16, 16, 100)
    y = np.linspace(-16, 16, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)

    # Flatten the mesh grid arrays and pair them into starting points
    empty = None
    initial_points_grid = np.column_stack((init_X.ravel(), init_Y.ravel()))
    initial_points_simple = np.array([[0, y] for y in [10]])  # , 8, 5, 2]])

    for initial_points, name in zip(
        [
            empty,
            initial_points_simple,
            initial_points_grid,
        ],
        [
            "Hill_eField.pdf",
            "Hill_simple.pdf",
            "Hill_grid.pdf",
        ],
    ):
        if initial_points is not None:
            results = run_optimizations(initial_points, f_func, df_func)
            summarize_end_points(results)
        else:
            results = None
        plot_results(X, Y, Z, minima, results, name)


def explore_six_hump_camel():
    X, Y, Z = define_six_hump_camelback()
    grad_Z = calculate_gradient(Z, X, Y)
    f_func, df_func = lambdify_functions(X, Y, Z, grad_Z)
    s = 1.2
    x_range = np.linspace(-3, 3, 10)
    y_range = np.linspace(-3, 3, 10)
    minima = find_minima(x_range, y_range, f_func, df_func)
    x = np.linspace(-0.2, s + 0.2, 100)
    y = np.linspace(0, s, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)

    initial_points_simple = np.array([[1.25, 0.3]])

    for initial_points, name in zip(
        [
            None,
            initial_points_simple,
        ],
        [
            "six_hump_eField.pdf",
            "six_hump_simple.pdf",
        ],
    ):
        if initial_points is not None:
            results = run_optimizations(initial_points, f_func, df_func)
            summarize_end_points(results)
        else:
            results = None
        plot_results(X, Y, Z, minima, results, name, loc="upper right")


def explore_three_hump_camel():
    X, Y, Z = define_three_hump_camelback()
    grad_Z = calculate_gradient(Z, X, Y)
    f_func, df_func = lambdify_functions(X, Y, Z, grad_Z)
    x_range = np.linspace(-3, 3, 10)
    y_range = np.linspace(-3, 3, 10)
    minima = find_minima(x_range, y_range, f_func, df_func)
    x = np.linspace(1, 2.2, 100)
    y = np.linspace(-1.5, 0.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)

    initial_points_simple = np.array([[1.25, 0.3]])

    for initial_points, name in zip(
        [
            None,
            initial_points_simple,
        ],
        [
            "three_hump_eField.pdf",
            "three_hump_simple.pdf",
        ],
    ):
        if initial_points is not None:
            results = run_optimizations(initial_points, f_func, df_func)
            summarize_end_points(results)
        else:
            results = None
        plot_results(X, Y, Z, minima, results, name, loc="upper right")


if __name__ == "__main__":
    # explore_hill_with_hole()
    # explore_six_hump_camel()
    explore_three_hump_camel()

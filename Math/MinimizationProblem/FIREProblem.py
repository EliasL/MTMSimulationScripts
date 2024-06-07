import numpy as np
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
def run_optimizations(x0s, f_func, df_func):
    FIRE_paths, LBFGS_paths = [], []
    Fire_nit, LBFGS_nit = [], []  # Initialize lists to store iterations
    LBFGS_nfev = []  # Initialize lists to store iterations

    for x0 in x0s:
        # FIRE algorithm
        result = optimize_fire2(
            x0,
            lambda x, params=None: f(x, f_func),
            lambda x, params=None: df(x, df_func),
            None,
            dt=0.01,
        )
        x_opt, f_opt, iterations, path = result
        FIRE_paths.append(np.array(path))
        Fire_nit.append(iterations)

        # LBFGS algorithm
        LBFGS_path = []
        result = minimize(
            lambda x: f(x, f_func),
            x0,
            method="L-BFGS-B",
            jac=lambda x: df(x, df_func),
            callback=lambda xk: LBFGS_path.append(xk.copy()),
        )
        LBFGS_paths.append(np.array([x0] + LBFGS_path))
        LBFGS_nit.append(
            result.nit
        )  # Extracting number of iterations directly from the result
        LBFGS_nfev.append(result.nfev)

    return FIRE_paths, LBFGS_paths, Fire_nit, LBFGS_nit, LBFGS_nfev


def scatter(ax, path, onlyStart=False, **kwargs):
    if "c" not in kwargs.keys():
        if tuple(np.round(path[-1]).astype(int)) == (2, -3):
            kwargs["c"] = "red"
        else:
            kwargs["c"] = "black"
    if onlyStart:
        ax.scatter(path[0, 0], path[0, 1], **kwargs)
    else:
        ax.scatter(path[:, 0], path[:, 1], **kwargs)


def plot(ax, path, **kwargs):
    ax.plot(path[:, 0], path[:, 1], **kwargs)


# Plotting functions
def plot_results(X, Y, Z, minima, FIRE_paths, LBFGS_paths):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")  # noqa: F841
    # ax.scatter(minima[:, 0], minima[:, 1], c='red', marker='x')

    # Initialize a set to keep track of unique labels
    unique_labels = set()

    # Function to safely add a label to the legend
    def safe_add_label(label):
        if label not in unique_labels:
            unique_labels.add(label)
            return label
        return ""

    # Plot the path of the optimization algorithm
    for FIRE_path, LBFGS_path in zip(FIRE_paths, LBFGS_paths):
        y0 = (
            f"{FIRE_path[0, 1]:.1f}"
            if FIRE_path[0, 1] % 1
            else f"{FIRE_path[0, 1]:.0f}"
        )
        # scatter(ax, FIRE_path, onlyStart=True, marker='x', label=safe_add_label(f'FIRE'))
        # scatter(ax, LBFGS_path, onlyStart=True, marker='+', label=safe_add_label(f'LBFGS'))

        plot(ax, FIRE_path, label=f"FIRE y0={y0}", c="r")
        plot(ax, LBFGS_path, linestyle="--", label=f"LBFGS y0={y0}", c="black")

    # Manually specify the order of the legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    sorted_labels = sorted(by_label.keys())
    sorted_handles = [by_label[label] for label in sorted_labels]

    ax.legend(handles=sorted_handles, loc="best")  # Adjust 'loc' as needed

    # fig.colorbar(contour, ax=ax, label='Height')
    ax.set_aspect("equal")
    plt.tight_layout()
    # Determine the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "fig.png"))
    plt.show()


def summarize_end_points(FIRE_paths, LBFGS_paths, Fire_nit, LBFGS_nit, LBFGS_nfev):
    # Extract and round the last positions for FIRE paths
    FIRE_end_points = [tuple(np.round(path[-1]).astype(int)) for path in FIRE_paths]
    # Extract and round the last positions for LBFGS paths
    LBFGS_end_points = [tuple(np.round(path[-1]).astype(int)) for path in LBFGS_paths]

    # Count unique points and their occurrences
    fire_counter = Counter(FIRE_end_points)
    lbfgs_counter = Counter(LBFGS_end_points)

    # Calculate and print the total number of iterations for FIRE
    total_Fire_nit = sum(Fire_nit)
    print(f"\nTotal number of FIRE iterations: {total_Fire_nit}")

    # Calculate and print the total number of function evaluations for LBFGS
    LBFGS_nit = sum(LBFGS_nit)
    print(f"Total number of LBFGS iterations: {LBFGS_nit}")
    # Calculate and print the total number of function evaluations for LBFGS
    total_LBFGS_nfev = sum(LBFGS_nfev)
    print(f"Total number of LBFGS function evaluations: {total_LBFGS_nfev}")

    # Print the results for FIRE path end points and their counts
    print("FIRE path end points and their counts:")
    for point, count in fire_counter.items():
        print(f"Point {point}: {count} paths")

    # Print the results for LBFGS path end points and their counts
    print("\nLBFGS path end points and their counts:")
    for point, count in lbfgs_counter.items():
        print(f"Point {point}: {count} paths")


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
    FIRE_paths, LBFGS_paths, Fire_nit, LBFGS_nit, LBFGS_nfev = run_optimizations(
        initial_points, f_func, df_func
    )
    summarize_end_points(FIRE_paths, LBFGS_paths, Fire_nit, LBFGS_nit, LBFGS_nfev)
    x = np.linspace(-16, 16, 100)
    y = np.linspace(-16, 16, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)
    plot_results(X, Y, Z, minima, FIRE_paths, LBFGS_paths)


if __name__ == "__main__":
    main()

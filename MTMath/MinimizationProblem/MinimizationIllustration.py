import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sp
from fire import optimize_fire2
from collections import Counter
import os
import matplotlib.patches as patches
import matplotlib.colors as mcolors

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

    # Define the three-Hump Camelback function
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
                tol=1e-8,
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


def add_solutions(results, f_func, df_func, tol=1e-17):
    for method, data in results.items():
        results[method]["min_energy"] = [0] * len(data["paths"])
        results[method]["min_pos"] = [[0, 0]] * len(data["paths"])
        for i, path in enumerate(data["paths"]):
            result = minimize(
                lambda x: f(x, f_func),
                path[-1],
                jac=lambda x: df(x, df_func),
                method="BFGS",
                tol=1e-8,
            )
            results[method]["min_pos"][i] = result.x
            results[method]["min_energy"][i] = f(result.x, f_func)


def run_optimizations(x0s, f_func, df_func, tol=1e-5):
    # Prepare data structures to store results
    # Define the optimization methods
    methods = ["FIRE", "CG", "L-BFGS"]

    # Initialize the results dictionary with the same structure for each method
    results = {
        method: {
            "paths": [],
            "feval_paths": [],
            "nit": [],
            "nfev": [],
            "path_energy": [],
            "f-eval_energy": [],
            "min_energy": [],
            "min_pos": [],
        }
        for method in methods
    }

    # Define the settings for the optimization algorithms
    optimizers = {
        "CG": {
            "method": "CG",
            "options": {"gtol": tol},
        },
        "L-BFGS": {
            "method": "L-BFGS",
            "options": {"gtol": tol},
        },
    }

    for x0 in x0s:
        # Loop over the optimizers that share similar structure
        for opt_name, opt_settings in optimizers.items():
            f_eval_points = []  # Track function evaluations
            path_energies = []  # Track energy at each point
            fEval_energies = []  # Track energy at each point

            # Wrapper for the objective function to track function evaluations
            def f_wrapper(x, func=f_func):
                f_eval_points.append(x.copy())
                (fEval_energies.append(func(x[0], x[1])),)  # Calculate and store energy
                return func(x[0], x[1])

            path = []  # Track optimization path

            # Perform the optimization using minimize
            result = minimize(
                lambda x: f_wrapper(x),
                x0,
                method=opt_settings["method"].replace("L-BFGS", "L-BFGS-B"),
                jac=lambda x: df(x, df_func),
                options=opt_settings["options"],
                callback=lambda xk: (
                    path.append(xk.copy()),  # Track the path
                    path_energies.append(
                        f_func(xk[0], xk[1])
                    ),  # Calculate and store energy
                ),
            )

            # Store results for this optimizer
            results[opt_name]["paths"].append(np.array([x0] + path))
            results[opt_name]["feval_paths"].append(np.array([x0] + f_eval_points))
            results[opt_name]["nit"].append(result.nit)
            results[opt_name]["nfev"].append(result.nfev)
            results[opt_name]["path_energy"].append(np.array(path_energies))
            results[opt_name]["f-eval_energy"].append(np.array(fEval_energies))

        # FIRE optimization requires separate handling
        result_fire = optimize_fire2(
            x0,
            lambda x, params=None: f(x, f_func),
            lambda x, params=None: df(x, df_func),
            None,
            atol=tol,
            dt=0.3,
        )
        x_opt, f_opt, nit, path = result_fire
        energies_fire = [f_func(p[0], p[1]) for p in path]

        results["FIRE"]["paths"].append(np.array(path))
        results["FIRE"]["feval_paths"].append(np.array(path))
        results["FIRE"]["nit"].append(nit)
        results["FIRE"]["nfev"].append(nit)
        results["FIRE"]["path_energy"].append(np.array(energies_fire))
        results["FIRE"]["f-eval_energy"].append(np.array(energies_fire))
    add_solutions(results, f_func, df_func)
    return results


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


def draw_lims_rectangle(ax, lims):
    # lims is given as ((minx, maxx), (miny, maxy))
    (minx, maxx), (miny, maxy) = lims

    # Calculate width and height of the rectangle
    width = maxx - minx
    height = maxy - miny

    # Create a rectangle patch
    rect = patches.Rectangle(
        (minx, miny),
        width,
        height,
        linewidth=1,
        edgecolor="w",
        facecolor="none",
        zorder=11,
    )

    # Add the rectangle to the Axes object
    ax.add_patch(rect)


# Function to mute a color (reduce saturation)
def mute_color(color, factor=0.5):
    """
    Takes a color and returns a muted version of it by reducing its saturation.

    Parameters:
    color: str or tuple
        The input color in any format recognizable by matplotlib (e.g., 'red', '#ff0000', (1, 0, 0)).
    factor: float
        A number between 0 and 1, where lower values reduce the saturation more.

    Returns:
    muted_color: tuple
        A muted version of the color.
    """
    # Convert color to RGB and then to HSV
    rgb = mcolors.to_rgb(color)
    hsv = mcolors.rgb_to_hsv(rgb)

    # Reduce the saturation (second value in HSV)
    hsv[1] *= factor

    # Convert back to RGB
    muted_rgb = mcolors.hsv_to_rgb(hsv)
    return muted_rgb


def normalize(v):
    # Normalize the vector
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def remove_backtracking_points(path):
    # List to store the new path after removing backtracking points
    filtered_path = [path[0]]  # Always keep the first point

    # Iterate through the path points, comparing directions
    for i in range(1, len(path) - 1):
        # Compute the direction vectors between consecutive points
        direction1 = normalize(np.array(path[i]) - np.array(path[i - 1]))
        direction2 = normalize(np.array(path[i + 1]) - np.array(path[i]))

        # Check if the directions are opposite
        if not np.allclose(direction1, -direction2, atol=1e-8):
            # If not opposite, keep the point
            filtered_path.append(path[i])

    # Always keep the last point
    filtered_path.append(path[-1])
    return np.array(filtered_path)


# Colors and styles for different method
colors = {"FIRE": "#d24646", "L-BFGS": "#008743", "CG": "#ffa701"}
muted_colors = {key: mute_color(color, factor=0.3) for key, color in colors.items()}
markers = {"FIRE": "x", "L-BFGS": "+", "CG": "o"}
styles = {"FIRE": "-", "L-BFGS": "-", "CG": "-"}


# Plotting functions
def plot_results(
    X,
    Y,
    Z,
    minima,
    results,
    name,
    ax=None,
    loc="upper right",
    lims=None,
    next_lims=None,
    alg="all",
    draw_f_evals=False,
):
    if ax is None:
        # Determine the aspect ratio from the lims
        if lims is not None:
            x_min, x_max = lims[0]
            y_min, y_max = lims[1]
            aspect_ratio = (x_max - x_min) / (y_max - y_min)  # Calculate aspect ratio
        else:
            aspect_ratio = 1  # Default to square if no limits are provided

        # Dynamically adjust figsize based on aspect ratio
        base_height = 11  # You can change this to your desired base height
        fig_width = (
            base_height * aspect_ratio
        )  # Adjust the width based on the aspect ratio
        figsize = (fig_width, base_height)  # Create the dynamic figsize
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Contour plot with limited data if zooming
    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")  # noqa: F841

    # Plot the paths for each optimization method
    for method, data in results.items() if results else []:
        if alg != "all" and alg != method:
            continue
        paths = data["paths"]
        for path in paths:
            # we want to remove overlapping points from the fire algorithm where it has taken a step backwards
            path = remove_backtracking_points(path)
            if len(paths) < 10:
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    styles[method],
                    label=method,
                    color=colors[method],
                    marker="o",
                    zorder=2,
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
        if "feval_paths" in data and method != "FIRE" and draw_f_evals:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for fPath in data["feval_paths"]:
                ax.plot(
                    fPath[:, 0],
                    fPath[:, 1],
                    styles[method],
                    label="f-evals",
                    color=muted_colors[method],
                    marker="^",
                    zorder=1,
                )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    # Add a color bar for the contour plot
    # fig.colorbar(contour, ax=ax, label="Height")

    # Handle legend and plot aesthetics
    # Assuming you have a plotting object ax
    handles, labels = ax.get_legend_handles_labels()
    order = ["FIRE", "CG", "L-BFGS"]  # Define custom order

    # Create a dictionary mapping labels to handles
    by_label = dict(zip(labels, handles))

    # Reorder handles using the predefined order
    sorted_handles = [by_label[label] for label in order if label in by_label]

    if (results is not None or alg != "all") and False:
        # Set the legend with sorted handles
        leg = ax.legend(handles=sorted_handles, loc=loc)
        # Make markers non-transparent
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    elif alg == "all" or alg in results.keys():
        ax.legend(loc=loc)
    # Use this to remove y-ticks of all but one image
    if alg in results.keys():  # and alg != "L-BFGS":
        ax.set_yticks([])

    ax.set_aspect("equal")

    # Save the current axis limits before adding the scatter plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Plot the scatter points
    if results is None or len(results["FIRE"]["paths"]) < 10:
        ax.scatter(minima[:, 0], minima[:, 1], c="#fffac5", marker="x", zorder=10)

    # Restore the original axis limits
    if lims is None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    if next_lims is not None:
        draw_lims_rectangle(ax, next_lims)

    # Layout and save figure
    plt.tight_layout()
    # This gets the directory of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    if alg != "all":
        name = name + f"_{alg}"
    path = os.path.join(script_dir, "Plots", name + ".pdf")
    # print(f"Saving to {path}")
    # plt.savefig(path, bbox_inches="tight")
    # plt.show()


def plot_energy(
    results,
    name,
    threshold=None,
    ax=None,
    itOrFeval="f-eval",
    alg="all",
    loc="lower left",
):
    if ax is None:
        aspect_ratio = 0.7  # Default to square if no limits are provided
        # Dynamically adjust figsize based on aspect ratio
        base_height = 9  # You can change this to your desired base height
        fig_width = (
            base_height * aspect_ratio
        )  # Adjust the width based on the aspect ratio
        figsize = (fig_width, base_height)  # Create the dynamic figsize
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # If the alg does not have results, we are done
    if alg not in results.keys():
        return
    # Plot the energy for each optimization method
    for method, data in results.items():
        if alg != "all" and alg != method:
            continue
        if alg == "all" or alg == "L-BFGS":
            ax.set_ylabel(r"$f(\mathbf{x}) - f(\mathbf{x^*})$")

        # Decide whether to plot against iterations or function evaluations
        if itOrFeval == "f-eval":
            # Get the energy data to plot
            energies = data["f-eval_energy"]
            ax.set_xlabel("Number of function evaluations")
        elif itOrFeval == "iterations":
            # Get the energy data to plot
            energies = data["path_energy"]
            ax.set_xlabel("Number of itterations")
        else:
            raise ValueError(f"Unknown option for itOrFeval: {itOrFeval}")
        # Plot energy vs iteration/f-eval for each path
        for i, energy in enumerate(energies):
            ax.plot(
                range(1, len(energy) + 1),  # X-axis is iteration or f-eval count
                energy - data["min_energy"][i],  # Y-axis is energy
                styles[method],  # Use predefined style for the method
                label=method,
                color=colors[method],
                marker="o",
                zorder=2,
            )
        if threshold:
            # Get the current x-axis limits
            xmin, xmax = ax.get_xlim()

            # Draw a horizontal line across the entire plot
            ax.hlines(threshold, xmin=xmin, xmax=xmax, color="b", linestyles="--")

    # Handling the legend and reordering
    handles, labels = ax.get_legend_handles_labels()
    order = ["FIRE", "CG", "L-BFGS"]  # Define custom order

    # Create a dictionary mapping labels to handles
    by_label = dict(zip(labels, handles))

    # Reorder handles using the predefined order
    sorted_handles = [by_label[label] for label in order if label in by_label]

    if alg != "all":
        # Set the legend with sorted handles
        leg = ax.legend(handles=sorted_handles, loc=loc)
        # Make markers non-transparent
        for lh in leg.legend_handles:
            lh.set_alpha(1)
    elif results is not None:
        ax.legend()

    ax.set_yscale("log")
    # Layout and save figure
    # plt.tight_layout()
    # This gets the directory of the script
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # path = os.path.join(script_dir, "Plots", name + "_energy.pdf")
    # print(f"Saving to {path}")
    # plt.savefig(path, bbox_inches="tight")
    # plt.show()
    return ax


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


def make_mesh(lims, f_func, resolution=100):
    x = np.linspace(lims[0][0], lims[0][1], resolution)
    y = np.linspace(lims[1][0], lims[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)
    return X, Y, Z


def calculate_lims(center, aspect_ratio, zoom_factor):
    if zoom_factor is None:
        return None
    x_center, y_center = center

    # Calculate the width and height of the zoomed region
    width = zoom_factor
    height = width / aspect_ratio

    # Calculate the limits based on the center point
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    return [[x_min, x_max], [y_min, y_max]]


def add_mark(ax, mark, x, y, color="black"):
    # Adding LaTeX-style bold font using \textbf{}
    ax.text(
        x,
        y,
        r"$\textbf{" + mark + "}$",  # LaTeX syntax for bold
        transform=ax.transAxes,
        fontsize=30,
        va="top",
        ha="left",
        color=color,
    )


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
            "Hill_eField",
            "Hill_simple",
            "Hill_grid",
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
    s = 2.3
    x_range = np.linspace(-3, 3, 10)
    y_range = np.linspace(-3, 3, 10)
    minima = find_minima(x_range, y_range, f_func, df_func)
    x = np.linspace(-0.2, s + 0.2, 100)
    y = np.linspace(0, s, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y), f_func)

    initial_points_simple = np.array([[1.7, 0.7]])

    for initial_points, name in zip(
        [
            None,
            initial_points_simple,
        ],
        [
            "six_hump_eField",
            "six_hump_simple",
        ],
    ):
        if initial_points is not None:
            results = run_optimizations(initial_points, f_func, df_func)
            summarize_end_points(results)
        else:
            results = None
        plot_results(X, Y, Z, minima, results, name, loc="upper right")


# MDPI Article plot
def explore_three_hump_camel():
    # Combine all algorithms or split them into different plots
    split = True

    X, Y, Z = define_three_hump_camelback()
    grad_Z = calculate_gradient(Z, X, Y)
    f_func, df_func = lambdify_functions(X, Y, Z, grad_Z)
    x_range = np.linspace(-3, 3, 10)
    y_range = np.linspace(-3, 3, 10)
    minima = find_minima(x_range, y_range, f_func, df_func)

    initial_points = np.array([[1.25, 0.3]])

    r = run_optimizations(initial_points, f_func, df_func)
    summarize_end_points(r)

    # Fixed center point and aspect ratio
    center = (1.747553, -0.873776)
    aspect_ratio = 9 / 16  # You can change this to your desired ratio

    # Define zoom levels using different zoom factors
    zoom_factors = [None, 0.05, 0.002, 0.0001, 0.00001, None]

    # Generate zoom limits for each zoom factor
    zooms = [
        calculate_lims(center, aspect_ratio, zoom_factor)
        for zoom_factor in zoom_factors
    ]
    zooms = [[[1, 2.2], [-1.5, 0.5]]] + zooms
    i = 0
    X, Y, Z = make_mesh(zooms[i], f_func)
    n = "three_hump_simple"

    w = 6
    h = 9
    # Create figure and axes for plot_results (1 row, 4 columns)
    fig_results, res_ax = plt.subplots(1, 3, figsize=(w * 3, h))

    # Create figure and axes for plot_energy (1 row, 4 columns)
    fig_energy, energy_ax = plt.subplots(1, 3, figsize=(w * 3, h))

    algorithms = ["L-BFGS", "CG", "FIRE"]
    labels = ["(a)", "(b)", "(c)"]
    if split:
        for idx, alg in enumerate(algorithms):
            # Pass each ax object to the plotting functions
            plot_results(X, Y, Z, minima, r, n, res_ax[idx], lims=zooms[i], alg=alg)
            add_mark(res_ax[idx], labels[idx], 0.03, 0.95, color="white")

            plot_energy(r, n, ax=energy_ax[idx], alg=alg)
            add_mark(energy_ax[idx], labels[idx], 0.8, 0.95)

    else:
        # Plot without splitting across multiple axes if required
        plot_results(X, Y, Z, minima, r, n, lims=zooms[i])
        plot_energy(r, n, alt=alg)

    # Layout and save figure
    fig_results.tight_layout()
    fig_energy.tight_layout()
    # This gets the directory of the script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rPath = os.path.join(script_dir, "Plots", n + "_results.pdf")
    ePath = os.path.join(script_dir, "Plots", n + "_energy.pdf")
    # sPath = os.path.join(script_dir, "Plots", n + "_stress.pdf")
    # TODO use stress as well instead of energy
    fig_results.savefig(rPath, bbox_inches="tight")
    fig_energy.savefig(ePath, bbox_inches="tight")


if __name__ == "__main__":
    # explore_hill_with_hole()
    # explore_six_hump_camel()
    explore_three_hump_camel()

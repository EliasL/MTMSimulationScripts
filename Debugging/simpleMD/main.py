#!/usr/bin/env python3
import numpy as np
from scipy.spatial.distance import pdist
import json
import os
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


def resample_curve(x, y, scale=3, kind="cubic"):
    """
    Resample (x, y) by an integer scale factor.
    If scale=1, returns the original points.
    If scale=3, returns 3× as many points, evenly spaced over [x[0], x[-1]].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if scale == 1:
        return x, y

    # build the interpolator
    f = interp1d(x, y, kind=kind, assume_sorted=True)

    # how many points total?
    n_new = len(x) * scale

    # new x‐grid from start to end
    x_new = np.linspace(x[0], x[-1], n_new)
    y_new = f(x_new)
    return x_new, y_new


# example
x = np.array([0, 1, 2, 3, 4])
y = np.sin(x)
x3, y3 = resample_curve(x, y, scale=3)
print(len(x3))  # 15


def make_key(calc_type, **params):
    parts = [calc_type] + [f"{k}={v}" for k, v in sorted(params.items())]
    return "_".join(parts)


def generate_grid(nx, ny, spacing=1.0):
    x = np.arange(nx) * spacing
    y = np.arange(ny) * spacing
    xv, yv = np.meshgrid(x, y, indexing="xy")
    return np.stack([xv.ravel(), yv.ravel()], axis=1)


# Generate a hexagonal grid of points
def generate_hex_grid(nx, ny, spacing=1.0):
    positions = []
    for j in range(ny):
        for i in range(nx):
            x = i * spacing + (spacing / 2 if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            positions.append([x, y])
    return np.array(positions)


def lennard_jones_energy(r, epsilon=1.0, sigma=1.0, K=4):
    # avoid divide-by-zero warnings (we never pass r=0 in practice)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr6 = (sigma / r) ** 6
        return K * epsilon * (sr6**2 - sr6)


def compute_energy(positions, energy_fn, cuttoff_radius, **energy_kwargs):
    # get all pairwise distances
    distances = pdist(positions)  # shape (N*(N-1)/2,)
    distances = distances[distances < cuttoff_radius]  # apply cutoff
    energies = energy_fn(distances, **energy_kwargs)
    return np.sum(energies)


def compute_energy_sparse(positions, energy_fn, cutoff_radius, **energy_kwargs):
    tree = cKDTree(positions)
    pairs = tree.query_pairs(cutoff_radius)
    pairs = np.array(list(pairs))
    pos_i = positions[pairs[:, 0]]
    pos_j = positions[pairs[:, 1]]
    distances = np.linalg.norm(pos_i - pos_j, axis=1)
    energies = energy_fn(distances, **energy_kwargs)
    return np.sum(energies)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    hex_grid = False
    show_space = False  # whether to visualize sheared configurations
    cutoff_radius = 10  # cutoff radius for LJ potential
    # shear sweep parameters
    if hex_grid:
        gamma = (4 / 3) ** (1 / 4)
    else:
        gamma = 1.0
    shear_start = 0
    shear_end = 5.0 * gamma**2
    n_steps = 100
    shear_values = np.linspace(shear_start, shear_end, n_steps)
    # energy parameters
    epsilon = 1.0  # LJ potential depth
    K = 1  # scaling factor for LJ potential
    if hex_grid:
        sigma = 1.0  #
    else:
        sigma = 1.0 / (2 ** (1 / 6))  # LJ potential distance scale

    # example usage
    # ---------------
    # create a 5×5 grid with spacing 1.2
    L_list = [8, 16, 32, 64, 128]  # grid sizes to test
    plt.figure(figsize=(6, 5))
    # create a colormap for gradual line colors
    cmap = plt.get_cmap("gray")

    # load existing results or initialize
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for idx, L in enumerate(L_list):
        # compute a color from the colormap
        color = cmap(0.7 - (idx / (len(L_list) - 1)))
        # compute dash style: gradually less dashed for larger L
        n = len(L_list)
        dash_on = 1 + idx * (10 - 1) / (n - 1)
        dash_off = 5 - idx * (5 - 0) / (n - 1)
        linestyle = (0, (dash_on, dash_off))
        if hex_grid:
            pos = generate_hex_grid(nx=L, ny=L)
        else:
            pos = generate_grid(nx=L, ny=L)
        key_params = {
            "hexagonal": hex_grid,
            "cutoff_radius": cutoff_radius,
            "L": L,
            "shear_start": shear_start,
            "shear_end": shear_end,
            "n_steps": n_steps,
            "epsilon": epsilon,
            "sigma": sigma,
            "K": K,
        }
        key_shear = make_key("lj_shear", **key_params)
        if key_shear in results:
            data = results[key_shear]
            shear_values = np.array(data["shear_values"])
            E_sheared = np.array(data["energies"])
            print(f"Loaded existing shear sweep: {key_shear}")
        else:
            # Apply a shear transformation
            # Build 2×2 shear matrices for each shear parameter
            shear_matrices = np.array(
                [[[1, s], [0, 1]] for s in shear_values]
            )  # shape (ns, 2, 2)
            # Apply each shear to all positions: result shape (ns, N, 2)
            sheared_positions = np.array([pos @ S.T for S in shear_matrices])
            # Compute Lennard-Jones energy for each sheared configuration
            print(f"Computing energies for L={L}...")
            E_sheared = np.array(
                [
                    compute_energy_sparse(
                        config,
                        lennard_jones_energy,
                        cutoff_radius=cutoff_radius,
                        epsilon=epsilon,
                        sigma=sigma,
                        K=K,
                    )
                    for config in tqdm(sheared_positions)
                ]
            )
            results[key_shear] = {
                "shear_values": shear_values.tolist(),
                "energies": E_sheared.tolist(),
            }
            print(f"Computed and saved shear sweep: {key_shear}")
            # save updated results
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        new_shear_values, E_sheared = resample_curve(
            shear_values, E_sheared, scale=3, kind="cubic"
        )
        plt.plot(
            new_shear_values / gamma**2,  # normalize shear by gamma^2
            E_sheared / L**2,
            color=color,
            linestyle=linestyle,
            label=f"L={L}",
            linewidth=2,
        )

    if hex_grid:
        plt.xlabel(r"$ (3/4)^{-1/4}\gamma$")
    else:
        plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$E(\gamma)/L^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/lj_shear_sweep.pdf", dpi=300)

    # Visualize configurations for a chosen L (e.g., L=64)
    if show_space:
        if hex_grid:
            pos = generate_hex_grid(nx=L_list[1], ny=L_list[1])
        else:
            pos = generate_grid(nx=L_list[1], ny=L_list[1])
        example_shear_values = np.linspace(
            shear_start, shear_end, 5
        )  # fewer steps for illustration
        shear_matrices = np.array([[[1, s], [0, 1]] for s in example_shear_values])
        sheared_configs = np.array([pos @ S.T for S in shear_matrices])

        fig, axs = plt.subplots(1, len(example_shear_values), figsize=(15, 3))
        for i, (ax, config, s) in enumerate(
            zip(axs, sheared_configs, example_shear_values)
        ):
            ax.scatter(config[:, 0], config[:, 1], s=1)
            ax.set_title(f"$\\gamma$={s:.2f}")
            ax.set_aspect("equal")
        plt.suptitle(f"Sheared Configurations (L={L_list[1]})")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        plt.show()

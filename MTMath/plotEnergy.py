import numpy as np
from .contiPotential import numericContiPotential
from matplotlib import pyplot as plt


def OneDPotential():
    # Load the potential and its derivatives
    phi, divPhi, divDivPhi = numericContiPotential()

    # Define size and variables
    distance = (-2, 2)
    size = 100
    shear = np.linspace(0, 1, size)

    # Create the deformation gradient tensor F for each value of shear
    F = np.array([[[1, s], [0, 1]] for s in shear])

    # Compute the right Cauchy-Green tensor C as F^T * F for each deformation gradient
    # Use matrix multiplication (@) for F.T @ F
    C = np.array([f.T @ f for f in F])

    # Extract the components from C for input into the potential function
    C_00 = C[:, 0, 0]  # First row, first column (C[0, 0])
    C_11 = C[:, 1, 1]  # Second row, second column (C[1, 1])
    C_01 = C[:, 0, 1]  # First row, second column (C[1, 0])

    # Pass the computed components to the phi function (assume constant extra arguments)
    # You may need to adjust these arguments to match the correct inputs for phi
    potential_values = phi(C_00, C_11, C_01, 1.0, 1.0, 1.0)

    # Assuming shear and potential_values are numpy arrays
    # Repeat shear and potential_values 4 times to extend the plot
    extended_potential_values = np.tile(
        potential_values, 4
    )  # Repeat potential_values 4 times

    # Plot the extended vectors
    plt.plot(np.linspace(0, 4, 100 * 4), extended_potential_values)
    plt.show()


def lagrange_reduction(C11, C22, C12, loops=600):
    for i in range(loops):
        mask1 = C12 < 0
        C12[mask1] *= -1

        mask2 = C22 < C11
        # Swap operation
        C11[mask2], C22[mask2] = C22[mask2].copy(), C11[mask2].copy()

        mask3 = 2 * C12 > C11
        C22[mask3] += C11[mask3] - 2 * C12[mask3]
        C12[mask3] -= C11[mask3]
    return C11, C22, C12


def generate_energy_grid(
    resolution=500, beta=-0.25, K=4, min_energy=3.9, max_energy=4.16
):
    # Load the potential and its derivatives (assuming this function is defined elsewhere)
    phi, divPhi, divDivPhi = numericContiPotential()

    # Define the range for x and y based on the unit circle
    radius = 1.0
    x_min, x_max = -radius, radius
    y_min, y_max = -radius, radius

    # Create the meshgrid for the x and y coordinates
    X, Y = np.meshgrid(
        np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
    )

    # Calculate the mask for points inside the unit circle
    mask = X**2 + Y**2 <= 1 - 1e-9

    # Precompute some common terms used in a, b, c12, c22, and c11 calculations
    denominator = X**2 - 2 * X + Y**2 + 1
    a = (2 * Y) / denominator
    b = -(X**2 + Y**2 - 1) / denominator

    # Avoid division by zero or near-zero by masking those values in b
    safe_b = np.where(b == 0, np.nan, b)

    # Calculate c12, c22, and c11 using vectorized operations
    C12 = a / safe_b
    C22 = 1 / safe_b
    C11 = (1 + C12**2) / C22
    C11, C22, C12 = lagrange_reduction(C11, C22, C12)

    # Initialize the energy grid with NaNs for points outside the unit circle
    energy_grid = np.full_like(X, np.nan)

    # Apply the phi function only to the points inside the unit circle
    energy_grid[mask] = phi(C11[mask], C22[mask], C12[mask], beta, K, 1)

    # Cap energies to avoid extreme values
    if min_energy is None:
        min_energy = energy_grid.min()
    if max_energy is None:
        max_energy = energy_grid.max()

    energy_grid = np.clip(energy_grid, min_energy, max_energy)
    return energy_grid

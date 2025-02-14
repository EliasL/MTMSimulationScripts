import numpy as np
import powerlaw
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)


# Generate synthetic power-law distributed data
def generate_powerlaw_data(alpha, size=10000, xmin=1e-8):
    return (np.random.pareto(alpha - 1, size) + 1) * xmin


# Generate power-law avalanche data
def generate_powerlaw_avalanche_data(alpha, size=10000, xmin=1e-8):
    increments = np.random.normal(xmin, xmin, size=size)  # Small incremental increases

    # Randomly select drop points
    drop_mask = np.random.uniform(size=size) > 0.6  # 30% chance of a drop
    drops = generate_powerlaw_data(alpha, drop_mask.sum(), xmin)

    # Apply drops, ensuring values don’t go negative
    increments[drop_mask] = -drops

    return increments


def get_only_drops(data):
    drop_mask = data < 0
    drops = -data[drop_mask]
    return drops


def get_true_data(csvPath):
    df = pd.read_csv(csvPath)
    diffs = df["Avg energy change"]
    drop_mask = diffs < 0
    drops = -diffs[drop_mask]
    return drops


# Generate data
alpha_true = 2
xmin = 1e-8
size = 10000
# data = generate_powerlaw_data(alpha_true, size, xmin)
data = generate_powerlaw_avalanche_data(alpha_true, size, xmin)
data = np.concatenate([data] * 50)
data = get_only_drops(data)
# data = get_true_data(
#     "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/macroData.csv"
# )
plt.plot(np.arange(len(data)), data)
plt.yscale("log")
plt.show()
# Fit using powerlaw library
fit = powerlaw.Fit(data)

xmin_estimated = fit.xmin
alpha_powerlaw = fit.alpha

# Print results
print(f"True alpha: {alpha_true}")
print(f"Powerlaw library estimate: {alpha_powerlaw:.3f}")
print(f"Estimated xmin: {xmin_estimated}")

# Interesting observations:
# My data from a single file gives an exponent of 2.5 (1.5?)
# When i duplicate fabricated data, the exponent drastically decreases

# Plot using powerlaw's built-in plotting function with scatter
plt.figure(figsize=(8, 6))
fig = fit.plot_ccdf(marker="o", linestyle="None", label="Empirical CCDF")

# Plot fitted power law
x_fit = np.logspace(np.log10(xmin_estimated), np.log10(max(data)), 1000)
plt.plot(
    x_fit,
    (x_fit / xmin_estimated) ** (-alpha_powerlaw + 1),
    label=f"Powerlaw Fit (alpha={alpha_powerlaw:.3f})",
    linestyle="dashed",
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("Complementary CDF")
plt.title("Power-law distributed data and estimated exponents")
plt.legend()
plt.show()

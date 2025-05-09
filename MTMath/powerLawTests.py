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
    drop_mask = np.random.uniform(size=size) > 0.7  # 40% chance of a drop
    drops = generate_powerlaw_data(alpha, drop_mask.sum(), xmin)
    # Apply drops
    increments[drop_mask] = -drops
    return increments


def get_only_drops(data):
    drop_mask = data < 0
    drops = -data[drop_mask]
    return drops


def get_true_data(csvPath):
    df = pd.read_csv(csvPath)
    diffs = df["avg_energy_change"]
    drop_mask = diffs < 0
    drops = -diffs[drop_mask]
    # e = df["avg_energy"][220000:]
    # plt.plot(e)
    # # plt.yscale("log")
    # plt.show()
    return drops


# Generate data
alpha_true = 1.5
xmin = 1e-6
size = 30000
# data = generate_powerlaw_data(alpha_true, size, xmin)
data = generate_powerlaw_avalanche_data(alpha_true, size, xmin)
# data = np.concatenate([data] * 50)
data = get_only_drops(data)
print(f"Number of drops: {len(data)}")
# data = get_true_data(
#     # "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/macroData.csv"
#     "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
# )
# Fit using powerlaw library
fit = powerlaw.Fit(data)

xmin_estimated = fit.xmin
alpha_powerlaw = fit.alpha

# Print results
print("")
print(f"True alpha: {alpha_true}")
print(f"Powerlaw library estimate: {alpha_powerlaw:.3f}")
print(f"Estimated xmin: {xmin_estimated}")
# Goodness-of-fit statistics
# Kolmogorov-Smirnov distance for power law fit
D = fit.power_law.D
print(f"KS distance (D): {D:.3f}")
# Likelihood ratio test comparing power law to lognormal
R, p = fit.distribution_compare("power_law", "lognormal")
print(f"Likelihood ratio test (power law vs lognormal): R={R:.3f}, p={p:.3f}")
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

# # Plot PDF of the data and fitted power-law
# plt.figure(figsize=(8, 6))
# fig_pdf = fit.plot_pdf(marker="o", linestyle="None", label="Empirical PDF")
# # fit.power_law.plot_pdf(
# #     label=f"Powerlaw Fit (alpha={alpha_powerlaw:.3f})", linestyle="dashed"
# # )

# plt.plot(
#     x_fit,
#     (alpha_powerlaw - 1)
#     / xmin_estimated
#     * (x_fit / xmin_estimated) ** (-alpha_powerlaw),
#     label=f"Powerlaw Fit (alpha={alpha_powerlaw:.3f})",
#     linestyle="dashed",
# )

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("x")
# plt.ylabel("PDF")
# plt.title("Power-law distributed data PDF and estimated exponents")
# plt.legend()
# plt.show()

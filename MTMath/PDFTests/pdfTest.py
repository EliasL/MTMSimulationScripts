import numpy as np
import matplotlib.pyplot as plt

# Parameters for the exponential distribution
lam = 1e8  # Rate parameter λ
scale = 1 / lam

# Parameters for the normal distribution
mu = 1e-8  # Mean (μ)
sigma = 1e-9  # Standard deviation (σ)

# Number of samples
n_samples = 10000

# Generate samples from the exponential distribution
exp_samples = np.random.exponential(scale=scale, size=n_samples)

# Generate samples from the normal distribution
norm_samples = np.random.normal(loc=mu, scale=sigma, size=int(n_samples / 10))

# Remove negative values (since log-log plot can't handle negatives)
norm_samples = norm_samples[norm_samples > 0]
# exp_samples = np.concatenate((norm_samples, exp_samples))
# Define logarithmic bins
min_value = max(min(exp_samples.min(), norm_samples.min()), 1e-12)
max_value = max(exp_samples.max(), norm_samples.max())
bins = np.logspace(np.log10(min_value), np.log10(max_value), 50)

# Plot histograms with logarithmic bins
plt.hist(
    exp_samples,
    bins=bins,
    density=False,
    alpha=0.6,
    color="skyblue",
    label="Exponential Sample Histogram",
)
# Set log-log scale
# plt.xscale("log")
# plt.yscale("log")


# Add labels and title
plt.xlabel("Value (log scale)")
plt.ylabel("Density (log scale)")
plt.title("Exponential and Normal Distributions on a Log-Log Plot")

# Add a legend
plt.legend()

# Display the plot
plt.show()

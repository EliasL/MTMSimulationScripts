import numpy as np
import matplotlib.pyplot as plt

# Number of samples
num_samples = 1000

# Generate log-normal distribution samples
mean = 0
sigma = 1
lognormal_samples = np.random.lognormal(mean, sigma, num_samples)

# Calculate the product of the samples
product_of_samples = np.prod(lognormal_samples)

# Plot the log-normal distribution
plt.hist(lognormal_samples, bins=30, alpha=0.6, color="blue", edgecolor="black")
plt.title("Log-Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(f"Product of all samples: {product_of_samples}")

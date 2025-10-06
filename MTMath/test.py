import powerlaw
from matplotlib import pyplot as plt
import numpy as np
# powerlaw Version: 1.5 (from pypi)

# I also tested the current github master branch

# Set the parameter range to be bad just to make
# sure it is being applied
parameterRanges = {"alpha": [1.9, 3.0]}

# Generate fake data distributed according to a power
# law with the alpha below (that is outside our parameter range)
realAlpha = 1.8
from scipy import stats

data = 1 / stats.powerlaw.rvs(realAlpha - 1, size=5000)

# Perform our fit
# (A) This line works
# fit = powerlaw.Fit(data, parameter_range=parameterRanges)

# (B) This line works
# fit = powerlaw.Fit(data, xmin=1)

# (C) This line will not work ("AttributeError: 'Power_Law' object has no attribute 'parent_Fit'")
fit = powerlaw.Power_Law(data, xmin=1, parameter_range=parameterRanges)

# Plotting functions
fig, ax = plt.subplots()

ax.hist(
    data,
    bins=np.logspace(0, np.log10(np.max(data)), 20),
    label=f"Data\n$\\alpha = {realAlpha}$",
    density=True,
    alpha=0.5,
)
ax.set_xscale("log")
ax.set_yscale("log")

fit.plot_pdf(ax=ax, label=f"PDF")
fit.power_law.plot_pdf(
    ax=ax, linestyle="--", label=f"power_law fit\n$\\alpha = {fit.power_law.alpha:.5}$"
)

plt.legend()
plt.show()

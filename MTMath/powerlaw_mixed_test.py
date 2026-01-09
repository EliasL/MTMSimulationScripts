from .plotPowerLaw import (
    make_fit,
    plot_data_and_fit,
    PLOTPATH,
    make_title_from_fit,
)
from .evaluatePowerlawFit import Truncated_Power_Law
import numpy as np


# Generate synthetic power-law distributed data
def generate_truncated_powerlaw_data(n, alpha, Lambda, xmin):
    dist = Truncated_Power_Law(xmin=xmin, alpha=alpha, Lambda=Lambda)
    return dist.generate_random(size=n)


# Generate power-law avalanche data
def generate_powerlaw_avalanche_data(alpha, size=5000, xmin=1e-8):
    increments = np.random.normal(xmin, xmin, size=size)  # Small incremental increases

    # Randomly select drop points
    drop_mask = np.random.uniform(size=size) > 0.7  # 40% chance of a drop
    Lambda = 1e4
    drops = generate_truncated_powerlaw_data(drop_mask.sum(), alpha, Lambda, xmin)
    # Apply drops
    increments[drop_mask] = -drops
    return increments


def get_only_drops(data):
    drop_mask = data < 0
    drops = -data[drop_mask]
    return drops


def testCombinedDists(alpha1=1.2, alpha2=1.4):
    # drops = generate_truncated_powerlaw_data(
    #     n=1000, alpha=alpha1, Lambda=1e4, xmin=1e-8
    # )
    # # It is different when i use the cache!
    # fit = make_fit(drops, xmin_range=(9.9e-9, 1.1e-8))

    # print(fit.xmin)
    # fit.evaluate_fit()

    drops = get_only_drops(
        generate_powerlaw_avalanche_data(alpha1)
        + generate_powerlaw_avalanche_data(alpha2)
    )

    fit = make_fit(drops)
    fit.evaluate_fit()

    filename = f"testing/{alpha1}_{alpha2}_lamb=1e4"
    title = make_title_from_fit(fit)
    ax = plot_data_and_fit(fit, title=title)
    ax.figure.savefig(PLOTPATH + filename + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved figure to {PLOTPATH + filename}.pdf")

from .plotPowerLaw import (
    find_best_xmin,
    plot_data_and_dist,
    PLOTPATH,
    make_title_from_dist,
)
from .powerlaw_lite import Truncated_Power_Law
import numpy as np
import os
from powerlaw import Fit


# Generate synthetic power-law distributed data
def generate_truncated_powerlaw_data(n, alpha, Lambda, xmin):
    dist = Truncated_Power_Law()
    dist.alpha = alpha
    dist.Lambda = Lambda
    dist.xmin = xmin
    return dist.generate_random(n=n)


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


def testCombinedDists(alpha1=1.1, alpha2=1.1):
    drops = get_only_drops(
        generate_powerlaw_avalanche_data(alpha1)
        + generate_powerlaw_avalanche_data(alpha2)
    )

    filename = f"testing/{alpha1}_{alpha2}_lamb=1e4"
    dist = find_best_xmin(drops, plotName=filename)
    title = make_title_from_dist(dist)
    ax = plot_data_and_dist(drops, dist, title=title)
    ax.figure.savefig(PLOTPATH + filename + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Saved figure to {PLOTPATH + filename}.pdf")

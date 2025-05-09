import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import powerlaw


def get_energy_drops(csvPath, xlim=[-np.inf, np.inf], debug=False):
    """
    Load energy drop data from CSV, filter by load limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    df = pd.read_csv(csvPath)
    diffs = df["avg_energy_change"]
    load = df["load"]
    lim_mask = (load > xlim[0]) & (load < xlim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = -diffs[mask]
    if debug:
        e = df["avg_energy"]
        fig, ax1 = plt.subplots()
        ax1.plot(load, e, label="Avg Energy")
        ax1.set_ylabel("Avg Energy")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        ax2.plot(load[mask], drops, label="Energy Drops")
        ax2.set_ylabel("Drops")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        ax2.set_ylim(0, drops.max() * 1.5)
        plt.show()
        print(f"Number of drops: {len(drops)}")
    return drops


def plot_data(ax, fit=None, data=None, xmin=None):
    if data is None and fit is not None:
        data = fit.data_original
    elif fit is None and data is not None:
        fit = powerlaw.Fit(data, xmin=xmin)
    else:
        raise ValueError("Either data or fit must be provided.")

    # full-data empirical
    fit.plot_ccdf(
        ax=ax,
        marker="o",
        linestyle="None",
        label="Empirical (all data)",
        original_data=True,
        facecolor="none",
        edgecolor="black",
    )
    return fit


def plot_fit(ax, fit, dist_name=None):
    # compute weight and x-grid
    data = fit.data_original
    xmin = fit.xmin
    mask = data >= xmin
    p_tail = mask.sum() / len(data)
    x_vals = np.logspace(
        np.log10(xmin),  # start at xmin
        np.log10(data.max()),
        num=200,
    )

    dist = getattr(fit, dist_name)
    y_tail = dist.ccdf(x_vals)
    y_full = p_tail * y_tail

    ax.plot(
        x_vals,
        y_full,
        linestyle="--",
        label=f"{dist_name.title()} (scaled)",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("Complementary CDF")
    plt.title(f"Fit for xmin={xmin}")
    plt.legend()


def compare_dists(fit):
    R, p = fit.distribution_compare(
        "truncated_power_law", "lognormal", normalized_ratio=True
    )
    print(
        f"Likelihood ratio test (truncated_power_law vs lognormal): R={R:.3f}, p={p:.3f}"
    )


if __name__ == "__main__":
    # User parameters
    debug = False
    csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    xlim = [2, 3]
    xmins = [1e-8, 1e-6, 1e-4]  # adjust as needed

    # Load data
    data = get_energy_drops(csvPath, xlim, debug)

    # Compare three distributions directly
    dist_names = ["truncated_power_law", "lognormal"]

    for xmin in xmins:
        fig, ax = plt.subplots()
        # Fit distributions
        fit = powerlaw.Fit(data, xmin=xmin)
        # Plot data and fit
        fit = plot_data(ax, fit)
        # Plot fitted distributions
        for dist_name in dist_names:
            plot_fit(ax, fit, dist_name)

        plt.show()
        # Compare distributions
        compare_dists(fit)

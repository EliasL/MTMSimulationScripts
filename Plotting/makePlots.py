from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import matplotlib.pylab
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import re
from datetime import timedelta
import powerlaw
import json
from simplification.cutil import simplify_coords_vwp
from tqdm import tqdm
from pathlib import Path
from .dataFunctions import get_data_from_name

if True:
    import warnings

    # This line converts all RuntimeWarnings into errors (exceptions).
    warnings.simplefilter("error", RuntimeWarning)


def durations_to_seconds(durations):
    # Create a mapping from unit to number of seconds
    unit_map = {
        "d": 86400,  # 24 hours * 3600 sec/hour
        "h": 3600,  # 60 minutes * 60 sec/minute
        "m": 60,  # 60 sec
        "s": 1,
    }

    result = []
    for duration in durations:
        total_seconds = 0
        # Split by space to handle multiple tokens like "1m 38s"
        parts = duration.split()
        for part in parts:
            # Last character is the unit, rest is the number
            number = float(part[:-1])
            unit = part[-1]
            # Convert and accumulate
            total_seconds += number * unit_map[unit]
        result.append(total_seconds)
    return result


def plotYOverX(
    X,
    Y,
    fig=None,
    ax=None,
    indicateLastPoint=False,
    tolerance=1e-12,
    xlim=None,
    ylim=None,
    **kwargs,
):
    # Ensure X and Y are NumPy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Input validation
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")

    # Apply xlim and ylim to crop the data
    mask = np.ones_like(X, dtype=bool)
    if xlim is not None:
        mask &= (X >= xlim[0]) & (X <= xlim[1])
    if ylim is not None:
        mask &= (Y >= ylim[0]) & (Y <= ylim[1])
    X = X[mask]
    Y = Y[mask]

    # Handle empty data after cropping
    if X.size == 0 or Y.size == 0:
        raise ValueError("No data points remain after applying xlim and ylim.")

    # Simplify the points after cropping
    if tolerance is not None:
        points = np.column_stack((X, Y))
        simplified_points = simplify_coords_vwp(points, tolerance)
        X_simplified = simplified_points[:, 0]
        Y_simplified = simplified_points[:, 1]
    else:
        X_simplified = X
        Y_simplified = Y

    # If no axis is provided, create
    # a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()

    # Plot on the provided axis
    (line,) = ax.plot(X_simplified, Y_simplified, **kwargs)

    # Optionally highlight the last point
    point = None
    if indicateLastPoint and X_simplified.size > 0:
        # Add a scatter point at the last point
        kwargs_without_label = {k: v for k, v in kwargs.items() if k != "label"}
        if "alpha" in kwargs_without_label:
            kwargs_without_label["alpha"] = min(
                kwargs_without_label["alpha"] * 1.5, 1.0
            )
        point = ax.scatter(X_simplified[-1], Y_simplified[-1], **kwargs_without_label)

    # Return the axis object for further use
    return fig, ax, line, point


def time_to_seconds(duration_str):
    pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?"
    matches = re.match(pattern, duration_str.strip())

    if not matches:
        return timedelta()

    days, hours, minutes, seconds = matches.groups(default="0")

    return timedelta(
        days=int(days), hours=int(hours), minutes=int(minutes), seconds=float(seconds)
    ).seconds


def plotColumns(cvs_files, Y, labels, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    values = []
    for file in cvs_files:
        df = pd.read_csv(file)
        last_entry = df[Y].values[-1][0]
        last_entry_seconds = time_to_seconds(last_entry)
        values.append(last_entry_seconds)

    ax.bar(labels, values)
    return fig, ax


def plotRollingAverage(X, Y, intervalSize=100, fig=None, ax=None, **kwargs):
    # Calculate rolling average
    rollingMean = Y.rolling(window=intervalSize, min_periods=1, center=True).mean()

    # Check if axis provided, if not, create a new one
    if ax is None:
        fig, ax = plt.subplots()
    del kwargs["label"]
    # Plotting the interval average
    (line,) = ax.plot(
        X, rollingMean, label=f"Rolling average (window={intervalSize})", **kwargs
    )

    # Return the axis object for further use
    return fig, ax, line


# Process drops
def pros_d(df, min_npd, strainLims):
    e = [key for key in df.columns if key not in ["Load", "Nr plastic deformations"]][0]

    if e == "Avg energy" and "Avg energy change" in df:
        diffs = df["Avg energy change"]
    else:
        diffs = np.diff(df[e])

    # Combine all conditions into a single mask using element-wise logical AND
    mask = (
        (df["Nr plastic deformations"] >= min_npd)
        & (df["Load"] >= strainLims[0])
        & (df["Load"] <= strainLims[1])
    )
    # Since np.diff reduces the length by 1, adjust the mask accordingly
    # The mask needs to exclude the first entry, so slice the mask by [1:]
    mask = mask[:-1]

    # Filter the diffs array
    diffs = diffs[mask]
    drop_mask = diffs < 0
    drops = -diffs[drop_mask]

    # plt.plot(range(len(drops)), drops)
    # plt.yscale("log")
    # plt.xlabel("Drop number")
    # plt.ylabel(r"$-\Delta E$")
    # plt.show()

    # load = df["Load"][:-1][mask][drop_mask]
    # plt.plot(load, drops)
    # plt.yscale("log")
    # plt.xlabel(r"$\gamma$")
    # plt.ylabel(r"$-\Delta E$")
    # plt.show()
    # Return the negative drops and flipp them
    return drops


def getPowerLawFit(
    dfs,
    minNrOfDeformations=0,
    dropLim=[None, None],
    maxEnergy=np.inf,
    bootstrap=False,
    split=True,
    innerStrainLims=(np.inf, -np.inf),
    outerStrainLims=(-np.inf, np.inf),
):
    # The purpose of the inner and outer strain lims is to isolate a middle
    # transition region from the analasys. If this is our "line" of data
    # ---|-----------------|--------|----------------------|---
    #    (O1)              (I1)     (I2)                   (O2)
    # I have labeled the locatinos of the outer (O) and inner (I) strain
    # limits. That way, the two regions we are interested in analysing is
    # given by (O1, I1) and (I2, O2).

    if split:
        combined_drops = [None, None]
        if outerStrainLims[0] < innerStrainLims[0]:
            pre_yield_drops = [
                pros_d(
                    df=df,
                    min_npd=minNrOfDeformations,
                    strainLims=(outerStrainLims[0], innerStrainLims[0]),
                )
                for df in dfs
            ]
            combined_drops[0] = np.concatenate(pre_yield_drops)

        if outerStrainLims[1] > innerStrainLims[1]:
            post_yield_drops = [
                pros_d(
                    df=df,
                    min_npd=minNrOfDeformations,
                    strainLims=(innerStrainLims[1], outerStrainLims[1]),
                )
                for df in dfs
            ]
            combined_drops[1] = np.concatenate(post_yield_drops)

    else:
        combined_drops = [
            np.concatenate(
                [pros_d(df, minNrOfDeformations, outerStrainLims) for df in dfs]
            )
        ]

    if bootstrap:
        # Parameters for bootstrapping
        n_bootstrap = 10  # Number of bootstrap samples
        if split and bootstrap:
            raise (RuntimeError("Bootstrap and split is not supported."))
        drops = combined_drops[0]

        # bootstrapping is very expensive, so we try to store unique names for
        # datasets we have already done and save the results so we only need
        # to do them once.

        # first we need to generate a name for the file. It is important that
        # each name is unique to the data set. Instead of garanteeing this, we
        # try to make it very unlikely that two datasets would get the same name
        name = f"bsFile.{n_bootstrap}_{len(drops)}_{np.sum(drops)}.json"
        folder = "bootstrapData"
        os.makedirs(folder, exist_ok=True)  # Ensure the folder exists
        filePath = os.path.join(folder, name)

        # Check if the result file already exists
        if os.path.exists(filePath):
            # Load the JSON result object if the file exists
            with open(filePath, "r") as f:
                result = json.load(f)
            # print(f"Loaded bootstrap results from {filePath}")
        else:
            # Perform bootstrapping and fit the power law model
            result = doBootstrap(
                drops,
                n_bootstrap,
                lambda drops: powerlaw.Fit(
                    drops, xmin=dropLim[0], xmax=dropLim[1], fit_method="Likelihood"
                ),
            )

            # Save the result to a JSON file
            with open(filePath, "w") as f:
                json.dump(result, f)

        return result, combined_drops

    else:
        combined_fits = []
        for drops in combined_drops:
            if drops is None:
                combined_fits.append(None)
                continue

            fit = powerlaw.Fit(
                drops, xmin=dropLim[0], xmax=dropLim[1], fit_method="Likelihood"
            )

            # plt.scatter(
            #     range(len(drops)),
            #     drops,
            #     facecolor="none",
            #     edgecolors=(0, 0, 1, 0.5),
            #     s=1,
            # )
            # plt.yscale("log")
            # plt.xlabel("Drop number")
            # plt.ylabel(r"$-\Delta E$")
            # plt.show()
            # fit.plot_pdf(original_data=True)
            # plt.ylabel("PDF")
            # plt.xlabel(r"$-\Delta E$")
            # plt.show()

            combined_fits.append(fit)

        return combined_fits, combined_drops


# List of distributions to compare
DISTRIBUTIONS = {
    "lognormal": ["mu", "sigma"],
    "exponential": ["Lambda"],
    "truncated_power_law": ["alpha", "Lambda"],
    "power_law": ["alpha"],
    # "negative_binomial": ["r", "p"],
    "stretched_exponential": ["Lambda", "beta"],
    # "gamma": ["k", "theta"],
}


def pdf(data, xmin=None, xmax=None, linear_bins=False, **kwargs):
    """
    Returns the probability density function (normalized histogram) of the
    data.

    Parameters
    ----------
    data : list or array
    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.
    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.
    linear_bins : float, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).

    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.
    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """
    from numpy import logspace, histogram, floor, unique, asarray
    from math import ceil, log10

    data = asarray(data)
    if not xmax:
        xmax = max(data)
    if not xmin:
        xmin = min(data)

    if (
        xmin < 1
    ):  # To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2 = xmax / xmin
        xmin2 = 1
    else:
        xmax2 = xmax
        xmin2 = xmin

    if "bins" in kwargs.keys():
        bins = kwargs.pop("bins")
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2) + 1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size - log_min_size) * 10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin < 1:  # Needed to include also data x<1 in pdf.
        hist, edges = histogram(data / xmin, bins, density=True)
        edges = edges * xmin  # transform result back to original
        hist = hist / xmin  # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True)

    return edges, hist


def plotPowerLaw(
    drops, ax, fit, label, part_label, dist="truncated_power_law", add_fit=True
):
    # np.savetxt(f"{label}{part_label}.csv", drops, delimiter=",")
    global color_index, index

    # Get the current color
    color = colors[label]

    fit.plot_pdf(
        original_data=True,
        ax=ax,
        marker=markers[index],
        linestyle="none",
        markerfacecolor="none",
        # markeredgecolors=color,
        c=color,
        # markersize=100,
        label=f"{label}{part_label}",
    )

    extraLabel = " ".join(
        [
            rf"$\{v.lower()}$: " + f"{getattr(getattr(fit,dist), v):.2f}"
            if len(v) > 1
            else (rf"${v}$: ") + f"{getattr(getattr(fit,dist), v):.2f}"
            for v in DISTRIBUTIONS[dist]
        ]
    )
    # Now, plot the best-fitting distribution
    if add_fit:
        # Plot power-law fit
        getattr(fit, dist).plot_pdf(
            ax=ax,
            linestyle="--",
            label=f"fit: {extraLabel}",
            color="black",
        )
        renormalizeMostRecentPlot(ax, fit, getattr(fit, dist))

    print(f"{label}{part_label}: dist: {dist}, {extraLabel}")

    return ax


def renormalizeMostRecentPlot(ax, fit, dist):
    """
    When we plott the ccdf fit over a subregion of the whole data, it will be
    normalized to a different region than the whole region of the data, so we
    need to readjust the plot.
    NB: Requires a line and a scatter plot to be the most recent additions to
    the ax.
    """
    # Get fit data
    fit_line = ax.lines[-1]
    fit_y_data = fit_line.get_ydata()
    fit_x_data = fit_line.get_xdata()
    fit_region = [np.min(fit_x_data), np.max(fit_x_data)]

    # Get original data
    edges, hist = pdf(fit.data_original)
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    y_data = hist
    x_data = bin_centers

    # Find the region for the scatter data that corresponds to the fit region
    scatter_region_indices = (x_data >= fit_region[0]) & (x_data <= fit_region[1])

    # Subset scatter data to the fit region
    scatter_x_data_region = x_data[scatter_region_indices]
    scatter_y_data_region = y_data[scatter_region_indices]

    # Integrate the y data over x for both fit and scatter data in the region of the fit
    area_fit = np.trapz(fit_y_data, fit_x_data)  # Numerical integration for fit data
    area_scatter = np.trapz(scatter_y_data_region, scatter_x_data_region)

    # Compute the scaling factor to adjust the fit data to match the scatter data area
    scaling_factor = area_scatter / area_fit

    # Apply the scaling factor to the fit y-data
    scaled_fit_y_data = fit_y_data * scaling_factor

    # Set the new y data
    fit_line.set_ydata(scaled_fit_y_data)
    # Redraw the figure to reflect the changes
    ax.figure.canvas.draw()


def doBootstrap(drops, n, fit_func):
    # Parameters for bootstrapping

    # Store bootstrap results for each dataset
    fits = []
    result = {}

    # Perform bootstrapping to estimate uncertainties
    for _ in range(n):
        # Resample the data with replacement
        bootstrap_sample = np.random.choice(drops, size=len(drops), replace=True)

        # Fit the model to the bootstrap sample
        fit = fit_func(bootstrap_sample)
        fits.append(fit)

    for dist, params in DISTRIBUTIONS.items():
        result[dist] = {}
        all_params = ["D", "xmin", "V", "Asquare", "Kappa"] + params
        for p in all_params:
            values = np.array(
                list(
                    map(
                        lambda fit: getattr(getattr(fit, dist), p),
                        fits,
                    )
                )
            )
            mean = np.mean(values)
            std = np.std(values)
            result[dist][p] = mean
            result[dist][f"{p}_std"] = std
    return result


# Define global variables
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "v", "^", "s", "D", "p", "*"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = {"FIRE": "#d24646", "L-BFGS": "#008743", "CG": "#ffa701"}
colors = {"L-BFGS": "#3BCDDD", "CG": "#CDDD3B", "FIRE": "#DD3BCD"}
colors = {"L-BFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}
color_index = 0
index = 0


def get_method(cvs_file_path):
    if not isinstance(cvs_file_path, str):
        cvs_file_path = cvs_file_path[0]
    if "minimizerFIRE" in cvs_file_path:
        return "FIRE"

    elif "minimizerCG" in cvs_file_path:
        return "CG"

    else:
        return "L-BFGS"


def plotSplitPowerLaw(
    dfs,
    fig=None,
    ax=None,
    label="",
    plot_pre_yield=True,
    plot_post_yield=True,
    add_fit=True,
    dist="truncated_power_law",
    **kwargs,
):
    global color_index, index, line_index
    if ax is None:
        fig, ax = plt.subplots()

    # Trim data based on dislocations
    minNrOfDeformations = 0

    c_fits, c_drops = getPowerLawFit(dfs, minNrOfDeformations, **kwargs)

    for fit, drops, part_label in zip(c_fits, c_drops, [" pre yield", " post yield"]):
        if drops is None:
            continue
        if part_label == " pre yield" and not plot_pre_yield:
            continue
        if part_label == " post yield" and not plot_post_yield:
            continue
        if not (plot_post_yield and plot_pre_yield):
            part_label = ""
        ax = plotPowerLaw(
            drops,
            ax,
            fit,
            label,
            part_label,
            dist=dist,
            add_fit=add_fit,
        )

    # Increment the global call count
    color_index += 1

    # Check if we've used all colors
    if color_index >= len(colors):
        # Switch to the next line style and marker
        color_index = 0

    # Set axes to logarithmic
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig, ax


def plotEnergyAvalancheHistogram(dfs, fig=None, axs=None, label=""):
    e = "Avg energy"
    pre_yield_df = [df[0 : np.argmax(df[e]) + 1] for df in dfs]
    post_yield_df = [df[np.argmax(df[e]) + 1 :] for df in dfs]

    # Prepare the figure and subplots for a 3x3 grid
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))  # Adjust size as necessary
        axs = axs.flatten()  # Flatten the array of axes for easier iteration

    min_group_index = 1
    max_group_index = 9  # This corresponds to 2^9 as the highest group (2^1 to 2^9)
    groups_indexes = range(min_group_index, max_group_index + 1)
    # Initialize a dictionary to store drops data for each group

    # Process each DataFrame split
    for split_dfs, label in zip(
        (pre_yield_df, post_yield_df),
        ("Pre yield", "Post yield"),
    ):
        groups_data = {
            i: [] for i in groups_indexes
        }  # Dictionary to store data for each group
        for df in split_dfs:
            # Filter out zero and NaN values
            df = df[df["Nr plastic deformations"] > 0]

            group_index = np.floor(np.log2(df["Nr plastic deformations"])).astype(int)
            group_index = np.clip(
                group_index, min_group_index, max_group_index
            )  # Clamp the group index

            drops = -np.diff(df[e])

            # Filter out negative drops
            group_index = group_index[1:][drops > 0]
            drops = drops[drops > 0]

            # Aggregate drops data by group
            for i in groups_indexes:
                mask = group_index == i  # Apply mask to align with `diffs` length
                if any(mask):
                    groups_data[i].extend(drops[mask])

        # Define logarithmic bins

        # Now plot the aggregated data for each group
        for i, ax in enumerate(axs):
            exp = groups_indexes[i]
            if not groups_data[exp]:
                continue
            min_v = min(groups_data[exp])
            max_v = max(groups_data[exp])
            if min_v == max_v:
                continue
            bins = np.logspace(
                np.log10(min_v), np.log10(max_v), 20
            )  # Generate 20 logarithmic bins
            ax.hist(groups_data[exp], bins=bins, alpha=0.75, label=label)
            energyCutoffVisualization = 1e-5
            if ax.get_xlim()[0] < energyCutoffVisualization:
                ax.vlines(
                    energyCutoffVisualization,
                    ymin=0,
                    ymax=ax.get_ylim()[1],
                    color="#1f77b4",
                )

            ax.set_title(f"{2**exp}-{2**(exp+1)-1} p.e.")
            if i == len(axs) - 1:
                ax.set_title(f"More than {2**exp} p.e.")
            ax.set_yscale("log")
            ax.set_xscale("log")
            if i == 0:  # or i == len(axs) - 1:
                ax.legend()
            # Only allow a maximum of 3 ticks along x-axis
            # ax.xaxis.set_major_locator(MaxNLocator(3))
            # Remove axis names for inner axes
            if i % 3 == 0:  # Not the first column
                ax.set_ylabel(r"$P>\langle E \rangle$")
            if i >= 6:  # Not the bottom row
                ax.set_xlabel(r"$-\Delta \langle E \rangle$")

    return fig, axs


def getPrettyLabel(string):
    s = ""
    if "minimizer=" in string:
        s = string.split("minimizer=")[1].split(",")[0]
    if "," not in string:
        s = string
    return s.replace("LBFGS", "L-BFGS")


# Example usage can be added as necessary with DataFrames having 'Nr plastic deformations' and 'Avg energy'


def plotSlidingPowerLaw(dfs, dist="truncated_power_law", fig=None, ax=None, label=""):
    global color_index, index, line_index
    if ax is None:
        fig, ax = plt.subplots()

    minNrOfDeformations = 0

    minEnergies = np.logspace(np.log10(1e-6), np.log10(1e-3), 30)
    exponents = []

    for minDropValue in tqdm(minEnergies):
        c_fits, c_drops = getPowerLawFit(dfs, minNrOfDeformations, minDropValue)
        exponents.append(c_fits[1].truncated_power_law.alpha)
    exponents = np.array(exponents)

    ax.plot(minEnergies, exponents, label=getPrettyLabel(label))
    ax.set_xscale("log")
    return fig, ax


def plotSlidingWindowPowerLaw(
    dfs,
    dist="truncated_power_law",
    dropLim=[None, None],
    windowRadius=0.1,
    fig=None,
    ax1=None,
    ax2=None,
    label="",
):
    if ax1 is None:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

    minNrOfDeformations = 0

    # Strain window centers
    strainWindowCenter = np.linspace(
        min(dfs[0]["Load"]) + windowRadius,
        max(dfs[0]["Load"]) - windowRadius,
        20,
    )

    # Initialize lists to store exponents, cutoffs, and errors
    exponents = []
    cutoffs = []
    exponent_errors = []
    cutoff_errors = []

    # Iterate over strain window centers and get power-law fits
    for center in tqdm(strainWindowCenter):
        r, c_drops = getPowerLawFit(
            dfs,
            minNrOfDeformations,
            dropLim=dropLim,
            bootstrap=True,
            split=False,
            outerStrainLims=(center - windowRadius, center + windowRadius),
        )
        # TODO Check individual plots
        # ax1 = plotPowerLaw(c_drops, ax1, r, label, f": {center}")
        # plt.show()
        exponents.append(r[dist]["alpha"])
        exponent_errors.append(r[dist]["alpha_std"])
        if "Lambda" in r[dist]:
            cutoffs.append(r[dist]["Lambda"])
            cutoff_errors.append(r[dist]["Lambda_std"])

    # Convert lists to numpy arrays for easier manipulation
    exponents = np.array(exponents)
    cutoffs = np.array(cutoffs)
    exponent_errors = np.array(exponent_errors)
    cutoff_errors = np.array(cutoff_errors)

    # Plot exponents (alpha) with error bars
    ax1.errorbar(
        strainWindowCenter,
        exponents,
        # Get the current color
        color=colors[label],
        yerr=exponent_errors,
        label="$\\alpha$ " + label,
        fmt="-o",  # Line with circle markers
        capsize=3,  # Error bar cap size
    )

    # Plot cutoffs (lambda) with error bars
    ax2.errorbar(
        strainWindowCenter,
        cutoffs,
        color=colors[label],
        yerr=cutoff_errors,
        label="$\\lambda$ " + label,
        fmt="--^",  # Line with triangular markers
        capsize=3,  # Error bar cap size
    )

    return fig, ax1, ax2


def makePlot(
    csv_file_paths,
    ax=None,
    fig=None,
    name="",
    Y="Avg energy",
    X="Load",
    x_name=None,
    y_name=None,
    use_y_axis_name=True,
    labels=None,
    use_title=False,
    title=None,
    plot_average=False,
    xlim=None,
    ylim=None,
    indicateLastPoint=False,
    plot_roll_average=False,
    plot_raw=True,
    plot_power_law=False,
    plot_columns=False,
    ylog=False,
    show=False,
    colors=None,
    plot_total=False,
    legend=None,
    add_shift=False,
    add_images=False,
    metric="energy",
    image_pos=None,
    image_size=0.4,
    add_cbar=True,
    save=True,
    mark=None,
    mark_pos=(0.8, 0.95),
    mark_fontsize=20,
    legend_loc="best",
    plot_pre_yield=True,
    plot_post_yield=True,
    dist="truncated_power_law",
):
    if len(csv_file_paths) == 0 or (
        len(csv_file_paths) > 0 and len(csv_file_paths[0]) == 0
    ):
        print("No files provided.")
        return
    if x_name is None:
        if X == "Load":
            x_name = r"Strain $\gamma$"
        else:
            x_name = X

    if y_name is None and use_y_axis_name:
        if Y == "Avg RSS":
            y_name = r"Stress $\langle \sigma \rangle$"
        elif Y == "Avg energy":
            y_name = r"Energy $\langle E \rangle$"
        else:
            y_name = Y

    # if we are not given a list, we make it into a list
    if isinstance(csv_file_paths, str):
        csv_file_paths = [csv_file_paths]

    if fig is None or ax is None:
        assert fig is None
        assert ax is None
        fig, ax = plt.subplots()

    lines = []
    data = []
    xData = []
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    for i, csv_file_path in enumerate(csv_file_paths):
        if X is None:
            break
        if isinstance(Y, str):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
        else:
            df = pd.read_csv(csv_file_path, usecols=[X] + Y)
            if plot_average:
                raise Warning("Cannot plot average with multiple Y columns")

        # Truncate data based on Lims
        if xlim:
            df = df[(df[X] >= xlim[0]) & (df[X] <= xlim[1])]
        if ylim:
            df = df[(df[Y] >= ylim[0]) & (df[Y] <= ylim[1])]

        if df[X].iloc[-1] < 1:
            print(f"{csv_file_path} is not done!")

        data.append(df[Y].values)
        xData.append(df[X].values)

        kwargs = {"fig": fig, "ax": ax, "indicateLastPoint": indicateLastPoint}

        if colors:
            kwargs["color"] = colors[i]
        if len(csv_file_paths) > 10:
            kwargs["color"] = "gray"
            kwargs["alpha"] = 0.5

        for Y_ in [Y] if isinstance(Y, str) else Y:
            if len(df[Y_]) == 0:
                continue
            if labels is None:
                kwargs["label"] = Y
            else:
                kwargs["label"] = labels[i]
                # +((" - " + Y_) if not isinstance(Y, str) else "")
            if add_shift:
                df[Y_] -= i * np.max(df[Y_]) / 500
            line = None
            point = None
            if plot_raw:
                fig, ax, line, point = plotYOverX(df[X], df[Y_], **kwargs)
            if plot_roll_average:
                fig, ax, line = plotRollingAverage(df[X], df[Y_], **kwargs)
            if line is not None:
                lines.append(line)
            if point is not None:
                lines.append(point)
        if plot_total:
            assert not isinstance(Y, str)
            if not plot_raw:
                kwargs["label"] = "total"
            fig, ax, line, point = plotYOverX(
                df[X], sum([df[Y_] for Y_ in Y]), **kwargs
            )
            lines.append(line)

    if plot_columns:
        fig, ax = plotColumns(csv_file_paths, Y, labels, fig, ax)

    if plot_average:
        # Determine the maximum length among all arrays
        max_length_index = np.argmax([len(d) for d in data])
        max_length = len(data[max_length_index])

        # Initialize the average array and a count array to track how many entries per index
        average = np.zeros(max_length)
        count = np.zeros(max_length)

        # Aggregate data
        for d in data:
            length = len(d)
            average[:length] += d
            count[:length] += 1

        # Compute average where count is non-zero to avoid division by zero
        average = np.divide(
            average, count, out=np.zeros_like(average), where=count != 0
        )

        kwargs = {"fig": fig, "ax": ax, "label": "Average", "color": "black"}
        if plot_raw:
            fig, ax, line, point = plotYOverX(
                xData[max_length_index], average, **kwargs
            )
        lines.append(line)

    if plot_power_law:
        kwargs = {
            "fig": fig,
            "ax": ax,
            "label": "Fit",
            "color": "black",
            "include_label": legend,
            "plot_pre_yeild": plot_pre_yield,
            "plot_post_yield": plot_post_yield,
            "dist": dist,
        }
        fig, ax, line = plotSplitPowerLaw(data, **kwargs)

    # cursor.connect(
    #   "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    if ylog:
        ax.set_yscale("log")

    # Create a list of line plots only for the legend
    handles, labels = ax.get_legend_handles_labels()
    line_handles = [
        handle for handle in handles if isinstance(handle, matplotlib.lines.Line2D)
    ]

    # Filter labels accordingly
    line_labels = [
        label
        for handle, label in zip(handles, labels)
        if isinstance(handle, matplotlib.lines.Line2D)
    ]

    # Set the legend with the filtered handles and labels
    if legend and isinstance(legend, bool):
        ax.legend(line_handles, line_labels, loc=legend_loc)
    elif isinstance(legend, list):
        custom_legend = [
            mlines.Line2D(
                [], [], color="blue", marker="o", linestyle="-", label="Custom Label 1"
            ),
            mlines.Line2D(
                [],
                [],
                color="green",
                marker="o",
                linestyle="--",
                label="Custom Label 2",
            ),
        ]
        # Create the legend with custom labels
        ax.legend(handles=custom_legend, loc=legend_loc)
    elif isinstance(legend, str):
        ax.legend(line_handles, [legend], loc=legend_loc)
    if add_images:
        i = 0
        addImagesToPlot(
            ax,
            fig,
            csv_file_paths[i],
            xData[i],
            data[i],
            image_pos,
            image_size,
            mesh_property=metric,
            add_cbar=add_cbar,
        )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if use_title:
        if title is None:
            ax.set_title(f"{y_name} over {x_name}")
        else:
            ax.set_title(title)

    if mark:
        assert mark_pos is not None
        add_mark(ax, f"({mark})", *mark_pos, fontsize=mark_fontsize)

    if save:
        figPath = os.path.join(os.path.dirname(csv_file_paths[0]), name)
        fig.savefig(figPath)
        print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()
    return fig, ax


def add_mark(ax, mark, x, y, color="black", fontsize=30):
    # Adding LaTeX-style bold font using \textbf{}
    ax.text(
        x,
        y,
        r"$\textbf{" + mark + "}$",  # LaTeX syntax for bold
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        color=color,
    )


def addImagesToPlot(
    ax,
    fig,
    csv_file_path,
    x,
    y,
    image_pos,
    size=0.4,
    mesh_property="energy",
    add_cbar=True,
):
    from .pyplotFunctions import plot_mesh

    # First we get the folder with vtu_files
    framesPath = Path(csv_file_path).parent / "data"

    # Define the regex pattern to match the file names and find the number between the dots
    pattern = re.compile(r".*\.(\d*)\.vtu")

    # Get all files in the folder matching the pattern and extract both the number and full path
    matching_files = [
        (
            int(pattern.match(f.name).group(1)),
            f,
        )  # Create a tuple with the number and the full path
        for f in framesPath.iterdir()
        if f.is_file() and pattern.match(f.name)
    ]

    # Sort the list of tuples by the number (the first element of the tuple)
    matching_files.sort(key=lambda x: x[0])

    # Extract the paths for the first, middle, and last files
    first_file = matching_files[0][1]
    middle_file = matching_files[int(len(matching_files) * 0.45)][1]
    last_file = matching_files[-1][1]

    if not isinstance(size, list):
        size = [size] * 3

    for pos, size, vtu_file, index_fraction in zip(
        image_pos,
        size,
        [first_file, middle_file, last_file],
        [0, 0.45, 0.999],
    ):
        # Top left for the first image
        #                           (left, bottom, width, height)
        ax_inset = ax.inset_axes((pos[0], pos[1], size, size))
        _, cmap, norm = plot_mesh(
            e_lims=[0, 0.37],
            vtu_file=vtu_file,
            ax=ax_inset,
            add_rombus=False,
            shift=False,
            mesh_property=mesh_property,
        )
        ax_inset.axis("off")

        # Now we want to make arrows that point to the graph where the image is
        # taken from

        # This should work, but there is a desync in my data somehow that means the
        # load in the vtu files are not accurate
        if False:
            # First we find the load
            load = get_data_from_name(vtu_file)["load"]

            x_value = float(load)
            # we find the index that is closest to this load,
            # and use that to find a y value
            index = np.abs(x - x_value).argmin()
            y_value = y[index]
        else:
            value_index = int(index_fraction * len(x))
            x_value = x[value_index]
            y_value = y[value_index]
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Convert normalized coordinates to actual axis coordinates
        arrow_start = (
            xlim[0] + (pos[0] + size / 2) * (xlim[1] - xlim[0]),
            ylim[0] + (pos[1] + size / 2) * (ylim[1] - ylim[0]),
        )

        # Arrow's ending point (the point on the main plot where the image corresponds to)
        arrow_end = (x_value, y_value)

        # Add the arrow using annotate
        ax.annotate(
            "",  # No text
            xy=arrow_end,  # End of the arrow (on the main plot)
            xytext=arrow_start,  # Start of the arrow (from the inset)
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.5, headwidth=5, headlength=5
            ),
        )
    # Create the color bar using the colormap and normalization
    if add_cbar:
        # Create a ScalarMappable object with the colormap and norm
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        if mesh_property == "stress":
            label = r"Stress $\sigma$"
        elif mesh_property == "energy":
            label = r"Energy $E$"
        elif mesh_property == "m":
            label = r"Dislocations $\textbf{m}_3$"
        # Add the color bar to the figure
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.005, label=label)


def removeBadData(df, crash_count, csv_file_path):
    # The max energy of an element should be around 4.2-6
    # If the energy of an element is 10, something has probably gone wrong
    max_e = "Max energy"
    max_value = 10

    if (df[max_e] > max_value).any():
        crash_count += 1
        mask = df[max_e] < max_value
        df = df[mask]
        # print(f"Crash in {csv_file_path}.")
    return df, crash_count


def makeAverageComparisonPlot(
    grouped_csv_file_paths,
    Y="Avg energy",
    name="",
    show=False,
    use_title=False,
    use_y_axis_name=True,
    ax=None,
    fig=None,
    save=True,
    xlim=None,
    ylim=None,
    mark=None,
    mark_pos=(0.8, 0.95),
    mark_fontsize=17,
    **kwargs,
):
    global color_index, index, line_index
    color_index, index, line_index = 0, 0, 0
    X = "Load"
    if Y == "Avg energy":
        y_name = r"Energy $\langle E \rangle$"
        if name == "":
            name = "Avg energy"
    elif Y == "Avg RSS":
        y_name = r"Stress $\langle \sigma \rangle$"
        if name == "":
            name = "Avg stress"
    elif "time" in Y:
        y_name = "Seconds"
        name = Y

    x_name = r"Strain $\gamma$"
    title = f"{name}"

    if fig is None or ax is None:
        assert fig is None
        assert ax is None
        fig, ax = plt.subplots()

    color_index = -1
    line_index = 0

    crash_count = 0

    # for each configuration
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        data = []
        # Increment the global call count
        color_index += 1
        # Check if we've used all colors
        if color_index >= len(colors):
            # Switch to the next line style and marker
            color_index = 0
            line_index += 1

        # Get the current color
        color = colors[get_method(csv_file_paths)]

        # For each seed using this config
        for j, csv_file_path in enumerate(csv_file_paths):
            # print(csv_file_path)
            df = pd.read_csv(csv_file_path, usecols=[X, Y, "Max energy"])
            # If Y contains strings, we will assume it is a time, and convert it to
            # seconds
            if isinstance(df[Y][0], str):
                df[Y] = durations_to_seconds(df[Y])
            # df = df[0:50000]
            if df.empty:
                continue
            df, crash_count = removeBadData(df, crash_count, csv_file_path)

            data.append(df[Y].values)

            e_kwargs = {
                "fig": fig,
                "ax": ax,
                "color": color,
                "linestyle": line_styles[line_index],
                "alpha": 0.05,
                "zorder": color_index - 10,
                "xlim": xlim,
                "ylim": ylim,
            }
            fig, ax, line, point = plotYOverX(df[X], df[Y], **e_kwargs)
        # Determine the maximum length among all arrays
        max_length_index = np.argmax([len(d) for d in data])
        max_length = len(data[max_length_index])

        # Initialize the average array and a count array to track how many entries per index
        average = np.zeros(max_length)
        count = np.zeros(max_length)

        # Aggregate data
        for d in data:
            length = len(d)
            average[:length] += d
            count[:length] += 1

        # Compute average where count is non-zero to avoid division by zero
        average = np.divide(
            average, count, out=np.zeros_like(average), where=count != 0
        )
        label = get_method(csv_file_paths)
        a_kwargs = {
            "fig": fig,
            "ax": ax,
            "label": label,
            "color": colors[label],
            "linestyle": line_styles[line_index],
            "zorder": -color_index,
            "xlim": xlim,
            "ylim": ylim,
        }
        df = pd.read_csv(csv_file_paths[max_length_index], usecols=[X])
        fig, ax, line, point = plotYOverX(df[X], average, **a_kwargs)
    if crash_count > 0:
        print(f"Found {crash_count} crashes using {label}.")
    # Set the legend with the filtered handles and labels
    ax.legend(loc="upper left")
    ax.set_xlabel(x_name)

    if use_y_axis_name:
        ax.set_ylabel(y_name)
    if use_title:
        ax.set_title(title)

    if mark:
        assert mark_pos is not None
        add_mark(ax, f"({mark})", *mark_pos, fontsize=mark_fontsize)

    if save:
        # Get the parent directory of the CSV file
        csv_directory = Path(grouped_csv_file_paths[0][0]).parent.parent
        # Move to the "plots" directory relative to the CSV file directory
        plotPath = csv_directory / "plots"

        figPath = os.path.join(plotPath, name + ".pdf")

        # fig.savefig(figPath)
        fig.savefig("Plots/" + name + ".pdf")
        # print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()
    return fig, ax


def add_power_law_line(ax, slope, x_lim, y_pos=1, c="black", linestyle="--", **kwargs):
    x = np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), 100)
    y = y_pos * x**slope
    ax.plot(x, y, label=rf"fit: $\alpha={slope}$", c=c, linestyle=linestyle, **kwargs)
    ax.legend()


def makeLogPlotComparison(
    grouped_csv_file_paths,
    Y="Avg energy",
    name="",
    show=False,
    slide=False,
    window=False,
    windowRadius=0.1,
    dropLim=[None, None],
    outerStrainLims=[-np.inf, np.inf],
    innerStrainLims=(0.45, 0.6),
    use_title=False,
    use_y_axis_name=True,
    ylim=None,
    ax=None,
    fig=None,
    save=True,
    show_lambda=False,
    plot_pre_yield=True,
    plot_post_yield=True,
    mark=None,
    mark_pos=(0.8, 0.95),
    mark_fontsize=17,
    add_fit=True,
    legend_loc="best",
    dist="truncated_power_law",
    **kwargs,
):
    global color_index, index, line_index
    color_index, index, line_index = 0, 0, 0

    if fig is None or ax is None:
        assert fig is None
        assert ax is None
        fig, ax = plt.subplots()

    X = "Load"
    oLims = (
        ""
        if outerStrainLims == [-np.inf, np.inf]
        else f" oLims: {outerStrainLims[0]}-{outerStrainLims[1]}, "
    )
    iLims = (
        ""
        if innerStrainLims == [np.inf, -np.inf]
        else f" iLims: {innerStrainLims[0]}-{innerStrainLims[1]}, "
    )
    title = f"{name}{oLims},"

    if name == "":
        if Y == "Avg energy":
            name == "Avg energy power law"
            unit = "E"
        elif Y == "Avg RSS":
            name == "Avg stress power law"
            unit = r"\sigma"

    if slide:
        x_name = "Energy cutoff"
        # Use LaTeX format for Greek letter alpha
        y_name = "Exponent for post yield $\\alpha$"
        name += " slide"
    if window:
        x_name = "Strain center"
        # Use LaTeX format for Greek letter alpha
        y_name = "Exponent $\\alpha$"
        title += f" wr={windowRadius},"
        ax2 = ax.twinx()
        # Use LaTeX format for Greek letter lambda
        ax2.set_ylabel("Cutoff $\\lambda$")
        name += " window"
    else:
        title += f" {iLims},"
        if Y == "Avg energy":
            x_name = "Magnitude of energy drops"
        elif Y == "Avg RSS":
            x_name = "Magnitude of stress drops"
        y_name = rf"$p(\Delta \langle {unit} \rangle)$"

    title += (
        r" $\\Delta "
        + f"{unit}"
        + "_{\\mathrm{min}}$="
        + f"{dropLim[0]}"
        + f"{unit}"
        + "_{\\mathrm{max}}$="
        + f"{dropLim[1]}"
    )

    crash_count = 0
    # For each configuration
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        dfs = []
        # for each seed using this config
        for j, csv_file_path in enumerate(csv_file_paths):
            df = pd.read_csv(
                csv_file_path,
                usecols=[
                    X,
                    Y,
                    "Nr plastic deformations",
                    "Max energy",
                    # "Avg energy change",
                ],
            )
            # Truncate data based on xlim
            df = df[(df[X] >= outerStrainLims[0]) & (df[X] <= outerStrainLims[1])]
            df, crash_count = removeBadData(df, crash_count, csv_file_path)
            dfs.append(df)
        label = getPrettyLabel(kwargs["labels"][i][0])
        log_kwargs = {
            "fig": fig,
            "ax": ax,
            "dist": dist,
            "label": label,
            "plot_pre_yield": plot_pre_yield,
            "plot_post_yield": plot_post_yield,
            "add_fit": add_fit,
        }

        if slide:
            fig, ax = plotSlidingPowerLaw(dfs, **log_kwargs)
        if window:
            fig, ax, ax2 = plotSlidingWindowPowerLaw(
                dfs,
                dist=dist,
                dropLim=dropLim,
                windowRadius=windowRadius,
                fig=fig,
                ax1=ax,
                ax2=ax2,
                label=label,
            )
        else:
            fig, ax = plotSplitPowerLaw(
                dfs,
                dropLim=dropLim,
                innerStrainLims=innerStrainLims,
                outerStrainLims=outerStrainLims,
                **log_kwargs,
            )
        if crash_count > 0:
            print(f"Found {crash_count} crashes using {label}.")
    # Set the legend with the filtered handles and labels
    if legend_loc:
        ax.legend(loc=legend_loc)
    else:
        if window:
            ax.legend(loc=("lower left"))
            ax2.legend(loc=("center right"))
        else:
            ax.legend(loc=("best"))

    if window and not show_lambda:  # Hide second axis (with lambda)
        hide_twinx_axis(ax2)

    if ylim:
        ax.set_ylim(ylim)

    if mark:
        assert mark_pos is not None
        add_mark(ax, f"({mark})", *mark_pos, fontsize=mark_fontsize)

    ax.set_xlabel(x_name)
    if use_y_axis_name:
        ax.set_ylabel(y_name)
    if use_title:
        ax.set_title(title)

    if save:
        # # Get the parent directory of the CSV file
        # csv_directory = Path(grouped_csv_file_paths[0][0]).parent.parent
        # # Move to the "plots" directory relative to the CSV file directory
        # plotPath = csv_directory / "plots"

        # figPath = os.path.join(plotPath, name + ".pdf")
        # fig.savefig(figPath)
        fig.savefig("Plots/" + name + ".pdf")
        # print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()
    return fig, ax


# Function to hide twinx axis
def hide_twinx_axis(ax_twin):
    # Set visibility of the twin axis to False
    ax_twin.set_visible(False)

    # Hide the ticks and labels
    ax_twin.tick_params(axis="both", which="both", length=0)  # Hide tick marks
    ax_twin.set_yticklabels([])  # Hide tick labels

    # Hide the twin axis' spines (the lines surrounding the axis)
    ax_twin.spines["right"].set_visible(False)

    # Remove any plot lines or elements associated with the twinx axis
    for line in ax_twin.get_lines():
        line.set_visible(False)


def makeEnergyAvalancheComparison(
    grouped_csv_file_paths,
    name,
    xlim=[-np.inf, np.inf],
    show=False,
    **kwargs,
):
    global color_index, index, line_index
    color_index, index, line_index = 0, 0, 0
    X = "Load"
    Y = "Avg energy"
    x_name = "Magnitude of energy drops"
    y_name = r"$P(>E)$"
    lims = "" if xlim == [-np.inf, np.inf] else f", xlim: {xlim[0]}-{xlim[1]}"

    crash_count = 0

    # for each config
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        dfs = []  # panda dataframes
        # for each seed using this config
        for j, csv_file_path in enumerate(csv_file_paths):
            df = pd.read_csv(
                csv_file_path, usecols=[X, Y, "Nr plastic deformations", "Max energy"]
            )
            # Truncate data based on xlim
            df = df[(df[X] >= xlim[0]) & (df[X] <= xlim[1])]
            # Truncate data based on dislocations
            df, crash_count = removeBadData(df, crash_count, csv_file_path)
            dfs.append(df)
        fig, ax = plotEnergyAvalancheHistogram(dfs)

        if crash_count > 0:
            print(f"Found {crash_count} crashes using {["L-BFGS", "CG", "FIRE"][i]}.")
        # Set the legend with the filtered handles and labels
        # ax.legend(loc=("best"))
        # ax.set_xlabel(x_name)
        # ax.set_ylabel(y_name)
        title = f"{name}{lims}" + f'-{["L-BFGS", "CG", "FIRE"][i]}'
        fig.suptitle(title, fontsize=16)  # Set the main
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        # Get the parent directory of the CSV file
        csv_directory = Path(grouped_csv_file_paths[0][0]).parent.parent
        # Move to the "plots" directory relative to the CSV file directory
        plotPath = csv_directory / "plots"

        figPath = os.path.join(plotPath, title + ".pdf")
        fig.savefig(figPath)
        fig.savefig("Plots/" + title + ".pdf")
        # print(f'Plot saved at: "{figPath}"')
        print(f'Plot saved at: "{figPath}"')
        if show:
            plt.show()


def makeItterationsPlot(csv_file_paths, name, **kwargs):
    if isinstance(csv_file_paths, str):
        makePlot(
            csv_file_paths,
            name,
            X="Load",
            Y=["Nr FIRE iterations", "Nr LBFGS iterations"],
            y_name="Nr itterations",
            title="Nr of Itterations",
            plot_raw=True,
            plot_roll_average=False,
            plot_total=True,
            **kwargs,
        )
    else:
        makePlot(
            csv_file_paths,
            name,
            X="Load",
            Y=["Nr FIRE iterations", "Nr LBFGS iterations"],
            y_name="Nr itterations",
            title="Nr of Itterations",
            plot_raw=True,
            plot_roll_average=False,
            plot_total=False,
            **kwargs,
        )


def makeTimePlot(csv_file_paths, name, **kwargs):
    makePlot(
        csv_file_paths,
        name,
        x_name="Settings",
        Y=["Run time"],
        y_name="Run time (s)",
        plot_raw=False,
        plot_columns=True,
        title="Runtime of simulations",
        **kwargs,
    )


def makePowerLawPlot(csv_file_paths, name, **kwargs):
    makePlot(
        csv_file_paths,
        name,
        X="Load",
        Y="Avg energy",
        x_name="Magnitude of energy drops",
        y_name=r"$P(>E)$",
        title="Powerlaw",
        plot_raw=False,
        plot_power_law=True,
        plot_average=False,
        **kwargs,
    )


if __name__ == "__main__":
    pass
    # The path should be the path from work directory to the folder inside the output folder.
    makePlot(
        [
            "/Volumes/data/MTS2D_output/simpleShear,s60x60l0.15,0.0002,1.0PBCt1minimizerFIRELBFGSEpsg0.0001eps0.01s0/macroData.csv",
        ],
        name="energy.pdf",
        Y="Avg energy",
    )
    # makeItterationsPlot(
    #     [
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
    #     ],
    #             name='nrIterations.pdf')

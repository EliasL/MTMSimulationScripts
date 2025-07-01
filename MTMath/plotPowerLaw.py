import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
import powerlaw
from tqdm import tqdm
from scipy.optimize import curve_fit

from concurrent.futures import ProcessPoolExecutor
import functools
import os
import glob

np.random.seed(0)
# Create directories for saving plots
PLOTPATH = "Plots/powerLaw/"
OUTPUTTYPE = ".png"
os.makedirs(PLOTPATH, exist_ok=True)
os.makedirs(PLOTPATH + "debug/", exist_ok=True)
MINIMIZER_COLORS = {"L-BFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}


def trim_to_range(data, xmin=None, xmax=None, **kwargs):
    """
    Removes elements of the data that are above xmin or below xmax (if present)
    """
    from numpy import asarray

    data = asarray(data)
    if xmin:
        data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]
    return data


def cumulative_distribution_function(
    data, xmin=None, xmax=None, survival=False, **kwargs
):
    """
    The cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). False by default.
    xmin : int or float, optional
        The minimum data size to include. Values less than xmin are excluded.
    xmax : int or float, optional
        The maximum data size to include. Values greater than xmin are
        excluded.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than or equal to X.
    """

    from numpy import array

    data = array(data)
    if not data.any():
        from numpy import nan

        return array([nan]), array([nan])

    data = trim_to_range(data, xmin=xmin, xmax=xmax)

    n = float(len(data))
    from numpy import sort

    data = sort(data)
    all_unique = not (any(data[:-1] == data[1:]))

    if all_unique:
        from numpy import arange

        CDF = arange(n) / n
    else:
        # This clever bit is a way of using searchsorted to rapidly calculate the
        # CDF of data with repeated values comes from Adam Ginsburg's plfit code,
        # specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
        from numpy import searchsorted, unique

        CDF = searchsorted(data, data, side="left") / n
        unique_data, unique_indices = unique(data, return_index=True)
        data = unique_data
        CDF = CDF[unique_indices]

    if survival:
        CDF = 1 - CDF
    return data, CDF


def power_law_ks_distance(data, alpha, xmin, xmax=None, discrete=False, kuiper=False):
    from numpy import arange, sort, mean

    data = data[data >= xmin]
    if xmax:
        data = data[data <= xmax]
    n = len(data)
    if n < 2:
        if kuiper:
            return 1, 1, 2, None, None
        return 1, None, None

    if not all(data[i] <= data[i + 1] for i in arange(n - 1)):
        data = sort(data)

    if not discrete:
        Actual_CDF = arange(n) / float(n)
        Theoretical_CDF = 1 - (data / xmin) ** (-alpha + 1)

    if discrete:
        from scipy.special import zeta

        if xmax:
            bins, Actual_CDF = cumulative_distribution_function(
                data, xmin=xmin, xmax=xmax
            )
            Theoretical_CDF = 1 - (
                (zeta(alpha, bins) - zeta(alpha, xmax + 1))
                / (zeta(alpha, xmin) - zeta(alpha, xmax + 1))
            )
        else:
            bins, Actual_CDF = cumulative_distribution_function(data, xmin=xmin)
            Theoretical_CDF = 1 - (zeta(alpha, bins) / zeta(alpha, xmin))

    diffs_plus = Theoretical_CDF - Actual_CDF
    diffs_minus = Actual_CDF - Theoretical_CDF

    D_plus = max(diffs_plus)
    D_minus = max(diffs_minus)

    D_plus_index = diffs_plus.argmax()
    D_minus_index = diffs_minus.argmax()

    Kappa = 1 + mean(Theoretical_CDF - Actual_CDF)

    if kuiper:
        return D_plus, D_minus, Kappa, D_plus_index, D_minus_index

    D = max(D_plus, D_minus)
    return D, D_plus_index, D_minus_index


# --- Truncated power-law KS distance helper ---
def truncated_power_law_ks_distance(data, dist):
    """
    Kolmogorov–Smirnov distance D, plus the indices of the maximum
    positive (D+) and negative (D–) deviations, for a *truncated* power‑law.

    Parameters
    ----------
    data : array‑like
        Raw sample. Only the tail (x >= dist.xmin) is used.
    dist : powerlaw.Truncated_Power_Law
        The fitted distribution instance whose .ccdf() gives the model CCDF.

    Returns
    -------
    D : float
        KS distance.
    D_plus_idx : int
        Index (in the sorted tail data array) where D+ occurs.
    D_minus_idx : int
        Index where D– occurs.
    """
    # Tail only
    xmin = dist.xmin
    data = np.asarray(data)
    data = data[data >= xmin]
    n = len(data)
    if n < 2:
        return np.nan, None, None

    sorted_data = np.sort(data)

    # Empirical left‑continuous CDF (same as powerlaw library)
    emp_cdf = np.arange(n) / float(n)

    # Theoretical CDF from the fitted *truncated* model
    model_cdf = 1.0 - dist.ccdf(sorted_data)

    diffs_plus = model_cdf - emp_cdf
    diffs_minus = emp_cdf - model_cdf

    D_plus_idx = diffs_plus.argmax()
    D_minus_idx = diffs_minus.argmax()

    D_plus = diffs_plus[D_plus_idx]
    D_minus = diffs_minus[D_minus_idx]

    D = max(D_plus, D_minus)
    return D, D_plus_idx, D_minus_idx


def get_energy_drops(
    csvPath, df=None, strainLim=[-np.inf, np.inf], debug=False, label=None
):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    if df is None:
        df = pd.read_csv(csvPath)

    if "avg_energy_change" not in df:
        # Add 0 in the beginning
        diffs = np.insert(np.diff(df["avg_energy"]), 0, 0)
    else:
        diffs = df["avg_energy_change"]

    strain = df["load"]
    lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = -diffs[mask]
    if debug:
        # Only debug first seed when using labels
        if label is not None and "seed=0" not in label:
            return drops

        strain_limited = strain[1:][lim_mask[1:]]
        plotDrops = np.clip(-diffs[1:][lim_mask[1:]], 0, np.inf)
        e = df["avg_energy"]
        debug_fig, ax1 = plt.subplots()
        ax1.plot(strain, e, label=r"$\langle E \rangle$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        ax2.plot(strain_limited, plotDrops, label=r"$-\Delta \langle E \rangle$")
        ax2.set_ylabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        ax2.set_ylim(0, drops.max() * 1.5)

        # ——— Compute 0.1%‐wide central slice ———
        mid = 0.5 * (strainLim[0] + strainLim[1])
        total_width = strainLim[1] - strainLim[0]
        slice_width = total_width * 0.05  # 1% of window
        x1, x2 = mid - slice_width / 2, mid + slice_width / 2
        zoom_mask = (strain >= x1) & (strain <= x2)

        # find energy‐axis extents in that slice
        y1, y2 = (-diffs[zoom_mask]).min(), (-diffs[zoom_mask]).max()
        zoomWidth = np.clip(x2 - x1, 0.01, None)
        zoomHeight = y2 - y1

        # draw red dashed box on main axes
        rect = Rectangle(
            (x1 - zoomWidth * 0.5, y1),  # lower‐left corner
            zoomWidth * 1.5,  # width
            zoomHeight,  # height
            linewidth=2,
            edgecolor="black",
            linestyle="--",
            facecolor="none",
            zorder=10,
        )
        ax2.add_patch(rect)

        # ——— Inset axes at top middle-left ———
        axins = inset_axes(
            ax1,
            width=1.5,
            height=0.7,
            loc="center",
            bbox_to_anchor=(0.5, 0.7, 0.0, 0.30),
            bbox_transform=ax1.transAxes,
        )
        # plot energy in inset
        axins.plot(strain[zoom_mask], e[zoom_mask], lw=0.8)
        axins.set_xlim(x1, x2)
        axins.set_title("Zoom", fontsize=8)

        # twin‐axis for drops in the inset
        axins2 = axins.twinx()
        zoom_strain_mask = strain_limited >= x1
        zoom_strain_mask &= strain_limited <= x2
        drops_zoom = plotDrops[zoom_strain_mask]
        axins2.plot(strain_limited[zoom_strain_mask], drops_zoom)
        axins2.set_ylim(0, drops_zoom.max() * 1.5)

        debug_fig.tight_layout()
        # Save debug energy plot
        minimizer = get_minimizer(label)
        filename = f"{PLOTPATH}debug/{minimizer}_{csvPath.split('/')[-1]}_energy_drops_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}{OUTPUTTYPE}"
        debug_fig.savefig(filename, dpi=300)
        # to save memory, close the figure
        plt.close(debug_fig)
    return drops


def plot_data(
    ax, fit=None, data=None, xmin=None, label="Energy drops", edgecolor="black", alpha=1
):
    raise RuntimeError("Don't use this function")
    if data is None and fit is not None:
        data = fit.data_original
    elif fit is None and data is not None:
        fit = powerlaw.Fit(data, xmin=xmin)
    else:
        raise ValueError("Either data or fit must be provided.")

    if edgecolor is None:
        # Automatically select the next color in the Matplotlib cycle when edgecolor is None
        edgecolor = ax._get_lines.get_next_color()
    # full-data empirical
    fit.plot_ccdf(
        ax=ax,
        marker="o",
        linestyle="None",
        label=label,
        original_data=True,
        facecolor="none",
        edgecolor=edgecolor,
        alpha=alpha,
    )
    return fit


def getHist(fit):
    data = fit.data_original
    # Find the start of the tail where Poisson noise exceeds threshold
    data_min = min(data)
    data_max = data.max()

    # Compute number of bins from x_min to data_max
    bins_per_decade = 5
    decades = np.log10(data_max) - np.log10(data_min)
    n_bins = int(np.ceil(decades * bins_per_decade))
    # Define bin edges from data_max downward
    log_edges = np.log10(data_max) - np.arange(n_bins + 1) / bins_per_decade
    bin_edges = np.power(10, log_edges)[::-1]  # Reverse to make it ascending

    # Compute the histogram for the tail (density=True → area under PDF = 1)
    hist_vals, edges = np.histogram(data, bins=bin_edges, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, hist_vals


def plot_data_pdf(
    ax,
    fit=None,
    data=None,
    label="Binned PDF of energy drops",
    edgecolor="black",
    alpha=1,
    color=None,
):
    """
    Plot the empirical PDF of the data on log–log axes using logarithmic bins.
    Automatically identifies x_min via find_x_min and uses 0.1-decade bin widths.
    If `fit` is provided and `data` is None, use `fit.data_original`.
    """
    # Determine which to use: fit or raw data
    if data is None and fit is not None:
        data = fit.data_original
    else:
        raise ValueError("Either data or fit must be provided (but not both).")

    # Choose edgecolor if None
    if edgecolor is None:
        edgecolor = ax._get_lines.get_next_color()

    bin_centers, hist_vals = getHist(fit)

    # Plot as points
    plot_kwargs = {
        "marker": "o",
        "linestyle": "None",
        "label": label,
        "alpha": alpha,
    }
    if color is not None:
        plot_kwargs["color"] = color
    ax.plot(
        bin_centers,
        hist_vals,
        **plot_kwargs,
    )

    # Set log–log axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    ax.set_ylabel(r"$p(-\Delta \langle E \rangle)$")
    ax.legend()
    return bin_centers, hist_vals


def plot_fit(
    ax,
    fit,
    dist_name="truncated_power_law",
    title=None,
    color=None,
    pre_label=None,
    label=None,
    alpha=1,
    linestyle="-",
):
    # compute weight and x-grid
    data = fit.data_original
    xmin = fit.xmin
    x_vals = np.logspace(
        np.log10(data.min()),  # start at xmin
        np.log10(data.max()),
        num=200,
    )

    dist = getattr(fit, dist_name)
    CDF = dist._cdf_base_function(x_vals)
    CCDF = 1 - CDF

    # Area under  CCDF in the fit region
    # try to use data from ax
    if len(ax.collections) > 0:
        x = ax.collections[0].get_offsets()[:, 0]  # CCDF x values
        mask = x > xmin
        empirical_area = np.trapezoid(
            ax.collections[0].get_offsets()[:, 1][mask],  # CCDF area
            x=x[mask],  # CCDF x values
        )  # CCDF area

        mask = x_vals > xmin
        # Area under fitted CCDF in the fit region
        fitted_area = np.trapezoid(CCDF[mask], x=x_vals[mask])  # fitted CCDF area
        # Scale the fitted CCDF to match the CCDF area
        CCDF = CCDF * empirical_area / fitted_area

    if label is None:
        label = f"{dist_name}: "
        params = zip(
            [
                dist.parameter1_name,
                dist.parameter2_name,
                dist.parameter3_name,
            ],
            [
                dist.parameter1,
                dist.parameter2,
                dist.parameter3,
            ],
        )
        for name, p in params:
            if name is not None:
                if name == "lambda":
                    label += f"1/{name}={(1 / p):.2e}, "
                else:
                    label += f"{name}={p:.3f}, "
                # print(f"{name}={p:.3f}")
            # For some reason, power_law does not have any parameters
            if dist_name == "power_law":
                label += f"alpha={dist.alpha:.3f}, "
                break
        # remove last comma
        label = label[:-2]
    if pre_label:
        label = pre_label + label

    ax.plot(
        x_vals,
        CCDF,
        label=pretty_label(label),
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    ax.set_ylabel(r"$P(X > x)$")
    ax.set_title(title)
    ax.legend(loc="lower left")


def plot_ks_distance_marker(ax, sorted_data, ecdf, model_ccdf, color="red"):
    diffs = np.abs(ecdf - model_ccdf)
    max_index = np.argmax(diffs)
    D_val = diffs[max_index]
    x_D = sorted_data[max_index]
    ax.vlines(
        x_D,
        model_ccdf[max_index],
        ecdf[max_index],
        color=color,
        linestyle="--",
        label=f"KS Distance D = {D_val:.3f}",
    )
    ax.scatter([x_D], [ecdf[max_index]], color="blue")
    ax.scatter([x_D], [model_ccdf[max_index]], color="gray")
    return D_val


# --- Helper for annotating KS distance on PDF plot ---
def annotate_ks_distance_pdf(ax, x_D, D_val, color="red"):
    """
    Mark the KS distance location on a PDF plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to draw.
    x_D : float
        Data value where the maximum KS distance occurs.
    D_val : float
        The KS distance (D statistic).
    color : str, optional
        Color for the marker and line.
    """
    # Vertical dashed line at x_D
    ax.axvline(
        x_D,
        color=color,
        linestyle="--",
        linewidth=1.2,
        label=f"KS Distance D = {D_val:.3f}",
    )


def make_and_plot_truncated_fit_pdf(
    ax,
    fit,
    bin_centers,
    hist_values,
    title=None,
    color=None,
    label=None,
    alpha=1,
    linestyle="-",
    truncated=True,
    pre_label=None,
    add_ks_marker=False,
    alpha_std=None,
):
    mask = (bin_centers > fit.xmin) & (hist_values > 0)
    xdata = bin_centers[mask]
    ydata = hist_values[mask]

    log_y = np.log(ydata)

    if truncated:
        # Truncated power law: log(y) = -alpha * log(x) - Lambda * x + logC
        def log_model(x, alpha, Lambda, logC):
            return -alpha * np.log(x) - Lambda * x + logC

        popt, _ = curve_fit(log_model, xdata, log_y, p0=[1.5, 1e-3, 0])
        alpha_fit, lambda_fit, logC_fit = popt
        y_fit = np.exp(log_model(bin_centers, *popt))
        if label is None:
            err = f" \\pm {alpha_std:.2f}" if alpha_std is not None else ""
            fit_label = (
                rf"Trunc. Fit: $\alpha={alpha_fit:.2f}{err}, \lambda={lambda_fit:.2f}$"
            )
        else:
            fit_label = label

    else:
        # Pure power law: log(y) = -alpha * log(x) + logC
        def log_model(x, alpha, logC):
            return -alpha * np.log(x) + logC

        popt, _ = curve_fit(log_model, xdata, log_y, p0=[1.5, 0])
        alpha_fit, logC_fit = popt
        y_fit = np.exp(log_model(bin_centers, *popt))
        if label is None:
            err = f" \\pm {alpha_std:.2f}" if alpha_std is not None else ""
            fit_label = rf"Powerlaw Fit: $\alpha={alpha_fit:.2f}{err}$"
        else:
            fit_label = label

    # Plot
    ax.plot(
        bin_centers,
        y_fit,
        label=(pre_label or "") + fit_label,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )
    if add_ks_marker:
        dist = fit.truncated_power_law
        ks_distance, D_plus_idx, D_minus_idx = truncated_power_law_ks_distance(
            fit.data_original, dist
        )

        # decide which index gives the max absolute deviation
        sorted_tail = np.sort(fit.data[fit.data >= dist.xmin])
        if ks_distance == (
            dist.cdf(sorted_tail)[D_plus_idx]
            - np.arange(len(sorted_tail))[D_plus_idx] / float(len(sorted_tail))
        ):
            max_idx = D_plus_idx
        else:
            max_idx = D_minus_idx

        x_D = sorted_tail[max_idx]
        annotate_ks_distance_pdf(ax, x_D, ks_distance)

        # Assert agreement with powerlaw's stored D
        assert np.isclose(ks_distance, dist.D, rtol=1e-9, atol=1e-12), (
            f"KS distance mismatch: computed {ks_distance:.6g}, powerlaw {dist.D:.6g}"
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    ax.set_ylabel(r"$p(-\Delta \langle E \rangle)$")
    ax.set_title(title)
    ax.legend()

    return ax


def compare_dists(fit):
    mainDist = "truncated_power_law"
    compareDists = ["lognormal", "power_law"]
    # Compare distributions
    for dist_name in compareDists:
        # Likelihood ratio test comparing mainDist to dist_name
        R, p = fit.distribution_compare(mainDist, dist_name, normalized_ratio=True)
        print(
            f"Likelihood ratio test ({mainDist} vs {dist_name}): R={R:.3f}, p={p:.3e}"
        )


def pretty_label(label):
    r"""
    Make the lable nicer.
    truncated_power_law: alpha=1.02, lambda=0.5
    ->
    Truncated Power Law: \alpha=1.02, \lambda=0.5
    """
    label = label.replace("_", " ")
    label = label.replace("alpha", r"$\alpha$")
    label = label.replace("lambda", r"$\lambda$")
    label = label.replace("mu", r"$\mu$")
    label = label.replace("sigma", r"$\sigma$")
    # Captialize first letter of the first word
    label = label.capitalize()
    return label


def get_drops_in_windows(
    csvPath=None,
    strainLim=None,
    df=None,
    steps=1,
    window_width=np.inf,
    debug=False,
    label=None,
):
    if df is None:
        df = pd.read_csv(csvPath)
    strain = df["load"]
    if strainLim is not None:
        lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
        df = df[lim_mask]
        strain = df["load"]

    global_max_strain = strain.max()
    global_min_strain = strain.min()
    if window_width == np.inf:
        window_width = global_max_strain - global_min_strain

    if global_max_strain - global_min_strain < window_width:
        centers = [global_min_strain + window_width / 2]
    else:
        # get list of window centers
        centers = np.linspace(
            global_min_strain + window_width / 2,
            global_max_strain - window_width / 2,
            steps,
        )

    drops_in_windows = []
    windows = []
    for center in centers:
        # get the window
        min_strain = center - window_width / 2
        max_strain = center + window_width / 2
        # get the data in the window
        drops = get_energy_drops(
            csvPath,
            df=df,
            strainLim=[min_strain, max_strain],
            debug=debug,
            label=label,
        )
        windows.append((min_strain, max_strain))
        drops_in_windows.append(drops)
    return drops_in_windows, windows, centers


def plot_data_and_fit(
    fit,
    ax=None,
    xmin=None,
    title="",
    dist_names=[
        "truncated_power_law",
        # "lognormal",
        # "power_law",
        # "exponential",
        # "stretched_exponential",
        # "lognormal_positive",
    ],
    pdf=True,
    alpha_std=None,
    p_val=None,
    color=None,
    addFit=True,
):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the data
    if pdf:
        bin_centers, hist_values = plot_data_pdf(ax, fit=fit, color=color)
    else:
        plot_data(ax, fit=fit)

    cmap_colors = ["green", "red", "yellow", "orange", "blue", "cyan"]
    # plot the fit
    if addFit:
        for dist_name, color_fit in zip(dist_names, cmap_colors):
            # If color is specified, use it for both data and fit, otherwise use colormap for fit
            fit_color = color if color is not None else color_fit
            if pdf:
                make_and_plot_truncated_fit_pdf(
                    ax,
                    fit,
                    bin_centers,
                    hist_values,
                    title=title,
                    color=fit_color,
                    alpha_std=alpha_std,
                )
            else:
                plot_fit(
                    ax,
                    fit,
                    dist_name=dist_name,
                    title=title,
                    color=fit_color,
                )

        # Add shaded fit region with formula in label
        ax.axvspan(
            xmin,
            fit.data.max(),
            color="gray",
            alpha=0.2,
            label=r"Fit region ($x \geq E_\mathrm{min}$, $p(x) \propto x^{-\alpha} e^{-\lambda x}$)",
        )
    ax.legend()
    ax.set_title(title)
    if p_val is not None:
        ax.set_title(ax.get_title() + f" p: {p_val:.2f}")
    return ax


def get_window_power_law_exponents(
    xmin=-np.inf,
    dist="truncated_power_law",
    syntheticData=False,
    syntheticExponent=1.0,
    **kwargs,
):
    """
    We slide this window over the data and plot the power law fit for each window.
    """
    drops_in_windows, windows, centers = get_drops_in_windows(**kwargs)
    fits = []
    ps = []
    debug = kwargs.get("debug", False)
    for drops, strainLim in zip(drops_in_windows, windows):
        if syntheticData:
            # We generate synthetic data instead of using the real data
            drops = create_synthetic_data(
                drops,
                xmin=xmin,
                nrSets=1,
                dist=dist,
                params={"alpha": syntheticExponent},
            )[0]
        # fit the data
        fit = powerlaw.Fit(drops, xmin=xmin)
        fits.append(fit)
        p, exp, std = evaluate_fit(
            drops,
            xmin=xmin,
            dist_name=dist,
            parallel=False,
            verbose=True,
            debug=debug,
        )
        ps.append(p)

        if debug:
            debug_fig, debug_ax = plt.subplots()
            title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"

            plot_data_and_fit(fit, debug_ax, xmin, title)
            debug_fig.tight_layout()
            debug_fig.show()
            # Save debug window power law plot
            filename = f"{PLOTPATH}window_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_xmin_{xmin:.2e}{OUTPUTTYPE}"
            debug_fig.savefig(filename)
            # to save memory, close the figure
            plt.close(debug_fig)

    # plot the exponents against the window centers
    exponents = [fit.truncated_power_law.alpha for fit in fits]

    return centers, exponents, ps


def worker_get_exponents(xmin, kwargs):
    import numpy as np

    seed = int((np.log10(xmin) * 1e6) % (2**32))  # Stable and unique
    np.random.seed(seed)
    return get_window_power_law_exponents(xmin=xmin, **kwargs)


def get_power_law_surface(xmins=None, **kwargs):
    # If we debug, we don't use multiprocessing
    if kwargs.get("debug", False):
        exponent_xmin_surface, p = [], []
        for xmin in tqdm(xmins):
            centers, exponents, p_ = get_window_power_law_exponents(xmin=xmin, **kwargs)
            exponent_xmin_surface.append(exponents)
            p.append(p_)
        return centers, np.array(exponent_xmin_surface), np.array(p)
    else:
        # Do the first call without the pool
        # centers, exponents, p = get_window_power_law_exponents(xmin=xmins[0], **kwargs)
        # Pre-bind kwargs using functools.partial
        with ProcessPoolExecutor() as executor:
            bound_worker = functools.partial(worker_get_exponents, kwargs=kwargs)
            results = list(tqdm(executor.map(bound_worker, xmins), total=len(xmins)))

        centers = results[0][0]  # All share same centers
        exponent_xmin_surface = np.array([r[1] for r in results])
        p = np.array([r[2] for r in results])
        return centers, exponent_xmin_surface, p


def plot_power_law_map(
    csvPath=None,
    xmins=None,
    df=None,
    strainLim=[-np.inf, np.inf],
    window_steps=20,
    window_width=0.4,
    debug=False,
    use_confidence_color=False,
    syntheticData=False,
    syntheticExponent=1.0,
):
    """
    Takes a csvPath or an already loaded file as a pandas dataframe (df)
    """

    # convert exponents to numpy array
    centers, exponent_xmin_surface, p = get_power_law_surface(
        csvPath=csvPath,
        xmins=xmins,
        df=df,
        strainLim=strainLim,
        steps=window_steps,
        window_width=window_width,
        debug=debug,
        syntheticData=syntheticData,
        syntheticExponent=syntheticExponent,
    )

    # Now we can plot a surface of the exponents on the z axis, centers on the x axis, and xmins on the y axis
    fig = plt.figure()
    if use_confidence_color:
        ax = fig.add_subplot(projection="3d")
        ax.set_zlabel(r"$\alpha$ (Exponent)")
    else:
        ax = fig.add_subplot()

    ax.set_xlabel("Strain window center")
    ax.set_ylabel(r"$\log_{10}(\Delta E_{\mathrm{min}})$")  # Changed label

    figType = "p" if use_confidence_color else "exp"

    # Choose plotting logic based on confidence‐color flag
    if use_confidence_color:
        # use p to color the surface
        facecolors = cm.viridis(p)  # Use colormap for p values

        X, Y = np.meshgrid(centers, np.log10(xmins))
        ax.plot_surface(
            X,
            Y,
            exponent_xmin_surface,
            facecolors=facecolors,
            shade=False,
            antialiased=False,
        )
        # Add colorbar for p values
        norm = colors.Normalize(vmin=np.nanmin(p), vmax=np.nanmax(p))
        sm = cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, aspect=10)
        cbar.set_label(r"$p$")

    else:
        X, Y = np.meshgrid(centers, np.log10(xmins))
        pcm = ax.pcolormesh(
            X,
            Y,
            exponent_xmin_surface,
            shading="auto",
            cmap="viridis",
            norm=colors.Normalize(
                vmin=np.nanmin(exponent_xmin_surface),
                vmax=np.nanmax(exponent_xmin_surface),
            ),
        )
        fig.colorbar(pcm, ax=ax, pad=0.1, aspect=10, label=r"$\alpha$")

    fig.tight_layout()
    # Save final power law surface plot
    syntheticTag = "synthetic_" if syntheticData else ""
    filename = (
        f"{PLOTPATH}{syntheticTag}power_law_surface_"
        f"strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_"
        f"xmin_{xmins[0]:.2e}_{xmins[-1]:.2e}_"
        f"steps_{window_steps}_width_{window_width:.2f}_"
        f"{figType}{OUTPUTTYPE}"
    )
    if not debug:
        # No need to save 3x3 map
        fig.savefig(filename, dpi=300)
    plt.show()


def make_debug_plot(xmins, strainLim=None):
    # Create debug plot grids for each xmin
    for xmin in xmins:
        # Find all saved fit plots for this xmin
        pattern = f"{PLOTPATH}window_strain_*_xmin_{xmin:.2e}{OUTPUTTYPE}"
        fit_files = sorted(glob.glob(pattern))
        if not fit_files:
            print(f"No fit debug files found for xmin {xmin:.2e}")
            continue

        n = len(fit_files)
        fig, axes = plt.subplots(2, n, figsize=(n * 6, 10))

        for i, fit_file in enumerate(fit_files):
            base = os.path.basename(fit_file)
            # Extract the strain range string between "window_strain_" and "_xmin"
            strain_range = base[len("window_strain_") : base.find("_xmin")]

            # Strain and display the energy drops image
            energy_file = f"{PLOTPATH}energy_drops_strain_{strain_range}{OUTPUTTYPE}"
            if os.path.exists(energy_file):
                img_energy = plt.imread(energy_file)
                axes[0, i].imshow(img_energy)
            else:
                axes[0, i].text(0.5, 0.5, "Missing image", ha="center", va="center")
            axes[0, i].axis("off")
            # axes[0, i].set_title(f"Energy drops\nstrain {strain_range}")

            # Strain and display the fit plot
            img_fit = plt.imread(fit_file)
            axes[1, i].imshow(img_fit)
            axes[1, i].axis("off")
            # axes[1, i].set_title(f"x_{\mathrm{min}}={xmin:.2e}")

        # fig.suptitle(f"Debug plots for x_{\mathrm{min}}={xmin:.2e}")
        fig.tight_layout()
        # Save the debug plot
        debug_filename = f"{PLOTPATH}debug_fit_plots_xmin_{xmin:.2e}{OUTPUTTYPE}"
        fig.savefig(debug_filename)
        # plt.show()


def plot_ks_distance(drops, xmin, dist_name="truncated_power_law"):
    """
    Plot the empirical CCDF vs the fitted CCDF and visually show the KS distance (D).
    """
    # Fit the distribution
    fit = powerlaw.Fit(drops, xmin=xmin)
    dist = getattr(fit, dist_name)
    # Get the ECDF and model CCDF
    data = fit.data
    sorted_data = np.sort(data[data >= xmin])
    ecdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    model_ccdf = dist.ccdf(sorted_data)

    # Plotting
    fig, ax = plt.subplots()
    ax.step(sorted_data, ecdf, where="post", label="Empirical CCDF", color="blue")
    ax.plot(sorted_data, model_ccdf, label="Model CCDF", color="gray")
    D_val = plot_ks_distance_marker(ax, sorted_data, ecdf, model_ccdf)
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$")
    ax.set_ylabel(r"$P(X > x)$")
    ax.set_title("Kolmogorov–Smirnov Distance (CCDF)")
    ax.legend()
    fig.tight_layout()
    # Save the plot
    filename = f"{PLOTPATH}ks_distance_xmin_{xmin:.2e}_{dist_name}{OUTPUTTYPE}"
    fig.savefig(filename, dpi=300)
    # plt.show()


def scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng):
    from scipy.special import gammainc, gammaincinv

    k = 1.0 - alpha
    theta = 1.0 / Lambda

    Fmin = gammainc(k, xmin / theta)
    u = Fmin + (1.0 - Fmin) * rng.random(size)
    y = gammaincinv(k, u)
    x = theta * y
    return x


def truncatedPowerlawGenerator(xmin, alpha, Lambda, size, rng):
    """
    Generate `size` samples from the PDF
        f(x) ∝ x**(-alpha) * exp(-Lambda*x),  x >= xmin
    using vectorized rejection sampling with an exponential proposal.
    """
    result = np.empty(size, dtype=float)
    n_done = 0
    if alpha < 1:
        return scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng)
    else:
        while n_done < size:
            n_remain = size - n_done
            proposals = xmin + rng.exponential(scale=1.0 / Lambda, size=n_remain)
            accept_probs = (xmin / proposals) ** alpha
            u = rng.random(n_remain)
            mask = u < accept_probs
            n_accept = mask.sum()
            if n_accept:
                result[n_done : n_done + n_accept] = proposals[mask]
                n_done += n_accept
        return result


def create_synthetic_data(
    drops,
    xmin=-np.inf,
    nrSets=2500,
    dist_name="truncated_power_law",
    params={},
    debug=False,
):
    """
    Create synthetic data for testing the power law fitting.
    If not all parameters are given, it will use the fitted parameters from the
    original data.
    """
    fit_orig = powerlaw.Fit(drops, xmin=xmin)
    dist = getattr(fit_orig, dist_name)

    tailDrops = drops[drops >= xmin]
    nonTailDrops = drops[drops < xmin]
    # Create a local RNG for reproducibility, seed based on xmin
    rng = np.random.default_rng(int((np.log10(xmin) * 1e6) % (2**32)))

    if len(nonTailDrops) == 0:
        total_samples = nrSets * len(drops)
        samples = truncatedPowerlawGenerator(
            xmin=dist.xmin,
            alpha=dist.alpha,
            Lambda=dist.Lambda,
            size=total_samples,
            rng=rng,
        )
        return samples.reshape((nrSets, len(drops)))

    nrTailObservations = len(tailDrops)
    nrObservations = len(drops)
    p_tail = nrTailObservations / nrObservations
    tail_counts = rng.binomial(nrObservations, p_tail, size=nrSets)

    for key, value in params.items():
        if hasattr(dist, key) and value is not None:
            setattr(dist, key, value)

    non_tail = rng.choice(nonTailDrops, size=(nrSets, nrObservations), replace=True)
    total_tails = tail_counts.sum()
    all_tails = truncatedPowerlawGenerator(
        xmin=dist.xmin,
        alpha=dist.alpha,
        Lambda=dist.Lambda,
        size=total_tails,
        rng=rng,
    )
    offsets = np.concatenate([[0], np.cumsum(tail_counts)])
    syntheticSets = non_tail.copy()
    for i in range(nrSets):
        k = tail_counts[i]
        if k:
            start = offsets[i]
            end = offsets[i + 1]
            syntheticSets[i, :k] = all_tails[start:end]
    if debug:
        # plot three sample sets
        debug_fig, debug_axes = plt.subplots(1, 4, figsize=(18, 6))
        plot_data_and_fit(fit_orig, debug_axes[0], xmin, "Original data")
        for i, drops in zip(
            range(1, 4), syntheticSets[[0, len(syntheticSets) // 2, -1]]
        ):
            fit_synth = powerlaw.Fit(drops, xmin=xmin)

            title = rf"Synthetic set {i}, $\alpha$: {dist.alpha:.3f}, $E_{{\mathrm{{min}}}}$={xmin:.2e}"

            plot_data_and_fit(fit_synth, debug_axes[i], xmin, title, pdf=True)
        debug_fig.tight_layout()
        debug_fig.show()
        # Save debug window power law plot
        filename = f"{PLOTPATH}Synthetic_sets_xmin_{xmin:.2e}{OUTPUTTYPE}"
        debug_fig.savefig(filename)
        # to save memory, close the figure
        plt.close(debug_fig)

    return syntheticSets


def _compute_D_for_set(args):
    """
    Worker for goodnessOfFit parallelization:
      args = (synthetic_dataset, xmin, dist_name)
    Returns the KS‐distance D for that dataset.
    Note that we fit the synthetic data and evaluate
    it's KS distance with new parameters (alpha and lambda)

    Here is an analogy which helps justify why we should fit
    with new parameters:
    Say you and your friend are trying to determine who is better at darts.
    To answer the question, you agree to place a dot on the dart board
    and you both try to hit the dot. Fitting a distribution to data is a bit
    like trying to find out where the dot was, after the darts have been trown.
    If you first find the center of your throws and assume the dot must have
    been in the middle of your throws (giving you the best score), and then
    judge your friend based on the assumption that the dot was where it would be
    most advantageous to you, there will be a significant bias in your favour.

    We also return the alpha exponent. This can give us a expected std
    """
    s, xmin, dist_name = args
    fit_s = powerlaw.Fit(s, xmin=xmin)
    dist_s = getattr(fit_s, dist_name)
    return dist_s.D, dist_s.alpha


# --- KS p-value computation function ---
def goodnessOfFit(
    drops,
    synthetic_sets,
    xmin=-np.inf,
    dist_name="truncated_power_law",
    parallel=False,
    debug=False,
):
    # Fit the original data
    fit_orig = powerlaw.Fit(drops, xmin=xmin)
    dist_orig = getattr(fit_orig, dist_name)
    D_orig = dist_orig.D

    # print("Computing KS distances for synthetic sets...")
    if parallel:
        # build arg‐tuples so each worker knows xmin & dist_name
        args_list = [(s, xmin, dist_name) for s in synthetic_sets]
        with ProcessPoolExecutor() as executor:
            # tqdm can wrap the map if you like progress bars
            D_synth = list(
                tqdm(
                    executor.map(_compute_D_for_set, args_list),
                    total=len(synthetic_sets),
                )
            )
    else:
        D_synth = []
        for s in synthetic_sets:
            D_synth.append(_compute_D_for_set((s, xmin, dist_name)))
    D_synth = np.array(D_synth)  # shape (N, 2): column 0=D, column 1=alpha
    # Split into D values and alpha values
    D_vals = D_synth[:, 0]
    alpha_vals = D_synth[:, 1]
    # p-value: proportion of synthetic distances >= original distance
    p_value = np.mean(D_vals >= D_orig)
    mean_alpha = np.mean(alpha_vals)
    std_alpha = np.std(alpha_vals)

    if debug:
        fig, ax = plt.subplots()

        bin_centers, hist_values = plot_data_pdf(ax, fit=fit_orig, label="Real data")
        make_and_plot_truncated_fit_pdf(
            ax,
            fit_orig,
            bin_centers=bin_centers,
            hist_values=hist_values,
        )

        for i, s_drops in enumerate(synthetic_sets[[0, len(synthetic_sets) // 2, -1]]):
            fit_synth = powerlaw.Fit(s_drops, xmin=xmin)

            # synth_bin_centers, synth_hist_values = getHist(fit_synth)
            synth_bin_centers, synth_hist_values = plot_data_pdf(
                ax,
                fit=fit_synth,
                label=f"Synthetic sample {i}",
                alpha=0.2,
            )
            dist_synth = getattr(fit_orig, dist_name)
            synthD = dist_synth.D
            # plot the fit

            make_and_plot_truncated_fit_pdf(
                ax,
                fit_synth,
                bin_centers=synth_bin_centers,
                hist_values=synth_hist_values,
                pre_label=f"Synth fit {i} D:{synthD:.2f}_",
                alpha=0.2,
                linestyle="--",
            )

        # Add shaded fit region
        ax.axvspan(
            xmin, fit_orig.data.max(), color="gray", alpha=0.2, label="Fit region"
        )
        ax.legend()
        ax.figure.show()

    return p_value, mean_alpha, std_alpha


def _compute_exponent(args):
    sample, xmin, dist_name = args
    sample = np.ravel(sample)  # Ensure 1D
    fit = powerlaw.Fit(sample, xmin=xmin)
    return getattr(fit, dist_name).alpha


def exponentUncertainty(
    drops,
    xmin=-np.inf,
    dist_name="truncated_power_law",
    parallel=False,
    debug=False,
):
    import json
    import os

    drop_sum = np.sum(drops)
    nr_sets = 2500 if not debug else 3
    filename = f"bootstrapData/uncertainty_{drop_sum}_{xmin}_{nr_sets}.json"

    if os.path.exists(filename) and not debug:
        with open(filename, "r") as f:
            result = json.load(f)
        return result["mean"], result["std"]

    rng = np.random.default_rng(0)
    synthetic_sets = rng.choice(drops, size=(nr_sets, len(drops)), replace=True)

    if parallel:
        with ProcessPoolExecutor() as executor:
            args = [(s, xmin, dist_name) for s in synthetic_sets]
            exponents = list(
                tqdm(executor.map(_compute_exponent, args), total=len(args))
            )
    else:
        exponents = [
            _compute_exponent((s, xmin, dist_name)) for s in tqdm(synthetic_sets)
        ]

    mean_exp = float(np.mean(exponents))
    std_exp = float(np.std(exponents))

    os.makedirs("bootstrapData", exist_ok=True)
    with open(filename, "w") as f:
        json.dump({"mean": mean_exp, "std": std_exp}, f)

    return mean_exp, std_exp


def evaluate_fit(
    drops,
    xmin,
    dist_name="truncated_power_law",
    parallel=True,
    verbose=False,
    debug=False,
):
    if verbose:
        print("nr of drops:", len(drops))
    if len(drops) < 200:
        print("Warning: this is not a lot of data, the p-value might not be reliable.")
    drop_sum = np.sum(drops)

    # Check if p has already been calculated for these drops
    # by assuming that the mean is unique enough
    # The files should be stored in the bootstrapData folder and saved as a json file
    # with the name "p_{mean}_{xmin}.json"
    import os
    import json

    nr_sets = 2500
    if debug:
        nr_sets = 3
    p_file = f"bootstrapData/p_{drop_sum}_{xmin}_{nr_sets}.json"
    if os.path.exists(p_file) and not debug:
        with open(p_file, "r") as f:
            result = json.load(f)
        p = result["p"]
        mean_s_exp = result["mean"]
        std_s_exp = result["std"]
        if verbose:
            print(f"Strained p-value from {p_file}")
    else:
        if verbose:
            print("Generating synthetic data...")
        sets = create_synthetic_data(
            drops,
            xmin=xmin,
            nrSets=nr_sets,
            dist_name=dist_name,
            debug=debug,
        )
        p, mean_s_exp, std_s_exp = goodnessOfFit(
            drops, sets, xmin, dist_name=dist_name, parallel=parallel, debug=debug
        )
        # Save the p-value to a file
        with open(p_file, "w") as f:
            json.dump({"p": p, "mean": mean_s_exp, "std": std_s_exp}, f)
    # if verbose:
    #     print(
    #         f"p-value for fit: {p:.3f}, ie. {p * 100:.1f}% of synthetic sets had a worse fit"
    #     )
    #     print(
    #         "If p > 0.1, the fit is likely a good fit. (This also depends on the number of drops.)"
    #     )
    return p, mean_s_exp, std_s_exp


def make_exponent_map():
    # User parameters
    res = 10
    debug = True
    synthetic = False
    if debug:
        res = 3
        synthetic = False
    csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"

    df = pd.read_csv(csvPath)
    xmins = np.logspace(-6.5, -5, num=res, base=10)
    strainLim = [0.3, 3]
    window_steps = res
    window_width = 0.5
    plot_power_law_map(
        df=df,
        strainLim=strainLim,
        xmins=xmins,
        window_steps=window_steps,
        window_width=window_width,
        debug=debug,
        use_confidence_color=True,
        syntheticData=synthetic,
        syntheticExponent=1,
    )

    if debug:
        make_debug_plot(xmins, strainLim=strainLim)


def make_exponent_fit():
    # csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    # csvPath = "/Volumes/data/MTS2D_output/simpleShear,s400x400l0.15,1e-05,1.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    xmin = 1e-6
    strainLim = [1.0, 3.0]
    debug = False
    fig, ax = plt.subplots()
    drops, _, _ = get_drops_in_windows(csvPath, strainLim)
    drops = drops  # we only have one window, so we take the first one
    # drops = np.tile(drops, 10)
    # drops = create_synthetic_data(
    #     drops,
    #     xmin=xmin,
    #     nrSets=10,
    #     params={"alpha": 1},
    # )
    for d in drops:
        fit = powerlaw.Fit(d, xmin=xmin)
        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"
        p, alpha, std = evaluate_fit(d, xmin, parallel=True, debug=debug)
        print("averageExponent:", alpha)
        plot_data_and_fit(fit, ax, xmin, title, pdf=True, alpha_std=std, p_val=p)
        print("P value:", p)
        # plot_ks_distance(d, xmin)
    plt.show()

    filename = f"{PLOTPATH}simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")


def get_minimizer(label):
    d = {
        k.strip(): v.strip() for k, v in (item.split("=") for item in label.split(","))
    }
    minimizer = d["minimizer"]
    minimizer = minimizer.replace("LBFGS", "L-BFGS")
    return minimizer


def plot_powerlaw(
    algorithms_paths,
    alg_labels=None,
    strainLim=[0.15, 0.4],
    xmin=1e-6,
    debug=False,
    show=False,
    evaluate=True,
    save=True,
    addFit=True,
):
    for paths, labels in zip(algorithms_paths, alg_labels):
        fig, ax = plt.subplots()
        all_drops = []
        for path, label in zip(paths, labels):
            drops, _, _ = get_drops_in_windows(
                path, strainLim, debug=debug, label=label
            )
            all_drops.extend(drops)  # drops is a list of arrays

        # After the loop
        all_drops = np.concatenate(all_drops)

        # Remove large drops (something strange happened)
        all_drops = all_drops[all_drops < 0.05]

        fit = powerlaw.Fit(all_drops, xmin=xmin)
        xmin = fit.xmin
        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"
        minimizer = get_minimizer(labels[0])
        title = minimizer + " " + title
        color = MINIMIZER_COLORS[minimizer]

        if evaluate:
            # p = evaluate_fit(all_drops, xmin, parallel=True, debug=debug)

            p, mean_s_exp, std_s_exp = evaluate_fit(
                all_drops, xmin, parallel=True, debug=debug
            )

            mean_exp, std_exp = exponentUncertainty(
                all_drops, xmin, parallel=True, debug=debug
            )

            rating = ["bad", "poor", "good", "excellent"]
            scores = [0.05, 0.1, 0.3]
            for threshold, r in zip(scores, rating):
                if p < threshold:
                    break
            else:
                r = rating[-1]
            print(
                f"{minimizer}: P value: {p:.2f} ({r}), mean: {mean_exp}, std: {std_exp}"
            )
            print(f"{minimizer}: Synthetic mean: {mean_s_exp}, std: {std_s_exp}")
            ax = plot_data_and_fit(
                fit,
                ax,
                xmin,
                title,
                pdf=True,
                p_val=p,
                alpha_std=std_s_exp,
                color=color,
                addFit=addFit,
            )
        else:
            ax = plot_data_and_fit(
                fit, ax, xmin, title, pdf=True, color=color, addFit=addFit
            )

        # plot_ks_distance(d, xmin)
        if show:
            plt.show()
        if save and not debug:
            # Save the figure as PDF using the title as filename
            safe_title = (
                title.replace(" ", "_")
                .replace("$", "")
                .replace("\\", "")
                .replace("{", "")
                .replace("}", "")
                .replace(":", "")
                .replace("__", "_")
                .replace("_-_", "-")
                .replace("mathrm", "")
                .replace(".00", "")
            )
            if not addFit:
                safe_title += "_noFit"
            filename = f"{PLOTPATH}{safe_title}.pdf"
            fig.savefig(filename, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_exponent_fit()

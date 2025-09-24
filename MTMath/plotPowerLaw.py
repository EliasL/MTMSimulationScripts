import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
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
    if xmin is not None:
        data = data[data >= xmin]
    if xmax is not None:
        data = data[data <= xmax]
    return data


def cumulative_distribution_function(data, xmin=None, xmax=None, ccdf=False, **kwargs):
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
    if data.size == 0:
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

    if ccdf:
        CDF = 1 - CDF
    return data, CDF


def power_law_ks_distance(
    data,
    alpha,
    xmin,
    xmax=None,
    truncated=True,
    Lambda=None,
):
    """
    data : array-like
    alpha : float
    xmin : float
    xmax : float or None
    truncated : bool
        If True, use truncated power law p(x) ∝ x^{-alpha} exp(-Lamda * x).
    Lambda : float or None
        Exponential cutoff λ. Required when truncated=True.

    Returns
    -------
    D : float KS distance
    D_index : int
    """
    import numpy as np
    from scipy.integrate import quad

    data = np.asarray(data)

    # Reject integer data
    if np.issubdtype(data.dtype, np.integer):
        raise TypeError("Discrete/integer data not supported. Use continuous data.")

    # Apply range
    data = data[data >= xmin]
    if xmax is not None:
        data = data[data <= xmax]
    n = len(data)
    if n < 2:
        return 1, None

    # Ensure sorted
    if not np.all(data[:-1] <= data[1:]):
        data = np.sort(data)

    # Empirical CDF
    # TODO: Check and verify
    Actual_CDF = np.arange(n) / float(n)

    # ---- Theoretical CDF ----
    if not truncated:
        if xmax is None:
            Theoretical_CDF = 1.0 - (data / float(xmin)) ** (-alpha + 1)
        else:
            num = 1.0 - (data / float(xmin)) ** (1.0 - alpha)
            den = 1.0 - (float(xmax) / float(xmin)) ** (1.0 - alpha)
            Theoretical_CDF = num / den
    else:
        if Lambda is None or Lambda <= 0:
            raise ValueError("When truncated=True, you must provide lam > 0.")

        def f(t):
            return (t ** (-alpha)) * np.exp(-Lambda * t)

        upper = np.inf if xmax is None else float(xmax)
        Z, _ = quad(f, float(xmin), upper, epsabs=1e-10, epsrel=1e-10, limit=200)

        cdf_vals = np.empty_like(data, dtype=float)
        for i, x in enumerate(data):
            xi = float(x)
            if xi <= xmin:
                cdf_vals[i] = 0.0
            elif xmax is not None and xi >= xmax:
                cdf_vals[i] = 1.0
            else:
                num, _ = quad(f, float(xmin), xi, epsabs=1e-10, epsrel=1e-10, limit=200)
                cdf_vals[i] = num / Z
        Theoretical_CDF = cdf_vals

    # ---- KS statistic ----
    diffs_plus = Theoretical_CDF - Actual_CDF
    diffs_minus = Actual_CDF - Theoretical_CDF

    D_plus = np.max(diffs_plus)
    D_minus = np.max(diffs_minus)

    D_plus_index = int(np.argmax(diffs_plus))
    D_minus_index = int(np.argmax(diffs_minus))

    if D_plus > D_minus:
        D = D_plus
        D_index = D_plus_index
    else:
        D = D_minus
        D_index = D_minus_index

    return float(D), D_index


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
        # handle drops.max() = NaN case
        if np.isnan(drops.max()):
            ax2.set_ylim(0, 1)
        else:
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
        if np.isnan(x1) or np.isnan(x2):
            x1, x2 = 0, 1
        axins.set_xlim(x1, x2)
        axins.set_title("Zoom", fontsize=8)

        # twin‐axis for drops in the inset
        axins2 = axins.twinx()
        zoom_strain_mask = strain_limited >= x1
        zoom_strain_mask &= strain_limited <= x2
        drops_zoom = plotDrops[zoom_strain_mask]
        axins2.plot(strain_limited[zoom_strain_mask], drops_zoom)
        if np.isnan(drops_zoom.max()):
            axins2.set_ylim(0, 1)
        else:
            axins2.set_ylim(0, drops_zoom.max() * 1.5)

        debug_fig.tight_layout()
        # Save debug energy plot
        minimizer = get_attribute(label)
        filename = f"{PLOTPATH}debug/{minimizer}_{csvPath.split('/')[-1]}_energy_drops_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}{OUTPUTTYPE}"
        debug_fig.savefig(filename, dpi=300)
        # to save memory, close the figure
        plt.close(debug_fig)
    return drops


def get_bins(data):
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
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return bin_edges, bin_centers


def get_hist(data):
    bin_edges, bin_centers = get_bins(data)

    # Compute the histogram for the tail (density=True → area under PDF = 1)
    hist_vals, edges = np.histogram(data, bins=bin_edges, density=True)
    return bin_centers, hist_vals


def plot_data_pdf(
    ax,
    data,
    label="Binned PDF of energy drops",
    edgecolor="black",
    alphaColor=1,
    color=None,
):
    # Choose edgecolor if None
    if edgecolor is None:
        edgecolor = ax._get_lines.get_next_color()

    bin_centers, hist_vals = get_hist(data)

    # Plot as points
    plot_kwargs = {
        "marker": "o",
        "linestyle": "None",
        "label": label,
        "alpha": alphaColor,
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


def fit_power_law(data, xmin, truncated=True):
    bin_centers, hist_values = get_hist(data)
    mask = (bin_centers > xmin) & (hist_values > 0)
    xdata = bin_centers[mask]
    ydata = hist_values[mask]

    log_y = np.log(ydata)
    if len(xdata) < 3:
        print("Not enough data points above xmin for fitting.")
        return None, None, None, None

    if truncated:
        # Truncated power law: log(y) = -alpha * log(x) - Lambda * x + logC
        def log_model(x, alpha, Lambda, logC):
            return -alpha * np.log(x) - Lambda * x + logC

        popt, _ = curve_fit(log_model, xdata, log_y, p0=[1.5, 1e-3, 0])
        alpha_fit, lambda_fit, logC_fit = popt
        y_fit = np.exp(log_model(bin_centers, *popt))

    else:
        # Pure power law: log(y) = -alpha * log(x) + logC
        def log_model(x, alpha, logC):
            return -alpha * np.log(x) + logC

        popt, _ = curve_fit(log_model, xdata, log_y, p0=[1.5, 0])
        alpha_fit, logC_fit = popt
        y_fit = np.exp(log_model(bin_centers, *popt))

    return alpha_fit, lambda_fit, y_fit, bin_centers, hist_values


def make_and_plot_truncated_fit_pdf(
    ax,
    data,
    xmin,
    title=None,
    color=None,
    label=None,
    linestyle="-",
    pre_label=None,
    add_ks_marker=False,
    alpha_std=None,
):
    alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(data, xmin)

    if label is None:
        err = f" \\pm {alpha_std:.2f}" if alpha_std is not None else ""
        fit_label = (
            rf"Trunc. Fit: $\alpha={alpha_fit:.2f}{err}, \lambda={lambda_fit:.2f}$"
        )
    else:
        fit_label = label

    # Plot
    ax.plot(
        bin_centers,
        y_fit,
        label=(pre_label or "") + fit_label,
        color=color,
        linestyle=linestyle,
    )
    if add_ks_marker:
        ks_distance, ks_index = power_law_ks_distance(
            data, alpha_fit, xmin, Lambda=lambda_fit
        )

        # decide which index gives the max absolute deviation
        sorted_tail = np.sort(data[data >= xmin])

        x_D = sorted_tail[ks_index]
        annotate_ks_distance_pdf(ax, x_D, ks_distance)

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
    data,
    ax=None,
    xmin=None,
    title="",
    alpha_std=None,
    p_val=None,
    color=None,
    addFit=True,
):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the data

    bin_centers, hist_values = plot_data_pdf(ax, data=data, color=color)

    # plot the fit
    if addFit:
        make_and_plot_truncated_fit_pdf(
            ax,
            data,
            xmin,
            title=title,
            alpha_std=alpha_std,
        )

        # Add shaded fit region with formula in label
        ax.axvspan(
            xmin,
            data.max(),
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
                params={"alpha": syntheticExponent},
            )[0]
        # fit the data
        fit = fit_power_law(drops, xmin)
        fits.append(fit)
        p, alpha, mean, std = evaluate_fit(
            drops,
            xmin=xmin,
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
    # fit = (alpha_fit, lambda_fit, y_fit, bin_centers, hist_values)

    exponents = [fit[0] for fit in fits]

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


def scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng):
    from scipy.special import gammainc, gammaincinv

    k = 1.0 - alpha  # only valid when alpha < 1
    theta = 1.0 / Lambda
    Fmin = gammainc(k, xmin / theta)
    u = Fmin + (1.0 - Fmin) * rng.random(size)
    y = gammaincinv(k, u)
    x = theta * y
    return x


def _plain_powerlaw(xmin, alpha, size, rng):
    """
    Sample from f(x) ∝ x^(-alpha), x >= xmin (proper only if alpha > 1).
    Inverse-CDF: X = xmin * U^(-1/(alpha-1)).
    """
    if alpha <= 1:
        raise ValueError("For Lambda=None, a proper power-law requires alpha > 1.")
    u = rng.random(size)
    return xmin * u ** (-1.0 / (alpha - 1.0))


def truncatedPowerlawGenerator(xmin, alpha, Lambda, size, rng):
    """
    f(x) ∝ x^(-alpha) * exp(-Lambda * x), x >= xmin.
    If Lambda is None: use the *plain* power law f(x) ∝ x^(-alpha), x >= xmin (alpha > 1).
    """
    if xmin <= 0:
        raise ValueError("xmin must be positive.")

    # No truncation: fall back to standard power-law
    if Lambda is None:
        return _plain_powerlaw(xmin, alpha, size, rng)

    if Lambda <= 0:
        raise ValueError("Lambda must be positive when provided.")

    # Heavy tail ⇒ inverse-CDF path is faster/numerically stable for alpha < 1
    if alpha < 1:
        return scipy_truncated_powerlaw(xmin, alpha, Lambda, size, rng)

    accepted = []
    need = size

    # Heuristic initial acceptance-rate estimate
    r_est = min(0.8, max(0.05, (xmin * Lambda) / (xmin * Lambda + alpha)))

    while need > 0:
        overshoot = 1.15
        n_prop = max(32, int(np.ceil(overshoot * need / r_est)))

        # Proposals from exponential tail anchored at xmin
        prop = xmin + rng.exponential(scale=1.0 / Lambda, size=n_prop)

        # Accept with probability (xmin / x)^alpha (log-space)
        log_u = np.log(rng.random(n_prop))
        mask = log_u < alpha * (np.log(xmin) - np.log(prop))

        if mask.any():
            accepted.extend(prop[mask])

        got = min(len(accepted), size)
        need = size - got

        # Update acceptance-rate estimate (EMA)
        inst_rate = mask.mean()
        if inst_rate > 0:
            r_est = 0.7 * r_est + 0.3 * float(inst_rate)
            r_est = min(0.95, max(0.02, r_est))

    if len(accepted) > size:
        accepted = accepted[:size]
    return np.asarray(accepted, dtype=float)


def create_synthetic_data(
    drops,
    xmin=-np.inf,
    nrSets=2500,
    debug=False,
):
    """
    Create synthetic data sets for testing the power law fitting.

    Parameters
    ----------
    drops : array-like
        Original dataset of energy drops.
    xmin : float
        Lower cutoff for the power-law tail.
    nrSets : int
        Number of synthetic datasets to generate.
    dist_name : str
        Which distribution to use (default: truncated_power_law).
    debug : bool
        If True, make diagnostic plots.

    Returns
    -------
    syntheticSets : ndarray, shape (nrSets, len(drops))
        Synthetic datasets, each with the same size as `drops`.
    """
    # 1. Fit the chosen distribution (default: truncated power law) to the real data.
    alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(drops, xmin)

    # 2. Split the data into "tail" (>= xmin, assumed to follow the model)
    #    and "non-tail" (< xmin, arbitrary background).
    tailDrops = drops[drops >= xmin]
    nonTailDrops = drops[drops < xmin]

    # 3. Create a reproducible RNG, seeded deterministically by xmin.
    rng = np.random.default_rng(int((np.log10(xmin) * 1e6) % (2**32)))

    # 4. Special case: if there are no non-tail values,
    #    just sample *everything* from the truncated power law.
    if len(nonTailDrops) == 0:
        total_samples = nrSets * len(drops)
        samples = truncatedPowerlawGenerator(
            xmin=xmin,
            alpha=alpha_fit,
            Lambda=lambda_fit,
            size=total_samples,
            rng=rng,
        )
        return samples.reshape((nrSets, len(drops)))

    # 5. Otherwise, compute the fraction of tail observations in the real data.
    nrTailObservations = len(tailDrops)
    nrObservations = len(drops)
    p_tail = nrTailObservations / nrObservations

    # 6. For each synthetic dataset, randomly decide how many points
    #    should come from the tail (binomial draw with prob = p_tail).
    tail_counts = rng.binomial(nrObservations, p_tail, size=nrSets)

    # 8. Generate the non-tail part by bootstrap resampling from the original non-tail data.
    non_tail = rng.choice(nonTailDrops, size=(nrSets, nrObservations), replace=True)

    # 9. Generate all required tail samples in bulk from the truncated power law.
    total_tails = tail_counts.sum()
    all_tails = truncatedPowerlawGenerator(
        xmin=xmin,
        alpha=alpha_fit,
        Lambda=lambda_fit,
        size=total_tails,
        rng=rng,
    )

    # 10. Split the tail samples into slices of the right size for each dataset.
    offsets = np.concatenate([[0], np.cumsum(tail_counts)])

    # 11. Construct the synthetic datasets:
    #     start with resampled non-tail values, then overwrite the first k entries
    #     with tail values for each dataset.
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
        plot_data_and_fit(drops, debug_axes[0], xmin, "Original data")
        for i, synth_drops in zip(
            range(1, 4), syntheticSets[[0, len(syntheticSets) // 2, -1]]
        ):
            synth_alpha_fit, _, _, _, _ = fit_power_law(synth_drops, xmin)

            title = rf"Synthetic set {i}, $\alpha$: {synth_alpha_fit:.3f}, $E_{{\mathrm{{min}}}}$={xmin:.2e}"

            plot_data_and_fit(synth_drops, debug_axes[i], xmin, title, pdf=True)
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
    data, xmin = args

    alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(data, xmin)
    D, _ = power_law_ks_distance(data, alpha_fit, xmin, Lambda=lambda_fit)
    return D, alpha_fit


# --- KS p-value computation function ---
def goodnessOfFit(
    drops,
    synthetic_sets,
    xmin=-np.inf,
    parallel=False,
    debug=False,
):
    # Fit the original data
    alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(drops, xmin)
    D_orig, _ = power_law_ks_distance(drops, alpha_fit, xmin, Lambda=lambda_fit)

    # print("Computing KS distances for synthetic sets...")
    if parallel:
        # build arg‐tuples so each worker knows xmin & dist_name
        args_list = [(synthetic_data, xmin) for synthetic_data in synthetic_sets]
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
            D_synth.append(_compute_D_for_set((s, xmin)))
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

        bin_centers, hist_values = plot_data_pdf(ax, drops, label="Real data")
        make_and_plot_truncated_fit_pdf(
            ax,
            drops,
        )

        for i, s_drops in enumerate(synthetic_sets[[0, len(synthetic_sets) // 2, -1]]):
            alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(
                s_drops, xmin
            )

            # synth_bin_centers, synth_hist_values = getHist(fit_synth)
            synth_bin_centers, synth_hist_values = plot_data_pdf(
                ax,
                s_drops,
                label=f"Synthetic sample {i}",
                alphaColor=0.2,
            )
            synth_d, _ = power_law_ks_distance(s_drops, alpha_fit, xmin)

            # plot the fit

            make_and_plot_truncated_fit_pdf(
                ax,
                s_drops,
                xmin=xmin,
                bin_centers=synth_bin_centers,
                hist_values=synth_hist_values,
                pre_label=f"Synth fit {i} D:{synth_d:.2f}_",
                alpha=0.2,
                linestyle="--",
            )

        # Add shaded fit region
        ax.axvspan(xmin, drops.max(), color="gray", alpha=0.2, label="Fit region")
        ax.legend()
        ax.figure.show()

    return p_value, mean_alpha, std_alpha


def evaluate_fit(
    drops,
    xmin,
    parallel=True,
    verbose=False,
    debug=False,
    nr_sets=2500,
):
    """
    Compute (a) KS p-value using parametric synthetic sets *and* (b) exponent
    uncertainty. This consolidates work so synthetic sets are generated once
    and reused for both metrics.

    Returns
    """
    import os
    import json
    from pathlib import Path

    if verbose:
        print("nr of drops:", len(drops))
    if len(drops) < 200:
        print("Warning: this is not a lot of data, the p-value might not be reliable.")

    nr_sets = nr_sets if not debug else 3

    # paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    bootstrap_dir = repo_root / "bootstrapData"
    os.makedirs(bootstrap_dir, exist_ok=True)

    drop_sum = float(np.sum(drops))
    p_file = bootstrap_dir / f"p_{drop_sum}_{xmin}_{nr_sets}.json"

    p = None
    mean_exponent = None
    exponent_std = None

    # --- (A) KS p-value + parametric exponent uncertainty via synthetic sets ---
    if p_file.exists() and not debug:
        with open(p_file, "r") as f:
            result = json.load(f)
        p = result["p"]
        mean_exponent = result.get("mean")
        exponent_std = result.get("std")
        exponent = result.get("exp")
        if verbose:
            print(f"Loaded p-value + synthetic exponent stats from {p_file}")
    else:
        if verbose:
            print("Generating synthetic data for KS test and parametric uncertainty…")

        exponent, _, _, _, _ = fit_power_law(drops, xmin)

        sets = create_synthetic_data(
            drops,
            xmin=xmin,
            nrSets=nr_sets,
            debug=debug,
        )

        p, mean_exponent, exponent_std = goodnessOfFit(
            drops,
            sets,
            xmin,
            parallel=parallel,
            debug=debug,
        )

        with open(p_file, "w") as f:
            json.dump(
                {"p": p, "exp": exponent, "mean": mean_exponent, "std": exponent_std}, f
            )

    return p, exponent, mean_exponent, exponent_std


def find_best_xmin(
    drops,
    debug=False,
    start_xmin=1e-7,
    max_xmin=1e-4,
    nr_sets_start=250,
    nr_sets_final=2500,
):
    """
    Heuristically choose the smallest x_min whose truncated power‑law fit
    passes the KS goodness‑of‑fit at p >= 0.1.

    Strategy
    --------
    1) Build a log-spaced grid of candidate x_min values between
       start_xmin and max_xmin.
    2) Perform a binary search over that grid to find the *minimum* index
       where p >= 0.1. To reduce variance, increase the number of synthetic
       sets (nr_sets) as we move to larger x_min (closer to acceptance).
    3) If no candidate reaches p >= 0.1, return the x_min with the largest
       p observed.

    Returns
    -------
    best_xmin : float
        Selected x_min.
    best_stats : dict
        Dictionary with 'p', 'alpha', 'mean_alpha', 'std_alpha', 'nr_sets' for the best_xmin.
    history : list of dict
        One entry per evaluated candidate in ascending x_min order, each with:
        {'xmin', 'p', 'alpha', 'mean_alpha', 'std_alpha', 'nr_sets'}.
    """
    # Candidates and corresponding synthetic-set budgets
    xmins = np.logspace(np.log10(start_xmin), np.log10(max_xmin), num=100)
    # Prepare a per-iteration schedule for the number of synthetic sets.
    # We use the *maximum* number of iterations a binary search may take,
    # then increase the budget monotonically along the search path.
    max_iters = int(np.ceil(np.log2(len(xmins)))) + 1
    nr_sets_schedule = np.logspace(
        np.log10(nr_sets_start), np.log10(nr_sets_final), num=max_iters
    ).astype(int)

    p_threshold = 0.10

    # Cache: idx -> {'p','alpha', 'mean_alpha','std_alpha','nr_sets'}
    evaluated = {}
    history = []

    def eval_idx(idx, nr_sets_min):
        """
        Evaluate candidate 'idx' using at least 'nr_sets_min' synthetic sets.
        If we've already evaluated this idx with fewer sets, re-evaluate with the larger budget.
        """
        xmin = float(xmins[idx])
        nr_sets = int(nr_sets_min)

        prev = evaluated.get(idx)
        if prev is not None and prev["nr_sets"] >= nr_sets:
            # Already have an evaluation with >= this budget
            return (
                prev["p"],
                prev["alpha"],
                prev["mean_alpha"],
                prev["std_alpha"],
                prev["nr_sets"],
            )

        p, alpha, mean_alpha, std_alpha = evaluate_fit(
            drops,
            xmin,
            parallel=True,
            debug=debug,
            nr_sets=nr_sets,
        )
        rec = {
            "p": float(p),
            "alpha": float(alpha),
            "mean_alpha": None if mean_alpha is None else float(mean_alpha),
            "std_alpha": None if std_alpha is None else float(std_alpha),
            "nr_sets": int(nr_sets),
        }
        evaluated[idx] = rec

        history.append(
            {
                "xmin": xmin,
                **rec,
            }
        )
        if debug:
            print(
                f"[find_best_xmin] xmin={xmin:.3e} -> p={rec['p']:.3f}, "
                f"alpha={rec['alpha']:.3f}, "
                f"mean_alpha={rec['mean_alpha']}, std_alpha={rec['std_alpha']}, "
                f"nr_sets={rec['nr_sets']}"
            )
        return (
            rec["p"],
            rec["alpha"],
            rec["mean_alpha"],
            rec["std_alpha"],
            rec["nr_sets"],
        )

    # Binary search on indices [lo, hi] with increasing evaluation budgets
    lo, hi = 0, len(xmins) - 1
    candidate_idx = None
    iter_idx = 0  # how many evaluations we've attempted (upper-bounded by max_iters)

    while lo <= hi and iter_idx < max_iters:
        mid = (lo + hi) // 2
        nr_sets_now = nr_sets_schedule[min(iter_idx, max_iters - 1)]
        p_mid, _, _, _, _ = eval_idx(mid, nr_sets_now)

        if p_mid >= p_threshold:
            candidate_idx = mid
            hi = mid - 1  # try smaller xmin
        else:
            lo = mid + 1  # need larger xmin

        iter_idx += 1

    if candidate_idx is None:
        # No candidate reached threshold — choose the max-p option evaluated.
        # Ensure all were evaluated so we can pick the true max p over the grid.
        for i in range(len(xmins)):
            eval_idx(i, nr_sets_schedule[-1])
        # Find the index with the highest p (tie-breaker: smaller xmin)
        best_i = min(
            range(len(xmins)),
            key=lambda i: (-evaluated[i]["p"], xmins[i]),
        )
    else:
        best_i = candidate_idx

    rec = evaluated[best_i]
    best_xmin = float(xmins[best_i])
    best_stats = {
        "p": rec["p"],
        "alpha": rec["alpha"],
        "mean_alpha": rec["mean_alpha"],
        "std_alpha": rec["std_alpha"],
        "nr_sets": rec["nr_sets"],
    }

    # Sort history by xmin for readability
    history.sort(key=lambda d: d["xmin"])

    return best_xmin, best_stats, history


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


def plot_history(history, savePath):
    """
    Plot KS p-value and exponent (with std error bars) versus xmin.
    """
    if not history:
        return

    hist_sorted = sorted(history, key=lambda d: d["xmin"])
    x = np.array([h["xmin"] for h in hist_sorted], dtype=float)
    pvals = np.array([h.get("p", np.nan) for h in hist_sorted], dtype=float)
    alphas = np.array([h.get("alpha", np.nan) for h in hist_sorted], dtype=float)
    stds = np.array([h.get("std_alpha", np.nan) for h in hist_sorted], dtype=float)

    # Colors assigned per axis (consistent with Matplotlib defaults)
    c_p = "tab:blue"  # left axis (p-values)
    c_a = "tab:orange"  # right axis (alpha)

    fig, ax1 = plt.subplots()
    ax1.set_xscale("log")

    # --- Left axis: p-values ---
    (p_line,) = ax1.plot(
        x,
        pvals,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label="KS p-value",
        color=c_p,
        zorder=3,
    )
    thr_line = ax1.axhline(
        0.10,
        linestyle="--",
        linewidth=1.0,
        color=c_p,
        alpha=0.5,
        label="p = 0.10 threshold",
        zorder=2,
    )
    ax1.set_xlabel(r"$E_{\mathrm{min}}$")
    ax1.set_ylabel("KS p-value", color=c_p)
    ax1.tick_params(axis="y", colors=c_p)
    ax1.spines["left"].set_color(c_p)
    ax1.set_ylim(0, 1)
    ax1.grid(True, which="both", axis="both", alpha=0.25)

    # --- Right axis: alpha (+ error bars) ---
    ax2 = ax1.twinx()
    yerr = None if np.all(np.isnan(stds)) else stds
    alpha_err = ax2.errorbar(
        x,
        alphas,
        yerr=yerr,
        marker="s",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label=r"Exponent $\alpha$",
        color=c_a,
        ecolor=c_a,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=4,
    )
    ax2.set_ylabel(r"Exponent $\alpha$", color=c_a)
    ax2.tick_params(axis="y", colors=c_a)
    ax2.spines["right"].set_color(c_a)
    ax1.legend(handles=[p_line, thr_line, alpha_err], loc="upper left")
    # ax2.legend()

    fig.tight_layout()

    if savePath:
        fig.savefig(savePath, bbox_inches="tight")
    else:
        fig.show()


def make_exponent_fit(csvPath, strainLim=[0.15, 1.0], debug=False):
    # get name from path
    name = os.path.basename(os.path.dirname(csvPath))
    fig, ax = plt.subplots()
    dropWindows, _, _ = get_drops_in_windows(csvPath, strainLim)
    # drops = create_synthetic_data(
    #     drops,
    #     xmin=xmin,
    #     nrSets=10,
    #     params={"alpha": 1},
    # )
    for drops in dropWindows:
        # find best xmin
        xmin, stats, history = find_best_xmin(drops, debug=debug)
        p = stats["p"]
        alpha = stats["alpha"]
        std = stats["std_alpha"]
        alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(
            drops, xmin
        )

        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"
        plot_data_and_fit(drops, ax, xmin, title, pdf=True, alpha_std=std, p_val=p)
        print(f"E_min: {xmin:.2e}")
        print(f"P value: {p:.2f}")
        print(f"Exponent: {alpha:.3f} +/- {std:.3f}")
        # Print p value for surrounding xmins
        print("History:")
        for rec in history:
            print(
                f"  xmin={rec['xmin']:.2e} -> p={rec['p']:.3f}, "
                f"alpha={rec['alpha']:.3f}, "
                f"std_alpha={rec['std_alpha']:.3f}, "
                f"nr_sets={rec['nr_sets']}"
            )
        plot_history(history, savePath=PLOTPATH + f"searchHistory_{name}.pdf")
        print(f"Saved history plot to {PLOTPATH}searchHistory_{name}.pdf")

        # plot_ks_distance(d, xmin)
    plt.show()

    filename = f"{PLOTPATH}{name}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")


def get_attribute(label):
    d = {
        k.strip(): v.strip() for k, v in (item.split("=") for item in label.split(","))
    }
    if "minimizer" in d:
        attribute = d["minimizer"]
        attribute = attribute.replace("LBFGS", "L-BFGS")
    elif "L" in d:
        attribute = "L=" + d["L"]
    else:
        attribute = "Unknown"
    return attribute


def plot_powerlaw(
    algorithms_paths,
    alg_labels=None,
    strainLim=[0.15, 0.4],
    xmin=1e-4,
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
        if len(all_drops) == 0:
            print(f"No valid drops found for {label} in strain range {strainLim}.")
            continue

        # alpha_fit, lambda_fit, y_fit, bin_centers, hist_values = fit_power_law(
        #     drops, xmin
        # )

        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"
        attribute = get_attribute(labels[0])
        title = attribute + " " + title
        if attribute in MINIMIZER_COLORS:
            color = MINIMIZER_COLORS[attribute]
        else:
            color = "black"

        if evaluate:
            p, exp, mean_exp, exp_std = evaluate_fit(
                all_drops, xmin, parallel=True, debug=debug
            )

            rating = ["bad", "poor", "good", "excellent"]
            scores = [0.05, 0.1, 0.3]
            for threshold, r in zip(scores, rating):
                if p < threshold:
                    break
            else:
                r = rating[-1]
            print(f"Number of drops: {len(all_drops)}")
            print(f"{attribute}: P value: {p:.2f} ({r}), exp: {exp}, std: {exp_std}")
            ax = plot_data_and_fit(
                all_drops,
                ax,
                xmin,
                title,
                pdf=True,
                p_val=p,
                alpha_std=exp_std,
                color=color,
                addFit=addFit,
            )
        else:
            ax = plot_data_and_fit(
                all_drops, ax, xmin, title, pdf=True, color=color, addFit=addFit
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
            print(f"Saved figure to {filename}")
        plt.close(fig)


if __name__ == "__main__":
    # csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Users/elias/Downloads/macroData.csv"
    strainLim = [1.0, 3.0]
    # csvPath = "/Volumes/data/MTS2D_output/simpleShear,s400x400l0.15,1e-05,1.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"

    make_exponent_fit(csvPath=csvPath, strainLim=strainLim)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
import powerlaw
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
import functools
import os
import glob

np.random.seed(0)
# Create directories for saving plots
os.makedirs("Plots/powerLaw", exist_ok=True)


def get_energy_drops(csvPath, df=None, strainLim=[-np.inf, np.inf], debug=False):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    if df is None:
        df = pd.read_csv(csvPath)
    diffs = df["avg_energy_change"]
    strain = df["load"]
    lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = -diffs[mask]
    if debug:
        e = df["avg_energy"]
        debug_fig, ax1 = plt.subplots()
        ax1.plot(strain, e, label=r"$\langle E \rangle$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        ax2.plot(strain[mask], drops, label=r"$-\Delta \langle E \rangle$")
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
            bbox_to_anchor=(0.45, 0.7, 0.0, 0.30),
            bbox_transform=ax1.transAxes,
        )
        # plot energy in inset
        axins.plot(strain[zoom_mask], e[zoom_mask], lw=0.8)
        axins.set_xlim(x1, x2)
        axins.set_title("Zoom", fontsize=8)

        # twin‐axis for drops in the inset
        axins2 = axins.twinx()
        drops_zoom = -diffs[zoom_mask]
        axins2.plot(strain[zoom_mask], drops_zoom)
        axins2.set_ylim(0, drops_zoom.max() * 1.5)

        debug_fig.tight_layout()
        # Save debug energy plot
        filename = f"{plotPath}energy_drops_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}{outputType}"
        debug_fig.savefig(filename, dpi=300)
        # to save memory, close the figure
        plt.close(debug_fig)
    return drops


def plot_data(
    ax, fit=None, data=None, xmin=None, label="Energy drops", edgecolor="black", alpha=1
):
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


# --- Find x_min function for PDF tail threshold ---
def find_x_min(data, threshold=0.02, num_micro_bins=1000, verbose=False):
    """
    Identify the starting point for logarithmic binning by finding the first bin
    where the relative Poisson error (1/sqrt(count)) exceeds `threshold`.
    - data: 1D numpy array of positive values
    - threshold: fractional error cutoff (e.g., 0.02 for 2% relative uncertainty)
      This represents the 1-sigma Poisson fractional error: sigma/N = 1/sqrt(N).
    - num_micro_bins: number of fine-grained log-spaced bins for initial scan
    - verbose: if True, print information about threshold, x_min, and data proportion dropped

    Returns the geometric center of the first micro-bin where 1/sqrt(count) > threshold.
    If no bin exceeds the threshold, returns data.min(). In verbose mode, also prints
    the percentage of data below x_min (not log-binned).
    """
    data = np.asarray(data)
    n_total = data.size

    data_min = data.min()
    data_max = data.max()

    # Create micro-bins in log-space
    micro_edges = np.logspace(
        np.log10(data_min), np.log10(data_max), num=num_micro_bins + 1
    )
    counts, edges = np.histogram(data, bins=micro_edges)

    x_min = data_min
    count_at_xmin = None

    # Scan micro-bins for the first one where 1/sqrt(count) > threshold
    for i, count in enumerate(counts):
        if count > 0:
            rel_err = 1.0 / np.sqrt(count)
            # rel_err is the 1-sigma fractional (Poisson) error in this micro-bin
            if rel_err > threshold:
                x_min = np.sqrt(edges[i] * edges[i + 1])
                count_at_xmin = count
                break

    # Verbose reporting
    if verbose:
        pct_error = threshold * 100
        if count_at_xmin is not None:
            print(
                f"Chosen relative error threshold: {pct_error:.1f}% (1/sqrt(N) > {threshold:.3f})."
            )
            print(
                f"At x_min = {x_min:.3e}, micro-bin count = {count_at_xmin}, "
                f"relative error = {1.0 / np.sqrt(count_at_xmin):.3f}."
            )
            n_dropped = np.sum(data < x_min)
            pct_dropped = n_dropped / n_total * 100
            print(
                f"Data points below x_min (not log-binned): {n_dropped}/{n_total} = {pct_dropped:.2f}%."
            )
        else:
            print(
                f"No micro-bin had relative error > {pct_error:.1f}%; "
                f"defaulting x_min to data.min() = {data_min:.3e}."
            )

    return x_min


def plot_data_pdf(
    ax,
    fit=None,
    data=None,
    label="Energy drops",
    edgecolor="black",
    alpha=1,
    threshold=0.02,
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

    # Filter out values ≤ 0 (since log–bins require positive values)
    data = data[data > 0]

    # Find the start of the tail where Poisson noise exceeds threshold
    x_min = find_x_min(data, threshold=threshold)
    # Only keep data ≥ x_min
    tail_data = data[data >= x_min]

    # If there are no points beyond x_min, do nothing
    if len(tail_data) == 0:
        return

    data_max = tail_data.max()

    # Compute number of 0.1-decade bins from x_min to data_max
    decades = np.log10(data_max) - np.log10(x_min)
    n_bins = int(np.ceil(decades / 0.1))
    # Define bin edges in log-space
    bin_edges = np.logspace(
        np.log10(x_min),
        np.log10(x_min) + n_bins * 0.1,
        num=n_bins + 1,
    )

    # Compute the histogram for the tail (density=True → area under PDF = 1)
    hist_vals, edges = np.histogram(tail_data, bins=bin_edges, density=True)

    # Compute the geometric center of each bin for plotting
    bin_centers = np.sqrt(edges[:-1] * edges[1:])

    # Choose edgecolor if None
    if edgecolor is None:
        edgecolor = ax._get_lines.get_next_color()

    # Plot as points
    ax.plot(
        bin_centers,
        hist_vals,
        marker="o",
        linestyle="None",
        label=label,
        alpha=alpha,
    )

    # Set log–log axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    ax.set_ylabel(r"$p(-\Delta \langle E \rangle)$")
    ax.legend()


def plot_fit_pdf(
    ax,
    fit,
    dist_name="truncated_power_law",
    title=None,
    color=None,
    pre_label=None,
    label=None,
    alpha=1,
    linestyle="-",
    num_points=200,
):
    """
    Overplot the theoretical PDF of the fitted distribution (e.g., truncated_power_law)
    on log–log axes by approximating the derivative of the CCDF. Uses a log‐spaced
    grid from `fit.xmin` to `data.max()`. If `label` is None, construct it from
    distribution parameters.
    """
    # Get the original data and xmin from the fit
    data = fit.data_original
    xmin_val = fit.xmin

    # Define a log‐spaced grid starting at xmin up to data.max()
    x_vals = np.logspace(
        np.log10(xmin_val),
        np.log10(data.max()),
        num=num_points,
    )

    # Retrieve the fitted distribution object (e.g., fit.truncated_power_law)
    dist = getattr(fit, dist_name)

    # Compute the CDF on the grid, then CCDF = 1 – CDF
    CDF = dist._cdf_base_function(x_vals)
    CCDF = 1.0 - CDF

    # Approximate the PDF as –d(CCDF)/dx using numpy.gradient
    # Provide x_vals as spacing to gradient
    dCCDF_dx = np.gradient(CCDF, x_vals)
    pdf_vals = -dCCDF_dx

    # If no explicit label is given, build one from the dist parameters
    if label is None:
        label_parts = []
        # Some distributions expose parameter1_name, parameter1, etc.
        params = zip(
            [dist.parameter1_name, dist.parameter2_name, dist.parameter3_name],
            [dist.parameter1, dist.parameter2, dist.parameter3],
        )
        for name, val in params:
            if name is not None:
                if name == "lambda":
                    label_parts.append(r"$\lambda$")
                    label_parts.append(f"= {val:.2e}")
                else:
                    # For power_law, alpha is named dist.alpha
                    label_parts.append(rf"${name}$")
                    label_parts.append(f"= {val:.3f}")
        # Special case: pure power_law has dist.alpha
        if dist_name == "power_law":
            label_parts = [r"$\alpha$" + f"= {dist.alpha:.3f}"]
        label_str = ", ".join(label_parts)
        label = f"{dist_name.replace('_', ' ').title()}: {label_str}".strip()

    # Prepend a prefix if provided
    if pre_label:
        label = pre_label + label

    # Plot the PDF curve
    ax.plot(
        x_vals,
        pdf_vals,
        label=label,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )

    # Set log‐log scales
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
    csvPath=None, strainLim=None, df=None, steps=1, window_width=np.inf, debug=False
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
            csvPath, df=df, strainLim=[min_strain, max_strain], debug=debug
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
    pdf=False,
):
    if ax is None:
        fig, ax = plt.subplots()
    # plot the data
    if pdf:
        plot_data_pdf(ax, fit=fit)
    else:
        plot_data(ax, fit=fit)
    # plot the fit

    cmap_colors = ["green", "red", "yellow", "orange", "blue", "cyan"]

    for dist_name, color in zip(dist_names, cmap_colors):
        if pdf:
            plot_fit_pdf(
                ax,
                fit,
                dist_name=dist_name,
                title=title,
                color=color,
            )
        else:
            plot_fit(
                ax,
                fit,
                dist_name=dist_name,
                title=title,
                color=color,
            )

    # Add shaded fit region
    ax.axvspan(xmin, fit.data.max(), color="gray", alpha=0.2, label="Fit region")
    ax.legend()
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
        p = evaluate_fit(
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
            filename = f"{plotPath}window_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_xmin_{xmin:.2e}{outputType}"
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
        f"{plotPath}{syntheticTag}power_law_surface_"
        f"strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_"
        f"xmin_{xmins[0]:.2e}_{xmins[-1]:.2e}_"
        f"steps_{window_steps}_width_{window_width:.2f}_"
        f"{figType}{outputType}"
    )
    if not debug:
        # No need to save 3x3 map
        fig.savefig(filename, dpi=300)
    plt.show()


def make_debug_plot(xmins, strainLim=None):
    # Create debug plot grids for each xmin
    for xmin in xmins:
        # Find all saved fit plots for this xmin
        pattern = f"{plotPath}window_strain_*_xmin_{xmin:.2e}{outputType}"
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
            energy_file = f"{plotPath}energy_drops_strain_{strain_range}{outputType}"
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
        debug_filename = f"{plotPath}debug_fit_plots_xmin_{xmin:.2e}{outputType}"
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

    model_cdf = dist._cdf_base_function(sorted_data)
    model_ccdf = 1 - model_cdf
    # Normalize model CCDF to match area under ECDF
    empirical_area = np.trapezoid(ecdf, x=sorted_data)
    model_area = np.trapezoid(model_ccdf, x=sorted_data)
    model_ccdf *= empirical_area / model_area

    # Compute the KS statistic location
    diffs = np.abs(ecdf - model_ccdf)
    max_index = np.argmax(diffs)
    D_val = diffs[max_index]
    x_D = sorted_data[max_index]

    # Plotting
    fig, ax = plt.subplots()
    ax.step(sorted_data, ecdf, where="post", label="Empirical CCDF", color="blue")
    ax.plot(sorted_data, model_ccdf, label="Model CCDF", color="gray")

    # Highlight KS distance
    ax.vlines(
        x_D,
        model_ccdf[max_index],
        ecdf[max_index],
        color="red",
        linestyle="--",
        label=f"KS Distance D = {D_val:.3f}",
    )
    ax.scatter([x_D], [ecdf[max_index]], color="blue")
    ax.scatter([x_D], [model_ccdf[max_index]], color="gray")

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$")
    ax.set_ylabel(r"$P(X > x)$")
    ax.set_title("Kolmogorov–Smirnov Distance (CCDF)")
    ax.legend()
    fig.tight_layout()
    # Save the plot
    filename = f"{plotPath}ks_distance_xmin_{xmin:.2e}_{dist_name}{outputType}"
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
        filename = f"{plotPath}Synthetic_sets_xmin_{xmin:.2e}{outputType}"
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


    """
    s, xmin, dist_name = args
    fit_s = powerlaw.Fit(s, xmin=xmin)
    dist_s = getattr(fit_s, dist_name)
    return dist_s.D


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
    D_synth = np.array(D_synth)
    # p-value: proportion of synthetic distances >= original distance
    p_value = np.mean(D_synth >= D_orig)

    if debug:
        fig, ax = plt.subplots()
        plot_data(ax, fit=fit_orig, label="Real data")
        plot_fit(ax, fit_orig, dist_name=dist_name)
        for i, s_drops in enumerate(synthetic_sets[[0, len(synthetic_sets) // 2, -1]]):
            fit_synth = powerlaw.Fit(s_drops, xmin=xmin)
            plot_data(
                ax,
                fit=fit_synth,
                label=f"Synthetic sample {i}",
                edgecolor=None,
                alpha=0.2,
            )
            dist_synth = getattr(fit_orig, dist_name)
            synthD = dist_synth.D
            # plot the fit

            plot_fit(
                ax,
                fit_synth,
                dist_name=dist_name,
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

    return p_value


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
            p = json.load(f)
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
        p = goodnessOfFit(
            drops, sets, xmin, dist_name=dist_name, parallel=parallel, debug=debug
        )
        # Save the p-value to a file
        with open(p_file, "w") as f:
            json.dump(p, f)
    # if verbose:
    #     print(
    #         f"p-value for fit: {p:.3f}, ie. {p * 100:.1f}% of synthetic sets had a worse fit"
    #     )
    #     print(
    #         "If p > 0.1, the fit is likely a good fit. (This also depends on the number of drops.)"
    #     )
    return p


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
    csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    csvPath = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    xmin = 1e-6
    strainLim = [1, 3]
    debug = True
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
        plot_data_and_fit(fit, ax, xmin, title)
        p = evaluate_fit(d, xmin, parallel=True, debug=debug)
        print(p)
        plot_ks_distance(d, xmin)
    plt.show()


if __name__ == "__main__":
    plotPath = "Plots/powerLaw/"
    outputType = ".png"

    make_exponent_fit()

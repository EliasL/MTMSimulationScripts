import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
import powerlaw
from tqdm import tqdm
from scipy.optimize import curve_fit
from .powerlaw_lite import Truncated_Power_Law, Power_Law, Distribution
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


def get_minimizer(df):
    # We can find out what minimizer we have by looking at the third line in the
    # csv file, and seeing which of
    # LBFGS_Term_reason,CG_Term_reason,FIRE_Term_reason is non-zero
    if not len(df["LBFGS_Term_reason"]) >= 2:
        print("No data in file!")
        return "Unkown"

    # We use iloc so that if for example the second half of the file is loaded,
    # we take the third line from that arbitrary startingpoint, instead of trying
    # to find the third line from the top of the file (which might not be loaded)
    LBFGS = df["LBFGS_Term_reason"].iloc[2]
    CG = df["CG_Term_reason"].iloc[2]
    FIRE = df["FIRE_Term_reason"].iloc[2]
    assert sum([LBFGS == 0, CG == 0, FIRE == 0]) == 2, (
        "There is not exactly one non-zero term reason!"
    )
    if LBFGS != 0:
        return "L-BFGS"
    elif CG != 0:
        return "CG"
    elif FIRE != 0:
        return "FIRE"
    else:
        raise RuntimeError("Minimizer not found")


def get_system_size(csvPaths):
    import re

    sizes = set()
    for path in csvPaths:
        match = re.search(r"s(\d+)x(\d+)l", path)
        if match:
            n1, n2 = match.groups()
            if n1 != n2:
                return -1
            else:
                sizes.add(int(n1))
        else:
            print("Not able to find system size")
            re
    if len(sizes) > 1:
        print("More than one size!")
        return -1
    else:
        return sizes.pop()


def get_energy_drops(
    csvPaths, df=None, strainLim=[-np.inf, np.inf], debug=False, label=None
):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    If debug=True, plot intermediate energy and drop traces.
    """
    if isinstance(csvPaths, str):
        csvPaths = [csvPaths]

    drops = []
    for singlePath in csvPaths:
        if df is None:
            df = pd.read_csv(singlePath)

        if "avg_energy_change" not in df:
            # Add 0 in the beginning
            diffs = np.insert(np.diff(df["avg_energy"]), 0, 0)
        else:
            diffs = df["avg_energy_change"]

        strain = df["load"]
        lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
        drop_mask = diffs < 0
        mask = drop_mask & lim_mask
        drops.extend(-diffs[mask])

    drops = np.array(drops)

    data_info = {}
    data_info["minimizer"] = get_minimizer(df)
    data_info["nrSimulations"] = len(csvPaths)
    data_info["strainLim"] = strainLim
    data_info["L"] = get_system_size(csvPaths)

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

        name = make_path_name(data_info)
        filename = f"{PLOTPATH}debug/{name}_energy_drops_strain{OUTPUTTYPE}"
        debug_fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        # to save memory, close the figure
        plt.close(debug_fig)

    return drops, data_info


def getHist(data):
    # Find the start of the tail where Poisson noise exceeds threshold
    data_min = data.min()
    data_max = data.max()

    # Compute number of bins from x_min to data_max
    bins_per_decade = 5
    decades = np.log10(data_max) - np.log10(data_min)
    n_bins = int(np.ceil(decades * bins_per_decade))
    # Define bin edges from data_max downward
    log_edges = np.log10(data_max) - np.arange(n_bins + 1) / bins_per_decade
    bin_edges = np.power(10, log_edges)[::-1]  # Reverse to make it ascending

    # Compute the histogram for the tail (density=True → area under PDF = 1)
    # To make the fit more easily align with the data plot, we normalize
    # to density manually with the bin centers. Technically slightly incorrect.
    hist_vals, edges = np.histogram(data, bins=bin_edges, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, hist_vals


def plot_data_pdf(
    ax,
    data,
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

    # Choose edgecolor if None
    if edgecolor is None:
        edgecolor = ax._get_lines.get_next_color()

    bin_centers, hist_vals = getHist(data)

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
    ax.axvline(
        x_D,
        color=color,
        linestyle="--",
        linewidth=1.2,
        label=f"KS Distance D = {D_val:.3f}",
        zorder=-1,
        alpha=0.5,
    )


def plot_dist_pdf(
    ax,
    data,
    dist,
    title=None,
    color=None,
    alpha=1,
    linestyle="-",
    pre_label=None,
    add_ks_marker=False,
):
    # Plot

    tail_frac = (data >= dist.xmin).mean()
    bins_for_model = np.unique(data)
    # scale down to match full-data density
    pdf_model = dist.pdf(bins_for_model) * tail_frac
    ax.plot(
        bins_for_model,
        pdf_model,
        label=(pre_label or "") + pretty_text(dist.name),
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )
    if add_ks_marker:
        annotate_ks_distance_pdf(ax, dist.D_x, dist.D)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$-\Delta \langle E \rangle$ (Energy Drop)")
    ax.set_ylabel(r"$p(-\Delta \langle E \rangle)$")
    ax.set_title(title)
    ax.legend()
    return ax


def pretty_text(text, addEquation=True):
    r"""
    Make the text nicer.
    truncated_power_law: alpha=1.02, lambda=0.5
    ->
    Truncated Power Law: \alpha=1.02, \lambda=0.5
    """
    if addEquation:
        if "truncated_power_law" in text:
            text = text.replace(
                "truncated_power_law",
                "truncated_power_law " + r"$p(x) = x^{-\alpha} e^{-\lambda x}$)",
            )
        elif "power_law" in text:
            text = text.replace(
                "power_law",
                "power_law " + r"$p(x) = x^{-\alpha}$)",
            )

    text = text.replace("_", " ")
    text = text.replace(" alpha", r" $\alpha$")
    text = text.replace(" lambda", r" $\lambda$")
    text = text.replace(" mu", r" $\mu$")
    text = text.replace(" sigma", r" $\sigma$")
    # Captialize first letter of the first word
    text = text.capitalize()
    return text


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
        drops, data_info = get_energy_drops(
            csvPath,
            df=df,
            strainLim=[min_strain, max_strain],
            debug=debug,
            label=label,
        )
        windows.append((min_strain, max_strain))
        drops_in_windows.append(drops)
    return drops_in_windows, windows, centers


def make_title_from_dist(dist: Distribution):
    title = rf"$E_{{\mathrm{{min}}}}$={dist.xmin:.2e}"

    # add first parameter (assume greek variable name)
    title += rf" $\{dist.parameter1_name}={dist.parameter1:.2f}$"

    try:
        title += rf"$\pm{getattr(dist, dist.parameter1_name + '_std'):.2f}$"
    except AttributeError:
        pass

    if dist.p is not None:
        title += f" p: {dist.p:.2f}"
    return title


def make_title_from_data_info(data_info):
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    n = data_info["nrSimulations"]
    samples_string = f"{n} sample{'s' if n != 1 else ''}"
    title = rf"{data_info['minimizer']} {L}x{L} {samples_string} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f}"
    return title


def make_title(data_info=None, dist=None):
    title = ""
    if data_info:
        title += make_title_from_data_info(data_info)
    if dist:
        if data_info:
            title += " "
        title += make_title_from_dist(dist)
    return title


def make_path_name(data_info):
    if data_info is None:
        return "unkown"
    strainLim = data_info["strainLim"]
    path_name = f"{data_info['minimizer']}_s{strainLim[0]:.2f}-{strainLim[1]:.2f}_samples{data_info['nrSimulations']}"
    return path_name


def plot_data_and_dist(
    data,
    dist,
    ax=None,
    title="",
    color=None,
    addFit=True,
):
    if ax is None:
        fig, ax = plt.subplots()

    plot_data_pdf(ax, data)

    # plot the fit
    if addFit:
        plot_dist_pdf(ax, data, dist, color=color)

        # Add shaded fit region with formula in label
        if dist.xmax is None:
            xmax = max(data)
        else:
            xmax = dist.xmax
        ax.axvspan(
            dist.xmin,
            xmax,
            color="gray",
            alpha=0.2,
            label="Fit region",
        )

    ax.legend()
    ax.set_title(title)
    return ax


def get_window_power_law_exponents(
    xmin=-np.inf,
    dist=Truncated_Power_Law,
    syntheticData=False,
    syntheticExponent=1.0,
    **kwargs,
):
    """
    We slide this window over the data and plot the power law fit for each window.
    """
    drops_in_windows, windows, centers = get_drops_in_windows(**kwargs)
    dists = []
    ps = []
    debug = kwargs.get("debug", False)
    for drops, strainLim in zip(drops_in_windows, windows):
        dist = dist(drops, xmin=xmin)
        dists.append(dist)
        dist.evaluate_fit()
        ps.append(dist.p)

        if debug:
            debug_fig, debug_ax = plt.subplots()
            title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}"

            plot_data_and_dist(drops, dist, debug_ax, title)
            debug_fig.tight_layout()
            debug_fig.show()
            # Save debug window power law plot
            filename = f"{PLOTPATH}window_strain_{strainLim[0]:.2f}_{strainLim[1]:.2f}_xmin_{xmin:.2e}{OUTPUTTYPE}"
            debug_fig.savefig(filename)
            # to save memory, close the figure
            plt.close(debug_fig)

    # plot the exponents against the window centers
    exponents = [dist.alpha for dist in dists]

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


def explore_xmin(
    drops,
    min_xmin=None,
    max_xmin=None,
    nr_evaluation=10,
    confidence=0.1,
    DistType=Truncated_Power_Law,
    debug=False,
    xmax=None,
):
    if min_xmin is None:
        min_xmin = min(drops)
    if max_xmin is None:
        max_xmin = max(drops)
    # The last xmin usually have too few datapoints to be interesting,
    # so we remove the few last xmin, but make sure to still have
    # the correct number of evaluations
    remove_nr = int(nr_evaluation * 0.2)
    xmin_values = np.logspace(
        np.log10(min_xmin), np.log10(max_xmin), nr_evaluation + remove_nr
    )[:-remove_nr]

    test_dists = []
    for i, trial_xmin in enumerate(xmin_values):
        print(f"xmin:{trial_xmin:.2e}: {i + 1}/{len(xmin_values)}")
        dist = DistType(data=drops, xmin=trial_xmin, xmax=xmax)
        dist.evaluate_fit(drops, confidence=confidence, parallel=not debug)
        test_dists.append(dist)
    return test_dists


def find_best_xmin(
    drops,
    debug=False,
    min_xmin=None,
    max_xmin=None,
    xmax=None,
    min_p=0.1,
    nr_evaluation=20,
    start_accuracy=0.05,
    max_accuracy=0.01,
    DistType: Distribution = Truncated_Power_Law,
    data_info=None,
):
    """
    We scan many possible xmin values. We try to identify a plateau region
    in the exponents. We make sure the p-value is larger than min_p. If the
    p-value is close to the min_p limit, we need to increaes the accuracy.
    """

    path_name = make_path_name(data_info)
    title = make_title(data_info)

    print(f"Testing xmins for {title}")

    test_dists = explore_xmin(
        drops,
        min_xmin,
        max_xmin,
        nr_evaluation,
        start_accuracy,
        DistType,
        debug,
        xmax=xmax,
    )

    # We now have a rough sample on possible xmin values
    exponents = [d.alpha for d in test_dists]
    p_values = [d.p for d in test_dists]
    xmins = [d.xmin for d in test_dists]

    first_p_criteria = min_p / 2

    # Vectorize for robust indexing and easy neighborhood expansion
    x = np.array(xmins, dtype=float)
    p = np.array(p_values, dtype=float)

    if not np.isfinite(p).any() or p.max() < first_p_criteria:
        print("No pure power law found.")
        best_dist = test_dists[0]
    else:
        # Identify contiguous region where p > threshold-start_accuracy
        valid_idx = np.flatnonzero(p > first_p_criteria - start_accuracy)
        i_min, i_max = valid_idx[0], valid_idx[-1]

        # Expand the search window by one neighbor on each side when available
        i0 = max(0, i_min - 1)
        i1 = min(len(x) - 1, i_max + 1)
        new_min_xmin = x[i0]
        new_max_xmin = x[i1]

        # remove dists that we are about to replace
        test_dists = [
            d for d in test_dists if d.xmin < new_min_xmin or d.xmin > new_max_xmin
        ]

        new_dists = explore_xmin(
            drops,
            new_min_xmin,
            new_max_xmin,
            nr_evaluation,
            start_accuracy / 2,
            DistType,
            debug,
            xmax,
        )
        idx = int(np.argmax([d.p for d in new_dists]))
        best_dist = new_dists[idx]

        test_dists.extend(new_dists)

    # Plot p and exponent
    plot_dists_over_xmin(
        test_dists, best_dist, PLOTPATH + f"{path_name}_xMins.pdf", title=title
    )

    return best_dist


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


def plot_dists_over_xmin(dists, best_dist=None, savePath=None, title=None):
    """
    Plot KS p-value and exponent (with std error bars) versus xmin.
    """
    dists.sort(key=lambda d: d.xmin)
    x = np.array([d.xmin for d in dists], dtype=float)
    pvals = np.array([d.p for d in dists], dtype=float)
    p_stds = np.array([d.p_std for d in dists], dtype=float)
    alphas = np.array([d.alpha for d in dists], dtype=float)
    alpha_stds = np.array([d.alpha_std for d in dists], dtype=float)

    # Colors assigned per axis (consistent with Matplotlib defaults)
    c_p = "tab:blue"  # left axis (p-values)
    c_a = "tab:orange"  # right axis (alpha)

    fig, ax1 = plt.subplots()
    ax1.set_xscale("log")

    # --- Left axis: p-values ---
    p_line = ax1.errorbar(
        x,
        pvals,
        yerr=p_stds,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label="KS p-value",
        color=c_p,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
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

    # --- Right axis: alpha (+ error bars) ---
    ax2 = ax1.twinx()
    alpha_err = ax2.errorbar(
        x,
        alphas,
        yerr=alpha_stds,
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

    if best_dist:
        best_line = ax1.axvline(
            best_dist.xmin,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Best xmin: {best_dist.xmin:.2e}",
            zorder=-1,
            alpha=0.5,
        )

    ax1.legend(handles=[p_line, thr_line, alpha_err, best_line], loc="upper left")
    fig.tight_layout()
    ax1.set_title(title)

    if savePath:
        fig.savefig(savePath, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {savePath}")


def make_exponent_fit(
    csvPaths,
    strainLim=[0.15, 1.0],
    debug=False,
    DistType: Distribution = Truncated_Power_Law,
    xmax=None,
    show=True,
):
    fig, ax = plt.subplots()
    drops, data_info = get_energy_drops(csvPaths, strainLim=strainLim, debug=debug)
    if drops.size == 0:
        # No energy drops in strain region
        print(f"No energy drops in {strainLim} strain region: {data_info}")
        return

    # find best xmin
    dist = find_best_xmin(
        drops, DistType=DistType, debug=debug, xmax=xmax, data_info=data_info
    )
    title = make_title(data_info, dist)
    pathName = make_path_name(data_info)
    plot_data_and_dist(drops, dist, ax, title=title)

    filename = f"{PLOTPATH}{pathName}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    if show:
        plt.show()


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
    DistType: Distribution = Truncated_Power_Law,
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

        dist = DistType(data=all_drops, xmin=xmin)

        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}, $\alpha=${dist.alpha:.2f}"
        attribute = get_attribute(labels[0])
        title = attribute + " " + title
        if attribute in MINIMIZER_COLORS:
            color = MINIMIZER_COLORS[attribute]
        else:
            color = "black"

        if evaluate:
            p, mean_exp, exp_std = dist.evaluate_fit(all_drops, parallel=True)

            rating = ["bad", "poor", "good", "excellent"]
            scores = [0.05, 0.1, 0.3]
            for threshold, r in zip(scores, rating):
                if p < threshold:
                    break
            else:
                r = rating[-1]
            print(f"Number of drops: {len(all_drops)}")
            print(
                f"{attribute}: P value: {p:.2f} ({r}), exp: {dist.alpha}, std: {exp_std}"
            )

        plot_data_and_dist(
            all_drops,
            dist,
            ax,
            title,
            color=color,
            addFit=addFit,
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
    import powerlaw

    # csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    strainLim = [0.65, 1.0]
    paths = [
        f"/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s{i}/macroData.csv"
        for i in range(10)
    ]
    # paths = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    # strainLim = [1, 3]
    # csvPath = "/Volumes/data/MTS2D_output/simpleShear,s400x400l0.15,1e-05,1.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"

    make_exponent_fit(
        csvPaths=paths,
        strainLim=strainLim,
        debug=True,
        DistType=Truncated_Power_Law,
        # xmax=1e-4,
    )

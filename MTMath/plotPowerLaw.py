import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .evaluatePowerlawFit import Fit, Truncated_Power_Law
from powerlaw import Distribution
from Management.updateCSV import update_df_header
from Plotting.makePlots import safePath
import os
import glob
from Plotting.makePlots import safePath
from tqdm import tqdm

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
    if "LBFGS_Term_reason" in df:
        if not len(df["LBFGS_Term_reason"]) >= 2:
            print("No data in file!")
            return "Unkown"
    else:
        return "Unkown"

    # We use iloc so that if for example the second half of the file is loaded,
    # we take the third line from that arbitrary startingpoint, instead of trying
    # to find the third line from the top of the file (which might not be loaded)
    LBFGS = df["LBFGS_Term_reason"].iloc[2]
    CG = df["CG_Term_reason"].iloc[2]
    FIRE = df["FIRE_Term_reason"].iloc[2]
    assert sum([int(LBFGS) == 0, int(CG) == 0, int(FIRE) == 0]) == 2, (
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
        match = re.search(r"(\d+)x(\d+)", path)
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
    csvPaths,
    use_avg_e_change_from_init=True,
    df=None,
    strainLim=[-np.inf, np.inf],
    debug=False,
    label=None,
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
            df = update_df_header(df)
        if use_avg_e_change_from_init:
            assert "avg_e_change_from_init" in df, (
                "Uh oh. If the data is old, set use_avg_e_change_from_inti."
            )
            diffs = df["avg_e_change_from_init"]
        elif "avg_energy_change" not in df:
            # Add 0 in the beginning
            diffs = np.insert(np.diff(df["avg_energy"]), 0, 0)
        else:
            diffs = df["avg_energy_change"]

        if "Iteration" in df:
            # This is umut data, so he has already flipped his drops
            diffs = -diffs

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
    data_info["label"] = label

    if debug:
        # Only debug first seed when using labels
        if label is not None and "seed=" in label and "seed=0" not in label:
            return drops, data_info

        strain_limited = strain[1:][lim_mask[1:]]
        plotDrops = np.clip(-diffs[1:][lim_mask[1:]], 0, np.inf)
        e = df["avg_energy"]
        debug_fig, ax1 = plt.subplots()
        ax1.plot(strain, e, label=r"$\langle E \rangle$")
        ax1.set_ylabel(r"$\langle E \rangle$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        if use_avg_e_change_from_init:
            label = r"$- \langle E_{\mathrm{pre}-E_{\mathrm{post}}} \rangle$"
        else:
            label = r"$-\Delta \langle E \rangle$"
        ax2.plot(strain_limited, plotDrops, label=label)
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
        axins.set_itle("Zoom", fontsize=8)

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

        name = safePath(make_title(data_info))

        filename = f"{PLOTPATH}debug/{name}_energy_drops_strain{OUTPUTTYPE}"
        debug_fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        # to save memory, close the figure
        plt.close(debug_fig)

    return (drops, data_info)


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
    hist_vals, edges = np.histogram(data, bins=bin_edges, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    return bin_centers, hist_vals


def plot_data_pdf(
    ax,
    data,
    label=None,
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

    if label is None:
        if data.size < 1e4:
            nrDrops = data.size
        else:
            nrDrops = f"{data.size:.1e}"
        label = f"Binned PDF of {nrDrops} energy drops"

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
def annotate_ks_distance_pdf(ax, xmin, D_val, color="red"):
    ax.axvline(
        xmin,
        color=color,
        linestyle="--",
        linewidth=1.2,
        label=f"KS Distance D = {D_val:.3f}",
        zorder=-1,
        alpha=0.5,
    )


def plot_fit_pdf(
    ax,
    fit: Fit,
    title=None,
    color=None,
    alpha=1,
    linestyle="-",
    pre_label=None,
    add_ks_marker=False,
):
    dist = dist_from_fit(fit)

    # Work on a copy to avoid mutating fit.data_original in-place
    data = np.asarray(fit.data_original).copy()
    data.sort()

    # Fraction of samples in the fitted tail; used to scale the tail PDF to the
    # full-data density (to visually align with the empirical PDF computed on all data).
    tail_frac = float((data >= fit.xmin).mean())

    # Evaluate the model on a monotone x-grid. Using unique values avoids repeated
    # x-points when aggregating many simulations.
    bins_for_model = np.unique(data)

    # Guard against non-positive values on log axes.
    bins_for_model = bins_for_model[bins_for_model > 0]

    # Evaluate the fitted PDF at the plotting grid (NOT at the full, duplicated data).
    f = dist._pdf_base_function(bins_for_model)
    C = dist._pdf_continuous_normalizer
    likelihoods = f * C * tail_frac

    ax.plot(
        bins_for_model,
        likelihoods,
        label=(pre_label or "") + pretty_text(dist.name),
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )
    if add_ks_marker:
        annotate_ks_distance_pdf(
            ax, fit.xmin_fitting_results["xmins"][dist.D_i], dist.D
        )

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


def findPrePostSplit(csvPath="", df=None):
    # Return strain value where the stress is the largest
    # Uses P12 (off diagonal of first piola kirchhoff stress)
    if df is None:
        df = pd.read_csv(csvPath)
        df = update_df_header(df)
    if "avg_Pxy" in df:
        i = df["avg_Pxy"].argmax()
    elif "avg_sigmaxy" in df:
        i = df["avg_sigmaxy"].argmax()
    else:
        raise KeyError("No stress key found!")
    return df["load"].iloc[i]


def get_drops_in_windows(
    csvPath,
    strainLim=None,
    df=None,
    steps=1,
    window_width=np.inf,
    debug=False,
    label=None,
):
    if df is None:
        df = pd.read_csv(csvPath)
        df = update_df_header(df)

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


def make_title_from_fit(fit: Fit):
    title = rf"$E_{{\mathrm{{min}}}}$={fit.xmin:.2e}"
    dist = dist_from_fit(fit)
    # add first parameter (assume greek variable name)
    title += rf" $\{dist.parameter1_name}={dist.parameter1:.2f}$"

    try:
        title += rf"$\pm{getattr(dist, dist.parameter1_name + '_std'):.2f}$"
    except AttributeError:
        pass

    if hasattr(fit, "p") and fit.p is not None:
        title += f" p: {fit.p:.2f}"
    return title


def make_title_from_data_info(data_info):
    if "customTitle" in data_info:
        return data_info["customTitle"]
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    n = data_info["nrSimulations"]
    samples_string = f"{n} sample{'s' if n != 1 else ''}"
    title = rf"{data_info['minimizer']} {L}x{L} {samples_string} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f} ({data_info['label']})"
    return title


def make_title(data_info=None, fit: Fit | None = None):
    title = ""
    if data_info:
        title += make_title_from_data_info(data_info)
    if fit:
        if data_info:
            title += " "
        title += make_title_from_fit(fit)
    if title == "":
        title = "unknown"
    return title


def plot_data_and_fit(
    fit: Fit,
    ax=None,
    title="",
    data_info=None,
    color=None,
    addFit=True,
    save=True,
    extraPath="",
    show=False,
):
    if ax is None:
        fig, ax = plt.subplots()

    plot_data_pdf(ax, fit.data_original)

    # plot the fit
    if addFit:
        plot_fit_pdf(ax, fit, color=color)

        # Add shaded fit region with formula in label
        if fit.xmax is None:
            xmax = max(fit.data_original)
        else:
            xmax = fit.xmax
        dist = dist_from_fit(fit)
        ax.axvspan(
            fit.xmin,
            xmax,
            color="gray",
            alpha=0.2,
            label=rf"Fit region. $\alpha={dist.alpha:.2f}, \lambda=$ {dist.Lambda:.2e}",
        )

    ax.legend()
    if title == "" and data_info is not None:
        title = make_title(data_info)
    ax.set_title(title)

    if show:
        plt.show()
    if save:
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
        filename = f"{PLOTPATH}{extraPath}{safe_title}.pdf"
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {filename}")
        plt.close(fig)
    else:
        return ax


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


def plot_ks_distance(drops, xmin, dist_name="truncated_power_law", data_info=None):
    """
    Plot the empirical CCDF vs the fitted CCDF and visually show the KS distance (D).
    """
    # Fit the distribution
    fitObj = Fit(drops, xmin=xmin)
    dist = getattr(fitObj, dist_name)
    # Get the ECDF and model CCDF
    data = fitObj.data
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
    title = "Kolmogorov–Smirnov Distance (CCDF)"
    if data_info is not None:
        title = make_title_from_data_info(data_info)
        title += rf" $E_{{\mathrm{{min}}}}$={xmin:.2e}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    # Save the plot
    if data_info is not None:
        safe_title = safePath(title)
        filename = f"{PLOTPATH}{safe_title}_ks_distance{OUTPUTTYPE}"
    else:
        filename = f"{PLOTPATH}ks_distance_xmin_{xmin:.2e}_{dist_name}{OUTPUTTYPE}"
    fig.savefig(filename, dpi=300)
    print(f"Saved fig to {filename}")
    # plt.show()


def _fit_single_xmin_task(args):
    drops, trial_xmin, xmax, dist_name, confidence = args
    fit = Fit(
        data=drops,
        xmin=trial_xmin,
        xmax=xmax,
        xmin_distribution=dist_name,
    )
    # Avoid nested multiprocessing inside each worker.
    fit.evaluate_fit(drops, confidence=confidence, parallel=False)
    return fit


def explore_xmin(
    drops,
    min_xmin=None,
    max_xmin=None,
    nr_evaluation=10,
    confidence=0.1,
    distType: type[Distribution] = Truncated_Power_Law,
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

    tasks = [
        (drops, float(trial_xmin), xmax, distType.name, confidence)
        for trial_xmin in xmin_values
    ]

    try:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor() as ex:
            fits_iter = ex.map(_fit_single_xmin_task, tasks)
            test_fits = list(
                tqdm(
                    fits_iter,
                    total=len(tasks),
                    desc="Fitting xmins",
                    disable=False,
                )
            )
    except Exception as e:
        print(f"Parallel xmin exploration failed, falling back to serial: {e}")
        test_fits = []
        for i, trial_xmin in enumerate(xmin_values):
            desc = f"xmin:{trial_xmin:.2e}: {i + 1}/{len(xmin_values)}:"
            fit = Fit(
                data=drops,
                xmin=trial_xmin,
                xmax=xmax,
                xmin_distribution=distType.name,
            )
            fit.evaluate_fit(
                drops,
                confidence=confidence,
                parallel=False,
                tqdmDesc=desc,
            )
            test_fits.append(fit)

    return test_fits


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
    DistType: type[Distribution] = Truncated_Power_Law,
    data_info=None,
    xmin_results=None,
):
    """
    We scan many possible xmin values. We try to identify a plateau region
    in the exponents. We make sure the p-value is larger than min_p. If the
    p-value is close to the min_p limit, we need to increaes the accuracy.
    """

    title = make_title(data_info)
    path_name = safePath(title)

    print(f"Testing xmins for {title}")

    test_fits = explore_xmin(
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
    exponents = [dist_from_fit(f).alpha for f in test_fits]
    p_values = [f.p for f in test_fits]
    xmins = [f.xmin for f in test_fits]

    first_p_criteria = min_p / 2

    # Vectorize for robust indexing and easy neighborhood expansion
    x = np.array(xmins, dtype=float)
    p = np.array(p_values, dtype=float)

    if not np.isfinite(p).any() or p.max() < first_p_criteria:
        print("No pure power law found.")
        best_fit = test_fits[0]
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
        test_fits = [
            f for f in test_fits if f.xmin < new_min_xmin or f.xmin > new_max_xmin
        ]

        new_fits = explore_xmin(
            drops,
            new_min_xmin,
            new_max_xmin,
            nr_evaluation,
            start_accuracy / 2,
            DistType,
            debug,
            xmax,
        )
        p_vals = np.array([f.p for f in new_fits], dtype=float)
        local_max_idx = None
        if len(p_vals) >= 3:
            for i in range(1, len(p_vals) - 1):
                if (
                    p_vals[i] >= min_p
                    and p_vals[i] >= p_vals[i - 1]
                    and p_vals[i] >= p_vals[i + 1]
                ):
                    local_max_idx = i
                    break
        if local_max_idx is None:
            local_max_idx = int(np.argmax(p_vals))
        best_fit = new_fits[local_max_idx]

        test_fits.extend(new_fits)

    # Plot p and exponent
    plot_fits_over_xmin(
        test_fits,
        best_fit,
        PLOTPATH + f"{path_name}_xMins.pdf",
        title=title,
        xmin_results=xmin_results,
    )

    return best_fit


def find_start_of_plastic_events(
    paths, postRegime, data_info, debug=False, binsPerDecade=5
):
    all_data = None
    strainLim = None
    if data_info is not None:
        strainLim = data_info.get("strainLim", None)

    for path in paths:
        df = pd.read_csv(path)
        df = update_df_header(df)
        if strainLim is None or strainLim == "auto":
            gamma_max_stress = findPrePostSplit(df=df)
            if postRegime:
                strainLim = [gamma_max_stress + 1e-2, df["load"].max()]
            else:
                strainLim = [df["load"].min(), gamma_max_stress - 1e-4]
        df = df[(df["load"] > strainLim[0]) & (df["load"] < strainLim[1])]
        if all_data is None:
            all_data = df
        else:
            all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data is None or all_data.empty:
        print("No data found for plastic event analysis.")
        return None
    # The plan is to look at the nr_plastic_deformations associated with each
    # drop, and make histogram of the number of plastic deformations associated
    # with each drop size

    if "avg_e_change_from_init" in all_data:
        diffs = all_data["avg_e_change_from_init"]
    else:
        raise RuntimeError("Avoid using old data")
        if "avg_energy_change" not in df:
            # Add 0 in the beginning
            diffs = np.insert(np.diff(df["avg_energy"]), 0, 0)
        else:
            diffs = df["avg_energy_change"]

    if "Iteration" in all_data:
        # This is umut data, so he has already flipped his drops
        diffs = -diffs

    strain = all_data["load"]
    lim_mask = (strain > strainLim[0]) & (strain < strainLim[1])
    drop_mask = diffs < 0
    mask = drop_mask & lim_mask
    drops = (-diffs[mask]).to_numpy()
    if "nr_plastic_deformations" in all_data:
        plastics = all_data["nr_plastic_deformations"][mask].to_numpy()
    else:
        raise KeyError("nr_plastic_deformations column not found.")

    xmin_peak = None
    drops_pos = drops[drops > 0]
    if drops_pos.size > 0:
        decades = np.log10(drops_pos.max()) - np.log10(drops_pos.min())
        nr_bins = max(1, int(np.ceil(decades * binsPerDecade)))
        bins = np.logspace(
            np.log10(drops_pos.min()), np.log10(drops_pos.max()), nr_bins + 1
        )
        weights = plastics[drops > 0]
        bin_sums, _ = np.histogram(drops_pos, bins=bins, weights=weights)
        bin_density, _ = np.histogram(
            drops_pos, bins=bins, weights=weights, density=True
        )
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        if len(bin_sums) >= 3:
            for i in range(1, len(bin_sums) - 1):
                if bin_sums[i] >= bin_sums[i - 1] and bin_sums[i] >= bin_sums[i + 1]:
                    xmin_peak = bin_centers[i]
                    break

    if debug:
        fig, ax1 = plt.subplots(1, 1, figsize=(6.4, 4.2))
        ax2 = ax1.twinx()
        c_pdf = "tab:blue"
        c_plastic = "tab:orange"
        plot_data_pdf(ax1, drops)
        ax1.set_title("Energy drop PDF and plastic events")
        ax1.set_xlabel(r"$-\Delta \langle E \rangle$")
        ax1.set_ylabel(r"$p(-\Delta \langle E \rangle)$", color=c_pdf)
        ax1.tick_params(axis="y", colors=c_pdf)
        ax1.spines["left"].set_color(c_pdf)
        ax1.set_xscale("log")

        if drops_pos.size > 0:
            ax2.plot(
                bin_centers,
                bin_density,
                marker="o",
                linestyle="-",
                color=c_plastic,
                label="Plastic-event density (weighted by drop size bins)",
            )
            if xmin_peak is not None:
                ax1.axvline(
                    xmin_peak,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.6,
                )
            ax2.set_xscale("log")
            ax2.set_ylabel(
                "Plastic-event density vs drop size (weighted)", color=c_plastic
            )
            ax2.tick_params(axis="y", colors=c_plastic)
            ax2.spines["right"].set_color(c_plastic)
        else:
            ax2.set_ylabel(
                "Plastic-event density vs drop size (weighted)", color=c_plastic
            )

        # Keep ax1 legend above ax2 plot elements
        ax2.set_zorder(0)
        ax1.set_zorder(1)
        ax1.patch.set_visible(False)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        legend.set_zorder(10)

        fig.tight_layout()
        title = make_title_from_data_info(data_info) if data_info else "plastic_events"
        safe_title = safePath(title)
        filename = f"{PLOTPATH}debug/{safe_title}_plastic_events.pdf"
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)

    return xmin_peak


def dist_from_fit(fit: Fit) -> type[Distribution]:
    dist = getattr(fit, fit.xmin_distribution.name)
    return dist


def _get_cache_path(cache_dir, data, extra_string=""):
    import os
    import hashlib
    from numpy import asarray

    # Creates a unique filename based on data and an optional extra string

    data_bytes = asarray(data).tobytes()
    h = hashlib.sha1()
    h.update(data_bytes)
    data_sig = h.hexdigest() + extra_string
    cache_name = hashlib.sha1(data_sig.encode("utf-8")).hexdigest() + ".json"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)
    return cache_path


def make_fit(
    data,
    xmin_range: tuple | list | None = None,
    distType: type[Distribution] = Truncated_Power_Law,
    use_cache=True,
    cache_dir: str = ".xmin_values",
    fast_xmin=False,
    xmin_accuracy=1.0,
) -> Fit:
    """
    This is a wrapper for the Fit function. Finding xmin takes a long
    time, so we save the results locally and use a precomputed xmin if available.
    """

    # --- try cache
    cache_path = None
    if use_cache:
        import os
        import json
        import gzip

        xmin_fitting_results = None
        cache_path = _get_cache_path(
            cache_dir, data, distType.name + f"{xmin_range}{fast_xmin}{xmin_accuracy}"
        )
        cache_path_json = cache_path
        cache_path_gz = cache_path + ".gz"
        # Prefer the compressed cache if present; fall back to legacy .json
        if os.path.exists(cache_path_gz) or os.path.exists(cache_path_json):
            try:
                with gzip.open(cache_path_gz, "rt", encoding="utf-8") as f:
                    cache = json.load(f)

                xmin_range = cache["xmin"]
                xmin_fitting_results = cache.get("xmin_fitting_results")
                # xmin_range is no longer a tuple, but a single value
                # That means that the fit will be much faster

            except Exception as e:
                # fall through to recompute if loading fails
                print(e)

    # If xmin_range is a tuple, Fit will search for a good xmin, but if we
    # have loaded a precomputed xmin, it will be much faster
    fitObj = Fit(
        data,
        xmin=xmin_range,
        xmin_distribution=distType.name,
        fast_xmin=fast_xmin,
        xmin_accuracy=xmin_accuracy,
    )
    if xmin_fitting_results:
        fitObj.xmin_fitting_results = xmin_fitting_results

    # save xmin if the file does not exsist
    if (
        use_cache
        and cache_path is not None
        and not (os.path.exists(cache_path + ".gz") or os.path.exists(cache_path))
    ):
        try:
            with gzip.open(cache_path + ".gz", "wt", encoding="utf-8") as f:
                json.dump(
                    {
                        "xmin": fitObj.xmin,
                        "xmin_fitting_results": fitObj.xmin_fitting_results,
                    },
                    f,
                )
        except Exception as e:
            # don't fail the computation if saving fails
            print(e)

    return fitObj


def make_exponent_fit(
    csvPaths,
    strainLim=[0.15, 1.0],
    debug=False,
    distType: type[Distribution] = Truncated_Power_Law,
    xmin_range=(1e-7, 1e-5),
    show=True,
    evaluate=True,
):
    raise RuntimeError("Use plot_powerlaw instead")
    fig, ax = plt.subplots()
    drops, data_info = get_energy_drops(csvPaths, strainLim=strainLim, debug=debug)
    if drops.size == 0:
        # No energy drops in strain region
        print(f"No energy drops in {strainLim} strain region: {data_info}")
        return

    # find best xmin
    fit = make_fit(drops, xmin_range=xmin_range, distType=distType)

    if evaluate:
        p, mean_exp, exp_std = fit.evaluate_fit(parallel=True)

    title = make_title(data_info, fit)
    pathName = make_path_name(data_info)
    plot_data_and_fit(fit, ax, title=title)

    filename = f"{PLOTPATH}{pathName}.pdf"
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure to {filename}")
    if show:
        fig.show()


def get_attribute(label):
    if label == "":
        return "Unknown"
    if "=" not in label:
        return label
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


def plot_fits_over_xmin(
    fits, best_fit=None, savePath=None, title=None, xmin_results=None
):
    """
    Plot KS p-value and exponent (with std error bars) versus xmin.
    """
    fits.sort(key=lambda f: f.xmin)
    x = np.array([f.xmin for f in fits], dtype=float)
    pvals = np.array([f.p for f in fits], dtype=float)
    p_stds = np.array([f.p_std for f in fits], dtype=float)
    alphas = np.array([dist_from_fit(f).alpha for f in fits], dtype=float)
    alpha_stds = np.array([f.alpha_std for f in fits], dtype=float)

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
        zorder=2,
    )
    thr_line = ax1.axhline(
        0.10,
        linestyle="--",
        linewidth=1.0,
        color=c_p,
        alpha=0.5,
        label="p = 0.10 threshold",
        zorder=1,
    )
    ax1.set_xlabel(r"$E_{\mathrm{min}}$")
    ax1.set_ylabel("KS p-value", color=c_p)
    ax1.tick_params(axis="y", colors=c_p)
    ax1.spines["left"].set_color(c_p)
    ax1.set_ylim(0, 1)

    # Optional overlay: KS distance (scaled) from xmin fitting results
    ks_line = None
    if xmin_results is not None:
        r = xmin_results
        xmins = np.asarray(r["xmins"], dtype=float)
        distances = np.asarray(r["distances"], dtype=float)
        valid_fits = r.get("valid_fits", None)
        if valid_fits is not None:
            valid_fits = np.asarray(valid_fits, dtype=bool)
        mask = np.isfinite(distances)
        # if valid_fits is not None:
        #    mask &= valid_fits
        if mask.any():
            x_d = xmins[mask]
            d = distances[mask]
            order = np.argsort(x_d)
            x_d = x_d[order]
            d = d[order]
            ks_line = ax1.plot(
                x_d,
                d,
                linestyle="--",
                linewidth=1.2,
                color="0.5",
                alpha=0.7,
                label="KS distance (scaled)",
                zorder=0,
            )[0]
            ax1.axvline(
                x_d[np.argmin(d)],
                color="0.5",
                linestyle="--",
                linewidth=1,
                label=f"Best KS xmin: {best_fit.xmin:.2e}",
                zorder=-1,
                alpha=0.5,
            )

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
        zorder=1,
    )
    ax2.set_ylabel(r"Exponent $\alpha$", color=c_a)
    ax2.tick_params(axis="y", colors=c_a)
    ax2.spines["right"].set_color(c_a)

    if best_fit:
        best_line = ax1.axvline(
            best_fit.xmin,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=rf"Best $p$-xmin: {best_fit.xmin:.2e}",
            zorder=-1,
            alpha=0.5,
        )

    legend_handles = [p_line, thr_line, alpha_err]
    if ks_line is not None:
        legend_handles.append(ks_line)
    if best_fit:
        legend_handles.append(best_line)
    ax1.legend(handles=legend_handles)
    fig.tight_layout()
    ax1.set_title(title)

    if savePath:
        fig.savefig(savePath, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {savePath}")
    plt.close(fig)


def plot_xmin_fitting(fit, save=True, show=False):
    """Plot xmin search diagnostics.

    Expects `fit.xmin_fitting_results` to be a dict with keys:
        distances, alphas, sigmas, xmins, valid_fits

    Produces a plot of KS distance (D) and alpha as a function of candidate xmin.
    NaNs are ignored. A vertical line is drawn at the chosen xmin (`fit.xmin`).
    """

    if not hasattr(fit, "xmin_fitting_results") or fit.xmin_fitting_results is None:
        print("No xmin_fitting_results found on fit object.")
        return None

    r = fit.xmin_fitting_results

    xmins = np.asarray(r["xmins"], dtype=float)
    distances = np.asarray(r["distances"], dtype=float)
    alphas = np.asarray(r["alphas"], dtype=float)

    # Optional keys
    valid_fits = r.get("valid_fits", None)
    if valid_fits is not None:
        valid_fits = np.asarray(valid_fits, dtype=bool)

    # Build mask: finite values and (optionally) valid fits
    mask = np.isfinite(distances)
    if valid_fits is not None:
        mask &= valid_fits

    if mask.sum() == 0:
        print("No valid xmin fitting points after filtering NaNs/invalid fits.")
        return None

    x = xmins[mask]
    D = distances[mask]
    a = alphas[mask]

    # Sort by xmin for nicer curves
    order = np.argsort(x)
    x = x[order]
    D = D[order]
    a = a[order]

    fig, ax1 = plt.subplots()
    c_d = "tab:blue"  # left axis (KS distance)
    c_a = "tab:orange"  # right axis (alpha)

    # Left axis: KS distance
    ax1.plot(
        x,
        D,
        marker="o",
        linestyle="-",
        label="KS distance (D)",
        color=c_d,
        markerfacecolor="none",
        markeredgecolor=c_d,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$x_{\min}$")
    ax1.set_ylabel("KS distance (D)")
    ax1.tick_params(axis="y", colors=c_d)
    ax1.spines["left"].set_color(c_d)

    # Right axis: alpha
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        a,
        marker="s",
        linestyle="--",
        label=r"$\alpha$",
        color=c_a,
        markerfacecolor="none",
        markeredgecolor=c_a,
    )
    ax2.set_ylabel(r"$\alpha$")
    ax2.tick_params(axis="y", colors=c_a)
    ax2.spines["right"].set_color(c_a)

    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    # Chosen xmin
    chosen_xmin = getattr(fit, "xmin", None)
    if chosen_xmin is not None and np.isfinite(chosen_xmin):
        ax1.axvline(
            chosen_xmin,
            linestyle=":",
            linewidth=1.5,
            label=rf"Chosen $x_{{\min}}$ = {chosen_xmin:.2e}",
            alpha=0.9,
        )

    # Title with distribution name when available
    try:
        dist_name = fit.xmin_distribution.name
    except Exception:
        dist_name = ""
    title = "xmin fitting"
    if dist_name:
        title += f" ({pretty_text(dist_name, addEquation=False)})"
    ax1.set_title(title)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    if show:
        plt.show()

    if save:
        os.makedirs(PLOTPATH + "debug/", exist_ok=True)
        dist_tag = dist_name if dist_name else "dist"
        xmin_tag = f"{chosen_xmin:.2e}" if chosen_xmin is not None else "unknown"
        filename = (
            f"{PLOTPATH}debug/xmin_fitting_{dist_tag}_xmin_{xmin_tag}{OUTPUTTYPE}"
        )
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)
        return filename

    return fig, (ax1, ax2)


def plot_powerlaw(
    group_paths=None,
    group_labels=None,
    strainLim: str | list[float] = "auto",
    postRegime=True,
    xmin_range=None,
    debug=False,
    show=False,
    evaluate=True,
    distType: type[Distribution] = Truncated_Power_Law,
    save=True,
    addFit=True,
    fast_xmin=True,
    xmin_accuracy=1.0,
    csvPaths=None,
):
    if group_paths is None and csvPaths is not None:
        group_paths = csvPaths

    if group_paths is None:
        print("No paths provided.")
        return

    def _is_nested_paths(paths):
        if len(paths) == 0:
            return False
        if isinstance(paths, (str, os.PathLike)):
            return False
        return isinstance(paths[0], (list, tuple, np.ndarray))

    if isinstance(group_paths, (str, os.PathLike)):
        group_paths = [str(group_paths)]

    if _is_nested_paths(group_paths):
        normalized_paths = [list(p) for p in group_paths]
    else:
        normalized_paths = [list(group_paths)]

    if group_labels is None:
        normalized_labels = [[""] * len(paths) for paths in normalized_paths]
    else:
        if isinstance(group_labels, (str, os.PathLike)):
            group_labels = [str(group_labels)]

        if _is_nested_paths(group_labels):
            normalized_labels = [list(l) for l in group_labels]
        else:
            if len(normalized_paths) == 1:
                normalized_labels = [list(group_labels)]
            elif len(group_labels) == len(normalized_paths):
                normalized_labels = [
                    [str(label)] * len(paths)
                    for label, paths in zip(group_labels, normalized_paths)
                ]
            else:
                normalized_labels = [[""] * len(paths) for paths in normalized_paths]

    for paths, labels in zip(normalized_paths, normalized_labels):
        if len(labels) < len(paths):
            labels = labels + [""] * (len(paths) - len(labels))
        elif len(labels) > len(paths):
            labels = labels[: len(paths)]
        all_drops = []
        data_info = None
        for path, label in zip(paths, labels):
            df = pd.read_csv(path)
            df = update_df_header(df)
            if strainLim == "auto":
                gamma_max_stress = findPrePostSplit(df=df)
                if postRegime:
                    strainLim = [gamma_max_stress + 1e-2, df["load"].max()]
                else:
                    strainLim = [df["load"].min(), gamma_max_stress - 1e-4]
            if data_info is None:
                _, data_info = get_energy_drops(
                    paths, strainLim=strainLim, debug=debug, label=label
                )
            drops, _, _ = get_drops_in_windows(
                path, df=df, strainLim=strainLim, debug=debug, label=label
            )
            all_drops.extend(drops)  # drops is a list of arrays

        find_start_of_plastic_events(paths, postRegime, data_info, debug=True)

        # After the loop
        all_drops = np.concatenate(all_drops)

        if len(all_drops) == 0:
            group_label = labels[0] if labels else ""
            print(
                f"No valid drops found for {group_label} in strain range {strainLim}."
            )
            continue
        if xmin_range is None and not fast_xmin:
            # Not using fast_xmin is brutally slow. We add a default range here
            xmin_range = [1e-9, 1]
        fit = make_fit(
            data=all_drops,
            xmin_range=xmin_range,
            distType=distType,
            fast_xmin=fast_xmin,
            xmin_accuracy=xmin_accuracy,
        )

        best_fit = find_best_xmin(
            all_drops,
            debug=debug,
            data_info=data_info,
            xmin_results=fit.xmin_fitting_results,
        )

        d = dist_from_fit(fit)

        attribute = get_attribute(labels[0])
        if attribute == "Unkown":
            assert isinstance(data_info, dict)
            if "minimizer" in data_info:
                attribute = data_info["minimizer"]

        if attribute in MINIMIZER_COLORS:
            color = MINIMIZER_COLORS[attribute]
        else:
            color = "black"

        if evaluate:
            p, mean_exp, exp_std = fit.evaluate_fit(all_drops, parallel=True)

            thresholds = [0.05, 0.1, 0.3, float("inf")]
            ratings = ["bad", "poor", "good", "excellent"]

            # Set r
            for t, r in zip(thresholds, ratings):
                if p < t:
                    break

            print(f"Number of drops: {len(all_drops)}")
            print(
                f"{attribute}: P value: {p:.2f} ({r}), exp: {d.alpha}, std: {exp_std}"
            )

        min_drop = np.min(all_drops)
        # We exclude the first decade. Usually not interesting
        exclude_factor = 10
        plot_ks_distance(all_drops, min_drop * exclude_factor)  # fit.xmin)

        title = make_title(data_info=data_info, fit=fit)
        if attribute:
            title = attribute + " " + title
        plot_data_and_fit(
            fit,
            title=title,
            color=color,
            addFit=addFit,
            save=save,
            show=show,
        )


if __name__ == "__main__":
    # csvPath = "/Volumes/data/MTS2D_output/unfixed_simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    strainLim = [0.65, 1.0]
    paths = [
        f"/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s{i}/macroData.csv"
        for i in range(10)
    ]
    # paths = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    # strainLim = [1, 3]
    # csvPath = "/Volumes/data/MTS2D_output/simpleShear,s400x400l0.15,1e-05,1.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"

    plot_powerlaw(
        csvPaths=paths,
        strainLim=strainLim,
        debug=True,
        distType=Truncated_Power_Law,
        # xmax=1e-4,
    )

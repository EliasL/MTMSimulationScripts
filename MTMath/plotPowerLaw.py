import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .evaluatePowerlawFit import Fit, Truncated_Power_Law
from powerlaw import Distribution

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
    data = fit.data_original
    data.sort()
    # Plot
    tail_frac = (data >= fit.xmin).mean()
    bins_for_model = np.unique(data)
    # scale down to match full-data density
    f = dist._pdf_base_function(data)
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
        annotate_ks_distance_pdf(ax, fit.xmin, dist.D)

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
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    n = data_info["nrSimulations"]
    samples_string = f"{n} sample{'s' if n != 1 else ''}"
    title = rf"{data_info['minimizer']} {L}x{L} {samples_string} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f}"
    return title


def make_title(data_info=None, fit: Fit = None):
    title = ""
    if data_info:
        title += make_title_from_data_info(data_info)
    if fit:
        if data_info:
            title += " "
        title += make_title_from_fit(fit)
    return title


def make_path_name(data_info):
    if data_info is None:
        return "unkown"
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    path_name = f"{L}x{L}{data_info['minimizer']}_s{strainLim[0]:.2f}-{strainLim[1]:.2f}_{data_info['nrSimulations']}samples"
    return path_name


def plot_data_and_fit(
    fit: Fit,
    ax=None,
    title="",
    color=None,
    addFit=True,
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
    ax.set_title(title)
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


def plot_ks_distance(drops, xmin, dist_name="truncated_power_law"):
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
    ax.set_title("Kolmogorov–Smirnov Distance (CCDF)")
    ax.legend()
    fig.tight_layout()
    # Save the plot
    filename = f"{PLOTPATH}ks_distance_xmin_{xmin:.2e}_{dist_name}{OUTPUTTYPE}"
    fig.savefig(filename, dpi=300)
    # plt.show()


def dist_from_fit(fit: Fit) -> Distribution:
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
    xmin_range: tuple = None,
    distType: Distribution = Truncated_Power_Law,
    use_cache=True,
    cache_dir: str = ".xmin_values",
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

        cache_path = _get_cache_path(cache_dir, data, distType.name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    cache = json.load(f)
                    xmin_range = cache["xmin"]
                    # xmin_range is no longer a tuple, but a single values
                    # That means that the fit will be much faster

            except Exception as e:
                # fall through to recompute if loading fails
                print(e)

    # If xmin_range is a tuple, Fit will search for a good xmin, but if we
    # have loaded a precomputed xmin, it will be much faster
    fitObj = Fit(data, xmin=xmin_range, xmin_distribution=distType.name)

    # save xmin if the file does not exsist
    if use_cache and cache_path is not None and not os.path.exists(cache_path):
        try:
            with open(cache_path, "w") as f:
                json.dump({"xmin": fitObj.xmin}, f, indent=4)
        except Exception as e:
            # don't fail the computation if saving fails
            print(e)

    return fitObj


def make_exponent_fit(
    csvPaths,
    strainLim=[0.15, 1.0],
    debug=False,
    DistType: Distribution = Truncated_Power_Law,
    xmin_range=(1e-7, 1e-5),
    show=True,
    evaluate=True,
):
    fig, ax = plt.subplots()
    drops, data_info = get_energy_drops(csvPaths, strainLim=strainLim, debug=debug)
    if drops.size == 0:
        # No energy drops in strain region
        print(f"No energy drops in {strainLim} strain region: {data_info}")
        return

    # find best xmin
    fit = make_fit(drops, xmin_range=xmin_range, distType=DistType)

    if evaluate:
        p, mean_exp, exp_std = fit.evaluate_fit(parallel=True)

    title = make_title(data_info, fit)
    pathName = make_path_name(data_info)
    plot_data_and_fit(fit, ax, title=title)

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
    xmin=None,
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

        fit = make_fit(data=all_drops, xmin=xmin)

        title = rf"$\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f},  $E_{{\mathrm{{min}}}}$={xmin:.2e}, $\alpha=${fit.alpha:.2f}"
        attribute = get_attribute(labels[0])
        title = attribute + " " + title
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
                f"{attribute}: P value: {p:.2f} ({r}), exp: {fit.alpha}, std: {exp_std}"
            )

        plot_data_and_fit(
            all_drops,
            fit,
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

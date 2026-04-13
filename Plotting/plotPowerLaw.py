from .findXmin import find_xmin, find_xmin_rising_level
from MTMath.energyFunction import ContiEnergy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from MTMath.evaluatePowerlawFit import Fit, Truncated_Power_Law
from powerlaw import Distribution
from Management.updateCSV import update_df_header
from .makePlots import safePath, maybe_avg
import os
import glob
import tempfile
import uuid
from tqdm import tqdm
import warnings

np.random.seed(0)
# Create directories for saving plots
PLOTPATH = "Plots/powerLaw/"
OUTPUTTYPE = ".png"
os.makedirs(PLOTPATH, exist_ok=True)
os.makedirs(PLOTPATH + "debug/", exist_ok=True)
MINIMIZER_COLORS = {"L-BFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}


def _resolve_fast_xmin(fast_xmin=None, fit=None, fits=None):
    if fast_xmin is not None:
        return bool(fast_xmin)
    if fit is not None:
        return bool(getattr(fit, "fast_xmin", False))
    if fits:
        vals = [bool(getattr(f, "fast_xmin", False)) for f in fits]
        if vals and all(v == vals[0] for v in vals):
            return vals[0]
        return any(vals)
    return False


def ks_tag(*, fast_xmin=None, fit=None, fits=None, lower=False):
    use_fast = _resolve_fast_xmin(fast_xmin=fast_xmin, fit=fit, fits=fits)
    if lower:
        return "sks" if use_fast else "ks"
    return "SKS" if use_fast else "KS"


def _append_sample_suffix(name, n_samples):
    if n_samples is None:
        return name
    try:
        n_int = int(n_samples)
    except (TypeError, ValueError):
        return name
    return f"{name}_n{n_int}"


# def ax.figure -> mpl.figure.Figure:
#     fig = ax.figure
#     # SubFigure doesn't implement tight_layout; use the parent Figure instead.
#     if isinstance(fig, mpl.figure.Figure):
#         return fig
#     return fig.figure


def get_minimizer(df):
    # We can find out what minimizer we have by looking at the third line in the
    # csv file, and seeing which of
    # LBFGS_Term_reason,CG_Term_reason,FIRE_Term_reason is non-zero
    if "LBFGS_Term_reason" in df:
        if not len(df["LBFGS_Term_reason"]) >= 2:
            print("No data in file!")
            return "Unknown"
    else:
        return "Unknown"

    # We use iloc so that if for example the second half of the file is loaded,
    # we take the third line from that arbitrary startingpoint, instead of trying
    # to find the third line from the top of the file (which might not be loaded)
    LBFGS = df["LBFGS_Term_reason"].iloc[2]
    CG = df["CG_Term_reason"].iloc[2]
    FIRE = df["FIRE_Term_reason"].iloc[2]
    zeros = [int(LBFGS) == 0, int(CG) == 0, int(FIRE) == 0]
    if all(zeros):
        return "None"
    assert sum(zeros) == 2, "There is not exactly one non-zero term reason!"
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
        match = re.search(r"(\d+)x(\d+)", str(path))
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


def _resolve_strain_lim(strainLim, *, df, postRegime):
    """
    Resolve "auto"/"all"/None strain limits based on the stress peak.
    """
    if strainLim == "all":
        return [-np.inf, np.inf]
    if strainLim is None or strainLim == "auto":
        gamma_max_stress = findPrePostSplit(df=df)
        if postRegime is None:
            return [df["load"].min(), df["load"].max()]
        if postRegime:
            return [gamma_max_stress + 1e-2, df["load"].max()]
        return [df["load"].min(), gamma_max_stress - 1e-4]
    return strainLim


def get_energy_drops(
    csvPaths,
    df=None,
    strainLim: str | list[float] = "auto",
    debug=False,
    label=None,
    onlyStrainedEnergyDrops=False,
    postRegime=True,
    energy_type="e_change_from_init",
    averageEnergy=False,
):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    When onlyStrainedEnergyDrops is true, we still use e_change_from_init, but only
    when energy_chage is negative (there is a drop between relaxed states).
    If debug=True, plot intermediate energy and drop traces.
    """
    if isinstance(csvPaths, str):
        csvPaths = [csvPaths]

    drops = []
    masks = []
    dfs = []
    L = get_system_size(csvPaths)
    read_from_paths = df is None

    if averageEnergy is None:
        prefix = ""
    else:
        prefix = "avg_" if averageEnergy else "total_"
    # energyType = "e_change_from_init"
    # energyType = "energy_change"
    energy_key = prefix + energy_type

    for singlePath in csvPaths:
        resolved_strain_lim = strainLim
        if read_from_paths:
            df_local = pd.read_csv(singlePath)
            df_local = update_df_header(df_local, L=L)
            dfs.append(df_local)
        else:
            df_local = df
        if resolved_strain_lim in ("auto", "all", None):
            resolved_strain_lim = _resolve_strain_lim(
                resolved_strain_lim, df=df_local, postRegime=postRegime
            )

        diffs = df_local[energy_key]

        if np.all(diffs >= 0):
            # Umut code compatability: I'm just assuming if all the drops are positive
            # they should probably be flipped.
            diffs = -diffs

        strain = df_local["load"]

        # The first minimization always has a unaturally large jump
        # set the diff at load step 1 = 0
        if "load_step" in df_local and df_local["load_step"].iloc[0] == 1:
            diffs.iloc[0] = 0

        lim_mask = (strain > resolved_strain_lim[0]) & (strain < resolved_strain_lim[1])
        drop_mask = diffs < 0
        mask = drop_mask & lim_mask

        if onlyStrainedEnergyDrops:
            # We use a negative change between relaxed states as a proxy to
            # distinguish affine relaxation from plastic relaxation.
            realPlasticDrop = df_local["nr_elements_with_m3_fix_change"] >= 1
            mask = mask & realPlasticDrop

        drops.extend(-diffs[mask])
        masks.append(mask)

    drops = np.array(drops)
    concat_df = pd.concat(dfs, ignore_index=True) if read_from_paths else df

    data_info = get_energy_drops_info(
        csvPaths=csvPaths,
        drops=drops,
        energy_key=energy_key,
        df=concat_df,
        strainLim=resolved_strain_lim,
        label=label,
        masks=masks,
    )

    if debug:
        # Only debug first seed when using labels
        if label is not None and "seed=" in label and "seed=0" not in label:
            return drops, data_info

        strain_limited = strain[1:][lim_mask[1:]]
        plotDrops = np.clip(-diffs[1:][lim_mask[1:]], 0, np.inf)
        e = df_local[energy_key]
        debug_fig, ax1 = plt.subplots()
        avg_label = maybe_avg("E", averageEnergy)
        ax1.plot(strain, e, label=rf"${avg_label}$")
        ax1.set_ylabel(rf"${avg_label}$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        if averageEnergy:
            pre_post = maybe_avg("E_{\\mathrm{pre}-E_{\\mathrm{post}}}", True)
            label = rf"$- {pre_post}$"
        else:
            delta_e = maybe_avg("E", False)
            label = rf"$-\Delta {delta_e}$"
        ax2.plot(strain_limited, plotDrops, label=label)
        avg_delta = maybe_avg("E", averageEnergy)
        ax2.set_ylabel(rf"$-\Delta {avg_delta}$ (Energy Drop)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines + lines2,
            labels + labels2,
            loc="upper right",
            ncol=2,
            frameon=True,
        )
        # handle drops.max() = NaN case
        if np.isnan(drops.max()):
            ax2.set_ylim(0, 1)
        else:
            ax2.set_ylim(0, drops.max() * 1.5)

        # ——— Compute 0.1%‐wide central slice ———
        mid = 0.5 * (resolved_strain_lim[0] + resolved_strain_lim[1])
        total_width = resolved_strain_lim[1] - resolved_strain_lim[0]
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
        debug_fig.tight_layout(rect=[0, 0, 1, 0.9])
        debug_fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        # to save memory, close the figure
        plt.close(debug_fig)

    return drops, data_info


def get_energy_drops_info(
    csvPaths,
    drops,
    energy_key,
    *,
    df=None,
    strainLim=None,
    label=None,
    masks=None,
):
    """
    Collect metadata for energy-drop data without recomputing the drops.
    """
    if isinstance(csvPaths, str):
        csvPaths = [csvPaths]
    info = {}
    if df is not None:
        info["df"] = df
        info["minimizer"] = get_minimizer(df)
    info["nrSimulations"] = len(csvPaths)
    info["drops"] = drops
    info["key"] = energy_key
    if strainLim is not None:
        info["strainLim"] = strainLim
    info["L"] = get_system_size(csvPaths)
    if label is not None:
        info["label"] = label
    if masks is not None:
        info["masks"] = masks
    if df is not None and masks:
        if isinstance(masks, (list, tuple)):
            if len(masks) == 1:
                combined_mask = np.asarray(masks[0])
            else:
                combined_mask = np.concatenate([np.asarray(mask) for mask in masks])
        else:
            combined_mask = np.asarray(masks)
        if combined_mask.size == len(df):
            info["mask"] = combined_mask
        else:
            warnings.warn(
                "Combined mask length does not match dataframe length; "
                "skipping data_info['mask'].",
                RuntimeWarning,
            )
    return info


def getHist(data, weights=None, density=True, bins_per_decade=5):
    data = np.asarray(data)
    if data.size == 0:
        return np.array([]), np.array([])

    weights_arr = None
    if weights is not None:
        weights_arr = np.asarray(weights)

    # Histogram bins require positive values for log scaling
    if np.any(data <= 0):
        mask = data > 0
        data = data[mask]
        if weights_arr is not None:
            if weights_arr.shape == mask.shape:
                weights_arr = weights_arr[mask]
            else:
                weights_arr = None
        if data.size == 0:
            return np.array([]), np.array([])

    # Find the start of the tail where Poisson noise exceeds threshold
    data_min = data.min()
    data_max = data.max()

    # Compute number of bins from x_min to data_max
    decades = np.log10(data_max) - np.log10(data_min)
    n_bins = max(10, int(np.ceil(decades * bins_per_decade)))
    # Define bin edges from data_max downward
    log_edges = np.log10(data_max) - np.arange(n_bins + 1) / bins_per_decade
    bin_edges = np.power(10, log_edges)[::-1]  # Reverse to make it ascending

    # Compute the histogram for the tail (density=True → area under PDF = 1)
    hist_vals, edges = np.histogram(
        data, bins=bin_edges, weights=weights_arr, density=density
    )
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

    data = np.asarray(data)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return

    # Choose edgecolor if None
    if edgecolor is None:
        edgecolor = ax._get_lines.get_next_color()

    bin_centers, hist_vals = getHist(data)

    if label is None:
        if data.size < 1e4:
            nrDrops = data.size
        else:
            nrDrops = f"{data.size:.1e}"
        label = f"PDF of {nrDrops} energy drops"

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
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$ (Energy Drop)")
    ax.set_ylabel(rf"$p(-\Delta {maybe_avg('E')})$")
    ax.legend()


def plot_data_cdf(
    ax,
    data,
    label=None,
    color=None,
    alpha=1,
    use_ccdf=True,
):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        return

    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    if use_ccdf:
        y_vals = 1.0 - cdf
        ylabel = r"$P(X > x)$"
    else:
        y_vals = cdf
        ylabel = r"$P(X \leq x)$"

    if label is None:
        if data.size < 1e4:
            nrDrops = data.size
        else:
            nrDrops = f"{data.size:.1e}"
        label = f"{'CCDF' if use_ccdf else 'CDF'} of {nrDrops} energy drops"

    plot_kwargs = {
        "label": label,
        "alpha": alpha,
        "where": "post",
    }
    if color is not None:
        plot_kwargs["color"] = color

    ax.step(sorted_data, y_vals, **plot_kwargs)

    ax.set_xscale("log")
    if use_ccdf:
        ax.set_yscale("log")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$ (Energy Drop)")
    ax.set_ylabel(ylabel)
    ax.legend()


def plot_ks_distance_marker(
    ax, sorted_data, ecdf, model_ccdf, color="red", fast_xmin=None
):
    diffs = np.abs(ecdf - model_ccdf)
    max_index = np.argmax(diffs)
    D_val = diffs[max_index]
    x_D = sorted_data[max_index]
    tag = ks_tag(fast_xmin=fast_xmin)
    ax.vlines(
        x_D,
        model_ccdf[max_index],
        ecdf[max_index],
        color=color,
        linestyle="--",
        label=f"{tag} Distance D = {D_val:.3f}",
    )
    ax.scatter([x_D], [ecdf[max_index]], color="blue")
    ax.scatter([x_D], [model_ccdf[max_index]], color="gray")
    return D_val


# --- Helper for annotating KS distance on PDF plot ---
def annotate_ks_distance_pdf(ax, xmin, D_val, color="red", fast_xmin=None, fit=None):
    tag = ks_tag(fast_xmin=fast_xmin, fit=fit)
    ax.axvline(
        xmin,
        color=color,
        linestyle="--",
        linewidth=1.2,
        label=f"{tag} Distance D = {D_val:.3f}",
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
            ax, fit.xmin_fitting_results["xmins"][dist.D_i], dist.D, fit=fit
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$ (Energy Drop)")
    ax.set_ylabel(rf"$p(-\Delta {maybe_avg('E')})$")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_fit_cdf(
    ax,
    fit: Fit,
    title=None,
    color=None,
    alpha=1,
    linestyle="-",
    pre_label=None,
    use_ccdf=True,
):
    dist = dist_from_fit(fit)

    data = np.asarray(fit.data_original).copy()
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        return ax
    data.sort()

    tail_frac = float((data >= fit.xmin).mean())
    bins_for_model = np.unique(data)
    bins_for_model = bins_for_model[bins_for_model > 0]
    bins_for_model = bins_for_model[bins_for_model >= fit.xmin]
    if bins_for_model.size == 0:
        return ax

    model_ccdf = dist.ccdf(bins_for_model)
    if use_ccdf:
        y_vals = model_ccdf * tail_frac
        ylabel = r"$P(X > x)$"
    else:
        y_vals = 1.0 - model_ccdf * tail_frac
        ylabel = r"$P(X \leq x)$"

    ax.plot(
        bins_for_model,
        y_vals,
        label=(pre_label or "") + pretty_text(dist.name),
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )

    ax.set_xscale("log")
    if use_ccdf:
        ax.set_yscale("log")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$ (Energy Drop)")
    ax.set_ylabel(ylabel)
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
        df = update_df_header(df, L=get_system_size([csvPath]))
    if "avg_P12" in df:
        i = df["avg_P12"].argmax()
    elif "avg_sigma12" in df:
        i = df["avg_sigma12"].argmax()
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
        df = update_df_header(df, L=get_system_size([csvPath]))

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
    def _strip_seed(label):
        if label is None:
            return ""
        tokens = [tok.strip() for tok in str(label).split(",") if tok.strip()]
        tokens = [t for t in tokens if not t.startswith("seed=")]
        return ", ".join(tokens)

    if "customTitle" in data_info:
        return data_info["customTitle"]
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    n = data_info["nrSimulations"]
    samples_string = f"{n} sample{'s' if n != 1 else ''}"
    title = (
        rf"{L}x{L} {samples_string} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f}"
    )
    if data_info["minimizer"] != "Unknown":
        title = rf"{data_info['minimizer']} " + title
    if "label" in data_info:
        l = data_info["label"]
        if isinstance(l, list):
            normalized = [_strip_seed(item) for item in l]
            normalized_non_empty = [item for item in normalized if item]
            if normalized_non_empty and len(set(normalized_non_empty)) == 1:
                title += normalized_non_empty[0]
            else:
                assert len(set(normalized)) == 1, (
                    f"Labels in group are different: {set(normalized)}"
                )  # Which should we use?
                title += normalized[0]
        else:
            title += _strip_seed(l)
    title = title.strip().replace("  ", " ")
    return title


def make_title(data_info=None, fit: Fit | None = None):
    title = ""
    if data_info:
        title += make_title_from_data_info(data_info)
    if fit:
        if data_info:
            title += " "
        title += make_title_from_fit(fit)
    return title


def plot_data_and_fit(
    fit: Fit,
    ax=None,
    title="",
    data_info=None,
    color=None,
    addFit=True,
    useCDF=False,
    useCCDF=True,
    save=True,
    extraPath="",
    show=False,
    close=True,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if useCDF:
        plot_data_cdf(ax, fit.data_original, use_ccdf=useCCDF)
    else:
        plot_data_pdf(ax, fit.data_original)

    # plot the fit
    if addFit:
        if useCDF:
            plot_fit_cdf(ax, fit, color=color, use_ccdf=useCCDF)
        else:
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
        if useCDF:
            safe_title += "_ccdf" if useCCDF else "_cdf"
        else:
            safe_title += "_pdf"
        safe_title = _append_sample_suffix(safe_title, len(fit.data_original))
        filename = f"{PLOTPATH}{extraPath}{safe_title}.pdf"
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {filename}")
        setattr(fig, "path", filename)
    else:
        setattr(fig, "path", None)

    if close:
        plt.close(fig)
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


def plot_ks_distance(
    drops,
    xmin,
    dist_name="truncated_power_law",
    data_info=None,
    name="",
    ax=None,
    save=True,
    close=True,
    extraPath="",
    fast_xmin=None,
):
    """
    Plot the empirical CCDF vs the fitted CCDF and visually show the KS distance (D).
    """
    # Fit the distribution
    fitObj = Fit(drops, xmin=xmin, xmin_distribution=dist_name)
    dist = getattr(fitObj, dist_name)
    # Get the ECDF and model CCDF
    data = fitObj.data
    sorted_data = np.sort(data[data >= xmin])
    ecdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    model_ccdf = dist.ccdf(sorted_data)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.step(sorted_data, ecdf, where="post", label="Empirical CCDF", color="blue")
    ax.plot(
        sorted_data,
        model_ccdf,
        label=rf"Model CCDF ($\alpha={dist.alpha:.2f}, \lambda=$ {dist.Lambda:.2e})",
        color="gray",
    )
    D_val = plot_ks_distance_marker(
        ax, sorted_data, ecdf, model_ccdf, fast_xmin=fast_xmin
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$")
    ax.set_ylabel(r"$P(X > x)$")
    tag = ks_tag(fast_xmin=fast_xmin)
    title = rf"{tag} Distance $E_{{\mathrm{{min}}}}$={xmin:.2e}" + (
        " " + name if name != "" else ""
    )
    if data_info is not None:
        title += make_title_from_data_info(data_info)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    # Save the plot
    safe_title = safePath(title)
    safe_title = _append_sample_suffix(safe_title, len(drops))
    filename = f"{PLOTPATH}{extraPath}{safe_title}.pdf"
    if save:
        fig.savefig(filename, dpi=300)
        print(f"Saved fig to {filename}")
        setattr(fig, "path", filename)
    else:
        setattr(fig, "path", None)
    # plt.show()
    if close:
        plt.close(fig)

    return ax


_MEMMAP_CACHE: dict[tuple[str, str, tuple[int, ...]], np.memmap] = {}


def _load_memmap(spec):
    key = (
        spec["memmap_path"],
        spec["dtype"],
        tuple(spec["shape"]),
    )
    if key not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[key] = np.memmap(
            spec["memmap_path"],
            mode="r",
            dtype=np.dtype(spec["dtype"]),
            shape=tuple(spec["shape"]),
        )
    return _MEMMAP_CACHE[key]


def _fit_single_xmin_task(args):
    drops_spec, trial_xmin, xmax, dist_name, confidence = args
    if isinstance(drops_spec, dict) and "memmap_path" in drops_spec:
        drops = _load_memmap(drops_spec)
    else:
        drops = drops_spec
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
    parallel=True,
    max_workers=None,
    use_memmap=True,
    memmap_min_size=5e4,
    memmap_dir=None,
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

    drops_spec = drops
    memmap_path = None
    if parallel and use_memmap and len(drops) >= memmap_min_size:
        drops_array = np.asarray(drops)
        memmap_dir = memmap_dir or tempfile.gettempdir()
        os.makedirs(memmap_dir, exist_ok=True)
        memmap_path = os.path.join(
            memmap_dir, f"explore_xmin_{os.getpid()}_{uuid.uuid4().hex}.dat"
        )
        mmap = np.memmap(
            memmap_path,
            dtype=drops_array.dtype,
            mode="w+",
            shape=drops_array.shape,
        )
        mmap[:] = drops_array[:]
        mmap.flush()
        drops_spec = {
            "memmap_path": memmap_path,
            "dtype": str(drops_array.dtype),
            "shape": drops_array.shape,
        }

    tasks = [
        (drops_spec, float(trial_xmin), xmax, distType.name, confidence)
        for trial_xmin in xmin_values
    ]

    try:
        if parallel:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                fits_iter = ex.map(_fit_single_xmin_task, tasks)
                test_fits = list(
                    tqdm(
                        fits_iter,
                        total=len(tasks),
                        desc="Fitting xmins",
                        disable=False,
                    )
                )
        else:
            raise RuntimeError("Parallel disabled")
    except KeyboardInterrupt:
        print("Parallel xmin exploration interrupted.")
        raise
    except Exception as e:
        if parallel:
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
    finally:
        if memmap_path and os.path.exists(memmap_path):
            try:
                os.remove(memmap_path)
            except OSError:
                pass

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
    extraPath="",
    fast_xmin=None,
    parallel=True,
    max_workers=None,
    use_memmap=True,
    memmap_min_size=5e4,
    memmap_dir=None,
):
    """
    We scan many possible xmin values. We try to identify a plateau region
    in the exponents. We make sure the p-value is larger than min_p. If the
    p-value is close to the min_p limit, we need to increaes the accuracy.
    """

    title = make_title(data_info)
    path_name = safePath(title)
    path_name = _append_sample_suffix(path_name, len(drops))

    print(f"Testing xmins for {title}")

    if parallel and len(drops) > 5e4 and not use_memmap:
        parallel = False

    test_fits = explore_xmin(
        drops,
        min_xmin,
        max_xmin,
        nr_evaluation,
        start_accuracy,
        DistType,
        debug,
        xmax=xmax,
        parallel=parallel,
        max_workers=max_workers,
        use_memmap=use_memmap,
        memmap_min_size=memmap_min_size,
        memmap_dir=memmap_dir,
    )

    # We now have a rough sample on possible xmin values
    # exponents = [dist_from_fit(f).alpha for f in test_fits]
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
            parallel=parallel,
            max_workers=max_workers,
            use_memmap=use_memmap,
            memmap_min_size=memmap_min_size,
            memmap_dir=memmap_dir,
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
    xmin_plot_path = f"{PLOTPATH}{extraPath}{path_name}_xMins.pdf"
    plot_fits_over_xmin(
        test_fits,
        best_fit,
        xmin_plot_path,
        title=title,
        xmin_results=xmin_results,
        fast_xmin=fast_xmin,
    )
    setattr(best_fit, "xmin_plot_path", xmin_plot_path)

    return best_fit


def find_start_of_plastic_events(data_info, debug=False, binsPerDecade=5):
    if data_info is None or "df" not in data_info or "mask" not in data_info:
        warnings.warn("Missing df/mask in data_info; cannot analyze plastic events.")
        return None

    all_data = data_info["df"]
    mask = data_info["mask"]
    drops = data_info["drops"]
    if "nr_elements_with_m3_fix_change" in all_data:
        plastics = all_data["nr_elements_with_m3_fix_change"][mask].to_numpy()
    else:
        warnings.warn("nr_elements_with_m3_fix_change column not found.")
        return None

    xmin_loc_min = None
    if drops.size > 0:
        bin_centers, bin_sums = getHist(
            drops, weights=plastics, density=False, bins_per_decade=binsPerDecade
        )
        # Find first local minimum
        if len(bin_sums) >= 3:
            for i in range(1, len(bin_sums) - 1):
                if bin_sums[i] <= bin_sums[i - 1] and bin_sums[i] <= bin_sums[i + 1]:
                    xmin_loc_min = bin_centers[i]
                    break

    if debug:
        ax = plot_plastic_counts(
            info=data_info, binsPerDecade=binsPerDecade, save=False
        )
        # Optional: mark xmin_peak on x-axis
        if xmin_loc_min is not None:
            ax.axvline(
                xmin_loc_min,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
            )
        fig = ax.figure
        fig.tight_layout()
        title = make_title_from_data_info(data_info) if data_info else "plastic_events"
        safe_title = safePath(title)
        filename = f"{PLOTPATH}debug/{safe_title}_plastic_events.pdf"
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)
    return xmin_loc_min


def plot_plastic_counts(
    paths=None,
    info=None,
    binsPerDecade=5,
    postRegime=True,
    strainLim="auto",
    ax=None,
    show=False,
    save=True,
):
    if info is not None:
        assert paths is None, "Only provide info or paths"
        drops = info["drops"]
    else:
        drops, info = get_energy_drops(
            paths,
            strainLim=strainLim,
            debug=False,
            label=None,
            postRegime=postRegime,
        )

    drops = np.asarray(drops)
    valid = np.isfinite(drops) & (drops > 0)
    if not np.any(valid):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title("Energy-drop PDF and plasticity vs drop size")
            ax.set_xlabel(rf"$-\Delta {maybe_avg('W')}$")
            ax.set_ylabel("Density (normalized)")
        return ax

    drops = drops[valid]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax2 = ax.twinx()

    mask = info.get("mask")
    df = info["df"]
    assert mask is not None
    plastics = df["nr_elements_with_m3_fix_change"][mask].to_numpy()
    if plastics.size == valid.size:
        plastics = plastics[valid]

    bin_centers, bin_sums = getHist(
        drops, weights=plastics, density=False, bins_per_decade=binsPerDecade
    )
    bin_centers, bin_density = getHist(
        drops, weights=plastics, density=True, bins_per_decade=binsPerDecade
    )

    c_drop_pdf = "tab:blue"
    c_plastic_pdf = "tab:green"
    c_plastic_counts = "tab:orange"

    # 1) Energy-drop PDF (normalized) on ax1
    plot_data_pdf(ax, drops)
    ax.set_title("Energy-drop PDF and plasticity vs drop size")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('W')}$")
    ax.set_ylabel("Density (normalized)", color=c_drop_pdf)
    ax.tick_params(axis="y", colors=c_drop_pdf)
    ax.spines["left"].set_color(c_drop_pdf)
    ax.set_xscale("log")

    # Plastic-event PDF vs drop size (normalized over plastic events) on ax1
    plastic_pdf = bin_density  # W_i / (sum W) / Δx_i
    ax.plot(
        bin_centers,
        plastic_pdf,
        marker="o",
        linestyle="--",
        color=c_plastic_pdf,
        label="Plastic-event PDF",
    )

    # 2) Raw plastic-event counts per bin (no scaling) on ax2
    ax2.plot(
        bin_centers,
        bin_sums,
        marker="o",
        linestyle="-",
        color=c_plastic_counts,
        label="Plastic events per drop-size bin",
    )
    ax2.set_xscale("log")
    ax2.set_ylabel("Nr plastic events", color=c_plastic_counts)
    ax2.tick_params(axis="y", colors=c_plastic_counts)
    ax2.spines["right"].set_color(c_plastic_counts)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        ncol=2,
        frameon=True,
    )
    fig.tight_layout()
    if save:
        title = make_title_from_data_info(info) if info else "plastic_events"
        safe_title = safePath(title)
        filename = f"{PLOTPATH}debug/{safe_title}_plastic_events.pdf"
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)
    if show:
        plt.show()
    return ax


def plot_plastic_counts_compare(
    paths,
    labels=None,
    postRegime=True,
    strainLim="auto",
    binsPerDecade=5,
    ax=None,
    show=False,
    save=True,
    filename=None,
    name=None,
):
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    if labels is None:
        labels = ["" for _ in paths]
    if len(labels) < len(paths):
        labels = labels + [""] * (len(paths) - len(labels))
    elif len(labels) > len(paths):
        labels = labels[: len(paths)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    else:
        fig = ax.figure

    blues = mpl.colormaps["Blues"]
    oranges = mpl.colormaps["Oranges"]

    n_edge = sum("edgeFlip" in p for p in paths)
    n_norm = len(paths) - n_edge
    edgeflip_colors = iter(blues(np.linspace(0.3, 0.9, max(1, n_edge))))
    normal_colors = iter(oranges(np.linspace(0.3, 0.9, max(1, n_norm))))
    for path, label in zip(paths, labels):
        drops, info = get_energy_drops(
            [path],
            strainLim=strainLim,
            debug=False,
            label=None,
            postRegime=postRegime,
        )

        mask = info.get("mask")
        df = info["df"]
        assert mask is not None
        plastics = df["nr_elements_with_m3_fix_change"][mask].to_numpy()

        bin_centers, bin_sums = getHist(
            drops, weights=plastics, density=False, bins_per_decade=binsPerDecade
        )
        if bin_centers.size == 0:
            continue

        if "edgeFlip" in path:
            marker = "s"
            color = next(edgeflip_colors)
        else:
            marker = "o"
            color = next(normal_colors)

        ax.plot(
            bin_centers,
            bin_sums,
            marker=marker,
            linestyle="-",
            color=color,
            label=label,
            markerfacecolor="none",
        )

    if not ax.lines:
        print("No data found for plastic count plotting.")
        return None

    ax.set_xscale("log")
    ax.set_xlabel(rf"$-\Delta {maybe_avg('E')}$")
    ax.set_ylabel("Nr plastic events")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    if name:
        title = name
    elif info:
        title = make_title_from_data_info(info)
    else:
        title = "plastic_counts"
    ax.set_title(title)
    fig.tight_layout()

    if save:
        if filename is None:
            filename = f"{PLOTPATH}{safePath(title)}_plastic_counts.pdf"
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
    if show:
        plt.show()

    return ax


def plot_plastic_energy_scatter(
    paths,
    labels=None,
    postRegime=True,
    strainLim="auto",
    ax=None,
    show=False,
    save=True,
    filename=None,
    name=None,
    color_by_label=False,
):
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    if labels is None:
        labels = ["" for _ in paths]
    if len(labels) < len(paths):
        labels = labels + [""] * (len(paths) - len(labels))
    elif len(labels) > len(paths):
        labels = labels[: len(paths)]

    def _split_label(label):
        if label is None:
            return [], None
        tokens = [tok.strip() for tok in str(label).split(",") if tok.strip()]
        seed = None
        rest = []
        for t in tokens:
            if t.startswith("seed="):
                try:
                    seed = int(t.split("=", 1)[1])
                except ValueError:
                    seed = None
            elif t.startswith("s="):
                try:
                    seed = int(t.split("=", 1)[1])
                except ValueError:
                    seed = None
            else:
                rest.append(t)
        return rest, seed

    def _base_label(label):
        tokens, _ = _split_label(label)
        return ", ".join(tokens)

    # Build legend labels that collapse only-seed differences
    grouped_seeds = {}
    for label in labels:
        base = _base_label(label)
        if base not in grouped_seeds:
            grouped_seeds[base] = []
        _, seed = _split_label(label)
        if seed is not None:
            grouped_seeds[base].append(seed)

    legend_label_for_base = {}
    for base, seeds in grouped_seeds.items():
        if seeds:
            seeds = sorted(seeds)
            legend_label_for_base[base] = f"{base}, s={seeds[0]}-{seeds[-1]}"
        else:
            legend_label_for_base[base] = base

    used_legend_bases = set()
    used_model_labels_by_L = set()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.2))
    else:
        fig = ax.figure

    # For scaling model
    # mu, _ = ContiEnergy.moduli_at_F(np.eye(2))
    mu = 6.08

    label_color_map = {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_by_label:
        blues = mpl.colormaps["Blues"]
        oranges = mpl.colormaps["Oranges"]
        n_edge = sum("edgeFlip" in str(p) for p in paths)
        n_norm = len(paths) - n_edge
        edgeflip_colors = iter(blues(np.linspace(0.3, 0.9, max(1, n_edge))))
        normal_colors = iter(oranges(np.linspace(0.3, 0.9, max(1, n_norm))))
    for path, label in zip(paths, labels):
        base = _base_label(label)
        if base in used_legend_bases:
            plot_label = "_nolegend_"
        else:
            plot_label = legend_label_for_base.get(base, base)
            used_legend_bases.add(base)

        df = pd.read_csv(path)
        df = update_df_header(df, L=get_system_size([path]))
        drops, info = get_energy_drops(
            [path],
            df=df,
            strainLim=strainLim,
            debug=False,
            label=None,
            postRegime=postRegime,
        )

        mask = info.get("mask")
        assert mask is not None
        plastics = df["nr_elements_with_m3_fix_change"][mask].to_numpy()
        # idx = df["nr_elements_with_m3_fix_change"][mask].idxmax()
        # print(df.loc[idx])

        if plastics.size != drops.size:
            warnings.warn(
                "Plastic-event count length does not match drop count; skipping.",
                RuntimeWarning,
            )
            continue

        positive_mask = drops > 0
        if not np.any(positive_mask):
            continue
        drops_pos = drops[positive_mask]
        plastics = plastics[positive_mask]

        x_vals = plastics.astype(float) ** 2
        y_vals = drops_pos

        if color_by_label:
            marker = "o"
            if base not in label_color_map and default_cycle:
                label_color_map[base] = default_cycle[
                    len(label_color_map) % len(default_cycle)
                ]
            color = label_color_map.get(base, None)
        else:
            if "edgeFlip" in str(path):
                marker = "s"
                color = next(edgeflip_colors)
            else:
                marker = "o"
                color = next(normal_colors)

        ax.scatter(
            x_vals,
            y_vals,
            marker=marker,
            color=color,
            label=plot_label,
            alpha=0.6,
            s=14,
            edgecolors="none",
        )

        # Scaling model
        L = info.get("L", 1)

        Np = plastics[plastics > 0]
        x = np.array([Np.min(), Np.max()])
        b = 1
        y = mu * b**2 * x**2 / L**2
        if L in used_model_labels_by_L:
            model_label = "_nolegend_"
        else:
            print(f"Using mu: {mu:.2f}")
            print(f"Using L: {L}")
            model_label = rf"$\frac{{\mu b^2 N_p^2}}{{L^2}}, \mu={mu:.2f}$"
            used_model_labels_by_L.add(L)
        ax.plot(
            x**2,
            y,
            label=model_label,
        )

    if not ax.collections:
        print("No data found for plastic energy scatter plotting.")
        return None

    ax.set_xlabel(r"Nr plastic events$^2$, ($N_p^2$)")
    ax.set_ylabel(rf"$-\Delta {maybe_avg('E')}$")
    ax.legend(loc="best")
    ax.loglog()
    if name:
        title = name
    elif info:
        title = make_title_from_data_info(info)
    else:
        title = "plastic_energy_scatter"
    ax.set_title(title)
    fig.tight_layout()

    if save:
        if filename is None:
            filename = f"{PLOTPATH}{safePath(title)}_plastic_energy_scatter.png"
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)
    if show:
        plt.show()

    return ax


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
    xmin_range: tuple | list | float | None = None,
    distType: type[Distribution] = Truncated_Power_Law,
    use_cache=True,
    cache_dir: str = ".xmin_values",
    fast_xmin=False,
    xmin_accuracy=1.0,
    parallel_xmin=None,
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
    )
    if parallel_xmin is not None:
        fitObj.parallel_xmin = bool(parallel_xmin)
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
    fits, best_fit=None, savePath=None, title=None, xmin_results=None, fast_xmin=None
):
    """
    Plot KS p-value and exponent (with std error bars) versus xmin.
    """
    fits.sort(key=lambda f: f.xmin)
    tag = ks_tag(fits=fits, fast_xmin=fast_xmin)
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
    ax1.errorbar(
        x,
        pvals,
        yerr=p_stds,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label=f"{tag} p-value",
        color=c_p,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=2,
    )
    ax1.axhline(
        0.10,
        linestyle="--",
        linewidth=1.0,
        color=c_p,
        alpha=0.5,
        label="p = 0.10 threshold",
        zorder=1,
    )
    ax1.set_xlabel(r"$E_{\mathrm{min}}$")
    ax1.set_ylabel(f"{tag} p-value", color=c_p)
    ax1.tick_params(axis="y", colors=c_p)
    ax1.spines["left"].set_color(c_p)
    ax1.set_ylim(0, 1)

    ax3 = None
    if xmin_results is not None:
        r = xmin_results
        xmins = np.asarray(r["xmins"], dtype=float)
        distances = np.asarray(r["distances"], dtype=float)
        valid_fits = r.get("valid_fits", None)
        if valid_fits is not None:
            valid_fits = np.asarray(valid_fits, dtype=bool)
        mask = np.isfinite(distances)
        if valid_fits is not None:
            mask &= valid_fits
        max_xmin_filter = (xmins < np.nanmax(x)) & (np.isfinite(distances))
        if mask.any():
            # Filtered (optionally-valid) KS minimum
            x_d = xmins[mask]
            d = distances[mask]
            order = np.argsort(x_d)
            x_d = x_d[order]
            d = d[order]
            ks_xmin_filtered = float(x_d[np.argmin(d)])

            # Global KS minimum
            global_idx = np.argmin(distances[max_xmin_filter])
            ks_xmin_global = xmins[max_xmin_filter][global_idx]

            # KS distance curve (plotted over the range used for the p/alpha curves)
            ax1.plot(
                xmins[max_xmin_filter],
                distances[max_xmin_filter],
                linestyle="--",
                linewidth=1.2,
                color="0.5",
                alpha=0.7,
                label=f"{tag} distance",
                zorder=0,
            )

            # # Log-derivative of KS distance on a third axis
            # if np.any(max_xmin_filter):
            #     x_k = xmins[max_xmin_filter]
            #     d_k = distances[max_xmin_filter]
            #     order_k = np.argsort(x_k)
            #     x_k = x_k[order_k]
            #     d_k = d_k[order_k]
            #     if x_k.size >= 2:
            #         logx = np.log10(x_k)
            #         dD = np.gradient(d_k, logx)
            #         ax3 = ax1.twinx()
            #         ax3.spines["right"].set_position(("axes", 1.15))
            #         c_dd = "tab:green"
            #         ax3.plot(
            #             x_k,
            #             dD,
            #             marker="^",
            #             linestyle=":",
            #             linewidth=1.2,
            #             markersize=4,
            #             label=r"$dD/d\log_{10}(x_{\min})$",
            #             color=c_dd,
            #             markerfacecolor="none",
            #             markeredgecolor=c_dd,
            #             zorder=1,
            #         )
            #         ax3.set_ylabel(r"$dD/d\log_{10}(x_{\min})$", color=c_dd)
            #         ax3.tick_params(axis="y", colors=c_dd)
            #         ax3.spines["right"].set_color(c_dd)

            # Only draw both markers if they actually differ.
            if np.isclose(ks_xmin_filtered, ks_xmin_global, rtol=1e-12, atol=0.0):
                ax1.axvline(
                    ks_xmin_global,
                    color="0.5",
                    linestyle="-.",
                    linewidth=1,
                    label=f"Global {tag} xmin: {ks_xmin_global:.2e}",
                    zorder=-1,
                    alpha=0.7,
                )
            else:
                ax1.axvline(
                    ks_xmin_filtered,
                    color="0.5",
                    linestyle="-",
                    linewidth=1,
                    label=f"Filtered {tag} xmin: {ks_xmin_filtered:.2e}",
                    zorder=-1,
                    alpha=0.7,
                )
                ax1.axvline(
                    ks_xmin_global,
                    color="0.5",
                    linestyle="-.",
                    linewidth=1,
                    label=f"Global {tag} xmin: {ks_xmin_global:.2e}",
                    zorder=-1,
                    alpha=0.7,
                )
            if xmin_results.get("plateau_xmin", None):
                ax1.axvline(
                    xmin_results["plateau_xmin"],
                    color="0.5",
                    linestyle="-.",
                    linewidth=1,
                    label=f"Plateau xmin: {xmin_results['plateau_xmin']:.2e}",
                    zorder=-1,
                    alpha=0.7,
                )

    # --- Right axis: alpha (+ error bars) ---
    ax2 = ax1.twinx()
    ax2.errorbar(
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
        ax1.axvline(
            best_fit.xmin,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=rf"Best $p$-xmin: {best_fit.xmin:.2e}",
            zorder=-1,
            alpha=0.7,
        )

    # --- Legend: collect handles from both axes ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # handles3, labels3 = ([], [])
    # if ax3 is not None:
    #     handles3, labels3 = ax3.get_legend_handles_labels()

    # Deduplicate by label while preserving order
    seen = set()
    handles = []
    labels = []
    for h, l in (
        list(zip(handles1, labels1)) + list(zip(handles2, labels2))
        # + list(zip(handles3, labels3))
    ):
        if l not in seen and l != "":
            seen.add(l)
            handles.append(h)
            labels.append(l)

    ax2.legend(
        handles,
        labels,
        loc="upper right",
        ncol=2,
        frameon=True,
    )
    if ax3 is not None:
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()
    ax1.set_title(title)

    if savePath:
        fig.savefig(savePath, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {savePath}")
        setattr(fig, "path", savePath)
    else:
        setattr(fig, "path", None)
    plt.close(fig)
    return savePath


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

    tag = ks_tag(fit=fit)
    fig, ax1 = plt.subplots()
    c_d = "tab:blue"  # left axis (KS distance)
    c_a = "tab:orange"  # right axis (alpha)

    # Left axis: KS distance
    ax1.plot(
        x,
        D,
        marker="o",
        linestyle="-",
        label=f"{tag} distance (D)",
        color=c_d,
        markerfacecolor="none",
        markeredgecolor=c_d,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$x_{\min}$")
    ax1.set_ylabel(f"{tag} distance (D)")
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
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        ncol=2,
        frameon=True,
    )

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


def plot_KS_fitting(fit, save=True, show=False):
    """Plot KS distance and its derivative versus candidate xmin.

    Expects `fit.xmin_fitting_results` to be a dict with keys:
        distances, xmins, valid_fits

    Produces a plot of KS distance (D) and dD/dlog10(xmin) as a function of
    candidate xmin. NaNs are ignored. A vertical line is drawn at the chosen
    xmin (`fit.xmin`).
    """

    if not hasattr(fit, "xmin_fitting_results") or fit.xmin_fitting_results is None:
        print("No xmin_fitting_results found on fit object.")
        return None

    r = fit.xmin_fitting_results

    xmins = np.asarray(r["xmins"], dtype=float)
    distances = np.asarray(r["distances"], dtype=float)

    valid_fits = r.get("valid_fits", None)
    if valid_fits is not None:
        valid_fits = np.asarray(valid_fits, dtype=bool)

    mask = np.isfinite(distances) & np.isfinite(xmins)
    if valid_fits is not None:
        mask &= valid_fits
    mask &= xmins > 0

    if mask.sum() == 0:
        print("No valid xmin fitting points after filtering NaNs/invalid fits.")
        return None

    x = xmins[mask]
    D = distances[mask]

    order = np.argsort(x)
    x = x[order]
    D = D[order]

    logx = np.log10(x)
    dD = np.gradient(D, logx)

    tag = ks_tag(fit=fit)
    tag_lower = ks_tag(fit=fit, lower=True)
    fig, ax1 = plt.subplots()
    c_d = "tab:blue"
    c_dd = "tab:green"

    ax1.plot(
        x,
        D,
        marker="o",
        linestyle="-",
        label=f"{tag} distance (D)",
        color=c_d,
        markerfacecolor="none",
        markeredgecolor=c_d,
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$x_{\min}$")
    ax1.set_ylabel(f"{tag} distance (D)")
    ax1.tick_params(axis="y", colors=c_d)
    ax1.spines["left"].set_color(c_d)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        dD,
        marker="s",
        linestyle="--",
        label=r"$dD/d\log_{10}(x_{\min})$",
        color=c_dd,
        markerfacecolor="none",
        markeredgecolor=c_dd,
    )
    ax2.set_ylabel(r"$dD/d\log_{10}(x_{\min})$")
    ax2.tick_params(axis="y", colors=c_dd)
    ax2.spines["right"].set_color(c_dd)

    chosen_xmin = getattr(fit, "xmin", None)
    if chosen_xmin is not None and np.isfinite(chosen_xmin):
        ax1.axvline(
            chosen_xmin,
            linestyle=":",
            linewidth=1.5,
            label=rf"Chosen $x_{{\min}}$ = {chosen_xmin:.2e}",
            alpha=0.9,
        )

    try:
        dist_name = fit.xmin_distribution.name
    except Exception:
        dist_name = ""
    title = f"{tag} fitting"
    if dist_name:
        title += f" ({pretty_text(dist_name, addEquation=False)})"
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()
    if show:
        plt.show()

    if save:
        os.makedirs(PLOTPATH + "debug/", exist_ok=True)
        dist_tag = dist_name if dist_name else "dist"
        xmin_tag = f"{chosen_xmin:.2e}" if chosen_xmin is not None else "unknown"
        filename = (
            f"{PLOTPATH}debug/{tag_lower}_fitting_{dist_tag}_xmin_{xmin_tag}{OUTPUTTYPE}"
        )
        fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        plt.close(fig)
        return filename

    return fig, (ax1, ax2)


def get_group_structure(group_paths, group_labels):
    """
    Given a single path, a list of paths, or a list of lists of paths, it always
    returns a list of list of paths.
    Exampel:
    Groups:
        LBFSG:
            csv1
            csv2
        FIRE:
            csv1
    """
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
    return normalized_paths, normalized_labels


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
    useCDF=False,
):
    if group_paths is None and csvPaths is not None:
        group_paths = csvPaths

    grouped_paths, grouped_labels = get_group_structure(group_paths, group_labels)

    # We only deal with one group
    all_drops, data_info = get_energy_drops(
        grouped_paths[0],
        strainLim=strainLim,
        debug=debug,
        label=grouped_labels[0],
        postRegime=postRegime,
    )
    if all_drops is None or len(all_drops) == 0:
        print("No energy drops found; skipping powerlaw fit.")
        return None
    # find_xmin_rising_level(all_drops, debug=True)

    event_min_xmin = find_start_of_plastic_events(data_info, debug=True)

    if xmin_range is None and not fast_xmin:
        # Not using fast_xmin is brutally slow. We add a default range here
        xmin_range = [1e-9, 1]
    KS_fit = make_fit(
        data=all_drops,
        xmin_range=xmin_range,
        distType=distType,
        fast_xmin=fast_xmin,
        xmin_accuracy=xmin_accuracy,
    )
    if evaluate:
        KS_fit.evaluate_fit()

    p_fit = find_best_xmin(
        all_drops,
        debug=debug,
        data_info=data_info,
        xmin_results=getattr(KS_fit, "xmin_fitting_results", None),
        fast_xmin=fast_xmin,
    )

    d = dist_from_fit(p_fit)

    attribute = get_attribute(data_info["label"][0])
    if attribute == "Unknown":
        assert isinstance(data_info, dict)
        if "minimizer" in data_info:
            attribute = data_info["minimizer"]

    if attribute in MINIMIZER_COLORS:
        color = MINIMIZER_COLORS[attribute]
    else:
        color = "black"
    if attribute == "Unknown":
        attribute = ""

    if evaluate:
        p, mean_exp, exp_std = p_fit.evaluate_fit(all_drops, parallel=True)

        thresholds = [0.05, 0.1, 0.3, float("inf")]
        ratings = ["bad", "poor", "good", "excellent"]

        # Set r
        for t, r in zip(thresholds, ratings):
            if p < t:
                break

        print(f"Number of drops: {len(all_drops)}")
        print(f"{attribute}: P value: {p:.2f} ({r}), exp: {d.alpha}, std: {exp_std}")

    tag = ks_tag(fast_xmin=fast_xmin)
    if event_min_xmin:
        plot_ks_distance(
            all_drops,
            event_min_xmin,
            data_info=data_info,
            name="event-min-fit",
            fast_xmin=fast_xmin,
        )

    min_drop = np.min(all_drops)
    # We exclude the first two decades.
    exclude_factor = 100
    rmEnd_xmin = min_drop * exclude_factor
    plot_ks_distance(
        all_drops,
        rmEnd_xmin,
        data_info=data_info,
        name=f"rmEnd{exclude_factor}",
        fast_xmin=fast_xmin,
    )

    plot_ks_distance(
        all_drops, p_fit.xmin, data_info=data_info, name=r"$p$-fit", fast_xmin=fast_xmin
    )
    plot_ks_distance(
        all_drops,
        KS_fit.xmin,
        data_info=data_info,
        name=f"{tag}-fit",
        fast_xmin=fast_xmin,
    )

    title = make_title(data_info=data_info, fit=p_fit)
    if attribute and attribute not in title:
        title = attribute + " " + title
    plot_data_and_fit(
        p_fit,
        title=title,
        color=color,
        addFit=addFit,
        save=save,
        show=show,
        useCDF=useCDF,
    )
    title = make_title(data_info=data_info, fit=KS_fit)
    plot_data_and_fit(
        KS_fit,
        title=title + f"_{tag}_xmin",
        color=color,
        addFit=addFit,
        save=save,
        show=show,
        useCDF=useCDF,
    )

    # rmEnd_fit = Fit(all_drops, xmin=rmEnd_xmin, xmin_distribution=distType.name)
    # if evaluate:
    #     rmEnd_fit.evaluate_fit()

    # title = make_title(data_info=data_info, fit=rmEnd_fit)
    # plot_data_and_fit(
    #     rmEnd_fit,
    #     title=title + "_rmEnd_Emin",
    #     color=color,
    #     addFit=addFit,
    #     save=save,
    #     show=show,
    #     useCDF=useCDF,
    # )


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

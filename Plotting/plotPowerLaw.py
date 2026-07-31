from .findXmin import (
    analyze_xmin,
    annotate_xmin_choices,
    plot_xmin_analysis,
    xmin_global_differs,
)
from MTMath.energyFunction import ContiEnergy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from MTMath.evaluatePowerlawFit import Fit, Truncated_Power_Law
from powerlaw import Distribution
from Management.updateCSV import update_df_header, read_macrodata_csv
from .makePlots import safePath, maybe_avg, energy_drop_label
from .dataFunctions import get_metadata
from .energyDropCalculations import (
    extract_energy_drops_from_dataframe,
    infer_plastic_event_column,
    infer_stress_column,
)
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


def _make_json_serializable(value):
    if isinstance(value, np.ndarray):
        return [_make_json_serializable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(v) for v in value]
    return value


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


def get_lowest_distance_xmin(xmin_results):
    if not xmin_results:
        return None

    xmins = np.asarray(xmin_results.get("xmins", []), dtype=float)
    distances = np.asarray(xmin_results.get("distances", []), dtype=float)
    if xmins.size == 0 or distances.size == 0 or xmins.shape != distances.shape:
        return None

    mask = np.isfinite(xmins) & np.isfinite(distances) & (xmins > 0)
    valid_fits = xmin_results.get("valid_fits", None)
    if valid_fits is not None:
        valid_fits = np.asarray(valid_fits, dtype=bool)
        if valid_fits.shape == mask.shape:
            mask &= valid_fits
    if not np.any(mask):
        return None

    masked_indices = np.flatnonzero(mask)
    best_idx = masked_indices[int(np.argmin(distances[mask]))]
    return float(xmins[best_idx])


def _append_sample_suffix(name, n_samples):
    if n_samples is None:
        return name
    try:
        n_int = int(n_samples)
    except (TypeError, ValueError):
        return name
    return f"{name}_n{n_int}"


def strip_seed_from_label(label):
    if label is None:
        return ""
    tokens = [tok.strip() for tok in str(label).split(",") if tok.strip()]
    tokens = [token for token in tokens if not token.startswith("seed=")]
    return ", ".join(tokens)


def pretty_variant_label(label):
    if label is None:
        return ""

    tokens = [tok.strip() for tok in str(label).split(",") if tok.strip()]
    pretty_tokens = []
    for token in tokens:
        if token.startswith("seed=") or token.startswith("s="):
            continue
        if token.startswith("loadIncrement="):
            value = token.split("=", 1)[1]
            pretty_tokens.append(rf"$\delta \gamma$: {value}")
            continue
        if token.startswith("reconnectionMethod="):
            value = token.split("=", 1)[1]
            pretty_tokens.append(f"re.met: {value}")
            continue
        pretty_tokens.append(token)

    return ", ".join(pretty_tokens)


def fit_equation_label(dist_name):
    if dist_name == "truncated_power_law":
        return r"Fit: $p(x) = x^{-\alpha} e^{-\lambda x}$"
    if dist_name == "power_law":
        return r"Fit: $p(x) = x^{-\alpha}$"
    return f"Fit: {pretty_text(dist_name, addEquation=False)}"


def fit_parameter_label(dist):
    parameter_names = list(getattr(dist, "parameter_names", []))
    if not parameter_names:
        return ""

    symbol_map = {
        "alpha": r"\alpha",
        "Lambda": r"\lambda",
        "mu": r"\mu",
        "sigma": r"\sigma",
        "beta": r"\beta",
    }
    fixed_point_params = {"alpha", "mu", "sigma", "beta"}
    parts = []
    for name in parameter_names:
        value = getattr(dist, name, np.nan)
        if not np.isfinite(value):
            continue
        symbol = symbol_map.get(name, name)
        if name in fixed_point_params:
            parts.append(rf"{symbol}={value:.2f}")
        else:
            parts.append(rf"{symbol}={value:.2e}")

    if not parts:
        return ""
    return "$" + ", ".join(parts) + "$"


def _count_fit_drops(fit):
    data = _clean_positive_data(fit.data_original)
    return int(np.count_nonzero(data >= fit.xmin))


def _format_drop_count(nr_drops):
    if nr_drops < 1e4:
        return str(int(nr_drops))
    return f"{float(nr_drops):.1e}"


def compare_legend_label(label, fit, nr_samples=None, nr_drops=None):
    dist = dist_from_fit(fit)
    variant_label = pretty_variant_label(label)
    params = fit_parameter_label(dist)
    stats = []
    if nr_samples is not None:
        stats.append(f"S:{int(nr_samples)}")
    if nr_drops is not None:
        stats.append(f"D:{float(nr_drops):.1e}")
    stats_label = ", ".join(stats)

    parts = [part for part in [variant_label, stats_label, params] if part]
    if parts:
        return "; ".join(parts)
    return pretty_text(dist.name, addEquation=False)


def _make_fit_x_grid(data, xmin=None, num=256):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        return np.asarray([], dtype=float)

    x_max = float(np.max(data))
    x_min = float(np.min(data)) if xmin is None else float(xmin)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min <= 0 or x_max <= 0:
        return np.asarray([], dtype=float)
    if x_max < x_min:
        return np.asarray([], dtype=float)
    if np.isclose(x_min, x_max):
        return np.asarray([x_min], dtype=float)
    return np.logspace(np.log10(x_min), np.log10(x_max), num=num)


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
        if len(df["LBFGS_Term_reason"]) < 3:
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
    sizes = set()
    for path in csvPaths:
        meta = get_metadata(path)
        L = meta.get("L")
        if L is None:
            dims = meta.get("dims") or meta.get("N")
            if dims:
                n1, n2 = dims
                if n1 != n2:
                    return -1
                L = n1
        if L is None:
            print("Not able to find system size")
            continue
        sizes.add(int(L))
    if len(sizes) > 1:
        print("More than one size!")
        return -1
    if not sizes:
        raise ValueError("Could not infer system size from any CSV path.")
    return sizes.pop()


def _resolve_drop_label(info=None, *, energy_type=None, stress_corrected=False, averageEnergy=None):
    if info is not None and "drop_label" in info:
        return info["drop_label"]
    return energy_drop_label(
        energy_type=energy_type,
        stress_corrected=stress_corrected,
        use_avg=averageEnergy,
    )


def _stress_corrected_drop_label(use_piola=False, use_avg=None):
    # The numerical correction uses averaged Cauchy shear stress for MTS2D's
    # left-multiplicative affine loading. Keep ``use_piola`` only for API
    # compatibility with older callers.
    return maybe_avg("E_S", use_avg)


def get_elastic_mu(report=False):
    mu = ContiEnergy.moduli_at_F(np.eye(2)).mu
    mu = float(np.asarray(mu, dtype=float).reshape(-1)[0])
    if report:
        print(f"Using mu: {mu:.2f}")
    return mu

def get_mu(df):
    sigma_col = infer_stress_column(df)
    sigma = df[sigma_col]
    delta_gamma = np.diff(df["load"])
    mu = np.diff(sigma)/delta_gamma
    return mu

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
    stress_corrected=True,
    stress_correction_order=2,
    stress_tangent="current",
    drop_sign="negative",
    min_drop=0.0,
):
    """
    Strain energy drop data from CSV, filter by strain limits, and return drops.
    When ``onlyStrainedEnergyDrops`` is true, retain only steps with a recorded
    element-level plastic event.
    Stress-corrected drops use the shared Taylor expansion helper; by default
    this is second order with a1212 evaluated at the current strain gamma_i.
    If debug=True, plot intermediate energy and drop traces.

    For stored energy-change columns, ``drop_sign`` explicitly specifies
    whether drops are negative or positive. Taylor-corrected drops always use
    their defined positive-drop convention.
    """
    if isinstance(csvPaths, str):
        csvPaths = [csvPaths]
    if not np.isfinite(min_drop) or min_drop < 0:
        raise ValueError("min_drop must be finite and nonnegative.")

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
    energy_type_key = energy_type
    if isinstance(energy_type, str) and energy_type.lower() in {
        "inter-strain",
        "inter_strain",
        "interstrain",
    }:
        energy_type_key = "energy_change"
    energy_key = prefix + energy_type_key
    drop_label = None
    used_piola_stress = False
    use_average_label = averageEnergy

    for singlePath in csvPaths:
        resolved_strain_lim = strainLim
        if read_from_paths:
            df_local = read_macrodata_csv(singlePath, L=L)
            dfs.append(df_local)
        else:
            df_local = df
        if resolved_strain_lim in ("auto", "all", None):
            resolved_strain_lim = _resolve_strain_lim(
                resolved_strain_lim, df=df_local, postRegime=postRegime
            )

        extracted, mask, signed_step_change, step_info = extract_energy_drops_from_dataframe(
            df_local,
            csv_file_path=singlePath,
            metadata=get_metadata(singlePath),
            strain_lim=resolved_strain_lim,
            energy_key=energy_key,
            average_energy=averageEnergy,
            stress_corrected=stress_corrected,
            correction_order=stress_correction_order,
            tangent=stress_tangent,
            drop_sign=drop_sign,
            min_drop=min_drop,
            plastic_only=onlyStrainedEnergyDrops,
        )
        strain = df_local["load"]
        lim_mask = (strain > resolved_strain_lim[0]) & (strain < resolved_strain_lim[1])
        energy_col = step_info["energy_col"]
        if stress_corrected:
            used_piola_stress |= step_info["used_piola_stress"]
            use_average = step_info["converted_avg_energy_to_total"]
            if use_average_label is None:
                use_average_label = use_average
            correction_name = (
                "first_order"
                if stress_correction_order == 1
                else f"second_order_{stress_tangent}"
            )
            energy_key = (
                f"avg_stress_corrected_energy_drop_{correction_name}"
                if use_average
                else f"total_stress_corrected_energy_drop_{correction_name}"
            )
        elif drop_label is None:
            drop_label = energy_drop_label(
                energy_type=energy_type,
                stress_corrected=False,
                use_avg=averageEnergy,
            )

        drops.extend(extracted)
        masks.append(mask)

    drops = np.array(drops)
    concat_df = pd.concat(dfs, ignore_index=True) if read_from_paths else df

    if drop_label is None:
        if stress_corrected:
            drop_label = _stress_corrected_drop_label(
                use_piola=used_piola_stress,
                use_avg=use_average_label,
            )
        else:
            drop_label = energy_drop_label(
                energy_type=energy_type,
                stress_corrected=stress_corrected,
                use_avg=averageEnergy,
            )

    data_info = get_energy_drops_info(
        csvPaths=csvPaths,
        drops=drops,
        energy_key=energy_key,
        df=concat_df,
        strainLim=resolved_strain_lim,
        label=label,
        masks=masks,
        drop_label=drop_label,
    )

    if debug:
        # Only debug first seed when using labels
        label_text = ", ".join(map(str, label)) if isinstance(label, list) else str(label)
        if label is not None and "seed=" in label_text and "seed=0" not in label_text:
            return drops, data_info

        strain_limited = strain[1:][lim_mask[1:]]
        plotDrops = np.clip(-signed_step_change[1:][lim_mask[1:]], 0, np.inf)
        if stress_corrected:
            e = df_local[energy_col]
        else:
            e = df_local[energy_key]
        debug_fig, ax1 = plt.subplots()
        avg_label = maybe_avg("E", averageEnergy)
        ax1.plot(strain, e, label=rf"${avg_label}$")
        ax1.set_ylabel(rf"${avg_label}$")
        ax1.set_xlabel(r"$\gamma$")
        ax2 = ax1.twinx()
        ax2.plot([])  # advance color cycle
        drop_label_local = drop_label or energy_drop_label(
            energy_type=energy_type,
            stress_corrected=stress_corrected,
            use_avg=averageEnergy,
        )
        label = rf"$-\Delta {drop_label_local}$"
        ax2.plot(strain_limited, plotDrops, label=label)
        ax2.set_ylabel(rf"$-\Delta {drop_label_local}$ (Energy Drop)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines + lines2,
            labels + labels2,
            loc="upper right",
            ncol=2,
            frameon=True,
        )
        finite_drops = drops[np.isfinite(drops)]
        if finite_drops.size == 0:
            ax2.set_ylim(0, 1)
        else:
            ax2.set_ylim(0, finite_drops.max() * 1.5)

        # ——— Compute 0.1%‐wide central slice ———
        mid = 0.5 * (resolved_strain_lim[0] + resolved_strain_lim[1])
        total_width = resolved_strain_lim[1] - resolved_strain_lim[0]
        slice_width = total_width * 0.05  # 1% of window
        x1, x2 = mid - slice_width / 2, mid + slice_width / 2
        zoom_mask = (strain >= x1) & (strain <= x2)

        # find energy‐axis extents in that slice
        zoom_values = -signed_step_change[zoom_mask]
        if zoom_values.size == 0:
            raise ValueError("Debug zoom window contains no load points.")
        y1, y2 = zoom_values.min(), zoom_values.max()
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
        finite_zoom = drops_zoom[np.isfinite(drops_zoom)]
        if finite_zoom.size == 0:
            axins2.set_ylim(0, 1)
        else:
            axins2.set_ylim(0, finite_zoom.max() * 1.5)

        name = safePath(make_title(data_info))

        filename = f"{PLOTPATH}debug/{name}_energy_drops_strain{OUTPUTTYPE}"
        debug_fig.tight_layout(rect=[0, 0, 1, 0.9])
        debug_fig.savefig(filename, dpi=300)
        print(f"Saved figure to {filename}")
        # to save memory, close the figure
        plt.close(debug_fig)

    return drops, data_info

def get_stress_drops(
    csvPaths,
    df=None,
    strainLim: str | list[float] = "auto",
    label=None,
    postRegime=True,
):
    """
    Stress drops defined as
        Δσ = σ_n − σ_{n+1} + μ δγ
    keeping only positive drops inside the requested strain window.
    """
    if isinstance(csvPaths, str):
        csvPaths = [csvPaths]

    drops = []
    masks = []
    dfs = []
    L = get_system_size(csvPaths)
    read_from_paths = df is None
    mu = get_elastic_mu(report=False)

    for singlePath in csvPaths:
        resolved_strain_lim = strainLim
        if read_from_paths:
            df_local = read_macrodata_csv(singlePath, L=L)
            dfs.append(df_local)
        else:
            df_local = df

        if resolved_strain_lim in ("auto", "all", None):
            resolved_strain_lim = _resolve_strain_lim(
                strainLim, df=df_local, postRegime=postRegime
            )

        sigma_col = infer_stress_column(df_local)
        sigma_arr = np.asarray(df_local[sigma_col], dtype=float)
        load_arr = np.asarray(df_local["load"], dtype=float)
        delta_gamma_arr = np.diff(load_arr)
        if delta_gamma_arr.ndim == 0:
            delta_gamma_arr = np.full(len(df_local) - 1, float(delta_gamma_arr))
        elif delta_gamma_arr.ndim != 1 or delta_gamma_arr.size != len(df_local) - 1:
            raise ValueError(
                f"loadIncrement for {singlePath} must be scalar or have "
                f"length len(df)-1={len(df_local) - 1}, got shape {delta_gamma_arr.shape}."
            )

        stress_drop = sigma_arr[:-1] - sigma_arr[1:] + mu * delta_gamma_arr
        strain = load_arr[1:]
        mask = (
            (stress_drop > 0)
            & (strain > resolved_strain_lim[0])
            & (strain < resolved_strain_lim[1])
        )

        drops.extend(stress_drop[mask])
        full_mask = np.zeros(len(df_local), dtype=bool)
        full_mask[1:] = mask
        masks.append(full_mask)

    drops = np.asarray(drops, dtype=float)
    concat_df = pd.concat(dfs, ignore_index=True) if read_from_paths else df
    data_info = get_energy_drops_info(
        csvPaths=csvPaths,
        drops=drops,
        energy_key="stress_drop",
        df=concat_df,
        strainLim=resolved_strain_lim,
        label=label,
        masks=masks,
        drop_label=r"\sigma",
    )
    data_info["mu"] = float(mu)
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
    drop_label=None,
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
    if drop_label is not None:
        info["drop_label"] = drop_label
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


def _clean_positive_data(data):
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    data = data[data > 0]
    if data.size == 0:
        return data
    return np.sort(data)


def _drop_quantity_label(drop_label):
    return rf"-\Delta {drop_label}"


def _drop_quantity_name(drop_label):
    if drop_label == r"\sigma":
        return "Stress Drop"
    return "Energy Drop"


def _set_distribution_axes(
    ax,
    *,
    drop_label=None,
    y_mode="pdf",
    title=None,
    set_title=False,
    show_legend=True,
):
    if drop_label is None:
        drop_label = maybe_avg("E")
    quantity_label = _drop_quantity_label(drop_label)
    quantity_name = _drop_quantity_name(drop_label)

    ax.set_xscale("log")
    ax.set_xlabel(rf"${quantity_label}$ ({quantity_name})")

    if y_mode == "pdf":
        ax.set_yscale("log")
        ax.set_ylabel(rf"$p({quantity_label})$")
    elif y_mode == "ccdf":
        ax.set_yscale("log")
        ax.set_ylabel(r"$P(X > x)$")
    elif y_mode == "cdf":
        ax.set_ylabel(r"$P(X \leq x)$")
    else:
        raise ValueError(f"Unknown y_mode: {y_mode}")

    if set_title:
        ax.set_title("" if title is None else title)
    if show_legend:
        ax.legend()
    return ax


def _get_fit_curve_data(fit: Fit, *, use_ccdf=True, x_grid_mode="data", xmin_only=False):
    dist = dist_from_fit(fit)

    data = _clean_positive_data(fit.data_original)
    if data.size == 0:
        return dist, np.array([]), np.array([])

    tail_frac = float((data >= fit.xmin).mean())

    if x_grid_mode == "data":
        x_vals = np.unique(data)
    elif x_grid_mode == "smooth":
        x_vals = _make_fit_x_grid(data, xmin=fit.xmin if xmin_only else None)
    else:
        raise ValueError(f"Unknown x_grid_mode: {x_grid_mode}")

    x_vals = x_vals[x_vals > 0]
    if xmin_only:
        x_vals = x_vals[x_vals >= fit.xmin]
    if x_vals.size == 0:
        return dist, np.array([]), np.array([])

    if use_ccdf is None:
        f = dist._pdf_base_function(x_vals)
        C = dist._pdf_continuous_normalizer
        y_vals = f * C * tail_frac
    else:
        model_ccdf = dist.ccdf(x_vals)
        if use_ccdf:
            y_vals = model_ccdf * tail_frac
        else:
            y_vals = 1.0 - model_ccdf * tail_frac

    return dist, x_vals, y_vals


def plot_data_pdf(
    ax,
    data,
    label=None,
    edgecolor="black",
    alpha=1,
    color=None,
    drop_label=None,
    show_legend=True,
):
    """
    Plot the empirical PDF of the data on log–log axes using logarithmic bins.
    Automatically identifies x_min via find_x_min and uses 0.1-decade bin widths.
    If `fit` is provided and `data` is None, use `fit.data_original`.
    """

    data = _clean_positive_data(data)
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

    _set_distribution_axes(
        ax,
        drop_label=drop_label,
        y_mode="pdf",
        show_legend=show_legend,
    )


def plot_data_cdf(
    ax,
    data,
    label=None,
    color=None,
    alpha=1,
    use_ccdf=True,
    drop_label=None,
    show_legend=True,
):
    data = _clean_positive_data(data)
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

    _set_distribution_axes(
        ax,
        drop_label=drop_label,
        y_mode="ccdf" if use_ccdf else "cdf",
        show_legend=show_legend,
    )


def plot_energy_drop_trace(
    ax,
    strain,
    energy,
    drop_strain,
    drops,
    *,
    energy_label=r"$E$",
    drop_label=r"$s$",
    color_energy=None,
    color_drop=None,
    min_drop=0.0,
    log_drop_axis=False,
    drop_marker=None,
    drop_linestyle="-",
    drop_linewidth=0.8,
    zoom_center=None,
    zoom_width=None,
    inset_bounds=(0.49, 0.08, 0.47, 0.43),
    inset_background_alpha=0.9,
    inset_show_x_ticks=True,
    inset_show_y_ticks=True,
    title=None,
    set_title=False,
    show_legend=True,
):
    """Plot an energy--strain trace with energy-drop magnitudes and an inset.

    This is the reusable form of the debug visualization used by
    :func:`get_energy_drops`.  The input arrays are deliberately decoupled from
    CSV loading so the plot can be embedded in composite figures without
    generating a standalone file or a simulation-derived title.

    Returns
    -------
    tuple
        ``(drop_axis, inset_energy_axis, inset_drop_axis)``.
    """

    strain = np.asarray(strain, dtype=float)
    energy = np.asarray(energy, dtype=float)
    drop_strain = np.asarray(drop_strain, dtype=float)
    drops = np.asarray(drops, dtype=float)
    if strain.ndim != 1 or energy.ndim != 1 or strain.shape != energy.shape:
        raise ValueError("strain and energy must be matching one-dimensional arrays.")
    if drop_strain.ndim != 1 or drops.ndim != 1 or drop_strain.shape != drops.shape:
        raise ValueError("drop_strain and drops must be matching one-dimensional arrays.")

    finite_energy = np.isfinite(strain) & np.isfinite(energy)
    min_drop = float(min_drop)
    plot_drops = np.clip(drops, 0.0, np.inf)
    finite_drop_values = np.isfinite(drop_strain) & np.isfinite(plot_drops)
    if min_drop == 0.0:
        finite_drops = finite_drop_values & (plot_drops >= min_drop)
    else:
        finite_drops = finite_drop_values & (plot_drops > min_drop)
    positive_drops = (
        np.isfinite(drop_strain)
        & np.isfinite(plot_drops)
        & (plot_drops > min_drop)
    )
    if not np.any(finite_energy):
        raise ValueError("No finite energy--strain values to plot.")
    if not np.any(positive_drops):
        raise ValueError("No finite positive drops above min_drop to plot.")

    energy_line = ax.plot(
        strain[finite_energy],
        energy[finite_energy],
        color=color_energy,
        linewidth=1.0,
        label=energy_label,
    )[0]
    if color_energy is None:
        color_energy = energy_line.get_color()
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(energy_label)

    drop_ax = ax.twinx()
    if color_drop is None:
        color_drop = "C1"
    drop_line = drop_ax.plot(
        drop_strain[finite_drops],
        plot_drops[finite_drops],
        marker=drop_marker,
        markersize=2.0,
        linestyle=drop_linestyle,
        linewidth=drop_linewidth,
        alpha=0.85,
        color=color_drop,
        label=drop_label,
    )[0]
    drop_ax.set_ylabel(drop_label)
    if log_drop_axis:
        drop_ax.set_yscale("log")
    else:
        drop_ax.set_ylim(0.0, 1.5 * float(np.nanmax(plot_drops[positive_drops])))

    finite_drop_strain = drop_strain[finite_drops]
    retained_drop_values = plot_drops[finite_drops]
    if zoom_center is None:
        zoom_center = float(finite_drop_strain[np.argmax(retained_drop_values)])
    else:
        zoom_center = float(zoom_center)

    finite_strain = strain[finite_energy]
    if zoom_width is None:
        unique_strain = np.unique(finite_strain)
        positive_steps = np.diff(unique_strain)
        positive_steps = positive_steps[positive_steps > 0]
        median_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
        zoom_width = max(80.0 * median_step, 0.002 * np.ptp(finite_strain))
    zoom_width = float(zoom_width)
    if not np.isfinite(zoom_width) or zoom_width <= 0.0:
        raise ValueError("zoom_width must be finite and positive.")

    x_lo = zoom_center - 0.5 * zoom_width
    x_hi = zoom_center + 0.5 * zoom_width
    zoom_energy = finite_energy & (strain >= x_lo) & (strain <= x_hi)
    if np.count_nonzero(zoom_energy) < 3:
        nearest = np.argsort(np.abs(strain - zoom_center))[: min(20, strain.size)]
        x_lo = float(np.min(strain[nearest]))
        x_hi = float(np.max(strain[nearest]))
        zoom_energy = finite_energy & (strain >= x_lo) & (strain <= x_hi)

    inset_ax = ax.inset_axes(inset_bounds)
    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(float(inset_background_alpha))
    inset_ax.plot(
        strain[zoom_energy],
        energy[zoom_energy],
        color=color_energy,
        linewidth=0.8,
    )
    inset_ax.set_xlim(x_lo, x_hi)
    inset_ax.tick_params(labelsize=6, pad=1)
    if not inset_show_x_ticks:
        inset_ax.tick_params(axis="x", labelbottom=False)
    if not inset_show_y_ticks:
        inset_ax.tick_params(axis="y", labelleft=False)

    inset_drop_ax = inset_ax.twinx()
    zoom_drops = finite_drops & (drop_strain >= x_lo) & (drop_strain <= x_hi)
    if np.any(zoom_drops):
        inset_drop_ax.plot(
            drop_strain[zoom_drops],
            plot_drops[zoom_drops],
            marker=drop_marker,
            markersize=1.8,
            linestyle=drop_linestyle,
            linewidth=drop_linewidth,
            color=color_drop,
            alpha=0.9,
        )
        if log_drop_axis:
            inset_drop_ax.set_yscale("log")
        else:
            zoom_max = float(np.nanmax(plot_drops[zoom_drops]))
            inset_drop_ax.set_ylim(0.0, 1.5 * zoom_max if zoom_max > 0.0 else 1.0)
    inset_drop_ax.tick_params(labelsize=6, pad=1)
    if not inset_show_y_ticks:
        inset_drop_ax.tick_params(axis="y", labelright=False)

    zoom_energy_values = energy[zoom_energy]
    if zoom_energy_values.size:
        y_lo = float(np.nanmin(zoom_energy_values))
        y_hi = float(np.nanmax(zoom_energy_values))
        y_pad = max(0.08 * (y_hi - y_lo), np.finfo(float).eps)
        rect = Rectangle(
            (x_lo, y_lo - y_pad),
            x_hi - x_lo,
            (y_hi - y_lo) + 2.0 * y_pad,
            linewidth=0.9,
            edgecolor="0.25",
            linestyle="--",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

    if set_title:
        ax.set_title("" if title is None else title)
    if show_legend:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = drop_ax.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

    return drop_ax, inset_ax, inset_drop_ax


def plot_ks_distance_marker(
    ax,
    sorted_data,
    ecdf,
    model_ccdf,
    color="red",
    empirical_color="blue",
    model_color="gray",
    fast_xmin=None,
    ecdf_before=None,
):
    sorted_data = np.asarray(sorted_data, dtype=float)
    ecdf = np.asarray(ecdf, dtype=float)
    model_ccdf = np.asarray(model_ccdf, dtype=float)
    if ecdf.size == 0:
        raise ValueError("Cannot calculate a KS distance from an empty CCDF.")
    if ecdf_before is None:
        ecdf_before = np.concatenate(([1.0], ecdf[:-1]))
    else:
        ecdf_before = np.asarray(ecdf_before, dtype=float)
    if (
        sorted_data.shape != ecdf.shape
        or ecdf.shape != model_ccdf.shape
        or ecdf_before.shape != ecdf.shape
    ):
        raise ValueError("Empirical and model CCDF arrays must have matching shapes.")

    # The empirical CCDF jumps at every observed value.  The KS supremum can
    # occur on either side of a jump, so evaluate both the pre- and post-jump
    # values rather than sampling only the plotted post-jump curve.
    post_jump_diffs = np.abs(ecdf - model_ccdf)
    pre_jump_diffs = np.abs(ecdf_before - model_ccdf)
    if np.max(pre_jump_diffs) > np.max(post_jump_diffs):
        max_index = int(np.argmax(pre_jump_diffs))
        empirical_at_D = ecdf_before[max_index]
        D_val = pre_jump_diffs[max_index]
    else:
        max_index = int(np.argmax(post_jump_diffs))
        empirical_at_D = ecdf[max_index]
        D_val = post_jump_diffs[max_index]
    x_D = sorted_data[max_index]
    tag = ks_tag(fast_xmin=fast_xmin)
    ax.vlines(
        x_D,
        model_ccdf[max_index],
        empirical_at_D,
        color=color,
        linestyle="--",
        label=rf"{tag} Distance $D={D_val:.3f}$",
    )
    ax.scatter([x_D], [empirical_at_D], color=empirical_color)
    ax.scatter([x_D], [model_ccdf[max_index]], color=model_color)
    return D_val


# --- Helper for annotating KS distance on PDF plot ---
def annotate_ks_distance_pdf(ax, xmin, D_val, color="red", fast_xmin=None, fit=None):
    tag = ks_tag(fast_xmin=fast_xmin, fit=fit)
    ax.axvline(
        xmin,
        color=color,
        linestyle="--",
        linewidth=1.2,
        label=rf"{tag} Distance $D={D_val:.3f}$",
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
    drop_label=None,
    label=None,
    show_legend=True,
    set_title=True,
    x_grid_mode="data",
    xmin_only=True,
    linewidth=None,
):
    dist, bins_for_model, likelihoods = _get_fit_curve_data(
        fit,
        use_ccdf=None,
        x_grid_mode=x_grid_mode,
        xmin_only=xmin_only,
    )
    if bins_for_model.size == 0:
        return ax

    if label is None:
        label = (pre_label or "") + pretty_text(dist.name)

    plot_kwargs = {
        "label": label,
        "color": color,
        "alpha": alpha,
        "linestyle": linestyle,
    }
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth

    ax.plot(
        bins_for_model,
        likelihoods,
        **plot_kwargs,
    )
    if add_ks_marker:
        annotate_ks_distance_pdf(
            ax, fit.xmin_fitting_results["xmins"][dist.D_i], dist.D, fit=fit
        )

    _set_distribution_axes(
        ax,
        drop_label=drop_label,
        y_mode="pdf",
        title=title,
        set_title=set_title,
        show_legend=show_legend,
    )
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
    drop_label=None,
    label=None,
    show_legend=True,
    set_title=True,
    x_grid_mode="data",
    xmin_only=True,
    linewidth=None,
):
    dist, bins_for_model, y_vals = _get_fit_curve_data(
        fit,
        use_ccdf=use_ccdf,
        x_grid_mode=x_grid_mode,
        xmin_only=xmin_only,
    )
    if bins_for_model.size == 0:
        return ax

    if label is None:
        label = (pre_label or "") + pretty_text(dist.name)

    plot_kwargs = {
        "label": label,
        "color": color,
        "alpha": alpha,
        "linestyle": linestyle,
    }
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth

    ax.plot(
        bins_for_model,
        y_vals,
        **plot_kwargs,
    )

    _set_distribution_axes(
        ax,
        drop_label=drop_label,
        y_mode="ccdf" if use_ccdf else "cdf",
        title=title,
        set_title=set_title,
        show_legend=show_legend,
    )
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
    # Prefer Cauchy shear stress, matching the spatial affine loading path.
    if df is None:
        df = read_macrodata_csv(csvPath, L=get_system_size([csvPath]))
    if "avg_sigma12" in df:
        i = df["avg_sigma12"].argmax()
    elif "avg_P12" in df:
        i = df["avg_P12"].argmax()
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
        df = read_macrodata_csv(csvPath, L=get_system_size([csvPath]))

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
    title = (
        rf"{L}x{L} {samples_string} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f} "
    )
    if data_info["minimizer"] != "Unknown":
        title = rf"{data_info['minimizer']} " + title
    if "label" in data_info:
        l = data_info["label"]
        if isinstance(l, list):
            normalized = [strip_seed_from_label(item) for item in l]
            normalized_non_empty = [item for item in normalized if item]
            if normalized_non_empty and len(set(normalized_non_empty)) == 1:
                if normalized_non_empty[0] != f"L={L}":
                    title += normalized_non_empty[0]
            else:
                if(not len(set(normalized)) == 1):
                    print(f"Labels in group are different: {set(normalized)}")
                title += normalized[0]
        else:
            title += strip_seed_from_label(l)
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


def make_compare_title_from_data_info(data_info):
    if "customTitle" in data_info:
        return data_info["customTitle"]
    strainLim = data_info["strainLim"]
    L = data_info["L"]
    title = rf"{L}x{L} $\gamma$: {strainLim[0]:.2f} - {strainLim[1]:.2f}"
    if data_info["minimizer"] != "Unknown":
        title = rf"{data_info['minimizer']} " + title
    return title.strip().replace("  ", " ")


def plot_data_and_fit(
    fit: Fit,
    ax=None,
    title="",
    data_info=None,
    color=None,
    data_color=None,
    addFit=True,
    useCDF=False,
    useCCDF=True,
    save=True,
    extraPath="",
    show=False,
    close=True,
    show_fit_region=True,
    show_cutoff=True,
    show_title=True,
    show_legend=True,
    xmin_analysis=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    drop_label = data_info.get("drop_label") if data_info else None
    fit_drop_count = _count_fit_drops(fit)
    fit_drop_count_label = _format_drop_count(fit_drop_count)
    if useCDF:
        dist_label = "CCDF" if useCCDF else "CDF"
        plot_data_cdf(
            ax,
            fit.data_original,
            label=f"{dist_label} of {fit_drop_count_label} drops in fit",
            color=data_color,
            use_ccdf=useCCDF,
            drop_label=drop_label,
            show_legend=False,
        )
    else:
        plot_data_pdf(
            ax,
            fit.data_original,
            label=f"PDF of {fit_drop_count_label} drops in fit",
            color=data_color,
            drop_label=drop_label,
            show_legend=False,
        )

    # plot the fit
    if addFit:
        if useCDF:
            plot_fit_cdf(
                ax,
                fit,
                color=color,
                use_ccdf=useCCDF,
                drop_label=drop_label,
                set_title=False,
                show_legend=False,
            )
        else:
            plot_fit_pdf(
                ax,
                fit,
                color=color,
                drop_label=drop_label,
                set_title=False,
                show_legend=False,
            )

        # Add shaded fit region with formula in label
        if fit.xmax is None:
            xmax = max(fit.data_original)
        else:
            xmax = fit.xmax
        dist = dist_from_fit(fit)
        if show_fit_region:
            ax.axvspan(
                fit.xmin,
                xmax,
                color="gray",
                alpha=0.2,
                label=(
                    rf"Fit region. $\hat{{\alpha}}={dist.alpha:.2f}, "
                    rf"\hat{{\lambda}}={dist.Lambda:.2e}$"
                ),
            )

        # Mark x = 1/lambda with a dashed vertical line through the full plot height.
        lambda_val = float(getattr(dist, "Lambda", np.nan))
        if show_cutoff and np.isfinite(lambda_val) and lambda_val > 0.0:
            x_inv_lambda = 1.0 / lambda_val
            ax.axvline(
                x_inv_lambda,
                color="tab:green",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label=r"$1/\lambda$",
            )

    if xmin_analysis is None:
        xmin_analysis = getattr(fit, "xmin_analysis", None)
    if xmin_analysis is not None:
        annotate_xmin_choices(ax, xmin_analysis)

    if show_legend:
        ax.legend()
    if show_title:
        if title == "" and data_info is not None:
            title = make_title(data_info)
        ax.set_title(title)
    else:
        ax.set_title("")

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
    set_title=True,
    show_legend=True,
    empirical_color="blue",
    model_color="gray",
    ks_color="red",
    show_inset=False,
    inset_bounds=(0.57, 0.33, 0.39, 0.34),
    inset_x_factor=1.15,
    inset_background_alpha=0.92,
    inset_grid=False,
    legend_usetex=False,
    tight_layout=True,
):
    """
    Plot the empirical CCDF vs the fitted CCDF and visually show the KS distance (D).
    """
    # Fit the distribution
    fitObj = Fit(drops, xmin=xmin, xmin_distribution=dist_name)
    dist = getattr(fitObj, dist_name)
    # Get the ECDF and model CCDF
    data = fitObj.data
    sorted_tail = np.sort(data[data >= xmin])
    sorted_data, counts = np.unique(sorted_tail, return_counts=True)
    cumulative_counts = np.cumsum(counts)
    n_tail = len(sorted_tail)
    ecdf = 1.0 - cumulative_counts / n_tail
    ecdf_before = 1.0 - (cumulative_counts - counts) / n_tail

    model_ccdf = dist.ccdf(sorted_data)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.step(
        sorted_data,
        ecdf,
        where="post",
        label=r"$\widehat{P}_{>}(x)$",
        color=empirical_color,
    )
    ax.plot(
        sorted_data,
        model_ccdf,
        label=(
            rf"$P_>^{{\mathrm{{TPL}}}}(x)$ "
            rf"($\hat{{\alpha}}={dist.alpha:.2f}, "
            rf"\hat{{\lambda}}={dist.Lambda:.2e}$)"
        ),
        color=model_color,
    )
    D_val = plot_ks_distance_marker(
        ax,
        sorted_data,
        ecdf,
        model_ccdf,
        color=ks_color,
        empirical_color=empirical_color,
        model_color=model_color,
        fast_xmin=fast_xmin,
        ecdf_before=ecdf_before,
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")
    drop_label = _resolve_drop_label(data_info)
    ax.set_xlabel(rf"${_drop_quantity_label(drop_label)}$")
    ax.set_ylabel(r"$P(X > x)$")
    tag = ks_tag(fast_xmin=fast_xmin)
    title = rf"{tag} Distance $E_{{\mathrm{{min}}}}$={xmin:.2e}" + (
        " " + name if name != "" else ""
    )
    if data_info is not None:
        title += make_title_from_data_info(data_info)
    if set_title:
        ax.set_title(title)
    else:
        ax.set_title("")
    if show_legend:
        legend = ax.legend()
        for legend_text in legend.get_texts():
            legend_text.set_usetex(bool(legend_usetex))

    if show_inset:
        if not np.isfinite(inset_x_factor) or inset_x_factor <= 1.0:
            raise ValueError("inset_x_factor must be finite and greater than 1.")
        post_jump_diffs = np.abs(ecdf - model_ccdf)
        pre_jump_diffs = np.abs(ecdf_before - model_ccdf)
        if np.max(pre_jump_diffs) > np.max(post_jump_diffs):
            max_index = int(np.argmax(pre_jump_diffs))
            empirical_at_D = ecdf_before[max_index]
        else:
            max_index = int(np.argmax(post_jump_diffs))
            empirical_at_D = ecdf[max_index]
        x_distance = float(sorted_data[max_index])
        x_low = x_distance / float(inset_x_factor)
        x_high = x_distance * float(inset_x_factor)
        local = (sorted_data >= x_low) & (sorted_data <= x_high)
        if np.count_nonzero(local) < 4:
            lo = max(0, max_index - 8)
            hi = min(sorted_data.size, max_index + 9)
            local = np.zeros(sorted_data.size, dtype=bool)
            local[lo:hi] = True
            x_low = float(sorted_data[lo])
            x_high = float(sorted_data[hi - 1])

        inset_ax = ax.inset_axes(inset_bounds)
        inset_ax.set_facecolor("white")
        inset_ax.patch.set_alpha(float(inset_background_alpha))
        inset_ax.step(
            sorted_data[local],
            ecdf[local],
            where="post",
            color=empirical_color,
            linewidth=1.0,
        )
        inset_ax.plot(
            sorted_data[local],
            model_ccdf[local],
            color=model_color,
            linewidth=1.0,
        )
        inset_ax.vlines(
            x_distance,
            model_ccdf[max_index],
            empirical_at_D,
            color=ks_color,
            linestyle="--",
            linewidth=1.2,
            zorder=4,
        )
        inset_ax.scatter(
            [x_distance],
            [empirical_at_D],
            color=empirical_color,
            s=10,
            zorder=5,
        )
        inset_ax.scatter(
            [x_distance],
            [model_ccdf[max_index]],
            color=model_color,
            s=10,
            zorder=5,
        )
        inset_ax.set_xscale("log")
        inset_ax.set_xlim(x_low, x_high)
        y_low = float(min(empirical_at_D, model_ccdf[max_index]))
        y_high = float(max(empirical_at_D, model_ccdf[max_index]))
        y_span = max(y_high - y_low, 1.0e-4)
        inset_ax.set_ylim(
            max(0.0, y_low - 0.8 * y_span),
            min(1.0, y_high + 0.8 * y_span),
        )
        inset_ax.tick_params(labelsize=5.5, pad=1)
        inset_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        inset_ax.set_xticks([], minor=False)
        inset_ax.set_xticks([], minor=True)
        inset_ax.xaxis.offsetText.set_visible(False)
        if inset_grid:
            inset_ax.grid(True, color="0.9", linewidth=0.4)
        else:
            inset_ax.grid(False, which="both")
    if tight_layout:
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
    selected_fit=None,
):
    """
    We scan many possible xmin values. We try to identify a plateau region
    in the exponents. We make sure the p-value is larger than min_p. If the
    p-value is close to the min_p limit, we need to increaes the accuracy.
    """
    if not 0 < max_accuracy <= start_accuracy:
        raise ValueError("Require 0 < max_accuracy <= start_accuracy.")

    title = make_title(data_info)
    path_name = safePath(title)
    path_name = _append_sample_suffix(path_name, len(drops))

    print(f"Testing xmins for {title}")

    selected_xmin = getattr(selected_fit, "xmin", None)

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

    # Ensure selected_fit is represented in the sampled fit list (for plotting and comparison).
    if selected_fit is not None and selected_xmin is not None:
        has_eval_attrs = all(
            hasattr(selected_fit, attr) for attr in ("p", "p_std", "alpha_std")
        )
        if not has_eval_attrs:
            desc = f"xmin:{selected_xmin:.2e}: selected fit"
            selected_fit.evaluate_fit(parallel=False, tqdmDesc=desc)

        test_fits = [
            f
            for f in test_fits
            if not np.isclose(f.xmin, selected_xmin, rtol=1e-12, atol=0.0)
        ]
        test_fits.append(selected_fit)

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

        refined_accuracy = max(max_accuracy, start_accuracy / 2)
        new_fits = explore_xmin(
            drops,
            new_min_xmin,
            new_max_xmin,
            nr_evaluation,
            refined_accuracy,
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

    # Keep selected_fit in the final collection (the refinement window may have removed it).
    if selected_fit is not None and selected_xmin is not None:
        test_fits = [
            f
            for f in test_fits
            if not np.isclose(f.xmin, selected_xmin, rtol=1e-12, atol=0.0)
        ]
        test_fits.append(selected_fit)

    # Plot p and exponent
    xmin_plot_path = f"{PLOTPATH}{extraPath}{path_name}_xMins.pdf"
    plot_fits_over_xmin(
        test_fits,
        best_fit,
        xmin_plot_path,
        title=title,
        xmin_results=xmin_results,
        fast_xmin=fast_xmin,
        selected_xmin=selected_xmin,
        ks_xmin=get_lowest_distance_xmin(xmin_results),
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
    plastic_col = infer_plastic_event_column(all_data, required=False)
    if plastic_col is not None:
        plastics = all_data[plastic_col][mask].to_numpy()
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

    drop_label = _resolve_drop_label(info)
    drops = np.asarray(drops)
    valid = np.isfinite(drops) & (drops > 0)
    if not np.any(valid):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title(r"$E$-drop PDF and plasticity vs drop size")
            ax.set_xlabel(rf"$-\Delta {drop_label}$")
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
    plastic_col = infer_plastic_event_column(df)
    plastics = df[plastic_col][mask].to_numpy()
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
    plot_data_pdf(ax, drops, drop_label=drop_label)
    ax.set_title(r"$E$-drop PDF and plasticity vs drop size")
    ax.set_xlabel(rf"$-\Delta {drop_label}$")
    ax.set_ylabel("Density (normalized)", color=c_drop_pdf)
    ax.tick_params(axis="y", colors=c_drop_pdf)
    ax.spines["left"].set_color(c_drop_pdf)
    ax.set_xscale("log")

    # Plastic-event PDF vs drop size (normalized over plastic events) on ax1
    plastic_pdf = bin_density  # bin_weight_i / (sum weights) / bin_width_i
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
        plastic_col = infer_plastic_event_column(df)
        plastics = df[plastic_col][mask].to_numpy()

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

    drop_label = _resolve_drop_label(info)
    ax.set_xscale("log")
    ax.set_xlabel(rf"$-\Delta {drop_label}$")
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
    mu = get_elastic_mu()

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

        df = read_macrodata_csv(path, L=get_system_size([path]))
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
        plastic_col = infer_plastic_event_column(df)
        plastics = df[plastic_col][mask].to_numpy()
        # idx = df[plastic_col][mask].idxmax()
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
    drop_label = _resolve_drop_label(info)
    ax.set_ylabel(rf"$-\Delta {drop_label}$")
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
    parallel_xmin=False,
    xmin_search_kwargs=None,
    debug=False,
) -> Fit:
    """Fit at a fixed xmin or at the canonical simpleDrop-selected xmin."""
    if xmin_range is not None:
        if not np.isscalar(xmin_range):
            raise ValueError(
                "Automatic xmin ranges are no longer supported. Pass one fixed "
                "xmin or leave xmin_range=None for the canonical simpleDrop search."
            )
        xmin_range = float(xmin_range)
        if not np.isfinite(xmin_range) or xmin_range <= 0:
            raise ValueError("A fixed xmin must be finite and positive.")
    search_kwargs = dict(xmin_search_kwargs or {})
    forbidden = {"distType", "parallel"} & search_kwargs.keys()
    if forbidden:
        raise ValueError(
            f"xmin_search_kwargs must not override {sorted(forbidden)}."
        )
    parallel_xmin = bool(parallel_xmin)

    # --- try cache
    cache_path = None
    cache_loaded = False
    xmin_analysis = None
    if use_cache:
        import os
        import json
        import gzip

        cache_path = _get_cache_path(
            cache_dir,
            data,
            f"canonical-simpleDrop-v1:{distType.name}:{xmin_range}:"
            f"{parallel_xmin}:{search_kwargs}",
        )
        cache_path_json = cache_path
        cache_path_gz = cache_path + ".gz"
        # Prefer the compressed cache if present; fall back to legacy .json
        if os.path.exists(cache_path_gz) or os.path.exists(cache_path_json):
            try:
                opener = gzip.open if os.path.exists(cache_path_gz) else open
                path = cache_path_gz if os.path.exists(cache_path_gz) else cache_path_json
                with opener(path, "rt", encoding="utf-8") as f:
                    cache = json.load(f)

                xmin_range = cache["xmin"]
                xmin_analysis = cache.get("xmin_analysis")
                if xmin_analysis is not None:
                    for key, dtype in (
                        ("xmins", float),
                        ("distances", float),
                        ("alphas", float),
                        ("sigmas", float),
                        ("valid_fits", bool),
                        ("tail_counts", int),
                    ):
                        xmin_analysis[key] = np.asarray(
                            xmin_analysis[key],
                            dtype=dtype,
                        )
                cache_loaded = True

            except Exception as e:
                # fall through to recompute if loading fails
                print(e)

    if xmin_range is None:
        xmin_analysis = analyze_xmin(
            data,
            distType=distType,
            parallel=parallel_xmin,
            **search_kwargs,
        )
        xmin_range = xmin_analysis["simple_drop_xmin"]

    fitObj = Fit(
        data,
        xmin=xmin_range,
        xmin_distribution=distType.name,
        verbose=0,
    )
    fitObj.xmin_analysis = xmin_analysis
    if xmin_analysis is not None:
        fitObj.xmin_fitting_results = xmin_analysis

    # Save new results and replace any unreadable cache.
    if use_cache and cache_path is not None and not cache_loaded:
        try:
            with gzip.open(cache_path + ".gz", "wt", encoding="utf-8") as f:
                json.dump(
                    _make_json_serializable(
                        {
                            "xmin": fitObj.xmin,
                            "xmin_analysis": xmin_analysis,
                        }
                    ),
                    f,
                )
        except Exception as e:
            # don't fail the computation if saving fails
            print(e)
    if debug:
        if xmin_analysis is None:
            raise ValueError("Cannot plot xmin diagnostics for a fixed-xmin fit.")
        plot_xmin_analysis(xmin_analysis)
        plt.show()
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
    fits,
    best_fit=None,
    savePath=None,
    title=None,
    xmin_results=None,
    fast_xmin=None,
    selected_xmin=None,
    ks_xmin=None,
):
    """
    Plot KS distance/p-value, exponent alpha, and inverse cutoff 1/lambda versus xmin.
    """
    fits.sort(key=lambda f: f.xmin)
    tag = ks_tag(fits=fits, fast_xmin=fast_xmin)
    x = np.array([f.xmin for f in fits], dtype=float)
    pvals = np.array([f.p for f in fits], dtype=float)
    p_stds = np.array([f.p_std for f in fits], dtype=float)
    dists = [dist_from_fit(f) for f in fits]
    alphas = np.array([d.alpha for d in dists], dtype=float)
    alpha_stds = np.array([f.alpha_std for f in fits], dtype=float)
    lambdas = np.array([getattr(d, "Lambda", np.nan) for d in dists], dtype=float)
    inv_lambda = np.full_like(lambdas, np.nan, dtype=float)
    positive_lambda = np.isfinite(lambdas) & (lambdas > 0.0)
    inv_lambda[positive_lambda] = 1.0 / lambdas[positive_lambda]

    # Colors assigned per axis (consistent with Matplotlib defaults)
    c_p = "tab:blue"  # left axis (p-values)
    c_a = "tab:orange"  # right axis (alpha)

    fig, ax1 = plt.subplots(figsize=(8,4))
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
        label="KS-distance / p-value",
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
    ax1.set_ylabel("KS-distance / p-value", color=c_p)
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
            # KS distance curve (plotted over the range used for the p/alpha curves)
            ax1.plot(
                xmins[max_xmin_filter],
                distances[max_xmin_filter],
                linestyle="--",
                linewidth=1.2,
                color="0.5",
                alpha=0.7,
                label="KS distance",
                zorder=0,
            )

    if np.isfinite(inv_lambda).any():
        c_l = "tab:green"
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.14))
        ax3.spines["right"].set_visible(True)
        ax3.plot(
            x,
            inv_lambda,
            marker="^",
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            color=c_l,
            label=r"Inverse cutoff $1/\lambda$",
        )
        ax3.set_ylabel(r"Inverse cutoff $1/\lambda$", color=c_l)
        ax3.set_yscale("log")
        finite_inv = inv_lambda[np.isfinite(inv_lambda) & (inv_lambda > 0.0)]
        if finite_inv.size > 0:
            inv_min = float(np.min(finite_inv))
            inv_max = float(np.max(finite_inv))
            # Keep log scaling, but use plain float tick labels in a human range.
            if inv_min >= 0.1 and inv_max <= 99.0:
                if inv_max / inv_min < 10.0:
                    ax3.yaxis.set_major_locator(
                        mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0))
                    )
                ax3.yaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(lambda val, _pos: f"{val:g}")
                )
        ax3.tick_params(axis="y", colors=c_l, which="both")
        ax3.spines["right"].set_color(c_l)
        ax3.yaxis.label.set_color(c_l)
        ax3.yaxis.get_offset_text().set_color(c_l)

    # --- Right axis: alpha (+ error bars) ---
    # Keep this axis creation after ax3 so alpha/legend remain on top.
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
    )
    ax2.set_ylabel(r"Exponent $\alpha$", color=c_a)
    ax2.tick_params(axis="y", colors=c_a)
    ax2.spines["right"].set_color(c_a)

    if selected_xmin is not None and np.isfinite(selected_xmin):
        ax1.axvline(
            selected_xmin,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"{tag} xmin: {selected_xmin:.2e}",
            zorder=-1,
            alpha=0.7,
        )
    if (
        ks_xmin is not None
        and np.isfinite(ks_xmin)
        and (selected_xmin is None or not np.isclose(ks_xmin, selected_xmin, rtol=1e-12, atol=0.0))
    ):
        ax1.axvline(
            ks_xmin,
            color="tab:green",
            linestyle="-.",
            linewidth=1.2,
            label=f"KS xmin: {ks_xmin:.2e}",
            zorder=-0.95,
            alpha=0.7,
        )
    if best_fit and np.isfinite(best_fit.xmin):
        ax1.axvline(
            best_fit.xmin,
            color="tab:purple",
            linestyle=":",
            linewidth=1.2,
            label=rf"$p$-fit xmin: {best_fit.xmin:.2e}",
            zorder=-0.9,
            alpha=0.7,
        )

    # --- Legend: collect handles from both axes ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ([], [])
    if ax3 is not None:
        handles3, labels3 = ax3.get_legend_handles_labels()

    # Deduplicate by label while preserving order
    seen = set()
    handles = []
    labels = []
    for h, l in (
        list(zip(handles1, labels1)) + list(zip(handles2, labels2))
        + list(zip(handles3, labels3))
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
            label=f"{tag} xmin: {chosen_xmin:.2e}",
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
            label=f"{tag} xmin: {chosen_xmin:.2e}",
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


def plot_powerlaw_compare(
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
    parallel_xmin=False,
    xmin_search_kwargs=None,
    csvPaths=None,
    useCDF=False,
    drop_type="energy",
):
    if group_paths is None and csvPaths is not None:
        group_paths = csvPaths
    if drop_type not in {"energy", "stress"}:
        raise ValueError(
            f"Unsupported drop_type: {drop_type!r}. Use 'energy' or 'stress'."
        )
    grouped_paths, grouped_labels = get_group_structure(group_paths, group_labels)
    if len(grouped_paths) <= 1:
        return plot_powerlaw(
            group_paths=grouped_paths,
            group_labels=grouped_labels,
            strainLim=strainLim,
            postRegime=postRegime,
            xmin_range=xmin_range,
            debug=debug,
            show=show,
            evaluate=evaluate,
            distType=distType,
            save=save,
            addFit=addFit,
            parallel_xmin=parallel_xmin,
            xmin_search_kwargs=xmin_search_kwargs,
            useCDF=useCDF,
            drop_type=drop_type,
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = list(MINIMIZER_COLORS.values())

    selected_tag = "simpleDrop"
    compare_infos = []
    compare_fits = []
    legend_handles = []
    equation_entry = None

    for idx, (paths, labels) in enumerate(zip(grouped_paths, grouped_labels)):
        if drop_type == "stress":
            drops, data_info = get_stress_drops(
                paths,
                strainLim=strainLim,
                label=labels,
                postRegime=postRegime,
            )
        else:
            drops, data_info = get_energy_drops(
                paths,
                strainLim=strainLim,
                debug=debug,
                label=labels,
                postRegime=postRegime,
            )
        if drops is None or len(drops) == 0:
            print(f"No {drop_type} drops found for compare group {idx}; skipping.")
            continue

        fit = make_fit(
            data=drops,
            xmin_range=xmin_range,
            distType=distType,
            parallel_xmin=parallel_xmin,
            xmin_search_kwargs=xmin_search_kwargs,
        )
        if evaluate:
            fit.evaluate_fit()

        color = colors[idx % len(colors)]
        source_label = labels[0] if labels else ""
        group_name = pretty_variant_label(source_label) or f"group {idx + 1}"
        legend_label = compare_legend_label(
            source_label,
            fit,
            nr_samples=data_info.get("nrSimulations"),
            nr_drops=_count_fit_drops(fit),
        )
        drop_label = data_info.get("drop_label")
        if equation_entry is None:
            equation_entry = fit_equation_label(dist_from_fit(fit).name)

        if useCDF:
            plot_data_cdf(
                ax,
                fit.data_original,
                label="_nolegend_",
                color=color,
                alpha=0.9,
                use_ccdf=True,
                drop_label=drop_label,
                show_legend=False,
            )
            if addFit:
                plot_fit_cdf(
                    ax,
                    fit,
                    label="_nolegend_",
                    color=color,
                    use_ccdf=True,
                    drop_label=drop_label,
                    show_legend=False,
                    set_title=False,
                    x_grid_mode="smooth",
                    xmin_only=True,
                    linewidth=1.8,
                )
        else:
            plot_data_pdf(
                ax,
                fit.data_original,
                label="_nolegend_",
                color=color,
                alpha=0.9,
                drop_label=drop_label,
                show_legend=False,
            )
            if addFit:
                plot_fit_pdf(
                    ax,
                    fit,
                    label="_nolegend_",
                    color=color,
                    drop_label=drop_label,
                    show_legend=False,
                    set_title=False,
                    x_grid_mode="smooth",
                    xmin_only=True,
                    linewidth=1.8,
                )

        ax.axvline(
            fit.xmin,
            color=color,
            linestyle=":",
            linewidth=1.2,
            alpha=0.45,
            label="_nolegend_",
        )
        xmin_analysis = getattr(fit, "xmin_analysis", None)
        if xmin_analysis is not None and xmin_global_differs(xmin_analysis):
            ax.axvline(
                xmin_analysis["global_min_xmin"],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.55,
                label="_nolegend_",
            )
            legend_label += (
                rf"; global $x_{{\min}}={xmin_analysis['global_min_xmin']:.1e}$"
            )

        print(f"{selected_tag} xmin ({group_name}): {fit.xmin}")
        legend_handles.append(
            mpl.lines.Line2D(
                [],
                [],
                color=color,
                linewidth=1.8,
                linestyle="-",
                marker=None if useCDF else "o",
                markersize=5,
                label=legend_label,
            )
        )
        compare_infos.append(data_info)
        compare_fits.append(fit)

    if not compare_infos:
        plt.close(fig)
        print("No valid groups found for powerlaw comparison.")
        return None

    shared_info = dict(compare_infos[0])
    shared_info.pop("label", None)
    shared_info["nrSimulations"] = sum(
        info.get("nrSimulations", 0) for info in compare_infos
    )
    title = make_compare_title_from_data_info(shared_info).strip()
    if title:
        title += f" {selected_tag} compare"
    else:
        title = f"{selected_tag}_powerlaw_compare"

    ax.set_title(title)
    if equation_entry:
        legend_handles = [
            mpl.lines.Line2D(
                [],
                [],
                color="none",
                linestyle="none",
                label=equation_entry,
            )
        ] + legend_handles
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()

    if show:
        plt.show()
    if save:
        suffix = "_ccdf" if useCDF else "_pdf"
        filename = f"{PLOTPATH}{safePath(title)}{suffix}.pdf"
        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"Saved figure to {filename}")
        setattr(fig, "path", filename)
        plt.close(fig)
    else:
        setattr(fig, "path", None)
    setattr(fig, "fits", compare_fits)

    return fig


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
    parallel_xmin=False,
    xmin_search_kwargs=None,
    csvPaths=None,
    useCDF=False,
    drop_type="energy",
):
    if group_paths is None and csvPaths is not None:
        group_paths = csvPaths
    if drop_type not in {"energy", "stress"}:
        raise ValueError(
            f"Unsupported drop_type: {drop_type!r}. Use 'energy' or 'stress'."
        )

    grouped_paths, grouped_labels = get_group_structure(group_paths, group_labels)
    if len(grouped_paths) > 1:
        return plot_powerlaw_compare(
            group_paths=grouped_paths,
            group_labels=grouped_labels,
            strainLim=strainLim,
            postRegime=postRegime,
            xmin_range=xmin_range,
            debug=debug,
            show=show,
            evaluate=evaluate,
            distType=distType,
            save=save,
            addFit=addFit,
            parallel_xmin=parallel_xmin,
            xmin_search_kwargs=xmin_search_kwargs,
            useCDF=useCDF,
            drop_type=drop_type,
        )

    # We only deal with one group
    if drop_type == "stress":
        all_drops, data_info = get_stress_drops(
            grouped_paths[0],
            strainLim=strainLim,
            label=grouped_labels[0],
            postRegime=postRegime,
        )
    else:
        all_drops, data_info = get_energy_drops(
            grouped_paths[0],
            strainLim=strainLim,
            debug=debug,
            label=grouped_labels[0],
            postRegime=postRegime,
        )
    if all_drops is None or len(all_drops) == 0:
        print(f"No {drop_type} drops found; skipping powerlaw fit.")
        return None
    reported_fit = make_fit(
        data=all_drops,
        xmin_range=xmin_range,
        distType=distType,
        parallel_xmin=parallel_xmin,
        xmin_search_kwargs=xmin_search_kwargs,
    )
    print("simpleDrop xmin:", reported_fit.xmin)
    if evaluate:
        reported_fit.evaluate_fit()

    d = dist_from_fit(reported_fit)

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
        p, mean_exp, exp_std = reported_fit.evaluate_fit(all_drops, parallel=True)

        thresholds = [0.05, 0.1, 0.3, float("inf")]
        ratings = ["bad", "poor", "good", "excellent"]

        # Set r
        for t, r in zip(thresholds, ratings):
            if p < t:
                break

        print(f"Number of drops in fit: {_count_fit_drops(reported_fit)}")
        print(f"{attribute}: P value: {p:.2f} ({r}), exp: {d.alpha}, std: {exp_std}")
    title = make_title(data_info=data_info, fit=reported_fit)
    if attribute and attribute not in title:
        title = attribute + " " + title
    xmin_analysis = getattr(reported_fit, "xmin_analysis", None)
    if xmin_analysis is not None:
        global_differs = xmin_global_differs(xmin_analysis)
        if global_differs:
            print("Refined global-min xmin:", xmin_analysis["global_min_xmin"])
        fig, ax = plot_xmin_analysis(xmin_analysis)
        ax.set_title(f"{title} xmin search")
        fig.tight_layout()
        if save:
            filename = f"{PLOTPATH}{safePath(title)}_xmin_search.pdf"
            fig.savefig(filename, format="pdf", bbox_inches="tight")
            print(f"Saved figure to {filename}")
            plt.close(fig)
        elif not show:
            plt.close(fig)
        if not global_differs:
            print("Refined global minimum agrees with simpleDrop.")
        plot_ks_distance(
            all_drops,
            reported_fit.xmin,
            data_info=data_info,
            name="simpleDrop-fit",
            save=save,
        )
    plot_data_and_fit(
        reported_fit,
        title=title,
        data_info=data_info,
        color=color,
        addFit=addFit,
        save=save,
        show=show,
        useCDF=useCDF,
    )

    return reported_fit


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

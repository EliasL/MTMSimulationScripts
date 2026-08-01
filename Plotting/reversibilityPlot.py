import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Management.configGenerator import SimulationConfig
from Management.updateCSV import update_df_header, read_macrodata_csv
from Plotting.remotePlotting import get_csv_files
from Plotting.makePlots import maybe_avg
from Plotting.plotPowerLaw import (
    get_energy_drops,
    getHist,
    get_system_size,
    pretty_variant_label,
)


def _resolve_csv_path(config_file, useOldFiles=False, forceUpdate=False):
    if isinstance(config_file, SimulationConfig):
        config = config_file
    elif isinstance(config_file, (str, os.PathLike)):
        path = str(config_file)
        if path.endswith(".csv"):
            return path
        config = SimulationConfig(path)
    else:
        raise ValueError(
            "config_file must be a .conf path, .csv path, or SimulationConfig"
        )

    paths, _ = get_csv_files([config], useOldFiles=useOldFiles, forceUpdate=forceUpdate)
    if not paths:
        raise RuntimeError("No CSV files found for the provided config.")
    return paths[0]


def _collect_reversibility_data(csv_paths, strainLim="auto", postRegime=True):
    if isinstance(csv_paths, (str, os.PathLike)):
        csv_paths = [str(csv_paths)]
    if not csv_paths:
        raise RuntimeError("No CSV paths provided.")

    L = get_system_size(csv_paths)
    all_drops = []
    all_is_rev = []
    all_rev_u = []
    has_rev_u = True

    for csv_path in csv_paths:
        df = read_macrodata_csv(csv_path, L=L)

        rev_col = "is_reversible"
        if rev_col not in df:
            raise RuntimeError(f"Missing '{rev_col}' column in {csv_path}")

        drops, data_info = get_energy_drops(
            csv_path,
            df=df,
            strainLim=strainLim,
            postRegime=postRegime,
        )
        drop_mask = data_info["masks"][0]
        is_rev = np.array(df[rev_col], dtype=bool)[drop_mask]

        all_drops.append(drops)
        all_is_rev.append(is_rev)

        if "rev_u_diff" in df:
            all_rev_u.append(df["rev_u_diff"].to_numpy()[drop_mask])
        else:
            has_rev_u = False

    drops = np.concatenate(all_drops) if all_drops else np.array([])
    is_rev = np.concatenate(all_is_rev) if all_is_rev else np.array([], dtype=bool)
    rev_u = None
    if has_rev_u and all_rev_u:
        rev_u = np.concatenate(all_rev_u)
    return drops, is_rev, rev_u


def _collect_reversible_state_differences(
    csv_paths, strainLim="auto", postRegime=True
):
    """Collect state-4 minus state-0 diagnostics for reversible drops."""
    if isinstance(csv_paths, (str, os.PathLike)):
        csv_paths = [str(csv_paths)]
    if not csv_paths:
        raise RuntimeError("No CSV paths provided.")

    L = get_system_size(csv_paths)
    all_load = []
    all_energy = []
    all_sigma12 = []

    required_columns = {
        "is_reversible",
        "rev_energy_diff",
        "rev_sigma_12_diff",
    }

    for csv_path in csv_paths:
        df = read_macrodata_csv(csv_path, L=L)
        missing = required_columns.difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise RuntimeError(
                f"Missing reversibility columns ({missing_text}) in {csv_path}"
            )

        # Reuse the same event/drop mask as the existing reversibility plots.
        _, data_info = get_energy_drops(
            csv_path,
            df=df,
            strainLim=strainLim,
            postRegime=postRegime,
        )
        drop_mask = data_info["masks"][0]
        is_reversible = df["is_reversible"].to_numpy(dtype=bool)[drop_mask]

        load = df["load"].to_numpy(dtype=float)[drop_mask]
        energy = df["rev_energy_diff"].to_numpy(dtype=float)[drop_mask]
        sigma12 = df["rev_sigma_12_diff"].to_numpy(dtype=float)[drop_mask]
        finite = (
            is_reversible
            & np.isfinite(load)
            & np.isfinite(energy)
            & np.isfinite(sigma12)
        )

        all_load.append(load[finite])
        all_energy.append(energy[finite])
        all_sigma12.append(sigma12[finite])

    return (
        np.concatenate(all_load) if all_load else np.array([]),
        np.concatenate(all_energy) if all_energy else np.array([]),
        np.concatenate(all_sigma12) if all_sigma12 else np.array([]),
    )


def plot_reversible_state_differences(
    config_file,
    postRegime: bool | None = True,
    strainLim="auto",
    show=True,
    save_path=None,
    group_labels=None,
):
    """Plot energy and shear-stress return differences for reversible drops."""
    if isinstance(config_file, (list, tuple)) and config_file:
        if isinstance(config_file[0], (list, tuple)):
            groups = list(config_file)
        else:
            groups = [config_file]
    else:
        groups = [[config_file]]

    if group_labels is None:
        group_labels = [f"group_{i}" for i in range(len(groups))]
    elif isinstance(group_labels, (str, os.PathLike)):
        group_labels = [str(group_labels)]

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(5.0, 4.5))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        raise RuntimeError("Matplotlib default color cycle is empty.")

    plotted = 0
    for idx, entries in enumerate(groups):
        csv_paths = [_resolve_csv_path(entry) for entry in entries]
        load, energy, sigma12 = _collect_reversible_state_differences(
            csv_paths,
            strainLim=strainLim,
            postRegime=postRegime,
        )
        if load.size == 0:
            continue

        order = np.argsort(load)
        load = load[order]
        energy = energy[order]
        sigma12 = sigma12[order]
        label = group_labels[idx] if idx < len(group_labels) else f"group_{idx}"
        display_label = pretty_variant_label(label) or str(label)
        color = default_colors[idx % len(default_colors)]

        for ax, values in zip(axes, (energy, sigma12)):
            ax.plot(
                load,
                values,
                linestyle="None",
                marker="o",
                markersize=3.0,
                color=color,
                alpha=0.7,
                label=display_label,
            )
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        raise RuntimeError("No reversible state-difference data found.")

    axes[0].set_ylabel(r"$|E_4-E_0|$")
    axes[1].set_ylabel(r"$|\sigma_{12,4}-\sigma_{12,0}|$")
    axes[1].set_xlabel(r"strain $\gamma$")
    axes[0].legend(loc="best", frameon=True)
    figure.tight_layout()

    if save_path:
        figure.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, axes


def _smooth_histogram_curve(bin_centers, counts):
    """Smooth positive histogram points in log-log coordinates."""
    mask = (
        np.isfinite(bin_centers)
        & np.isfinite(counts)
        & (bin_centers > 0)
        & (counts > 0)
    )
    x = np.asarray(bin_centers)[mask]
    y = np.asarray(counts)[mask]
    if x.size < 3:
        return x, y

    log_x = np.log10(x)
    log_y = np.log10(y)
    smooth_log_x = np.linspace(log_x[0], log_x[-1], max(100, 4 * x.size))
    smooth_log_y = np.interp(smooth_log_x, log_x, log_y)
    window = min(31, smooth_log_y.size)
    if window % 2 == 0:
        window -= 1
    if window >= 3:
        half_window = window // 2
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(smooth_log_y, half_window, mode="edge")
        smooth_log_y = np.convolve(padded, kernel, mode="valid")
    return 10**smooth_log_x, 10**smooth_log_y


def _plot_histogram_series(ax, centers, counts, *, color, marker, linestyle, label):
    smooth_centers, smooth_counts = _smooth_histogram_curve(centers, counts)
    ax.plot(
        smooth_centers,
        smooth_counts,
        linestyle=linestyle,
        linewidth=1.5,
        color=color,
        label=label,
    )
    mask = counts > 0
    ax.plot(
        centers[mask],
        counts[mask],
        marker=marker,
        linestyle="None",
        color=color,
        alpha=0.2,
        markerfacecolor="none",
        label="_nolegend_",
    )


def plot_reversibility_histograms(
    config_file,
    bins=50,
    postRegime: bool | None = True,
    strainLim="auto",
    show=True,
    save_path=None,
    group_labels=None,
):
    """
    Load CSV associated with a config file (or list of configs/paths) and plot
    histograms of energy drops split by reversibility. Prints the reversible drop
    number with max rev_d.
    """
    if isinstance(config_file, (list, tuple)) and config_file:
        if isinstance(config_file[0], (list, tuple)):
            groups = list(config_file)
        else:
            groups = [config_file]
    else:
        groups = [[config_file]]

    if group_labels is None:
        group_labels = [f"group_{i}" for i in range(len(groups))]
    elif isinstance(group_labels, (str, os.PathLike)):
        group_labels = [str(group_labels)]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        raise RuntimeError("Matplotlib default color cycle is empty.")

    infos = []
    for idx, entries in enumerate(groups):
        csv_paths = [_resolve_csv_path(entry) for entry in entries]
        label = group_labels[idx] if idx < len(group_labels) else f"group_{idx}"
        display_label = pretty_variant_label(label) or str(label)

        drops, is_rev, rev_u = _collect_reversibility_data(
            csv_paths, strainLim=strainLim, postRegime=postRegime
        )

        if drops.size == 0:
            print(f"No drops found for {display_label}.")
            continue

        rev_order = np.zeros(len(drops), dtype=int)
        irrev_order = np.zeros(len(drops), dtype=int)
        rev_count = 0
        irrev_count = 0
        for i, rev in enumerate(is_rev):
            if rev:
                rev_count += 1
                rev_order[i] = rev_count
            else:
                irrev_count += 1
                irrev_order[i] = irrev_count

        if rev_u is not None and np.any(is_rev):
            rev_indices = np.where(is_rev)[0]
            idx_max = rev_indices[np.argmax(rev_u[is_rev])]
            print(
                f"{display_label} highest rev_u_diff reversible drop #: "
                f"{rev_order[idx_max]} (rev_u_diff={rev_u[idx_max]:.6g}, "
                f"drop={drops[idx_max]:.6g})"
            )
        else:
            print(
                f"{display_label}: No reversible drops found "
                "(or 'rev_u_diff' missing)."
            )

        if rev_u is not None and np.any(~is_rev):
            irrev_indices = np.where(~is_rev)[0]
            idx_max = irrev_indices[np.argmax(rev_u[~is_rev])]
            print(
                f"{display_label} highest rev_u_diff irreversible drop #: "
                f"{irrev_order[idx_max]} (rev_u_diff={rev_u[idx_max]:.6g}, "
                f"drop={drops[idx_max]:.6g})"
            )
        else:
            print(
                f"{display_label}: No irreversible drops found "
                "(or 'rev_u_diff' missing)."
            )

        rev_drops = drops[is_rev]
        irrev_drops = drops[~is_rev]
        rev_drops = rev_drops[rev_drops > 0]
        irrev_drops = irrev_drops[irrev_drops > 0]

        color = default_colors[idx % len(default_colors)]

        if rev_drops.size:
            rev_centers, rev_counts = getHist(
                rev_drops, density=False, bins_per_decade=bins
            )
            _plot_histogram_series(
                ax1,
                rev_centers,
                rev_counts,
                color=color,
                marker="x",
                linestyle="--",
                label=f"{display_label} (rev)",
            )
        if irrev_drops.size:
            irrev_centers, irrev_counts = getHist(
                irrev_drops, density=False, bins_per_decade=bins
            )
            _plot_histogram_series(
                ax2,
                irrev_centers,
                irrev_counts,
                color=color,
                marker="o",
                linestyle="-",
                label=f"{display_label} (irrev)",
            )

        infos.append(
            {
                "csv_paths": csv_paths,
                "reversible_count": rev_count,
                "irreversible_count": irrev_count,
                "reversible_order": rev_order,
                "irreversible_order": irrev_order,
                "label": display_label,
            }
        )

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel(rf"$-\Delta {maybe_avg('E')}$")
    ax1.set_ylabel("Reversible count")
    ax2.set_ylabel("Irreversible count")
    ax1.tick_params(axis="y", which="both")
    ax2.tick_params(axis="y", which="both")
    ax1.set_title("Reversible vs. irreversible energy drops")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax1, infos

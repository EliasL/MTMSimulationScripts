import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from Management.configGenerator import SimulationConfig
from Management.updateCSV import update_df_header
from Plotting.remotePlotting import get_csv_files
from Plotting.makePlots import maybe_avg
from Plotting.plotPowerLaw import get_energy_drops, getHist, get_system_size


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
        df = pd.read_csv(csv_path)
        df = update_df_header(df, L=L)

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

    n_groups = max(1, len(groups))
    red_cmap = mpl.colormaps["Reds"]
    blue_cmap = mpl.colormaps["Blues"]
    shade_vals = np.linspace(0.35, 0.85, n_groups)
    group_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    infos = []
    for idx, entries in enumerate(groups):
        csv_paths = [_resolve_csv_path(entry) for entry in entries]
        label = group_labels[idx] if idx < len(group_labels) else f"group_{idx}"

        drops, is_rev, rev_u = _collect_reversibility_data(
            csv_paths, strainLim=strainLim, postRegime=postRegime
        )

        if drops.size == 0:
            print(f"No drops found for {label}.")
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
                f"{label} highest rev_u_diff reversible drop #: "
                f"{rev_order[idx_max]} (rev_u_diff={rev_u[idx_max]:.6g}, "
                f"drop={drops[idx_max]:.6g})"
            )
        else:
            print(f"{label}: No reversible drops found (or 'rev_u_diff' missing).")

        if rev_u is not None and np.any(~is_rev):
            irrev_indices = np.where(~is_rev)[0]
            idx_max = irrev_indices[np.argmax(rev_u[~is_rev])]
            print(
                f"{label} highest rev_u_diff irreversible drop #: "
                f"{irrev_order[idx_max]} (rev_u_diff={rev_u[idx_max]:.6g}, "
                f"drop={drops[idx_max]:.6g})"
            )
        else:
            print(f"{label}: No irreversible drops found (or 'rev_u_diff' missing).")

        rev_drops = drops[is_rev]
        irrev_drops = drops[~is_rev]
        rev_drops = rev_drops[rev_drops > 0]
        irrev_drops = irrev_drops[irrev_drops > 0]

        rev_color = red_cmap(shade_vals[idx % n_groups])
        irrev_color = blue_cmap(shade_vals[idx % n_groups])
        marker = group_markers[idx % len(group_markers)]

        if rev_drops.size:
            rev_centers, rev_counts = getHist(
                rev_drops, density=False, bins_per_decade=bins
            )
            rev_mask = rev_counts > 0
            ax1.plot(
                rev_centers[rev_mask],
                rev_counts[rev_mask],
                marker=marker,
                linestyle="None",
                color=rev_color,
                label=f"{label} (rev)",
                markerfacecolor="none",
            )
        if irrev_drops.size:
            irrev_centers, irrev_counts = getHist(
                irrev_drops, density=False, bins_per_decade=bins
            )
            irrev_mask = irrev_counts > 0
            ax2.plot(
                irrev_centers[irrev_mask],
                irrev_counts[irrev_mask],
                marker=marker,
                linestyle="None",
                color=irrev_color,
                label=f"{label} (irrev)",
                markerfacecolor="none",
            )

        infos.append(
            {
                "csv_paths": csv_paths,
                "reversible_count": rev_count,
                "irreversible_count": irrev_count,
                "reversible_order": rev_order,
                "irreversible_order": irrev_order,
                "label": label,
            }
        )

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel(rf"$-\Delta {maybe_avg('E')}$")
    ax1.set_ylabel("Reversible count", color="tab:red")
    ax2.set_ylabel("Irreversible count", color="tab:blue")
    ax1.tick_params(axis="y", which="both", colors="tab:red")
    ax2.tick_params(axis="y", which="both", colors="tab:blue")
    ax1.set_title("Reversible vs. irreversible energy drops")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        ncol=2,
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

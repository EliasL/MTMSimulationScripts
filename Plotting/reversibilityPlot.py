import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from Management.configGenerator import SimulationConfig
from Management.updateCSV import update_df_header
from Plotting.remotePlotting import get_csv_files
from Plotting.makePlots import maybe_avg
from Plotting.plotPowerLaw import (
    get_energy_drops,
    getHist,
    findPrePostSplit,
    get_system_size,
)


def _resolve_csv_path(config_file, useOldFiles=False, forceUpdate=False):
    if isinstance(config_file, SimulationConfig):
        config = config_file
    elif isinstance(config_file, str):
        if config_file.endswith(".csv"):
            return config_file
        config = SimulationConfig(config_file)
    else:
        raise ValueError(
            "config_file must be a .conf path, .csv path, or SimulationConfig"
        )

    paths, _ = get_csv_files([config], useOldFiles=useOldFiles, forceUpdate=forceUpdate)
    if not paths:
        raise RuntimeError("No CSV files found for the provided config.")
    return paths[0]


def plot_reversibility_histograms(
    config_file,
    bins=50,
    postRegime: bool | None = True,
    strainLim="auto",
    show=True,
    save_path=None,
):
    """
    Load CSV associated with a config file and plot histograms of energy drops
    split by reversibility. Prints the reversible drop number with max rev_d.
    """
    csv_path = _resolve_csv_path(config_file)

    df = pd.read_csv(csv_path)
    df = update_df_header(df, L=get_system_size([csv_path]))
    resolved_strain_lim = strainLim
    if resolved_strain_lim is None or resolved_strain_lim == "auto":
        gamma_max_stress = findPrePostSplit(df=df)
        if postRegime:
            resolved_strain_lim = [gamma_max_stress + 1e-2, df["load"].max()]
        elif postRegime is None:
            resolved_strain_lim = [df["load"].min(), df["load"].max()]
        else:
            resolved_strain_lim = [df["load"].min(), gamma_max_stress - 1e-4]

    if "avg_e_change_from_init" in df:
        energy_col = "avg_e_change_from_init"
    elif "e_change_from_init" in df:
        energy_col = "e_change_from_init"
    else:
        raise RuntimeError(
            "Missing energy change column. Expected 'avg_e_change_from_init' or "
            "'e_change_from_init'."
        )

    rev_col = "is_reversible"
    is_rev = np.array(df[rev_col], dtype=bool)

    drops, data_info = get_energy_drops(csv_path, df=df, strainLim=resolved_strain_lim)
    drop_mask = data_info["masks"][0]
    is_rev = is_rev[drop_mask]

    if "rev_d" in df:
        rev_d = df["rev_d"].to_numpy()[drop_mask]
    else:
        rev_d = None
        print("Warning: 'rev_d' column not found.")

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

    if rev_d is not None and np.any(is_rev):
        rev_indices = np.where(is_rev)[0]
        idx_max = rev_indices[np.argmax(rev_d[is_rev])]
        print(
            "Highest rev_d reversible drop #: "
            f"{rev_order[idx_max]} (rev_d={rev_d[idx_max]:.6g}, "
            f"drop={drops[idx_max]:.6g})"
        )
    else:
        print("No reversible drops found (or 'rev_d' missing).")

    if rev_d is not None and np.any(~is_rev):
        irrev_indices = np.where(~is_rev)[0]
        idx_max = irrev_indices[np.argmax(rev_d[~is_rev])]
        print(
            "Highest rev_d irreversible drop #: "
            f"{irrev_order[idx_max]} (rev_d={rev_d[idx_max]:.6g}, "
            f"drop={drops[idx_max]:.6g})"
        )
    else:
        print("No irreversible drops found (or 'rev_d' missing).")

    rev_drops = drops[is_rev]
    irrev_drops = drops[~is_rev]
    rev_drops = rev_drops[rev_drops > 0]
    irrev_drops = irrev_drops[irrev_drops > 0]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    blues = mpl.colormaps["Blues"]
    oranges = mpl.colormaps["Oranges"]
    rev_color = blues(0.7)
    irrev_color = oranges(0.7)

    if rev_drops.size:
        rev_centers, rev_counts = getHist(
            rev_drops, density=False, bins_per_decade=bins
        )
        rev_mask = rev_counts > 0
        ax1.plot(
            rev_centers[rev_mask],
            rev_counts[rev_mask],
            marker="o",
            linestyle="-",
            color=rev_color,
            label="Reversible",
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
            marker="s",
            linestyle="-",
            color=irrev_color,
            label="Irreversible",
            markerfacecolor="none",
        )

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel(rf"$-\Delta {maybe_avg('E')}$")
    ax1.set_ylabel("Reversible count", color=rev_color)
    ax2.set_ylabel("Irreversible count", color=irrev_color)
    ax1.tick_params(axis="y", which="both", colors=rev_color)
    ax2.tick_params(axis="y", which="both", colors=irrev_color)
    ax1.set_title("Reversible vs. irreversible energy drops")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    info = {
        "csv_path": csv_path,
        "energy_column": energy_col,
        "reversible_count": rev_count,
        "irreversible_count": irrev_count,
        "reversible_order": rev_order,
        "irreversible_order": irrev_order,
    }
    return fig, ax1, info

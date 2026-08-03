import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from Management.configGenerator import SimulationConfig
from Management.updateCSV import update_df_header, read_macrodata_csv
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import volume_from_metadata
from Plotting.remotePlotting import get_csv_files
from Plotting.plotPowerLaw import (
    get_energy_drops,
    getHist,
    findPrePostSplit,
    get_system_size,
    pretty_variant_label,
    strip_seed_from_label,
)
from Plotting.remoteDataPaths import RAW_DATA_PATH
from Plotting.vtuDataForSylvain import VTUData


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
            stress_corrected=True,
            stress_correction_order=2,
            stress_tangent="current",
        )
        mesh_volume = volume_from_metadata(get_metadata(csv_path))
        if mesh_volume is None or not np.isfinite(mesh_volume) or mesh_volume <= 0:
            raise ValueError(f"Could not infer a positive mesh volume for {csv_path}")
        drops = drops / mesh_volume
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
    histograms of first-order stress-corrected energy drops split by
    reversibility. Prints the reversible drop number with max rev_d.
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
                "color": color,
            }
        )

    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")

    ax1.set_xlabel(r"$-\Delta E_S/V_{\mathrm{mesh}}$")
    ax1.set_ylabel("Reversible count")
    ax2.set_ylabel("Irreversible count")
    ax1.tick_params(axis="y", which="both")
    ax2.tick_params(axis="y", which="both")
    ax1.set_title("Reversible vs. irreversible energy drops")

    setting_handles = [
        Line2D([], [], color=info["color"], linewidth=2, label=info["label"])
        for info in infos
    ]
    type_handles = [
        Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            marker="x",
            markersize=6,
            linewidth=1.5,
            label="rev",
        ),
        Line2D(
            [],
            [],
            color="black",
            linestyle="-",
            marker="o",
            markerfacecolor="none",
            markersize=6,
            linewidth=1.5,
            label="irrev",
        ),
    ]
    ax2.legend(
        handles=setting_handles + type_handles,
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


_REVERSIBILITY_EVENT_PATTERN = re.compile(
    r"(?P<kind>rev|irrev)_drop_l_(?P<load>[0-9.eE+-]+)$"
)


def _job_name_from_csv_path(csv_path):
    path = Path(csv_path)
    return path.parent.name if path.name == "macroData.csv" else path.stem


def _single_state_file(event_dir, state_name, *, allow_missing=False):
    matches = sorted(event_dir.glob(f"{state_name}.*.vtu"))
    if allow_missing and not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {state_name} VTU in {event_dir}, found {matches}."
        )
    return matches[0]


def _ordered_xy_points(vtu_path):
    data = VTUData(vtu_path)
    points = np.asarray(data.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected 2D point coordinates in {vtu_path}, got {points.shape}.")
    ref_index, location, _ = data.field("refIndex")
    if location != "point":
        raise ValueError(f"Expected point-wise refIndex in {vtu_path}, got {location}.")
    ref_index = np.asarray(ref_index)
    if ref_index.ndim != 1 or ref_index.shape[0] != points.shape[0]:
        raise ValueError(f"Invalid refIndex shape {ref_index.shape} in {vtu_path}.")
    # Periodic meshes can contain repeated refIndex values for image nodes, so
    # the point order itself is the unambiguous correspondence here.
    return points[:, :2], ref_index


def _delta_u_relaxation(event_dir):
    affine_file = _single_state_file(
        event_dir, "state1_affine_gamma_plus", allow_missing=True
    )
    relaxed_file = _single_state_file(
        event_dir, "state2_relaxed_gamma_plus", allow_missing=True
    )
    if affine_file is None and relaxed_file is None:
        return None
    if affine_file is None or relaxed_file is None:
        raise RuntimeError(f"Incomplete affine/relaxed VTU pair in {event_dir}.")
    affine_points, affine_refs = _ordered_xy_points(affine_file)
    relaxed_points, relaxed_refs = _ordered_xy_points(relaxed_file)
    if not np.array_equal(affine_refs, relaxed_refs):
        raise ValueError(f"The affine and relaxed meshes have different refIndex values in {event_dir}.")
    displacement = relaxed_points - affine_points
    displacement -= displacement.mean(axis=0)
    result = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"Invalid Delta u_R={result} calculated for {event_dir}.")
    return result


def _reversibility_event_records(csv_path):
    df = read_macrodata_csv(csv_path)
    required_columns = {"load", "is_reversible", "rev_u_diff"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {csv_path}.")

    metadata = get_metadata(csv_path)
    load_increment = float(metadata["loadIncrement"])
    if not np.isfinite(load_increment) or load_increment <= 0:
        raise ValueError(f"Invalid loadIncrement={load_increment} for {csv_path}.")
    yield_load = findPrePostSplit(df=df)
    loads = df["load"].to_numpy(dtype=float)
    event_root = Path(RAW_DATA_PATH) / _job_name_from_csv_path(csv_path) / "data/reversibilityData"
    if not event_root.is_dir():
        return []

    records = []
    skipped_missing_states = 0
    for event_dir in sorted(path for path in event_root.iterdir() if path.is_dir()):
        match = _REVERSIBILITY_EVENT_PATTERN.fullmatch(event_dir.name)
        if match is None:
            raise ValueError(f"Unexpected reversibility event directory name: {event_dir}")
        start_load = float(match.group("load"))
        event_load = start_load + load_increment
        candidate_indices = np.flatnonzero(
            np.isclose(loads, event_load, rtol=1e-9, atol=max(1e-12, load_increment * 1e-6))
        )
        if candidate_indices.size != 1:
            raise RuntimeError(
                f"Could not uniquely map {event_dir.name} to a macro row in {csv_path}; "
                f"target load={event_load}, candidates={candidate_indices.tolist()}."
            )
        row = df.iloc[int(candidate_indices[0])]
        expected_reversible = match.group("kind") == "rev"
        actual_reversible = bool(int(row["is_reversible"]))
        if actual_reversible != expected_reversible:
            raise ValueError(
                f"Reversibility mismatch for {event_dir}: folder says "
                f"{expected_reversible}, macro row says {actual_reversible}."
            )
        rev_u = float(row["rev_u_diff"])
        if not np.isfinite(rev_u):
            raise ValueError(f"Non-finite rev_u_diff for {event_dir} in {csv_path}.")
        delta_u_R = _delta_u_relaxation(event_dir)
        if delta_u_R is None:
            skipped_missing_states += 1
            continue
        records.append(
            {
                "delta_u_R": delta_u_R,
                "delta_rev_u": rev_u,
                "post_yield": float(row["load"]) > yield_load,
                "reversible": expected_reversible,
            }
        )
    if skipped_missing_states:
        print(
            f"Skipped {skipped_missing_states} event directories without saved "
            f"affine/relaxed VTUs in {csv_path}."
        )
    return records


def plot_reversibility_delta_u_relaxation(
    configs,
    labels=None,
    *,
    show=False,
    save=True,
    name="reversibility_delta_u_R_vs_delta_rev_u",
):
    """Plot post-processed relaxation displacement against reversibility displacement."""
    paths, labels = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )
    if paths is None or len(paths) == 0:
        raise RuntimeError("No CSV paths found for the reversibility displacement plot.")
    if not isinstance(paths[0], (list, tuple, np.ndarray)):
        paths = [paths]
    if labels is None:
        labels = [[""] * len(group) for group in paths]
    elif not isinstance(labels[0], (list, tuple, np.ndarray)):
        labels = [labels]

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        raise RuntimeError("Matplotlib default color cycle is empty.")

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_groups = []
    for group_index, group_paths in enumerate(paths):
        group_records = []
        for csv_path in group_paths:
            group_records.extend(_reversibility_event_records(csv_path))
        if not group_records:
            continue
        color = default_colors[group_index % len(default_colors)]
        for post_yield, marker in ((False, "^"), (True, "o")):
            selected = [record for record in group_records if record["post_yield"] == post_yield]
            x_values = np.asarray([record["delta_rev_u"] for record in selected], dtype=float)
            y_values = np.asarray([record["delta_u_R"] for record in selected], dtype=float)
            valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
            if np.any(valid):
                ax.scatter(
                    x_values[valid],
                    y_values[valid],
                    marker=marker,
                    s=18,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.8,
                    alpha=0.2,
                )
        group_labels = labels[group_index] if group_index < len(labels) else []
        cleaned = [strip_seed_from_label(label) for label in group_labels if label]
        display_label = pretty_variant_label(cleaned[0]) if cleaned else ""
        plotted_groups.append(
            (color, display_label or (cleaned[0] if cleaned else f"group {group_index + 1}"))
        )

    if not plotted_groups:
        raise RuntimeError("No valid displacement pairs were found in the reversibility data.")

    color_handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor=color, markersize=6, label=label)
        for color, label in plotted_groups
    ]
    shape_handles = [
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="none",
               markeredgecolor="black", markersize=6, label="post-yield"),
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="none",
               markeredgecolor="black", markersize=6, label="pre-yield"),
    ]
    ax.set_xlabel(r"$\Delta_{\mathrm{rev}} \mathbf{u}$")
    ax.set_ylabel(r"$\Delta \mathbf{u}_R$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Relaxation displacement vs. reversibility displacement")
    ax.legend(
        handles=[
            Line2D([], [], linestyle="None", label="Settings (color)"),
            *color_handles,
            Line2D([], [], linestyle="None", label="Yield regime (shape)"),
            *shape_handles,
        ],
        loc="upper left",
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()
    if save:
        save_path = Path("Plots") / f"{name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

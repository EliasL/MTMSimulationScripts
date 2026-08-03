from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import matplotlib.pylab
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from matplotlib.ticker import FuncFormatter
import powerlaw
import json
from simplification.cutil import simplify_coords_vwp
from tqdm import tqdm
from pathlib import Path
from .dataFunctions import (
    VTUData,
    get_data_from_name,
    extract_force_contribution_magnitude_series,
    extract_true_force_contribution_magnitude_series,
    resolve_vtu_files,
)
from .energyDropCalculations import (
    calculate_energy_step_data,
    infer_plastic_event_column,
)
from Management.updateCSV import update_df_header, read_macrodata_csv
from collections import defaultdict

# in Plotting/makePlots.py


def safePath(path):
    safe_path = (
        path.replace(" ", "_")
        .replace("$", "")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace("__", "_")
        .replace("_-_", "-")
        .replace("mathrm", "")
    )
    # Only strip ".00" when it is a trailing decimal (e.g. "1.00" -> "1"),
    # not when it is part of a longer decimal like "0.0001".
    safe_path = re.sub(r"(?<=\d)\.00(?!\d)", "", safe_path)
    return safe_path


USE_AVG_ENERGY = False


def maybe_avg(text, use_avg=None):
    if use_avg is None:
        use_avg = USE_AVG_ENERGY
    return rf"\langle {text} \rangle" if use_avg else text


def energy_drop_symbol(energy_type=None, stress_corrected=False):
    if stress_corrected:
        return "E_S"
    if energy_type is None:
        return "E"
    et = str(energy_type).lower()
    if "stress_corrected" in et:
        return "E_S"
    if "change_from_init" in et: 
        return "E_R" # Relaxation energy drop
    if "energy_change" in et or "inter-strain" in et or "inter_strain" in et:
        return "E_I"
    return "E"


def energy_drop_label(energy_type=None, stress_corrected=False, use_avg=None):
    return maybe_avg(energy_drop_symbol(energy_type, stress_corrected), use_avg)


def enable_strict_runtimewarnings():
    import warnings

    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    warnings.simplefilter("error", RuntimeWarning)


def duration_to_seconds(duration):
    try:
        return float(duration)
    except ValueError:
        pass
    # Create a mapping from unit to number of seconds
    unit_map = {
        "d": 86400,  # 24 hours * 3600 sec/hour
        "h": 3600,  # 60 minutes * 60 sec/minute
        "m": 60,  # 60 sec
        "s": 1,
    }

    total_seconds = 0
    # Split by space to handle multiple tokens like "1m 38s"
    parts = duration.split()
    for part in parts:
        # Last character is the unit, rest is the number
        number = float(part[:-1])
        unit = part[-1]
        # Convert and accumulate
        total_seconds += number * unit_map[unit]
    return total_seconds


def durations_to_seconds(durations):
    result = []
    for duration in durations:
        s = duration_to_seconds(duration)
        result.append(s)
    return result


def _format_seconds_dhm(seconds, _pos=None):
    if seconds is None or not np.isfinite(seconds):
        return ""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    total_minutes = int(seconds // 60)
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    if days > 0:
        return f"{sign}{days}d" if hours == 0 else f"{sign}{days}d {hours}h"
    if hours > 0:
        return f"{sign}{hours}h" if minutes == 0 else f"{sign}{hours}h {minutes}m"
    if minutes == 0 and seconds > 0:
        return f"{sign}<1m"
    return f"{sign}{minutes}m"


def _series_to_seconds(series):
    if series is None or len(series) == 0:
        return np.array([])
    first = series.iloc[0]
    if isinstance(first, str):
        return np.asarray(durations_to_seconds(series))
    return series.to_numpy(dtype=float)


def _estimate_total_runtime(df, csv_file_path):
    if "run_time" not in df.columns or df["run_time"].empty:
        return None, None
    run_time_sec = _series_to_seconds(df["run_time"])
    if run_time_sec.size == 0 or not np.isfinite(run_time_sec[-1]):
        return None, run_time_sec

    total_runtime = run_time_sec[-1]
    if "load" in df.columns and not df["load"].empty:
        meta = get_data_from_name(csv_file_path)
        start_load = meta.get("startLoad", df["load"].iloc[0])
        max_load = meta.get("maxLoad")
        if (
            max_load is not None
            and start_load is not None
            and np.isfinite(max_load)
            and np.isfinite(start_load)
        ):
            denom = max_load - start_load
            if denom > 0:
                progress = (df["load"].iloc[-1] - start_load) / denom
                if np.isfinite(progress) and progress > 0:
                    progress = min(progress, 1.0)
                    if progress < 1.0:
                        total_runtime = total_runtime / progress

    return total_runtime, run_time_sec


def plotYOverX(
    X,
    Y,
    fig=None,
    ax=None,
    indicateLastPoint=False,
    xlim=None,
    ylim=None,
    auto_scale="break",  # 'off' | 'log' | 'break'
    break_gap=None,  # (low_end, high_start) for the gap to skip
    sigma_thresh=40.0,
    **kwargs,
):
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")

    # crop + drop non-finite values
    mask = np.ones_like(X, dtype=bool)
    if xlim is not None:
        mask &= (X >= xlim[0]) & (X <= xlim[1])
    if ylim is not None:
        mask &= (Y >= ylim[0]) & (Y <= ylim[1])
    mask &= np.isfinite(X) & np.isfinite(Y)
    X = X[mask]
    Y = Y[mask]
    if X.size == 0:
        raise ValueError("No data after applying xlim/ylim.")

    Xs, Ys = X, Y  # (keep your simplification hook if you want)

    # --- auto detection of "spikes" (robust 6-sigma using MAD) ---
    def _spike_z(y):
        med = np.median(y)
        mad = np.median(np.abs(y - med))
        if not np.isfinite(mad) or mad == 0:
            return np.zeros_like(y)
        # For normal dist, std ≈ 1.4826 * MAD
        z = (y - med) / (1.4826 * mad)
        return z

    def has_spikes(y, k=sigma_thresh):
        z = _spike_z(y)
        if np.all(np.isnan(z)):
            return False
        return np.nanmax(z) > k

    if auto_scale == "break" and ax is not None and ax.get_yscale() == "log":
        # Already on log scale → skip broken axis
        auto_scale = "off"

    if auto_scale in ("log", "break") and has_spikes(Ys):
        if auto_scale == "log":
            # fall back to broken-axis if nonpositive values exist
            if np.any(Ys <= 0):
                auto_scale = "break"
            else:
                # single-axes, log scale
                if ax is None:
                    fig, ax = plt.subplots()
                (line,) = ax.plot(Xs, Ys, **kwargs)
                ax.set_yscale("log")
                if xlim:
                    ax.set_xlim(*xlim)
                if ylim:
                    ax.set_ylim(*ylim)
                point = None
                if indicateLastPoint and Xs.size > 0:
                    kw = {k: v for k, v in kwargs.items() if k != "label"}
                    if "alpha" in kw:
                        kw["alpha"] = min(kw["alpha"] * 1.5, 1.0)
                    point = ax.scatter(Xs[-1], Ys[-1], **kw)
                return fig, ax, line, point

        # --- broken y-axis ---
        # Use quantiles to focus the lower panel on the bulk of the data
        q_low, q_high, q_spike = 0.01, 0.99, 0.995
        y_low = np.quantile(Ys, q_low)
        if not np.isfinite(y_low):
            y_low = np.min(Ys)
        if break_gap is None:
            y_lo = np.quantile(Ys, q_high)
            y_hi = np.quantile(Ys, q_spike)
            if not np.isfinite(y_lo):
                y_lo = np.max(Ys)
            if not np.isfinite(y_hi) or y_hi <= y_lo:
                y_hi = np.max(Ys)
            break_gap = (y_lo, y_hi)

        if fig is None or ax is not None:
            fig = plt.figure()
        hRatio = 3
        gs = fig.add_gridspec(2, 1, height_ratios=[1, hRatio], hspace=0.05)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

        (line_bot,) = ax_bot.plot(Xs, Ys, **kwargs)
        (line_top,) = ax_top.plot(Xs, Ys, **kwargs)

        ymin, ymax = y_low, np.max(Ys)
        gap_low, gap_high = break_gap

        ax_bot.set_ylim(ymin, gap_low)
        ax_top.set_ylim(gap_high, ymax)

        if xlim:
            ax_top.set_xlim(*xlim)
        # hide spines between
        ax_top.spines.bottom.set_visible(False)
        ax_bot.spines.top.set_visible(False)
        ax_top.tick_params(labelbottom=False)  # no x-labels on top

        # after creating ax_top, ax_bot and setting GridSpec heights:
        fig.canvas.draw()  # ensure positions are finalized

        # same x offset for both; scale y offset for the shorter/taller axis
        dx = 0.015
        h_top = ax_top.get_position().height
        h_bot = ax_bot.get_position().height
        dy_top = 0.015
        dy_bot = dy_top * (h_top / h_bot)

        # draw diagonals (axes coordinates)
        kw_top = dict(transform=ax_top.transAxes, clip_on=False, color="k", lw=0.5)
        kw_bot = dict(transform=ax_bot.transAxes, clip_on=False, color="k", lw=0.5)

        # top axis (at its bottom edge)
        ax_top.plot((-dx, +dx), (-dy_top, +dy_top), **kw_top)
        ax_top.plot((1 - dx, 1 + dx), (-dy_top, +dy_top), **kw_top)

        # bottom axis (at its top edge) – note dy_bot
        ax_bot.plot((-dx, +dx), (1 - dy_bot, 1 + dy_bot), **kw_bot)
        ax_bot.plot((1 - dx, 1 + dx), (1 - dy_bot, 1 + dy_bot), **kw_bot)

        point = None
        if indicateLastPoint and Xs.size > 0:
            kw = {k: v for k, v in kwargs.items() if k != "label"}
            if "alpha" in kw:
                kw["alpha"] = min(kw["alpha"] * 1.5, 1.0)
            ax_top.scatter(Xs[-1], Ys[-1], **kw)
            ax_bot.scatter(Xs[-1], Ys[-1], **kw)

        # return the bottom axis for compatibility; you can also return both
        return fig, ax_bot, line_bot, point

    # --- regular single-axes plot (no special scaling) ---
    if ax is None:
        fig, ax = plt.subplots()
    (line,) = ax.plot(Xs, Ys, **kwargs)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    point = None
    if indicateLastPoint and Xs.size > 0:
        kw = {k: v for k, v in kwargs.items() if k != "label"}
        if "alpha" in kw:
            kw["alpha"] = min(kw["alpha"] * 1.5, 1.0)
        point = ax.scatter(Xs[-1], Ys[-1], **kw)
    return fig, ax, line, point


def time_to_seconds(duration_str):
    pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?"
    matches = re.match(pattern, duration_str.strip())

    if not matches:
        return timedelta()

    days, hours, minutes, seconds = matches.groups(default="0")

    return timedelta(
        days=int(days), hours=int(hours), minutes=int(minutes), seconds=float(seconds)
    ).seconds


def plotColumns(cvs_files, Y, labels, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    sigma_ax = ax.twinx()

    values = []
    for path in cvs_files:
        df = pd.read_csv(path)
        last_entry = df[Y].values[-1][0]
        last_entry_seconds = time_to_seconds(last_entry)
        values.append(last_entry_seconds)

    ax.bar(labels, values)
    return fig, ax


def plotRollingAverage(X, Y, intervalSize=100, fig=None, ax=None, **kwargs):
    # Calculate rolling average
    rollingMean = Y.rolling(window=intervalSize, min_periods=1, center=True).mean()

    # Check if axis provided, if not, create a new one
    if ax is None:
        fig, ax = plt.subplots()
    del kwargs["label"]
    # Plotting the interval average
    (line,) = ax.plot(
        X, rollingMean, label=f"Rolling average (window={intervalSize})", **kwargs
    )

    # Return the axis object for further use
    return fig, ax, line


# Define global variables
LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (1, 1)),  # Dotted
    (0, (5, 2, 1, 2)),  # Dash-dot variation
    (0, (3, 5, 1, 5)),  # Another custom
    (0, (2, 2, 3, 2)),  # Another variation
]
markers = ["o", "v", "^", "s", "D", "p", "*"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
colors = {"FIRE": "#d24646", "L-BFGS": "#008743", "CG": "#ffa701"}
colors = {"L-BFGS": "#3BCDDD", "CG": "#CDDD3B", "FIRE": "#DD3BCD"}
colors = {"L-BFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}
color_index = 0
index = 0


def _get_system_size_from_path(path):
    import re

    match = re.search(r"(\d+)x(\d+)", str(path))
    if not match:
        return None
    n1, n2 = match.groups()
    if n1 != n2:
        return None
    return int(n1)


def get_method(cvs_file_path):
    if isinstance(cvs_file_path, (list, tuple)):
        cvs_file_path = cvs_file_path[0]
    if isinstance(cvs_file_path, Path):
        cvs_file_path = str(cvs_file_path)
    if "minimizerFIRE" in cvs_file_path:
        return "FIRE"

    elif "minimizerCG" in cvs_file_path:
        return "CG"

    else:
        return "L-BFGS"


def plotEnergyAvalancheHistogram(dfs, fig=None, axs=None, label="", use_avg=None):
    if use_avg is None:
        if dfs and len(dfs) > 0:
            use_avg = "avg_energy" in dfs[0].columns
        else:
            use_avg = USE_AVG_ENERGY
    e = "avg_energy" if use_avg else "energy"
    drop_label = energy_drop_label("energy_change", use_avg=use_avg)
    pre_yield_df = [df[0 : np.argmax(df[e]) + 1] for df in dfs]
    post_yield_df = [df[np.argmax(df[e]) + 1 :] for df in dfs]

    # Prepare the figure and subplots for a 3x3 grid
    if axs is None:
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))  # Adjust size as necessary
        axs = axs.flatten()  # Flatten the array of axes for easier iteration

    min_group_index = 1
    max_group_index = 9  # This corresponds to 2^9 as the highest group (2^1 to 2^9)
    groups_indexes = range(min_group_index, max_group_index + 1)
    # Initialize a dictionary to store drops data for each group

    # Process each DataFrame split
    for split_dfs, label in zip(
        (pre_yield_df, post_yield_df),
        ("Pre yield", "Post yield"),
    ):
        groups_data = {
            i: [] for i in groups_indexes
        }  # Dictionary to store data for each group
        for df in split_dfs:
            # Filter out zero and NaN values
            plastic_col = infer_plastic_event_column(df)
            df = df[df[plastic_col] > 0]

            group_index = np.floor(np.log2(df[plastic_col])).astype(int)
            group_index = np.clip(
                group_index, min_group_index, max_group_index
            )  # Clamp the group index

            drops = -np.diff(df[e])

            # Filter out negative drops
            group_index = group_index[1:][drops > 0]
            drops = drops[drops > 0]

            # Aggregate drops data by group
            for i in groups_indexes:
                mask = group_index == i  # Apply mask to align with `diffs` length
                if any(mask):
                    groups_data[i].extend(drops[mask])

        # Define logarithmic bins

        # Now plot the aggregated data for each group
        for i, ax in enumerate(axs):
            exp = groups_indexes[i]
            if not groups_data[exp]:
                continue
            min_v = min(groups_data[exp])
            max_v = max(groups_data[exp])
            if min_v == max_v:
                continue
            bins = np.logspace(
                np.log10(min_v), np.log10(max_v), 20
            )  # Generate 20 logarithmic bins
            ax.hist(groups_data[exp], bins=bins, alpha=0.75, label=label)
            energyCutoffVisualization = 1e-5
            if ax.get_xlim()[0] < energyCutoffVisualization:
                ax.vlines(
                    energyCutoffVisualization,
                    ymin=0,
                    ymax=ax.get_ylim()[1],
                    color="#1f77b4",
                )

            ax.set_title(f"{2**exp}-{2 ** (exp + 1) - 1} p.e.")
            if i == len(axs) - 1:
                ax.set_title(f"More than {2**exp} p.e.")
            ax.set_yscale("log")
            ax.set_xscale("log")
            if i == 0:  # or i == len(axs) - 1:
                ax.legend()
            # Only allow a maximum of 3 ticks along x-axis
            # ax.xaxis.set_major_locator(MaxNLocator(3))
            # Remove axis names for inner axes
            if i % 3 == 0:  # Not the first column
                ax.set_ylabel(rf"$P>{drop_label}$")
            if i >= 6:  # Not the bottom row
                ax.set_xlabel(rf"$-\Delta {drop_label}$")

    return fig, axs


def getPrettyLabel(string):
    if not isinstance(string, str):
        return str(string)

    s = string
    eps_x = re.search(
        r"(?:LBFGSEpsx|LGBFSEpsx)=([^,]+)", string, flags=re.IGNORECASE
    )
    if eps_x:
        s = rf"$\epsilon_{{\mathbf{{x}}}}$: {eps_x.group(1)}"
    else:
        load_increment = re.search(r"loadIncrement=([^,]+)", string)
        if load_increment:
            s = rf"$\Delta \gamma$: {load_increment.group(1)}"
        elif "minimizer=" in string:
            s = string.split("minimizer=", 1)[1].split(",", 1)[0]
    s = s.replace("LBFGS", "L-BFGS")
    return s


def get_axis_labels(X, Y, x_name=None, y_name=None, use_y_axis_name=True, use_avg=None):
    """
    Determines appropriate axis labels based on given column names.
    """
    if x_name is None:
        x_name = r"Strain $\gamma$" if X == "load" else X

    if y_name is None and use_y_axis_name:
        # Remove total_ and avg_
        if use_avg is None:
            use_avg = isinstance(Y, str) and Y.startswith("avg_")
        Y_ = Y.replace("avg_", "").replace("total_", "")

        # Replace text with latex
        sigma12 = r"\sigma_{12}"
        p12 = r"P_{12}"
        y_labels_map = {
            "RSS": rf"Stress ${maybe_avg(p12, use_avg)}$",
            "P12": rf"Stress ${maybe_avg(p12, use_avg)}$",
            "sigma12": rf"Stress ${maybe_avg(sigma12, use_avg)}$",
            "energy": rf"Energy ${maybe_avg('E', use_avg)}$",
            "est_time_remaining": "Estimated time remaining (s)",
        }
        y_name = y_labels_map.get(Y_, Y)

    return x_name, y_name


def _coerce_series_numeric(series):
    if len(series) == 0:
        return series
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    if isinstance(non_null.iloc[0], str):
        return pd.Series(
            durations_to_seconds(series), index=series.index, name=series.name
        )
    return series


def _resolve_y_series(df, y_name):
    if y_name in df.columns:
        return _coerce_series_numeric(df[y_name])

    if not isinstance(y_name, str):
        raise KeyError(f"Unsupported Y type: {type(y_name)}")

    operators = [op for op in "+-*/" if op in y_name]
    operator_count = sum(y_name.count(op) for op in "+-*/")
    if operator_count != 1 or len(operators) != 1:
        raise KeyError(
            f"Y='{y_name}' not found. Expected exactly one operator expression like 'colA-colB'."
        )

    op = operators[0]
    left, right = [part.strip() for part in y_name.split(op, 1)]
    if not left or not right:
        raise KeyError(
            f"Invalid Y expression '{y_name}'. Expected format 'colA{op}colB'."
        )
    if left not in df.columns or right not in df.columns:
        missing = [name for name in (left, right) if name not in df.columns]
        raise KeyError(
            f"Could not evaluate Y='{y_name}'. Missing column(s): {', '.join(missing)}"
        )

    a = _coerce_series_numeric(df[left])
    b = _coerce_series_numeric(df[right])
    if op == "+":
        out = a + b
    elif op == "-":
        out = a - b
    elif op == "*":
        out = a * b
    elif op == "/":
        out = a / b
    else:
        raise KeyError(f"Unsupported operator '{op}' in Y expression '{y_name}'.")

    return pd.Series(out, index=df.index, name=y_name)


def makePlot(
    csv_file_paths,
    ax=None,
    fig=None,
    name="",
    Y="total_energy",
    X="load",
    x_name=None,
    y_name=None,
    use_y_axis_name=True,
    labels=None,
    use_title=False,
    title=None,
    plot_average=False,
    xlim=None,
    ylim=None,
    indicateLastPoint=False,
    plot_roll_average=False,
    plot_raw=True,
    plot_power_law=False,
    plot_columns=False,
    ylog=False,
    show=False,
    colors=None,
    linestyles=None,
    plot_total=False,
    legend=None,
    add_shift=False,
    add_images=False,
    metric="energy",
    image_pos=None,
    image_size=0.4,
    add_cbar=True,
    save=True,
    mark=None,
    mark_pos=(0.8, 0.95),
    mark_fontsize=20,
    legend_loc="best",
    plot_pre_yield=True,
    plot_post_yield=True,
    dist="truncated_power_law",
    reverse_x_axis=None,
    subtractFile=None,
):
    def _is_grouped_paths(paths):
        return (
            isinstance(paths, (list, tuple))
            and len(paths) > 0
            and isinstance(paths[0], (list, tuple))
        )

    if isinstance(csv_file_paths, (str, Path)):
        csv_file_paths = [csv_file_paths]

    if _is_grouped_paths(csv_file_paths):
        csv_file_paths = [p for group in csv_file_paths for p in group]
        if isinstance(labels, (list, tuple)) and _is_grouped_paths(labels):
            labels = [l for group in labels for l in group]

    if len(csv_file_paths) == 0:
        print("No files provided.")
        return

    x_name, y_name = get_axis_labels(X, Y, x_name, y_name, use_y_axis_name)

    # if we are not given a list, we make it into a list
    if isinstance(csv_file_paths, str):
        csv_file_paths = [csv_file_paths]

    if fig is None or ax is None:
        assert fig is None
        assert ax is None
        fig, ax = plt.subplots()

    lines = []
    data = []
    dfs = []
    xData = []
    y_targets = [Y] if isinstance(Y, str) else list(Y)

    sub_df = None
    if subtractFile is not None:
        if isinstance(subtractFile, str):
            sub_df = pd.read_csv(subtractFile)
        elif isinstance(subtractFile, int):
            sub_df = pd.read_csv(csv_file_paths[subtractFile])
        else:
            raise ValueError("Invalid type for subtractFile. Must be path or int.")

    line_index = 0
    for i, csv_file_path in enumerate(csv_file_paths):
        if i == subtractFile:
            continue
        if X is None:
            raise ValueError("X cannot be None in makePlot.")
        df = read_macrodata_csv(csv_file_path, L=_get_system_size_from_path(csv_file_path))
        for Y_ in y_targets:
            df[Y_] = _resolve_y_series(df, Y_)
        # Truncate data based on Lims
        # if xlim:
        #    df = df[(df[X] >= xlim[0]) & (df[X] <= xlim[1])]
        # if ylim:
        #    df = df[(df[Y] >= ylim[0]) & (df[Y] <= ylim[1])]

        dfs.append(df)

        if reverse_x_axis is None:
            # Check if we should reverse the x axis
            # We do this by checking which direction the X values go in by index
            if df[X][1] - df[X][0] < 0:
                reverse_x_axis = True
            else:
                reverse_x_axis = False

        if sub_df is not None:
            for Y_ in y_targets:
                sub_y = _resolve_y_series(sub_df, Y_)
                df[Y_] -= sub_y.reindex(df.index, fill_value=0)

        data.append(df[Y].values)
        xData.append(df[X].values)

        kwargs = {"fig": fig, "ax": ax, "indicateLastPoint": indicateLastPoint}

        if colors:
            if isinstance(colors, list):
                kwargs["color"] = colors[i]
            else:
                kwargs["color"] = colors

        if linestyles:
            if isinstance(linestyles, bool):
                kwargs["linestyle"] = LINE_STYLES[line_index]
                line_index += 1
                if line_index >= len(LINE_STYLES):
                    line_index = 0
            else:
                kwargs["linestyle"] = linestyles[i]

        for Y_ in y_targets:
            if len(df[Y_]) == 0:
                continue
            if labels is None:
                kwargs["label"] = Y_
            else:
                kwargs["label"] = labels[i]  # getPrettyLabel(labels[i])
                # +((" - " + Y_) if not isinstance(Y, str) else "")
            if add_shift:
                df[Y_] -= i * np.max(df[Y_]) / 500
            line = None
            point = None
            if plot_raw:
                fig, ax, line, point = plotYOverX(df[X], df[Y_], **kwargs)
                if Y_ == "est_time_remaining":
                    total_runtime, run_time_sec = _estimate_total_runtime(
                        df, csv_file_path
                    )
                    if total_runtime is not None and run_time_sec is not None:
                        x_vals = df[X].to_numpy()
                        n = min(len(x_vals), len(run_time_sec))
                        if n > 0:
                            true_remaining = total_runtime - run_time_sec[:n]
                            true_remaining = np.maximum(true_remaining, 0)
                            if labels is None:
                                true_label = "True remaining"
                            else:
                                true_label = f"{labels[i]} (true)"
                            ax.plot(
                                x_vals[:n],
                                true_remaining,
                                linestyle="--",
                                color=kwargs.get("color", None),
                                alpha=0.7,
                                label=true_label,
                            )
            if plot_roll_average:
                fig, ax, line = plotRollingAverage(df[X], df[Y_], **kwargs)
            if line is not None:
                lines.append(line)
            if point is not None:
                lines.append(point)
        if plot_total:
            assert not isinstance(Y, str)
            if not plot_raw:
                kwargs["label"] = "total"
            fig, ax, line, point = plotYOverX(
                df[X], sum([df[Y_] for Y_ in Y]), **kwargs
            )
            lines.append(line)

    if plot_columns:
        fig, ax = plotColumns(csv_file_paths, Y, labels, fig, ax)

    if plot_average:
        # Determine the maximum length among all arrays
        max_length_index = np.argmax([len(d) for d in data])
        max_length = len(data[max_length_index])

        # Initialize the average array and a count array to track how many entries per index
        average = np.zeros(max_length)
        count = np.zeros(max_length)

        # Aggregate data
        for d in data:
            length = len(d)
            average[:length] += d
            count[:length] += 1

        # Compute average where count is non-zero to avoid division by zero
        average = np.divide(
            average, count, out=np.zeros_like(average), where=count != 0
        )

        kwargs = {"fig": fig, "ax": ax, "label": "Average", "color": "black"}
        if plot_raw:
            fig, ax, line, point = plotYOverX(
                xData[max_length_index], average, **kwargs
            )
        lines.append(line)

    # cursor.connect(
    #   "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    if ylog:
        ax.set_yscale("log")
    if isinstance(Y, str) and "time" in Y.lower() and not ylog:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_seconds_dhm))

    if xlim:
        if len(xlim) == 1:
            ax.set_xlim(xmin=xlim[0])
        else:
            ax.set_xlim(*xlim)
        # Update y scale
        ax.relim()  # recompute data bounds based on all artists
        ax.autoscale_view(scalex=False, scaley=True)  # leave x alone, rescale y

    if ylim:
        ax.set_ylim(*ylim)

    # Create a list of line plots only for the legend
    handles, labels = ax.get_legend_handles_labels()
    line_handles = [
        handle for handle in handles if isinstance(handle, matplotlib.lines.Line2D)
    ]

    # Filter labels accordingly
    line_labels = [
        label
        for handle, label in zip(handles, labels)
        if isinstance(handle, matplotlib.lines.Line2D)
    ]

    # Set the legend with the filtered handles and labels
    if legend and not isinstance(legend, str):
        ax.legend(line_handles, line_labels, loc=legend_loc)

    elif isinstance(legend, str):
        ax.legend(line_handles, [legend], loc=legend_loc)
    if add_images:
        i = 0
        addImagesToPlot(
            ax,
            fig,
            csv_file_paths[i],
            xData[i],
            data[i],
            image_pos,
            image_size,
            mesh_property=metric,
            add_cbar=add_cbar,
        )
    if reverse_x_axis:
        print("Reversing x-axis!")
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_max, x_min)  # Swap the limits

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if use_title:
        if title is None:
            ax.set_title(f"{y_name} over {x_name}")
        else:
            ax.set_title(title)

    if mark:
        assert mark_pos is not None
        add_mark(ax, f"({mark})", *mark_pos, fontsize=mark_fontsize)

    if save:
        fig.tight_layout()

        name = safePath(name)
        figPath = os.path.join(os.path.dirname(csv_file_paths[0]), name)
        fig.savefig(figPath)
        print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()
    return fig, ax


def add_mark(ax, mark, x, y, color="black", fontsize=30):
    # Adding LaTeX-style bold font using \textbf{}
    ax.text(
        x,
        y,
        r"$\textbf{" + mark + "}$",  # LaTeX syntax for bold
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        color=color,
    )


def _extract_group_values_from_label(label):
    text = str(label)
    if not text:
        return None, None
    clean = text.replace("$", "").replace("\\", "")
    number = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    gp1_patterns = [
        rf"gamma_?0\s*=\s*{number}",
        rf"\bgp1\s*=\s*{number}",
    ]
    gp2_patterns = [
        rf"\bd\s*=\s*{number}",
        rf"\bgp2\s*=\s*{number}",
    ]

    def _find_value(patterns):
        for pattern in patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match is None:
                continue
            try:
                return float(match.group(1))
            except ValueError:
                continue
        return None

    return _find_value(gp1_patterns), _find_value(gp2_patterns)


def _build_auto_series_styles(labels):
    parsed = [_extract_group_values_from_label(label) for label in labels]
    gp1_values = sorted({gp1 for gp1, _ in parsed if gp1 is not None})
    gp2_values = sorted({gp2 for _, gp2 in parsed if gp2 is not None})
    if not gp1_values and not gp2_values:
        return None, None

    marker_pool = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_by_gp1 = {
        gp1: marker_pool[idx % len(marker_pool)] for idx, gp1 in enumerate(gp1_values)
    }

    color_pool = [plt.cm.tab10(i % 10) for i in range(max(10, len(gp2_values)))]
    color_by_gp2 = {
        gp2: color_pool[idx % len(color_pool)] for idx, gp2 in enumerate(gp2_values)
    }

    colors = []
    markers = []
    for gp1, gp2 in parsed:
        colors.append(color_by_gp2.get(gp2, None))
        markers.append(marker_by_gp1.get(gp1, "o"))
    return colors, markers


def plot_force_contribution_magnitudes(
    vtu_sources,
    labels=None,
    fig=None,
    ax=None,
    name="force_contribution_magnitude_vs_strain.pdf",
    show=False,
    save=True,
    use_tqdm=False,
    plot_mode="line",
    series_colors=None,
    series_markers=None,
    marker_size=26,
    connect_points=True,
    auto_style=True,
    compare_true_force=True,
    **kwargs,
):
    """
    Plot average element-force-contribution magnitude vs strain from VTU files.

    vtu_sources can be:
    - path to simulation folder (with collection.pvd or data/*.vtu)
    - path to .pvd
    - list of .vtu paths (single curve)
    - list of the above (multiple curves)
    """

    if isinstance(vtu_sources, (str, Path)):
        sources = [vtu_sources]
    elif (
        isinstance(vtu_sources, (list, tuple, np.ndarray))
        and len(vtu_sources) > 0
        and all(str(item).endswith(".vtu") for item in vtu_sources)
    ):
        sources = [list(vtu_sources)]
    else:
        sources = list(vtu_sources)

    if labels is None:
        labels = []
        for src in sources:
            if isinstance(src, (list, tuple, np.ndarray)):
                src0 = Path(src[0]) if len(src) > 0 else Path("series")
                labels.append(src0.parent.parent.name if src0.parent.name == "data" else src0.parent.name)
            else:
                labels.append(Path(src).stem)

    if len(labels) != len(sources):
        raise ValueError("Length of labels must match number of VTU source series.")

    if fig is None or ax is None:
        assert fig is None and ax is None
        if compare_true_force:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))
        else:
            fig, ax = plt.subplots()
            axes = [ax]
    else:
        if compare_true_force:
            raise ValueError(
                "compare_true_force=True requires fig and ax to be omitted."
            )
        axes = [ax]

    plot_mode = str(plot_mode).lower()
    if plot_mode not in {"line", "scatter"}:
        raise ValueError("plot_mode must be 'line' or 'scatter'.")

    def _resolve_style(spec, idx, label, default=None):
        if spec is None:
            return default
        if isinstance(spec, dict):
            return spec.get(label, default)
        if isinstance(spec, (list, tuple, np.ndarray)):
            if len(spec) == 0:
                return default
            return spec[idx % len(spec)]
        return spec

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    auto_colors = None
    auto_markers = None
    if auto_style:
        auto_colors, auto_markers = _build_auto_series_styles(labels)
    n_series = len(sources)
    if n_series <= 1:
        marker_sizes = np.array([float(marker_size)], dtype=float)
    else:
        size_scales = np.linspace(1.5, 0.5, n_series)
        marker_sizes = float(marker_size) * size_scales

    def _plot_series(ax_obj, xvals, yvals, *, label, color, marker, size, zorder):
        if plot_mode == "scatter":
            if connect_points:
                ax_obj.plot(
                    xvals,
                    yvals,
                    color=color,
                    linewidth=0.9,
                    alpha=0.35,
                    label="_nolegend_",
                    zorder=zorder - 0.1,
                )
            ax_obj.scatter(
                xvals,
                yvals,
                label=label,
                marker=marker,
                s=float(size),
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                zorder=zorder,
                **kwargs,
            )
        else:
            line_kwargs = dict(kwargs)
            line_kwargs.setdefault("color", color)
            line_kwargs.setdefault("marker", marker)
            line_kwargs.setdefault("markerfacecolor", "none")
            line_kwargs.setdefault("markeredgecolor", color)
            ax_obj.plot(xvals, yvals, label=label, **line_kwargs)[0]

    for idx, (source, label) in enumerate(zip(sources, labels)):
        strain_umut, mean_umut, _ = extract_force_contribution_magnitude_series(
            source, use_tqdm=use_tqdm
        )
        strain_umut = np.asarray(strain_umut[1:], dtype=float)
        mean_umut = np.asarray(mean_umut[1:], dtype=float)

        if compare_true_force:
            strain_true, mean_true, _ = extract_true_force_contribution_magnitude_series(
                source, use_tqdm=use_tqdm
            )
            strain_true = np.asarray(strain_true[1:], dtype=float)
            mean_true = np.asarray(mean_true[1:], dtype=float)

        default_color = default_colors[idx % len(default_colors)] if default_colors else None
        color = _resolve_style(series_colors, idx, label, default=None)
        if color is None and auto_colors is not None:
            color = auto_colors[idx]
        if color is None:
            color = default_color

        marker = _resolve_style(series_markers, idx, label, default=None)
        if marker is None and auto_markers is not None:
            marker = auto_markers[idx]
        if marker is None:
            marker = "o"

        series_zorder = 2 + idx
        _plot_series(
            axes[0],
            strain_umut,
            mean_umut,
            label=label,
            color=color,
            marker=marker,
            size=marker_sizes[idx],
            zorder=series_zorder,
        )

        if compare_true_force:
            _plot_series(
                axes[1],
                strain_true,
                mean_true,
                label=label,
                color=color,
                marker=marker,
                size=marker_sizes[idx],
                zorder=series_zorder,
            )

    axes[0].set_xlabel(r"Strain $\gamma$")
    axes[0].set_ylabel(r"$\langle |F_{ei}| \rangle$")
    axes[0].legend(loc="best")
    if compare_true_force:
        axes[0].set_title("Umut F")
        axes[1].set_title("True F")
        axes[1].set_xlabel(r"Strain $\gamma$")

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            base = sources[0][0] if isinstance(sources[0], (list, tuple, np.ndarray)) else sources[0]
            base_path = Path(base)
            if base_path.is_file():
                output_path = base_path.parent / output_path
            else:
                output_path = base_path / output_path
        fig.tight_layout()
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')

    if show:
        plt.show()
    if compare_true_force:
        return fig, axes
    return fig, axes[0]


def compute_predicted_next_energy(csv_file_path):
    r"""
    Compute predicted next-step total energy using first- and second-order
    simple-shear Taylor approximations:

        \widehat E_{n+1}
            = E_n
            + V_0 * <sigma_{12}>_n * delta_gamma_n
        \widehat E_{n+1}
            = E_n
            + V_0 * <sigma_{12}>_n * delta_gamma_n
            + 0.5 * V_0 * \mathfrak{a}_{1212,n}
              * delta_gamma_n^2
        \widehat E_{n+1}(\mathfrak{a}_{1212}(0))
            = E_n
            + V_0 * <sigma_{12}>_n * delta_gamma_n
            + 0.5 * V_0 * \mathfrak{a}_{1212}(0)
              * delta_gamma_n^2

    The Cauchy term approximates the generalized stress conjugate to MTS2D's
    left-multiplicative affine shear. A is dP/dF evaluated along
    F = [[1, gamma], [0, 1]], and the result is compared to measured E.
    """
    return calculate_energy_step_data(csv_file_path, average_energy=None)


def plot_predicted_energy_error(
    csv_file_paths,
    labels=None,
    fig=None,
    ax=None,
    name="pristine_crystal_energy_prediction_error.pdf",
    error_metric="prediction_error",
    property_keys=("L", "loadIncrement"),
    use_color_matrix_legend=True,
    y_log=True,
    strain_lim=(None, None),
    show_piola=False,
    show_first_order_reference=False,
    first_order_alpha=0.2,
    reference_prediction=None,
    reference_alpha=None,
    show_reference_line=False,
    x_column="load_ip1",
    show=False,
    save=True,
    legend_title=None,
    normalize_by_reference_volume=False,
    figsize=None,
    show_title=True,
):
    if isinstance(csv_file_paths, (str, Path)):
        csv_paths = [str(csv_file_paths)]
    else:
        csv_paths = [str(path) for path in csv_file_paths]
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    if labels is None:
        labels = [Path(path).parent.name for path in csv_paths]
    else:
        labels = list(labels)
    if len(labels) != len(csv_paths):
        raise ValueError("Length of labels must match length of csv_file_paths.")

    allowed_metrics = {
        "prediction_error",
        "abs_prediction_error",
        "relative_prediction_error",
        "second_order_prediction_error",
        "abs_second_order_prediction_error",
        "relative_second_order_prediction_error",
        "second_order_gamma0_prediction_error",
        "abs_second_order_gamma0_prediction_error",
        "relative_second_order_gamma0_prediction_error",
    }
    if error_metric not in allowed_metrics:
        raise ValueError(
            f"Unknown error_metric '{error_metric}'. Expected one of {sorted(allowed_metrics)}."
        )

    if normalize_by_reference_volume and error_metric.startswith("relative_"):
        raise ValueError(
            "Relative prediction errors are already dimensionless and cannot be "
            "normalized by reference volume."
        )

    reference_aliases = {
        "none": None,
        "off": None,
        "false": None,
        "first": "first_order",
        "first_order": "first_order",
        "gamma0": "second_order_gamma0",
        "second_order_gamma0": "second_order_gamma0",
        "second_order_at_gamma0": "second_order_gamma0",
    }
    if reference_prediction is not None:
        reference_key = str(reference_prediction).lower()
        if reference_key not in reference_aliases:
            raise ValueError(
                "reference_prediction must be one of "
                f"{sorted(key for key in reference_aliases if key not in {'false', 'off'})}."
            )
        reference_prediction = reference_aliases[reference_key]
    if show_first_order_reference and reference_prediction is None:
        reference_prediction = "first_order"
    if reference_alpha is None:
        reference_alpha = first_order_alpha

    x_label_map = {
        "load_i": r"Strain $\gamma$",
        "load_ip1": r"Strain $\gamma$",
    }
    if x_column not in x_label_map:
        raise ValueError(f"x_column must be one of {sorted(x_label_map)}.")

    if strain_lim is not None:
        if len(strain_lim) != 2:
            raise ValueError("strain_lim must be None or a two-value sequence.")
        strain_min, strain_max = strain_lim
    else:
        strain_min, strain_max = None, None

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    piola_ax = ax.twinx() if show_piola else None

    metric_label_map = {
        "prediction_error": r"$-\Delta E_S$",
        "abs_prediction_error": r"$|\Delta E_S|$",
        "relative_prediction_error": r"$|\Delta E_S|/|E|$",
        "second_order_prediction_error": r"$-\Delta E_S$",
        "abs_second_order_prediction_error": r"$|\Delta E_S|$",
        "relative_second_order_prediction_error": r"$|\Delta E_S|/|E|$",
        "second_order_gamma0_prediction_error": r"$-\Delta E_S(\mathfrak{a}_{1212}(0))$",
        "abs_second_order_gamma0_prediction_error": r"$|\Delta E_S(\mathfrak{a}_{1212}(0))|$",
        "relative_second_order_gamma0_prediction_error": r"$|\Delta E_S(\mathfrak{a}_{1212}(0))|/|E|$",
    }

    reference_metric_maps = {
        "first_order": {
            "second_order_prediction_error": "prediction_error",
            "abs_second_order_prediction_error": "abs_prediction_error",
            "relative_second_order_prediction_error": "relative_prediction_error",
            "second_order_gamma0_prediction_error": "prediction_error",
            "abs_second_order_gamma0_prediction_error": "abs_prediction_error",
            "relative_second_order_gamma0_prediction_error": "relative_prediction_error",
        },
        "second_order_gamma0": {
            "second_order_prediction_error": "second_order_gamma0_prediction_error",
            "abs_second_order_prediction_error": "abs_second_order_gamma0_prediction_error",
            "relative_second_order_prediction_error": "relative_second_order_gamma0_prediction_error",
        },
    }
    reference_text_map = {
        "first_order": "Transparent markers:\nfirst-order approximation",
        "second_order_gamma0": r"Transparent markers: second order with $\mathfrak{a}_{1212}(0)$",
    }
    signed_error_metrics = {
        "prediction_error",
        "second_order_prediction_error",
        "second_order_gamma0_prediction_error",
    }

    def _plot_values(prediction_df, metric):
        y_raw = np.asarray(prediction_df[metric], dtype=float)
        if y_log and metric in signed_error_metrics:
            return np.abs(y_raw)
        return y_raw

    def _sample_points(mask, delta_gamma):
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return indices

        positive_steps = delta_gamma[np.isfinite(delta_gamma) & (delta_gamma > 0)]
        if positive_steps.size == 0:
            return indices

        representative_step = float(np.median(positive_steps))
        target = max(1, int(round(20 * -np.log(representative_step))))
        retained_fraction = target / indices.size
        # Near-complete downsampling creates conspicuous isolated holes while
        # saving almost no markers. Keep all points unless at least 20% can be
        # removed.
        if retained_fraction >= 0.8:
            return indices

        sampled = np.linspace(0, indices.size - 1, target)
        return indices[np.unique(np.round(sampled).astype(int))]

    colors = [None] * len(csv_paths)
    marker_pool = ["o", "s", "^", "v", "D", "p", "h", "<", ">"]
    markers = [marker_pool[i % len(marker_pool)] for i in range(len(csv_paths))]
    marker_matrix = None
    if use_color_matrix_legend:
        if property_keys is None or len(property_keys) == 0:
            raise ValueError(
                "property_keys must be provided when use_color_matrix_legend=True."
            )
        prop1_values, prop2_values, _ = parse_labels(labels, property_keys)
        if np.any(pd.isna(prop1_values)):
            raise ValueError(
                f"Could not parse key '{property_keys[0]}' from all labels."
            )
        if len(property_keys) > 1 and np.any(pd.isna(prop2_values)):
            raise ValueError(
                f"Could not parse key '{property_keys[1]}' from all labels."
            )
        color_matrix, unique_p1, unique_p2 = create_color_matrix(
            prop1_values,
            prop2_values,
        )
        marker_matrix = np.empty(color_matrix.shape[:2], dtype=object)
        marker_matrix[:] = None
        cell_markers = {}
        for i in range(len(csv_paths)):
            row = (
                np.where(unique_p2 == prop2_values[i])[0][0]
                if unique_p2 is not None
                else 0
            )
            col = np.where(unique_p1 == prop1_values[i])[0][0]
            colors[i] = color_matrix[row, col]
            cell = (row, col)
            if cell not in cell_markers:
                cell_markers[cell] = marker_pool[len(cell_markers) % len(marker_pool)]
            markers[i] = cell_markers[cell]
            marker_matrix[row, col] = markers[i]

    plot_data = []
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        prediction_df, prediction_info = compute_predicted_next_energy(csv_path)
        error_scale = (
            float(prediction_info["reference_volume"])
            if normalize_by_reference_volume
            else 1.0
        )
        if not np.isfinite(error_scale) or error_scale <= 0:
            raise ValueError(
                f"Invalid reference volume {error_scale!r} for {csv_path}."
            )
        x = np.asarray(prediction_df[x_column], dtype=float)
        y = _plot_values(prediction_df, error_metric) / error_scale
        cauchy_stress = np.asarray(prediction_df["sigma12_i"], dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if strain_min is not None:
            finite &= x >= strain_min
        if strain_max is not None:
            finite &= x <= strain_max
        if y_log:
            finite &= y > 0
        if not np.any(finite):
            print(
                f"No plottable points for '{error_metric}' in {csv_path} "
                f"(after strain_lim={strain_lim} and y_log={y_log} filtering). Skipping."
            )
            continue

        reference_metric = None
        if reference_prediction is not None:
            reference_metric = reference_metric_maps[reference_prediction].get(
                error_metric
            )
        reference_y = None
        reference_indices = None
        if reference_metric is not None:
            reference_y = (
                _plot_values(prediction_df, reference_metric) / error_scale
            )
            reference_finite = np.isfinite(x) & np.isfinite(reference_y)
            if strain_min is not None:
                reference_finite &= x >= strain_min
            if strain_max is not None:
                reference_finite &= x <= strain_max
            if y_log:
                reference_finite &= reference_y > 0
            if np.any(reference_finite):
                reference_indices = _sample_points(
                    reference_finite,
                    np.asarray(prediction_df["delta_gamma"], dtype=float),
                )

        if show_piola:
            # ``show_piola`` is retained for API compatibility; the predictor
            # and this auxiliary curve now use averaged Cauchy shear stress.
            stress_finite = np.isfinite(x) & np.isfinite(cauchy_stress)
            if strain_min is not None:
                stress_finite &= x >= strain_min
            if strain_max is not None:
                stress_finite &= x <= strain_max
            if np.any(stress_finite):
                piola_ax.plot(
                    x[stress_finite],
                    cauchy_stress[stress_finite],
                    color="0.45",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.65,
                    label="_nolegend_",
                )

        plot_indices = _sample_points(
            finite, np.asarray(prediction_df["delta_gamma"], dtype=float)
        )
        color = colors[i] if colors[i] is not None else f"C{i % 10}"
        legend_label = getPrettyLabel(label) if label else Path(csv_path).parent.name
        plot_data.append(
            {
                "x": x,
                "y": y,
                "plot_indices": plot_indices,
                "reference_y": reference_y,
                "reference_indices": reference_indices,
                "color": color,
                "marker": markers[i],
                "legend_label": legend_label,
            }
        )

    for item in plot_data:
        if item["reference_y"] is None or item["reference_indices"] is None:
            continue
        if show_reference_line:
            ax.plot(
                item["x"][item["reference_indices"]],
                item["reference_y"][item["reference_indices"]],
                color=item["color"],
                linestyle="--",
                linewidth=0.8,
                alpha=reference_alpha,
                zorder=1,
                label="_nolegend_",
            )
        ax.scatter(
            item["x"][item["reference_indices"]],
            item["reference_y"][item["reference_indices"]],
            label="_nolegend_",
            marker=item["marker"],
            s=22,
            facecolors="none",
            edgecolors=item["color"],
            linewidths=0.9,
            alpha=reference_alpha,
            zorder=1,
        )

    for item in plot_data:
        ax.scatter(
            item["x"][item["plot_indices"]],
            item["y"][item["plot_indices"]],
            label=item["legend_label"],
            marker=item["marker"],
            s=22,
            facecolors="none",
            edgecolors=item["color"],
            linewidths=0.9,
            zorder=2,
        )

    if error_metric in signed_error_metrics and not y_log:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(x_label_map[x_column])
    if y_log and error_metric in signed_error_metrics:
        label_key = f"abs_{error_metric}"
        y_label = metric_label_map[label_key]
    else:
        y_label = metric_label_map[error_metric]
    if normalize_by_reference_volume:
        y_label = rf"${y_label[1:-1]}/V_0$"
    ax.set_ylabel(y_label)
    if y_log:
        ax.set_yscale("log")
    if show_piola:
        piola_ax.set_ylabel(r"$\langle\sigma_{12}\rangle$", color="0.35")
        piola_ax.tick_params(axis="y", colors="0.35")
    if "second_order_gamma0" in error_metric:
        title_formula = (
            r"$\widehat E_{n+1}(\mathfrak{a}_{1212}(0))="
            r"E_n+V_0\langle\sigma_{12}\rangle_n\Delta\gamma_n"
            r"+\frac{1}{2}V_0\mathfrak{a}_{1212}(0)(\Delta\gamma_n)^2$"
        )
    elif "second_order" in error_metric:
        title_formula = (
            r"$\widehat E_{n+1}=E_n"
            r"+V_0\langle\sigma_{12}\rangle_n\Delta\gamma_n"
            r"+\frac{1}{2}V_0\mathfrak{a}_{1212,n}(\Delta\gamma_n)^2$"
        )
    else:
        title_formula = (
            r"$\widehat E_{n+1}=E_n"
            r"+V_0\langle\sigma_{12}\rangle_n\Delta\gamma_n$"
        )
    title = (
        "Energy Prediction Error per Reference Area"
        if normalize_by_reference_volume
        else "Energy Prediction Error"
    )
    if show_title:
        ax.set_title(title + "\n" + title_formula)
    if reference_prediction is not None:
        ax.text(
            0.02,
            0.90,
            reference_text_map[reference_prediction],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color="0.35",
        )

    if use_color_matrix_legend:
        plot_color_matrix(
            ax,
            color_matrix,
            unique_p1,
            unique_p2,
            property_keys=property_keys,
            loc="upper left",
            bbox_to_anchor=(0.1, -0.05, 1, 1),
            marker_matrix=marker_matrix,
            marker_color="lower_left_white",
        )
    else:
        ax.legend(loc="upper right", title=legend_title)
    fig.tight_layout()

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            output_path = Path("Plots") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')

    if show:
        plt.show()
    return fig, ax


def plot_predicted_energy_error_distribution(
    csv_file_paths,
    *,
    name="predicted_energy_error_distribution.pdf",
    title=None,
    bins=120,
    drop_threshold=0.0,
    distance_steps=1,
    error_metric="second_order_prediction_error",
    normalize_by_reference_volume=True,
    pre_yield_min_gamma=None,
    figsize=(7.0, 4.8),
    show=False,
    save=True,
):
    """Plot probabilities per logarithmic bin for the second-order residual.

    The raw inter-load energy change is used to identify actual energy drops:
    transitions with ``total_energy_change < -drop_threshold`` are excluded.
    ``distance_steps=1`` retains the non-drop transition immediately before a
    drop; ``distance_steps=None`` retains all non-drop transitions.  Pre/post
    yield is classified from the strain at the maximum stored ``avg_sigma12``.

    The plotted residual is ``abs(Delta E_S)`` and is divided by the
    initial/reference volume of each run by default. If
    ``pre_yield_min_gamma`` is provided, pre-yield samples with
    ``load_i < pre_yield_min_gamma`` are excluded. The y values are normalized
    counts per logarithmic bin, not probability densities. The plotted curve
    connects the tops of adjacent bins.
    """
    if isinstance(csv_file_paths, (str, Path)):
        csv_paths = [Path(csv_file_paths)]
    else:
        csv_paths = [Path(path) for path in csv_file_paths]
    if not csv_paths:
        raise ValueError("No CSV paths provided.")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    if distance_steps is not None:
        if not isinstance(distance_steps, (int, np.integer)) or distance_steps < 1:
            raise ValueError("distance_steps must be None or a positive integer.")
    if not np.isfinite(drop_threshold) or drop_threshold < 0:
        raise ValueError("drop_threshold must be finite and nonnegative.")
    if error_metric not in {
        "second_order_prediction_error",
        "abs_second_order_prediction_error",
        "second_order_gamma0_prediction_error",
        "abs_second_order_gamma0_prediction_error",
    }:
        raise ValueError(
            "error_metric must be a second-order prediction residual, got "
            f"{error_metric!r}."
        )
    if pre_yield_min_gamma is not None:
        if not np.isfinite(pre_yield_min_gamma) or pre_yield_min_gamma < 0:
            raise ValueError(
                "pre_yield_min_gamma must be None or finite and nonnegative."
            )

    regions = ("pre", "post")
    distances = (None,) if distance_steps is None else (None, distance_steps)
    values = {(region, distance): [] for region in regions for distance in distances}
    counts = {(region, distance): 0 for region in regions for distance in distances}

    for csv_path in csv_paths:
        prediction_df, prediction_info = compute_predicted_next_energy(csv_path)
        raw_df = read_macrodata_csv(csv_path)
        load = np.asarray(raw_df["load"], dtype=float)
        load_i = np.asarray(prediction_df["load_i"], dtype=float)
        if len(load) != len(prediction_df) + 1:
            raise ValueError(
                f"Unexpected row alignment in {csv_path}: raw data has {len(load)} "
                f"rows, prediction data has {len(prediction_df)} steps."
            )

        if "total_energy_change" in raw_df:
            step_energy_change = np.asarray(
                raw_df["total_energy_change"], dtype=float
            )[1:]
        elif "total_energy" in raw_df:
            step_energy_change = np.diff(
                np.asarray(raw_df["total_energy"], dtype=float)
            )
        else:
            raise KeyError(
                f"{csv_path} has neither 'total_energy_change' nor 'total_energy'."
            )
        drop_mask = np.isfinite(step_energy_change) & (
            step_energy_change < -float(drop_threshold)
        )

        stress_col = "avg_sigma12" if "avg_sigma12" in raw_df else "avg_P12"
        if stress_col not in raw_df:
            raise KeyError(f"{csv_path} has no yield-stress column.")
        stress = np.asarray(raw_df[stress_col], dtype=float)
        finite_stress = np.isfinite(stress)
        if not np.any(finite_stress):
            raise ValueError(f"No finite yield-stress values found in {csv_path}.")
        yield_load = float(load[int(np.nanargmax(stress))])

        if distance_steps is None:
            distance_masks = {None: np.ones(len(prediction_df), dtype=bool)}
        else:
            distance_masks = {
                None: np.ones(len(prediction_df), dtype=bool),
                int(distance_steps): np.zeros(len(prediction_df), dtype=bool),
            }
            drop_indices = np.flatnonzero(drop_mask)
            targets = drop_indices - int(distance_steps)
            valid_targets = (targets >= 0) & (targets < len(prediction_df))
            distance_masks[int(distance_steps)][targets[valid_targets]] = True

        residual = np.abs(np.asarray(prediction_df[error_metric], dtype=float))
        if normalize_by_reference_volume:
            reference_volume = float(prediction_info["reference_volume"])
            if not np.isfinite(reference_volume) or reference_volume <= 0:
                raise ValueError(
                    f"Invalid initial volume {reference_volume!r} for {csv_path}."
                )
            residual /= reference_volume
        base_mask = np.isfinite(residual) & (residual > 0) & (~drop_mask)
        for region in regions:
            region_mask = (
                load_i < yield_load if region == "pre" else load_i >= yield_load
            )
            if region == "pre" and pre_yield_min_gamma is not None:
                region_mask &= load_i >= float(pre_yield_min_gamma)
            for distance in distances:
                mask = base_mask & region_mask & distance_masks[distance]
                values[(region, distance)].append(residual[mask])
                counts[(region, distance)] += int(np.count_nonzero(mask))

    for key in values:
        if values[key]:
            values[key] = np.concatenate(values[key])
        else:
            values[key] = np.empty(0, dtype=float)

    all_values = np.concatenate(
        [array for array in values.values() if array.size], dtype=float
    ) if any(array.size for array in values.values()) else np.empty(0, dtype=float)
    if all_values.size == 0:
        raise ValueError("No positive non-drop prediction errors remain after filtering.")
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= 0:
        raise ValueError("Prediction residuals must contain finite positive values.")
    if lower == upper:
        lower /= 2.0
        upper *= 2.0
    edges = np.geomspace(lower, upper, int(bins) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    styles = [
        ("pre", None, "Pre-yield, all non-drop", "C0"),
        ("post", None, "Post-yield, all non-drop", "C1"),
    ]
    if distance_steps is not None:
        styles.extend(
            [
                (
                    "pre",
                    distance_steps,
                    f"Pre-yield, {distance_steps}-step pre-drop",
                    "C2",
                ),
                (
                    "post",
                    distance_steps,
                    f"Post-yield, {distance_steps}-step pre-drop",
                    "C3",
                ),
            ]
        )
    positive_probabilities = []
    for region, distance, label, color in styles:
        sample = values[(region, distance)]
        if sample.size == 0:
            continue
        histogram, _ = np.histogram(sample, bins=edges)
        probability = histogram.astype(float) / float(sample.size)
        positive_probabilities.extend(probability[probability > 0].tolist())
        step_x = np.repeat(edges, 2)[1:-1]
        step_y = np.repeat(probability, 2)
        ax.plot(
            step_x,
            step_y,
            color=color,
            linewidth=1.8,
            label=f"{label} (N={sample.size:,})",
        )

    if not positive_probabilities:
        raise ValueError("All prediction-error histogram bins are empty.")
    ax.set_xscale("log")
    x_label = r"$|\Delta E_S|$"
    if normalize_by_reference_volume:
        x_label = r"$|\Delta E_S|/V_0$"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability per logarithmic bin")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            output_path = Path("Plots") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')
    if show:
        plt.show()

    summary = {
        "bins": edges,
        "counts": counts,
        "values": values,
        "yield_filter": "load_i below/at the maximum stored avg_sigma12",
        "drop_filter": "total_energy_change < -drop_threshold",
        "distance_filter": distance_steps,
        "error_metric": error_metric,
        "normalize_by_reference_volume": normalize_by_reference_volume,
        "pre_yield_min_gamma": pre_yield_min_gamma,
    }
    return fig, ax, summary


def plot_combined_predicted_energy_error_distribution(
    grouped_csv_file_paths,
    *,
    name="combined_predicted_energy_error_distribution.pdf",
    bins=60,
    drop_threshold=0.0,
    error_metric="second_order_prediction_error",
    normalize_by_reference_volume=True,
    pre_yield_min_gamma=None,
    main_xlim=None,
    main_ylim=None,
    full_loglog_inset=False,
    inset_bounds=(0.56, 0.36, 0.39, 0.48),
    figsize=(4.329, 2.808),
    show=False,
    save=True,
):
    """Compare pre/post-yield prediction errors for several simulation groups.

    ``grouped_csv_file_paths`` maps a legend label (for example, ``"No
    reconnection"``) to one or more macro-data CSV files.  Each group is
    filtered with the same non-drop, pre/post-yield rules as
    :func:`plot_predicted_energy_error_distribution`, with no special
    pre-drop subset.  The groups share one set of logarithmic bin edges so the
    four curves can be compared directly across the full data-derived x-range
    on a linear-probability y-axis.
    """
    if not hasattr(grouped_csv_file_paths, "items"):
        raise TypeError("grouped_csv_file_paths must be a mapping of labels to paths.")
    groups = []
    for label, csv_paths in grouped_csv_file_paths.items():
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        csv_paths = [Path(path) for path in csv_paths]
        if not csv_paths:
            raise ValueError(f"No CSV paths provided for group {label!r}.")
        temporary_fig, _, summary = plot_predicted_energy_error_distribution(
            csv_paths,
            bins=bins,
            drop_threshold=drop_threshold,
            distance_steps=None,
            error_metric=error_metric,
            normalize_by_reference_volume=normalize_by_reference_volume,
            pre_yield_min_gamma=pre_yield_min_gamma,
            show=False,
            save=False,
        )
        plt.close(temporary_fig)
        groups.append((str(label), summary))

    regions = ("pre", "post")
    all_values = [
        summary["values"][(region, None)]
        for _, summary in groups
        for region in regions
        if summary["values"][(region, None)].size
    ]
    if not all_values:
        raise ValueError("No positive prediction errors remain after filtering.")
    all_values = np.concatenate(all_values)
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= 0:
        raise ValueError("Prediction residuals must contain finite positive values.")
    if lower == upper:
        lower /= 2.0
        upper *= 2.0
    edges = np.geomspace(lower, upper, int(bins) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ("#9ecae1", "#2171b5", "#fdae6b", "#e6550d")
    counts = {}
    plot_data = []

    for group_index, (group_label, summary) in enumerate(groups):
        for region_index, region in enumerate(regions):
            sample = summary["values"][(region, None)]
            counts[(group_label, region)] = int(sample.size)
            if sample.size == 0:
                continue
            histogram, _ = np.histogram(sample, bins=edges)
            probability = histogram.astype(float) / float(sample.size)
            plot_data.append(
                (
                    group_label,
                    region,
                    sample,
                    probability,
                    colors[group_index * len(regions) + region_index],
                )
            )

    for group_label, region, sample, probability, color in plot_data:
        step_x = np.repeat(edges, 2)[1:-1]
        step_y = np.repeat(probability, 2)
        ax.plot(
            step_x,
            step_y,
            color=color,
            linewidth=1.2,
            label=f"{group_label}, {region}-yield",
        )

    ax.set_xscale("log")
    if main_xlim is not None:
        ax.set_xlim(*main_xlim)
    if main_ylim is not None:
        ax.set_ylim(*main_ylim)
    x_label = r"$|\Delta E_S|$"
    if normalize_by_reference_volume:
        x_label = r"$|\Delta E_S|/V_0$"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability per logarithmic bin")
    ax.legend(loc="upper left", fontsize=4.8, handlelength=1.43, borderpad=0.26)

    fig.tight_layout()
    if full_loglog_inset:
        inset_ax = ax.inset_axes(inset_bounds)
        for _, _, _, probability, color in plot_data:
            inset_ax.plot(
                np.repeat(edges, 2)[1:-1],
                np.repeat(probability, 2),
                color=color,
                linewidth=0.8,
            )
        inset_ax.set_xscale("log")
        inset_ax.set_yscale("log")
        positive_probability = np.concatenate(
            [probability[probability > 0] for _, _, _, probability, _ in plot_data]
        )
        if positive_probability.size:
            inset_ax.set_ylim(
                max(float(np.min(positive_probability)) * 0.5, 1.0e-12),
                min(float(np.max(positive_probability)) * 2.0, 1.0),
            )
        inset_ax.tick_params(axis="both", which="both", labelsize=4.2, pad=1.0)

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            output_path = Path("Plots") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')
    if show:
        plt.show()

    return fig, ax, {
        "bins": edges,
        "counts": counts,
        "groups": groups,
        "distance_filter": None,
        "normalize_by_reference_volume": normalize_by_reference_volume,
        "pre_yield_min_gamma": pre_yield_min_gamma,
        "main_xlim": main_xlim,
        "main_ylim": main_ylim,
        "full_loglog_inset": bool(full_loglog_inset),
    }


def plot_predicted_energy_error_distribution_with_reconnection_events(
    grouped_csv_file_paths,
    *,
    name="predicted_energy_error_with_reconnection_events.pdf",
    bins=120,
    drop_threshold=0.0,
    error_metric="second_order_prediction_error",
    normalize_by_reference_volume=True,
    pre_yield_min_gamma=None,
    reconnection_event_column="nr_total_edge_flips",
    show_reconnection_subsets=False,
    main_xlim=None,
    main_ylim=None,
    figsize=(4.329, 2.808),
    show=False,
    save=True,
):
    """Plot all non-drop errors plus reconnecting, increasing-energy steps.

    The four ordinary curves contain all non-drop transitions. If
    ``show_reconnection_subsets`` is enabled, reconnecting groups additionally
    show two dashed curves for transitions with
    ``reconnection_event_column > 0`` and ``total_energy_change > 0``, plus two
    dotted curves for non-drop transitions with
    ``reconnection_event_column == 0``.
    """
    if not hasattr(grouped_csv_file_paths, "items"):
        raise TypeError("grouped_csv_file_paths must be a mapping of labels to paths.")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    if pre_yield_min_gamma is not None and (
        not np.isfinite(pre_yield_min_gamma) or pre_yield_min_gamma < 0
    ):
        raise ValueError("pre_yield_min_gamma must be None or finite and nonnegative.")

    def collect(csv_paths, event_only=False, no_event_only=False):
        if event_only and no_event_only:
            raise ValueError("event_only and no_event_only are mutually exclusive.")
        values = {"pre": [], "post": []}
        for csv_path in csv_paths:
            prediction_df, prediction_info = compute_predicted_next_energy(csv_path)
            raw_df = read_macrodata_csv(csv_path)
            load = np.asarray(raw_df["load"], dtype=float)
            load_i = np.asarray(prediction_df["load_i"], dtype=float)
            if len(load) != len(prediction_df) + 1:
                raise ValueError(f"Unexpected row alignment in {csv_path}.")

            if "total_energy_change" in raw_df:
                step_energy_change = np.asarray(
                    raw_df["total_energy_change"], dtype=float
                )[1:]
            elif "total_energy" in raw_df:
                step_energy_change = np.diff(
                    np.asarray(raw_df["total_energy"], dtype=float)
                )
            else:
                raise KeyError(
                    f"{csv_path} has neither 'total_energy_change' nor 'total_energy'."
                )
            drop_mask = np.isfinite(step_energy_change) & (
                step_energy_change < -float(drop_threshold)
            )

            stress_col = "avg_sigma12" if "avg_sigma12" in raw_df else "avg_P12"
            if stress_col not in raw_df:
                raise KeyError(f"{csv_path} has no yield-stress column.")
            stress = np.asarray(raw_df[stress_col], dtype=float)
            finite_stress = np.isfinite(stress)
            if not np.any(finite_stress):
                raise ValueError(f"No finite yield-stress values found in {csv_path}.")
            yield_load = float(load[int(np.nanargmax(stress))])

            residual = np.abs(
                np.asarray(prediction_df[error_metric], dtype=float)
            )
            if normalize_by_reference_volume:
                reference_volume = float(prediction_info["reference_volume"])
                if not np.isfinite(reference_volume) or reference_volume <= 0:
                    raise ValueError(f"Invalid initial volume for {csv_path}.")
                residual /= reference_volume

            base_mask = np.isfinite(residual) & (residual > 0)
            if event_only:
                if reconnection_event_column not in raw_df:
                    raise KeyError(
                        f"{csv_path} has no {reconnection_event_column!r} column."
                    )
                event_mask = np.asarray(
                    raw_df[reconnection_event_column], dtype=float
                )[1:] > 0
                base_mask &= event_mask & (step_energy_change > 0)
            elif no_event_only:
                if reconnection_event_column not in raw_df:
                    raise KeyError(
                        f"{csv_path} has no {reconnection_event_column!r} column."
                    )
                event_mask = np.asarray(
                    raw_df[reconnection_event_column], dtype=float
                )[1:] > 0
                base_mask &= (~event_mask) & ~drop_mask
            else:
                base_mask &= ~drop_mask

            for region in ("pre", "post"):
                region_mask = (
                    load_i < yield_load if region == "pre" else load_i >= yield_load
                )
                if region == "pre" and pre_yield_min_gamma is not None:
                    region_mask &= load_i >= float(pre_yield_min_gamma)
                values[region].append(residual[base_mask & region_mask])

        for region in values:
            values[region] = (
                np.concatenate(values[region])
                if values[region]
                else np.empty(0, dtype=float)
            )
        return values

    def display_group_label(label):
        normalized = str(label).lower()
        if "no" in normalized and "recon" in normalized:
            return "No recon"
        if "recon" in normalized:
            return "Recon"
        return str(label)

    def is_reconnection_group(label):
        normalized = str(label).lower()
        return "recon" in normalized and "no" not in normalized

    def format_count(count):
        exponent = int(np.floor(np.log10(count)))
        mantissa = float(count) / (10.0**exponent)
        return rf"$n={mantissa:.1f}\times 10^{{{exponent}}}$"

    color_families = {
        "No recon": ("#9ecae1", "#2171b5"),
        "Recon": ("#fdae6b", "#e6550d"),
    }
    entries = []
    for label, csv_paths in grouped_csv_file_paths.items():
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        csv_paths = [Path(path) for path in csv_paths]
        if not csv_paths:
            raise ValueError(f"No CSV paths provided for group {label!r}.")
        display_label = display_group_label(label)
        colors = color_families.get(display_label, ("C0", "C1"))
        values = collect(csv_paths, event_only=False)
        for region_index, region in enumerate(("pre", "post")):
            entries.append(
                {
                    "label": f"{display_label}, {region}-yield",
                    "sample": values[region],
                    "color": colors[region_index],
                    "linestyle": "-",
                    "linewidth": 1.2,
                }
            )
        if show_reconnection_subsets and is_reconnection_group(label):
            event_values = collect(csv_paths, event_only=True)
            for region_index, region in enumerate(("pre", "post")):
                entries.append(
                    {
                        "label": f"Flips + increase, {region}-yield",
                        "sample": event_values[region],
                        "color": colors[region_index],
                        "linestyle": "--",
                        "linewidth": 1.0,
                    }
                )
            no_event_values = collect(csv_paths, no_event_only=True)
            for region_index, region in enumerate(("pre", "post")):
                entries.append(
                    {
                        "label": f"Recon, no flips, {region}-yield",
                        "sample": no_event_values[region],
                        "color": colors[region_index],
                        "linestyle": ":",
                        "linewidth": 1.0,
                    }
                )

    nonempty_entries = [entry for entry in entries if entry["sample"].size]
    if not nonempty_entries:
        raise ValueError("No positive prediction errors remain after filtering.")
    all_values = np.concatenate([entry["sample"] for entry in nonempty_entries])
    lower = float(np.nanmin(all_values))
    upper = float(np.nanmax(all_values))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= 0:
        raise ValueError("Prediction residuals must contain finite positive values.")
    if lower == upper:
        lower /= 2.0
        upper *= 2.0
    edges = np.geomspace(lower, upper, int(bins) + 1)

    plot_data = []
    for entry in nonempty_entries:
        histogram, _ = np.histogram(entry["sample"], bins=edges)
        probability = histogram.astype(float) / float(entry["sample"].size)
        plot_data.append((entry, probability))

    fig, ax = plt.subplots(figsize=figsize)
    for entry, probability in plot_data:
        ax.plot(
            np.repeat(edges, 2)[1:-1],
            np.repeat(probability, 2),
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=entry["linewidth"],
            label=f"{entry['label']} ({format_count(entry['sample'].size)})",
        )
    ax.set_xscale("log")
    if main_xlim is not None:
        ax.set_xlim(*main_xlim)
    if main_ylim is not None:
        ax.set_ylim(*main_ylim)
    x_label = r"$|\Delta E_S|$"
    if normalize_by_reference_volume:
        x_label = r"$|\Delta E_S|/V_0$"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability per logarithmic bin")
    ax.legend(loc="upper left", fontsize=8.0, handlelength=1.56, borderpad=0.26)

    fig.tight_layout()

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            output_path = Path("Plots") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')
    if show:
        plt.show()

    return fig, ax, {
        "bins": edges,
        "entries": entries,
        "event_column": reconnection_event_column,
        "main_xlim": main_xlim,
        "main_ylim": main_ylim,
    }


def plot_average_elastic_frobenius_norm_vs_gamma(
    grouped_vtu_file_paths,
    *,
    name="average_elastic_frobenius_norm_vs_gamma.pdf",
    energy_csv_file_paths=None,
    figsize=(3.5, 2.4),
    show=False,
    save=True,
):
    """Plot the element-average Frobenius norm of ``F_e = F M_e`` vs gamma.

    ``M_e`` is the elastic-domain reduction matrix returned by
    ``VTUData.get_M(elastic_M=True)``.  The norm is evaluated per element and
    then averaged, so each marker represents one VTU snapshot. The legend
    labels identify the reconnection condition; the sample seed is kept in
    the output filename rather than repeated in the legend.
    """
    if not hasattr(grouped_vtu_file_paths, "items"):
        raise TypeError("grouped_vtu_file_paths must be a mapping of labels to paths.")

    curves = []
    for label, vtu_paths in grouped_vtu_file_paths.items():
        if isinstance(vtu_paths, (str, Path)):
            vtu_paths = [vtu_paths]
        points = []
        for vtu_path in vtu_paths:
            vtu_path = Path(vtu_path)
            metadata = get_data_from_name(vtu_path)
            gamma = metadata.get("load")
            if gamma is None:
                raise ValueError(f"Could not infer gamma from VTU filename {vtu_path}.")
            data = VTUData(str(vtu_path))
            F = np.asarray(data.get_F(), dtype=float)
            M_elastic = np.asarray(data.get_M(elastic_M=True), dtype=float)
            if F.shape != M_elastic.shape or F.ndim != 3 or F.shape[-2:] != (2, 2):
                raise ValueError(
                    f"Unexpected F/M shape for {vtu_path}: {F.shape}, {M_elastic.shape}."
                )
            F_elastic = np.matmul(F, M_elastic)
            element_norm = np.linalg.norm(F_elastic, axis=(-2, -1))
            finite_norm = element_norm[np.isfinite(element_norm)]
            if finite_norm.size == 0:
                raise ValueError(f"No finite elastic norms found in {vtu_path}.")
            points.append((float(gamma), float(np.mean(finite_norm)), str(vtu_path)))
        if not points:
            raise ValueError(f"No VTU paths provided for group {label!r}.")
        points.sort(key=lambda point: point[0])
        curves.append((str(label), points))

    fig, ax = plt.subplots(figsize=figsize)
    summaries = {}
    curve_colors = {}
    for curve_index, (label, points) in enumerate(curves):
        gamma, mean_norm, paths = zip(*points)
        color = f"C{curve_index}"
        curve_colors[label] = color
        ax.plot(
            gamma,
            mean_norm,
            color=color,
            marker="o",
            linewidth=1.2,
            markersize=3.0,
            label=label,
        )
        summaries[label] = {
            "gamma": np.asarray(gamma),
            "mean_frobenius_norm": np.asarray(mean_norm),
            "paths": paths,
        }

    energy_ax = None
    if energy_csv_file_paths is not None:
        if not hasattr(energy_csv_file_paths, "items"):
            raise TypeError("energy_csv_file_paths must be a mapping of labels to CSV paths.")
        energy_ax = ax.twinx()
        energy_handles = []
        for label, csv_path in energy_csv_file_paths.items():
            csv_path = Path(csv_path)
            energy_df = read_macrodata_csv(csv_path)
            if "load" not in energy_df or "avg_energy" not in energy_df:
                raise KeyError(
                    f"{csv_path} must contain 'load' and 'avg_energy' columns."
                )
            load = np.asarray(energy_df["load"], dtype=float)
            energy = np.asarray(energy_df["avg_energy"], dtype=float)
            if load.shape != energy.shape:
                raise ValueError(f"Energy/load shape mismatch in {csv_path}.")
            stress_col = "avg_sigma12" if "avg_sigma12" in energy_df else "avg_P12"
            if stress_col not in energy_df:
                raise KeyError(f"{csv_path} has no yield-stress column.")
            stress = np.asarray(energy_df[stress_col], dtype=float)
            finite_stress = np.isfinite(stress)
            if not np.any(finite_stress):
                raise ValueError(f"No finite yield-stress values found in {csv_path}.")
            yield_gamma = float(load[int(np.nanargmax(stress))])
            color = curve_colors.get(label, f"C{len(curve_colors)}")
            (energy_line,) = energy_ax.plot(
                load,
                energy,
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.45,
                label=f"{label} energy",
            )
            energy_handles.append(energy_line)
            ax.axvline(
                yield_gamma,
                color=color,
                linestyle=":",
                linewidth=0.9,
                alpha=0.45,
            )
        energy_ax.set_ylabel(r"$E$")
        energy_ax.grid(False)
        if energy_handles:
            energy_ax.legend(
                handles=energy_handles,
                loc="lower right",
                fontsize=5.5,
                handlelength=1.5,
                borderpad=0.3,
            )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Mean elastic norm $\langle\|\mathbf{F}_e\|\rangle$")
    ax.legend(loc="best", fontsize=6, handlelength=1.5, borderpad=0.3)
    fig.tight_layout()

    if save:
        output_path = Path(name)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pdf")
        if not output_path.is_absolute():
            output_path = Path("Plots") / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f'Plot saved at: "{output_path}"')
    if show:
        plt.show()

    return fig, ax, summaries


def addImagesToPlot(
    ax,
    fig,
    csv_file_path,
    x,
    y,
    image_pos,
    size=0.4,
    mesh_property="energy",
    add_cbar=True,
):
    from .pyplotFunctions import plot_mesh

    # Get ordered VTU files for this simulation.
    vtu_files = resolve_vtu_files(Path(csv_file_path).parent)
    if len(vtu_files) == 0:
        raise ValueError(f"No VTU files found for {csv_file_path}")

    # Extract the paths for the first, middle, and last files.
    middle_idx = int(len(vtu_files) * 0.45)
    first_file = vtu_files[0]
    middle_file = vtu_files[middle_idx]
    last_file = vtu_files[-1]

    if not isinstance(size, list):
        size = [size] * 3

    for pos, size, vtu_file, index_fraction in zip(
        image_pos,
        size,
        [first_file, middle_file, last_file],
        [0, 0.45, 0.999],
    ):
        # Top left for the first image
        #                           (left, bottom, width, height)
        ax_inset = ax.inset_axes((pos[0], pos[1], size, size))
        _, cmap, norm = plot_mesh(
            e_lims=[0, 0.37],
            vtu_file=vtu_file,
            ax=ax_inset,
            add_rombus=False,
            shift=False,
            mesh_property=mesh_property,
        )
        ax_inset.axis("off")

        # Now we want to make arrows that point to the graph where the image is
        # taken from

        # This should work, but there is a desync in my data somehow that means the
        # load in the vtu files are not accurate
        if False:
            # First we find the load
            load = get_data_from_name(vtu_file)["load"]

            x_value = float(load)
            # we find the index that is closest to this load,
            # and use that to find a y value
            index = np.abs(x - x_value).argmin()
            y_value = y[index]
        else:
            value_index = int(index_fraction * len(x))
            x_value = x[value_index]
            y_value = y[value_index]
        # Get axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Convert normalized coordinates to actual axis coordinates
        arrow_start = (
            xlim[0] + (pos[0] + size / 2) * (xlim[1] - xlim[0]),
            ylim[0] + (pos[1] + size / 2) * (ylim[1] - ylim[0]),
        )

        # Arrow's ending point (the point on the main plot where the image corresponds to)
        arrow_end = (x_value, y_value)

        # Add the arrow using annotate
        ax.annotate(
            "",  # No text
            xy=arrow_end,  # End of the arrow (on the main plot)
            xytext=arrow_start,  # Start of the arrow (from the inset)
            arrowprops=dict(
                facecolor="black", shrink=0.05, width=0.5, headwidth=5, headlength=5
            ),
        )
    # Create the color bar using the colormap and normalization
    if add_cbar:
        # Create a ScalarMappable object with the colormap and norm
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        if mesh_property == "stress":
            label = r"Stress $\sigma$"
        elif mesh_property == "energy":
            label = r"Energy $E$"
        elif mesh_property == "m":
            label = r"Dislocations $\textbf{m}_3$"
        # Add the color bar to the figure
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.005, label=label)


def removeBadData(df, crash_count, csv_file_path):
    # The max energy of an element should be around 4.2-6
    # If the energy of an element is 10, something has probably gone wrong
    max_e = "max_energy"
    max_value = 10

    if (df[max_e] > max_value).any():
        crash_count += 1
        mask = df[max_e] < max_value
        df = df[mask]
        # print(f"Crash in {csv_file_path}.")
    return df, crash_count


def makeAverageComparisonPlot(
    grouped_csv_file_paths,
    Y="avg_energy",
    name="",
    group_labels=None,
    show=False,
    use_title=False,
    use_y_axis_name=True,
    ax=None,
    fig=None,
    save=True,
    xlim=None,
    ylim=None,
    mark=None,
    mark_pos=(0.8, 0.95),
    mark_fontsize=17,
    **kwargs,
):
    global color_index, index, line_index
    color_index, index, line_index = 0, 0, 0
    X = "load"
    x_name, y_name = get_axis_labels("load", Y, use_y_axis_name=use_y_axis_name)
    if name == "":
        if Y == "avg_energy":
            name = "Avg energy"
        elif Y == "avg_sigma12":
            name = "Avg Cauchy shear stress"
        elif Y == "avg_P12":
            name = "Avg Piola shear stress"
        elif "time" in Y:
            name = Y
    title = f"{name}"

    if fig is None or ax is None:
        assert fig is None
        assert ax is None
        fig, ax = plt.subplots()

    color_index = -1
    line_index = 0

    crash_count = 0

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    label_color_map = {}

    def _get_color(label, fallback_idx):
        if label in colors:
            return colors[label]
        if label in label_color_map:
            return label_color_map[label]
        if not default_cycle:
            return None
        color = default_cycle[len(label_color_map) % len(default_cycle)]
        label_color_map[label] = color
        return color

    # for each configuration
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        data = []
        # Increment the global call count
        color_index += 1
        # Check if we've used all colors
        if color_index >= len(colors):
            # Switch to the next line style and marker
            color_index = 0
            line_index += 1

        method_label = get_method(csv_file_paths)
        if group_labels and i < len(group_labels) and group_labels[i]:
            label = getPrettyLabel(group_labels[i])
        else:
            label = method_label

        # Get the current color
        color = _get_color(label, i)

        # For each seed using this config
        for j, csv_file_path in enumerate(csv_file_paths):
            # print(csv_file_path)
            df = pd.read_csv(csv_file_path, usecols=[X, Y, "max_energy"])
            # If Y contains strings, we will assume it is a time, and convert it to
            # seconds
            if isinstance(df[Y][0], str):
                df[Y] = durations_to_seconds(df[Y])
            # df = df[0:50000]
            if df.empty:
                continue
            df, crash_count = removeBadData(df, crash_count, csv_file_path)

            data.append(df[Y].values)

            e_kwargs = {
                "fig": fig,
                "ax": ax,
                "color": color,
                "linestyle": LINE_STYLES[line_index],
                "alpha": 0.05,
                "zorder": color_index - 10,
                "xlim": xlim,
                "ylim": ylim,
            }
            fig, ax, line, point = plotYOverX(df[X], df[Y], **e_kwargs)
        # Determine the maximum length among all arrays
        max_length_index = np.argmax([len(d) for d in data])
        max_length = len(data[max_length_index])

        # Initialize the average array and a count array to track how many entries per index
        average = np.zeros(max_length)
        count = np.zeros(max_length)

        # Aggregate data
        for d in data:
            length = len(d)
            average[:length] += d
            count[:length] += 1

        # Compute average where count is non-zero to avoid division by zero
        average = np.divide(
            average, count, out=np.zeros_like(average), where=count != 0
        )
        a_kwargs = {
            "fig": fig,
            "ax": ax,
            "label": label,
            "color": color,
            "linestyle": LINE_STYLES[line_index],
            "zorder": -color_index,
            "xlim": xlim,
            "ylim": ylim,
        }
        df = pd.read_csv(csv_file_paths[max_length_index], usecols=[X])
        fig, ax, line, point = plotYOverX(df[X], average, **a_kwargs)
    if crash_count > 0:
        print(f"Found {crash_count} crashes using {label}.")
    # Set the legend with the filtered handles and labels
    ax.legend(loc="upper left")
    ax.set_xlabel(x_name)

    if use_y_axis_name:
        ax.set_ylabel(y_name)
    if use_title:
        ax.set_title(title)

    if mark:
        assert mark_pos is not None
        add_mark(ax, f"({mark})", *mark_pos, fontsize=mark_fontsize)

    if save:
        # Get the parent directory of the CSV file
        csv_directory = Path(grouped_csv_file_paths[0][0]).parent.parent
        # Move to the "plots" directory relative to the CSV file directory
        plotPath = csv_directory / "plots"

        figPath = os.path.join(plotPath, name + ".pdf")

        # fig.savefig(figPath)
        fig.savefig("Plots/" + name + ".pdf")
        print(f'Plot saved at: "Plots/{name}.pdf"')
    if show:
        plt.show()
    return fig, ax


def add_power_law_line(ax, slope, x_lim, y_pos=1, c="black", linestyle="--", **kwargs):
    x = np.logspace(np.log10(x_lim[0]), np.log10(x_lim[1]), 100)
    y = y_pos * x**slope
    ax.plot(x, y, label=rf"fit: $\alpha={slope}$", c=c, linestyle=linestyle, **kwargs)
    ax.legend()


def makeEnergyAvalancheComparison(
    grouped_csv_file_paths,
    name,
    xlim=[-np.inf, np.inf],
    show=False,
    **kwargs,
):
    global color_index, index, line_index
    color_index, index, line_index = 0, 0, 0
    X = "load"
    Y = "avg_energy"
    x_name = "Magnitude of energy drops"
    y_name = r"$P(>E)$"
    lims = "" if xlim == [-np.inf, np.inf] else f", xlim: {xlim[0]}-{xlim[1]}"

    crash_count = 0

    # for each config
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        dfs = []  # panda dataframes
        # for each seed using this config
        for j, csv_file_path in enumerate(csv_file_paths):
            header = pd.read_csv(csv_file_path, nrows=0)
            plastic_col = infer_plastic_event_column(header)
            df = pd.read_csv(
                csv_file_path, usecols=[X, Y, plastic_col, "max_energy"]
            ).rename(columns={plastic_col: "nr_elements_with_m3_fix_change"})
            # Truncate data based on xlim
            df = df[(df[X] >= xlim[0]) & (df[X] <= xlim[1])]
            # Truncate data based on dislocations
            df, crash_count = removeBadData(df, crash_count, csv_file_path)
            dfs.append(df)
        fig, ax = plotEnergyAvalancheHistogram(dfs)

        if crash_count > 0:
            print(f"Found {crash_count} crashes using {['L-BFGS', 'CG', 'FIRE'][i]}.")
        # Set the legend with the filtered handles and labels
        # ax.legend(loc=("best"))
        # ax.set_xlabel(x_name)
        # ax.set_ylabel(y_name)
        title = f"{name}{lims}" + f"-{['L-BFGS', 'CG', 'FIRE'][i]}"
        fig.suptitle(title, fontsize=16)  # Set the main
        plt.tight_layout()  # Adjust subplots to fit into figure area.
        # Get the parent directory of the CSV file
        csv_directory = Path(grouped_csv_file_paths[0][0]).parent.parent
        # Move to the "plots" directory relative to the CSV file directory
        plotPath = csv_directory / "plots"

        figPath = os.path.join(plotPath, title + ".pdf")
        fig.savefig(figPath)
        fig.savefig("Plots/" + title + ".pdf")
        # print(f'Plot saved at: "{figPath}"')
        print(f'Plot saved at: "{figPath}"')
        if show:
            plt.show()


def sci_format(val):
    """Format numbers in scientific notation without leading zeros in exponents."""
    return f"{val:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def get_linestyle(pos):
    """Returns a linestyle that avoids overlap based on `pos`."""
    pos = np.array(pos)
    total_length = 13  # Increase total length to allow more variation

    if sum(pos) >= 4:
        return "-"
    else:
        pos += 1
        dash = pos[0] * 2 + 2  # Ensure a minimum dash length
        gap = max(2, total_length - dash - pos[1] * 2)  # Keep the gap reasonable
        return (sum(pos), (pos[1] * 2, gap))


def parse_labels(labels, property_keys):
    """Extract numerical property values and additional labels from provided labels."""
    prop1_values, prop2_values, extra_labels = [], [], []

    for label in labels:
        parts = label.split(",")
        p1, p2, extra = None, None, []
        for part in parts:
            key, value = map(str.strip, part.split("="))
            if key == property_keys[0]:
                p1 = float(value)
            elif len(property_keys) > 1 and key == property_keys[1]:
                p2 = float(value)
            else:
                extra.append(getPrettyLabel(part))
        prop1_values.append(p1)
        prop2_values.append(p2)
        extra_labels.append(", ".join(extra))

    return np.array(prop1_values), np.array(prop2_values), extra_labels


def safe_log_norm(values):
    min_val, max_val = np.log10(values.min()), np.log10(values.max())
    return (
        np.zeros_like(values)
        if min_val == max_val
        else (np.log10(values) - min_val) / (max_val - min_val)
    )


def safe_linear_norm(values):
    min_val, max_val = values.min(), values.max()
    return (
        np.zeros_like(values)
        if min_val == max_val
        else (values - min_val) / (max_val - min_val)
    )


def create_color_matrix(prop1_values, prop2_values, log_p1=True, log_p2=True):
    """Generate a color matrix based on normalized property values."""

    def _norm(values, use_log):
        if use_log:
            return safe_log_norm(values)
        return safe_linear_norm(values)

    unique_p1 = np.unique(prop1_values)

    if prop2_values[0] is None:
        unique_p2 = None
        p1_norm = _norm(unique_p1, log_p1)
        color_matrix = np.zeros((1, len(unique_p1), 4))  # Single row matrix
        for col, (p1, r) in enumerate(zip(unique_p1, p1_norm)):
            color_matrix[0, col] = [r, 0, 0, 1]  # Only using log_p1_norm
        return color_matrix, unique_p1, unique_p2

    unique_p2 = np.unique(prop2_values)

    p1_norm = _norm(prop1_values, log_p1)
    p2_norm = _norm(prop2_values, log_p2)

    color_matrix = np.zeros((len(unique_p2), len(unique_p1), 4))
    index_map = {}

    for i, (p1, p2, r, b) in enumerate(
        zip(prop1_values, prop2_values, p1_norm, p2_norm)
    ):
        row, col = np.where(unique_p2 == p2)[0][0], np.where(unique_p1 == p1)[0][0]
        color_matrix[row, col] = [r, abs(r - b), b, 1]
        index_map[(row, col)] = i

    return color_matrix, unique_p1, unique_p2


def plot_color_matrix(
    ax,
    color_matrix,
    unique_p1,
    unique_p2,
    property_keys,
    fmt_p1=None,
    fmt_p2=None,
    xlabel=None,
    ylabel=None,
    width="25%",
    height="25%",
    loc="upper left",
    bbox_to_anchor=None,
    marker_matrix=None,
    marker_color="black",
):
    """Inset a color matrix inside the main plot."""

    if bbox_to_anchor is None:
        bbox_to_anchor = (0.1, -0.05, 1, 1)

    inset_ax = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )
    inset_ax.matshow(color_matrix.transpose((0, 1, 2)), origin="upper", aspect="auto")
    if marker_matrix is not None:
        marker_matrix = np.asarray(marker_matrix, dtype=object)
        if marker_matrix.shape != color_matrix.shape[:2]:
            raise ValueError("marker_matrix must match the first two color_matrix dimensions.")
        marker_color_matrix = None
        if not isinstance(marker_color, str) and not callable(marker_color):
            marker_color_matrix = np.asarray(marker_color, dtype=object)
            if marker_color_matrix.shape != marker_matrix.shape:
                raise ValueError("marker_color matrix must match marker_matrix shape.")
        row_cut = int(np.ceil(marker_matrix.shape[0] / 2))
        col_cut = int(np.ceil(marker_matrix.shape[1] / 2))
        for row in range(marker_matrix.shape[0]):
            for col in range(marker_matrix.shape[1]):
                marker = marker_matrix[row, col]
                if marker is None:
                    continue
                if marker_color == "lower_left_white":
                    this_marker_color = (
                        "white" if row < row_cut and col < col_cut else "black"
                    )
                elif callable(marker_color):
                    this_marker_color = marker_color(row, col)
                elif marker_color_matrix is not None:
                    this_marker_color = marker_color_matrix[row, col]
                else:
                    this_marker_color = marker_color
                inset_ax.plot(
                    col,
                    row,
                    marker=marker,
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor=this_marker_color,
                    markeredgewidth=1.1,
                    markersize=5.5,
                )
    inset_ax.set_xticks(range(len(unique_p1)))
    if fmt_p1 is None:
        fmt_p1 = sci_format
    if fmt_p2 is None:
        fmt_p2 = sci_format
    inset_ax.set_xticklabels([fmt_p1(val) for val in unique_p1])
    inset_ax.set_xlabel(
        getPrettyLabel(property_keys[0]) if xlabel is None else xlabel, fontsize=10
    )
    if unique_p2 is not None:
        inset_ax.set_yticks(range(len(unique_p2)))
        inset_ax.set_yticklabels([fmt_p2(val) for val in unique_p2])
        inset_ax.set_ylabel(
            getPrettyLabel(property_keys[1]) if ylabel is None else ylabel, fontsize=10
        )
    inset_ax.set_title("Parameter color map", fontsize=10)
    inset_ax.invert_yaxis()
    inset_ax.xaxis.set_ticks_position("bottom")
    return inset_ax


def find_best_color_matrix_corner(
    x_vals,
    y_vals,
    logx=False,
    logy=False,
    default_loc="upper left",
    bbox_to_anchor=(0.1, -0.05, 1, 1),
    quantile=0.25,
    ax=None,
):
    x = np.asarray(x_vals).ravel()
    y = np.asarray(y_vals).ravel()
    if x.size == 0 or y.size == 0:
        return default_loc, bbox_to_anchor

    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    x = x[mask]
    y = y[mask]
    if x.size == 0 or y.size == 0:
        return default_loc, bbox_to_anchor

    x_use = np.log10(x) if logx else x
    y_use = np.log10(y) if logy else y

    x_min, x_max = np.min(x_use), np.max(x_use)
    y_min, y_max = np.min(y_use), np.max(y_use)
    q = np.clip(quantile * 1.6, 0.0, 0.5)
    x_lo = x_min + q * (x_max - x_min)
    x_hi = x_max - q * (x_max - x_min)
    y_lo = y_min + q * (y_max - y_min)
    y_hi = y_max - q * (y_max - y_min)

    counts = {
        "upper right": np.sum((x_use >= x_hi) & (y_use >= y_hi)),
        "upper left": np.sum((x_use <= x_lo) & (y_use >= y_hi)),
        "lower right": np.sum((x_use >= x_hi) & (y_use <= y_lo)),
        "lower left": np.sum((x_use <= x_lo) & (y_use <= y_lo)),
    }
    order = ["upper left", "upper right", "lower left", "lower right"]
    min_count = min(counts.values())
    for loc in order:
        if counts[loc] == min_count:
            chosen_loc = loc
            break
    else:
        chosen_loc = default_loc

    x_off, y_off, w, h = bbox_to_anchor
    x_off_base = abs(x_off)
    y_off_base = abs(y_off)
    if chosen_loc == "upper right":
        x_off = -x_off_base * 0.6
        y_off = -y_off_base
    elif chosen_loc == "upper left":
        x_off = x_off_base
        y_off = -y_off_base
    elif chosen_loc == "lower right":
        x_off = -x_off_base
        y_off = y_off_base * 1.4
    else:
        x_off = x_off_base
        y_off = y_off_base * 1.4

    return chosen_loc, (x_off, y_off, w, h)


def all_files_have_same_starting_point(csv_file_paths):
    X = "load"
    Y = "avg_energy"
    energy = defaultdict(list)
    for csv_file_path in csv_file_paths:
        seed = get_data_from_name(csv_file_path)["seed"]
        df = pd.read_csv(csv_file_path, usecols=[X, Y])
        energy[seed].append(df[Y][0])
        if len(set(energy[seed])) != 1:
            print("Files do not start from the same energy!")
            print(csv_file_path)


def makeSettingComparison(
    csv_file_paths=None,
    labels=None,
    property_keys=None,  # a tuple of two keys, e.g., ("epsR", "epsE")
    X="load",
    Y="avg_energy",
    title=None,
    xlim=None,
    ylim=None,
    show=False,
    save=True,
    name="setting_comparison",
    loc="lower right",
    subtract=False,  # Shows the difference from the most exact simulation
    detatchment=False,  # Shows where simulations detatch from the most accurate simulation
    detatchmentThreshold=1e-5,
    yPad=1,
    seedsToShow=[0, 1, 2, 3, 4],
    **kwargs,
):
    # Check files start from the same energy

    all_files_have_same_starting_point(csv_file_paths)

    if detatchment:
        subtract = True

    # --- Extract and parse the two properties ---
    prop1_values, prop2_values, extra_labels = parse_labels(labels, property_keys)

    pretty_keys = [getPrettyLabel(k) for k in property_keys]
    prop1_values, prop2_values = np.array(prop1_values), np.array(prop2_values)

    # --- Create unique property values for grid ---
    color_matrix, unique_p1, unique_p2 = create_color_matrix(prop1_values, prop2_values)

    # --- Preserve diagonal colors (as in original) ---
    if subtract:
        color_matrix[0, 0] = [1, 1, 1, 1]
    else:
        color_matrix[0, 0] = [1, 0, 0, 1]
    # color_matrix[1, 1] = [0, 1, 0, 1]
    # color_matrix[2, 2] = [0, 0, 1, 1]

    # --- Create the figure and the main plot ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # --- Find the best simulation ---
    # Dictionaries coresponding to the seed
    best_df = {}
    best_label = ""  # We'll just use the last one, should be the same anyway
    for i, csv_file_path in enumerate(csv_file_paths):
        pos = (
            np.where(unique_p2 == prop2_values[i])[0][0]
            if unique_p2 is not None
            else 0,
            np.where(unique_p1 == prop1_values[i])[0][0],
        )
        if pos == (0, 0):
            print("best setting:", csv_file_path)
            seed = get_data_from_name(csv_file_path)["seed"]
            best_df[seed] = pd.read_csv(csv_file_path, usecols=[X, Y])
            best_label = f"{pretty_keys[0]}={sci_format(prop1_values[i])}"
            if unique_p2 is not None:
                best_label += f", {pretty_keys[1]}={sci_format(prop2_values[i])}"

    # --- Data manipulation ---
    added_labels = []
    all_x = []
    all_y = []
    for i, csv_file_path in enumerate(csv_file_paths):
        seed = get_data_from_name(csv_file_path)["seed"]
        if seed not in seedsToShow:
            continue

        pos = (
            np.where(unique_p2 == prop2_values[i])[0][0]
            if unique_p2 is not None
            else 0,
            np.where(unique_p1 == prop1_values[i])[0][0],
        )
        color = color_matrix[pos]

        try:
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
        except Exception as e:
            print(f"Error reading {csv_file_path}: {e}")
            continue

        if len(df) == 0:
            print(f"No data in {csv_file_path}.")
            continue

        X_data, Y_data = df[X].values, df[Y].values
        if subtract:
            if pos == (0, 0):
                continue
            else:
                best_X = best_df[seed][X].values
                best_Y = best_df[seed][Y].values

                if max(best_X) > max(X_data):
                    print("Best is further!: " + csv_file_path)
                    continue

                # Ensure X_data is within range of best_X
                within_best = X_data < max(best_X)
                X_data = X_data[within_best]
                Y_data = Y_data[within_best]

                # Find indices of X_data in best_X
                indices = np.searchsorted(best_X, X_data)

                # Ensure indices are valid
                assert np.all(best_X[indices] == X_data), (
                    "Mismatch between X_data and best_X"
                )

                # Get corresponding best_Y values
                filtered_best_Y = best_Y[indices]

                # Compute absolute differences
                Y_data = np.abs(Y_data - filtered_best_Y)

                # Find the last point at which the difference is zero
                detach_Y_data = np.where(Y_data < detatchmentThreshold, 1, 0)
                if np.any(detach_Y_data):  # Check if there is at least one zero
                    detach_index = len(Y_data) - 1 - np.argmax(detach_Y_data[::-1])
                else:
                    detach_index = 0  # No zeros found
                if detatchment:
                    Y_data = [Y_data[detach_index]]
                    X_data = [X_data[detach_index]]

        if isinstance(Y_data[0], str):
            try:
                Y_data = np.array([duration_to_seconds(y_str) for y_str in Y_data])
            except Exception as e:
                print(f"Error converting Y data in {csv_file_path}: {e}")

        if xlim is not None:
            mask = (X_data >= xlim[0]) & (X_data <= xlim[1])
            X_data, Y_data = X_data[mask], Y_data[mask]
        if ylim is not None:
            mask = (Y_data >= ylim[0]) & (Y_data <= ylim[1])
            X_data, Y_data = X_data[mask], Y_data[mask]

        # PLOTTING
        # --- Ensure proper layering (smaller values drawn first) ---
        sorted_indices = np.argsort(prop1_values)
        z_order = len(sorted_indices) - np.where(sorted_indices == i)[0][0]
        linestyle = get_linestyle(pos)
        label = f"{pretty_keys[0]}={sci_format(prop1_values[i])}"
        if unique_p2 is not None:
            label += f", {pretty_keys[1]}={sci_format(prop2_values[i])}"
        if label in added_labels:
            label = None
        else:
            added_labels.append(label)
        if detatchment:
            marker = "o"
            if detach_index == -1:
                marker = "^"
            ax.scatter(
                prop1_values[i],
                X_data,
                label=label,
                zorder=z_order,
                edgecolors=color,
                facecolors="none",
                marker=marker,
                **kwargs,
            )
            all_x.extend(np.atleast_1d(prop1_values[i]))
            all_y.extend(np.atleast_1d(X_data))

        else:
            ax.plot(
                X_data,
                Y_data,
                color=color,
                label=label,
                zorder=z_order,
                linestyle=linestyle,
                **kwargs,
            )
            all_x.extend(np.atleast_1d(X_data))
            all_y.extend(np.atleast_1d(Y_data))
            if subtract:
                if detach_index == -1:
                    marker = "^"
                else:
                    marker = "+"
                ax.scatter(
                    X_data[detach_index],
                    Y_data[detach_index],
                    zorder=-40 - z_order,
                    # edgecolors=color,
                    facecolors=color,
                    marker=marker,
                    s=40 + z_order * 2,
                    **kwargs,
                )

    # --- Make space for legend at top ---
    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()

    # Adjust the upper y-limit
    ax.set_ylim(y_min, y_max * yPad)

    # --- Set labels and titles for main plot ---
    x_name, y_name = get_axis_labels(X, Y)
    if subtract:
        y_name = f"{y_name} difference from {best_label}"
    if detatchment:
        y_name = "Detatchment " + x_name
        x_name = f"{pretty_keys[0]}"
        ax.set_xscale("log")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    # ax.set_title(f"{', '.join(set(extra_labels))}")
    cm_loc, cm_bbox = find_best_color_matrix_corner(
        all_x, all_y, logx=detatchment, logy=False
    )
    plot_color_matrix(
        ax,
        color_matrix,
        unique_p1,
        unique_p2,
        property_keys,
        loc=cm_loc,
        bbox_to_anchor=cm_bbox,
    )
    ax.legend(fontsize=8, loc=loc, ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fig.savefig(f"Plots/{name}.pdf")
        print(f"Plot saved to: Plots/{name}.pdf")
    if show:
        plt.show()

    return fig, ax


def makeItterationsPlot(csv_file_paths, name, **kwargs):
    if isinstance(csv_file_paths, str):
        makePlot(
            csv_file_paths,
            name,
            X="load",
            Y=["Nr FIRE iterations", "Nr LBFGS iterations"],
            y_name="Nr itterations",
            title="Nr of Itterations",
            plot_raw=True,
            plot_roll_average=False,
            plot_total=True,
            **kwargs,
        )
    else:
        makePlot(
            csv_file_paths,
            name,
            X="load",
            Y=["Nr FIRE iterations", "Nr LBFGS iterations"],
            y_name="Nr itterations",
            title="Nr of Itterations",
            plot_raw=True,
            plot_roll_average=False,
            plot_total=False,
            **kwargs,
        )


def makeTimePlot(csv_file_paths, name, **kwargs):
    makePlot(
        csv_file_paths,
        name,
        x_name="Settings",
        Y=["Run_time"],
        y_name="Run time (s)",
        plot_raw=False,
        plot_columns=True,
        title="Runtime of simulations",
        **kwargs,
    )


def makePowerLawPlot(csv_file_paths, name, **kwargs):
    makePlot(
        csv_file_paths,
        name,
        X="load",
        Y="avg_energy",
        x_name="Magnitude of energy drops",
        y_name=r"$P(>E)$",
        title="Powerlaw",
        plot_raw=False,
        plot_power_law=True,
        plot_average=False,
        **kwargs,
    )


if __name__ == "__main__":
    pass
    # The path should be the path from work directory to the folder inside the Plots folder.
    makePlot(
        [
            "/Volumes/data/MTS2D_output/simpleShear,s60x60l0.15,0.0002,1.0PBCt1minimizerFIRELBFGSEpsg0.0001eps0.01s0/macroData.csv",
        ],
        name="energy.pdf",
        Y="avg_energy",
    )
    # makeItterationsPlot(
    #     [
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
    #     ],
    #             name='nrIterations.pdf')

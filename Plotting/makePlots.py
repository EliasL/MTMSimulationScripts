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
from .dataFunctions import get_data_from_name
from Management.updateCSV import update_df_header
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
        .replace(".00", "")
    )
    return safe_path


USE_AVG_ENERGY = False


def maybe_avg(text, use_avg=None):
    if use_avg is None:
        use_avg = USE_AVG_ENERGY
    return rf"\langle {text} \rangle" if use_avg else text


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
    if not isinstance(cvs_file_path, str):
        cvs_file_path = cvs_file_path[0]
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
            df = df[df["nr_plastic_deformations"] > 0]

            group_index = np.floor(np.log2(df["nr_plastic_deformations"])).astype(int)
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
                ax.set_ylabel(rf"$P>{maybe_avg('E', use_avg)}$")
            if i >= 6:  # Not the bottom row
                ax.set_xlabel(rf"$-\Delta {maybe_avg('E', use_avg)}$")

    return fig, axs


def getPrettyLabel(string):
    s = ""
    if "minimizer=" in string:
        s = string.split("minimizer=")[1].split(",")[0]
    if "," not in string:
        s = string
    if "eps" in string.lower():
        # Find the position of "eps" (case-insensitive)
        idx = string.lower().index("eps")
        # Extract the part after "eps" without changing its case
        stopType = string[idx + 3 :]
        if stopType == "g":
            stopType = r"\sigma"
        if stopType == "x":
            stopType = r"$|\Delta \mathbf{u}|$"
        s = rf"$\epsilon_{{{stopType}}}$"
    s = s.replace("LBFGS", "L-BFGS")
    s = s.replace("loadIncrement", r"$\Delta \gamma$")
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
            "energy": rf"Energy ${maybe_avg('W', use_avg)}$",
            "est_time_remaining": "Estimated time remaining (s)",
        }
        y_name = y_labels_map.get(Y_, Y)

    return x_name, y_name


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
    subtract=None,
):
    if len(csv_file_paths) == 0 or (
        len(csv_file_paths) > 0 and len(csv_file_paths[0]) == 0
    ):
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

    line_index = 0
    for i, csv_file_path in enumerate(csv_file_paths):
        if i == subtract:
            continue
        if X is None:
            breakpoint
        df = pd.read_csv(csv_file_path)
        df = update_df_header(df, L=_get_system_size_from_path(csv_file_path))
        # If it is a string, we assume it is a time that we can convert to seconds
        if isinstance(df[Y][0], str):
            df[Y] = durations_to_seconds(df[Y])
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

        if subtract is not None:
            if isinstance(subtract, str):
                sub_df = pd.read_csv(subtract, usecols=[Y])
            elif isinstance(subtract, int):
                sub_df = pd.read_csv(csv_file_paths[subtract], usecols=[Y])
            else:
                raise ValueError("Invalid type for subtract. Must be path or int.")

            df[Y] -= sub_df[Y].reindex(df.index, fill_value=0)

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

        for Y_ in [Y] if isinstance(Y, str) else Y:
            if len(df[Y_]) == 0:
                continue
            if labels is None:
                kwargs["label"] = Y
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
                                true_label = (
                                    f"{labels[i]} (true)" if j == 0 else "_nolegend_"
                                )
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

    # First we get the folder with vtu_files
    framesPath = Path(csv_file_path).parent / "data"

    # Define the regex pattern to match the file names and find the number between the dots
    pattern = re.compile(r".*\.(\d*)\.vtu")

    # Get all files in the folder matching the pattern and extract both the number and full path
    matching_files = [
        (
            int(pattern.match(f.name).group(1)),
            f,
        )  # Create a tuple with the number and the full path
        for f in framesPath.iterdir()
        if f.is_file() and pattern.match(f.name)
    ]

    # Sort the list of tuples by the number (the first element of the tuple)
    matching_files.sort(key=lambda x: x[0])

    # Extract the paths for the first, middle, and last files
    first_file = matching_files[0][1]
    middle_file = matching_files[int(len(matching_files) * 0.45)][1]
    last_file = matching_files[-1][1]

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
            label = r"Energy $W$"
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

        # Get the current color
        color = colors[get_method(csv_file_paths)]

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
        label = get_method(csv_file_paths)
        a_kwargs = {
            "fig": fig,
            "ax": ax,
            "label": label,
            "color": colors[label],
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
        # print(f'Plot saved at: "{figPath}"')
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
            df = pd.read_csv(
                csv_file_path, usecols=[X, Y, "nr_plastic_deformations", "max_energy"]
            )
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
    # The path should be the path from work directory to the folder inside the output folder.
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

"""Energy-density plot for a single logged minimization trajectory."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Management.configGenerator import SimulationConfig
from Plotting.dataFunctions import get_metadata
from Plotting.energyDropCalculations import volume_from_metadata
from Plotting.makePlots import getPrettyLabel


def _reference_volume(csv_path, volume):
    if volume is not None:
        volume = float(volume)
    else:
        config_path = next(
            (
                parent / "config.conf"
                for parent in csv_path.parents
                if (parent / "config.conf").is_file()
            ),
            None,
        )
        if config_path is not None:
            config = SimulationConfig(config_path)
            volume = float(config.rows * config.cols)
        else:
            volume = volume_from_metadata(get_metadata(str(csv_path)))
    if volume is None or not np.isfinite(volume) or volume <= 0:
        raise ValueError(
            f"Could not infer a positive reference volume for {csv_path}; "
            "pass volume explicitly."
        )
    return volume


def _narrow_lower_scale_fraction(limits, fraction, narrowing, scale):
    lower, upper = sorted(limits)
    if scale == "log":
        if lower <= 0:
            raise ValueError("A logarithmic energy axis requires positive limits.")
        log_lower, log_upper = np.log(lower), np.log(upper)
        center = log_lower + 0.5 * fraction * (log_upper - log_lower)
        half_width = 0.5 * fraction * (log_upper - log_lower) / narrowing
        lower, upper = np.exp(center - half_width), np.exp(center + half_width)
    else:
        center = lower + 0.5 * fraction * (upper - lower)
        half_width = 0.5 * fraction * (upper - lower) / narrowing
        lower, upper = center - half_width, center + half_width
    return lower, upper


def _narrow_previous_log_zoom(values, fraction, narrowing, margin):
    """Keep the former log-plot zoom center while returning linear limits."""
    if np.any(values <= 0):
        raise ValueError("Preserving the previous log-view center requires positive energy.")
    log_lower, log_upper = np.log([np.min(values), np.max(values)])
    padding = margin * (log_upper - log_lower)
    log_lower -= padding
    log_upper += padding
    old_bottom = np.exp(log_lower)
    old_top = np.exp(log_lower + fraction * (log_upper - log_lower))
    center = 0.5 * (old_bottom + old_top)
    half_width = 0.5 * (old_top - old_bottom) / narrowing
    return center - half_width, center + half_width


def plot_minimization_energy(
    csv_path,
    output_path=None,
    *,
    volume=None,
    zoom_fraction=(0.1, 0.1),
    zoom_y_narrowing=5,
    zoom_xlim=(None, 70000),
    zoom_ylim=(0.009665, 0.00968),
    ylog=False,
    show=False,
):
    """Plot ``total_energy / V_0`` with a top-right inset of the lower-right corner.

    ``zoom_xlim`` and ``zoom_ylim`` set exact zoom bounds. A ``None`` x bound
    retains the width given by ``zoom_fraction[0]``; a ``None`` y bound uses the
    former lower-corner behavior and ``zoom_y_narrowing``. The connectors join
    the zoom region's top corners to the inset's bottom corners.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if len(zoom_fraction) != 2 or not all(0 < float(value) <= 1 for value in zoom_fraction):
        raise ValueError("zoom_fraction must contain two values in (0, 1].")
    if not np.isfinite(zoom_y_narrowing) or zoom_y_narrowing < 1:
        raise ValueError("zoom_y_narrowing must be finite and at least 1.")
    if len(zoom_xlim) != 2 or len(zoom_ylim) != 2:
        raise ValueError("zoom_xlim and zoom_ylim must each contain two values.")

    frame = pd.read_csv(csv_path)
    required = {"nr_func_evals", "total_energy", "nr_reconnect_cycles"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(sorted(missing))}")

    x = frame["nr_func_evals"].to_numpy(dtype=float)
    energy_density = frame["total_energy"].to_numpy(dtype=float) / _reference_volume(
        csv_path, volume
    )
    finite = np.isfinite(x) & np.isfinite(energy_density)
    if ylog:
        finite &= energy_density > 0
    if not np.any(finite):
        raise ValueError("No finite positive energy-density samples to plot.")
    reconnect_cycles = frame["nr_reconnect_cycles"].to_numpy(dtype=float)[finite]
    x, energy_density = x[finite], energy_density[finite]
    if not np.all(np.isfinite(reconnect_cycles)):
        raise ValueError("nr_reconnect_cycles contains non-finite values.")
    reconnect_events = np.r_[
        reconnect_cycles[0] > 0,
        np.diff(reconnect_cycles) > 0,
    ]

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.19, right=0.97, bottom=0.14, top=0.97)
    label = r"$\langle E\rangle/V_0$"
    (line,) = ax.plot(x[:-1], energy_density[:-1], label=label)
    ax.plot(
        x[-1],
        energy_density[-1],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        label="Final energy",
    )
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(getPrettyLabel("nr_func_evals"))
    ax.set_ylabel(r"Energy $\langle E\rangle/V_0$")
    ax.legend(loc="upper left")

    fig.canvas.draw()
    main_ylim = ax.get_ylim()
    x_left, x_right = sorted(ax.get_xlim())
    x_zoom_right = x_right if zoom_xlim[1] is None else float(zoom_xlim[1])
    x_zoom_left = (
        x_zoom_right - float(zoom_fraction[0]) * (x_right - x_left)
        if zoom_xlim[0] is None
        else float(zoom_xlim[0])
    )
    if not x_zoom_left < x_zoom_right:
        raise ValueError("zoom_xlim must define an increasing x range.")
    if zoom_ylim[0] is not None and zoom_ylim[1] is not None:
        y_zoom_bottom, y_zoom_top = map(float, zoom_ylim)
    elif ylog:
        y_zoom_bottom, y_zoom_top = _narrow_lower_scale_fraction(
            main_ylim,
            float(zoom_fraction[1]),
            float(zoom_y_narrowing),
            ax.get_yscale(),
        )
    else:
        y_zoom_bottom, y_zoom_top = _narrow_previous_log_zoom(
            energy_density,
            float(zoom_fraction[1]),
            float(zoom_y_narrowing),
            ax.margins()[1],
        )
    if not y_zoom_bottom < y_zoom_top:
        raise ValueError("zoom_ylim must define an increasing y range.")
    ax.vlines(
        x[reconnect_events],
        min(main_ylim),
        energy_density[reconnect_events],
        color="0.35",
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )
    ax.set_ylim(main_ylim)

    inset = inset_axes(
        ax,
        width="32.3%",
        height="34%",
        loc="upper right",
        bbox_to_anchor=(-0.04, -0.04, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    inset.plot(x[:-1], energy_density[:-1], color=line.get_color())
    inset.plot(
        x[-1],
        energy_density[-1],
        linestyle="none",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
    )
    if ylog:
        inset.set_yscale("log")
    inset.set_xlim(x_zoom_left, x_zoom_right)
    inset.set_ylim(y_zoom_bottom, y_zoom_top)
    inset.vlines(
        x[reconnect_events],
        y_zoom_bottom,
        energy_density[reconnect_events],
        color="0.35",
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )
    inset.tick_params(labelsize="small")

    zoom_box = Rectangle(
        (x_zoom_left, y_zoom_bottom),
        x_zoom_right - x_zoom_left,
        y_zoom_top - y_zoom_bottom,
        fill=False,
        edgecolor="black",
        linewidth=1,
        zorder=4,
    )
    ax.add_patch(zoom_box)
    for main_x, inset_x in ((x_zoom_left, 0), (x_zoom_right, 1)):
        fig.add_artist(
            ConnectionPatch(
                xyA=(main_x, y_zoom_top),
                coordsA=ax.transData,
                xyB=(inset_x, 0),
                coordsB=inset.transAxes,
                color="black",
                linewidth=0.8,
                clip_on=False,
                zorder=3,
            )
        )

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if show:
        plt.show()
    return fig, ax, inset

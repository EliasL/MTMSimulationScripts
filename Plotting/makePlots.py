from matplotlib import pyplot as plt
import matplotlib
import os
import matplotlib.pylab
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import re
import mplcursors
from datetime import timedelta
import powerlaw
import pickle
from simplification.cutil import simplify_coords_vwp


def plotYOverX(
    X, Y, fig=None, ax=None, indicateLastPoint=True, tolerance=1e-12, **kwargs
):
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()

    # Simplify the points if tolerance is provided
    if tolerance is not None:
        points = np.column_stack((X, Y))
        simplified_points = simplify_coords_vwp(points, tolerance)
        X_simplified = simplified_points[:, 0]
        Y_simplified = simplified_points[:, 1]
    else:
        X_simplified = X
        Y_simplified = Y
    # Plot on the provided axis
    (line,) = ax.plot(X_simplified, Y_simplified, **kwargs)

    if indicateLastPoint:
        # Add a scatter point at the last point
        kwargs["label"] = ""
        point = ax.scatter(X_simplified[-1], Y_simplified[-1], **kwargs)
    else:
        point = None

    # Return the axis object for further use
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
    for file in cvs_files:
        df = pd.read_csv(file)
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
line_styles = ["-", "--", "-.", ":"]
markers = ["v", "o", "^", "s", "D", "p", "*"]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_index = 0
index = 0
line_index = 0


def plotPowerLaw(
    y_values,
    fig=None,
    ax=None,
    label="",
):
    global color_index, index, line_index
    if ax is None:
        fig, ax = plt.subplots()

    # Process drops
    def pros_d(y_values, start, end):
        diffs = np.diff(y_values[start:end])
        return -diffs[diffs < 0]

    pre_yield_drops = [pros_d(np.array(y), 0, np.argmax(y) + 1) for y in y_values]
    post_yield_drops = [pros_d(np.array(y), np.argmax(y) + 1, len(y)) for y in y_values]

    combined_drops = [
        np.concatenate(drops) for drops in (pre_yield_drops, post_yield_drops)
    ]

    for drops, part_label, index in zip(
        combined_drops, ["pre yield", "post yield"], [0, 1]
    ):
        if len(drops) == 0:
            continue

        # Configure the analysis range
        xmin, xmax = np.min(drops), np.max(drops)

        fit = powerlaw.Fit(drops, xmin=xmin, xmax=xmax, fit_method="Likelihood")

        # Set up bins and plot histogram
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 12)
        hist, bin_edges = np.histogram(drops, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get the current color
        color = colors[color_index]

        # Separated the pre and post lines
        shift = 1e2 * index

        # Plot the histogram with marker only, but don't show in legend
        ax.scatter(
            bin_centers,
            hist / (shift if index else 1),
            marker=markers[index],
            facecolors="none",
            edgecolors=color,
            s=100,
            zorder=-(color_index * 2 + 1),
        )

        # Plot the line only, but don't show in legend
        ax.plot(
            bin_centers,
            fit.truncated_power_law.pdf(bin_centers) / (shift if index else 1),
            line_styles[index],
            label="_nolegend_",
            color=color,
            zorder=-color_index * 2,
        )

        # Create a dummy plot for the legend with both marker and line style
        ax.plot(
            [],
            [],
            line_styles[index],
            marker=markers[index],
            label=rf"{label.split(' seed')[0]} {part_label} $\alpha$={fit.truncated_power_law.alpha:.2f}, $\lambda$={fit.truncated_power_law.Lambda:.2f}",
            color=color,
        )
        print(f"{label} {part_label}: {fit.truncated_power_law.alpha}")

    # Increment the global call count
    color_index += 1

    # Check if we've used all colors
    if color_index >= len(colors):
        # Switch to the next line style and marker
        color_index = 0

    # Set axes to logarithmic
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig, ax


def plotSlidingPowerLaw(
    y_values,
    fig=None,
    ax=None,
    label="",
):
    global color_index, index, line_index
    if ax is None:
        fig, ax = plt.subplots()

    # Process drops
    def pros_d(y_values, start, end):
        diffs = np.diff(y_values[start:end])
        return -diffs[diffs < 0]

    widths = [int(0.25 * len(y)) for y in y_values]
    intervals = 10
    start_values = [
        np.linspace(0, len(y - w), intervals).astype(int)
        for y, w in zip(y_values, widths)
    ]
    print(list(map(len, start_values)))
    drops_series_starts = [
        [
            pros_d(np.array(y), s[i], s[i] + w)
            for y, w, s in zip(y_values, widths, start_values)
        ]
        for i in range(intervals)
    ]
    combined_drops = [np.concatenate(drops) for drops in drops_series_starts]

    settings = []

    settings_file = "/tmp/settings.pkl"

    # Check if the settings file already exists
    if os.path.exists(settings_file) and False:
        with open(settings_file, "rb") as file:
            settings = pickle.load(file)
    else:
        for drops_series in combined_drops:
            lamb = []
            alpha = []
            for drops in drops_series:
                if len(drops) == 0:
                    continue

                # Configure the analysis range
                xmin, xmax = np.min(drops), np.max(drops)

                fit = powerlaw.Fit(drops, xmin=xmin, xmax=xmax, fit_method="Likelihood")
                alpha.append(fit.truncated_power_law.alpha)
                lamb.append(fit.truncated_power_law.Lambda)
            settings.append((np.array(lamb), np.array(alpha)))

        # Save the settings to a file
        with open(settings_file, "wb") as file:
            pickle.dump(settings, file)
    np.array(settings)
    ax.plot(
        range(len(settings)),
        settings[:, 0, 0],
        marker="o",
    )
    return fig, ax


def makePlot(
    csv_file_paths,
    name,
    X=None,
    Y=None,
    x_name=None,
    y_name=None,
    labels=None,
    title=None,
    plot_average=False,
    xLims=(-np.inf, np.inf),
    yLims=(-np.inf, np.inf),
    indicateLastPoint=False,
    plot_roll_average=False,
    plot_raw=True,
    plot_power_law=False,
    plot_columns=False,
    ylog=False,
    show=False,
    colors=None,
    plot_total=False,
    legend=None,
    addShift=False,
):
    if len(csv_file_paths) == 0 or (
        len(csv_file_paths) > 0 and len(csv_file_paths[0]) == 0
    ):
        print("No files provided.")
        return
    if x_name is None:
        if X == "Load":
            x_name = r"$\alpha$"
        else:
            x_name = X

    if y_name is None:
        y_name = Y

    # if we are not given a list, we make it into a list
    if isinstance(csv_file_paths, str):
        csv_file_paths = [csv_file_paths]

    fig, ax = plt.subplots()
    lines = []
    data = []
    for i, csv_file_path in enumerate(csv_file_paths):
        if X is None:
            break
        """
        /Volumes/data/MTS2D_output/simpleShear,s20x20l0.15,1e-05,1.0PBCt4minimizerLBFGSLBFGSEpsg1e-06s0/macroData.csv
        /Volumes/data/MTS2D_output/simpleShear,s20x20l0.15,1e-05,1PBCt4minimizerLBFGSLBFGSEpsg1e-06s0/macroData.csv
        """
        if isinstance(Y, str):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
        else:
            df = pd.read_csv(csv_file_path, usecols=[X] + Y)
            if plot_average:
                raise Warning("Cannot plot average with multiple Y columns")

        # Truncate data based on Lims
        df = df[(df[X] >= xLims[0]) & (df[X] <= xLims[1])]
        df = df[(df[Y] >= yLims[0]) & (df[Y] <= yLims[1])]

        data.append(df[Y].values)

        kwargs = {"fig": fig, "ax": ax, "indicateLastPoint": indicateLastPoint}

        if colors:
            kwargs["color"] = colors[i]
        if len(csv_file_paths) > 10:
            kwargs["color"] = "gray"
            kwargs["alpha"] = 0.5

        for Y_ in [Y] if isinstance(Y, str) else Y:
            if len(df[Y_]) == 0:
                continue
            if labels is None:
                kwargs["label"] = Y
            else:
                kwargs["label"] = labels[i] + (
                    (" - " + Y_) if not isinstance(Y, str) else ""
                )
            if addShift:
                df[Y_] -= i * np.max(df[Y_]) / 500
            line = None
            point = None
            if plot_raw:
                fig, ax, line, point = plotYOverX(df[X], df[Y_], **kwargs)
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
        df = pd.read_csv(csv_file_paths[max_length_index], usecols=[X])
        df = df[(df[X] >= xLims[0]) & (df[X] <= xLims[1])]
        if plot_raw:
            fig, ax, line, point = plotYOverX(df[X], average, **kwargs)
        lines.append(line)

    if plot_power_law:
        kwargs = {
            "fig": fig,
            "ax": ax,
            "label": "Fit",
            "color": "black",
            "include_label": legend,
        }
        fig, ax, line = plotPowerLaw(data, **kwargs)

    cursor = mplcursors.cursor(lines)  # noqa: F841
    # cursor.connect(
    #   "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    if ylog:
        ax.set_yscale("log")

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
    if legend:
        ax.legend(line_handles, line_labels, loc=("best"))
    if isinstance(legend, list):
        custom_legend = [
            mlines.Line2D(
                [], [], color="blue", marker="o", linestyle="-", label="Custom Label 1"
            ),
            mlines.Line2D(
                [],
                [],
                color="green",
                marker="o",
                linestyle="--",
                label="Custom Label 2",
            ),
        ]
        # Create the legend with custom labels
        ax.legend(handles=custom_legend)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if title is None:
        ax.set_title(f"{y_name} over {x_name}")
    else:
        ax.set_title(title)
    ax.autoscale_view()
    figPath = os.path.join(os.path.dirname(csv_file_paths[0]), name)
    fig.savefig(figPath)
    print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()


def makeEnergyPlotComparison(grouped_csv_file_paths, name, show=True, **kwargs):
    X = "Load"
    Y = "Avg energy"
    x_name = "Load"
    y_name = "Avg energy"
    title = f"{name}"

    fig, ax = plt.subplots()

    color_index = -1
    line_index = 0

    crash_count = 0

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
        color = colors[color_index]

        for j, csv_file_path in enumerate(csv_file_paths):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
            if df.empty:
                continue
            # Check for Y values greater than 10 and truncate
            if (df[Y] > 10).any():
                crash_count += 1
                df = df[df[Y] <= 1]
                print(f"Crash in {csv_file_path}.")

            data.append(df[Y].values)

            e_kwargs = {
                "fig": fig,
                "ax": ax,
                "color": color,
                "linestyle": line_styles[line_index],
                "alpha": 0.05,
                "zorder": -color_index,
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
            "label": kwargs["labels"][i][0].split(" seed")[0],
            "color": colors[color_index],
            "linestyle": line_styles[line_index],
            "zorder": -color_index,
        }
        df = pd.read_csv(csv_file_paths[max_length_index], usecols=[X])
        fig, ax, line, point = plotYOverX(df[X], average, **a_kwargs)
    if crash_count > 0:
        print(f"Found {crash_count} crashes.")
    # Set the legend with the filtered handles and labels
    ax.legend(loc=("best"))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.autoscale_view()
    figPath = os.path.join(os.path.dirname(grouped_csv_file_paths[0][0]), name + ".pdf")
    fig.savefig(figPath)
    print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()


def makeLogPlotComparison(
    grouped_csv_file_paths,
    name,
    xLims=[-np.inf, np.inf],
    show=True,
    slide=False,
    **kwargs,
):
    X = "Load"
    Y = "Avg energy"
    x_name = "Magnitude of energy drops"
    y_name = "Frequency"
    lims = "" if xLims == [-np.inf, np.inf] else f", xLims: {xLims[0]}-{xLims[1]}"
    title = f"{name}{lims}"

    fig, ax = plt.subplots()

    crash_count = 0
    for i, csv_file_paths in enumerate(grouped_csv_file_paths):
        data = []
        for j, csv_file_path in enumerate(csv_file_paths):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
            df = df[
                (df[X] >= xLims[0]) & (df[X] <= xLims[1])
            ]  # Truncate data based on xLims
            # Check for Y values greater than 10 and truncate
            if (df[Y] > 10).any():
                crash_count += 1
                df = df[df[Y] <= 1]
            data.append(df[Y].values)
        log_kwargs = {
            "fig": fig,
            "ax": ax,
            "label": kwargs["labels"][i][j],
        }
        if slide:
            fig, ax = plotSlidingPowerLaw(data, **log_kwargs)
        else:
            fig, ax = plotPowerLaw(data, **log_kwargs)
    if crash_count > 0:
        print(f"Found {crash_count} crashes.")
    # Set the legend with the filtered handles and labels
    ax.legend(loc=("best"))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.autoscale_view()
    figPath = os.path.join(os.path.dirname(grouped_csv_file_paths[0][0]), name + ".pdf")
    fig.savefig(figPath)
    print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()


def makeEnergyPlot(csv_file_paths, name, **kwargs):
    makePlot(csv_file_paths, name, X="Load", Y="Avg energy", **kwargs)


def makeItterationsPlot(csv_file_paths, name, **kwargs):
    if isinstance(csv_file_paths, str):
        makePlot(
            csv_file_paths,
            name,
            X="Load",
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
            X="Load",
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
        Y=["Run time"],
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
        X="Load",
        Y="Avg energy",
        x_name="Magnitude of energy drops",
        y_name="Frequency",
        title="Powerlaw",
        plot_raw=False,
        plot_power_law=True,
        plot_average=False,
        **kwargs,
    )


if __name__ == "__main__":
    pass
    # The path should be the path from work directory to the folder inside the output folder.
    makeEnergyPlot(
        [
            "/Volumes/data/MTS2D_output/simpleShear,s60x60l0.15,0.0002,1.0PBCt1minimizerFIRELBFGSEpsg0.0001eps0.01s0/macroData.csv",
        ],
        name="energy.pdf",
    )
    # makeItterationsPlot(
    #     [
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
    #     ],
    #             name='nrIterations.pdf')

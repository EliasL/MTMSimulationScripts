from matplotlib import pyplot as plt
import matplotlib
import os
import matplotlib.pylab
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import re
from pathlib import Path
import mplcursors
from datetime import timedelta
import powerlaw


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


def plotPowerLaw(
    y_values_series,
    fig=None,
    ax=None,
    include_label=False,
    indicate_last_point=True,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    # Extract and prepare the data
    combined_diffs = np.concatenate(
        [
            -np.diff(np.array(y_values))[np.diff(np.array(y_values)) < 0]
            for y_values in y_values_series
        ]
    )

    # Configure the analysis range
    xmin, xmax = 1e-6, np.max(combined_diffs)
    fit = powerlaw.Fit(combined_diffs, xmin=xmin, xmax=xmax, fit_method="Likelihood")

    # Set up bins and plot histogram
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 12)
    hist, bin_edges = np.histogram(combined_diffs, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, hist, linestyle="None", marker=".", markersize=20, color="b")

    # Plot the fitted power law
    ax.plot(
        bin_centers,
        fit.truncated_power_law.pdf(bin_centers),
        "--",
        color="r",
        label=rf"Truncated Power law PDF $\alpha$={fit.truncated_power_law.alpha:.2f}, $\lambda$={fit.truncated_power_law.Lambda:.2f}",
    )

    # Set axes to logarithmic
    ax.set_xscale("log")
    ax.set_yscale("log")
    return fig, ax, None


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
):
    if len(csv_file_paths) == 0:
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
    if type(csv_file_paths) == type(""):
        csv_file_paths = [csv_file_paths]

    fig, ax = plt.subplots()
    lines = []
    data = []
    for i, csv_file_path in enumerate(csv_file_paths):
        if X is None:
            break
        if isinstance(Y, str):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
            df = df[
                (df[X] >= xLims[0]) & (df[X] <= xLims[1])
            ]  # Truncate data based on xLims

            data.append(df[Y].values)
        else:
            df = pd.read_csv(csv_file_path, usecols=[X] + Y)
            df = df[
                (df[X] >= xLims[0]) & (df[X] <= xLims[1])
            ]  # Truncate data based on xLims
            if plot_average:
                raise Warning("Cannot plot average with multiple Y columns")

        kwargs = {"fig": fig, "ax": ax, "indicateLastPoint": indicateLastPoint}

        if colors:
            kwargs["color"] = colors[i]
        if len(csv_file_paths) > 7:
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

    cursor = mplcursors.cursor(lines)
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
    if legend == True:
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
        x_name=f"Magnitude of energy drops",
        y_name="Frequency",
        title=f"Powerlaw",
        plot_raw=False,
        plot_power_law=True,
        plot_average=False,
        **kwargs,
    )


if __name__ == "__main__":
    pass
    # The path should be the path from work directory to the folder inside the output folder.
    # makeEnergyPlot([
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
    #     ],
    #                name='energy.pdf')
    # makeItterationsPlot(
    #     [
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
    #         '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
    #     ],
    #             name='nrIterations.pdf')

from matplotlib import pyplot as plt
import matplotlib
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import mplcursors
from datetime import timedelta


from simplification.cutil import simplify_coords_vwp


def plotYOverX(X, Y, fig=None, ax=None, sub=0, indicateLastPoint=True, tolerance=1e-12, **kwargs):
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # Simplify the points if tolerance is provided
    if tolerance is not None:
        points = np.column_stack((X, Y))
        simplified_points = simplify_coords_vwp(points, tolerance)
        X_simplified = simplified_points[:, 0]
        Y_simplified = simplified_points[:, 1] - sub
    else:
        X_simplified = X
        Y_simplified = Y - sub
    # Plot on the provided axis
    line, = ax.plot(X_simplified, Y_simplified, **kwargs)
    
    if indicateLastPoint:
        # Add a scatter point at the last point
        point = ax.scatter(X_simplified[-1], 
                           Y_simplified[-1], 
                           **kwargs)
    else:
        point = None

    # Return the axis object for further use
    return fig, ax, line, point

def time_to_seconds(duration_str):
    pattern = r'(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?'
    matches = re.match(pattern, duration_str.strip())
    
    if not matches:
        return timedelta()
    
    days, hours, minutes, seconds = matches.groups(default='0')
    
    return timedelta(
        days=int(days),
        hours=int(hours),
        minutes=int(minutes),
        seconds=float(seconds)
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

def plotIntervalAverage(X, Y, intervalSize=100, fig=None,
                         ax=None, sub=0, **kwargs):

    # Calculate rolling average
    rollingMean = Y.rolling(window=intervalSize, min_periods=1, center=True).mean()

    # Check if axis provided, if not, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    # Plotting the interval average
    line, = ax.plot(X, rollingMean-sub, label=f'Rolling average (window={intervalSize})', **kwargs)

    # Return the axis object for further use
    return fig, ax, line

def plotPowerLaw(Y, fig=None, ax=None, sub=0, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if sub!=0:
        # It seems strange to do this
        raise Warning("Are you sure?")

    #Convert Y to np array
    Y=np.array(Y)

    # Calculate the difference in Y (and adjust Y if desired)
    Y_diff = np.diff(Y - sub)

    # We only care about the negative drops
    # and then we take abs to plot in log plot
    Y_diff = np.abs(Y_diff[Y_diff < 0])

    if(len(Y_diff)==0):
        return fig, ax, None

    # Create a histogram of the absolute values of Y differences
    bins = np.logspace(-7, 0, num=20, base=10)
    counts, bin_edges = np.histogram(Y_diff, bins=bins)
    
    # Remove bins with zero counts
    non_zero_indices = counts > 0
    counts = counts[non_zero_indices]
    bin_edges = bin_edges[:-1][non_zero_indices]  # Shift to match the counts length correctly

    probability = counts / sum(counts)
    
    # Plot on a log-log scale
    line, = ax.loglog(bin_edges, probability, **kwargs)
    #print(counts)
    # Return the figure, axis, and line for further use
    return fig, ax, line


def makePlot(csv_file_paths, name, X=None, Y=None, x_name=None, y_name=None, labels=None,
             title=None, 
             plot_average=False, plot_raw=True, plot_power_law=False, plot_columns=False, ylog=False,
             subtract=None, show=False, colors=None, SUM=False, legend=None):
    if x_name is None:
        if X == 'Load':
            x_name = r'$\alpha$'
        else:
            x_name=X
            
    if y_name is None:
        y_name = Y

    # if we are not given a list, we make it into a list
    if type(csv_file_paths)==type(""):
        csv_file_paths = [csv_file_paths]

    if subtract is not None:
        if isinstance(Y, str):
            sub = pd.read_csv(subtract, usecols=[X, Y])
        else:
            sub = pd.read_csv(subtract, usecols=[X]+Y)
    else:
        sub = 0

    fig, ax = plt.subplots()
    lines =[]
    for i, csv_file_path in enumerate(csv_file_paths):
        if X is None:
            break
        if isinstance(Y, str):
            df = pd.read_csv(csv_file_path, usecols=[X, Y])
        else:
            df = pd.read_csv(csv_file_path, usecols=[X]+Y)

        if not isinstance(sub, (int, float)):
            sub_ = np.interp(df[X], sub[X], sub[Y])
        else:
            sub_ = sub

            
        kwargs = {'fig': fig, 'ax': ax, 'sub': sub_}

        if colors:
            kwargs['color'] = colors[i]

        for Y_ in ([Y] if isinstance(Y, str) else Y):
            if len(df[Y_])==0:
                continue
            kwargs['label'] = labels[i] + ((" - " + Y_) if not isinstance(Y, str) else "")
            line=None
            point=None
            if plot_raw:
                fig, ax, line, point = plotYOverX(df[X], df[Y_], **kwargs)
            if plot_average:
                fig, ax, line = plotIntervalAverage(df[X], df[Y_], **kwargs)
            if plot_power_law:
                fig, ax, line = plotPowerLaw(df[Y_], **kwargs)
            if line is not None:
                lines.append(line)
            if point is not None:
                lines.append(point)
        if SUM:
            assert not isinstance(Y, str)
            if not plot_raw:
                kwargs['label']='total'
            fig, ax, line = plotYOverX(df[X], sum([df[Y_] for Y_ in Y]), **kwargs)

    if plot_columns:
        fig, ax = plotColumns(csv_file_paths, Y, labels, fig, ax)

    cursor = mplcursors.cursor(lines)
    #cursor.connect(
     #   "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    if ylog:
        ax.set_yscale('log')

    # Create a list of line plots only for the legend
    handles, labels = ax.get_legend_handles_labels()
    line_handles = [handle for handle in handles if isinstance(handle, matplotlib.lines.Line2D)]

    # Filter labels accordingly
    line_labels = [label for handle, label in zip(handles, labels) if isinstance(handle, matplotlib.lines.Line2D)]

    # Set the legend with the filtered handles and labels
    if (len(line_labels) < 10 and legend is None) or legend:

        ax.legend(line_handles, line_labels, loc=('best' if legend else 'right'))

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if title is None:
        ax.set_title(f'{y_name} over {x_name}')
    else:
        ax.set_title(title)
    ax.grid(True)
    ax.autoscale_view()
    figPath = os.path.join(os.path.dirname(csv_file_paths[0]), name)
    fig.savefig(figPath)
    print(f'Plot saved at: "{figPath}"')
    if show:
        plt.show()

def makeEnergyPlot(csv_file_paths, name, **kwargs):
    makePlot(csv_file_paths, name, X='Load', Y='Avg energy', **kwargs)

def makeItterationsPlot(csv_file_paths, name, **kwargs):
    if isinstance(csv_file_paths, str):
        makePlot(csv_file_paths, name, X='Load', Y=['Nr FIRE iterations', 'Nr LBFGS iterations'],
                y_name="Nr itterations", title='Nr of Itterations', plot_raw=True, 
                plot_average=False, SUM=True, **kwargs)
    else:
        makePlot(csv_file_paths, name, X='Load', Y=['Nr FIRE iterations', 'Nr LBFGS iterations'],
                y_name="Nr itterations", title='Nr of Itterations', plot_raw=True, 
                plot_average=False, SUM=False, **kwargs)
        
def makeTimePlot(csv_file_paths, name, **kwargs):
    makePlot(csv_file_paths, name, x_name='Settings', Y=['Run time'], y_name='Run time (s)', plot_raw=False,
             plot_columns=True, title='Runtime of simulations', **kwargs)


def makePowerLawPlot(csv_file_paths, name, **kwargs):
    makePlot(csv_file_paths, name, X='Load', Y='Avg energy',
             x_name=f'Size of energy drops', y_name='Probability (Frequency/Total)',
            title=f'Log-log plot of energy drops',
             plot_raw=False, plot_average=False, plot_power_law=True, **kwargs)


if __name__ == "__main__":
    # The path should be the path from work directory to the folder inside the output folder. 
    makeEnergyPlot([
            '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
            '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
        ],
                   name='energy.pdf')
    makeItterationsPlot(
        [
            '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/macroData.csv',
            '/Volumes/data/MTS2D_output/simpleShearPeriodicBoundary,s60x60l0.15,1e-05,20PBCt4s0/FullMacroData.csv',
        ],
                name='nrIterations.pdf')

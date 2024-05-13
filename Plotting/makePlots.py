from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
from pathlib import Path
import mplcursors


def plotYOverX(csv_file_path, X, Y, df=None, fig=None, ax=None, sub=0,
                **kwargs):
    # Load data
    if df is None:
        df = pd.read_csv(csv_file_path, usecols=[X, Y])
    
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot on the provided axis
    line, = ax.plot(df[X], df[Y]-sub, **kwargs)
    
    # Return the axis object for further use
    return fig, ax, line

def plotIntervalAverage(csv_file_path, X, Y, df=None, intervalSize=100, fig=None,
                         ax=None, sub=0, **kwargs):
    # Load data
    if df is None:
        df = pd.read_csv(csv_file_path, usecols=[X, Y])

    # Calculate rolling average
    df['RollingMean'] = df[Y].rolling(window=intervalSize, min_periods=1, center=True).mean()

    # Check if axis provided, if not, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    # Plotting the interval average
    line, = ax.plot(df[X], df['RollingMean']-sub, label=f'Rolling average (window={intervalSize})', **kwargs)

    # Return the axis object for further use
    return fig, ax, line

def makePlot(csv_file_paths, name, X, Y, x_name=None, y_name=None, labels=None, plot_average=False, 
             plot_raw=True, ylog=False, subtract=None, show=False):
    if x_name is None:
        if X == 'Load':
            x_name = r'$\alpha$'
        else:
            x_name=X
            
    if y_name is None:
        y_name = Y
    
    if labels is None:
        labels = y_name

    # if we are not given a list, we make it into a list
    if type(csv_file_paths)==type(""):
        csv_file_paths = [csv_file_paths]

    if subtract is not None:
        sub = pd.read_csv(subtract, usecols=[X, Y])
    else:
        sub = 0

    fig, ax = plt.subplots()
    lines =[]
    for i, csv_file_path in enumerate(csv_file_paths):
        df = pd.read_csv(csv_file_path, usecols=[X, Y])

        if not isinstance(sub, (int, float)):
            sub_ = np.interp(df[X], sub[X], sub[Y])
        else:
            sub_ = sub
        if not isinstance(labels, str):
            label = labels[i]
        else:
            label = labels
            
        kwargs = {'label': label, 'fig': fig, 'ax': ax, 'sub': sub_}

        if plot_raw:
            fig, ax, line = plotYOverX(csv_file_path, X, Y, df, **kwargs)
        if plot_average:
            fig, ax, line = plotIntervalAverage(csv_file_path, X, Y, df, **kwargs)

        lines.append(line)

    cursor = mplcursors.cursor(lines)

    #cursor.connect(
     #   "add", lambda sel: sel.annotation.set_text(labels[sel.index]))

    if ylog:
        ax.set_yscale('log')

    ax.legend()
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f'{y_name} over {x_name}')
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
    makePlot(csv_file_paths, name, X='Load', Y='Nr iterations',
             plot_raw=True, plot_average=False, **kwargs)


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

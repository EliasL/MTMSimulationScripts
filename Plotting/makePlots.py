from matplotlib import pyplot as plt
import os
import pandas as pd
 
def plotYOverX(csv_file_path, X, Y, df=None, fig=None, ax=None, **kwargs):
    # Load data
    if df is None:
        df = pd.read_csv(csv_file_path, usecols=[X, Y])
    
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot on the provided axis
    ax.plot(df[X], df[Y], **kwargs)
    
    # Return the axis object for further use
    return fig, ax

def plotIntervalAverage(csv_file_path, X, Y, df=None, intervalSize=100, fig=None, ax=None, **kwargs):
    # Load data
    if df is None:
        df = pd.read_csv(csv_file_path, usecols=[X, Y])

    # Calculate rolling average
    df['RollingMean'] = df[Y].rolling(window=intervalSize, min_periods=1, center=True).mean()

    # Check if axis provided, if not, create a new one
    if ax is None:
        fig, ax = plt.subplots()

    # Plotting the interval average
    ax.plot(df[X], df['RollingMean'], label=f'Rolling average (window={intervalSize})', **kwargs)

    # Return the axis object for further use
    return fig, ax

def makePlot(csv_file_paths, name, X, Y, x_name=None, y_name=None, plot_average=False, plot_raw=True, ylog=False):
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


    fig, ax = plt.subplots()
    for i, csv_file_path in enumerate(csv_file_paths):
        df = pd.read_csv(csv_file_path, usecols=[X, Y])
        #df = df[(df[X] >= 6) & (df[X] <= 6.5)]
 
        if plot_raw:
            fig, ax = plotYOverX(csv_file_path, X, Y, df,label=y_name, fig=fig, ax=ax, linewidth=3-i*2)
        if plot_average:
            fig, ax = plotIntervalAverage(csv_file_path, X, Y, df, fig=fig, ax=ax)

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
    print(f"Plot saved at: {figPath}")
    #plt.show()

def makeEnergyPlot(csv_file_paths, name):
    makePlot(csv_file_paths, name, X='Load', Y='Avg energy')

def makeItterationsPlot(csv_file_paths, name):
    makePlot(csv_file_paths, name, X='Load', Y='Nr iterations',
             plot_raw=True, plot_average=False)

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

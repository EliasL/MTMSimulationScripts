from matplotlib import pyplot as plt
import os
import pandas as pd
 
def plotEnergyOverLoad(csv_file_path, ax=None, **kwargs):
    # Load data
    df = pd.read_csv(csv_file_path, usecols=['Load', 'Avg. energy'], dtype={'Load': 'float64', 'Avg. energy': 'float64'})
    
    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot on the provided axis
    ax.plot(df['Load'], df['Avg. energy'], **kwargs)
    
    # Return the axis object for further use
    return ax

def makeSinglePlot(csv_file_path, name="energy.pdf", destination=None):
    print("Plotting...")

    fig, ax = plt.subplots()
    plotEnergyOverLoad(csv_file_path, ax)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Energy')
    ax.set_title(r'Average energy over stress $\alpha$')
    ax.grid(True)

    ax.autoscale_view()
    
    # Default destination
    if destination is None:
        destination = os.path.dirname(csv_file_path)

    figPath = os.path.join(destination, name)
    fig.savefig(figPath)
    print(f"Plot saved at: {figPath}")
    #plt.show()

if __name__ == "__main__":
    # The path should be the path from work directory to the folder inside the output folder. 
    makeSinglePlot('/media/elias/dataStorage/output/S100x100L0.15,1e-05,1t4n0.05M10s0/macroData.csv')

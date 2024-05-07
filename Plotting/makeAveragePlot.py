import sys
from pathlib import Path
from settings import settings
import pandas as pd
import matplotlib.pyplot as plt
import re

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / 'Management'))

# Now we can import from Management
from configGenerator import ConfigGenerator

# We want to plot several runs ontop of each other
# In order to find the folders, we use the name generator in ConfigGenerator

seeds = range(0,11)
configs = ConfigGenerator.generate_over_seeds(seeds, rows=100, cols=100, startLoad=0.15, 
                          loadIncrement=0.00001, maxLoad=1)


nrCorrections = [1, 3, 5, 7, 10]
configs = ConfigGenerator.generate_over_("nrCorrections", nrCorrections,
                                            rows=100, cols=100, startLoad=0.15,
                                            loadIncrement=0.00001, maxLoad=0.7,
                                            nrThreads=1)

outPath = "/media/elias/dataStorage/output/"
macroPath = f"/{settings['MACRODATANAME']}.csv"
filePaths = [outPath + config.generate_name(False) + macroPath for config in configs]

def make_average_plot(name, filePaths):
    # Initialize a list to store DataFrames for each run
    dfs = []

    # Set up the plot for all runs and the average
    plt.figure(figsize=(6, 5))

    # Loop through each file path to read CSV data
    for (i,filePath) in enumerate(filePaths):
        # Read the CSV data into a DataFrame
        df = pd.read_csv(filePath, usecols=['Load', 'Avg energy'], dtype={'Load': 'float64', 'Avg energy': 'float64'})
        dfs.append(df)

        # Plot individual run
        plt.plot(df['Load'], df['Avg energy'], label=f'seed: {i}', alpha=0.5) 

    # Concatenate all DataFrames for average calculation
    combined_df = pd.concat(dfs)

    # Group by Load and calculate mean of Avg energy for the average plot
    average_df = combined_df.groupby('Load')['Avg energy'].mean().reset_index()

    # Plot combined average with a distinct color and label
    plt.plot(average_df['Load'], average_df['Avg energy'], label='Average', color='black', linewidth=2)

    # Customize the plot
    plt.xlabel('Load')
    plt.ylabel('Average Energy')
    plt.title('Energy vs. Load + Average')
    plt.legend()
    plt.grid(True)

    # Save the plot
    visuals_folder = str(Path(__file__).resolve().parent.parent / 'Plots')

    # Local paths will be called macroData
    fileName = Path(filePaths[0]).parent.name
    if fileName == "macroData":
        # but downloaded files will be called the config name
        fileName = Path(filePaths[0].stem)
    
    filePath = visuals_folder+f"/Seeds/{name}_{fileName}.pdf" 
    plt.savefig(filePath)
    print(f"saved figure to {filePath}")

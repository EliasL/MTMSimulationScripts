from makeAnimations import makeAnimations
from makePlots import makeSinglePlot
from makeEnergyField import makeEnergyField
from settings import settings
import sys
from pathlib import Path

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / 'Management'))
# Now we can import from Management
from simulationManager import findOutputPath


if __name__ == "__main__":

    if len(sys.argv) < 2:
        raise Exception("Config file is required!")
        

    # We expect the argument to be path/name.conf, and we want just the name
    subfolderName = Path(sys.argv[1]).stem
    if len(sys.argv) < 3:
        print("Attempting to automatically find output path")
        dataPath = findOutputPath()
    else:
        dataPath = sys.argv[2]
    
    collection = f"{settings['COLLECTIONNAME']}.pvd"
    macroData = f"{settings['MACRODATANAME']}.csv"

    path = dataPath+subfolderName+'/'

    makeSinglePlot(path+macroData)
    makeAnimations(path, collection)

    # energyGridName = "energy_grid.csv"
    # makeEnergyField(path, energyGridName)

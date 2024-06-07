from makeAnimations import makeAnimations
from makePlots import makeEnergyPlot, makeItterationsPlot  # noqa: F401
from makeEnergyField import makeEnergyField  # noqa: F401
from settings import settings
import sys
import os
from pathlib import Path

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Management"))
# Now we can import from Management
from simulationManager import findOutputPath
from configGenerator import SimulationConfig


def plotAll(configFile, dataPath, noVideo=False):
    if configFile[0] != "/":
        # This means that the config file is relative
        configFile = os.path.join(dataPath, configFile)

    conf = SimulationConfig(configFile)
    subfolderName = conf.name

    macroData = f"{settings['MACRODATANAME']}.csv"

    path = os.path.join(dataPath, subfolderName)
    # path = '/Volumes/data/KeepSafe/150x150FireVsLBFGS/simpleShear,s150x150l0.15,1e-05,1PBCt4LBFGSEpsX1e-06s0/'
    # subfolderName = 'simpleShear,s150x150l0.15,1e-05,1PBCt4LBFGSEpsX1e-06s0'
    print(f"Plotting at {path}")
    makeEnergyPlot(os.path.join(path, macroData), subfolderName + "_energy.pdf")
    # makeItterationsPlot(path+macroData, subfolderName+"_itterations.pdf")
    if not noVideo:
        makeAnimations(path)

    # energyGridName = "energy_grid.csv"
    # makeEnergyField(path, energyGridName)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Config file is required!")

    if len(sys.argv) < 3:
        print("Attempting to automatically find output path")
        dataPath = findOutputPath()
    else:
        dataPath = sys.argv[2]

    noVideo = False
    if len(sys.argv) >= 4:
        noVideo = sys.argv[3] == "noVideo"

    plotAll(sys.argv[1], dataPath, noVideo)

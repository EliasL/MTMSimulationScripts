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


def plotAll(configFile, dataPath, noVideo=False, **kwargs):
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
        makeAnimations(path, **kwargs)

    # energyGridName = "energy_grid.csv"
    # makeEnergyField(path, energyGridName)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-c", "--config", required=True, help="Config file")
    parser.add_argument("-o", "--output", default=None, help="Data path")
    parser.add_argument("-nV", "--noVideo", action="store_true", help="Disable video")
    parser.add_argument(
        "-gif", "--makeGIF", action="store_true", default=False, help="Make gif"
    )

    args = parser.parse_args()

    outputPath = args.output if args.output else findOutputPath()

    plotAll(args.config, outputPath, args.noVideo, makeGIF=args.makeGIF)

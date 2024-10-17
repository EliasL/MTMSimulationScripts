from makeAnimations import makeAnimations
from makePlots import makePlot, makeItterationsPlot  # noqa: F401
from settings import settings
import sys
import os
from pathlib import Path

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Management"))
# Now we can import from Management
from simulationManager import findOutputPath
from configGenerator import SimulationConfig


def plotAll(configFile, noVideo=False, **kwargs):
    conf = SimulationConfig(configFile)
    subfolderName = conf.name

    macroData = f"{settings['MACRODATANAME']}.csv"

    path = Path(configFile).parent
    # path = '/Volumes/data/KeepSafe/150x150FireVsLBFGS/simpleShear,s150x150l0.15,1e-05,1PBCt4LBFGSEpsX1e-06s0/'
    # subfolderName = 'simpleShear,s150x150l0.15,1e-05,1PBCt4LBFGSEpsX1e-06s0'
    print(f"Plotting at {path}")
    csvPath = os.path.join(path, macroData)
    makePlot(csvPath, subfolderName + "_energy.pdf", Y="Avg energy")
    makePlot(csvPath, subfolderName + "_stress.pdf", Y="Avg RSS")

    # makeItterationsPlot(path+macroData, subfolderName+"_itterations.pdf")
    if not noVideo:
        makeAnimations(path, **kwargs)

    makePlot(
        csvPath,
        subfolderName + "_stress+.pdf",
        Y="Avg RSS",
        add_images=True,
        labels=conf.minimizer,
    )

    # energyGridName = "energy_grid.csv"
    # makeEnergyField(path, energyGridName)


def handle_args_and_plot():
    import argparse

    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-c", "--config", required=True, help="Config file")
    parser.add_argument("-nV", "--noVideo", action="store_true", help="Disable video")
    parser.add_argument(
        "-t", "--transparent", action="store_true", help="Make transparent videoes"
    )
    parser.add_argument(
        "-gif", "--makeGIF", action="store_true", default=False, help="Make gif"
    )

    args = parser.parse_args()

    plotAll(args.config, args.noVideo, makeGIF=args.makeGIF)


if __name__ == "__main__":
    handle_args_and_plot()

    # outputPath = findOutputPath()
    # # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    # config = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    # plotAll(config, outputPath, makeGIF=False, transparent=False)

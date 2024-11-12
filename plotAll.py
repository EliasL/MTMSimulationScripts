from Plotting.makeAnimations import makeAnimations
from Plotting.makePlots import makePlot, makeItterationsPlot  # noqa: F401
from Plotting.settings import settings

import os
from pathlib import Path

# Now we can import from Management
from Management.simulationManager import findOutputPath
from Management.configGenerator import SimulationConfig


def plotAll(configFile, noVideos=False, noPlots=False, **kwargs):
    conf = SimulationConfig(configFile)
    subfolderName = conf.name

    macroData = f"{settings['MACRODATANAME']}.csv"

    path = Path(configFile).parent
    print(f"Plotting at {path}")
    csvPath = os.path.join(path, macroData)
    if not noPlots:
        makePlot(csvPath, name=subfolderName + "_energy.pdf", Y="Avg energy")
        makePlot(csvPath, name=subfolderName + "_stress.pdf", Y="Avg RSS")
        makePlot(
            csvPath,
            name=subfolderName + "_stress+.pdf",
            Y="Avg RSS",
            add_images=True,
            image_pos=[
                [0.35, 0.02],  # first image, bottom middle
                [0.03, 0.5],  # second image, upper left
                [0.6, 0.55],  # upper right
            ],
            labels=conf.minimizer,
        )

    # makeItterationsPlot(path+macroData, subfolderName+"_itterations.pdf")
    if not noVideos:
        makeAnimations(path, **kwargs)


def handle_args_and_plot():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-c", "--config", required=True, help="Config file")
    parser.add_argument("-nP", "--noPlots", action="store_true", help="Disable plots")
    parser.add_argument("-nV", "--noVideo", action="store_true", help="Disable video")
    parser.add_argument(
        "-t", "--transparent", action="store_true", help="Make transparent videoes"
    )
    parser.add_argument(
        "-gif", "--makeGIF", action="store_true", default=False, help="Make gif"
    )

    if len(sys.argv) > 1:
        args = parser.parse_args()
        plotAll(args.config, args.noVideo, makeGIF=args.makeGIF, use_tqdm=False)
    else:
        return


if __name__ == "__main__":
    handle_args_and_plot()

    # outputPath = findOutputPath()
    # # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    config = "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    plotAll(config, makeGIF=False, transparent=False, noPlots=True)

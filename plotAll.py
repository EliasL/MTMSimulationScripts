from Plotting.makeAnimations import makeAnimations
from Plotting.makePlots import makePlot, makeItterationsPlot  # noqa: F401
from Plotting.settings import settings

import os
from pathlib import Path

# Now we can import from Management
from Management.simulationManager import findOutputPath
from Management.configGenerator import SimulationConfig, ConfigGenerator


def plotAll(configFile=None, noVideos=False, noPlots=False, **kwargs):
    if configFile is None:
        raise ValueError("No config file!")
    conf = SimulationConfig(configFile)
    subfolderName = conf.name

    macroData = f"{settings['MACRODATANAME']}.csv"

    path = Path(configFile).parent
    print(f"Plotting at {path}")
    csvPath = os.path.join(path, macroData)
    if not noPlots:
        makePlot(csvPath, name=subfolderName + "_energy.pdf", Y="Avg_energy")
        makePlot(csvPath, name=subfolderName + "_stress.pdf", Y="Avg_RSS")
        makePlot(
            csvPath,
            name=subfolderName + "_stress+.pdf",
            Y="Avg_RSS",
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
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process plotting and video options.")

    # Add arguments
    parser.add_argument("-c", "--configFile", required=True, help="Config file")
    parser.add_argument(
        "--noPlots", action="store_true", help="Disable plots (default: False)"
    )
    parser.add_argument(
        "-nV",
        "--noVideos",
        action="store_true",
        help="Disable video creation (default: False)",
    )
    parser.add_argument(
        "-t",
        "--transparent",
        action="store_true",
        help="Make videos transparent (default: False)",
    )
    parser.add_argument(
        "--makeGIF", action="store_true", help="Create GIFs (default: False)"
    )
    parser.add_argument(
        "--reuseImages",
        type=bool,
        choices=[True, False],
        default=True,
        help="Reuse existing images (default: True)",
    )
    parser.add_argument(
        "--combineVideos",
        type=bool,
        choices=[True, False],
        default=True,
        help="Combine videos into one (default: True)",
    )
    parser.add_argument(
        "--allImages",
        type=bool,
        choices=[True, False],
        default=True,
        help="Use all images for the process (default: False)",
    )

    args = parser.parse_args()

    # Convert Namespace to dict for **kwargs usage
    kwargs = vars(args)

    # clean the inputs
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.strip()

    # Pass the arguments directly to plotAll
    plotAll(
        **kwargs,
        fps=30,
        seconds_per_unit_shear=2,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        handle_args_and_plot()
    else:
        # outputPath = findOutputPath()
        # # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"

        configs = [
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s42/config.conf",
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerCGLBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s42/config.conf",
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s42/config.conf",
        ]
        configs = [
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06energyDropThreshold1e-10s41/config.conf"
        ]
        configs = [
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.1,1e-05,1.0PBCt3initialGuessNoise1e-06LBFGSEpsg1e-08CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06energyDropThreshold1e-10s41/config.conf",
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.1,1e-05,1.1PBCt3initialGuessNoise1e-06LBFGSEpsg1e-08CGEpsg1e-05eps1e-05energyDropThreshold1e-10s41/config.conf",
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.1,1e-05,1.1PBCt3initialGuessNoise1e-06LBFGSEpsg1e-08CGEpsg1e-05eps1e-05energyDropThreshold1e-10s42/config.conf",
        ]

        configs = [
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/config.conf"
        ]

        configs = [
            "/Volumes/data/MTS2D_output/singleDislocationTest,s10x10l0.0,0.001,2.0NPBCt3epsR1e-06s0/config.conf"
        ]

        for c in configs:
            plotAll(
                c,
                makeGIF=False,
                transparent=False,
                noPlots=True,
                combineVideos=True,
                fps=60,
                seconds_per_unit_shear=2,
                allImages=True,
                reuseImages=True,
            )

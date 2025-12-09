from Plotting.makeAnimations import makeAnimations
from Plotting.makePlots import makePlot, makeItterationsPlot
from Plotting.settings import settings
from Plotting.dataFunctions import parse_pvd_file, get_data_from_name
from Plotting.makePvd import create_collection

import os
from pathlib import Path

# Now we can import from Management
from Management.simulationManager import findOutputPath
from Management.configGenerator import SimulationConfig, ConfigGenerator

from matplotlib import pyplot as plt


def plotAll(unkownFile=None, noVideos=False, noPlots=False, **kwargs):
    # File can be either a .conf, .pvd or .csv file
    conf, csvPath, pvdFile = None, None, None
    X = "load"
    ylog = False
    subfolderName = "unkown_"

    if unkownFile is not None:
        if unkownFile.endswith(".conf"):
            conf = SimulationConfig(unkownFile)
        elif unkownFile.endswith(".pvd"):
            pvdFile = unkownFile
        elif unkownFile.endswith(".csv"):
            csvPath = unkownFile

    path = Path(unkownFile).parent
    # Try to find other files
    if os.path.isfile(path / (settings["MACRODATANAME"] + ".csv")):
        csvPath = str(path / (settings["MACRODATANAME"] + ".csv"))

    if os.path.isfile(path / settings["CONFIGNAME"]):
        conf = SimulationConfig(path / settings["CONFIGNAME"])
        subfolderName = conf.name

    if os.path.isfile(path / (settings["COLLECTIONNAME"] + ".pvd")):
        pvdFile = str(path / (settings["COLLECTIONNAME"] + ".pvd"))
        vtu_files = parse_pvd_file(path, pvdFile)
        first = get_data_from_name(vtu_files[0])
        subfolderName = first["name"]
        if "minStep" in first:
            X = "nr_func_evals"
            ylog = True

    # if there is no pvd file, we can create one if we find some vtu files
    if pvdFile is None:
        # check if there are any vtu files
        vtu_files = list(path.glob("*.vtu"))
        dataPath = path
        if len(vtu_files) == 0:
            # try data folder too
            vtu_files = list(path.glob(settings["DATAFOLDERPATH"] + "/*.vtu"))
            dataPath = path / settings["DATAFOLDERPATH"]
        if len(vtu_files) > 0:
            create_collection(dataPath, path, settings["COLLECTIONNAME"])
            pvdFile = str(path / (settings["COLLECTIONNAME"] + ".pvd"))
            vtu_files = parse_pvd_file(path, pvdFile)
            first = get_data_from_name(vtu_files[0])
            subfolderName = first["name"]
            if "minStep" in first:
                X = "nr_func_evals"
                ylog = True

    print(f"Plotting at {path}")
    if not noPlots and csvPath is not None:
        from Plotting.remotePlotting import update_headers_in_file

        update_headers_in_file(csvPath)
        makePlot(
            csvPath,
            name=subfolderName + "_energy.pdf",
            X=X,
            Y="avg_energy",
            ylog=ylog,
        )
        makePlot(
            csvPath,
            name=subfolderName + "_stress.pdf",
            X=X,
            Y="avg_RSS",
            # xlim=[0, 1],
        )

        for Y in [
            "minimization_time",
            "nr_iterations",
            "nr_func_evals",
            "est_time_remaining",
        ]:  # "Write_time", "Run_time", "Est_time_remaining"]:
            try:
                if Y == "est_time_remaining":
                    xlim = [0.16]
                else:
                    xlim = None
                makePlot(
                    csvPath,
                    Y=Y,
                    name=subfolderName + f"{Y.replace(' ', '_')}.pdf",
                    legend=True,
                    use_title=True,
                    xlim=xlim,
                )
            except KeyError as e:
                print(f"{e}")

        if X == "nr_func_evals":
            makePlot(
                csvPath,
                name=subfolderName + "_maxForce.pdf",
                ylog=ylog,
                X=X,
                Y="max_force",
            )
        # makePlot(
        #     csvPath,
        #     name=subfolderName + "subract_stress.pdf",
        #     Y="avg_RSS",
        #     xlim=[0, 1],
        #     subtract="/Volumes/data/MTS2D_output/singleDislocationTest,s10x10l0.0,0.001,4.0NPBCt3meshDiagonalminorepsR1e-06s0/macroData.csv",
        # )
        # if conf is not None:
        #     makePlot(
        #         csvPath,
        #         name=subfolderName + "_stress+.pdf",
        #         Y="avg_RSS",
        #         add_images=True,
        #         image_pos=[
        #             [0.35, 0.02],  # first image, bottom middle
        #             [0.03, 0.5],  # second image, upper left
        #             [0.6, 0.55],  # upper right
        #         ],
        #         labels=conf.minimizer,
        #     )
        # Close all plt plots
        plt.close("all")

    # makeItterationsPlot(path+macroData, subfolderName+"_itterations.pdf")
    if not noVideos and pvdFile is not None:
        makeAnimations(path, X=X, **kwargs)


def handle_args_and_plot():
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process plotting and video options.")

    # Add arguments

    parser.add_argument("-f", "--unkownFile", help=".conf, .pvd or .csv file")
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
        p = "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s400x400l0.15,1e-05,1.0PBCt8epsR1e-06s9/macroData.csv"
        plotAll(
            # "/Volumes/data/MTS2D_output/doubleDislocationTest,s100x100l0.0,0.001,4.0NPBCt3epsR1e-06s0/macroData.csv",
            p,
            makeGIF=False,
            transparent=False,
            noPlots=True,
            noVideos=False,
            combineVideos=False,
            fps=60,
            seconds_per_unit_shear=2,
            allImages=True,
            reuseImages=True,
        )

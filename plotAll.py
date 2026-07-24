from Plotting.makeAnimations import makeAnimations
from Plotting.makePlots import makePlot, makeItterationsPlot
from Plotting.settings import settings
from Plotting.dataFunctions import parse_pvd_file, get_data_from_name
from Plotting.makePvd import create_collection
from Plotting.remotePlotting import get_csv_files

import os
from pathlib import Path

# Now we can import from Management
from Management.simulationManager import findOutputPath
from Management.configGenerator import SimulationConfig, ConfigGenerator

from matplotlib import pyplot as plt


def plotAll(unkownFile="", plots=True, videoes=True, **kwargs):
    # Backward-compatible handling of old flags.
    if "noPlots" in kwargs:
        plots = not kwargs.pop("noPlots")
    if "noVideos" in kwargs:
        videoes = not kwargs.pop("noVideos")
    if "noVidoes" in kwargs:
        videoes = not kwargs.pop("noVidoes")

    element_subset = kwargs.pop("element_subset",  None)
    if element_subset == "none":
        element_subset = None
    kwargs["element_subset"] = element_subset

    video_variants = kwargs.pop("videoVariants", False)
    video_variants = bool(video_variants)

    X = "load"
    ylog = False

    if isinstance(unkownFile, list):
        csvPath, labels = get_csv_files(
            unkownFile, useOldFiles=False, forceUpdate=False
        )
        if kwargs.get("labels"):
            labels = kwargs["labels"]
        name = kwargs.get("name", "noName") + "_"
        path = Path(csvPath[0]).parent
    else:
        # unkownFile can be either a .conf, .pvd or .csv file
        conf, csvPath, pvdFile = None, None, None
        subfolderName = "unkown_"

        if unkownFile != "":
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

        name = subfolderName
        labels = None

    print(f"Plotting at {path}")
    if plots and csvPath is not None:
        makePlot(
            csvPath,
            name=name + "_energy.pdf",
            X=X,
            Y="avg_energy",
            ylog=ylog,
            labels=labels,
            legend=True,
        )
        try:
            makePlot(
                csvPath,
                name=name + "_stress.pdf",
                X=X,
                Y="avg_sigma12",
                legend=True,
                labels=labels,
                # xlim=[0, 1],
            )
        except KeyError as e:
            makePlot(
                csvPath,
                name=name + "_stress.pdf",
                X=X,
                Y="avg_P12",
                legend=True,
                labels=labels,
                # xlim=[0, 1],
            )

        for Y in [
            "minimization_time",
            "nr_iterations",
            "nr_func_evals",
            "est_time_remaining",
            "avg_P12",
            "avg_sigma12-avg_P12",
        ]:  # "Write_time", "Run_time", "Est_time_remaining"]:
            try:
                # if Y == "est_time_remaining":
                #     xlim = [0.16]
                # else:
                #     xlim = None
                makePlot(
                    csvPath,
                    Y=Y,
                    name=name + f"{Y.replace(' ', '_')}.pdf",
                    legend=True,
                    use_title=True,
                    labels=labels,
                    # xlim=xlim,
                )
            except KeyError as e:
                print(f"{e}")

        if X == "nr_func_evals":
            makePlot(
                csvPath,
                name=name + "_maxForce.pdf",
                ylog=ylog,
                X=X,
                Y="max_force",
                labels=labels,
            )
        # makePlot(
        #     csvPath,
        #     name=name + "subract_stress.pdf",
        #     Y="avg_RSS",
        #     xlim=[0, 1],
        #     subtract="/Volumes/data/MTS2D_output/singleDislocationTest,s10x10l0.0,0.001,4.0NPBCt3meshDiagonalminorepsR1e-06s0/macroData.csv",
        # )
        # if conf is not None:
        #     makePlot(
        #         csvPath,
        #         name=name + "_stress+.pdf",
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

    # makeItterationsPlot(path+macroData, name+"_itterations.pdf")
    if videoes and pvdFile is not None:
        if video_variants and False:
            variant_settings = [
                {"element_subset": None},
                {"element_subset": "even"},
            ]
            for variant in variant_settings:
                variant_kwargs = kwargs.copy()
                variant_kwargs["element_subset"] = variant["element_subset"]
                makeAnimations(path, X=X, **variant_kwargs)
        else:
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
        "--noVidoes",
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
    parser.add_argument(
        "--elementSubset",
        dest="element_subset",
        choices=["odd", "even", "none"],
        default="none",
        help="Only plot odd/even elements in mesh/disk videos (default: none).",
    )
    parser.add_argument(
        "--videoVariants",
        action="store_true",
        help="Render videos for C",
    )
    parser.add_argument(
        "--squarePeriodicMesh",
        dest="square_periodic_mesh",
        action="store_true",
        help="Map periodic mesh videos into a unit square before rendering.",
    )
    parser.add_argument(
        "--periodicBoxSize",
        dest="periodic_box_size",
        type=float,
        default=None,
        help="Periodic box size for square mesh rendering (default: inferred from VTU metadata).",
    )
    parser.add_argument(
        "--cartesianViewportCulling",
        dest="cartesian_viewport_culling",
        action="store_true",
        help="Cull old Cartesian mesh rendering to the current viewport.",
    )
    parser.add_argument(
        "--cartesianViewport",
        dest="cartesian_viewport",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=None,
        help="Physical mesh viewport used with Cartesian viewport culling.",
    )

    args = parser.parse_args()

    # Convert Namespace to dict for **kwargs usage
    kwargs = vars(args)

    # clean the inputs
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.strip()

    # Map legacy flags to new names
    kwargs["plots"] = not kwargs.pop("noPlots")
    kwargs["videoes"] = not kwargs.pop("noVideos")

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
        p = "/Volumes/data/MTS2D_output/doubleDislocationTest,s30x30l0.0,0.001,4.0NPBCt3meshDiagonalminorepsR1e-06logDuringMinimization1s0/macroData.csv"
        p = os.path.expanduser(
            "~/Work/PhD/Code/localData/MTS2D_output/doubleDislocationTest,s20x20l0.0,0.01,2.0NPBCt1meshDiagonalminorGP31.0epsR1e-06logDuringMinimization1s0/macroData.csv"
        )
        plotAll(
            # "/Volumes/data/MTS2D_output/doubleDislocationTest,s100x100l0.0,0.001,4.0NPBCt3epsR1e-06s0/macroData.csv",
            p,
            makeGIF=False,
            transparent=False,
            plots=True,
            videoes=True,
            combineVideos=False,
            fps=60,
            seconds_per_unit_shear=2,
            allImages=True,
            reuseImages=True,
        )

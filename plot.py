import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Management.jobs import *

from Management.updateCSV import fix_csv_files, read_macrodata_csv
from Management.configGenerator import ConfigGenerator
from Management.simulationManager import findOutputPath
from Plotting.makePlots import (
    makePlot,
    makeSettingComparison,
    makeAverageComparisonPlot,
    plot_force_contribution_magnitudes,
    plot_predicted_energy_error,
)
from Plotting.pyplotFunctions import plot_center_node_forces
from MTMath.powerlaw_mixed_test import (
    testDist,
    grid_compare_xmin,
    testSamplePiecewise,
    plot_compare_xmin,
    plot_convergence_xmin,
)
from Plotting.plotPowerLaw import (
    get_group_structure,
    make_exponent_fit,
    plot_powerlaw,
    make_fit,
    get_energy_drops,
    plot_KS_fitting,
    findPrePostSplit,
    plot_plastic_energy_scatter,
)
from Plotting.reversibilityPlot import plot_reversibility_histograms
from MTMath.meshGeometryReconnecting import run_reconnection_demo
from MTMath.poincareTiling import (
    elasticReductionPlots,
    tryAllRotations,
    bug_hunting,
    poincareTiling,
    plotStressFromRealF,
    plotsLotsOfRealFStress,
    calculateSimpleFiniteDifferenceDerivatives,
    calculateShearFiniteDifferenceDerivatives,
    plotShearFiniteDifferenceDerivatives,
    oldQuadrantIdentification,
    checkPoincareQuadrants,
    drawPoincareGrid,
    drawLeftRightExplanationFigs,
    drawRotationExplanationFigs,
    drawRotation2ExplanationFigs,
)
from MTMath.decomposeElasticPlastic import showDecomposition
from MTMath.poincareEnergy import generate_cauchy_stress_grid, generate_energy_grid
from plotAll import plotAll
from Plotting.remotePlotting import (
    plotLog2,
    plotLogCompare,
    plotPlasticCounts,
    plotReversibilityEnergyDropCorrelation,
    get_csv_files,
    plotEnergy,
    plotStress,
    stressPlotWithImages,
    energyPlotWithImages,
    plotLog,
    plotAverage,
    plotTime,
    get_folders_from_servers,
    createVideoes,
    get_csv_from_server,
)
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from Management.configGenerator import SimulationConfig
from Management.connectToCluster import Servers
from pathlib import Path
from Plotting.doubleDislocationComparison import (
    doubleDislocationComparison,
    doubleDislocationEnergy,
    doubleDislocationMeshMovies,
)


def plotPropperJob():
    nrThreads = 3
    nrSeeds = 40
    # configs, labels = propperJob(nrThreads, nrSeeds, group_by_variant=True)
    configs, labels = largePropperJob(group_by_variant=True)  # , FIREOnly=True)
    configs, labels = bigUmutJob(group_by_variant=True)
    # xlim = [0.25, 0.55]
    startLoad = configs[0][0].startLoad
    maxLoad = configs[0][0].maxLoad
    for confs, labs in zip(configs, labels):
        plotEnergy(confs, labels=labs)
    # plotLog(
    #     configs,
    #     labels=labels,
    #     # show=True,
    #     # xlim=xlim,
    # )

    paths_minimizers, labs = get_csv_files(configs, labels=labels)
    for paths in paths_minimizers:
        make_exponent_fit(
            csvPaths=paths,
            strainLim=[0.7, maxLoad],
            # debug=True,
            # xmax=1e-4,
            show=False,
        )
        make_exponent_fit(
            csvPaths=paths,
            strainLim=[startLoad, 0.4],
            # debug=True,
            # xmax=1e-4,
            show=False,
        )
        make_exponent_fit(
            csvPaths=paths,
            strainLim=[startLoad, 1.0],
            # debug=True,
            show=False,
            # xmax=1e-4,
        )

    # from MTMath.plotPowerLaw import plot_powerlaw  # import locally if re-enabling the block below
    # plot_powerlaw(
    #     paths,
    #     alg_labels=labs,
    #     show=True,
    #     strainLim=[0.7, 1.0],
    #     evaluate=True,
    #     # debug=True,
    # )


def plotLongJob():
    path = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/macroData.csv"
    strainLim = [1, 3]

    make_exponent_fit(
        csvPaths=path,
        strainLim=strainLim,
        debug=False,
        # xmax=1e-4,
    )


def compare_center_node_forces():
    sim_paths = [
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCt1meshDiagonalminors0",
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCt1s0",
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCedgeFlipt1s0",
    ]
    labels = [
        "minor",
        "major",
        "edgeFlip",
    ]
    fig, _ = plot_center_node_forces(
        sim_paths, labels=labels, pvd_file="collection.pvd"
    )
    os.makedirs("Plots", exist_ok=True)
    out_path = os.path.join("Plots", "center_node_forces_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def compare_energy_three_sims():
    sim_paths = [
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCt1meshDiagonalminors0",
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCt1s0",
        "/Volumes/data/MTS2D_output/reconnectSSTest,s3x3l0.0,0.01,3.0PBCedgeFlipt1s0",
    ]
    labels = [
        "minor",
        "major",
        "edgeFlip",
    ]
    energy_cols = ["avg_energy", "total_energy", "max_energy", "energy"]
    x_cols = ["load", "strain", "gamma", "load_step"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for path, label in zip(sim_paths, labels):
        csv_path = os.path.join(path, "macroData.csv")
        df = pd.read_csv(csv_path)

        x_col = next((c for c in x_cols if c in df.columns), None)
        if x_col is None:
            x = np.arange(len(df), dtype=float)
        else:
            x = df[x_col].to_numpy(dtype=float)

        y_col = next((c for c in energy_cols if c in df.columns), None)
        if y_col is None:
            raise ValueError(f"No energy column found in {csv_path}")
        y = df[y_col].to_numpy(dtype=float)

        if len(x) > 1:
            x = x[1:]
            y = y[1:]

        ax.plot(x, y, label=f"{label} ({y_col})")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"Energy $E$")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs("Plots", exist_ok=True)
    out_path = os.path.join("Plots", "energy_comparison_logy.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.show()
    plt.show()



# MDPI Article plot
def energyField():
    from matplotlib import pyplot as plt
    from MTMath.poincareEnergy import (
        generate_energy_grid,
        make3DEnergyField,
        plotEnergyField,
    )
    import numpy as np

    g, x, y = generate_energy_grid(
        resolution=400,
        return_XY=True,
        zoom=1,
        poincareDisk=True,
        energy_lim=[0, 50],
    )
    ax = plotEnergyField(g, save=False, scale=0.2)
    ax.figure.show()
    plt.show()
    # g = generate_energy_grid(9)
    # print(np.round(g, 2))

    # make3DEnergyField(g, x, y, zScale=0.6, add_front_hole=True)


def showPoincareDisk():
    from MTMath.poincareEnergy import plotPoincareDisk, plotPoincareLines

    #plotPoincareDisk(grid_size=500, depth=7, transformation="triangular")
    plotPoincareLines(grid_size=500)
    plt.show()


def showInstabilityAngle():
    from MTMath.poincareEnergy import generate_stability_min_angle_grid, prepPoincareFig

    res = 100
    # fig, ax = prepPoincareFig(
    #     grid_size=res, withCircle=False, withGrid=False, minimalTicks=True
    # )
    # theta = generate_stability_min_angle_grid(resolution=res)
    # ax.imshow(theta, cmap="twilight")
    # path = f"Plots/acoustic_tensor_min_det_angle_{res}.png"
    # fig.savefig(path, bbox_inches="tight")
    # print(f"Saved plot to {path}")
    stability = True

    fig, ax = prepPoincareFig(grid_size=res)
    theta = generate_stability_min_angle_grid(resolution=res, boolStability=stability)
    ax.imshow(theta, cmap="twilight")
    path = f"Plots/acoustic_tensor_min_det_{'stability' if stability else 'angle'}_{res}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot to {path}")


def oneDPlot():
    from MTMath.poincareEnergy import oneDPotential, oneDPotentialDissordered

    oneDPotential()


# oneDPlot()


# MDPI Article plot
def plotSampleRuns():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_variant=True)
    seedNr = 3
    configs = [c[seedNr] for c in configs]
    labels = [lab[seedNr] for lab in labels]

    paths = [
        "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s41",
        "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerCGLBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s41",
        "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05plasticityEventThreshold1e-06s41",
    ]
    with tqdm(total=len(configs) * 2 + 2) as pbar:
        # Loop through each config and path, updating the progress bar
        for config, path in zip(configs, paths):
            stressPlotWithImages([config], [path])
            pbar.update(1)
            energyPlotWithImages([config], [path])
            pbar.update(1)

        stressPlotWithImages(configs, paths)
        pbar.update(1)
        energyPlotWithImages(configs, paths)
        pbar.update(1)


def plotThreadTest():
    nrThreads = 1  # [1, 2, 4, 8, 16, 32, 64]
    nrSeeds = 1
    size = 100
    configs, labels = basicJob(nrThreads, nrSeeds, size, group_by_variant=True)
    configs, labels = smallJob(group_by_variant=True)
    # plotAverage(configs, labels)
    plotTime(configs, labels)


def plotSylvainBatches():
    fast_xmin = True
    xmin_accuracy = 1.0
    for batch in [-2, -1]:
        configs, labels = sylvainBatches(batch)
        
        if not configs:
            continue
        grouped_configs, grouped_labels, group_labels = (
            ConfigGenerator.group_by_settings(configs, labels=labels)
        )
        #plotReversibilityEnergyDropCorrelation(grouped_configs, grouped_labels, xAxisCol="rev_energy_diff")

        paths, _ = get_csv_files(
            grouped_configs,
            labels=grouped_labels,
            useOldFiles=False,
            forceUpdate=False,
        )
        if not paths:
            continue

        def _path_to_name(path):
            base = os.path.basename(path)
            if base == "macroData.csv":
                return os.path.basename(os.path.dirname(path))
            return os.path.splitext(base)[0]

        path_names = {_path_to_name(p) for group in paths for p in group}
        aligned_group_labels = []
        for confs, label in zip(grouped_configs, group_labels):
            if any(c.name in path_names for c in confs):
                aligned_group_labels.append(label)

        if len(aligned_group_labels) > len(paths):
            aligned_group_labels = aligned_group_labels[: len(paths)]
        elif len(aligned_group_labels) < len(paths):
            for i in range(len(aligned_group_labels), len(paths)):
                aligned_group_labels.append(f"group_{i}")

        # makeAverageComparisonPlot(
        #     paths,
        #     Y="avg_energy",
        #     name=f"sylvain_batch_{batch}_avg_energy",
        #     group_labels=aligned_group_labels,
        #     use_title=True,
        # )

        # flat_paths = []
        # flat_labels = []
        # for group_paths, group_label in zip(paths, aligned_group_labels):
        #     if not group_paths:
        #         continue
        #     flat_paths.extend(group_paths)
        #     flat_labels.extend([group_label] * len(group_paths))

        # for postRegime, suffix in [(True, "post"), (False, "pre")]:
        #     plot_reversibility_histograms(
        #         paths,
        #         postRegime=postRegime,
        #         show=False,
        #         save_path=f"Plots/sylvain_batch_{batch}_reversibility_{suffix}.pdf",
        #         group_labels=aligned_group_labels,
        #     )

        #     plot_plastic_energy_scatter(
        #         flat_paths,
        #         labels=flat_labels,
        #         postRegime=postRegime,
        #         name=f"sylvain_batch_{batch}_plastic_energy_{suffix}",
        #         color_by_label=True,
        #     )

        for group_idx, (group_paths, group_label) in enumerate(
            zip(paths, aligned_group_labels)
        ):
            if not group_paths:
                continue
            display_label = group_label or f"group_{group_idx}"
            display_label = f"batch={batch}, {display_label}"
            for postRegime in [True, False]:
                plot_powerlaw(
                    group_paths,
                    group_labels=display_label,
                    postRegime=postRegime,
                    fast_xmin=fast_xmin,
                    xmin_accuracy=xmin_accuracy,
                )


def print_remote_runtimes(load_increment=1e-5):
    import os
    import pandas as pd

    configs, labels = size_scaling_job()
    configs, labels = ConfigGenerator.filter(configs, labels, keys="L=300")
    configs = [c for group in configs for c in group]
    configs = [c for c in configs if abs(c.loadIncrement - load_increment) < 1e-12]
    if not configs:
        print("No configs found.")
        return

    configs_by_name = {c.name: c for c in configs}
    for server in Servers.servers:
        csv_paths = get_csv_from_server(server, configs)
        if not csv_paths:
            continue

        short = server.split(".")[0]
        print(f"{short}:")
        for csv_path in sorted(csv_paths):
            name = os.path.splitext(os.path.basename(csv_path))[0]
            config = configs_by_name.get(name)
            if config is None:
                print(f"  {name}: config missing")
                continue

            df = pd.read_csv(csv_path)
            col = None
            if "run_time" in df.columns:
                col = "run_time"
            elif "Run_time" in df.columns:
                col = "Run_time"
            elif "Run time" in df.columns:
                col = "Run time"

            if col is None:
                print(f"  seed {config.seed}: run_time column missing")
                continue

            series = df[col].dropna()
            if series.empty:
                print(f"  seed {config.seed}: run_time empty")
                continue

            print(f"  seed {config.seed}: {series.iloc[-1]}")


def debugPlotAll():
    # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    config = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    plotAll(config, makeGIF=False, transparent=False, noVideos=False)


def plotAvalanches():
    configs, labels, dump = avalanches(nrThreads=20)
    line_styles = [
        (0, (2, 2, 3, 2)),  # Another variation
        (0, (5, 2, 1, 2)),  # Dash-dot variation
        (0, (3, 5, 1, 5)),  # Another custom
        "-",
        "-",
        "-",
        "-",
        (0, (2, 2, 3, 2)),  # Another variation
        (0, (5, 2, 1, 2)),  # Dash-dot variation
        (0, (3, 5, 1, 5)),  # Another custom
        (0, (2, 2, 3, 2)),  # Another variation
        (0, (5, 2, 1, 2)),  # Dash-dot variation
        (0, (3, 5, 1, 5)),  # Another custom
    ]

    plotEnergy(configs, labels, linestyles=line_styles)
    plotTime(configs, labels)


def plotMaxForce():
    fig, ax = makePlot(
        [
            "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20epsR0.001s0/macroData.csv"
        ],
        Y="max_force",
        name="maxForce.pdf",
        labels=["EpsR=0.001"],
        legend=True,
        # ylog=True,
    )


def plotMinimizationCriteriaData():
    nrSeeds = 5
    configs, labels = findMinimizationCriteriaJobs(nrSeeds=nrSeeds)

    # configs, labels = compareWithOldStoppingCriteria()

    Ls = [40, 60, 80, 100]

    for L in Ls:
        confs, labs = zip(
            *[(conf, lab) for (conf, lab) in zip(configs, labels) if conf.rows == L]
        )
        paths, labs = get_csv_files(
            confs, labels=labs, useOldFiles=False, forceUpdate=False
        )
        # labs = [l + ", loadIncrement=1e-5" for l in labs]

        # Common kwargs for makeSettingComparison
        common_kwargs = {
            "csv_file_paths": paths,
            "labels": labs,
            # "property_keys": ["LBFGSEpsg"],
            "property_keys": ("epsR", "loadIncrement"),
            "loc": "upper right",
            "yPad": 1.3,
        }

        fig1, ax1 = makeSettingComparison(
            **common_kwargs,
            name=f"L={L}_Energy",
            seedsToShow=[2],
        )
        fig2, ax2 = makeSettingComparison(
            **common_kwargs,
            name=f"L={L}_SubtractEnergy",
            subtract=True,
            seedsToShow=[2],
        )
        fig3, ax3 = makeSettingComparison(
            **common_kwargs,
            name=f"L={L}_DetatchEnergy",
            detatchment=True,
            seedsToShow=range(nrSeeds),
        )

        # Save as separate PDF pages
        with PdfPages(f"Plots/combined_L{L}.pdf") as pdf:
            pdf.savefig(fig1, bbox_inches="tight")
            pdf.savefig(fig2, bbox_inches="tight")
            pdf.savefig(fig3, bbox_inches="tight")


def plotShowMinCriteria():
    configs, labels = showMinimizationCriteriaJobs(nrSeeds=1)
    linestyles = ["-" if "epsR=None" in label else "--" for label in labels]
    plotTime(configs, labels, linestyles=linestyles)


def plotEnergyRegion():
    csvFile = "/Volumes/data/KeepSafe/longSimulation0.15-4.75/macroData.csv"
    makePlot(csvFile, show=True, xlim=[2.9, 3.1])


def plotReconnectionJob():
    size = 100
    configs, labels = reconnectionTest(L=size)
    plotEnergy(configs, labels=labels, legend=True)


def investigateJobs():
    configs, labels = basicJob(nrSeeds=10, nrThreads=8, size=400)
    # Get config of L=400 with seed 0 and 7
    confs = []
    # for g in configs:
    for c in configs:
        assert isinstance(c, SimulationConfig)
        if c.rows == 400 and c.seed in [0, 7]:
            confs.append(c)
    get_folders_from_servers(confs)


def compareStop():
    # Compare using Epsx or EpsR to stop
    c1, l1 = bigUmutJob()
    c2, l2 = bigUmutJobWithEliasStop()
    # plotEnergy(configs=c1 + c2, labels=["Epsx=1e-5", "EpsR=1e-5"])

    configs, labels = stopConditionJob()
    # plotEnergy(configs=configs, labels=labels)
    plotAll(configs, noVideos=True, labels=labels, name="compareStop")


def compareStep():
    c, l = loadStepJob()
    fast_xmin = True
    xmin_accuracy = 0.1
    # plotPlasticCounts(c, l, postRegime=True)
    # plotPlasticCounts(c, l, postRegime=False)
    # plotLog2(
    #     c, labels=l, postRegime=True, fast_xmin=fast_xmin, xmin_accuracy=xmin_accuracy
    # )
    plotLogCompare(
        c, l, postRegime=True, fast_xmin=fast_xmin, xmin_accuracy=xmin_accuracy
    )


def plotReversibility():
    configs, labels = reversibilityJob()
    plot_reversibility_histograms(
        configs[0], postRegime=None, show=False, save_path="Plots/reversibility_All.pdf"
    )
    plot_reversibility_histograms(
        configs[0],
        postRegime=True,
        show=False,
        save_path="Plots/reversibility_post.pdf",
    )
    plot_reversibility_histograms(
        configs[0],
        postRegime=False,
        show=False,
        save_path="Plots/reversibility_pre.pdf",
    )
    fast_xmin = True
    plotLog2(configs, labels=labels, postRegime=True, fast_xmin=fast_xmin)
    plotLog2(configs, labels=labels, postRegime=False, fast_xmin=fast_xmin)







def plotLogAnalasys():
    drop_type = "energy"
    # configs, labels = bigUmutJob(group_by_variant=True)
    configs, labels = umutJobs(loadIncrement=1e-5)
    configs, labels = ConfigGenerator.filter(configs, labels, ["L=200"])
    matching_groups = [
        (group, group_labels) for group, group_labels in zip(configs, labels) if group
    ]
    if len(matching_groups) != 1:
        raise ValueError(
            f"Expected exactly one non-empty L=200 group, found {len(matching_groups)}."
        )
    our_configs, our_labels = matching_groups[0]
    # # Powerlaw
    plotEnergy(configs, labels=labels)

    # # Find split
    fast_xmin = True
    min_xmin = 1e-2
    useCDF = False
    plotLog2(
        configs, labels=labels, postRegime=True, fast_xmin=fast_xmin, min_xmin=min_xmin, useCDF=useCDF,drop_type=drop_type
    )
    plotLog2(
        configs, labels=labels, postRegime=False, fast_xmin=fast_xmin, min_xmin=min_xmin,  useCDF=useCDF,drop_type=drop_type
    )
    # p = [["/Users/eliaslundheim/Downloads/s400x400_energy_stress_log.csv"]]
    # p = [
    #     our_configs,
    #     [
    #         "/Users/eliaslundheim/work/PhD/Umut/UmutData/200x200DelaunayReconnecting/s200x200_energy_stress_log1.csv",
    #         "/Users/eliaslundheim/work/PhD/Umut/UmutData/200x200DelaunayReconnecting/s200x200_energy_stress_log2.csv",
    #     ]
    # ]

    # lab = [our_labels,["umut", "umut"]]
    # plotLogCompare(p, lab,postRegime=True, fast_xmin=fast_xmin,
    # )
    # plot_powerlaw(
    #     p,
    #     group_labels=lab,
    #     postRegime=True,
    #     fast_xmin=fast_xmin,
    #     useCDF=useCDF,
    # )
    # plot_powerlaw(
    #     p,
    #     group_labels=lab,
    #     postRegime=False,
    #     fast_xmin=fast_xmin,
    #     useCDF=useCDF,
    # )

    # p = [
    #     [
    #         "/Users/eliaslundheim/work/PhD/UmutCode/UmutData/s400x400_alpha_energy_drop1.csv",
    #     ],
    #     [
    #         "/Users/eliaslundheim/work/PhD/UmutCode/UmutData/s400x400_alpha_energy_drop2.csv",
    #     ],
    #     [
    #         "/Users/eliaslundheim/work/PhD/UmutCode/UmutData/s400x400_alpha_energy_drop3.csv",
    #     ],
    #     [
    #         "/Users/eliaslundheim/work/PhD/UmutCode/UmutData/s400x400_alpha_energy_drop4.csv",
    #     ],
    # ]
    # lab = [["umut_noRe_seed=1"], ["umut_noRe_seed=2"], ["umut_noRe_seed=3"], ["umut_noRe_seed=4"]]

    # p = [
    #     [
    #     "/Volumes/data/MTS2D_output/simpleShear,s500x500l0.138,2e-05,1.0PBCt8initialGuessNoise0.04epsR1e-05s0/macroData.csv"
    #     ],
    #     # [
    #     #     "/Users/eliaslundheim/work/PhD/Umut/UmutData/400x400NoReconnect/s400x400_alpha_energy_drop1.csv",
    #     #     "/Users/eliaslundheim/work/PhD/Umut/UmutData/400x400NoReconnect/s400x400_alpha_energy_drop2.csv",
    #     #     "/Users/eliaslundheim/work/PhD/Umut/UmutData/400x400NoReconnect/s400x400_alpha_energy_drop3.csv",
    #     #     "/Users/eliaslundheim/work/PhD/Umut/UmutData/400x400NoReconnect/s400x400_alpha_energy_drop4.csv",
    #     # ]
    # ]
    # lab = [
    #     ["500x500EliasR"],
    #     #["umut_noRe"] * 4
    # ]

    # plot_powerlaw(
    #     p,
    #     group_labels=lab,
    #     postRegime=True,
    #     fast_xmin=fast_xmin,
    #     useCDF=useCDF,
    #     drop_type=drop_type,
    # )
    # plot_powerlaw(
    #     p,
    #     group_labels=lab,
    #     postRegime=False,
    #     fast_xmin=fast_xmin,
    #     useCDF=useCDF,
    #     drop_type=drop_type,
    # )


def syntheticDataPlotting():
    ns = [1e2, 1e3, 1e4, 1e5]
    subgrid = (6, 6)
    datasets = [grid_compare_xmin(n=n, subgrid=subgrid) for n in ns]

    plot_compare_xmin(
        data=datasets,
        sample_sizes=[int(n) for n in ns],
        method="all",
        subgrid=subgrid,
    )
    plot_convergence_xmin(data=datasets, subgrid=subgrid)


def testRealData():
    import numpy as np

    umut_data = Path("~/work/PhD/UmutCode/UmutData").expanduser()
    paths = [
        "/Volumes/data/MTS2D_output/simpleShear,s500x500l0.138,2e-05,1.0PBCt8initialGuessNoise0.04LBFGSEpsx1e-05s0/macroData.csv",
        "/Volumes/data/MTS2D_output/simpleShear,s500x500l0.138,2e-05,1.0PBCt8initialGuessNoise0.04epsR1e-05s0/macroData.csv",
        str(umut_data / "s400x400_alpha_energy_drop1.csv"),
        str(umut_data / "s400x400_alpha_energy_drop2.csv"),
        str(umut_data / "s400x400_alpha_energy_drop3.csv"),
        str(umut_data / "s400x400_alpha_energy_drop4.csv"),
    ]
    for path in paths:
        try:
            split = findPrePostSplit(csvPath=path)
        except Exception as e:
            print(f"Failed to find pre/post split for {path}: {e}")
            continue
        drops, _ = get_energy_drops(path, averageEnergy=True, strainLim=[split, np.inf])
        drops = np.asarray(drops, dtype=float)
        drops = drops[np.isfinite(drops)]
        if drops.size < 10:
            print(f"Not enough drops for {path}")
            continue
        fit = make_fit(drops, fast_xmin=True, xmin_accuracy=0.1, parallel_xmin=True)
        plot_KS_fitting(fit, save=True, show=False)


def analyseLongData():
    configs, labels = longJob(8, 1, size=300)

    from Plotting.findXmin import find_xmin_derivative

    paths, labels = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )

    paths = fix_csv_files(paths)
    paths, labels = get_group_structure(paths, labels)
    drops, info = get_energy_drops(
        paths[0],
        debug=False,
        label=None,
        postRegime=True,
    )
    plateau_xmin = find_xmin_derivative(drops, debug=True)

    # plotLog2(configs, labels, xmin_range=1e-1)
    # plotLog2(configs, labels, xmin_range=1e-4)

def plotReferenceTest():

    configs, labels = reconnectSSTest(reconnectionMethod="none")
    plotStress(configs, labels)
    flat_configs = (
        [c for group in configs for c in group]
        if configs and isinstance(configs[0], list)
        else list(configs)
    )
    if not flat_configs:
        print("No reference-test configs found.")
        return

    csv_paths, _ = get_csv_files(
        flat_configs, labels=None, useOldFiles=False, forceUpdate=False
    )
    if not csv_paths:
        print("No CSV files found for reference-test plots.")
        return

    label_by_name = {
        cfg.name: rf"$\gamma_0$={cfg.GP1:g}, $d$={cfg.GP2:g}"
        for cfg in flat_configs
    }

    sim_paths = []
    sim_labels = []
    for csv_path in csv_paths:
        sim_dir = str(Path(csv_path).parent)
        sim_name = Path(sim_dir).name
        sim_paths.append(sim_dir)
        sim_labels.append(label_by_name.get(sim_name, sim_name))

    force_contrib_out = (
        Path.cwd() / "Plots" / "reference_test_force_contrib_vs_strain.pdf"
    )
    plot_force_contribution_magnitudes(
        sim_paths,
        labels=sim_labels,
        name=str(force_contrib_out),
        plot_mode="scatter",
        marker_size=80,
        connect_points=False,
        show=True,
    )


def plotPristineCrystalPredictionError():
    configs, labels = smallPristineCrystal(group_by_variant=True)
    paths, labels = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )
    if not paths:
        print("No CSV files found for pristineCrystal.")
        return

    paths = fix_csv_files(paths)
    paths, labels = get_group_structure(paths, labels)
    flat_paths = [path for group in paths for path in group]
    flat_labels = [label for group in labels for label in group]
    if not flat_paths:
        print("No valid CSV paths found for pristineCrystal.")
        return

    # Switch to "first_order" for the first-order transparent reference.
    #reference_prediction = "second_order_gamma0"
    reference_prediction = "first_order"
    output_suffix = (
        "_gamma0_reference" if reference_prediction == "second_order_gamma0" else ""
    )
    output_path = (
        Path.cwd()
        / "Plots"
        / f"pristine_crystal_energy_prediction_error{output_suffix}.pdf"
    )
    plot_predicted_energy_error(
        flat_paths,
        labels=flat_labels,
        name=str(output_path),
        show=False,
        error_metric="abs_second_order_prediction_error",
        property_keys=("L", "loadIncrement"),
        use_color_matrix_legend=True,
        reference_prediction=reference_prediction,
        reference_alpha=0.2,
        show_reference_line=True,
        strain_lim=(0,0.14), #Loss of strong elipticity
        x_column="load_i",
        y_log=True,
    )


if __name__ == "__main__":
    # calculateSimpleFiniteDifferenceDerivatives()
    # plotShearFiniteDifferenceDerivatives()
    # calculateShearFiniteDifferenceDerivatives()
    #run_reconnection_demo()
    # from MTMath.triangleError import test, test_Kappa

    # test()
    # plotReconnectionJob()
    # test_Kappa()
    # plotEnergyRegion()
    # plotSampleRuns()
    # plotLongJob()
    # plotPropperJob()

    # debugPlotAll()
    # energyField()
    #showPoincareDisk()
    # showInstabilityAngle()
    # plotThreadTest()
    # configs, labels = allPlasticEventsJob()
    # createVideoes(configs, all_images=True)

    # plotAvalanches()
    # plotMaxForce()
    # plotMinimizationCriteriaData()
    # plotShowMinCriteria()
    # poincareTiling()

    # checkPoincareQuadrants()
    # drawLeftRightExplanationFigs()
    # drawRotationExplanationFigs()
    # drawRotation2ExplanationFigs()
    # plotStressFromRealF(grid_size=400, nr_theta=400, stress_type="stability")
    # tryAllRotations()
    # plotsLotsOfRealFStress("stability", reduced=True)
    # bug_hunting()
    # elasticReductionPlots()
    # showDecomposition()
    # compareStop()
    # compareStep()
    # plotReversibility()
    #plotReferenceTest()
    #plotLogAnalasys()
    # analyseLongData()
    # testSamplePiecewise(alpha=1.35, xmin=1e-5, xlow=1e-7)
    # syntheticDataPlotting()
    # testDist()
    # testRealData()
    # investigateJobs()
    # print_remote_runtimes()
    plotSylvainBatches()
    # plotPristineCrystalPredictionError()
    # compare_center_node_forces()
    # compare_energy_three_sims()
    pass

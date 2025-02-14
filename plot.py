from Management.jobs import (
    smallJob,
    basicJob,
    allPlasticEventsJob,
    propperJob,
    propperJob3,
    avalanches,
    findMinimizationCriteriaJobs,
)


from Management.simulationManager import findOutputPath
from Plotting.makePlots import makePlot, makeSettingComparison
from plotAll import plotAll
from Plotting.remotePlotting import (
    get_csv_files,
    plotEnergy,
    stressPlotWithImages,
    energyPlotWithImages,
    plotLog,
    plotAverage,
    plotTime,
    get_folders_from_servers,
    createVideoes,
)
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def plotPropperJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_seeds=True)
    # xlim = [0.25, 0.55]
    plotLog(
        configs,
        labels=labels,
        # show=True,
        # xlim=xlim,
    )


def plotPropperJob3():
    configs, labels = propperJob3(group_by_seeds=True)
    # xlim = [0.25, 0.55]
    # get_folders_from_servers(configs)
    createVideoes(configs)
    # plotLog(configs, labels=labels)
    # plotAverage(configs, labels)


# MDPI Article plot
def energyField():
    from MTMath.plotEnergy import generate_energy_grid, make3DEnergyField

    g, x, y = generate_energy_grid(
        resolution=200,
        return_XY=True,
        zoom=1,
        energy_lim=[None, 0.37],
        poincareDisk=True,
    )

    make3DEnergyField(g, x, y, zScale=0.6, add_front_hole=True)


def oneDPlot():
    from MTMath.plotEnergy import oneDPotential, oneDPotentialDissordered

    oneDPotential()


# oneDPlot()


# MDPI Article plot
def plotSampleRuns():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_seeds=True)
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
    configs, labels = basicJob(nrThreads, nrSeeds, size, group_by_seeds=True)
    configs, labels = smallJob(group_by_seeds=True)
    # plotAverage(configs, labels)
    plotTime(configs, labels)


def debugPlotAll():
    outputPath = findOutputPath()
    # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    config = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    plotAll(config, outputPath, makeGIF=False, transparent=False, noVideos=False)


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
        Y="Max_force",
        name="maxForce.pdf",
        labels=["EpsR=0.001"],
        legend=True,
        # ylog=True,
    )


def plotMinimizationCriteriaData():
    nrSeeds = 5
    configs, labels = findMinimizationCriteriaJobs(nrSeeds=nrSeeds)
    Ls = [40, 60, 80, 100]

    for L in Ls:
        confs, labs = zip(
            *[(conf, lab) for (conf, lab) in zip(configs, labels) if conf.rows == L]
        )
        paths, labs = get_csv_files(
            confs, labels=labs, useOldFiles=False, forceUpdate=False
        )

        # Common kwargs for makeSettingComparison
        common_kwargs = {
            "csv_file_paths": paths,
            "labels": labs,
            "property_keys": ("epsR", "loadIncrement"),
            "loc": "upper right",
            "yPad": 1.3,
        }

        fig1, ax1 = makeSettingComparison(**common_kwargs, name=f"L={L}_Energy")
        fig2, ax2 = makeSettingComparison(
            **common_kwargs, name=f"L={L}_SubtractEnergy", subtract=True
        )
        fig3, ax3 = makeSettingComparison(
            **common_kwargs, name=f"L={L}_CumSumSubEnergy", cumSumSubtract=True
        )
        fig4, ax4 = makeSettingComparison(
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
            pdf.savefig(fig4, bbox_inches="tight")


if __name__ == "__main__":
    # plotSampleRuns()
    # plotPropperJob3()
    # debugPlotAll()
    # energyField()
    # plotThreadTest()
    # configs, labels = allPlasticEventsJob()
    # createVideoes(configs, all_images=True)

    # plotAvalanches()
    # plotMaxForce()
    plotMinimizationCriteriaData()

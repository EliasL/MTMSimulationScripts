from Management.multiServerJob import (
    bigJob,
    confToCommand,
    basicJob,
    smallJob,
    propperJob,
    propperJob1,
    propperJob2,
    propperJob3,
    JobManager,
    get_server_short_name,
)

from Management.simulationManager import findOutputPath
from plotAll import plotAll
from Plotting.remotePlotting import (
    stressPlotWithImages,
    energyPlotWithImages,
    plotLog,
    plotAverage,
    plotTime,
)
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
    plotLog(configs, labels=labels)
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


if __name__ == "__main__":
    # plotSampleRuns()
    plotPropperJob3()
    # debugPlotAll()
    # energyField()
    # plotThreadTest()

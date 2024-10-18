from Management.multiServerJob import (
    bigJob,
    confToCommand,
    basicJob,
    propperJob,
    propperJob1,
    propperJob2,
    propperJob3,
    generateCommands,
    JobManager,
    get_server_short_name,
)
from Management.simulationManager import findOutputPath
from Plotting.plotAll import plotAll
from Plotting.remotePlotting import stressPlotWithImages, energyPlotWithImages, plotLog
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


# MDPI Article plot
def energyField():
    from MTMath.plotEnergy import (
        generate_energy_grid,
        make3DEnergyField,
    )

    g, x, y = generate_energy_grid(
        resolution=1000, return_XY=True, energy_lim=[None, 4.3]
    )
    make3DEnergyField(g, x, y, zScale=0.8, zoom=0.2, add_front_hole=True)


# MDPI Article plot
def plotSampleRuns():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_seeds=True)
    seedNr = 3
    configs = [c[seedNr] for c in configs]
    labels = [lab[seedNr] for lab in labels]

    paths = [
        "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-05CGEpsg1e-05eps1e-05s3",
        "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerCGLBFGSEpsg1e-05CGEpsg1e-05eps1e-05s3",
        "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s100x100l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s3",
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


# MDPI Article plot
def plotEnergyPowerLaw():
    pass


def debugPlotAll():
    outputPath = findOutputPath()
    # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    config = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    plotAll(config, outputPath, makeGIF=False, transparent=False, noVideo=False)


if __name__ == "__main__":
    # plotPropperJob()
    plotSampleRuns()
    # debugPlotAll()

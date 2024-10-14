from Management import parameterExploring as pe
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
from Management.clusterStatus import get_all_server_info, display_server_info
from Management.simulationManager import findOutputPath
from Plotting.plotAll import plotAll


def plotPropperJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_seeds=True)
    # xLims = [0.25, 0.55]
    pe.plotLog(
        configs,
        "100x100, load:0.15-1, PBC, seeds:40",
        labels=labels,
        # show=True,
        # xLims=xLims,
    )


# Article plot
def energyField():
    from MTMath.plotEnergy import (
        generate_energy_grid,
        make3DEnergyField,
    )

    g, x, y = generate_energy_grid(
        resolution=1000, return_XY=True, energy_lim=[None, 4.3]
    )
    make3DEnergyField(g, x, y, zScale=0.8, zoom=0.2, add_front_hole=True)


def jobs():
    plotPropperJob()


def debugPlotAll():
    outputPath = findOutputPath()
    # config = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s16x16l0.0,1e-05,1.0NPBCt4LBFGSEpsg1e-10s0/config.conf"
    config = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1.0PBCt3minimizerFIRELBFGSEpsg1e-05CGEpsg1e-05eps1e-05s0/config.conf"
    plotAll(config, outputPath, makeGIF=False, transparent=False, noVideo=False)


if __name__ == "__main__":
    jobs()
    # debugPlotAll()

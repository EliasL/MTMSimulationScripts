from .simulationManager import SimulationManager
from .configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool

import logging

# Suppress Paramiko logging
logging.getLogger("paramiko").setLevel(logging.CRITICAL)

# Add Management to sys.path (used to import files)
# sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))

# Define dumpPath as a global variable
dumpPath = "/Volumes/data/MTS2D_output/simpleShear,s60x60l0.15,0.0002,1.0PBCt1minimizerFIRELBFGSEpsg0.0001s0/dumps//Dump_l0.447600_16.27~24.06.2024.mtsb"


def task(config):
    manager = SimulationManager(config)

    time = manager.runSimulation(build=False, resumeIfPossible=False)
    # time = manager.resumeSimulation(
    #     dumpFile=dumpPath, overwriteSettings=True, build=False
    # )

    # manager.plot()

    return time


def assignColors(configs, keyValueColors, defaultColor="black"):
    color_for_config = [defaultColor] * len(configs)

    for conf_i, config in enumerate(configs):
        for key, value, color in keyValueColors:
            if getattr(config, key) == value:
                color_for_config[conf_i] = color
    return color_for_config


def runSims(configs):
    # Build and test (Fail early)
    manager = SimulationManager(
        SimulationConfig(rows=3, cols=3, loadIncrement=0.1, showProgress=0)
    )
    try:
        manager.runSimulation(resumeIfPossible=False, silent=True)
    except Exception as e:
        Warning(e)
        manager.clean()
        try:
            manager.runSimulation()
        except Exception as e:
            raise (Exception(e))

    with Pool(processes=len(configs)) as pool:
        pool.map(task, configs)


def plotSims(configs, name, **kwargs):
    # Now we can import from Management
    from remotePlotting import get_csv_files

    from makePlots import makeEnergyPlot, makePowerLawPlot, makeItterationsPlot  # noqa: F401

    paths = get_csv_files(configs, useOldFiles=False)
    print("Plotting...")
    makeEnergyPlot(paths, f"{name} Energy.pdf", **kwargs)
    for k in ["plot_average"]:
        if k in kwargs:
            del kwargs[k]
    # makePowerLawPlot(paths, f"{name}PowerLaw.pdf", legend=True, **kwargs)
    # makeItterationsPlot(paths, f"{name}Itterations.pdf", **kwargs)


def plotLog(config_groups, name, labels, **kwargs):
    # Now we can import from Management
    from remotePlotting import get_csv_files

    from makePlots import (
        makeLogPlotComparison,
        makeEnergyPlotComparison,
        makeEnergyAvalancheComparison,
    )

    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=True
    )
    kwargs["labels"] = labels
    print("Plotting...")
    makeEnergyPlotComparison(paths, f"{name} - Energy", **kwargs)
    makeLogPlotComparison(paths, f"{name} - PowerLaw", window=False, **kwargs)
    # makeLogPlotComparison(paths, f"{name} - PowerLaw", window=True, **kwargs)
    # makeEnergyAvalancheComparison(paths, f"{name} - Histogram", **kwargs)
    # makeItterationsPlot(paths, f"{name}Itterations.pdf", **kwargs)


def plotOldStuff():
    from OldConfigGenerator import ConfigGenerator as OldConf

    configs, labels = OldConf.generate(
        rows=150,
        cols=150,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=[1e-5, 2e-5, 1e-4, 2e-4],
        maxLoad=0.8,
        epsx=[0],
        epsg=[0, 1e-2, 1e-3, 1e-4],
        epsf=[0, 1e-4, 1e-5, 1e-6],
        scenario="simpleShear",
    )
    c = assignColors(
        configs,
        [
            ["loadIncrement", 1e-5, "black"],
            ["loadIncrement", 2e-5, "red"],
            ["loadIncrement", 1e-4, "blue"],
            ["loadIncrement", 2e-4, "orange"],
        ],
    )
    plotSims(configs, "FireExplore1", labels=labels, colors=c, show=True)


def plotLessOldStuff():
    configs, labels = ConfigGenerator.generate(
        seed=[4, 5, 1, 2],
        rows=100,
        cols=100,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=[1e-5],
        maxLoad=1,
        alphaStart=[0.01, 0.1, 0.3],
        eps=[1e-3, 1e-4, 1e-5],
        LBFGSEpsg=[9e-4, 9e-5, 9e-6],
        scenario="simpleShear",
    )
    c = assignColors(
        configs,
        [
            ["eps", 1e-4, "black"],
            ["eps", 1e-5, "red"],
            ["eps", 1e-6, "blue"],
        ],
    )
    plotSims(configs, "FireExplore2", labels=labels, colors=c, show=True)


def statStuff():
    seeds = range(0, 60)
    configs = ConfigGenerator.generate_over_seeds(
        seeds,
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=1e-5,
        maxLoad=1,
        LBFGSEpsx=1e-6,
        minimizer="LBFGS",
        scenario="simpleShear",
    )
    plotSims(
        configs,
        "powerlaw",
        labels=[f"s:{i}" for i in seeds],
        show=True,
        plot_average=False,
        xLims=(0.15, 0.55),
    )


def fastStatStuff():
    seeds = range(0, 60)
    configs = ConfigGenerator.generate_over_seeds(
        seeds,
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=3e-5,
        maxLoad=1,
        LBFGSEpsx=1e-5,
        minimizer="LBFGS",
        scenario="simpleShear",
    )
    runSims(configs)
    # plotSims(configs, "powerlaw", labels=[f"s:{i}" for i in seeds], show=True,
    # plot_average=False, xLims=(0.15, 0.55))


def loadingSpeeds():
    nrThreads = 1
    nrSeeds = 40
    size = 60
    configs, labels = ConfigGenerator.generate(
        group_by_seeds=True,
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        # loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        # LBFGSEpsg=[1e-6, 1e-5, 5e-5, 1e-4],
        # loadIncrement=[1e-5],
        # LBFGSEpsg=[1e-6, 1e-5, 1e-4],
        maxLoad=1.0,
        scenario="simpleShear",
    )
    extra_configs, extra_labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.0,
        nrThreads=nrThreads,
        loadIncrement=[1e-5],
        maxLoad=1.0,
        LBFGSEpsx=[1e-6],
        scenario="simpleShear",
    )
    # configs.extend([extra_configs])
    # labels.extend([["loadIncrement=1e-5, LBFGSEpsx=1e-6"]])
    plotLog(configs, "60x60, load:0.15-1, PBC, LBFGS, t1, seeds:40", labels=labels)


def smallLoadingSpeeds():
    nrThreads = 4
    nrSeeds = 1
    size = 30
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.0,
        nrThreads=nrThreads,
        loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        maxLoad=1.0,
        LBFGSEpsg=[1e-8, 1e-9],
        scenario="simpleShear",
    )
    extra_configs, extra_labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.0,
        nrThreads=nrThreads,
        loadIncrement=[1e-5],
        maxLoad=1.0,
        LBFGSEpsx=[1e-6],
        scenario="simpleShear",
    )
    configs.extend(extra_configs)
    labels.extend(["loadIncrement=1e-5, LBFGSEpsx=1e-6"])
    # runSims(configs)
    plotSims(
        configs,
        "Loading settings",
        labels=labels,
        show=True,
        legend=True,
        title=f"{size}x{size}",
        addShift=True,
    )


def FIRELoading():
    nrThreads = 1
    nrSeeds = 40
    size = 60
    configs, labels = ConfigGenerator.generate(
        group_by_seeds=True,
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer="FIRE",
        # loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        # LBFGSEpsg=[1e-6, 1e-5, 5e-5, 1e-4],
        # loadIncrement=[1e-5],
        # eps=[1e-6, 1e-5, 1e-4],
        maxLoad=1.0,
        scenario="simpleShear",
    )
    extra_configs, extra_labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.0,
        nrThreads=nrThreads,
        loadIncrement=[1e-5],
        maxLoad=1.0,
        LBFGSEpsx=[1e-6],
        scenario="simpleShear",
    )
    # configs.extend([extra_configs])
    # labels.extend([["loadIncrement=1e-5, LBFGSEpsx=1e-6"]])
    plotLog(
        configs, "60x60, load:0.15-1, PBC, FIRE, t1, seeds:40", labels=labels, show=True
    )


def CGLoading():
    nrThreads = 1
    nrSeeds = 40
    size = 60
    configs, labels = ConfigGenerator.generate(
        group_by_seeds=True,
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer="CG",
        # loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        # CGEpsg=[1e-6, 1e-5, 5e-5, 1e-4],
        loadIncrement=[1e-5],
        CGEpsg=[1e-6, 1e-5, 1e-4],
        maxLoad=1.0,
        scenario="simpleShear",
    )
    extra_configs, extra_labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.0,
        nrThreads=nrThreads,
        loadIncrement=[1e-5],
        maxLoad=1.0,
        LBFGSEpsx=[1e-6],
        scenario="simpleShear",
    )
    # configs.extend([extra_configs])
    # labels.extend([["loadIncrement=1e-5, LBFGSEpsx=1e-6"]])
    plotLog(
        configs, "60x60, load:0.15-1, PBC, CG, t1, seeds:40", labels=labels, show=True
    )


if __name__ == "__main__":
    # plotOldStuff()
    # plotLessOldStuff()
    # fastStatStuff()
    # plotLessOldStuff()
    # runSims()
    # loadingSpeeds()
    # smallLoadingSpeeds()
    # FIRELoading()
    CGLoading()
    pass

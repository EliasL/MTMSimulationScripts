from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool


# Add Management to sys.path (used to import files)
# sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))

# Define dumpPath as a global variable
dumpPath = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1PBCt4EpsG0.01EpsF0.001s0/dumps/Dump_l0.650400_19.03~10.05.2024.mtsb"


def task(config):
    manager = SimulationManager(config)
    try:
        time = manager.runSimulation(build=False)
        # time = manager.resumeSimulation(dumpFile=dumpPath,
        #                                 overwriteSettings=True,
        #                                 build=False)
    except Exception as e:
        print(e)
        return f"Error: {e}"

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
    manager = SimulationManager(SimulationConfig(rows=3, cols=3, loadIncrement=0.1))
    try:
        manager.runSimulation()
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

    paths = get_csv_files(configs, useOldFiles=True)
    print("Plotting...")
    makeEnergyPlot(paths, f"{name}Energy.pdf", legend=["test"], **kwargs)
    for k in ["plot_average"]:
        if k in kwargs:
            del kwargs[k]
    # makePowerLawPlot(paths, f"{name}PowerLaw.pdf", legend=True, **kwargs)
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


if __name__ == "__main__":
    # plotOldStuff()
    # plotLessOldStuff()
    # fastStatStuff()
    # plotLessOldStuff()

    # runSims()
    pass

from collections.abc import Iterable
from pathlib import Path

from .configGenerator import ConfigGenerator, SimulationConfig


def LBFGSconfs(nrThreads, nrSeeds):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=nrThreads,
        loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        maxLoad=1.0,
        LBFGSEpsg=[1e-4, 5e-5, 1e-5, 1e-6],
        experiment="simpleShear",
    )
    return configs, labels


def CGconfs(nrThreads, nrSeeds):
    size = 60
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        L=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer="CG",
        loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        CGEpsg=[1e-6, 1e-5, 5e-5, 1e-4],
        # missing epsg 5e-5
        # loadIncrement=[1e-5],
        # eps=[1e-6, 1e-5, 1e-4],
        maxLoad=1.0,
        experiment="simpleShear",
    )
    return configs, labels


def bigJob(nrThreads, nrSeeds, size=200, group_by_variant=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        L=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer=["LBFGS", "CG", "FIRE"],
        loadIncrement=2e-4,
        LBFGSEpsg=1e-4,
        CGEpsg=1e-4,
        eps=1e-4,
        maxLoad=1.0,
        experiment="simpleShear",
    )
    return configs, labels


def allPlasticEventsJob():
    configs, labels = ConfigGenerator.generate(
        seed=[0],
        group_by_variant=False,
        L=100,
        startLoad=0.15,
        # initialGuessNoise=0.000001,
        nrThreads=20,
        minimizer=["LBFGS"],
        loadIncrement=1e-5,
        LBFGSEpsg=1e-8,
        # CGEpsg=1e-5,
        # eps=1e-8,
        maxLoad=1.0,
        experiment="simpleShear",
        # Save all events
        # plasticityEventThreshold=1e-6,
        energyDropThreshold=1e-10,
    )
    return configs, labels


def initalInstability():
    configs, labels = ConfigGenerator.generate(
        seed=[0],
        group_by_variant=False,
        L=200,
        startLoad=0.12,
        loadIncrement=1e-5,
        maxLoad=0.2,
        # initialGuessNoise=0.000001,
        nrThreads=8,
        minimizer=["LBFGS"],
        # LBFGSEpsg=1e-6,
        # CGEpsg=1e-5,
        LBFGSEpsx=1e-6,
        # eps=1e-8,
        usingPBC="false",
        experiment="simpleShearFixedBoundary",
    )
    return configs, labels

def pristineCrystal(size=[100,200, 300], loadIncrement=[1e-4, 1e-5, 1e-6], group_by_variant=True, reconnection="none", maxLoad=0.2):
    configs, labels = ConfigGenerator.generate(
        seed=0,
        group_by_variant=group_by_variant,
        L=size,
        startLoad=0.0,
        maxLoad=maxLoad,
        nrThreads=1,
        minimizer="LBFGS",
        loadIncrement=loadIncrement,
        LBFGSEpsx=1e-7,
        experiment="simpleShear",
        reconnectionMethod=reconnection,
        initialGuessNoise = 0.00,
    )
    return configs, labels

def smallPristineCrystal(size=[10,20, 30], loadIncrement=[1e-3, 1e-4, 1e-5], group_by_variant=True, reconnection="none", maxLoad=0.2):
    configs, labels = ConfigGenerator.generate(
        seed=0,
        group_by_variant=group_by_variant,
        L=size,
        startLoad=0.0,
        maxLoad=maxLoad,
        nrThreads=1,
        minimizer="LBFGS",
        loadIncrement=loadIncrement,
        LBFGSEpsx=1e-7,
        experiment="simpleShear",
        reconnectionMethod=reconnection,
        initialGuessNoise = 0.00,
    )
    return configs, labels

def propperJob(
    nrThreads, nrSeeds=0, size=100, group_by_variant=False, seeds=None, minimizer=None
):
    if minimizer is None:
        minimizer = ["LBFGS", "CG", "FIRE"]
    elif "L-BFGS" in minimizer:
        print("Warning! Maybe you meant LBFGS?")
    if seeds is None:
        seeds = range(nrSeeds)
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer=minimizer,
        loadIncrement=1e-5,
        LBFGSEpsg=1e-5,
        CGEpsg=1e-5,
        eps=1e-5,
        maxLoad=1.0,
        experiment="simpleShear",
    )
    return configs, labels


def propperJob1(**kwargs):
    return propperJob(3, 40, 60, **kwargs)


def propperJob2(**kwargs):
    return propperJob(6, 20, 100, **kwargs)


def propperJobCGANDLBFGS(**kwargs):
    return propperJob(3, nrSeeds=10, size=200, minimizer=["LBFGS", "CG"], **kwargs)


def largerPropperJobCGANDLBFGS(**kwargs):
    return propperJob(8, nrSeeds=10, size=300, minimizer=["LBFGS", "CG"], **kwargs)


def largePropperJob(FIREOnly=False, notFIRE=False, **kwargs):
    # set minimizer
    assert not (FIREOnly and notFIRE), "Cannot be both FIREOnly and notFIRE"
    if notFIRE:
        minimizer = ["LBFGS", "CG"]
    elif FIREOnly:
        minimizer = ["FIRE"]
    else:
        minimizer = ["LBFGS", "CG", "FIRE"]
    return propperJob(
        3,
        nrSeeds=10,
        size=200,
        minimizer=minimizer,
        **kwargs,
    )


def propperJobFIRE():
    return propperJob(3, nrSeeds=10, size=200, group_by_variant=False, minimizer=["FIRE"])


def basicJob(
    nrThreads, nrSeeds, size=100, group_by_variant=False, maxLoad=1.0, reconnection="none", logDuringMinimization=0,loadIncrement=1e-5, **kwargs,
):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.15,
        maxLoad=maxLoad,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=loadIncrement,
        # epsR=1e-6,
        LBFGSEpsx=1e-6,
        # LBFGSEpsg=1e-8,
        experiment="simpleShear",
        reconnectionMethod=reconnection,
        # remesh=1,
        # temp
        energyDropThreshold=1e-4,
        logDuringMinimization=logDuringMinimization,
        **kwargs
    )
    return configs, labels


def restartWithLogDuringMinimization(config, dump):
    config_path = Path(config)
    dump_path = Path(dump)
    if not config_path.is_file():
        raise FileNotFoundError(f"No config file found at {config_path}")
    if not dump_path.is_file():
        raise FileNotFoundError(f"No dump file found at {dump_path}")

    conf = SimulationConfig(config_path)
    conf.logDuringMinimization = 1
    conf.name = conf.generate_name(withExtension=False)

    from .runSimulation import run_locally

    run_locally(conf, dump=str(dump_path), newOutput=True)


def debugJob(
    nrThreads=1,
    nrSeeds=1,
    size=20,
    group_by_variant=False,
    maxLoad=1.0,
    reconnection="edgeFlip",
):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.2,
        maxLoad=maxLoad,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-3,
        epsR=1e-5,
        LBFGSEpsx=1e-6,
        experiment="simpleShear",
        reconnectionMethod=reconnection,
        energyDropThreshold=1e-10,
        # logDuringMinimization=1,
    )
    return configs, labels


def longJob(nrThreads, nrSeeds, size=100, group_by_variant=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.15,
        maxLoad=5.1,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        LBFGSEpsx=1e-6,
        epsR=1e-5,
        LBFGSEpsg=1e-8,
        experiment="simpleShear",
        reconnectionMethod="edgeFlip",
    )
    return configs, labels


def longJobStatic(nrThreads, nrSeeds, size=100, group_by_variant=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.15,
        maxLoad=1.0,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        LBFGSEpsx=1e-6,
        epsR=1e-5,
        LBFGSEpsg=1e-8,
        experiment="simpleShear",
        reconnectionMethod="none",
    )
    return configs, labels


def smallJob(**kwargs):
    return basicJob(nrThreads=1, nrSeeds=1, **kwargs)


def largeAvalanche(nrThreads, nrSeeds=1, seeds=None, LBFGSEpsg=1e-8):
    if seeds is None:
        seeds = range(nrSeeds)
    dump = "/Volumes/data/KeepSafe/large_avalanche_dump_simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0_l0.62787.mtsb"
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=False,
        rows=100,
        cols=100,
        startLoad=0.62787,
        maxLoad=0.628,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        LBFGSEpsg=LBFGSEpsg,
        experiment="simpleShear",
    )
    return configs, labels, dump


def smallAvalanches(nrThreads, nrSeeds=1, seeds=None):
    if seeds is None:
        seeds = range(nrSeeds)
    dump = "/Volumes/data/KeepSafe/smal_avalanche_dump_simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0_l0.63922.mtsb"
    # In the end, it might be better to use a dump that is a bit further back.
    # For example, this one from the large avalanche
    # dump = "/Volumes/data/KeepSafe/large_avalanche_dump_simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0_l0.62787.mtsb"
    LBFGSEpsg = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    kwargs = {
        "seed": seeds,
        "group_by_variant": False,
        "rows": 100,
        "cols": 100,
        "startLoad": 0.62787,
        "maxLoad": 0.6422,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "LBFGSEpsg": LBFGSEpsg,
        "experiment": "simpleShear",
    }
    configs, labels = ConfigGenerator.generate(**kwargs)
    # Also add a simulation using Epsx
    del kwargs["LBFGSEpsg"]
    kwargs["LBFGSEpsx"] = 1e-6
    configsX, labelsX = ConfigGenerator.generate(**kwargs)
    configs.append(configsX[0])
    labels.append("LBFGSEpsx=1e-06")
    return configs, labels, dump


def avalanches(nrThreads, nrSeeds=1, seeds=None, size=100):
    if seeds is None:
        seeds = range(nrSeeds)
    dump = "/Volumes/data/KeepSafe/dump_l0.53.mtsb"
    LBFGSEpsg = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    kwargs = {
        "seed": seeds,
        "group_by_variant": False,
        "rows": size,
        "cols": size,
        "startLoad": 0.15,
        "maxLoad": 1.0,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "LBFGSEpsg": LBFGSEpsg,
        "experiment": "simpleShear",
    }
    configs, labels = ConfigGenerator.generate(**kwargs)
    configs, labels = [configs[0]], [labels[0]]
    epsR = [1e-6, 1e-5, 1e-4, 1e-3]
    kwargs = {
        "seed": seeds,
        "group_by_variant": False,
        "rows": size,
        "cols": size,
        "startLoad": 0.15,
        "maxLoad": 1.0,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "epsR": epsR,
        "experiment": "simpleShear",
    }
    epsRconfigs, epsRlabels = ConfigGenerator.generate(**kwargs)
    configs.extend(epsRconfigs)
    labels.extend(epsRlabels)
    # Also add a simulation using Epsx
    del kwargs["epsR"]
    kwargs["LBFGSEpsx"] = 1e-6
    configsX, labelsX = ConfigGenerator.generate(**kwargs)
    # configs.append(configsX[0])
    # labels.append("LBFGSEpsx=1e-06")
    return configs, labels, dump


def backwards(nrThreads, nrSeeds=1, seeds=None, LBFGSEpsg=1e-8):
    if seeds is None:
        seeds = range(nrSeeds)
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=False,
        rows=100,
        cols=100,
        startLoad=-0.15,
        maxLoad=-1.0,
        loadIncrement=-1e-5,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        LBFGSEpsg=LBFGSEpsg,
        experiment="simpleShear",
    )
    return configs, labels


def cyclicLoading(nrThreads, nrSeeds=1, seeds=None):
    if seeds is None:
        seeds = range(nrSeeds)
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=False,
        rows=100,
        cols=100,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=1e-5,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        experiment="cyclicSimpleShear",
    )
    return configs, labels


def findMinimizationCriteriaJobs(nrSeeds=5, seeds=None):
    L = [30, 40, 60, 80, 100]
    loadIncrement = [1e-5, 1e-4, 1e-3]
    epsR = [1e-6, 1e-5, 1e-4, 1e-3]

    if seeds is None:
        seeds = range(nrSeeds)

    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=False,
        L=L,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=loadIncrement,
        nrThreads=4,
        minimizer="LBFGS",
        epsR=epsR,
    )
    return configs, labels


def compareWithOldStoppingCriteria(nrSeeds=5, seeds=None):
    L = [30, 40, 60, 80, 100]
    loadIncrement = [1e-5]
    LBFGSEpsg = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    LBFGSEpsx = [1e-6]

    if seeds is None:
        seeds = range(nrSeeds)

    configs, labels = ConfigGenerator.generate(
        LBFGSEpsg=LBFGSEpsg,
        seed=seeds,
        group_by_variant=False,
        L=L,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=loadIncrement,
        nrThreads=4,
        minimizer="LBFGS",
    )

    configsx, labelsx = ConfigGenerator.generate(
        LBFGSEpsx=LBFGSEpsx,
        seed=seeds,
        group_by_variant=False,
        L=L,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=loadIncrement,
        nrThreads=4,
        minimizer="LBFGS",
    )

    configs.extend(configsx)
    labels.extend(labelsx)

    return configs, labels


def reconnectionTest(L=100):
    configs1, labels1 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="minor", reconnecton="none"
    )
    configs2, labels2 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="major", reconnecton="none"
    )
    configs3, labels3 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="major", reconnecton="edgeFlip"
    )
    configs4, labels4 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="minor", reconnecton="edgeFlip"
    )
    configs5, labels5 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="major", reconnecton="delaunay"
    )
    configs6, labels6 = doubleDislocationTest(
        nrThreads=3, nrSeeds=1, L=L, diagonal="minor", reconnecton="delaunay"
    )
    configs = configs1 + configs2 + configs3 + configs4 + configs5 + configs6
    labels = [
        "Minor",
        "Major",
        "Minor with edge flip",
        "Major with edge flip",
        "Major with Delaunay",
        "Minor with Delaunay",
    ]
    return configs, labels


def fixedBoundaries(nrThreads, nrSeeds=1, seeds=None, L=40, fixed=True):
    if seeds is None:
        seeds = range(nrSeeds)
    experiment = "simpleShearFixedBoundary" if fixed else "simpleShear"
    usingPBC = "false" if fixed else "true"
    configs, labels = ConfigGenerator.generate(
        usingPBC=usingPBC,
        seed=seeds,
        group_by_variant=False,
        rows=L,
        cols=L,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=1e-6,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        # LBFGSEpsx=1e-6,
        experiment=experiment,
    )
    return configs, labels


def showMinimizationCriteriaJobs(nrSeeds=5, seeds=None):
    L = [400, 200, 100]
    loadIncrement = [1e-5]
    epsR = [1e-5, None]
    LBFGSEpsx = [1e-6, None]
    LBFGSEpsg = [1e-7, None]

    if seeds is None:
        seeds = range(nrSeeds)

    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_variant=False,
        L=L,
        startLoad=0.15,
        maxLoad=0.16,
        loadIncrement=loadIncrement,
        nrThreads=6,
        minimizer="LBFGS",
        LBFGSEpsx=LBFGSEpsx,
        LBFGSEpsg=LBFGSEpsg,
        epsR=epsR,
    )
    # Filter out configs and labels where labels contain either zero or two instances of None
    filtered_data = [(c, l) for c, l in zip(configs, labels) if l.count("None") == 1]

    # Unpack filtered configs and labels
    configs, labels = zip(*filtered_data) if filtered_data else ([], [])
    return configs, labels


def doubleDislocationTest(
    nrThreads=3,
    nrSeeds=1,
    seeds=None,
    L=10,
    diagonal=["minor"],
    reconnecton=["none", "edgeFlip"],
    GP1=[0.0],# 1.0],
    GP2=[2.0],# 1.0],
    GP3=1.0, # Load direction change point
):
    if seeds is None:
        seeds = range(nrSeeds)
    experiment = "doubleDislocationTest"
    # experiment = "simpleShear"
    configs, labels = ConfigGenerator.generate(
        usingPBC="false",
        seed=seeds,
        group_by_variant=False,
        L=L,
        #rows=10,
        #cols=[50, 75, 100],
        startLoad=0.0,
        maxLoad=4.0,
        loadIncrement=1e-3,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        experiment=experiment,
        meshDiagonal=diagonal,
        reconnectionMethod=reconnecton,
        logDuringMinimization=1,
        GP1=GP1,
        GP2=GP2,
        GP3=GP3,
    )
    return configs, labels


def singleDislocationTest(
    nrThreads=3, nrSeeds=1, seeds=None, L=50, diagonal=["major", "minor"]
):
    if seeds is None:
        seeds = range(nrSeeds)
    experiment = "singleDislocationFixedBoundaryTest"
    configs, labels = ConfigGenerator.generate(
        usingPBC="false",
        seed=seeds,
        group_by_variant=False,
        L=L,
        startLoad=0.0,
        maxLoad=1.0,
        loadIncrement=1e-3,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        experiment=experiment,
        meshDiagonal=diagonal,
    )
    return configs, labels


def reversibilityJob(
    nrThreads=4,
    nrSeeds=1,
    size=150,
    group_by_variant=False,
    maxLoad=1.0,
    reconnection="edgeFlip",
    loadIncrement=1e-5,
    LBFGSEpsx=1e-6,
):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.14,
        maxLoad=maxLoad,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=loadIncrement,
        # epsR=1e-14,
        LBFGSEpsx=LBFGSEpsx,
        # LBFGSEpsg=1e-8,
        experiment="reversibilityProtocolTest",
        reconnectionMethod=reconnection,
        # remesh=1,
        # temp
        energyDropThreshold=1e-5,
        #logDuringMinimization=1,
    )
    return configs, labels


def sylvainBatches(
    batch: int,
    nrSeeds: int = 4,
    size: int = 100,
    threads=3,
    group_by_variant: bool = False,
    reconnection= "none",
):
    """
    Batch plan (all based on reversibilityJob):
      1) Basic batch (L=100, threads 3/4, reconnection none/edgeFlip, 4 seeds)
      2) loadIncrement=1e-4
      3) loadIncrement=5e-5
      4) loadIncrement=5e-6
      5) loadIncrement=1e-6
      6) LBFGSEpsx=1e-4
      7) LBFGSEpsx=1e-5
      8) LBFGSEpsx=1e-7
    """
    batch_map = {
        -2: {"loadIncrement": 1e-5, "LBFGSEpsx": [1e-7, 1e-6,1e-5,1e-4][::-1]},
        -1: {"loadIncrement": [1e-6, 5e-6,1e-5,5e-5, 1e-4][::-1], "LBFGSEpsx": 1e-6},
        1: {"loadIncrement": 1e-5, "LBFGSEpsx": 1e-6},
        2: {"loadIncrement": 1e-4, "LBFGSEpsx": 1e-6},
        3: {"loadIncrement": 5e-5, "LBFGSEpsx": 1e-6},
        4: {"loadIncrement": 5e-6, "LBFGSEpsx": 1e-6},
        5: {"loadIncrement": 1e-6, "LBFGSEpsx": 1e-6},
        6: {"loadIncrement": 1e-5, "LBFGSEpsx": 1e-4},
        7: {"loadIncrement": 1e-5, "LBFGSEpsx": 1e-5},
        8: {"loadIncrement": 1e-5, "LBFGSEpsx": 1e-7},
    }
    if batch not in batch_map:
        raise ValueError(f"Unknown batch {batch}. Expected 1..8.")

    params = batch_map[batch]
    configs, labels = reversibilityJob(
        nrThreads=threads,
        nrSeeds=nrSeeds,
        size=size,
        group_by_variant=group_by_variant,
        reconnection=reconnection,
        loadIncrement=params["loadIncrement"],
        LBFGSEpsx=params["LBFGSEpsx"],
    )
    return configs, labels


def reconnectTest(diagonal="major", reconnectionMethod="edgeFlip"):
    configs, labels = ConfigGenerator.generate(
        usingPBC="false",
        L=3,
        meshDiagonal=diagonal,
        startLoad=0.0,
        maxLoad=1.0,
        loadIncrement=0.001,
        minimizer="LBFGS",
        experiment="reconnectTest",
        reconnectionMethod=reconnectionMethod,
    )
    return configs, labels

def reconnectSSTest(diagonal="major", reconnectionMethod="edgeFlip",
                    refConfig=[0.0,1.0,2.0, 4.0], disp=[0.0, 1e-2, 1e-1], vertLoad=False):
    configs, labels = ConfigGenerator.generate(
        usingPBC="true",
        L=3,
        meshDiagonal=diagonal,
        startLoad=0.0,
        maxLoad=3.0,
        loadIncrement=0.01,
        minimizer="LBFGS",
        experiment="reconnectSSTest",
        reconnectionMethod=reconnectionMethod,
        GP1=refConfig,
        GP2=disp,
        GP3=float(vertLoad),
    )
    return configs, labels

def reconnectionJob(L=100):
    configs, labels = ConfigGenerator.generate(
        usingPBC="true",
        L=L,
        meshDiagonal="major",
        reconnectionMethod="edgeFlip",
        epsR=1e-5,
        startLoad=0.15,
        maxLoad=2.0,
        nrThreads=8,
        loadIncrement=1e-5,
        minimizer="LBFGS",
        experiment="simpleShear",
        # logDuringMinimization=1,
    )
    return configs, labels


def umutTestJob(group_by_variant=False):
    configs, labels = ConfigGenerator.generate(
        seed=0,
        L=100,
        startLoad=0.138,
        maxLoad=0.3,
        initialGuessNoise=0.04,
        nrThreads=4,
        loadIncrement=1e-5,
        minimizer="LBFGS",
        LBFGSEpsg=1e-5,
        experiment="simpleShear",
        group_by_variant=group_by_variant,
    )
    return configs, labels


def umutJob(
    L,
    loadIncrement: float | list[float] = 2e-5,
    EliasStop=False,
    group_by_variant=False,
    reconnecton: str | list[str] = "none",
):
    if EliasStop:
        stop = {"epsR": 1e-5}
    else:
        stop = {"LBFGSEpsx": 1e-5}
    configs, labels = ConfigGenerator.generate(
        seed=0,
        L=L,
        startLoad=0.138,
        maxLoad=1.0,
        initialGuessNoise=0.04,
        nrThreads=8,
        loadIncrement=loadIncrement,
        minimizer="LBFGS",
        experiment="simpleShear",
        group_by_variant=group_by_variant,
        reconnectionMethod=reconnecton,
        **stop,
    )
    return configs, labels


def bigUmutJob(group_by_variant=False):
    return umutJob(500, EliasStop=False, group_by_variant=group_by_variant)


def bigUmutJobWithEliasStop(group_by_variant=False):
    return umutJob(500, EliasStop=True, group_by_variant=group_by_variant)


def stopConditionJob():
    c1, l1 = umutJob([200, 100])
    c2, l2 = umutJob([200, 100], EliasStop=True)
    c = c1 + c2
    labs = list(map(lambda lab: f"epsX=1e-5, {lab}", l1)) + list(
        map(lambda lab: f"epsR=1e-5, {lab}", l2)
    )
    return c, labs


def loadStepJob(group_by_variant=True, reconnection=["none", "edgeFlip"]):
    return umutJob(
        L=150,
        loadIncrement=[5e-6, 1e-5, 2e-5, 4e-5],
        group_by_variant=group_by_variant,
        reconnecton=reconnection,
    )


def umutJobs(loadIncrement=2e-5):
    """
    Generates a job for size scaling tests.
    """
    sizes = [50, 100, 200, 250, 300]
    nr_samples = [10, 10, 10, 10, 10]
    nr_threads = [4, 4, 4, 8, 8]
    sizes.reverse()
    nr_samples.reverse()
    nr_threads.reverse()
    all_configs = []
    all_labels = []
    for size, samples, threads in zip(sizes, nr_samples, nr_threads):
        configs, labels = ConfigGenerator.generate(
            seed=range(samples),
            rows=size,
            cols=size,
            startLoad=0.138,
            maxLoad=1.0,
            loadIncrement=loadIncrement,
            nrThreads=threads,
            minimizer="LBFGS",
            LBFGSEpsx=1e-6,
            experiment="simpleShear",
        )
        # Append the generated configs and labels to the main lists
        all_configs.append(configs)
        all_labels.append(list(map(lambda x: f"L={size}, " + x, labels)))

    return all_configs, all_labels


def size_scaling_job():
    """
    Generates a job for size scaling tests.
    """
    sizes = [50, 100, 150, 200, 250]  # , 300]
    nr_samples = [10, 10, 10, 10, 10]  # , 10]
    nr_threads = [2, 3, 4, 8, 8]  # , 8]
    sizes.reverse()
    nr_samples.reverse()
    nr_threads.reverse()
    all_configs = []
    all_labels = []
    for size, samples, threads in zip(sizes, nr_samples, nr_threads):
        configs, labels = ConfigGenerator.generate(
            seed=range(samples),
            rows=size,
            cols=size,
            startLoad=0.15,
            maxLoad=1.0,
            loadIncrement=1e-5,
            nrThreads=threads,
            minimizer="LBFGS",
            LBFGSEpsx=1e-6,
            # epsR=1e-6,
            experiment="simpleShear",
        )
        # Append the generated configs and labels to the main lists
        all_configs.append(configs)
        all_labels.append(list(map(lambda x: f"L={size}, " + x, labels)))

    return all_configs, all_labels


def triangular_edge_flip_job(
    size=200,
    group_by_variant=False,
    maxLoad=1.0,
    reconnection="edgeFlip",
):
    configs, labels = ConfigGenerator.generate(
        seed=range(1),
        group_by_variant=group_by_variant,
        rows=size,
        cols=size,
        startLoad=0.29,
        maxLoad=maxLoad,
        nrThreads=4,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        # epsR=1e-14,
        LBFGSEpsx=1e-6,
        # LBFGSEpsg=1e-8,
        experiment="simpleShear",
        reconnectionMethod=reconnection,
        energyFunction="contiTriangular",
        # remesh=1,
        # temp
        # energyDropThreshold=1e-10,
        # logDuringMinimization=1,
    )
    return configs, labels


def stiffnessMatrixComparisonJob(
    size=50,
    nrSeeds=1,
    seeds=None,
    nrThreads=4,
    maxLoad=1.0,
    loadIncrement=1e-5,
    referenceShear=0.5,
    referenceReconnection="none",
    nrVTUFrames=400,
    energyDropThreshold=1e-10,
    plasticityEventThreshold=0.0,
    writeDumps=0,
    showProgress=1,
    group_by_variant=False,
    startLoad=0.0,
):
    if startLoad != 0.0:
        raise ValueError(
            "Use startLoad=0.0 for this comparison. simpleShearReferenceTest "
            "initializes the reference state differently from simpleShear."
        )

    if seeds is None:
        seeds = range(nrSeeds)

    if isinstance(referenceShear, str) or not isinstance(referenceShear, Iterable):
        reference_shears = [float(referenceShear)]
    else:
        reference_shears = [float(shear) for shear in referenceShear]
    if len(reference_shears) == 0:
        raise ValueError("referenceShear must contain at least one value.")

    if isinstance(referenceReconnection, str):
        reference_reconnections = [referenceReconnection]
    else:
        reference_reconnections = list(referenceReconnection)
    if len(reference_reconnections) == 0:
        raise ValueError("referenceReconnection must contain at least one value.")

    common = dict(
        seed=seeds,
        rows=size,
        cols=size,
        startLoad=startLoad,
        maxLoad=maxLoad,
        loadIncrement=loadIncrement,
        nrThreads=nrThreads,
        initialGuessNoise=0.0,
        minimizer="LBFGS",
        LBFGSEpsx=1e-6,
        energyDropThreshold=energyDropThreshold,
        plasticityEventThreshold=plasticityEventThreshold,
        nrVTUFrames=nrVTUFrames,
        writeDumps=writeDumps,
        showProgress=showProgress,
    )
    variants = [
        (
            "simple shear, no reconnection",
            dict(experiment="simpleShear", reconnectionMethod="none", GP1=0.0),
        ),
        (
            "simple shear, edge flip",
            dict(experiment="simpleShear", reconnectionMethod="edgeFlip", GP1=0.0),
        ),
    ]
    for shear in reference_shears:
        for reconnection_method in reference_reconnections:
            reconnection_label = (
                "no reconnection"
                if reconnection_method == "none"
                else reconnection_method
            )
            variants.append(
                (
                    f"reference shear {shear:g}, {reconnection_label}",
                    dict(
                        experiment="simpleShearReferenceTest",
                        reconnectionMethod=reconnection_method,
                        GP1=shear,
                    ),
                )
            )

    all_configs = []
    all_labels = []
    for label, variant in variants:
        configs, labels = ConfigGenerator.generate(**common, **variant)
        labels = [
            label if seed_label == "" else f"{label}, {seed_label}"
            for seed_label in labels
        ]
        if group_by_variant:
            all_configs.append(configs)
            all_labels.append(labels)
        else:
            all_configs.extend(configs)
            all_labels.extend(labels)

    return all_configs, all_labels


def currentDistortionComparisonJob(
    size=50,
    nrSeeds=1,
    seeds=None,
    nrThreads=4,
    startLoads=(0.0, 2.0, 5.0, 10.0),
    window=1.0,
    reconnection=("none", "edgeFlip"),
    loadIncrement=1e-5,
    nrVTUFrames=400,
    energyDropThreshold=1e-10,
    plasticityEventThreshold=0.0,
    writeDumps=0,
    showProgress=1,
    group_by_variant=False,
):
    if seeds is None:
        seeds = range(nrSeeds)

    if isinstance(startLoads, str) or not isinstance(startLoads, Iterable):
        start_loads = [float(startLoads)]
    else:
        start_loads = [float(start_load) for start_load in startLoads]
    if len(start_loads) == 0:
        raise ValueError("startLoads must contain at least one value.")
    if window <= 0.0:
        raise ValueError("window must be positive.")

    if isinstance(reconnection, str):
        reconnections = [reconnection]
    else:
        reconnections = list(reconnection)
    if len(reconnections) == 0:
        raise ValueError("reconnection must contain at least one value.")

    common = dict(
        seed=seeds,
        rows=size,
        cols=size,
        loadIncrement=loadIncrement,
        nrThreads=nrThreads,
        initialGuessNoise=0.0,
        minimizer="LBFGS",
        LBFGSEpsx=1e-6,
        experiment="simpleShear",
        energyDropThreshold=energyDropThreshold,
        plasticityEventThreshold=plasticityEventThreshold,
        nrVTUFrames=nrVTUFrames,
        writeDumps=writeDumps,
        showProgress=showProgress,
    )

    all_configs = []
    all_labels = []
    for start_load in start_loads:
        max_load = start_load + window
        for reconnection_method in reconnections:
            reconnection_label = (
                "no reconnection"
                if reconnection_method == "none"
                else reconnection_method
            )
            label = (
                f"current shear {start_load:g} to {max_load:g}, "
                f"{reconnection_label}"
            )
            configs, labels = ConfigGenerator.generate(
                **common,
                startLoad=start_load,
                maxLoad=max_load,
                reconnectionMethod=reconnection_method,
            )
            labels = [
                label if seed_label == "" else f"{label}, {seed_label}"
                for seed_label in labels
            ]
            if group_by_variant:
                all_configs.append(configs)
                all_labels.append(labels)
            else:
                all_configs.extend(configs)
                all_labels.extend(labels)

    return all_configs, all_labels

from .configGenerator import ConfigGenerator


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
        scenario="simpleShear",
    )
    return configs, labels


def CGconfs(nrThreads, nrSeeds):
    size = 60
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer="CG",
        loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        CGEpsg=[1e-6, 1e-5, 5e-5, 1e-4],
        # missing epsg 5e-5
        # loadIncrement=[1e-5],
        # eps=[1e-6, 1e-5, 1e-4],
        maxLoad=1.0,
        scenario="simpleShear",
    )
    return configs, labels


def bigJob(nrThreads, nrSeeds, size=200, group_by_seeds=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_seeds=group_by_seeds,
        rows=size,
        cols=size,
        startLoad=0.15,
        nrThreads=nrThreads,
        minimizer=["LBFGS", "CG", "FIRE"],
        loadIncrement=2e-4,
        LBFGSEpsg=1e-4,
        CGEpsg=1e-4,
        eps=1e-4,
        maxLoad=1.0,
        scenario="simpleShear",
    )
    return configs, labels


def allPlasticEventsJob():
    configs, labels = ConfigGenerator.generate(
        seed=[0],
        group_by_seeds=False,
        rows=100,
        cols=100,
        startLoad=0.15,
        # initialGuessNoise=0.000001,
        nrThreads=20,
        minimizer=["LBFGS"],
        loadIncrement=1e-5,
        LBFGSEpsg=1e-8,
        # CGEpsg=1e-5,
        # eps=1e-8,
        maxLoad=1.0,
        scenario="simpleShear",
        # Save all events
        # plasticityEventThreshold=1e-6,
        energyDropThreshold=1e-10,
    )
    return configs, labels


def propperJob(
    nrThreads, nrSeeds=0, size=100, group_by_seeds=False, seeds=None, minimizer=None
):
    if minimizer is None:
        minimizer = ["LBFGS", "CG", "FIRE"]
    if seeds is None:
        seeds = range(nrSeeds)
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_seeds=group_by_seeds,
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
        scenario="simpleShear",
    )
    return configs, labels


def propperJob1(**kwargs):
    return propperJob(3, 40, 60, **kwargs)


def propperJob2(**kwargs):
    return propperJob(6, 20, 100, **kwargs)


def propperJob3(**kwargs):
    return propperJob(56, 2, 200, minimizer=["LBFGS", "CG"], **kwargs)


def basicJob(nrThreads, nrSeeds, size=100, group_by_seeds=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_seeds=group_by_seeds,
        rows=size,
        cols=size,
        startLoad=0.15,
        maxLoad=1.0,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        epsR=1e-5,
        LBFGSEpsg=1e-8,
        scenario="simpleShear",
    )
    return configs, labels


def longJob(nrThreads, nrSeeds, size=100, group_by_seeds=False):
    configs, labels = ConfigGenerator.generate(
        seed=range(nrSeeds),
        group_by_seeds=group_by_seeds,
        rows=size,
        cols=size,
        startLoad=0.15,
        maxLoad=2.0,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        epsR=1e-5,
        LBFGSEpsg=1e-8,
        scenario="simpleShear",
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
        group_by_seeds=False,
        rows=100,
        cols=100,
        startLoad=0.62787,
        maxLoad=0.628,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        loadIncrement=1e-5,
        LBFGSEpsg=LBFGSEpsg,
        scenario="simpleShear",
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
        "group_by_seeds": False,
        "rows": 100,
        "cols": 100,
        "startLoad": 0.62787,
        "maxLoad": 0.6422,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "LBFGSEpsg": LBFGSEpsg,
        "scenario": "simpleShear",
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
        "group_by_seeds": False,
        "rows": size,
        "cols": size,
        "startLoad": 0.15,
        "maxLoad": 1.0,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "LBFGSEpsg": LBFGSEpsg,
        "scenario": "simpleShear",
    }
    configs, labels = ConfigGenerator.generate(**kwargs)
    configs, labels = [configs[0]], [labels[0]]
    epsR = [1e-6, 1e-5, 1e-4, 1e-3]
    kwargs = {
        "seed": seeds,
        "group_by_seeds": False,
        "rows": size,
        "cols": size,
        "startLoad": 0.15,
        "maxLoad": 1.0,
        "nrThreads": nrThreads,
        "minimizer": "LBFGS",
        "loadIncrement": 1e-5,
        "epsR": epsR,
        "scenario": "simpleShear",
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
        group_by_seeds=False,
        rows=100,
        cols=100,
        startLoad=-0.15,
        maxLoad=-1.0,
        loadIncrement=-1e-5,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        LBFGSEpsg=LBFGSEpsg,
        scenario="simpleShear",
    )
    return configs, labels


def cyclicLoading(nrThreads, nrSeeds=1, seeds=None, LBFGSEpsg=1e-8):
    if seeds is None:
        seeds = range(nrSeeds)
    configs, labels = ConfigGenerator.generate(
        seed=seeds,
        group_by_seeds=False,
        rows=100,
        cols=100,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=1e-5,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        LBFGSEpsg=LBFGSEpsg,
        scenario="cyclicSimpleShear",
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
        group_by_seeds=False,
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
        group_by_seeds=False,
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
        group_by_seeds=False,
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


def fixedBoundaries(nrThreads, nrSeeds=1, seeds=None, L=40, fixed=True):
    if seeds is None:
        seeds = range(nrSeeds)
    scenario = "simpleShearFixedBoundary" if fixed else "simpleShear"
    usingPBC = "false" if fixed else "true"
    configs, labels = ConfigGenerator.generate(
        usingPBC=usingPBC,
        seed=seeds,
        group_by_seeds=False,
        rows=L,
        cols=L,
        startLoad=0.15,
        maxLoad=1.0,
        loadIncrement=1e-5,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        # LBFGSEpsx=1e-6,
        scenario=scenario,
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
        group_by_seeds=False,
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


def singleDislocationTest(
    nrThreads=3, nrSeeds=1, seeds=None, L=10, diagonal=["major", "minor"]
):
    if seeds is None:
        seeds = range(nrSeeds)
    scenario = "singleDislocationTest"
    configs, labels = ConfigGenerator.generate(
        usingPBC="false",
        seed=seeds,
        group_by_seeds=False,
        rows=L,
        cols=L,
        startLoad=0.0,
        maxLoad=2.0,
        loadIncrement=1e-3,
        nrThreads=nrThreads,
        minimizer="LBFGS",
        epsR=1e-6,
        scenario=scenario,
        meshDiagonal=diagonal,
    )
    return configs, labels

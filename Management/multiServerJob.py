from itertools import product
from collections import OrderedDict
from .configGenerator import ConfigGenerator, SimulationConfig

from .runOnCluster import queue_remote_job, build_on_all_servers, run_remote_command
from .clusterStatus import get_all_server_info, get_server_short_name
from .jobManager import JobManager
from .dataManager import DataManager


def confToCommand(config, **kwargs):
    base_command = "python3 ~/simulation/SimulationScripts/runSimulations.py"
    return (
        base_command
        # Get args from config
        + dictToString(ConfigGenerator.get_kwargs(config))
        # We add other arguments as well
        + dictToString(kwargs)
    )


def dictToString(dictionary):
    return " " + " ".join(  # We need to add "" to strings
        f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"
        for key, value in dictionary.items()
    )


def distributeConfigs(configs, threads_per_seed=1):
    """
    We give each seed its own slurm job and we just fill the entire cluster with
    lots of small jobs with a high nice value so that others can get past if they
    want.

    Importantly, we also need to check the servers for exsisting folders, since
    we might want to resume a simulation instead of starting a new one.
    """

    dm = DataManager()
    dm.findData(autoUpdate=True)

    # A dictionary with server names as keys, and all configs that should be run on that server
    serverConfigDict = OrderedDict()

    remaining_configs = []

    # Assign exsisting jobs to servers with that job
    for config in configs:
        configSolved = False
        data = {k: v for k, v in dm.data.items() if k != "date"}
        for server, (folders, sizes, free_space) in data.items():
            folders = map(lambda s: s.split("/")[-1], folders)
            if config.name in folders:
                if server not in serverConfigDict:
                    serverConfigDict[server] = []
                serverConfigDict[server].append(config)
                configSolved = True
                break
            else:
                print(f"{config.name} not found.")
        if not configSolved:
            remaining_configs.append(config)

    print(
        f"Resuming {sum(len(lst) for lst in serverConfigDict.values())} exsisting jobs."
    )
    if len(remaining_configs) > 0:
        print(f"Finding cpu space for {len(remaining_configs)} new jobs.")
    else:
        return serverConfigDict
    print("Getting server info... ")
    serverInfo = get_all_server_info()

    # Sort the server information by server name
    for si in sorted(serverInfo.values(), key=lambda x: x.sName):
        # Skip servers that are down or drained
        if "down" in si.nodeState or "drained" in si.nodeState:
            continue

        # Skip the condorcet server as it is slow
        if "condorcet" in si.sName:
            continue

        # We try to avoid almost filled servers
        if si.nrFreeCores < 10:
            continue

        # Initialize command list for the server if not already present
        if si.sName not in serverConfigDict:
            serverConfigDict[si.sName] = []

        # Calculate the number of available slots (leave one CPU free)
        available_slots = (si.nrFreeCores) // threads_per_seed - len(
            serverConfigDict[si.sName]
        )

        # Fill servers until they are full or no remaining configurations
        while available_slots > 0 and remaining_configs:
            serverConfigDict[si.sName].append(remaining_configs.pop(0))
            available_slots -= 1

        # If no remaining configurations, return the filled commands
        if not remaining_configs:
            return serverConfigDict
    raise RuntimeError(
        f"Not enough cores to run simulations! Need {len(remaining_configs) * threads_per_seed} cores."
    )


def queueJobs(server, configs, job_name="el", **kwargs):
    """
    Kwargs:
    resume=True,
    dump=None,
    plot=False,
    newOutput=False,
    """

    pre_command = "python3 ~/simulation/SimulationScripts/Management/queueLocalJobs.py"

    if isinstance(configs, SimulationConfig):
        configs = [configs]

    commands = [confToCommand(conf, **kwargs) for conf in configs]

    full_pre_command = (
        pre_command
        + " "
        + str(
            {
                '"commands"': str(commands).replace('"', "\u203d"),
                '"job_name"': f'"{job_name}"',
            }
        )
    )
    run_remote_command(server, full_pre_command)
    # result = subprocess.run(full_pre_command, shell=True, text=True)

    # for command in commands:
    #    jobId = queue_remote_job(server, command, "preciceJs", nrThreads)
    #    # print(command)
    #    pass
    print(f"Started {len(commands)} jobs on {get_server_short_name(server)}")


def oldGenerateCommands(configs, threads_per_seed=1):
    """
    The idea here is that we will have a huge number of configuration files, and we
    want to distribute them among all the available servers.

    We are going to send the arguments to the generate function from ConfigGenerator
    to the runSimulations script, and this scipt will generate all the configurations
    that will run on that server
    """
    # the base command
    base_command = "python3 ~/simulation/SimulationScripts/Management/runSimulations.py"

    # get the kwargs used to generate the configs
    kwargs = ConfigGenerator.get_kwargs(configs)

    # Find the number of seeds and the number of configs
    nr_seeds = len(set(config.seed for config in configs))

    seeds = kwargs["seed"]
    del kwargs["seed"]
    # Prepare the remaining arguments, ensuring they are iterable
    processed_kwargs = OrderedDict()
    for key, value in sorted(kwargs.items()):
        if isinstance(value, str) or not isinstance(value, list):
            processed_kwargs[key] = [value]  # Treat single values as a list
        else:
            processed_kwargs[key] = sorted(value)

    # Generate all combinations of non-seed parameters
    kwargs_combi = list(product(*processed_kwargs.values()))
    kwarg_index = 0  # Used to index kwargs_combi

    # First we want to find the number of available cores on each server
    # serverInfo[0].nrFreeCores
    print("Getting server info... ")
    serverInfo = get_all_server_info()

    print("Creating commadns... ")
    commands = OrderedDict()
    # If the number of servers with more free cores than nr_seeds is larger than
    # nr_max_batches, we simply put one batch on each of these servers
    # We alphabetically sort just so that the order is not random
    for si in sorted(serverInfo.values(), key=lambda x: x.sName):
        if "down" in si.nodeState or "drained" in si.nodeState:
            continue
        if "condorcet" in si.sName:
            # condorcet is slow
            continue

        if si.sName not in commands:
            commands[si.sName] = []
        while si.nrFreeCores > nr_seeds * threads_per_seed * (
            1 + len(commands[si.sName])
        ):
            # This creates a dictionary from keys and tuple values
            combi_dict = zip(kwargs.keys(), kwargs_combi[kwarg_index])
            kwarg_index += 1

            cmd = (
                base_command
                + " "
                + " ".join(  # We need to add "" to strings
                    f'{key}="{value}"' if isinstance(value, str) else f"{key}={value}"
                    for key, value in combi_dict
                )
            )

            full_command = f'{cmd} seed="{seeds}"'
            commands[si.sName].append(full_command)
            if kwarg_index >= len(kwargs_combi):
                for key, value in commands.items():
                    pass
                # print("All commands wiritten")
                return commands
    remaining_settings = len(kwargs_combi) - kwarg_index
    raise RuntimeError(
        f"Not enough cores to run simulations! Need {remaining_settings} batches of {nr_seeds} cores."
    )


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
        seed=[42],
        group_by_seeds=False,
        rows=100,
        cols=100,
        startLoad=0.15,
        initialGuessNoise=0.000001,
        nrThreads=8,
        minimizer=["LBFGS"],
        loadIncrement=1e-5,
        LBFGSEpsg=1e-8,
        # CGEpsg=1e-5,
        # eps=1e-5,
        maxLoad=1.03,
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
    import numpy as np

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
        # eps=1e-8,
        LBFGSEpsg=1e-8,
        scenario="simpleShear",
    )
    return configs, labels


def smallJob(**kwargs):
    return basicJob(nrThreads=1, nrSeeds=1, **kwargs)

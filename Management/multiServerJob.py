from itertools import product
from configGenerator import ConfigGenerator

from runOnCluster import queue_remote_job, build_on_all_servers  # noqa: F401
from clusterStatus import get_all_server_info, get_server_short_name
from jobManager import JobManager


def generateCommands(configs, threads_per_seed=1):
    """
    The idea here is that we will have a huge number of configuration files, and we
    want to distribute them among all the available servers.

    We are going to send the arguments to the generate function from ConfigGenerator
    to the runSimulations script, and this scipt will generate all the configurations
    that will run on that server
    """
    # the base command
    base_command = "python3 /home/elundheim/simulation/SimulationScripts/Management/runSimulations.py"

    # get the kwargs used to generate the configs
    kwargs = ConfigGenerator.get_kwargs(configs)

    # Find the number of seeds and the number of configs
    nr_seeds = len(set(config.seed for config in configs))

    seeds = kwargs["seed"]
    del kwargs["seed"]
    # Prepare the remaining arguments, ensuring they are iterable
    processed_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str) or not isinstance(value, list):
            processed_kwargs[key] = [value]  # Treat single values as a list
        else:
            processed_kwargs[key] = value

    # Generate all combinations of non-seed parameters
    kwargs_combi = list(product(*processed_kwargs.values()))
    kwarg_index = 0  # Used to index kwargs_combi

    # First we want to find the number of available cores on each server
    # serverInfo[0].nrFreeCores
    serverInfo = get_all_server_info()

    commands = {}
    # If the number of servers with more free cores than nr_seeds is larger than
    # nr_max_batches, we simply put one batch on each of these servers
    # We alphabetically sort just so that the order is not random
    for si in sorted(serverInfo.values(), key=lambda x: x.sName):
        if si.theNodeCanAcceptMoreJobs is False:
            continue
        if si.sName not in commands:
            commands[si.sName] = []
        while si.nrFreeCores > nr_seeds * threads_per_seed * (
            1 + len(commands[si.sName])
        ):
            # This creates a dictionary from keys and tuple values
            combi_dict = dict(zip(kwargs.keys(), kwargs_combi[kwarg_index]))
            kwarg_index += 1
            cmd = (
                base_command
                + " "
                + " ".join(f"{key}={value}" for key, value in combi_dict.items())
            )

            full_command = f'{cmd} seed="{seeds}"'
            commands[si.sName].append(full_command)

            if kwarg_index >= len(kwargs_combi):
                for key, value in commands.items():
                    pass
                # print("All commands wiritten")
                return commands
    raise RuntimeError("Not enough cores to run simulations!")


if __name__ == "__main__":
    nrThreads = 1
    nrSeeds = 40
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

    print("Generating commands...")
    commands = generateCommands(configs)
    print("Building on all servers... ")
    build_on_all_servers()
    print("Starting jobs...")
    j = JobManager()
    for server, commands in commands.items():
        for command in commands:
            # jobId = queue_remote_job(server, command, "bigJ", nrThreads * nrSeeds)
            # print(command)
            pass
        print(f"Started {len(commands)} jobs on {get_server_short_name(server)}")
    print("Done!")
    j.showProcesses()

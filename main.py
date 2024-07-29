from Management import parameterExploring as pe
from Management.runOnCluster import uploadProject, run_remote_script, queue_remote_job
from Management.connectToCluster import Servers
from Management.multiServerJob import (
    bigJob,
    generateCommands,
    build_on_all_servers,
    JobManager,
    get_server_short_name,
)
from Management.clusterStatus import get_all_server_info, display_server_info


def parameterExploring():
    # pe.loadingSpeeds()
    pe.FIRELoading()
    # pe.CGLoading()


def runOnServer():
    server = Servers.galois
    uploadProject(server)
    # Choose script to run
    remote_script_path = (
        "/home/elundheim/simulation/SimulationScripts/Management/runSimulation.py"
    )
    run_remote_script(server, remote_script_path)


def startJobs():
    build_on_all_servers()
    return
    j = JobManager()
    j.showSlurmJobs()
    # j.cancelAllJobs()

    nrThreads = 3
    nrSeeds = 40
    configs, labels = bigJob(nrThreads, nrSeeds)
    commands = generateCommands(configs, nrThreads)
    print("Building on all servers... ")
    print("Starting jobs...")
    for server, commands in commands.items():
        for command in commands:
            # jobId = queue_remote_job(server, command, "allJs", nrThreads * nrSeeds)
            # print(command)
            pass
        print(f"Started {len(commands)} jobs on {get_server_short_name(server)}")
    print("Done!")
    # j.showProcesses()


# runOnServer()
# parameterExploring()
startJobs()

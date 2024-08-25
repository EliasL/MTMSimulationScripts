from Management import parameterExploring as pe
from Management.runOnCluster import (
    uploadProject,
    run_remote_script,
    queue_remote_job,
    run_remote_command,
)
from Management.connectToCluster import Servers
from Management.multiServerJob import (
    bigJob,
    basicJob,
    propperJob,
    generateCommands,
    build_on_all_servers,
    JobManager,
    get_server_short_name,
)
from Management.clusterStatus import get_all_server_info, display_server_info

import subprocess


def parameterExploring():
    # pe.loadingSpeeds()
    pe.FIRELoading()
    # pe.CGLoading()


def plotBigJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = bigJob(nrThreads, nrSeeds, group_by_seeds=True)
    # xLims = [0.25, 0.55]
    pe.plotLog(
        configs,
        "200x200, load:0.15-1, PBC, t3, seeds:40",
        labels=labels,
        show=True,
        # xLims=xLims,
    )


def plotPropperJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = propperJob(nrThreads, nrSeeds, group_by_seeds=True)
    # xLims = [0.25, 0.55]
    pe.plotLog(
        configs,
        "100x100, load:0.15-1, PBC, t3, seeds:40",
        labels=labels,
        show=True,
        # xLims=xLims,
    )


def lotsOThreads():
    nrThreads = 64
    nrSeeds = 3
    size = 150
    configs, labels = propperJob(nrThreads, nrSeeds, size=size, group_by_seeds=True)
    # xLims = [0.25, 0.55]
    pe.plotLog(
        configs,
        f"{size}x{size}, load:0.15-1, PBC, t{nrThreads}, seeds:{nrSeeds}",
        labels=labels,
        show=True,
        # xLims=xLims,
    )


def runOnServer():
    server = Servers.galois
    uploadProject(server)
    # Choose script to run
    remote_script_path = (
        "/home/elundheim/simulation/SimulationScripts/Management/runSimulation.py"
    )
    run_remote_script(server, remote_script_path)


def startJobs():
    # j = JobManager()
    # j.findAndShowSlurmJobs()
    # j.cancelAllJobs(force=True)

    nrThreads = 3
    nrSeeds = 40
    size = 100
    print("Building on all servers... ")
    # build_on_all_servers()
    for job in [propperJob]:
        configs, labels = job(nrThreads, nrSeeds, size)
        servers_commands = generateCommands(configs, nrThreads)
        print("Starting jobs...")
        pre_command = "python3 /home/elundheim/simulation/SimulationScripts/Management/queueLocalJobs.py"
        # pre_command = "python3 main.py start_jobs"
        for server, commands in servers_commands.items():
            full_pre_command = (
                pre_command
                + " "
                + str(
                    {
                        '"commands"': str(commands).replace('"', "\u203d"),
                        '"job_name"': '"ej"',
                        '"nrThreads"': nrThreads,
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
    print("Done!")
    # j.showProcesses()


def stopJobs():
    j = JobManager()
    j.findAndShowSlurmJobs()
    # j.cancel_jobs_on_server(Servers.descartes, 80164)
    # j.cancel_jobs_on_server(Servers.descartes, 80165)
    # j.cancel_jobs_on_server(Servers.schwartz, 466525)
    j.cancelAllJobs(force=True)
    # j.showProcesses()


# 150x150 64 threads -> 23 days
# 150x150 32 threads -> 22 days
# 150x150 16 threads -> 16 days
# 150x150 8  threads -> 22 days


# runOnServer()
# parameterExploring()
# stopJobs()
startJobs()
# plotBigJob()
# plotPropperJob()

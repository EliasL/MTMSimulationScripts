from Management import parameterExploring as pe
from Management.runOnCluster import (
    uploadProject,
    run_remote_script,
    queue_remote_job,
    run_remote_command,
    build_on_all_servers,
    build_on_server,
)
from Management.runSimulations import run_many_locally
from Management.connectToCluster import Servers
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
        "200x200, load:0.15-1, PBC, seeds:40",
        labels=labels,
        # show=True,
        # xLims=xLims,
    )


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
        # show=True,
        # xLims=xLims,
    )


def threadTest():
    nrThreads = [8, 16, 32, 64]
    nrSeeds = 1
    size = 150
    build_on_server(Servers.mesopsl)
    configs, labels = propperJob(nrThreads, nrSeeds, size)
    print("Starting jobs...")
    commands = confToCommand(configs)
    pre_command = "python3 ~/simulation/SimulationScripts/Management/queueLocalJobs.py"
    # pre_command = "python3 main.py start_jobs"
    full_pre_command = (
        pre_command
        + " "
        + str(
            {
                '"commands"': str(commands).replace('"', "\u203d"),
                '"job_name"': '"ej"',
                '"nrThreads"': sum(nrThreads),
            }
        )
    )
    print(full_pre_command)
    # run_remote_command(Servers.mesopsl, full_pre_command)


def runOnServer():
    server = Servers.galois
    uploadProject(server)
    # Choose script to run
    remote_script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    run_remote_script(server, remote_script_path)


def runOnLocalMachine():
    configs, labels = propperJob(3, 1, size=150, group_by_seeds=False)
    run_many_locally(configs)


def startJobs():
    # j = JobManager()
    # j.findAndShowSlurmJobs()
    # j.cancelAllJobs(force=True)

    nrThreads = 3
    nrSeeds = 40
    print("Building on all servers... ")

    build_on_all_servers()
    for job in [propperJob1, propperJob2, propperJob3]:
        configs, labels = job()
        servers_commands = generateCommands(configs, configs[0].nrThreads)
        print("Starting jobs...")
        pre_command = (
            "python3 ~/simulation/SimulationScripts/Management/queueLocalJobs.py"
        )
        # pre_command = "python3 main.py start_jobs"
        for server, commands in servers_commands.items():
            full_pre_command = (
                pre_command
                + " "
                + str(
                    {
                        '"commands"': str(commands).replace('"', "\u203d"),
                        '"job_name"': '"ej"',
                        '"nrThreads"': configs[0].nrThreads,
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
runOnLocalMachine()
# startJobs()
# plotBigJob()
# plotPropperJob()
# threadTest()

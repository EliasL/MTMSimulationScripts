from Management import parameterExploring as pe
from Management.connectToCluster import uploadProject
from Management.runOnCluster import (
    run_remote_script,
    queue_remote_job,
    run_remote_command,
    build_on_all_servers,
    build_on_server,
)
from runSimulations import run_many_locally, run_locally
from Management.connectToCluster import Servers
from Management.multiServerJob import (
    bigJob,
    smallJob,
    confToCommand,
    basicJob,
    allPlasticEventsJob,
    queueJobs,
    propperJob,
    propperJob1,
    propperJob2,
    propperJob3,
    distributeConfigs,
    confToCommand,
    JobManager,
    get_server_short_name,
)
from Management.clusterStatus import get_all_server_info, display_server_info
from time import sleep


def benchmark():
    configs, labels = basicJob(nrThreads=3, nrSeeds=1, size=50)
    run_locally(configs[0])

    # log
    # 1% RT: 1m 57s  ETR: 2h 34m 36s Load: 0.160600

    # Better bounding box
    #


def parameterExploring():
    # pe.loadingSpeeds()
    pe.FIRELoading()
    # pe.CGLoading()


def plotBigJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = bigJob(nrThreads, nrSeeds, group_by_seeds=True)
    # xlim = [0.25, 0.55]
    pe.plotLog(
        configs,
        "200x200, load:0.15-1, PBC, seeds:40",
        labels=labels,
        # show=True,
        # xlim=xlim,
    )


def lotsOThreads():
    nrThreads = 64
    nrSeeds = 3
    size = 150
    configs, labels = propperJob(nrThreads, nrSeeds, size=size, group_by_seeds=True)
    # xlim = [0.25, 0.55]
    pe.plotLog(
        configs,
        f"{size}x{size}, load:0.15-1, PBC, t{nrThreads}, seeds:{nrSeeds}",
        labels=labels,
        # show=True,
        # xlim=xlim,
    )


def threadTest():
    nrThreads = 1  # [1, 2, 4, 8, 16, 32, 64]
    nrSeeds = 1
    size = 100
    #    build_on_server(Servers.poincare)
    configs, labels = basicJob(nrThreads, nrSeeds, size)
    print("Starting jobs...")
    queueJobs(Servers.poincare, configs, resume=False)
    # run_many_locally(configs)


def runOnServer():
    server = Servers.poincare
    uploadProject(server)
    # Choose script to run
    remote_script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    run_remote_script(server, remote_script_path)


def runOnLocalMachine():
    # configs, labels = propperJob(3, seeds=[0], size=100, group_by_seeds=False)
    configs, labels = allPlasticEventsJob()
    dump = "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.03PBCt8initialGuessNoise1e-06LBFGSEpsg1e-08energyDropThreshold1e-10s41/dumps//dump_l0.3.mtsb"
    run_many_locally(configs)  # , dump=dump)


def startJobs():
    # j = JobManager()
    # j.findAndShowSlurmJobs()
    # j.cancelAllJobs(force=True)

    nrThreads = 3
    nrSeeds = 40
    print("Building on all servers... ")

    build_on_all_servers(uploadOnly=False)
    for job in [smallJob]:
        configs, labels = job()
        servers_confs = distributeConfigs(configs, configs[0].nrThreads)
        print("Starting jobs...")
        for server, configs in servers_confs.items():
            queueJobs(server, configs, job_name="opt")
            pass
    print("Done!")
    # sleep(1)
    # j = JobManager()
    # j.findAndShowSlurmJobs()
    # j.showProcesses()


def stopJobs():
    j = JobManager()
    j.findAndShowSlurmJobs()
    # j.cancel_jobs_on_server(Servers.descartes, 80164)
    # j.cancel_jobs_on_server(Servers.descartes, 80165)
    # j.cancel_jobs_on_server(Servers.schwartz, 466525)
    j.cancel_jobs_on_server(
        Servers.poincare,
        [
            654061,
            654070,
        ],
    )
    # j.cancelAllJobs(force=True)
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
# threadTest()
# benchmark()

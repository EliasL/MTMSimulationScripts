from Management import parameterExploring as pe
from Management.connectToCluster import uploadProject
from Management.runOnCluster import build_on_all_servers
from runSimulations import run_many_locally, run_locally
from Management.connectToCluster import Servers
from Management.multiServerJob import distributeConfigs, JobManager, queueJobs
from Management.dataManager import DataManager
from Management.jobs import (
    cyclicLoading,
    fixedBoundaries,
    backwards,
    largeAvalanche,
    avalanches,
    bigJob,
    smallJob,
    basicJob,
    allPlasticEventsJob,
    propperJob,
    propperJob1,
    propperJob2,
    propperJob3,
    findMinimizationCriteriaJobs,
    compareWithOldStoppingCriteria,
    showMinimizationCriteriaJobs,
)


def benchmark():
    configs, labels = basicJob(nrThreads=3, nrSeeds=1, size=50)
    run_locally(configs[0])

    # log (nov. 2024)
    # 1% RT: 1m 57s  ETR: 2h 34m 36s Load: 0.160600

    # Lots of changes (05.02.25) (still good)
    # 1% RT: 1m 53s  ETR: 2h 30m 21s Load: 0.160470

    # Ghost nodes (24.02.25) (Running together with another system using 6 threads)


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
    # run_many_locally(configs,taskNames=labels)


def runOnServer():
    server = Servers.poincare
    uploadProject(server, verbose=True)  # , setup=True)
    # Choose script to run
    # remote_script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    # run_remote_script(server, remote_script_path)

    configs, labels = allPlasticEventsJob()
    configs, labels = backwards(nrThreads=20, seeds=[1])
    queueJobs(server, configs, job_name="bkw")


def runOnLocalMachine():
    # configs, labels = propperJob(3, seeds=[0], size=100, group_by_seeds=False)
    configs, labels = allPlasticEventsJob()
    # configs, labels = basicJob(20, 1, size=100)
    # dump = "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/dumps/dump_l0.89.mtsb"
    configs, labels, dump = largeAvalanche(nrThreads=20)
    configs, labels, dump = avalanches(nrThreads=20, size=100)
    # 12 threads:
    # [LBFGS] 1% RT: 1h 31m 38s       ETR: 3d 23h 37m 19s     Load: 0.163360
    configs, labels = fixedBoundaries(nrThreads=6, fixed=False, L=200)
    configs, labels = showMinimizationCriteriaJobs(nrSeeds=1)

    # configs, labels = backwards(nrThreads=20)
    # configs, labels = cyclicLoading(nrThreads=20)
    # run_locally(configs[0], dump=dump)
    run_many_locally(configs, taskNames=labels, resume=False)  # , dump=dump)


def startJobs():
    nrThreads = 3
    nrSeeds = 40
    print("Building on all servers... ")

    build_on_all_servers()
    for job in [findMinimizationCriteriaJobs, compareWithOldStoppingCriteria]:
        configs, labels = job()
        print("Distributing jobs and searching for already exsisting folders...")
        servers_confs = distributeConfigs(
            configs, configs[0].nrThreads, allowWaiting=True
        )
        for server, configs in servers_confs.items():
            if configs:
                queueJobs(server, configs, job_name="opt", stopExsistingJobs=False)
            pass


def stopJobs():
    j = JobManager()
    j.findAndShowSlurmJobs()
    # j.cancel_jobs_on_server(Servers.descartes, 80164)
    # j.cancel_jobs_on_server(Servers.descartes, 80165)
    # j.cancel_jobs_on_server(Servers.schwartz, 466525)
    # j.cancel_jobs_on_server(
    #     Servers.poincare,
    #     [
    #         654061,
    #         654070,
    #     ],
    # )
    j.cancelAllJobs(force=True)
    # j.showProcesses()


def cleanData():
    dm = DataManager()
    dm.findData()
    dm.clean_projects_on_servers()
    configs, labels = findMinimizationCriteriaJobs()
    dm.delete_data_from_configs(configs, dryRun=False)
    configs, labels = compareWithOldStoppingCriteria()
    dm.delete_data_from_configs(configs, dryRun=False)


# 150x150 64 threads -> 23 days
# 150x150 32 threads -> 22 days
# 150x150 16 threads -> 16 days
# 150x150 8  threads -> 22 days


# runOnServer()
# parameterExploring()
# runOnLocalMachine()

# stopJobs()
# cleanData()
# startJobs()

# plotBigJob()
# threadTest()
benchmark()

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
    doubleDislocationTest,
    singleDislocationTest,
    longJob,
    remeshTest,
)


def benchmark():
    configs, labels = basicJob(nrThreads=3, nrSeeds=1, size=50)
    run_locally(configs[0], resume=False)

    """
      - Config File: /Users/eliaslundheim/work/PhD/MTS2D/build-release/simpleShear,s50x50l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-08s0.conf
        - Data Path: /Volumes/data/MTS2D_output/
        Name: simpleShear,s50x50l0.15,1e-05,1.0PBCt3LBFGSEpsg1e-08s0
        Rows, Cols: 50, 50
        Boundary Conditions: PBC
        Scenario: simpleShear
        Number of Threads: 3
        Seed: 0
        Quenched disorder standard deviation: 0
        Initial guess noise: 0.05
        Loading Settings:
        Start Load: 0.15
        Load Increment: 1e-05
        Max Load: 1
        Minimizer: LBFGS
        LBFGS Settings:
            Number of Corrections: 10
            Scale: 1
            EpsR: 1e-20
            EpsG: 1e-08
            EpsF: 0
            EpsX: 0
        Max LBFGS Iterations: 0
        Plasticity event threshold: 0.05
        Energy drop threshold: 0.0001
        Show progress: 1
        Log during minimization: 0

    Load_step,Load,Avg_energy,Avg_energy_change,Max_energy,Max_force,Avg_RSS,Nr_plastic_deformations,Max_plastic_deformation,Max_positive_plastic_jump,Max_negative_plastic_jump,Nr_LBFGS_iterations,Nr_LBFGS_func_evals,LBFGS_Term_reason,Nr_CG_iterations,Nr_CG_iterations,CG_Term_reason,Nr_FIRE_iterations,Nr_FIRE_func_evals,FIRE_Term_reason,Run_time,Minimization_time,Write_time,Est_time_remaining,maxX,minX,maxY,minY
    1,0.15,0.0029348038974,0,0.116149747,7.5951451434e-07,-0.027403998156,777,2,2,0,1961,4176,8,0,0,0,0,0,0,1.005s,1.004s,0.000s,0.000s,-inf,inf,-inf,inf
    2,0.15001,0.002934666803,-1.3709446572e-07,0.11616923863,3.0124384019e-07,-0.027433627681,0,2,2,0,477,1117,1,0,0,0,0,0,0,1.289s,1.266s,0.021s,6h 42m 19s,-inf,inf,-inf,inf

    """

    # log (nov. 2024)
    # 1% RT: 1m 57s  ETR: 2h 34m 36s Load: 0.160600

    # Lots of changes (05.02.25) (still good)
    # 1% RT: 1m 53s  ETR: 2h 30m 21s Load: 0.160470

    # Ghost nodes (27.02.25)
    # 1% RT: 2m 2s   ETR: 2h 28m 44s Load: 0.160880

    # Without charger (28.02.25)
    # 0% RT: 2m 1s   ETR: 3h 23m 5s  Load: 0.158330

    # Remeshing! (05.03.25)
    # 3% RT: 1m 52s	ETR: 1h 50s	Load: 0.175500

    # Working remeshing (26.03.25) (with another simulation running)
    #  1% RT: 1m 51s  ETR: 2h 9m 59s  Load: 0.161220
    # Alone (27.03.25 still has some force problems)
    #  3% RT: 1m 56s  ETR: 1h 3m 37s  Load: 0.175500

    # Really working remeshing (17.04.25)
    # Still using acos, room for even faster
    # 2% RT: 2m 3s   ETR: 1h 3m 53s  Load: 0.174050

    # remesh-locking (24.04.25)
    # 2% RT: 1m 58s  ETR: 52m 34s    Load: 0.173380


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
    server = Servers.pascal
    uploadProject(server, verbose=True)  # , setup=True)
    # Choose script to run
    # remote_script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    # run_remote_script(server, remote_script_path)

    configs, labels = allPlasticEventsJob()
    configs, labels = backwards(nrThreads=20, seeds=[1])
    configs, labels = basicJob(6, 1)
    queueJobs(server, configs, job_name="bkw")


def runOnLocalMachine():
    # configs, labels = propperJob(3, seeds=[0], size=100, group_by_seeds=False)
    # configs, labels = allPlasticEventsJob()
    dump = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/dumps/dump_l1.0.xml.gz"
    configs, labels = basicJob(6, 1, size=200, maxLoad=3.0)

    # configs, labels = remeshTest(diagonal="major")
    # run_many_locally(configs, taskNames=labels, resume=False)
    # configs, labels = remeshTest(diagonal="alternate")

    # configs, labels = longJob(6, 1, size=100)
    # dump = "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/dumps/dump_l0.89.mtsb"
    # configs, labels, dump = largeAvalanche(nrThreads=20)
    # configs, labels, dump = avalanches(nrThreads=20, size=100)
    # 12 threads:
    # [LBFGS] 1% RT: 1h 31m 38s       ETR: 3d 23h 37m 19s     Load: 0.163360
    # configs, labels = fixedBoundaries(nrThreads=6, fixed=True, L=101)
    # dump = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s100x100l0.38,1e-05,0.383NPBCt6epsR1e-06LBFGSEpsx1e-06s0/dumps/dump_l0.3814.xml.gz"
    # configs, labels = showMinimizationCriteriaJobs(nrSeeds=1)

    # configs, labels = singleDislocationTest(diagonal="minor", L=10)
    # configs, labels = singleDislocationTest(diagonal="minor", L=20)

    # configs, labels = backwards(nrThreads=20)
    # configs, labels = cyclicLoading(nrThreads=20)
    run_locally(configs[0], resume=True, dump=dump)
    # run_many_locally(configs, taskNames=labels, resume=False)


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


if __name__ == "__main__":
    # 150x150 64 threads -> 23 days
    # 150x150 32 threads -> 22 days
    # 150x150 16 threads -> 16 days
    # 150x150 8  threads -> 22 days

    # runOnServer()
    # parameterExploring()
    runOnLocalMachine()

    # stopJobs()
    # cleanData()
    # startJobs()

    # plotBigJob()
    # threadTest()
    # benchmark()

from pathlib import Path

from Plotting.remotePlotting import plotLog, plotLog2, plotEnergy
from Management import parameterExploring as pe
from Management.configGenerator import SimulationConfig
from Management.connectToCluster import uploadProject, get_server_short_name
from Management.runOnCluster import build_on_all_servers, build_on_server
from runSimulations import run_many_locally, run_locally
from Management.connectToCluster import Servers
from Management.multiServerJob import distributeConfigs, JobManager, queueJobs
from Management.dataManager import DataManager
from Management.simulationManager import findOutputPath
from Management.jobs import (
    loadStepJob,
    cyclicLoading,
    fixedBoundaries,
    backwards,
    stopConditionJob,
    umutTestJob,
    umutJobs,
    bigUmutJob,
    bigUmutJobWithEliasStop,
    largeAvalanche,
    avalanches,
    bigJob,
    smallJob,
    basicJob,
    debugJob,
    allPlasticEventsJob,
    triangular_edge_flip_job,
    propperJob,
    propperJob1,
    propperJob2,
    propperJobCGANDLBFGS,
    largePropperJob,
    largerPropperJobCGANDLBFGS,
    findMinimizationCriteriaJobs,
    compareWithOldStoppingCriteria,
    showMinimizationCriteriaJobs,
    doubleDislocationTest,
    singleDislocationTest,
    longJob,
    size_scaling_job,
    reconnectionJob,
    remeshTest,
    initalInstability,
    reconnectionTest,
    reversibilityJob,
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

    # Sylvain remesh (12.02.26) (with another simulation running)
    # 0% RT: 2m 1s   ETR: 4h 4m 52s  Load: 0.156910

    # noAlias (16.02.26) (with another simulation running)
    # 1% RT: 1m 53s  ETR: 3h 5m 24s  Load: 0.158500

    # Merged energy and stress (16.02.26) (with another simulation running)
    # 0% RT: 2m 2s   ETR: 4h 23m 25s Load: 0.156440

    # Unmerged again (16.02.26) (with another simulation running)
    # 1% RT: 1m 58s  ETR: 3h 14m 21s Load: 0.158500

    # New functions (slight algebraic alteration) (16.02.26) (with another simulation running)
    # 1% RT: 2m 5s   ETR: 2h 48m 18s Load: 0.160390

    # Elastic reduction (20.02.26) (with another simulation running)
    # 0% RT: 2m 2s   ETR: 4h 32m 57s Load: 0.156210
    # 0% RT: 20s     ETR: 2h 36m 39s Load: 0.151810
    # 1% RT: 2m 3s   ETR: 3h 22m 42s Load: 0.158500
    # Too much variability.


def reconnectingBenchmark():
    configs, labels = basicJob(nrThreads=3, nrSeeds=1, size=50, reconnection="edgeFlip")
    run_locally(configs[0], resume=False)
    """
        - Config File: /Users/eliaslundheim/work/PhD/MTS2D/build-release/simpleShear,s50x50l0.15,1e-05,1.0PBCReCONt3LBFGSEpsx1e-06s0.conf
        - Data Path: /Volumes/data/MTS2D_output/
        Name: simpleShear,s50x50l0.15,1e-05,1.0PBCReCONt3LBFGSEpsx1e-06s0
        Rows, Cols: 50, 50
        Boundary Conditions: PBC
        Reconnection enabled: True
        Scenario: simpleShear
        Number of Threads: 3
        Seed: 0
        Quenched disorder standard deviation: 0
        Initial guess noise: 0.05
        Mesh diagonal: major
        Loading Settings:
        Start Load: 0.15
        Load Increment: 1e-05
        Max Load: 1
        Minimizer: LBFGS
        LBFGS Settings:
            Number of Corrections: 3
            Scale: 1
            EpsR: 1e-20
            EpsG: 1e-15
            EpsF: 0
            EpsX: 1e-06
            Max LBFGS Iterations: 0
        Plasticity event threshold: 0.05
        Energy drop threshold: 0.0001
        Show progress: 1
        Log during minimization: 0
    Load_step,Load,Avg_energy,Avg_energy_change,Max_energy,Max_force,Avg_RSS,Nr_plastic_deformations,Max_plastic_deformation,Max_positive_plastic_jump,Max_negative_plastic_jump,Nr_LBFGS_iterations,Nr_LBFGS_func_evals,LBFGS_Term_reason,Nr_CG_iterations,Nr_CG_iterations,CG_Term_reason,Nr_FIRE_iterations,Nr_FIRE_func_evals,FIRE_Term_reason,Run_time,Minimization_time,Write_time,Est_time_remaining,maxX,minX,maxY,minY
    1,0.15,0.0029348038974,0,0.116149747,7.5951451434e-07,-0.027403998156,777,2,2,0,1961,4176,8,0,0,0,0,0,0,1.005s,1.004s,0.000s,0.000s,-inf,inf,-inf,inf
    2,0.15001,0.002934666803,-1.3709446572e-07,0.11616923863,3.0124384019e-07,-0.027433627681,0,2,2,0,477,1117,1,0,0,0,0,0,0,1.289s,1.266s,0.021s,6h 42m 19s,-inf,inf,-inf,inf

    """
    # New reconnecting (15.09.25)
    # 2% RT: 2m 5s   ETR: 1h 9m 50s  Load: 0.174020


def parameterExploring():
    # pe.loadingSpeeds()
    pe.FIRELoading()
    # pe.CGLoading()


def plotBigJob():
    nrThreads = 3
    nrSeeds = 40
    configs, labels = bigJob(nrThreads, nrSeeds, group_by_seeds=True)
    # Energy

    # Powerlaw
    # xlim = [0.25, 0.55]
    strainLim = [0.15, 0.4]
    plotLog2(
        configs,
        labels=labels,
        strainLim=strainLim,
        # show=True,
        # debug=True,
    )
    strainLim = [0.5, 1.0]
    # strainLim = [0.6, 1.0]
    # strainLim = [0.7, 1.0]
    plotLog2(
        configs,
        labels=labels,
        strainLim=strainLim,
        # show=True,
        # debug=True,
    )


def resumeWithLogDuringMin(configPath, dump, newOutput=True):
    conf = SimulationConfig()
    conf.parse(configPath)
    conf.logDuringMinimization = 1
    if newOutput:
        conf.name = conf.generate_name(False)
    run_locally(conf, dump=dump, newOutput=newOutput)


def resumeSim(dumpPath, configPath=None, newOutput=False, **kwargs):
    dump_path = Path(dumpPath)
    output_path = None
    new_output_flag = False
    if isinstance(newOutput, (str, Path)):
        output_path = str(newOutput)
        new_output_flag = True
    elif newOutput:
        output_path = findOutputPath()
        new_output_flag = True

    if output_path is not None:
        kwargs = dict(kwargs)
        kwargs["outputPath"] = output_path

    conf = SimulationConfig()
    run_locally(
        conf,
        dump=str(dump_path),
        configPath=configPath,
        autoConfig=True,
        newOutput=new_output_flag,
        **kwargs,
    )


def sylvainSmallDrop():
    confPath = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.138,4e-05,1.0PBCt8initialGuessNoise0.04LBFGSEpsx1e-05s0/config.conf"
    dump = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.138,4e-05,1.0PBCt8initialGuessNoise0.04LBFGSEpsx1e-05s0/dumps/dump_l0.66.xml.gz"
    resumeWithLogDuringMin(configPath=confPath, dump=dump)


def plotPropperJob():
    nrThreads = 3
    nrSeeds = 10
    size = 200
    mini = ["LBFGS", "CG"]  # , "FIRE"]
    configs, labels = propperJob(
        nrThreads, nrSeeds, group_by_seeds=True, size=size, minimizer=mini
    )

    # Energy
    for c, lab, m in zip(configs, labels, mini):
        plotEnergy(c, lab, f"{m}-Energy", plot_average=True)

    # Powerlaw
    # xlim = [0.25, 0.55]
    strainLim = [0.15, 0.5]
    plotLog2(
        configs,
        labels=labels,
        # xmin=None,
        strainLim=strainLim,
        # xmin=1e-5,
        # show=True,
        # debug=True,
        # addFit=False,
    )
    # strainLim = [0.5, 1.0]
    # strainLim = [0.6, 1.0]
    strainLim = [0.7, 1.0]
    plotLog2(
        configs,
        xmin=1e-5,
        labels=labels,
        strainLim=strainLim,
        # show=True,
        # debug=True,
        # addFit=False,
    )


def plotSizeScaling():
    configs, labels = size_scaling_job()

    for (
        c,
        lab,
    ) in zip(configs, labels):
        plotEnergy(c, lab, f"{lab[0].split(', ')[0]}-Energy", plot_average=True)
    # Powerlaw
    # xlim = [0.25, 0.55]
    strainLim = [0.15, 0.5]
    plotLog2(
        configs,
        labels=labels,
        # xmin=None,
        strainLim=strainLim,
        # xmin=1e-5,
        # show=True,
        # debug=True,
        # addFit=False,
    )
    # strainLim = [0.5, 1.0]
    # strainLim = [0.6, 1.0]
    strainLim = [0.7, 1.0]
    plotLog2(
        configs,
        xmin=1e-5,
        labels=labels,
        strainLim=strainLim,
        # show=True,
        # debug=True,
        # addFit=False,
    )


def plotSizeJob():
    configs, labels = size_scaling_job()
    # Energy
    # for c, lab in zip(configs, labels):
    #     plotEnergy(c, lab, f"{lab[0]}-Energy", plot_average=True)
    # Powerlaw
    strainLim = [0.15, 0.5]
    plotLog2(
        configs,
        labels=labels,
        # xmin=None,
        strainLim=strainLim,
        # xmin=1e-5,
        # show=True,
        # debug=True,
        # addFit=False,
    )
    strainLim = [0.7, 1.0]
    plotLog2(
        configs,
        labels=labels,
        # xmin=None,
        strainLim=strainLim,
        # xmin=1e-5,
        # show=True,
        # debug=True,
        # addFit=False,
    )


def lotsOThreads():
    nrThreads = 64
    nrSeeds = 3
    size = 150
    configs, labels = propperJob(nrThreads, nrSeeds, size=size, group_by_seeds=True)
    # xlim = [0.25, 0.55]
    plotLog(
        configs,
        # f"{size}x{size}, load:0.15-1, PBC, t{nrThreads}, seeds:{nrSeeds}",
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
    configs, labels = stopConditionJob()
    configs, labels = basicJob(nrThreads=3, nrSeeds=1, size=50)

    stopJobs(configs)

    server = Servers.poincare
    uploadProject(server, verbose=True)  # , setup=True)
    build_on_server(server)
    # Choose script to run
    # remote_script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    # run_remote_script(server, remote_script_path)

    # configs, labels = allPlasticEventsJob()
    # configs, labels = backwards(nrThreads=20, seeds=[1])
    # configs, labels = basicJob(3, 1, size=20)
    queueJobs(server, configs, resume=False, jobCopies=1)


def runReconnectionJob(L=20):
    # configs, labels = reconnectionTest(L=L)
    configs, labels = debugJob(size=L, maxLoad=1.0)
    run_locally(configs[0], resume=False)
    # run_many_locally(configs, taskNames=labels, resume=False)


def runOnLocalMachine():
    configs, labels = propperJob(3, nrSeeds=10, size=200, group_by_seeds=False)
    # configs, labels = allPlasticEventsJob()
    dump = "/Volumes/data/MTS2D_output/simpleShear,s200x200l0.15,1e-05,3.0PBCt8epsR1e-05LBFGSEpsg1e-08s0/dumps/dump_l3.0.xml.gz"
    dump = "/Volumes/data/MTS2D_output/cyclicSimpleShear,s200x200l0.15,1e-05,1.0PBCt3epsR1e-06s0/dumps/dump_l0.28.xml.gz"
    # configs, labels = basicJob(8, 1, size=400, maxLoad=1.0)
    # configs, labels = reconnectionJob(L=300)
    # configs, labels = fixedBoundaries(1, 1, L=3)
    # configs, labels = umutTestJob()
    # configs, labels = bigUmutJob()
    # configs, labels = bigUmutJobWithEliasStop()
    # configs, labels = loadStepJob()
    # configs, labels = reversibilityJob()
    configs, labels = triangular_edge_flip_job(size=50)

    # configs, labels = doubleDislocationTest(
    #     nrThreads=1, nrSeeds=1, L=100, diagonal="minor", reconnecting=True
    # )

    # configs, labels = remeshTest(diagonal="major")
    # run_many_locally(configs, taskNames=labels, resume=False)
    # configs, labels = remeshTest(diagonal="alternate")
    # run_many_locally(configs, taskNames=labels, resume=False)
    # configs, labels = remeshTest(diagonal="minor")
    # run_many_locally(configs, taskNames=labels, resume=False)

    # configs, labels = longJob(6, 1, size=100)
    # dump = "/Volumes/data/MTS2D_output/simpleShear,s100x100l0.15,1e-05,1.0PBCt20LBFGSEpsg1e-08energyDropThreshold1e-10s0/dumps/dump_l0.89.mtsb"
    # configs, labels, dump = largeAvalanche(nrThreads=20)
    # configs, labels, dump = avalanches(nrThreads=20, size=100)
    # 12 threads:
    # [LBFGS] 1% RT: 1h 31m 38s       ETR: 3d 23h 37m 19s     Load: 0.163360
    # configs, labels = fixedBoundaries(nrThreads=6, fixed=True, L=101)
    # dump = "/Volumes/data/MTS2D_output/simpleShearFixedBoundary,s100x100l0.38,1e-05,0.383NPBCt6epsR1e-06LBFGSEpsx1e-06s0/dumps/dump_l0.3814.xml.gz"
    # configs, labels = showMinimizationCriteriaJobs(nrSeeds=1)

    # configs, labels = backwards(nrThreads=20)
    # configs, labels = cyclicLoading(nrThreads=3)
    # run_locally(configs[0], resume=True)  # , dump=dump)
    run_many_locally(configs, taskNames=labels, resume=True)


def startJobs():
    # print("Building on all servers... ")
    build_on_all_servers(onlyPrefered=ONLYPREFERED)

    # Make largeProperJob with notFIRE=True to exclude FIRE
    def notFIRE_largePropperJob():
        return largePropperJob(notFIRE=True)

    # for job in [notFIRE_largePropperJob, size_scaling_job]:
    for job in [umutJobs]:
        configs, labels = job()

        # Normalize to batches so we handle both a single list of configs
        # and a list of lists of configs uniformly.
        if configs and isinstance(configs[0], list):
            batches = zip(configs, labels)
        else:
            batches = [(configs, labels)]

        for c, l in batches:
            print("Distributing jobs and searching for already existing folders...")
            servers_confs = distributeConfigs(
                c, c[0].nrThreads, allowWaiting=True, onlyPrefered=ONLYPREFERED
            )
            for server, confs in servers_confs.items():
                print(f"Server: {get_server_short_name(server)}, jobs: {len(confs)}")
                if confs:
                    # Queue jobs (uncomment to actually submit)
                    queueJobs(server, confs, build=False, jobCopies=20)
                    pass


def stopJobs(configs):
    j = JobManager()
    j.findSlurmJobs()
    j.cancelJobs(configs, dryRun=False)
    # j.findAndShowSlurmJobs()
    # j.cancel_jobs_on_server(Servers.descartes, 80164)
    # j.cancelJobsByNameSubstring("500x500", force=True)
    # j.cancelAllJobs(force=True, on=Servers.lagrange)

    # j.cancel_jobs_on_server(Servers.schwartz, 466525)
    # j.cancel_jobs_on_server(Servers.galois, 559077)
    # j.cancel_jobs_on_server(
    #     Servers.poincare,
    #     [
    #         654061,
    #         654070,
    #     ],
    # )
    # j.cancelAllJobs(force=True)
    # j.showProcesses()


def cleanData():
    dm = DataManager()
    dm.findData()
    dm.clean_projects_on_servers(onlyPrefered=ONLYPREFERED)
    # configs, labels = largePropperJob(notFIRE=True)
    # dm.delete_data_from_configs(configs, dryRun=False)
    # configs, labels = compareWithOldStoppingCriteria()
    # dm.delete_data_from_configs(configs, dryRun=False)


if __name__ == "__main__":
    ONLYPREFERED = True
    # build_on_all_servers(onlyPrefered=ONLYPREFERED)
    # 150x150 64 threads -> 23 days
    # 150x150 32 threads -> 22 days
    # 150x150 16 threads -> 16 days
    # 150x150 8  threads -> 22 days

    runOnServer()
    # parameterExploring()
    # runReconnectionJob()
    # runOnLocalMachine()
    # sylvainSmallDrop()
    # plotSizeJob()

    # stopJobs()
    # cleanData()
    # startJobs()

    # plotPropperJob()
    # plotSizeScaling()
    # plotBigJob()
    # stopConditionJob()
    # threadTest()
    # benchmark()
    # reconnectingBenchmark()
    # resumeSim(
    #     "/Users/eliaslundheim/work/PhD/remoteData/data/simpleShear,s400x400l0.138,2e-05,1.0PBCt8LBFGSEpsx1e-06s0/dumps/dump_l0.16.xml.gz",
    #     newOutput=True,
    # )

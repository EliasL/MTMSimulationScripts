if __name__ == "__main__":
    from simulationManager import SimulationManager
    from configGenerator import SimulationConfig

    config = SimulationConfig(
        rows=100,
        cols=100,
        startLoad=0.15,
        nrThreads=4,
        loadIncrement=1e-5,
        maxLoad=1.0,
        # LBFGSEpsg=9e-5,
        LBFGSEpsx=1e-6,
        # eps=1e-4,
        minimizer="LBFGS",
        scenario="cyclicSimpleShear",
    )
    config = SimulationConfig(
        rows=20,
        cols=20,
        startLoad=0.15,
        nrThreads=4,
        loadIncrement=1e-5,
        maxLoad=1.0,
        LBFGSEpsg=1e-6,
        minimizer="LBFGS",
        scenario="simpleShear",
    )
    FireConfig = SimulationConfig(
        rows=40,
        cols=40,
        startLoad=0.15,
        nrThreads=4,
        loadIncrement=1e-5,
        maxLoad=1.0,
        # LBFGSEpsg=9e-5,
        # LBFGSEpsx=1e-4,
        eps=1e-5,
        # epsRel=1e-16,
        # minimizer="LBFGS",
        scenario="simpleShear",
    )

    resume = True
    manager = SimulationManager(config, useProfiling=False, overwriteData=not resume)
    manager.runSimulation(resumeIfPossible=resume)
    # manager.resumeSimulation()
    # manager.resumeSimulation(
    #     dumpFile="/Volumes/data/MTS2D_output/simpleShear,s40x40l0.15,0.0001,1PBCt4eps0.0001s0/dumps//Dump_l0.447600_11.33~03.06.2024.mtsb",
    #     overwriteSettings=True,
    # )
    manager.plot()

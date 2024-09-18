if __name__ == "__main__":
    from simulationManager import SimulationManager
    from configGenerator import SimulationConfig

    config = SimulationConfig(
        rows=16,
        cols=16,
        startLoad=0.0,
        nrThreads=4,
        loadIncrement=1e-5,
        maxLoad=1.0,
        LBFGSEpsg=1e-10,
        QDSD=0.0,
        usingPBC=0,
        minimizer="LBFGS",
        scenario="simpleShearFixedBoundary",
    )

    resume = False
    manager = SimulationManager(
        config, useProfiling=False, debugBuild=False, overwriteData=not resume
    )
    dump = "/Users/elias/Work/PhD/Code/localData/MTS2D_output/simpleShear,s100x100l0.15,0.0001,1.0PBCt4LBFGSEpsg0.0001s0/dumps//Dump_l0.957600_17.01~24.07.2024.mtsb"
    manager.runSimulation(resumeIfPossible=resume)
    # manager.resumeSimulation()
    # manager.resumeSimulation(dumpFile=dump, overwriteSettings=True)
    # manager.plot()

if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig


    config = SimulationConfig(rows=150, cols=150, startLoad=0.15, nrThreads=4,
                            loadIncrement=1E-5, maxLoad=1,
                            epsx=1e-6, epsg=0, epsf=0,
                            #epsx=0, epsg=1e-2, epsf=1e-3,
                            minimizer="LBFGS",
                            #scenario="createDumpBeforeEnergyDrop")
                            scenario="simpleShear")
                            #scenario="cyclicSimpleShearPeriodicBoundary")


    manager = SimulationManager(config, useProfiling=False)
    #manager.runSimulation()
    manager.resumeSimulation(0)
    manager.plot()

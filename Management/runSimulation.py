if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig


    config = SimulationConfig(rows=60, cols=60, startLoad=0.15, nrThreads=4,
                            loadIncrement=0.00001, maxLoad=10,
                            # scenario="simpleShearPeriodicBoundary")
                            scenario="resettingSimpleShearPeriodicBoundary")


    manager = SimulationManager(config, useProfiling=False)
    manager.runSimulation()
    #manager.resumeSimulation(3)
    manager.plot()

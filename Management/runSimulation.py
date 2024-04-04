if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig


    config = SimulationConfig(rows=4, cols=4, startLoad=0, nrThreads=1,
                            loadIncrement=0.00001, maxLoad=1,
                            scenario="periodicBoundaryTest")

    manager = SimulationManager(config)
    manager.runSimulation()
    manager.plot()

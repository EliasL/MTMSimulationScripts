if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig


    config = SimulationConfig(rows=30, cols=30, startLoad=0.15, nrThreads=4,
                            loadIncrement=0.0001, maxLoad=1,
                            scenario="simpleShearPeriodicBoundary")

    manager = SimulationManager(config)
    manager.runSimulation()
    manager.plot()

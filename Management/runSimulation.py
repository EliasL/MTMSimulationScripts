if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig


    config = SimulationConfig(nx=10, ny=10, startLoad=0.15, nrThreads=1,
                            loadIncrement=0.001, maxLoad=1)

    manager = SimulationManager(config)
    manager.runSimulation()
    manager.plot()

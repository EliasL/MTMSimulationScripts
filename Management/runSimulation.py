from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig


config = SimulationConfig(nx=30, ny=30, startLoad=0.15, nrThreads=1,
                          loadIncrement=0.0001, maxLoad=0.2)

manager = SimulationManager(config, useProfiling=True)
manager.runSimulation()
#manager.plot()

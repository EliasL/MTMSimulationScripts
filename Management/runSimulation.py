from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig


config = SimulationConfig(nx=15, ny=15, startLoad=0, 
                          loadIncrement=0.05, maxLoad=2)

manager = SimulationManager(config)
manager.runSimulation()
manager.plot()

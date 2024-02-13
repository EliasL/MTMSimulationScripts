from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
from datetime import time

def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(False)
        manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


seeds = range(0,10)
configs = ConfigGenerator.generate_over_seeds(seeds, nx=100, ny=100, startLoad=0.15, 
                          loadIncrement=0.0001, maxLoad=1, nrThreads=4) 

#Build and test (Fail early)
manager = SimulationManager(SimulationConfig())
try:
    manager.runSimulation()
except Exception as e:     
    Warning(e)
    manager.clean()
    try:
        manager.runSimulation()
    except Exception as e:
        raise(Exception(e))
    

with Pool(processes=len(seeds)) as pool: 
     results = pool.map(task, configs)
from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
from datetime import time

def task(config):
    try:
        manager = SimulationManager(config)
        #time = manager.runSimulation(False)
        manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


if __name__ == '__main__':
    seeds = range(0,10)
    configs = ConfigGenerator.generate_over_seeds(seeds, rows=300, cols=300, startLoad=0.15, 
                            loadIncrement=0.00001, maxLoad=1, nrThreads=2) 
    
    configs = [
        SimulationConfig(rows=150, cols=150, startLoad=0.15, nrThreads=4,
                            loadIncrement=0.00001, maxLoad=1,
                            scenario="simpleShearPeriodicBoundary"),
        SimulationConfig(rows=150, cols=150, startLoad=0.15, nrThreads=4,
                            loadIncrement=0.00001, maxLoad=1,
                            scenario="simpleShearFixedBoundary")
    ]

    task(configs[1])
    
    # #Build and test (Fail early)
    # manager = SimulationManager(SimulationConfig())
    # try:
    #     manager.runSimulation()
    # except Exception as e:     
    #     Warning(e)
    #     manager.clean()
    #     try:
    #         manager.runSimulation()
    #     except Exception as e:
    #         raise(Exception(e))
        

    # with Pool(processes=len(seeds)) as pool: 
    #     results = pool.map(task, configs)
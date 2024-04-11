from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
from datetime import time

def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(False)
        #manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


if __name__ == '__main__':
    seeds = range(0,10)
    configs = ConfigGenerator.generate_over_seeds(seeds, rows=100, cols=100, startLoad=0.0, 
                            loadIncrement=0.00001, maxLoad=0.00001, nrThreads=1,
                            # strange bug with large noise leading to nodes going to 0,0
                            plasticityEventThreshold=-1, noise=0.2) 
    
    
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
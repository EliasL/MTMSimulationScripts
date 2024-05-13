from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
from datetime import time
import numpy as np
import sys
from pathlib import Path

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))
# Now we can import from Management
from remotePlotting import get_csv_files

from makePlots import makeEnergyPlot

# Define dumpPath as a global variable
dumpPath = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1PBCt4EpsG0.01EpsF0.001s0/dumps/Dump_l0.650400_19.03~10.05.2024.mtsb"        

def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation()
        # time = manager.resumeSimulation(dumpFile=dumpPath,
        #                                 overwriteSettings=True,
        #                                 build=False)
        #manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


if __name__ == '__main__':
    configs, labels = ConfigGenerator.generate(rows=150, cols=150, startLoad=0.15, nrThreads=1,
                            loadIncrement=[1e-5,2e-5,1e-4,2e-4], maxLoad=0.8, 
                            epsx=[0], epsg=[0,1e-2,1e-3, 1e-4], epsf=[0,1e-4,1e-5,1e-6],
                            scenario="simpleShear")

    # Build and test (Fail early)
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


    # with Pool(processes=len(configs)) as pool: 
    #     results = pool.map(task, configs)

    paths = get_csv_files(configs)
    makeEnergyPlot(paths, "ParamExploration.pdf", labels=labels, show=True)    
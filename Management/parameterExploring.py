from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
from datetime import time
import numpy as np

# Define dumpPath as a global variable
dumpPath = "/Volumes/data/MTS2D_output/simpleShear,s150x150l0.15,1e-05,1PBCt4EpsG0.01EpsF0.001s0/dumps/Dump_l0.650400_19.03~10.05.2024.mtsb"        

def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(build=False)
        # time = manager.resumeSimulation(dumpFile=dumpPath,
        #                                 overwriteSettings=True,
        #                                 build=False)
        #manager.plot()
    except Exception as e:
        print(e)
        return f"Error: {e}"
    return time

def assignColors(configs, keyValueColors, defaultColor='black'):
    color_for_config=[defaultColor]*len(configs)

    for conf_i, config in enumerate(configs):
        for key, value, color in keyValueColors:
            if getattr(config, key) == value:
                color_for_config[conf_i] = color
    return color_for_config

def runSims():
    # Build and test (Fail early)
    manager = SimulationManager(SimulationConfig(rows=3, cols=3, loadIncrement=0.1))
    try:
        manager.runSimulation()
    except Exception as e:     
        Warning(e)
        manager.clean()
        try:
            manager.runSimulation()
        except Exception as e:
            raise(Exception(e))


    with Pool(processes=len(configs)) as pool: 
        results = pool.map(task, configs)

def plotOldStuff():
    from OldConfigGenerator import ConfigGenerator as OldConf
    configs, labels = OldConf.generate(rows=150, cols=150, startLoad=0.15, nrThreads=1,
                        loadIncrement=[1e-5,2e-5,1e-4,2e-4], maxLoad=0.8, 
                        epsx=[0], epsg=[0,1e-2,1e-3, 1e-4], epsf=[0,1e-4,1e-5,1e-6],
                        scenario="simpleShear") 
    c = assignColors(configs, [
        ['loadIncrement', 1e-5, 'black'],
        ['loadIncrement', 2e-5, 'red'],
        ['loadIncrement', 1e-4, 'blue'],
        ['loadIncrement', 2e-4, 'orange'],
                         ])

    import sys
    from pathlib import Path

    # Add Management to sys.path (used to import files)
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))
    # Now we can import from Management
    from remotePlotting import get_csv_files

    from makePlots import makeEnergyPlot, makePowerLawPlot, makeItterationsPlot
    paths = get_csv_files(configs)
    print("Plotting...")
    makeEnergyPlot(paths, "ParamExploration.pdf", colors=c, labels=labels, show=True, legend=False) 
    makePowerLawPlot(paths, "ParamExplorationPowerLaw.pdf", colors=c, labels=labels, show=True)    




if __name__ == '__main__':
    #plotOldStuff()

    c=None
    configs=[]
    labels=[]
    # configs, labels = ConfigGenerator.generate(rows=16, cols=16, startLoad=0, nrThreads=1,
    #                         loadIncrement=1E-5, maxLoad=1, alphaStart=[0.01, 0.1, 0.3],
    #                         LBFGSEpsg=[9e-4,9e-5,9e-6], eps=[1e-3,1e-4,1e-5],
    #                         scenario="simpleShear")


    # configs, labels = ConfigGenerator.generate(rows=16, cols=16, startLoad=0, nrThreads=1,
    #                         loadIncrement=1E-5, maxLoad=1, alphaStart=[0.1],
    #                         LBFGSEpsg=[9e-9, 9e-10, 9e-11], eps=[1e-8,1e-9,1e-10], LBFGSEpsx=[1e-6, 1e-5, 2e-6],
    #                         scenario="simpleShear")

    extra_config1 = SimulationConfig(rows=16, cols=16, startLoad=0, nrThreads=1,
                            loadIncrement=1E-5, maxLoad=1, alphaStart=0.1,
                            LBFGSEpsg=1e-12,eps=1e-2,
                            scenario="simpleShear")

    extra_config2 = SimulationConfig(rows=16, cols=16, startLoad=0, nrThreads=1,
                            loadIncrement=1E-5, maxLoad=1, alphaStart=0.1,
                            LBFGSEpsx=1e-6,eps=1e-2,
                            scenario="simpleShear")

    extra_config3 = SimulationConfig(rows=16, cols=16, startLoad=0, nrThreads=1,
                            loadIncrement=1E-5, maxLoad=1, alphaStart=0.2,
                            LBFGSEpsx=1e-6,minimizer="LBFGS",
                            scenario="simpleShear")

    # extra_config2 = SimulationConfig(rows=16, cols=16, startLoad=0, nrThreads=1,
    #                         loadIncrement=1E-5, maxLoad=1,
    #                         LBFGSEpsg=9e-5,
    #                         minimizer="LBFGS",
    #                         scenario="simpleShear")

    configs += [extra_config2, extra_config3]
    labels += ["UmutParamWithFire", "UmutParam"]

    c = assignColors(configs, [
        ['minimizer', 'LBFGS', 'red'],
        ['minimizer', 'FIRE', 'blue'],
                         ])


    runSims()

    import sys
    from pathlib import Path

    # Add Management to sys.path (used to import files)
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))
    # Now we can import from Management
    from remotePlotting import get_csv_files

    from makePlots import makeEnergyPlot, makePowerLawPlot, makeItterationsPlot, makeTimePlot
    paths = get_csv_files(configs)
    print("Plotting...")
    makeEnergyPlot(paths, "ParamExploration.pdf", colors=c, labels=labels, show=True, legend=True)
    makeTimePlot(paths, "Run time.pdf", colors=c, labels=labels, show=True, legend=True)    
    #makeItterationsPlot(paths, "ParamExploration.pdf", colors=c, labels=labels, show=True)
    #makePowerLawPlot(paths, "ParamExplorationPowerLaw.pdf", colors=c, labels=labels, show=True)    
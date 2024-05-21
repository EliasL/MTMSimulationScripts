if __name__ == "__main__":

    from simulationManager import SimulationManager
    from configGenerator import ConfigGenerator, SimulationConfig



    config = SimulationConfig(rows=60, cols=60, startLoad=0.15, nrThreads=1,
                            loadIncrement=1E-5, maxLoad=1,
                            #LBFGSEpsg=9e-5,
                            LBFGSEpsx=1e-6,
                            #eps=1e-4, 
                            minimizer="LBFGS",
                            scenario="simpleShear")
                            
    # config = SimulationConfig(rows=100, cols=100, startLoad=0.0, nrThreads=4,
    #                         loadIncrement=1E-5, maxLoad=0.1373,
    #                         eps=9e-3,
    #                         minimizer="LBFGS",
    #                         LBFGSEpsg=1e-3,
    #                         #LBFGSEpsx=1e-6,
    #                         #scenario = "simpleShear")
    #                         scenario = "simpleShearWithNoise")
                            

    manager = SimulationManager(config, useProfiling=False)
    manager.runSimulation()
    #manager.resumeSimulation(0)
    #manager.resumeSimulation(dumpFile="/Volumes/data/KeepSafe/FireGetsStuckSimpleShear,s100x100l0.15,1e-05,1PBCt4LBFGSEpsg9e-05eps0.0001s0_Dump_l0.315700_08.08~21.05.2024.mtsb",
                             #overwriteSettings=True)
    #manager.plot()

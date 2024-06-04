from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
import sys


def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(False)
        # manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv)
    (configs,) = ConfigGenerator.generate(**kwargs)

    # Build and test (Fail early)
    manager = SimulationManager(SimulationConfig())
    try:
        manager.runSimulation()
    except Exception as e:
        Warning(e)
        manager.clean()
        try:
            manager.runSimulation()
        except Exception as e:
            raise (Exception(e))

    with Pool(processes=len(configs)) as pool:
        results = pool.map(task, configs)

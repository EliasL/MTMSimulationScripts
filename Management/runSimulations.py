from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
import sys
import ast


def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(False)
        # manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


def parse_args(args):
    kwargs = {}
    for arg in args[1:]:  # Exclude the script name itself
        key, value = arg.split("=")
        kwargs[key] = ast.literal_eval(value)

    return kwargs


if __name__ == "__main__":
    kwargs = parse_args(sys.argv)
    (configs, labels) = ConfigGenerator.generate(**kwargs)

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

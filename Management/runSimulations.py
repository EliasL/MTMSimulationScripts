from simulationManager import SimulationManager
from configGenerator import ConfigGenerator, SimulationConfig
from multiprocessing import Pool
import ast
import sys


def task(config):
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(False)
        manager.plot()
    except Exception as e:
        return f"Error: {e}"
    return time


def parse_args():
    # Skip the first argument (script path)
    args = sys.argv[1:]
    kwargs = {}

    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                # Try to evaluate the value (e.g., for lists, numbers)
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If it fails, keep it as a string
                pass
            kwargs[key] = value

    return kwargs


if __name__ == "__main__":
    kwargs = parse_args()
    if len(kwargs) == 0:
        kwargs = {
            "seed": 0,
            "minimizer": ["CG"],
            "rows": 16,
            "cols": 16,
            "eps": 1e-5,
            "LBFGSEpsg": 1e-5,
            "CGEpsg": 1e-2,
            "loadIncrement": 1e-6,
        }

    (configs, labels) = ConfigGenerator.generate(**kwargs)

    # Build and test (Fail early)
    manager = SimulationManager(SimulationConfig())
    try:
        manager.runSimulation(resumeIfPossible=False, silent=True)
    except Exception as e:
        Warning(e)
        manager.clean()
        try:
            manager.runSimulation()
        except Exception as e:
            raise (Exception(e))

    with Pool(processes=len(configs)) as pool:
        results = pool.map(task, configs)
        pass

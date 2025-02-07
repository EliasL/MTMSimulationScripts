from Management.simulationManager import SimulationManager
from Management.configGenerator import ConfigGenerator, SimulationConfig
from Management.runSimulation import run_locally
import ast
import sys
import concurrent.futures
import functools


def task(config, **kwargs):
    # This is where the task is executed
    if "taskName" not in kwargs:
        kwargs["taskName"] = config.minimizer
    run_locally(config, **kwargs)


def run_many_locally(configs, taskNames=None, **kwargs):
    # Ensure taskNames is a list of correct length
    if taskNames is None:
        taskNames = [None] * len(configs)

    if len(taskNames) != len(configs):
        raise ValueError("Length of taskNames must match length of configs")

    # First build the project once so each task does not have to build it
    manager = SimulationManager(SimulationConfig())
    manager.build()
    kwargs["build"] = False

    # Use ThreadPoolExecutor to run the task on different threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Partial function with additional kwargs
        def task_with_name(config, taskName):
            if taskName:
                kwargs["taskName"] = taskName
            task(config, **kwargs)

        # Run the tasks in parallel with corresponding names
        executor.map(task_with_name, configs, taskNames)


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

    confKwargs, runKwargs = ConfigGenerator.splitKwargs(kwargs)

    (configs, labels) = ConfigGenerator.generate(**confKwargs)
    run_many_locally(configs, taskNames=labels, **runKwargs)

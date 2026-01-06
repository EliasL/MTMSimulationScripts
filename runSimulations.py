from Management.simulationManager import SimulationManager
from Management.configGenerator import ConfigGenerator, SimulationConfig
from Management.runSimulation import run_locally, parse_args

import concurrent.futures


def task(config, **kwargs):
    # This is where the task is executed
    if "taskName" not in kwargs:
        kwargs["taskName"] = config.minimizer
    run_locally(config, **kwargs)


def run_many_locally(configs, taskNames=None, build=False, **kwargs):
    # Ensure taskNames is a list of correct length
    if taskNames is None:
        taskNames = [None] * len(configs)

    if len(taskNames) != len(configs):
        raise ValueError("Length of taskNames must match length of configs")

    # First build the project once so each task does not have to build it
    if build:
        manager = SimulationManager(SimulationConfig())
        manager.build()
    # This makes sure that the run_locally function tells the manager to not build
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

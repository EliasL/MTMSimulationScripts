from Management.simulationManager import SimulationManager
from Management.configGenerator import ConfigGenerator, SimulationConfig
from Management.runSimulation import run_locally
import ast
import sys
import concurrent.futures
import functools


# Custom class to prepend thread names to output
class ThreadOutputWrapper:
    def __init__(self, prefix, original_stdout):
        self.prefix = prefix
        self.original_stdout = original_stdout

    def write(self, message):
        # Check if the message is not an empty string or a newline
        if message.strip():
            # Add thread prefix and print to original stdout
            self.original_stdout.write(f"[{self.prefix}] {message}")
        else:
            self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()


def task(config, **kwargs):
    # This is where the task is executed
    run_locally(config, taskName=config.minimizer, **kwargs)


def run_many_locally(configs, **kwargs):
    # First build the project once so the task does not have to build it
    manager = SimulationManager(SimulationConfig())
    manager.build()

    # Use ThreadPoolExecutor to run the task on different threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use functools.partial to pre-bind the dump argument to the task function
        task_with_dump = functools.partial(task, build=False, **kwargs)

        # Map the configs to threads and run the work
        # Pass only configs to the mapped function, dump is already bound
        executor.map(task_with_dump, configs)


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
    run_many_locally(configs, **runKwargs)

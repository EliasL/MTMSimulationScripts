from .simulationManager import SimulationManager
from .configGenerator import ConfigGenerator, SimulationConfig
from .runSimulation import run_locally
import ast
import sys
import concurrent.futures


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


def task(config):
    run_locally(config, build=False, taskName=config.minimizer)


# Function to launch the work on different threads
def run_many_locally(configs):
    # First build the project once so the task does not have to build it
    manager = SimulationManager(SimulationConfig())
    manager.build()
    # Use ThreadPoolExecutor to run the task on different threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the configs to threads and run the work
        executor.map(task, configs)


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
    run_many_locally(configs)

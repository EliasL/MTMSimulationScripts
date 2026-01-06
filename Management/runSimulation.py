from .simulationManager import SimulationManager
from .configGenerator import SimulationConfig, ConfigGenerator
import ast
import sys


def run_locally(
    config=SimulationConfig(),
    resume=True,
    dump=None,
    plot=False,
    build=True,
    newOutput=False,
    **kwargs,
):
    manager = SimulationManager(config, overwriteData=not resume, **kwargs)
    if dump:
        manager.resumeSimulation(
            dumpFile=dump, overwriteSettings=True, build=build, newOutput=newOutput
        )
    else:
        manager.runSimulation(resumeIfPossible=resume, build=build)
    if plot:
        manager.plot()


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
    assert len(configs) == 1
    config = configs[0]
    run_locally(config, **runKwargs)

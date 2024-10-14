from .simulationManager import SimulationManager
from .configGenerator import SimulationConfig


def run_locally(
    config=SimulationConfig(),
    resume=True,
    dump=None,
    plot=False,
    build=True,
    **kwargs,
):
    manager = SimulationManager(config, overwriteData=not resume, **kwargs)
    if dump:
        manager.resumeSimulation(dumpFile=dump, overwriteSettings=True, build=build)
    else:
        manager.runSimulation(resumeIfPossible=resume, build=build)
    if plot:
        manager.plot()


if __name__ == "__main__":
    config = SimulationConfig(
        rows=16,
        cols=16,
        startLoad=0.0,
        nrThreads=4,
        loadIncrement=1e-5,
        maxLoad=1.0,
        LBFGSEpsg=1e-10,
        QDSD=0.0,
        usingPBC=0,
        minimizer="LBFGS",
        scenario="simpleShearFixedBoundary",
    )
    run_locally(config)

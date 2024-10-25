from simulationManager import SimulationManager, findOutputPath
from configGenerator import ConfigGenerator, SimulationConfig
import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


def thread_benchmark():
    # Warmup run
    manager = SimulationManager(SimulationConfig())  # Default config values
    manager.runSimulation()

    threads = [1, 10, 20, 30, 32, 34, 40, 50, 60, 63, 64]
    threads = [1, 2, 3, 4]
    configs = ConfigGenerator.generate_over_threads(
        threads, rows=50, cols=50, startLoad=0.15, loadIncrement=0.001, maxLoad=1
    )

    # Real runs
    runtimes = []
    for config in configs:
        manager = SimulationManager(config)
        nrRuns = 1
        times = [manager.runSimulation(build=False) for _ in range(nrRuns)]
        time = sum(times) / nrRuns
        runtimes.append(time)

    outpath = findOutputPath()
    plt.plot(threads, runtimes)
    plt.title("Runtime over number of threads")
    plt.xlabel("Number of threads")
    plt.ylabel("Time (s)")
    path = outpath + f"/Thread_effect_{configs[0].generate_name(False)}.pdf"
    plt.savefig(path)
    print(f"Figure saved at {path}")


def task(args):
    config, run_id = args  # Unpack arguments
    try:
        manager = SimulationManager(config)
        time = manager.runSimulation(build=False)
        manager.plot()
    except Exception as e:
        return f"Error: {e}", run_id
    return time, run_id


def speed_tests():
    # Warmup run
    manager = SimulationManager(SimulationConfig())  # Default config values
    manager.runSimulation()

    nrCorrections = [1, 3, 5, 7, 10]
    configs = ConfigGenerator.generate_over_(
        "nrCorrections",
        nrCorrections,
        rows=100,
        cols=100,
        startLoad=0.15,
        loadIncrement=0.00001,
        maxLoad=0.7,
        threads=1,
    )

    nrTimesRun = 5
    # Prepare tasks with each config to be run 3 times
    tasks = [
        (config, f"{config.generate_name(withExtension=False)}-run{run_id}")
        for config in configs
        for run_id in range(1, nrTimesRun + 1)
    ]

    with Pool(processes=len(configs) * nrTimesRun) as pool:
        results = pool.map(task, tasks)

        # Organize results by config
        organized_results = {}
        for time, run_id in results:
            config_id = run_id.rsplit("-", 1)[0]
            if config_id not in organized_results:
                organized_results[config_id] = []
            organized_results[config_id].append(time)

        # Display results
        for config_id, times in organized_results.items():
            if all(
                isinstance(time, float) for time in times
            ):  # Ensure no errors before calculating average
                avg_time = np.mean(times)
                print(f"Config: {config_id}, Times: {times}, Average: {avg_time}")
            else:
                print(f"Config: {config_id} encountered an error during execution.")


def baseSpeed():
    # Warmup run
    manager = SimulationManager(SimulationConfig())  # Default config values
    manager.runSimulation()

    # linux
    # Without writing
    #   0:38
    # With spdlog csv and no vtu
    #   0:39
    # With spdlog csv and vtu
    #   0:42
    # With private nodes
    #   0:37

    # mac
    # With private nodes
    #   0:58
    # With 1 thread instead of 4
    #   0:22
    # No log, 1 thread
    #   0:22
    # Very nice! Logging takes up no time.

    config = SimulationConfig(
        rows=10,
        cols=10,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=0.00001,
        maxLoad=1,
        scenario="simpleShearFixedBoundary",
    )
    manager = SimulationManager(config)
    nrRuns = 1
    times = [manager.runSimulation(build=False) for _ in range(nrRuns)]
    time = sum(times) / nrRuns
    print(time)
    # manager.plot()


def resume_simulation_midway():
    config = SimulationConfig(
        rows=10, cols=10, startLoad=0.15, nrThreads=4, loadIncrement=0.00001, maxLoad=1
    )
    manager = SimulationManager(config)
    manager.runSimulation()
    manager.resumeSimulation()
    manager.plot()


if __name__ == "__main__":
    print("Starting benchmark run...")
    baseSpeed()
    # resume_simulation_midway()
    # thread_benchmark()

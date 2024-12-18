import multiprocessing
import time
import os
import platform
from collections import defaultdict


# Set environment variables to limit NumPy's BLAS to a single thread
os.environ["OMP_NUM_THREADS"] = "1"  # For OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # For Intel MKL
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For OpenBLAS
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Apple's Accelerate
os.environ["BLIS_NUM_THREADS"] = "1"  # For BLIS

import numpy as np  # Import NumPy after setting environment variables


def np_cpu_bound_task(matrix_size=1000):
    """
    Performs a matrix multiplication using NumPy.

    Args:
        matrix_size (int): Size of the square matrices to multiply.

    Returns:
        np.ndarray: Result of the matrix multiplication.
    """
    # Generate two random matrices
    for i in range(100):
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)

        # Perform matrix multiplication
        C = np.matmul(A, B)

    return C


def cpu_bound_task(n=10**8):
    """
    An arbitrary CPU-bound task.
    Performs a simple calculation that consumes CPU resources.

    Args:
        n (int): Number of iterations.

    Returns:
        int: The result of the calculation.
    """
    count = 0
    for i in range(n):
        count += i * i
    return count


def set_cpu_affinity(core_id):
    """
    Sets the CPU affinity of the current process to the specified core on Linux.
    On macOS, this function does nothing as CPU affinity is not straightforwardly supported.

    Args:
        core_id (int): The CPU core to bind to.
    """
    current_os = platform.system()

    if current_os == "Linux":
        try:
            os.sched_setaffinity(0, {core_id})
            # Optional: Verify affinity
            # actual_affinity = os.sched_getaffinity(0)
            # if core_id not in actual_affinity:
            #     print(f"Warning: Core {core_id} was not set for affinity.")
            return True
        except AttributeError:
            print("os.sched_setaffinity is not available on this system.")
        except PermissionError:
            print(
                "Permission denied when setting CPU affinity. Are you running as root?"
            )
        except Exception as e:
            print(f"An error occurred while setting CPU affinity: {e}")
    elif current_os == "Darwin":
        # CPU affinity setting is not straightforward on macOS.
        # Proceeding without setting affinity.
        print(f"[macOS] Skipping setting CPU affinity for Core {core_id}.")
    else:
        print(
            f"[{current_os}] Unsupported OS for setting CPU affinity. Proceeding without binding."
        )
    return False


def run_task(core_id, run_id, results):
    """
    Binds the current process to a specific CPU core (if supported) and runs the CPU-bound task.

    Args:
        core_id (int): The CPU core to bind to.
        run_id (int): The iteration number.
        results (dict): A shared dictionary to store results.
    """
    success = set_cpu_affinity(core_id)

    # Optional: Give some time for the OS to bind the process
    if success:
        time.sleep(0.1)

    # Measure execution time
    start_time = time.time()
    cpu_bound_task()
    # np_cpu_bound_task()
    end_time = time.time()
    elapsed = end_time - start_time

    # Store the result
    results[(core_id, run_id)] = elapsed
    print(f"Run {run_id} on Core {core_id}: {elapsed:.4f} seconds")


def main():
    NUM_CORES = 10  # Number of CPU cores to test
    NUM_RUNS = 5  # Number of runs per core

    available_cores = os.cpu_count()
    if available_cores is None:
        print("Unable to determine the number of CPU cores.")
        return

    if available_cores < NUM_CORES:
        print(
            f"Warning: Requested {NUM_CORES} cores, but only {available_cores} available."
        )
        NUM_CORES = available_cores

    print(f"Using {NUM_CORES} CPU cores for testing.")

    manager = multiprocessing.Manager()
    results = manager.dict()

    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== Run {run} ===")
        processes = []
        for core in range(NUM_CORES):
            p = multiprocessing.Process(target=run_task, args=(core, run, results))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # Analyze results
    timings = defaultdict(list)
    for (core, run), elapsed in results.items():
        timings[core].append(elapsed)

    print("\n=== Timing Results ===")
    avg_times = {}
    std_dev_within = {}
    for core in range(NUM_CORES):
        core_timings = timings.get(core, [])
        if core_timings:
            avg_time = np.mean(core_timings)
            std_dev = np.std(core_timings)
            avg_times[core] = avg_time
            std_dev_within[core] = std_dev
            print(
                f"Core {core}: Average Time = {avg_time:.4f} seconds over {len(core_timings)} runs | Std Dev = {std_dev:.6f}"
            )
        else:
            print(f"Core {core}: No data.")

    # Calculate standard deviation across the averages of all cores
    avg_time_values = list(avg_times.values())
    std_dev_across = np.std(avg_time_values)
    print(
        f"\nStandard Deviation Across All Cores' Averages: {std_dev_across:.6f} seconds"
    )

    # Identify the best and worst cores
    best_core = min(avg_times, key=avg_times.get)
    worst_core = max(avg_times, key=avg_times.get)
    best_time = avg_times[best_core]
    worst_time = avg_times[worst_core]

    # Calculate percentage difference between best and worst
    percentage_diff = ((worst_time - best_time) / best_time) * 100
    print(f"\nBest Core: Core {best_core} with Average Time = {best_time:.4f} seconds")
    print(f"Worst Core: Core {worst_core} with Average Time = {worst_time:.4f} seconds")
    print(f"Percentage Difference Between Best and Worst Core: {percentage_diff:.2f}%")

    # Rank cores by average execution time
    sorted_cores = sorted(avg_times.items(), key=lambda x: x[1])

    print("\n=== Cores Ranked by Average Execution Time ===")
    for rank, (core, avg_time) in enumerate(sorted_cores, start=1):
        print(f"{rank}. Core {core} - Avg Time: {avg_time:.4f} seconds")

    print("\nTest completed.")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------
# Below we will write a Python script that:
# 1) Runs a CPU-intensive benchmark (like a large floating-point computation).
# 2) Measures how long it takes to run that computation.
# 3) Optionally sets CPU affinity so the computation runs on a single CPU core.
# 4) Optionally tries to raise priority to reduce interference from other processes.
# ----------------------------------------------------------
# Note:
# - This script is intended for Linux, as it uses `os.sched_setaffinity`
#   which is Linux-specific. Mac does not have this by default.
# - To set CPU affinity, you will need the appropriate permissions or to run
#   as a user with that capability.
# - To raise priority, you can run this script with `sudo` and use `nice` or `chrt`.
#   For example: `sudo chrt -f 99 python3 test_cpu_perf.py --cpu 0`
#   This gives the process a FIFO scheduler with a high priority.
# - The script runs a fixed number of floating-point operations as a benchmark.
#   Adjust the workload as needed for your tests.
# ----------------------------------------------------------

import time
import argparse
import os
import math


def cpu_intensive_work(iterations):
    # ------------------------------------------------------
    # Perform a large number of floating-point operations
    # to stress the CPU and measure performance.
    # ------------------------------------------------------
    x = 0.0
    for i in range(iterations):
        x += math.sin(i) * math.cos(i)
    return x


if __name__ == "__main__":
    # ------------------------------------------------------
    # Parse arguments to allow choosing a specific CPU core
    # and number of iterations for the benchmark.
    # ------------------------------------------------------
    parser = argparse.ArgumentParser(description="CPU performance tester")
    parser.add_argument(
        "--cpu",
        type=int,
        default=None,
        help="CPU core to run on (0-indexed). If not provided, no affinity is set.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10_000_000,
        help="Number of iterations of floating-point ops.",
    )
    args = parser.parse_args()

    if args.cpu is not None:
        # --------------------------------------------------
        # Set CPU affinity to run on a single core.
        # Note: Requires Linux and appropriate permissions.
        # --------------------------------------------------
        os.sched_setaffinity(0, {args.cpu})

    # ------------------------------------------------------
    # Start timing the CPU-intensive task.
    # ------------------------------------------------------
    start_time = time.time()
    result = cpu_intensive_work(args.iterations)
    end_time = time.time()

    # ------------------------------------------------------
    # Print the results.
    # ------------------------------------------------------
    elapsed = end_time - start_time
    ops_per_sec = args.iterations / elapsed
    print(f"CPU: {args.cpu if args.cpu is not None else 'None'}")
    print(f"Iterations: {args.iterations}")
    print(f"Time (s): {elapsed:.4f}")
    print(f"Ops/s: {ops_per_sec:.2f}")
    print(f"Final result (ignore this): {result}")

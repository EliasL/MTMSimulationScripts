import subprocess
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configuration
test_script_path = "/tmp/test_cpu_perf.py"
cpus = range(128)  # or however many cores you want to test
iterations = 5_000_000
num_runs = 7
max_workers = int(max(cpus) / 2)
output_dir = "/tmp"  # Directory on the cluster to store the results


def run_single_cpu_benchmark(cpu):
    # Use `nice` to give the task the highest priority (-20)
    cmd = [
        "nice",
        "-n",
        "-20",  # Highest priority
        "taskset",
        "-c",
        str(cpu),  # Pin to a specific CPU
        "python3",
        test_script_path,
        f"--cpu={cpu}",
        f"--iterations={iterations}",
    ]
    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    cpu_num, ops_s, time_s = parse_output(output)
    return (cpu_num, ops_s, time_s)


def parse_output(output):
    cpu_num = None
    ops_s = None
    time_s = None
    for line in output.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("CPU:"):
            parts = line_stripped.split(":")
            cpu_num = int(parts[1].strip())
        elif line_stripped.startswith("Time (s):"):
            parts = line_stripped.split(":")
            time_s = float(parts[1].strip())
        elif line_stripped.startswith("Ops/s:"):
            parts = line_stripped.split(":")
            ops_s = float(parts[1].strip())
    return cpu_num, ops_s, time_s


def run_benchmark_one_round():
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_cpu_benchmark, cpu): cpu for cpu in cpus}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x[0])
    return results


def main():
    all_runs_results = []
    for i in range(num_runs):
        print(f"Starting run {i+1}/{num_runs} ...")
        run_results = run_benchmark_one_round()
        all_runs_results.append(run_results)

    # Plot results
    cpu_indices = [r[0] for r in all_runs_results[0]]
    plt.figure(figsize=(12, 6))
    for run_id, results in enumerate(all_runs_results, start=1):
        time_values = [r[2] for r in results]
        plt.plot(
            cpu_indices, time_values, marker="o", linestyle="-", label=f"Run {run_id}"
        )

    plt.title("Execution Time per Core (s) for Multiple Runs")
    plt.xlabel("CPU Core")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_file = os.path.join(output_dir, "cpu_time_per_core_multiple_runs.png")
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    main()

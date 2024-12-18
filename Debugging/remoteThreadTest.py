import subprocess
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

remote_host = "poincare"
local_script_path = "Debugging/test_cpu_perf.py"
remote_script_path = "/tmp/test_cpu_perf.py"
iterations = 1_000_000
cpus = range(12)


def upload_script():
    try:
        scp_cmd = ["scp", local_script_path, f"{remote_host}:{remote_script_path}"]
        print(f"Uploading {local_script_path} to {remote_host}:{remote_script_path}...")
        subprocess.run(
            scp_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Upload successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading script: {e.stderr.decode().strip()}")
        sys.exit(1)


def set_permissions():
    try:
        ssh_chmod_cmd = ["ssh", remote_host, "chmod", "+x", remote_script_path]
        print(f"Setting execute permissions on {remote_script_path}...")
        subprocess.run(
            ssh_chmod_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Permissions set successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting permissions: {e.stderr.decode().strip()}")
        sys.exit(1)


def run_benchmark():
    results = []
    for cpu in tqdm(cpus):
        print(f"Running benchmark on CPU core {cpu}...")
        try:
            cmd = [
                "ssh",
                remote_host,
                "python3",
                remote_script_path,
                f"--cpu={cpu}",
                f"--iterations={iterations}",
            ]
            # To set higher priority, prepend with 'nice' or 'sudo chrt' if necessary
            # Example with 'nice':
            # cmd = [
            #     "ssh", remote_host,
            #     "nice", "-n", "-20", "python3", remote_script_path,
            #     f"--cpu={cpu}",
            #     f"--iterations={iterations}"
            # ]
            output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, universal_newlines=True
            )
            cpu_num, ops_s, time_s = parse_output(output)
            results.append((cpu_num, ops_s, time_s))
            print(f"CPU {cpu_num}: Ops/s={ops_s}, Time={time_s}s")
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark on CPU {cpu}: {e.output}")
    return results


def parse_output(output):
    cpu_num = None
    ops_s = None
    time_s = None

    for line in output.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("CPU:"):
            parts = line_stripped.split(":")
            if len(parts) == 2:
                cpu_num = int(parts[1].strip())
        elif line_stripped.startswith("Time (s):"):
            parts = line_stripped.split(":")
            if len(parts) == 2:
                time_s = float(parts[1].strip())
        elif line_stripped.startswith("Ops/s:"):
            parts = line_stripped.split(":")
            if len(parts) == 2:
                ops_s = float(parts[1].strip())
    return cpu_num, ops_s, time_s


def plot_results(results):
    if not results:
        print("No results to plot.")
        return

    results.sort(key=lambda x: x[0])
    cpu_indices = [r[0] for r in results]
    ops_values = [r[1] for r in results]
    time_values = [r[2] for r in results]

    # Plot Ops/s
    # plt.figure(figsize=(12, 6))
    # plt.plot(cpu_indices, ops_values, marker="o", linestyle="-", color="blue")
    # plt.title("CPU Performance per Core (Ops/s)")
    # plt.xlabel("CPU Core")
    # plt.ylabel("Operations per Second")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("cpu_performance_per_core.png")
    # plt.show()

    # Plot Time
    plt.figure(figsize=(12, 6))
    plt.plot(cpu_indices, time_values, marker="o", linestyle="-", color="red")
    plt.title("Execution Time per Core (s)")
    plt.xlabel("CPU Core")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Debugging/cpu_time_per_core.png")
    plt.show()


def main():
    upload_script()
    set_permissions()
    results = run_benchmark()
    plot_results(results)


if __name__ == "__main__":
    main()

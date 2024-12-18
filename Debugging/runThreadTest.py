import subprocess


def run_command(cmd, description):
    # ----------------------------------------------------------
    # This function runs a given command (cmd) using subprocess.run
    # and prints a description of what it's doing.
    # If there's an error (CalledProcessError), it prints the full stdout/stderr
    # to help diagnose the actual issue.
    # ----------------------------------------------------------
    print(description)
    try:
        # Run the command and capture stdout/stderr
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print("Success.")
    except Exception as e:
        # Print command, return code, and both stdout and stderr
        print("Error running command:")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print("Standard output:")
        print(e.stdout)
        print("Standard error:")
        print(e.stderr)
        exit()


def main():
    remote_host = "poincare"
    local_test_script = "Debugging/test_cpu_perf.py"
    local_run_script = "Debugging/run_local_cpu_test.py"
    remote_test_script = "/tmp/test_cpu_perf.py"
    remote_run_script = "/tmp/run_local_cpu_test.py"
    remote_output_file = "/tmp/cpu_time_per_core_multiple_runs.png"
    local_output_file = "Debugging/cpu_time_per_core_multiple_runs.png"
    # Upload test_cpu_perf.py
    run_command(
        ["scp", local_test_script, f"{remote_host}:{remote_test_script}"],
        f"Uploading {local_test_script} to {remote_host}:{remote_test_script}...",
    )

    # Upload run_local_cpu_test.py
    run_command(
        ["scp", local_run_script, f"{remote_host}:{remote_run_script}"],
        f"Uploading {local_run_script} to {remote_host}:{remote_run_script}...",
    )

    # # Set permissions
    # run_command(
    #     ["ssh", remote_host, "chmod", "+x", remote_run_script, remote_test_script],
    #     f"Setting execute permissions on {remote_run_script} and {remote_test_script}...",
    # )

    # Run run_cpu_test.py on cluster
    run_command(
        ["ssh", remote_host, f"python3 {remote_run_script}"],
        f"Running {remote_run_script} on {remote_host}...",
    )

    # Download the resulting plot
    run_command(
        ["scp", f"{remote_host}:{remote_output_file}", local_output_file],
        f"Downloading the resulting plot from {remote_output_file}...",
    )
    print(f"Plot downloaded as {local_output_file}")


if __name__ == "__main__":
    main()

from Management.connectToCluster import (
    uploadProject,
    Servers,
    get_server_short_name,
    getServerUserName,
)
from Management.queueLocalJobs import get_batch_script
from fabric import Connection
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from invoke import Responder
import subprocess
import os


import os
import debugpy
from fabric import Connection
import sys


def run_remote_script_with_debug(server_hostname, script_path, silent=False):
    """
    Executes a Python script on a remote server, with optional support for debugging.
    If the local script is in debug mode, it enables a remote debug session.

    :param server_hostname: The hostname of the remote server.
    :param script_path: The path to the script on the remote server.
    :param silent: Suppress output if True.
    """
    user = getServerUserName(server_hostname)

    # Check if the current script is running in debug mode
    is_debug_mode = any(
        getattr(debugpy, "is_client_connected", lambda: False)() or "pydevd" in module
        for module in sys.modules
    )

    # Establish the SSH connection
    with Connection(host=server_hostname, user=user) as c:
        # Construct the remote command
        command = f"python3 -u {script_path}"

        if is_debug_mode:
            # Add debugpy to the remote command
            # Ensure debugpy is installed on the remote server
            command = f"python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client {script_path}"
            print("Running in debug mode: waiting for debugger to attach...")

        # Execute the command on the remote server
        result = c.run(
            command,
            hide=True,
            warn=True,
        )

        # Output and errors are captured
        output = result.stdout
        errors = result.stderr

        # Check the result
        if result.ok and not silent:
            print("Script executed successfully.")
            print("Output from the script:")
            print(output)
        else:
            print("Script execution failed:")
            print(errors)

        return output  # Optionally return output for further processing


def run_remote_script(server_hostname, script_path, silent=False):
    user = getServerUserName(server_hostname)
    # Establish the SSH connection
    with Connection(host=server_hostname, user=user) as c:
        # Execute the remote command (your Python script)
        # Set `hide=True` to suppress real-time output and capture it instead
        result = c.run(
            f"python3 -u {script_path}",
            hide=True,
            warn=True,
        )

        # Output and errors are captured
        output = result.stdout
        errors = result.stderr

        # Check the result
        if result.ok and not silent:
            print("Script executed successfully.")
            print("Output from the script:")
            print(output)
        else:
            print("Script execution failed:")
            print(errors)

        return output  # Optionally return output for further processing


output = {}
pbars = {}


class ProgressBarUpdater(Responder):
    def __init__(self, server, pbar_index, silent):
        self.server = get_server_short_name(server)
        self.pbar_index = pbar_index
        self.total = -1
        self.current_line = 0
        self.silent = silent

    def submit(self, all_lines):
        global output, pbars
        for line in all_lines.splitlines(keepends=True)[self.current_line :]:
            if not line.endswith(("\n", "\r")):
                return []
            else:
                line = line.strip()
            # Initialize or update the progress bar based on the output
            print(line)
            if self.total == -1:
                self.total = int(line)
                pbars[self.pbar_index] = tqdm(
                    desc=self.server,
                    total=self.total,
                    position=self.pbar_index,
                    disable=self.silent,
                    unit=" folders",
                )
                output[self.pbar_index] = []
                self.current_line += 1
            else:
                output[self.pbar_index].append(line)  # Store the line in the list
                pbars[self.pbar_index].update(1)
                self.current_line += 1
        return []  # Return an empty iterable


def run_remote_script_with_progress(server, script_path, pbar_index, silent):
    """Run a remote script and process its output in real time, updating a tqdm progress bar."""
    updater = ProgressBarUpdater(server, pbar_index, silent)
    user = getServerUserName(server)
    with Connection(host=server, user=user) as c:
        # Execute the script with unbuffered Python output and pseudo-terminal
        command = f"python3 -u {script_path}"
        c.run(command, watchers=[updater], hide=True)
        if pbar_index in output:
            pbars[pbar_index].close()
            return output[pbar_index]
        else:
            return []


def find_outpath_on_server(server_hostname):
    # Establish the SSH connection
    user = getServerUserName(server_hostname)
    with Connection(host=server_hostname, user=user) as c:
        # Execute the remote command (your Python script)
        result = c.run(
            "python3 -u ~/simulation/SimulationScripts/Management/simulationManager.py",
            hide=True,
            warn=True,
        )
    return result.stdout.strip()


def run_remote_command(server_hostname, command, hide=False, silent=True):
    # Establish the SSH connection
    user = getServerUserName(server_hostname)
    with Connection(host=server_hostname, user=user) as c:
        # Execute the remote command (your Python script)
        result = c.run(command, hide=hide, warn=True)

        # `hide=False` means output and errors are printed in real time
        # `warn=True` means execution won't stop on errors (similar to try/except)

        # Check the result
        if result.ok:
            if not silent:
                print(f"{server_hostname}: Command executed successfully.")
        else:
            print(f"{server_hostname}: Command execution failed: {result.stderr}")


# This function is depricated. It's slow to start jobs like this.
# It's better to run queueLocalJobs on the server
def queue_remote_job(server_hostname, command, job_name, nrThreads):
    base_path = "~/simulation/MTS2D/"
    outPath = os.path.join(base_path, "JobOutput")
    error_file = os.path.join(outPath, f"err-{job_name}.err")

    # Establish the SSH connection
    user = getServerUserName(server_hostname)
    with Connection(host=server_hostname, user=user) as c:
        # Check if the simulation directory exists
        if c.run(f"test -d {base_path}", warn=True).failed:
            raise Exception(f"The directory {base_path} does not exist.")

        # Ensure the JobOutput directory exists
        c.run(f"mkdir -p {outPath}")

        # Truncate the error file to clear old errors
        c.run(f"truncate -s 0 {error_file}")

        batch_script = get_batch_script(command, job_name, nrThreads, outPath)

        # Create the batch script on the server
        batch_script_path = outPath + job_name + ".sh"
        c.run(f'cat << "EOF" > {batch_script_path}\n{batch_script}\nEOF')

        # Submit the batch script to Slurm
        result = c.run(f"sbatch {batch_script_path}", hide=True, warn=True)

        # Check the submission result
        if result.ok:
            # print("Batch script submitted successfully.")
            print(result.stdout)  # This will include the Slurm job ID
            # Extract the Slurm job ID from the result
            try:
                job_id_line = result.stdout.strip().split()[-1]
                slurm_job_id = int(job_id_line)
                return slurm_job_id  # Return the Slurm job ID
            except ValueError as e:
                raise Exception(f"Error parsing jobID: {e}")
        else:
            print(f"Batch script submission failed: {result.stderr}")
            return None


def build_on_server(server, uploadOnly=False):
    shortName = get_server_short_name(server)
    print(f"Uploading to {shortName}...")
    uploadProject(server)

    if uploadOnly:
        print(f"Upload only to {shortName} completed.")
        return  # Exit the function after uploading

    project_path = "~/simulation/MTS2D/build-release/"
    build_command = (
        f"mkdir -p {project_path} && "
        f"cd {project_path} && "
        f"cmake -DCMAKE_BUILD_TYPE=Release .. && "
        f"make"
    )

    print(f"Building on {shortName}...")
    run_remote_command(server, build_command, hide=True)
    print(f"{shortName} is ready!")


def build_on_all_servers(uploadOnly=False):
    # This function should start build jobs on all servers, and then wait until
    # a build fails or all builds are completed
    servers = Servers.servers
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        # Future to server mapping
        future_to_server = {
            executor.submit(build_on_server, server, uploadOnly): server
            for server in servers
        }

        for future in as_completed(future_to_server):
            server = future_to_server[future]
            try:
                future.result()  # Get the result from future
            except Exception as exc:
                print(f"{server} generated an exception: {exc}")
                continue


if __name__ == "__main__":
    # This can be used to run something on the server, but don't use this
    # to run a job. Use JobManager.
    # Choose which server to run on
    server = Servers.galois
    # Upload/sync the project
    uploadProject(server)
    # Choose script to run
    script_path = "~/simulation/SimulationScripts/Management/runSimulation.py"
    # Generate sbatch script

    # Queue the script on the server
    run_remote_script(server, script_path)
    # build_on_all_servers()

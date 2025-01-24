import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from tabulate import tabulate
import time
import random
from datetime import timedelta
import re
import threading
from paramiko.ssh_exception import SSHException, NoValidConnectionsError

from concurrent.futures import ThreadPoolExecutor, as_completed
from .clusterStatus import Servers, get_server_short_name
from .connectToCluster import uploadProject, connectToCluster  # noqa: F401
from .configGenerator import SimulationConfig
from .dataManager import get_directory_size
from .runOnCluster import build_on_all_servers, build_on_server, queue_remote_job
from Plotting.settings import settings


import logging

# Suppress Paramiko logging
logging.getLogger("paramiko").setLevel(logging.CRITICAL)


def parse_duration(duration_str):
    pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?"
    matches = re.match(pattern, duration_str.strip())

    if not matches:
        return timedelta()

    days, hours, minutes, seconds = matches.groups(default="0")

    return timedelta(
        days=int(days), hours=int(hours), minutes=int(minutes), seconds=float(seconds)
    )


def calculate_percentage_completed(runtime_str, estimated_remaining_str):
    runtime = parse_duration(runtime_str)
    estimated_remaining = parse_duration(estimated_remaining_str)

    total_time = runtime + estimated_remaining
    percentage_completed = (runtime / total_time) * 100
    return percentage_completed


# Shared variables
nr_processes_found = 0
nr_jobs_found = 0
lock = threading.Lock()  # Create a lock for thread-safe operations


def update_progress(jobs=False, processes=False):
    if jobs:
        print(f"\r{nr_jobs_found} jobs found", end="")
    if processes:
        print(f"\r{nr_processes_found} processes found", end="")


class Process:
    """
    NB This does not find slurm jobs! It checks the processes running on the
    cluster and finds all instances of MTS2D running.
    """

    paris_zone = ZoneInfo("Europe/Paris")
    gmt_zone = ZoneInfo("Europe/London")

    def __init__(self, ssh, processID, server, timeRunning) -> None:
        self.ssh = ssh
        self.name = ""
        self.p_id = processID
        self.command = ""
        self.server = server
        self.timeEstimation = ""
        self.progress = ""
        self.progress_timestamp = None
        self.dataSize = 0
        self.output_path = ""
        self.configObj = None
        self.timeRunning = timeRunning

        self.getInfoFromProcess()

        with lock:
            global nr_processes_found
            nr_processes_found += 1
            update_progress(processes=True)

    def getInfoFromProcess(self):
        stdin, stdout, stderr = self.ssh.exec_command(f"ps -p {self.p_id} -o args=")
        command_line = stdout.read().decode("utf-8").strip()
        parts = command_line.split()
        self.command = command_line
        # Extracting the paths based on the -c and -o flags
        if "-c" in parts:
            c_index = parts.index("-c") + 1
            config_path = parts[c_index]
            self.get_config_file(config_path)
            self.name = os.path.splitext(os.path.basename(config_path))[0]

        if "-o" in parts:
            o_index = parts.index("-o") + 1
            self.output_path = parts[o_index]
        elif "-d" in parts:
            d_index = parts.index("-d") + 1
            # We can extract the name and output path from the dump path
            self.name = parts[d_index].split("/")[-3]
            self.output_path = "/".join(parts[d_index].split("/")[:-3])

        if self.output_path != "":
            self.get_progress()
            # self.dataSize = get_directory_size(self.ssh, self.output_path + self.name)

    def get_config_file(self, config_path):
        # Download the config file using SFTP
        sftp = self.ssh.open_sftp()
        local_config_filename = f"/tmp/{self.p_id}.conf"  # Extract filename from path
        sftp.get(config_path, local_config_filename)  # Download the file
        sftp.close()

        # Now parse the downloaded config file
        self.configObj = SimulationConfig()
        self.configObj.parse(local_config_filename)
        os.remove(local_config_filename)

    def get_progress(self):
        remote_file_path = os.path.join(
            self.output_path, self.name, settings["MACRODATANAME"] + ".csv"
        )
        "/data/elundheim/MTS2D_output/simpleShear,s200x200l0.15,0.0002,1.0PBCt3minimizerCGLBFGSEpsg0.0001CGEpsg0.0001eps0.0001s14/simpleShear,s200x200l0.15,0.0002,1.0PBCt3minimizerCGLBFGSEpsg0.0001CGEpsg0.0001eps0.0001s14/macroData.csv"

        with self.ssh.open_sftp() as sftp:
            with sftp.file(remote_file_path, "r") as file:
                # Read the first line for headers
                headers = file.readline().strip().split(",")
                header_indices = {header: idx for idx, header in enumerate(headers)}

                # Now we want to find the last chunk of the file
                file_size = file.stat().st_size
                chunk_size = 1024  # Read last 1024 bytes, adjust if necessary
                start_pos = max(file_size - chunk_size, 0)
                file.seek(start_pos)
                chunk = file.read(file_size - start_pos)
                lines = chunk.decode("utf-8").splitlines()
                if not lines or len(lines) <= 1:
                    self.timeEstimation = "N/A"
                    self.progress = 0
                    return

                # Find the last complete line of data
                if len(lines[-1].split(",")) == len(headers):
                    last_line = lines[-1]
                else:
                    last_line = lines[-2]

                last_line_values = last_line.split(",")

                # I decided to remove the line number, that might give a miss match
                if len(last_line_values) == len(headers) - 1:
                    for header in header_indices:
                        header_indices[header] -= 1

                # load = last_line_values[header_indices["Load"]]
                runTime = last_line_values[header_indices["Run time"]]
                timeRemaining = last_line_values[header_indices["Est time remaining"]]

                # Log the results
                self.timeEstimation = f"RT: {runTime}, ETR: {timeRemaining}"
                self.progress = calculate_percentage_completed(runTime, timeRemaining)


class JobManager:
    def __init__(self) -> None:
        self.processes = []
        self.slurmJobs = []
        self.users = ["elundheim", "uog82gz"]

    # Function to be executed in each thread

    def find_processes_on_server(self, server):
        ssh = connectToCluster(server, False)  # Single SSH connection
        command = "ps -eo pid,etime,cmd | grep [M]TS2D | grep -v '/bin/sh'"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode("utf-8").strip().split("\n")

        if "CMakeFiles" in stdout_lines[0]:
            s = get_server_short_name(server)
            ssh.close()  # Ensure the connection is closed after use
            return [f"{s}:\n  Building..."]

        # Filter out empty lines
        stdout_lines = [line for line in stdout_lines if line.strip()]

        def fetch_job(line):
            attempts = 0
            max_attempts = 3
            e = ""
            while attempts < max_attempts:
                try:
                    # Each call gets its own channel but uses the same SSH connection
                    parts = line.split()
                    p_id = parts[0]  # PID
                    time_running = parts[1]  # Elapsed time
                    return Process(ssh, p_id, server, time_running)
                except Exception as er:
                    e = er
                    attempts += 1
                    time.sleep(
                        random.uniform(1, 3)
                    )  # Random delay to prevent synchronized reconnection attempts
                    # print(f"Attempt {attempts} failed for {server}: {e}")

            print(f"Error processing {line}: {e}")

        # Use ThreadPoolExecutor to process lines in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_jobs = [executor.submit(fetch_job, line) for line in stdout_lines]
            local_jobs = [future.result() for future in future_jobs]

        ssh.close()  # Ensure the connection is closed after use
        return local_jobs

    @staticmethod
    def find_jobs_waiting_in_queue(ssh):
        # Fetch all running jobs (once)
        command = 'squeue -h -t PENDING -o "%A"'
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode("utf-8").strip().split("\n")

        # Extract job IDs as integers
        pending_job_ids = [int(job_id) for job_id in stdout_lines if job_id.strip()]

        # Define a helper function to estimate jobs ahead for a given job
        def estimate_jobs_ahead(job_id):
            return sum(
                [pending_job_id < int(job_id) for pending_job_id in pending_job_ids]
            )

        # Return a function to calculate jobs ahead for any given job
        return estimate_jobs_ahead

    def find_slurm_jobs_on_server(self, server):
        slurm_jobs = []
        ssh = connectToCluster(server, False)

        # Updated squeue command to include more details
        # Function for finding job position in queue
        estimate_jobs_ahead = None
        for user in self.users:
            command = f'squeue -u {user} -h -o "%A %T %C %l %L %M %D %R"'
            stdin, stdout, stderr = ssh.exec_command(command)
            stdout_lines = stdout.read().decode("utf-8").strip().split("\n")
            # Filter out empty lines and split each line into fields
            for line in stdout_lines:
                if line.strip():
                    fields = line.strip().split()
                    job_details = {
                        "server": server,
                        "job_id": fields[0],
                        "state": fields[1],
                        "cpus": fields[2],
                        "time_limit": fields[3],
                        "time_left": fields[4],
                        "elapsed": fields[5],
                        "nodes": fields[6],
                        "node_list": fields[7],
                    }
                    # Check if there is a point in getting the queue position
                    if job_details["state"] == "PENDING":
                        if estimate_jobs_ahead is None:
                            # We only want to define this function once
                            estimate_jobs_ahead = self.find_jobs_waiting_in_queue(ssh)
                        job_details["wait_position"] = estimate_jobs_ahead(
                            job_details["job_id"]
                        )
                    slurm_jobs.append(job_details)
                    with lock:
                        global nr_jobs_found
                        nr_jobs_found += 1
                        update_progress(jobs=True)

        return slurm_jobs

    # Generalized method for executing a command on all servers in parallel
    def execute_command_on_servers(self, command_function):
        results = []
        with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
            future_to_server = {
                executor.submit(command_function, server): server
                for server in Servers.servers
            }
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.extend(result)
                except Exception as exc:
                    print(f"{server} generated an exception: {exc}")
        return results

    def showProcesses(self):
        self.processes = self.execute_command_on_servers(self.find_processes_on_server)
        print("")
        if not self.processes:
            print("No processes found")
        else:
            print("### PROCESSES ###")
            headers = [
                "ID",
                "Name",
                "Server",
                "Progress",
                "Run time",
                "Estimated time remaining",
            ]
            table = []

            for process in self.processes:
                if isinstance(process, str):
                    row = ["N/A", "Building", process.split(":")[0], "0%", "0", "N/A"]
                    table.append(row)
                    continue
                server_short_name = get_server_short_name(process.server)
                if process.timeEstimation == "N/A":
                    run_time = "N/A"
                    estimated_time_remaining = "N/A"
                else:
                    time_parts = process.timeEstimation.split(",")
                    run_time = time_parts[0].strip().replace("RT: ", "")
                    if len(time_parts) > 1:
                        estimated_time_remaining = (
                            time_parts[1].strip().replace("ETR: ", "")
                        )
                    else:
                        estimated_time_remaining = f"Error. Time_parts:{time_parts}"
                if isinstance(process.progress, str):
                    process.progress = -1

                row = [
                    process.p_id,
                    process.name,
                    server_short_name,
                    f"{process.progress:.1f}%",
                    run_time,
                    estimated_time_remaining,
                ]
                table.append(row)

            print(tabulate(table, headers=headers, tablefmt="grid"))
            print(f"Found {len(self.processes)} processes.")

    def findAndShowSlurmJobs(self):
        self.slurmJobs = self.execute_command_on_servers(self.find_slurm_jobs_on_server)
        print("")
        if not self.slurmJobs:
            print("No jobs found")
        else:
            print("### JOBS ###")
            table = []
            headers = [
                "Server",
                "Job ID",
                "State",
                "CPUs",
                # "Time Limit",
                "Time Left",
                "Elapsed",
                # "Nodes",
                "Node List",
            ]
            for job in self.slurmJobs:
                state = job["state"]
                if state == "PENDING":
                    state += f" ({job['wait_position']})"
                row = [
                    get_server_short_name(job["server"]),
                    job["job_id"],
                    state,
                    job["cpus"],
                    # job["time_limit"],
                    job["time_left"],
                    job["elapsed"],
                    # job["nodes"],
                    job["node_list"],
                ]
                table.append(row)
            print(tabulate(table, headers=headers, tablefmt="grid"))

    def cancel_jobs_on_server(self, server, job_ids):
        """
        Cancel jobs on a specific server after verifying their existence using self.slurmJobs.

        Parameters:
            server (str): The server on which to cancel the jobs.
            job_ids (str or list): A single job ID or a list of job IDs to cancel.
        """

        # Ensure job_ids is a list
        if not isinstance(job_ids, list):
            job_ids = [job_ids]

        # Check if self.slurmJobs is available
        if self.slurmJobs is None:
            print("WARNING: self.slurmJobs is None. Job existence will not be checked.")

        try:
            # Establish SSH connection
            ssh_client = connectToCluster(server, False)

            for job_id in job_ids:
                # Step 1: Check if the job exists using self.slurmJobs
                job_exists = False
                if self.slurmJobs is not None:
                    for job in self.slurmJobs:
                        if job["server"] == server and str(job["job_id"]) == str(
                            job_id
                        ):
                            job_exists = True
                            break
                    if not job_exists:
                        print(
                            f"Job {job_id} does not exist on {server}. Skipping cancellation."
                        )
                        continue  # Skip to the next job_id
                else:
                    # If self.slurmJobs is None, skip existence check
                    pass

                # Step 2: Attempt to cancel the job
                cancel_command = f"scancel {job_id}"
                try:
                    stdin, stdout, stderr = ssh_client.exec_command(cancel_command)
                    cancel_error = stderr.read().decode().strip()

                    if cancel_error:
                        print(
                            f"Error canceling job {job_id} on {server}: {cancel_error}"
                        )
                        continue  # Skip verification if cancellation failed

                    print(f"Cancellation command sent for job {job_id} on {server}.")
                except SSHException as ssh_exc:
                    print(
                        f"SSH error while canceling job {job_id} on {server}: {ssh_exc}"
                    )
                    continue
                except Exception as exc:
                    print(
                        f"Unexpected error while canceling job {job_id} on {server}: {exc}"
                    )
                    continue  # Depending on your needs, you might want to handle this differently

                # Step 3: Verify cancellation using squeue
                check_command = f"squeue -j {job_id} -h"
                try:
                    stdin, stdout, stderr = ssh_client.exec_command(check_command)
                    verify_output = stdout.read().decode().strip()
                    verify_error = stderr.read().decode().strip()

                    if verify_error:
                        print(
                            f"Error verifying cancellation for job {job_id} on {server}: {verify_error}"
                        )
                        continue  # Proceed to the next job_id

                    if not verify_output:
                        print(f"Successfully canceled job {job_id} on {server}.")
                    else:
                        print(
                            f"Failed to cancel job {job_id} on {server}. It may still be running."
                        )
                except SSHException as ssh_exc:
                    print(
                        f"SSH error while verifying cancellation for job {job_id} on {server}: {ssh_exc}"
                    )
                    continue
                except Exception as exc:
                    print(
                        f"Unexpected error while verifying cancellation for job {job_id} on {server}: {exc}"
                    )
                    continue

        except (SSHException, NoValidConnectionsError) as conn_exc:
            print(f"Error connecting to server {server}: {conn_exc}")
        except Exception as exc:
            print(
                f"An unexpected error occurred while connecting to server {server}: {exc}"
            )
        finally:
            # Close SSH connection if it's open
            try:
                ssh_client.close()
            except (AttributeError, SSHException) as close_exc:
                print(f"Error closing SSH connection to {server}: {close_exc}")

    def cancelAllJobs(self, force=False, on=None):
        """Cancel all Slurm jobs listed in self.slurmJobs."""
        if len(self.slurmJobs) == 0:
            print("No jobs found. Do you run showSlurmJobs first?")
        if force:
            jobs = {}
            for job in self.slurmJobs:
                if job["server"] not in jobs:
                    jobs[job["server"]] = [job["job_id"]]
                else:
                    jobs[job["server"]].append(job["job_id"])
            for server, jobs in jobs.items():
                self.cancel_jobs_on_server(server, jobs)
        else:
            for job in self.slurmJobs:
                if on is None or on == job["server"] or job["server"] in on:
                    print(
                        f"Are you sure you want to cancle job {job['job_id']} on {job['server']}?:"
                    )
                    if input("yes/no: ") != "yes":
                        continue

                    self.cancel_jobs_on_server(job["server"], job["job_id"])

    def kill_all_processes(self, server):
        """Kill all processes related to the user on the specified server."""
        # Warning, this will disconnect ssh connections as well
        ssh = connectToCluster(server, False)
        command = "pkill -u $(whoami)"  # This kills all processes for the user
        ssh.exec_command(command)
        print(f"All processes for user on {server} have been terminated.")

    def kill_process(self, server, pid):
        """Kill a specific process by PID on the specified server."""
        ssh = connectToCluster(server, False)
        command = f"kill {pid}"
        ssh.exec_command(command)
        print(f"Process {pid} on {server} has been terminated.")


if __name__ == "__main__":
    config = SimulationConfig()
    config.startLoad = 0.15
    config.loadIncrement = 0.0001
    config.rows = 10
    config.cols = 10
    config.maxLoad = 0.2

    minNrThreads = 1
    script = "benchmarking.py"
    script = "runSimulations.py"
    script = "parameterExploring.py"
    server = Servers.dalembert
    # server = Servers.condorcet
    command = f"python3 ~/simulation/SimulationScripts/Management/{script}"

    j = JobManager()
    j.findAndShowSlurmJobs()
    # j.cancel_job_on_server(server, 876296)
    # j.cancel_job_on_server(server, 876297)
    # server = find_server(minNrThreads)
    # j.cancelAllJobs()
    # build_on_server(server)
    # build_on_all_servers()

    # jobId = queue_remote_job(server, command, "FIRETest", minNrThreads)
    # j.showSlurmJobs()
    # j.showProcesses()

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
    if percentage_completed == 100:
        print("hi")
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
        # Example path:
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

                try:
                    # load = last_line_values[header_indices["load"]]
                    runTime = last_line_values[header_indices["run_time"]]
                    timeRemaining = last_line_values[
                        header_indices["est_time_remaining"]
                    ]
                except KeyError:
                    try:
                        # Try old header names
                        runTime = last_line_values[header_indices["Run_time"]]
                        timeRemaining = last_line_values[
                            header_indices["Est_time_remaining"]
                        ]
                    except KeyError as e:
                        print(
                            f"Error parsing progress data for process {self.p_id} on {self.server}: {e}"
                        )
                        self.timeEstimation = "N/A"
                        self.progress = "N/A"
                        return

                # Log the results
                self.timeEstimation = f"RT: {runTime}, ETR: {timeRemaining}"
                self.progress = calculate_percentage_completed(runTime, timeRemaining)


class JobManager:
    def __init__(self) -> None:
        self.processes: list[Process] = []
        self.slurmJobs = []
        self.users = ["elundheim", "uog82gz"]

    # Function to be executed in each thread

    def find_processes_on_server(self, server):
        ssh = connectToCluster(server, False)  # Single SSH connection
        command = "ps -eo pid,etime,cmd | grep [M]TS2D | grep -v '/bin/sh'"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode("utf-8").strip().split("\n")

        s = get_server_short_name(server)
        if "CMakeFiles" in stdout_lines[0]:
            ssh.close()  # Ensure the connection is closed after use
            return [f"{s}:\n  Building..."]

        # Filter out empty lines
        stdout_lines = [line for line in stdout_lines if line.strip()]

        def fetch_process(line):
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
            return f"{s}:\n {e}"

        # Use ThreadPoolExecutor to process lines in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_p = [executor.submit(fetch_process, line) for line in stdout_lines]
            local_p = [future.result() for future in future_p]

        ssh.close()  # Ensure the connection is closed after use
        return local_p

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
            command = f'squeue -u {user} -h -o "%A %j %T %C %l %L %M %D %R"'
            stdin, stdout, stderr = ssh.exec_command(command)
            stdout_lines = stdout.read().decode("utf-8").strip().split("\n")
            # Filter out empty lines and split each line into fields
            for line in stdout_lines:
                if line.strip():
                    fields = line.strip().split()
                    job_details = {
                        "server": server,
                        "job_id": fields[0],
                        "job_name": fields[1],
                        "state": fields[2],
                        "cpus": fields[3],
                        "time_limit": fields[4],
                        "time_left": fields[5],
                        "elapsed": fields[6],
                        "nodes": fields[7],
                        "node_list": fields[8],
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

    def findProcesses(self):
        global nr_processes_found
        nr_processes_found = 0
        self.processes = self.execute_command_on_servers(self.find_processes_on_server)
        print("")

    def findAndShowProcesses(self):
        self.findProcesses()
        if not self.processes:
            print("No processes found")
        else:
            print("### PROCESSES ###")
            headers = [
                "ID",
                "Name",
                "Server",
                "Progress",
                "Run_time",
                "Estimated_time_remaining",
            ]
            table = []

            for process in self.processes:
                if process is None:
                    row = ["N/A", "Error", "Error", "0%", "0", "N/A"]
                    table.append(row)
                    continue
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

    def findSlurmJobs(self):
        global nr_jobs_found
        nr_jobs_found = 0
        self.slurmJobs = self.execute_command_on_servers(self.find_slurm_jobs_on_server)
        print("")

    def findAndShowSlurmJobs(self):
        self.findSlurmJobs()

        if not self.slurmJobs:
            print("No jobs found")
        else:
            print("### JOBS ###")
            table = []
            headers = [
                "Server",
                "Job ID",
                "Job Name",
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
                    job.get("job_name", ""),
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

    def getJobData(self):
        self.findSlurmJobs()
        self.findProcesses()

    def cancel_jobs_on_server(self, server, job_ids="all", force=False):
        """
        Cancel jobs on a specific server after verifying their existence using self.slurmJobs.

        Parameters
        ----------
        server : str
            The server on which to cancel the jobs.
        job_ids : str | int | Iterable
            A single job ID, an iterable of job IDs, or the string "all".
        """

        # 1. Special case: cancel all and stop.
        if job_ids == "all":
            return self.cancelAllJobs(force=force, on=server)

        # 2. Normalize job_ids to a list of strings.
        if not isinstance(job_ids, (list, tuple, set)):
            job_ids = [job_ids]
        job_ids = [str(j) for j in job_ids]

        # 3. Precompute known jobs for this server (if available).
        if self.slurmJobs is not None:
            known_ids = {
                str(job["job_id"])
                for job in self.slurmJobs
                if job.get("server") == server
            }
        else:
            known_ids = None
            print("WARNING: self.slurmJobs is None. Job existence will not be checked.")

        results = {}

        ssh_client = None
        try:
            # 4. Connect once.
            try:
                ssh_client = connectToCluster(server, False)
            except (SSHException, NoValidConnectionsError) as exc:
                msg = f"Error connecting to server {server}: {exc}"
                print(msg)
                # Mark all jobs as failed at connect stage.
                for jid in job_ids:
                    results[jid] = {"ok": False, "stage": "connect", "error": msg}
                return results
            except Exception as exc:
                msg = f"Unexpected error connecting to server {server}: {exc}"
                print(msg)
                for jid in job_ids:
                    results[jid] = {"ok": False, "stage": "connect", "error": msg}
                return results

            # 5. Process each job.
            for job_id in job_ids:
                # 5a. Existence check (if we have slurmJobs).
                if known_ids is not None and job_id not in known_ids:
                    msg = f"Job {job_id} does not exist on {server}. Skipping."
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "precheck", "error": msg}
                    continue

                # 5b. Cancel job.
                cancel_cmd = f"scancel {job_id}"
                try:
                    stdin, stdout, stderr = ssh_client.exec_command(cancel_cmd)
                    cancel_err = stderr.read().decode().strip()
                except SSHException as exc:
                    msg = f"SSH error while canceling job {job_id} on {server}: {exc}"
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "cancel", "error": msg}
                    continue
                except Exception as exc:
                    msg = f"Unexpected error while canceling job {job_id} on {server}: {exc}"
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "cancel", "error": msg}
                    continue

                if cancel_err:
                    msg = f"Error canceling job {job_id} on {server}: {cancel_err}"
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "cancel", "error": msg}
                    continue

                print(f"Cancellation command sent for job {job_id} on {server}.")

                # 5c. Verify cancellation.
                check_cmd = f"squeue -j {job_id} -h"
                try:
                    stdin, stdout, stderr = ssh_client.exec_command(check_cmd)
                    verify_out = stdout.read().decode().strip()
                    verify_err = stderr.read().decode().strip()
                except SSHException as exc:
                    msg = (
                        f"SSH error while verifying cancellation for job {job_id} "
                        f"on {server}: {exc}"
                    )
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "verify", "error": msg}
                    continue
                except Exception as exc:
                    msg = (
                        f"Unexpected error while verifying cancellation for job {job_id} "
                        f"on {server}: {exc}"
                    )
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "verify", "error": msg}
                    continue

                if verify_err:
                    msg = (
                        f"Error verifying cancellation for job {job_id} on {server}: "
                        f"{verify_err}"
                    )
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "verify", "error": msg}
                elif verify_out:
                    # Job still appears in squeue.
                    msg = (
                        f"Job {job_id} still appears in squeue output on {server} "
                        f"after cancellation."
                    )
                    print(msg)
                    results[job_id] = {"ok": False, "stage": "verify", "error": msg}
                else:
                    msg = f"Successfully canceled job {job_id} on {server}."
                    print(msg)
                    results[job_id] = {"ok": True, "stage": "done", "error": None}

        finally:
            if ssh_client is not None:
                try:
                    ssh_client.close()
                except (SSHException, Exception) as exc:
                    print(f"Error closing SSH connection to {server}: {exc}")

        return results

    def cancelAllJobs(self, force=False, on=None):
        """Cancel Slurm jobs listed in self.slurmJobs.

        Args:
            force: if True, cancel without asking.
            on:    None, a single server name, or an iterable of server names.
        """
        if len(self.slurmJobs) == 0:
            print("No jobs found. Do you run showSlurmJobs first?")
            return

        assert isinstance(force, bool), "Must be bool"

        def _matches_server(server, on):
            if on is None:
                return True
            if isinstance(on, str):
                return server == on
            # assume iterable of server names
            return server in on

        if force:
            jobs_by_server = {}
            for job in self.slurmJobs:
                if not _matches_server(job["server"], on):
                    continue
                jobs_by_server.setdefault(job["server"], []).append(job["job_id"])

            for server, job_ids in jobs_by_server.items():
                self.cancel_jobs_on_server(server, job_ids)
        else:
            for job in self.slurmJobs:
                if not _matches_server(job["server"], on):
                    continue

                print(
                    f"Are you sure you want to cancel job {job['job_id']} on {job['server']}?:"
                )
                if input("yes/no: ") != "yes":
                    continue

                # See point 2 below
                self.cancel_jobs_on_server(job["server"], [job["job_id"]])

    def kill_all_processes(self, server):
        """Kill all processes related to the user on the specified server."""
        # Warning, this will disconnect ssh connections as well
        ssh = connectToCluster(server, False)
        command = "pkill -u $(whoami)"  # This kills all processes for the user
        ssh.exec_command(command)
        print(f"All processes for user on {server} have been terminated.")

    def kill_processes(self, server, pids, verbal=True):
        if isinstance(pids, (str, int)):
            pids = [pids]
        """Kill a specific process by PID on the specified server."""
        ssh = connectToCluster(server, False)
        for pid in pids:
            command = f"kill {pid}"
            ssh.exec_command(command)
            if verbal:
                print(f"Process {pid} on {server} has been terminated.")

    def cancelJobs(self, configsToStop: list[SimulationConfig], dryRun=False):
        for conf in configsToStop:
            self.cancelJobsByNameSubstring(
                conf.name, case_sensitive=True, dryRun=dryRun
            )

    def cancelJobsByNameSubstring(
        self,
        substring: str,
        *,
        force=False,
        on=None,
        case_sensitive=False,
        dryRun=False,
    ):
        """Cancel Slurm jobs whose job name contains `substring`.

        Notes
        -----
        - Requires that `self.slurmJobs` has been populated (this method refreshes it).
        - Match is done on the Slurm job name (%j), not your simulation config name.

        Args
        ----
        substring:
            Substring to match in the job name.
        force:
            If True, cancel without prompting.
        on:
            None (all servers), a single server name, or an iterable of server names.
        case_sensitive:
            If False (default), match is case-insensitive.
        """

        if not substring:
            print("ERROR: substring must be non-empty")
            return

        if not self.slurmJobs:
            print("No jobs found. Did you run findSlurmJobs?")
            return

        def _matches_server(server, on):
            if on is None:
                return True
            if isinstance(on, str):
                return server == on
            return server in on

        needle = substring if case_sensitive else substring.lower()

        # Group matches by server.
        jobs_by_server = {}
        for job in self.slurmJobs:
            if not _matches_server(job.get("server"), on):
                continue

            name = str(job.get("job_name", ""))
            hay = name if case_sensitive else name.lower()
            if needle in hay:
                jobs_by_server.setdefault(job["server"], []).append(job["job_id"])

        if not jobs_by_server:
            print(f"No jobs matched substring '{substring}'.")
            return

        # Show what will be canceled.
        matched = [
            {
                "server": get_server_short_name(j["server"]),
                "job_id": j["job_id"],
                "job_name": j.get("job_name", ""),
                "state": j.get("state", ""),
            }
            for j in self.slurmJobs
            if j.get("server") in jobs_by_server
            and (
                needle
                in (
                    str(j.get("job_name", ""))
                    if case_sensitive
                    else str(j.get("job_name", "")).lower()
                )
            )
        ]
        print("### MATCHED JOBS ###")
        print(
            tabulate(
                matched,
                headers={
                    "server": "Server",
                    "job_id": "Job ID",
                    "job_name": "Job Name",
                    "state": "State",
                },
                tablefmt="grid",
            )
        )

        if not force:
            print(
                f"Cancel ALL matched jobs containing '{substring}'? (Total: {sum(len(v) for v in jobs_by_server.values())})"
            )
            if input("yes/no: ") != "yes":
                print("Aborted.")
                return
        if dryRun:
            print("Dry run, not canceling...")
        else:
            # Cancel per server.
            for server, job_ids in jobs_by_server.items():
                self.cancel_jobs_on_server(server, job_ids, force=True)


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

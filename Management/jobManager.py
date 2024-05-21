import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from tabulate import tabulate
import time
import random
from datetime import timedelta
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from runOnCluster import queue_remote_job, find_outpath_on_server
from clusterStatus import find_server, Servers, get_server_short_name
from connectToCluster import uploadProject, connectToCluster
from configGenerator import SimulationConfig
from dataManager import get_directory_size

sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))
# Now we can import from Management
from settings import settings


def parse_duration(duration_str):
    pattern = r'(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?'
    matches = re.match(pattern, duration_str.strip())
    
    if not matches:
        return timedelta()
    
    days, hours, minutes, seconds = matches.groups(default='0')
    
    return timedelta(
        days=int(days),
        hours=int(hours),
        minutes=int(minutes),
        seconds=float(seconds)
    )

def calculate_percentage_completed(runtime_str, estimated_remaining_str):
    runtime = parse_duration(runtime_str)
    estimated_remaining = parse_duration(estimated_remaining_str)
    
    total_time = runtime + estimated_remaining
    percentage_completed = (runtime / total_time) * 100
    return percentage_completed

class Process:
    """
    NB This does not find slurm jobs! It checks the processes running on the 
    cluster and finds all instances of MTS2D running.
    """
    paris_zone = ZoneInfo("Europe/Paris")
    gmt_zone = ZoneInfo("Europe/London")
    
    def __init__(self, ssh, processID, server, timeRunning) -> None:
        self.ssh = ssh
        self.name=""
        self.p_id=processID
        self.command=""
        self.server=server
        self.timeEstimation=""
        self.progress=""
        self.progress_timestamp=None
        self.dataSize=0
        self.output_path=""
        self.configObj=None
        self.timeRunning=timeRunning

        self.getInfoFromProcess()


    def getInfoFromProcess(self):
        stdin, stdout, stderr = self.ssh.exec_command(f"ps -p {self.p_id} -o args=")
        command_line = stdout.read().decode('utf-8').strip()
        parts = command_line.split()
        self.command = command_line
        # Assuming the second and third parts of the command are what you're interested in
        config_path = parts[-2]  # This seems to be new or unused; ensure it's handled as needed
        self.output_path = parts[-1]  # Assuming the last part is the output path
        self.get_config_file(config_path)
        self.name=os.path.splitext(os.path.basename(config_path))[0]
        self.get_progress()
        self.dataSize = get_directory_size(self.ssh, self.output_path+self.name)


    def get_config_file(self, config_path):
        # Download the config file using SFTP
        sftp = self.ssh.open_sftp()
        local_config_filename = f"/tmp/{self.p_id}.conf" # Extract filename from path
        sftp.get(config_path, local_config_filename)  # Download the file
        sftp.close()

        # Now parse the downloaded config file
        self.configObj = SimulationConfig()
        self.configObj.parse(local_config_filename)
        os.remove(local_config_filename)

    def get_progress(self):
        remote_file_path = (self.output_path +
                            self.name + '/' +
                            settings['MACRODATANAME'] + '.csv')

        with self.ssh.open_sftp() as sftp:
            with sftp.file(remote_file_path, 'r') as file:
                # Read the first line for headers
                headers = file.readline().strip().split(',')
                header_indices = {header: idx for idx, header in enumerate(headers)}

                # Now we want to find the last chunk of the file
                file_size = file.stat().st_size
                chunk_size = 1024  # Read last 1024 bytes, adjust if necessary
                start_pos = max(file_size - chunk_size, 0)
                file.seek(start_pos)
                chunk = file.read(file_size - start_pos)
                lines = chunk.decode('utf-8').splitlines()
                if not lines or len(lines)<=1:
                    self.timeEstimation = "N/A"
                    self.progress = 0
                    return

                # Find the last complete line of data
                if len(lines[-1].split(',')) == len(headers):
                    last_line = lines[-1]
                else:
                    last_line = lines[-2]


                last_line_values = last_line.split(',')
                load = last_line_values[header_indices['Load']]
                runTime = last_line_values[header_indices['Run time']]
                timeRemaining = last_line_values[header_indices['Est time remaining']]

                # Log the results
                self.timeEstimation = f"RT: {runTime}, ETR: {timeRemaining}"
                self.progress = calculate_percentage_completed(runTime, timeRemaining)
    

class JobManager:
    def __init__(self) -> None:        
        self.processes = []
        self.slurmJobs = []
        self.user="elundheim"

    # Function to be executed in each thread

    def find_processes_on_server(self, server):
        ssh = connectToCluster(server, False)  # Single SSH connection
        command = f"ps -eo pid,etime,cmd | grep [M]TS2D | grep -v '/bin/sh'"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode('utf-8').strip().split('\n')

        if 'CMakeFiles' in stdout_lines[0]:
            s = get_server_short_name(server)
            ssh.close()  # Ensure the connection is closed after use
            return [f"{s}:\n  Building..."]

        # Filter out empty lines
        stdout_lines = [line for line in stdout_lines if line.strip()]

        def fetch_job(line):
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts:
                try:
                    # Each call gets its own channel but uses the same SSH connection
                    parts = line.split()
                    p_id = parts[0]  # PID
                    time_running = parts[1]  # Elapsed time
                    return Process(ssh, p_id, server, time_running)
                except Exception as e:
                    attempts += 1
                    time.sleep(random.uniform(1, 3))  # Random delay to prevent synchronized reconnection attempts
                    print(f"Attempt {attempts} failed for {server}: {e}")
                    if attempts >= max_attempts:
                        print(f"Error processing {line}: {e}")


        # Use ThreadPoolExecutor to process lines in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_jobs = [executor.submit(fetch_job, line) for line in stdout_lines]
            local_jobs = [future.result() for future in future_jobs]

        ssh.close()  # Ensure the connection is closed after use
        return local_jobs
    
    def find_slurm_jobs_on_server(self, server):
        slurm_jobs = []
        ssh = connectToCluster(server, False)
        # Updated squeue command to include more details
        command = f"squeue -u {self.user} -h -o \"%A %T %C %l %L %M %D %R\""
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode('utf-8').strip().split('\n')
        # Filter out empty lines and split each line into fields
        for line in stdout_lines:
            if line.strip():
                fields = line.strip().split()
                job_details = {
                    'server':server,
                    'job_id':fields[0],
                    'state':fields[1],
                    'cpus':fields[2],
                    'time_limit':fields[3],
                    'time_left':fields[4],
                    'elapsed':fields[5],
                    'nodes':fields[6],
                    'node_list':fields[7]
                }
                slurm_jobs.append(job_details)
        return slurm_jobs
    
    # Generalized method for executing a command on all servers in parallel
    def execute_command_on_servers(self, command_function):
        results = []
        with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
            future_to_server = {executor.submit(command_function, server): server for server in Servers.servers}
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.extend(result)
                except Exception as exc:
                    print(f'{server} generated an exception: {exc}')
        return results


    def showProcesses(self):
        self.processes = self.execute_command_on_servers(self.find_processes_on_server)

        if not self.processes:
            print("No processes found")
        else:
            print("### PROCESSES ###")
            headers = ["ID", "Name", "Server", "Progress", "Run time", "Estimated time remaining"]
            table = []

            for process in self.processes:
                if isinstance(process, str):
                    row = ['N/A','Building', process.split(':')[0], '0%', '0', 'N/A']
                    table.append(row)
                    continue
                server_short_name = get_server_short_name(process.server)
                if(process.timeEstimation=='N/A'):
                    run_time = 'N/A'
                    estimated_time_remaining = 'N/A'
                else:
                    time_parts = process.timeEstimation.split(',')
                    run_time = time_parts[0].strip().replace('RT: ', '')
                    estimated_time_remaining = time_parts[1].strip().replace('ETR: ', '')
                row = [
                    process.p_id,
                    process.name,
                    server_short_name,
                    f"{process.progress:.1f}%",
                    run_time,
                    estimated_time_remaining
                ]
                table.append(row)

            print(tabulate(table, headers=headers, tablefmt="grid"))
            print(f"Found {len(self.processes)} processes.")


    def showSlurmJobs(self):
        self.slurmJobs = self.execute_command_on_servers(self.find_slurm_jobs_on_server)
        if not self.slurmJobs:
            print("No jobs found")
        else:
            print("### JOBS ###")
            table = []
            headers = ["Server", "Job ID", "State", "CPUs", "Time Limit", "Time Left", "Elapsed", "Nodes", "Node List"]
            for job in self.slurmJobs:
                row = [
                    job['server'],
                    job['job_id'],
                    job['state'],
                    job['cpus'],
                    job['time_limit'],
                    job['time_left'],
                    job['elapsed'],
                    job['nodes'],
                    job['node_list']
                ]
                table.append(row)
            print(tabulate(table, headers=headers, tablefmt="grid"))


    def cancel_job_on_server(self, server, job_id):
        """Function to cancel a job on a specific server."""
        try:
            ssh = connectToCluster(server, False)
            command = f"scancel {job_id}"
            ssh.exec_command(command)
            print(f"Cancelled job {job_id} on {server}")
        except Exception as exc:
            print(f"Error canceling job {job_id} on {server}: {exc}")

    def cancelAllJobs(self):
        """Cancel all Slurm jobs listed in self.slurmJobs."""
        for server, job_id in self.slurmJobs:
            print(f"Are you sure you want to cancle job {job_id} on {server}?:") 
            if input(f"yes/no : ")!="yes":
                continue
            else:
                self.cancel_job_on_server(server, job_id)
    
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

    import sys


    config = SimulationConfig()
    config.startLoad=0.15
    config.loadIncrement=0.0001
    config.rows=10
    config.cols=10
    config.maxLoad=0.2

    if len(sys.argv) >= 2:
        onlyCheckJobs = sys.argv[1]
    else: 
        onlyCheckJobs = 'False'

    minNrThreads = 61
    script = "benchmarking.py"
    script = "parameterExploring.py"
    script = "runSimulations.py"
    server = Servers.dalembert
    server = Servers.condorcet
    server = Servers.galois
    command=f"python3 /home/elundheim/simulation/SimulationScripts/Management/{script}"
    if onlyCheckJobs.upper() == "TRUE":
        j=JobManager()
        j.showSlurmJobs()
        j.showProcesses()
    else:
        j=JobManager()
        j.cancel_job_on_server(server, 558366)
        # server = find_server(minNrThreads)
        uploadProject(server)

        jobId = queue_remote_job(server, command, "energy", minNrThreads)
        #j.showProcesses()
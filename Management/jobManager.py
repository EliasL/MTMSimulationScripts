import os
from pathlib import Path
import sys
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import humanize

from concurrent.futures import ThreadPoolExecutor, as_completed
from runOnCluster import queue_remote_job, find_outpath_on_server
from clusterStatus import find_server, Servers, get_server_short_name
from connectToCluster import uploadProject, connectToCluster
from configGenerator import SimulationConfig
from dataManager import get_directory_size

sys.path.append(str(Path(__file__).resolve().parent.parent / 'Plotting'))
# Now we can import from Management
from settings import settings

class Job:
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
                self.progress = float(load)/float(self.configObj.maxLoad)
        


    def __str__(self) -> str:
        return (
                f"P_ID {self.p_id}: {self.name} on {get_server_short_name(self.server)}\n"
                #f"\tCommand: {self.command}\n"
                f"  Progress: {self.progress*100:.1f}%\n"
                f"  Time: {self.timeEstimation}\n"
                f"  {self.output_path} : {self.dataSize}\n"
            )
    

class JobManager:
    def __init__(self) -> None:        
        self.processes = []
        self.slurmJobs = []
        self.user="elundheim"

    # Function to be executed in each thread
    def find_processes_on_server(self, server):
        local_jobs = []
        ssh = connectToCluster(server, False)
        # The [M] is so that we don't find the grep process searching for MTS2D
        command = f"ps -eo pid,etime,cmd | grep [M]TS2D | grep -v '/bin/sh'"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode('utf-8').strip().split('\n')
        if 'CMakeFiles' in stdout_lines[0]:
            s = get_server_short_name(server)
            return [f"{s}:\n  Building..."]

        # Filter out empty lines
        stdout_lines = [line for line in stdout_lines if line.strip()]
        for line in stdout_lines:
            parts = line.split()
            p_id = parts[0]  # PID
            time_running = parts[1]  # Elapsed time
            local_jobs.append(Job(ssh, p_id, server, time_running))
        return local_jobs
    
    def find_slurm_jobs_on_server(self, server):
        slurm_jobs = []
        ssh = connectToCluster(server, False)
        # Use squeue to list jobs for the user, outputting only the job ID
        command = f"squeue -u {self.user} -h -o %A"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode('utf-8').strip().split('\n')
        # Filter out empty lines
        stdout_lines = [line for line in stdout_lines if line.strip()]
        for line in stdout_lines:
            job_id = line.strip()  # Slurm job ID
            slurm_jobs.append((server,job_id))
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

    def getProcesses(self):
        self.processes = self.execute_command_on_servers(self.find_processes_on_server)

        if not self.processes:
            print("No jobs found")
        else:
            for process in self.processes:
                print(process)
            print(f"Found {len(self.processes)} processes.")

    def getSlurmJobs(self):
        self.slurmJobs = self.execute_command_on_servers(self.find_slurm_jobs_on_server)
        print("Jobs:")
        [print("  ", job) for job in self.slurmJobs]


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

    minNrThreads = 65
    script = "benchmarking.py"
    script = "runSimulation.py"
    script = "parameterExploring.py"
    server = Servers.dalembert
    server = Servers.condorcet
    command=f"python3 /home/elundheim/simulation/SimulationScripts/Management/{script}"
    if onlyCheckJobs.upper() == "TRUE":
        j=JobManager()
        j.getSlurmJobs()
        j.getProcesses()
    else:
        j=JobManager()
        # server = find_server(minNrThreads)
        uploadProject(server)
        jobId = queue_remote_job(server, command, "expl", minNrThreads)
        j.getProcesses()
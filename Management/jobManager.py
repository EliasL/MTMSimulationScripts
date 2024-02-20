import os
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
        self.progress=0
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
        config_path = parts[1]  # This seems to be new or unused; ensure it's handled as needed
        self.output_path = parts[2]  # Assuming the last part is the output path

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
                            self.name + '.log')
        # Open the remote file for reading
        # We read the second last line because the very last line might not be complete
        with self.ssh.open_sftp().file(remote_file_path, 'rb') as file:
            # Seek to the second last line
            file.seek(-2, 2)  # Seek to the second last byte
            while file.read(1) != b'\n':  # Move to the start of the last line
                file.seek(-2, 1)  # Move back one byte
            file.seek(-2, 1)  # Move back one byte
            while file.read(1) != b'\n':  # Move to the start of the second last line
                file.seek(-2, 1)  # Move back one byte

            # Read and return the second last line
            second_last_line = file.readline().decode('utf-8').strip()
            # Sample log line
            #log_line = "[2024-02-12 08:32:44.395] [infoLog] [info] 23% runTime: 2d 11h 48m 44.513s ETR: 8d 0h 41m 8.965s Load: 0.351350"

            # Extract timestamp from the square brackets
            timestamp_match = re.search(r"\[(.*?)\]", second_last_line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                # Parse the timestamp string into a datetime object
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                # Manually adjust for timezone
                timestamp = timestamp.replace(tzinfo=self.gmt_zone)
                self.progress_timestamp = timestamp
            if not '[' in second_last_line:
                # This means that we are not getting propper progress updates yet
                self.progress = "..."
            else:
                # Remove all contents within square brackets from the log line
                cleaned_log_line = re.sub(r"\[.*?\]", "", second_last_line).strip()
                self.progress = cleaned_log_line



    def __str__(self) -> str:
        if self.progress_timestamp:
            time_since_update = humanize.naturaltime(
                datetime.now(self.paris_zone) - self.progress_timestamp
            )
            formatted_time = self.progress_timestamp.strftime('%H:%M:%S')
            timeUpdate = f"{formatted_time}, {time_since_update}"
        else:
            timeUpdate = '...'
        return (
                f"P_ID {self.p_id}: {self.name} on {get_server_short_name(self.server)}\n"
                #f"\tCommand: {self.command}\n"
                f"  Progress: {self.progress}\n"
                f"  Time since update: {timeUpdate}\n"
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
        command = f"ps -eo pid,etime,cmd | grep [M]TS2D | grep -v '/bin/sh'"
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout_lines = stdout.read().decode('utf-8').strip().split('\n')
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
        [print(job) for job in self.slurmJobs]


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

if __name__ == "__main__":
    minNrThreads = 4
    script = "runSimulations.py"
    script = "benchmarking.py"
    server = Servers.galois
    server2 = Servers.condorcet
    command=f"python3 /home/elundheim/simulation/SimulationScripts/Management/{script}"
    
    j=JobManager()
    # j.getSlurmJobs()
    # j.cancelAllJobs()

    # server = find_server(minNrThreads)
    # uploadProject(server)
    # uploadProject(server2)
    # jobId = queue_remote_job(server, command, "bench", minNrThreads)
    # jobId = queue_remote_job(server2, command, "bench", minNrThreads)
    j.getProcesses()
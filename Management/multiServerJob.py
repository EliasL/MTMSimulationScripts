from configGenerator import SimulationConfig, ConfigGenerator

from concurrent.futures import ThreadPoolExecutor, as_completed
from runOnCluster import queue_remote_job, find_outpath_on_server
from clusterStatus import find_server, Servers, get_server_short_name
from connectToCluster import uploadProject, connectToCluster
from configGenerator import SimulationConfig
from dataManager import get_directory_size
from jobManager import JobManager

"""
The idea here is that we will have a huge number of configuration files, and we 
want to distribute them among all the available servers.

We are going to send the arguments to the generate function from ConfigGenerator
to the runSimulations script, and this scipt will generate all the configurations
that will run on that server
"""


def generateCommand(configs):
    pass


if __name__ == "__main__":
    script = "runSimulations.py"
    server = Servers.dalembert
    server = Servers.condorcet
    server = Servers.galois
    command = (
        f"python3 /home/elundheim/simulation/SimulationScripts/Management/{script}"
    )

    j = JobManager()
    # j.cancel_job_on_server(server, 558366)
    # server = find_server(minNrThreads)
    uploadProject(server)

    jobId = queue_remote_job(server, command, "energy", 0)
    # j.showProcesses()

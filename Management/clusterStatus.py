from concurrent.futures import ThreadPoolExecutor, as_completed
from connectToCluster import Servers, connectToCluster, get_server_short_name
from tabulate import tabulate

class ServerInfo:
    def __init__(self):
        self.nrTotalCores=0
        self.nrUsedCores=0
        self.nrFreeCores=0
        self.nrJobsRunning=0
        self.nrJobsWaitingInQueue=0
        self.theNodeCanAcceptMoreJobs=False

def get_server_info(ssh_client):
    # Create a ServerInfo object
    si = ServerInfo()

    # Get the total number of cores in the system
    stdin, stdout, stderr = ssh_client.exec_command("grep -c ^processor /proc/cpuinfo")
    si.nrTotalCores = int(stdout.read().decode().strip())

    # Get the total number of cores allocated to running jobs
    command_busy_cores = "squeue -t R -o '%.6C' | awk '{s+=$1} END {print s}'"
    stdin, stdout, stderr = ssh_client.exec_command(command_busy_cores)
    si.nrUsedCores = int(stdout.read().decode().strip())

    si.nrFreeCores = si.nrTotalCores-si.nrUsedCores

    # Calculate the number of jobs running and waiting in the queue
    command_jobs_running = "squeue -t R | wc -l"
    stdin, stdout, stderr = ssh_client.exec_command(command_jobs_running)
    si.nrJobsRunning = int(stdout.read().decode().strip()) - 1  # Adjust for header line
    
    command_jobs_waiting = "squeue -t PD | wc -l"
    stdin, stdout, stderr = ssh_client.exec_command(command_jobs_waiting)
    si.nrJobsWaitingInQueue = int(stdout.read().decode().strip()) - 1  # Adjust for header line
    
   # Check for exclusive job settings
    command_exclusive_jobs = "squeue -h -o '%i %t %p %C %D %R' | grep ' R ' | awk '{print $6}'"
    stdin, stdout, stderr = ssh_client.exec_command(command_exclusive_jobs)
    exclusive_job_settings = stdout.read().decode().strip().split('\n')

    # Simplified logic to set theNodeCanAcceptMoreJobs
    # Adjust this based on how you define exclusivity or constraints in your jobs
    si.theNodeCanAcceptMoreJobs = 'exclusive' not in exclusive_job_settings

    return si

# Function to add color based on the value
def colorize(value, good_value, bad_value):
    # Attempt to safely evaluate the expression to a float
    evaluated_value = eval(str(value))
    if evaluated_value is None:
        return "Invalid input"

    # Determine coloring based on comparison with good and bad values
    if good_value < bad_value:  # Lower values are better
        if evaluated_value <= good_value:
            color = "\033[92m"  # Green
        elif evaluated_value >= bad_value:
            color = "\033[91m"  # Red
        else:
            color = "\033[93m"  # Yellow
    else:  # Higher values are better
        if evaluated_value >= good_value:
            color = "\033[92m"  # Green
        elif evaluated_value <= bad_value:
            color = "\033[91m"  # Red
        else:
            color = "\033[93m"  # Yellow

    return f"{color}{value}\033[0m"

def display_server_info(server_info):
    data = []
    for server, info in server_info.items():
        if isinstance(info, str):  # Error handling case
            data.append([server, "Error", info, "N/A"])
            continue
        server_short_name = get_server_short_name(server)
        nr_unused_cores = f"{info.nrFreeCores}/{info.nrTotalCores}"
        colored_cores = colorize(info.nrFreeCores, 50, 15) + f"/{info.nrTotalCores}"
        jobs_running = info.nrJobsRunning
        jobs_waiting = colorize(info.nrJobsWaitingInQueue, 0, 2)
        data.append([server_short_name, colored_cores, jobs_running, jobs_waiting])

    headers = ['Server', 'Free Cores', 'Jobs Running', 'Jobs Waiting']
    print(tabulate(data, headers=headers, tablefmt='grid'))

def task(server):
    try:
        ssh = connectToCluster(server, False)
        info = get_server_info(ssh)
        return info
    except Exception as e:
        return f"Error connecting to {server}: {e}"

def get_all_server_info(servers=Servers.servers):
    # A dictionary to hold server information, keyed by server
    server_info = {}

    # Use ThreadPoolExecutor for threading instead of multiprocessing Pool
    with ThreadPoolExecutor(max_workers=len(servers)) as executor:
        # Future to server mapping
        future_to_server = {executor.submit(task, server): server for server in servers}
        
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            try:
                info = future.result()  # Get the result from future
                server_info[server] = info
            except Exception as exc:
                print(f'{server} generated an exception: {exc}')
                server_info[server] = f"Error: {exc}"

    return server_info

def find_server(minNrThreads):
    print("Finding available server...")
    server_info = get_all_server_info()

    eligible_servers = []
    for server, info in server_info.items():
        if isinstance(info, str):  # Skip servers with errors
            continue
        if info.nrFreeCores >= minNrThreads and info.theNodeCanAcceptMoreJobs:
            eligible_servers.append((server, info))

    if not eligible_servers:
        print("No server currently meets the core requirement and can accept more jobs.")
        return None

    # Choose the server with the fewest jobs in the queue
    server, info = min(eligible_servers, key=lambda x: x[1].nrJobsWaitingInQueue)
    print(f"Selected {get_server_short_name(server)} with {info.nrJobsWaitingInQueue} jobs in the queue.")
    return server

if __name__ == "__main__":
    server_info = get_all_server_info()
    display_server_info(server_info)

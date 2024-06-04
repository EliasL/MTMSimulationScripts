from concurrent.futures import ThreadPoolExecutor, as_completed
from connectToCluster import Servers, connectToCluster, get_server_short_name
from tabulate import tabulate


class ServerInfo:
    def __init__(self):
        self.nrTotalCores = 0
        self.nrUsedCores = 0
        self.nrFreeCores = 0
        self.nrJobsRunning = 0
        self.nrJobsWaitingInQueue = 0
        self.theNodeCanAcceptMoreJobs = False
        self.totalRAM = 0  # GB
        self.usedRAM = 0  # GB
        self.freeRAM = 0  # GB

        self.totalScore = 0


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

    si.nrFreeCores = si.nrTotalCores - si.nrUsedCores

    # Calculate the number of jobs running and waiting in the queue
    command_jobs_running = "squeue -t R | wc -l"
    stdin, stdout, stderr = ssh_client.exec_command(command_jobs_running)
    si.nrJobsRunning = int(stdout.read().decode().strip()) - 1  # Adjust for header line

    command_jobs_waiting = "squeue -t PD | wc -l"
    stdin, stdout, stderr = ssh_client.exec_command(command_jobs_waiting)
    si.nrJobsWaitingInQueue = (
        int(stdout.read().decode().strip()) - 1
    )  # Adjust for header line

    # Check for exclusive job settings
    command_exclusive_jobs = (
        "squeue -h -o '%i %t %p %C %D %R' | grep ' R ' | awk '{print $6}'"
    )
    stdin, stdout, stderr = ssh_client.exec_command(command_exclusive_jobs)
    exclusive_job_settings = stdout.read().decode().strip().split("\n")

    # Simplified logic to set theNodeCanAcceptMoreJobs
    # Adjust this based on how you define exclusivity or constraints in your jobs
    si.theNodeCanAcceptMoreJobs = "exclusive" not in exclusive_job_settings

    # Get total RAM in GB
    stdin, stdout, stderr = ssh_client.exec_command(
        "free -m | grep Mem: | awk '{print $2}'"
    )
    si.totalRAM = round(int(stdout.read().decode().strip()) / 1000)

    # Get used RAM in GB (total - free)
    stdin, stdout, stderr = ssh_client.exec_command(
        "free -m | grep Mem: | awk '{print $3}'"
    )
    si.usedRAM = round(int(stdout.read().decode().strip()) / 1000)
    si.freeRAM = si.totalRAM - si.usedRAM

    return si


# Function to add color based on the value
def colorize(value, good_value, bad_value, string=None):
    # Attempt to safely evaluate the expression to a float
    evaluated_value = eval(str(value))
    if evaluated_value is None:
        return "Invalid input"

    score = 0  # -1 bad, 0 ok, +1 good

    # Determine coloring based on comparison with good and bad values
    if good_value < bad_value:  # Lower values are better
        if evaluated_value <= good_value:
            color = "\033[92m"  # Green
            score = 1
        elif evaluated_value >= bad_value:
            color = "\033[91m"  # Red
            score = -1
        else:
            color = "\033[93m"  # Yellow
    else:  # Higher values are better
        if evaluated_value >= good_value:
            color = "\033[92m"  # Green
            score = 1
        elif evaluated_value <= bad_value:
            color = "\033[91m"  # Red
            score = -1
        else:
            color = "\033[93m"  # Yellow
    if string:
        return f"{color}{string}\033[0m", score
    else:
        return f"{color}{value}\033[0m", score


def score_and_color_server(info):
    total_score = 0

    # Define the metrics to evaluate
    metrics = {
        "Free Cores": (info.nrFreeCores, info.nrTotalCores, 50, 15),
        "GB RAM": (info.freeRAM, info.totalRAM, 50, 15),
        "Jobs R": (info.nrJobsRunning, None, 2, 10),
        "Jobs W": (info.nrJobsWaitingInQueue, None, 0, 2),
    }

    results = []
    headers = []

    for label, (value, total, good_value, bad_value) in metrics.items():
        if total:
            colorized_value, score = colorize(
                value, good_value, bad_value, f"{value}/{total}"
            )
        else:
            colorized_value, score = colorize(value, good_value, bad_value)
        total_score += score
        results.append(colorized_value)
        headers.append(label)
    return results, headers, total_score


def display_server_info(server_info):
    data = []
    for server, info in server_info.items():
        if isinstance(info, str):  # Error handling case
            data.append([server, "Error", info, "N/A"])
            continue

        name = get_server_short_name(server)

        results, headers, total_score = score_and_color_server(info)

        # Insert the server name in the front
        results.insert(0, colorize(total_score, 1, -1, name)[0])
        headers.insert(0, "Server")
        info.totalScore = total_score
        # Append results with total_score for sorting
        data.append((total_score, results))

    # Sort by total_score, then extract sorted results
    sorted_data = [item[1] for item in sorted(data, key=lambda x: x[0], reverse=True)]

    print(tabulate(sorted_data, headers=headers, tablefmt="grid"))


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
                print(f"{server} generated an exception: {exc}")
                server_info[server] = f"Error: {exc}"

    return server_info


def find_server(minNrThreads=16, minRAM=16):
    print("Finding available server...")
    server_info = get_all_server_info()

    eligible_servers = []
    for server, info in server_info.items():
        if isinstance(info, str):  # Skip servers with errors
            continue
        if (
            info.nrFreeCores >= minNrThreads
            and info.freeRAM >= minRAM
            and info.theNodeCanAcceptMoreJobs
        ):
            eligible_servers.append((server, info))

    if not eligible_servers:
        print("No server currently meets the requirements.")
        return None

    # Choose the server with the best score
    server, info = max(eligible_servers, key=lambda x: score_and_color_server(x[1])[2])
    print(
        f"Selected {get_server_short_name(server)} with {info.nrJobsWaitingInQueue} jobs in the queue."
    )
    return server


if __name__ == "__main__":
    server_info = get_all_server_info()
    display_server_info(server_info)

from concurrent.futures import ThreadPoolExecutor, as_completed
from connectToCluster import Servers, connectToCluster, get_server_short_name
from tabulate import tabulate


class ServerInfo:
    def __init__(self):
        self.sName = ""  # Server name
        self.nrTotalCores = 0
        self.nrUsedCores = 0
        self.nrFreeCores = 0
        self.nrJobsRunning = 0
        self.nrJobsWaitingInQueue = 0
        self.totalRAM = 0  # GB
        self.usedRAM = 0  # GB
        self.freeRAM = 0  # GB
        self.nodeState = ""  # Node state (e.g., up, down, drained)
        self.nrIdleNodes = 0
        self.nrNodesTotal = 0
        self.totalScore = 0


def get_server_info(ssh_client):
    # Create a ServerInfo object
    si = ServerInfo()

    # Execute combined command for cores and node information
    # This returns "jobsRunning\njobsWaiting\nAlocated/Idle/Other/TotalCores"
    command = "squeue -h -t R | wc -l;\squeue -h -t PD | wc -l;\sinfo -h -o '%C'"
    stdin, stdout, stderr = ssh_client.exec_command(command)
    outputs = stdout.read().decode().split("\n")
    si.nrJobsRunning = int(outputs[0])
    si.nrJobsWaitingInQueue = int(outputs[1])
    alocated, idle, other, total = outputs[2].split("/")
    si.nrFreeCores = idle
    si.nrUsedCores = alocated
    si.nrTotalCores = total

    command_state = "sinfo -N --noheader -o '%t'"
    stdin, stdout, stderr = ssh_client.exec_command(command_state)
    outputs = stdout.read().decode().strip().split("\n")

    # Count the occurrences of each node status
    status_counts = {}
    for status in outputs:
        if status not in status_counts:
            status_counts[status] = 1
        else:
            status_counts[status] += 1
    # Determine the predominant status for each node
    si.nodeState = max(status_counts, key=status_counts.get)

    # Count total and idle nodes
    total_nodes = sum(status_counts.values())  # Total nodes based on all lines returned
    idle_count = sum(
        count
        for status, count in status_counts.items()
        if "idle" in status or "mix" in status
    )
    si.nrIdleNodes = idle_count
    si.nrNodesTotal = total_nodes

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
        "Idle Cores": (info.nrFreeCores, info.nrTotalCores, 50, 15),
        "Idle Nodes": (info.nrIdleNodes, info.nrNodesTotal, 1, 0),
        # "GB RAM": (info.freeRAM, info.totalRAM, 50, 15),
        "Jobs R": (info.nrJobsRunning, None, 2, 10),
        "Jobs W": (info.nrJobsWaitingInQueue, None, 0, 4),
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
    total_score = total_score * 100 + int(info.nrFreeCores) * 2
    return results, headers, total_score


def display_server_info(server_info):
    data = []
    headers = []
    for server, info in server_info.items():
        if isinstance(info, str):  # Error handling case
            data.append([server, "Error", info, "N/A"])
            continue

        name = get_server_short_name(server)
        results, headers, total_score = score_and_color_server(info)
        if "drain" in info.nodeState or "down" in info.nodeState:
            name += "(d)"
            total_score -= 1000
        # Insert the server name in the front
        results.insert(0, colorize(total_score, 200, -100, name)[0])
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
    except Exception as e:
        print(f"Error connecting to {server}: {e}")

    info = get_server_info(ssh)
    return info


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
            except Exception as exc:
                print(f"{server} generated an exception: {exc}")
                server_info[server] = f"Error: {exc}"
                continue
            info.sName = server
            server_info[server] = info

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

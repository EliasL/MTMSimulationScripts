import os
from paramiko import ProxyCommand, SSHClient, AutoAddPolicy, SSHConfig, SSHException
import subprocess


class Servers:
    # Server variables
    galois = "galois.pmmh-cluster.espci.fr"
    pascal = "pascal.pmmh-cluster.espci.fr"
    schwartz = "schwartz.pmmh-cluster.espci.fr"
    lagrange = "lagrange.pmmh-cluster.espci.fr"
    condorcet = "condorcet.pmmh-cluster.espci.fr"
    dalembert = "dalembert.pmmh-cluster.espci.fr"
    poincare = "poincare.pmmh-cluster.espci.fr"
    fourier = "fourier.pmmh-cluster.espci.fr"
    descartes = "descartes.pmmh-cluster.espci.fr"

    mesopsl = "mesopsl.obspm.fr"

    local_path_mac = "/Volumes/data/"

    # List of server variables for iteration or list-like access
    servers = [
        galois,
        pascal,
        schwartz,
        lagrange,
        condorcet,
        dalembert,
        poincare,
        fourier,
        descartes,
        mesopsl,
    ]

    # If we want to search all the servers including the local storage, we can do that
    serversAndLocal = servers + [local_path_mac]

    # Default server
    default = pascal


def uploadProject(cluster_address="Servers.default", verbose=False):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the local paths based on the script directory

    # Define the local paths based on the script directory and resolve them to absolute paths
    local_path_MTS2D = os.path.abspath(os.path.join(script_dir, "..", "..", "MTS2D"))
    local_path_SS = os.path.abspath(os.path.join(script_dir, ".."))
    clusterPath = f"elundheim@{cluster_address}:/home/elundheim/simulation/"

    ssh_command = [
        "ssh",
        f"elundheim@{cluster_address}",
        "mkdir -p /home/elundheim/simulation /home/elundheim/simulation/MTS2D /home/elundheim/simulation/SimulationScripts",
    ]

    output_options = None if verbose else subprocess.DEVNULL

    if verbose:
        print("Creating directories")
    subprocess.run(
        ssh_command, check=True, stdout=output_options, stderr=output_options
    )

    rsync_command_MTS2D = [
        "rsync",
        "-avz",
        "--progress",
        "--exclude",
        ".git",
        "--exclude",
        "build",
        "--exclude",
        "build-release",
        "--exclude",
        "libs/**-build",
        "--exclude",
        "libs/**-subbuild",
        "--exclude",
        "Visuals/",
        local_path_MTS2D,
        clusterPath,
    ]

    rsync_command_SS = [
        "rsync",
        "-avz",
        "--progress",
        "--exclude",
        "Plots/",
        "--exclude",
        ".git",
        local_path_SS,
        clusterPath,
    ]

    subprocess.run(
        rsync_command_MTS2D,
        check=True,
        stdout=output_options,
        stderr=output_options,
    )
    subprocess.run(
        rsync_command_SS, check=True, stdout=output_options, stderr=output_options
    )

    if verbose:
        print("Project folders successfully uploaded.")


def get_ssh_config():
    config_file = os.path.expanduser("~/.ssh/config")
    ssh_config = SSHConfig()
    with open(config_file) as f:
        ssh_config.parse(f)
    return ssh_config


def connectToCluster(cluster_address=Servers.default, verbose=True):
    # Get the SSH configuration
    ssh_config = get_ssh_config()
    config = ssh_config.lookup(cluster_address)

    # Create the SSH client
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())

    # Define the proxy command using ProxyJump from the SSH config
    proxy_jump = config.get("proxyjump")
    try:
        if proxy_jump:
            proxy_command = ProxyCommand(
                f"ssh -W {config['hostname']}:{22} {proxy_jump}"
            )
            # Connect to the server through the proxy with increased timeout
            ssh.connect(
                config["hostname"],
                username=config["user"],
                sock=proxy_command,
            )
        else:
            # Connect using the private key instead of a password
            ssh.connect(cluster_address, username=config["user"])
    except Exception as e:
        raise SSHException(f"Error connecting to {cluster_address}: {e}")

    if verbose:
        print(f"SSH connection established to {cluster_address}.")
    return ssh


def get_server_short_name(full_address):
    return full_address.split(".")[0]


if __name__ == "__main__":
    ssh = connectToCluster(Servers.mesopsl)
    ssh = connectToCluster(Servers.galois)

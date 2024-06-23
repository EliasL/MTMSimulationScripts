import os
from paramiko import SSHClient, AutoAddPolicy, AuthenticationException
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


def connectToCluster(cluster_address=Servers.default, verbose=True):
    username = "elundheim"

    # Step 1: Establish an SSH connection to the cluster using Paramiko.
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(AutoAddPolicy())

    try:
        # Connect using the private key instead of a password
        ssh.connect(cluster_address, username=username)
        if verbose:
            print(f"SSH connection established to {cluster_address}.")
    except AuthenticationException:
        raise AuthenticationFailedException(
            f"Authentication with {cluster_address} failed. Please check your SSH key."
        )
    except Exception as e:
        raise SSHConnectionException(f"Error connecting to {cluster_address}: {e}")

    return ssh


def get_server_short_name(full_address):
    return full_address.split(".")[0]


class AuthenticationFailedException(Exception):
    pass


class SSHConnectionException(Exception):
    pass

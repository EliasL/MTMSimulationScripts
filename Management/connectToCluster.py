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
        descartes
    ]

    # Default server
    default = galois

def uploadProject(cluster_address=Servers.default):
    try:
        # Define the rsync command as a list of arguments
        rsync_command = [
            "rsync",
            "-avz",
            "--progress",
            "--exclude", ".git",
            "--exclude", "build",
            "--exclude", "build-release",
            "/home/elias/Work/PhD/Code/MTS2D/",
            f"elundheim@{cluster_address}:/home/elundheim/simulation/"
        ]
        
        # Run the rsync command
        subprocess.run(rsync_command, check=True)
        print("Project folder successfully uploaded.")
        
    except subprocess.CalledProcessError as e:
        raise(Exception(f"An error occurred while uploading the project: {e}"))

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
        raise AuthenticationFailedException(f"Authentication with {cluster_address} failed. Please check your SSH key.")
    except Exception as e:
        raise SSHConnectionException(f"Error connecting to {cluster_address}: {e}")

    return ssh



def get_server_short_name(full_address):
    return full_address.split('.')[0]

class AuthenticationFailedException(Exception):
    pass

class SSHConnectionException(Exception):
    pass

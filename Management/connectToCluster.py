import os
import subprocess
from paramiko import SSHException


class Servers:
    # Server variables
    galois = "galois.pmmh-cluster.espci.fr"
    pascal = "pascal.pmmh-cluster.espci.fr"
    schwartz = "schwartz.pmmh-cluster.espci.fr"
    lagrange = "lagrange.pmmh-cluster.espci.fr"
    # Condorcet is slow
    # condorcet = "condorcet.pmmh-cluster.espci.fr"
    dalembert = "dalembert.pmmh-cluster.espci.fr"
    poincare = "poincare.pmmh-cluster.espci.fr"
    fourier = "fourier.pmmh-cluster.espci.fr"
    descartes = "descartes.pmmh-cluster.espci.fr"

    # mesopsl = "mesopsl.obspm.fr"

    jeanZay = "jean-zay.idris.fr"

    local_path_mac = "/Volumes/data/"

    # List of server variables for iteration or list-like access
    servers = [
        galois,
        pascal,
        schwartz,
        lagrange,
        # condorcet,
        dalembert,
        poincare,
        fourier,
        descartes,
        # mesopsl,
        jeanZay,
    ]

    # If we want to search all the servers including the local storage, we can do that
    serversAndLocal = servers + [local_path_mac]

    # Default server
    default = pascal


def create_directories(cluster_address, cluster_base_path, verbose=False):
    """
    Connect to the cluster and create required directories.
    """
    try:
        # Connect to the cluster
        ssh = connectToCluster(cluster_address, verbose=verbose)

        # Commands to create directories
        commands = [
            f"mkdir -p {cluster_base_path}",
            # f"mkdir -p {cluster_base_path}MTS2D",
            # f"mkdir -p {cluster_base_path}SimulationScripts",
        ]
        for cmd in commands:
            if verbose:
                print(f"Executing command: {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            if stdout.channel.recv_exit_status() != 0:
                error_message = stderr.read().decode()
                raise SSHException(f"Command failed: {cmd}\nError: {error_message}")

        if verbose:
            print("Directories created successfully.")
    except SSHException as e:
        print(f"Error during SSH operations: {e}")
    finally:
        # Close SSH connection
        if "ssh" in locals():
            ssh.close()
            if verbose:
                print("SSH connection closed.")


def sync_folders(
    cluster_address, cluster_base_path, exclude_list, local_paths, verbose=False
):
    """
    Sync local folders to the cluster using rsync.
    """

    # Helper function to build rsync command with exclude list
    """
                                        /Users/eliaslundheim/work/PhD/MTS2D     jean-zay.idris.fr:~/simulation/
    rsync -avz     -e "ssh -vvv -T"     /Users/eliaslundheim/work/PhD/MTS2D     jean-zay.idris.fr:~/simulation/
    """

    def build_rsync_command(source_path):
        rsync_command = [
            "rsync",
            "-avz",
            "-e",
            "ssh -T",
            # "--protocol=31",
            "--progress",
        ]
        for item in exclude_list:
            rsync_command.extend(["--exclude", item])
        rsync_command.extend([source_path, f"{cluster_address}:{cluster_base_path}"])
        return rsync_command

    output_options = None if verbose else subprocess.DEVNULL

    # Execute rsync commands for each local path
    for local_path in local_paths:
        rsync_command = build_rsync_command(local_path)
        if verbose:
            print(f"Executing rsync: {' '.join(rsync_command)}")
        subprocess.run(
            rsync_command, check=True, stdout=output_options, stderr=output_options
        )

    if verbose:
        print("Folders successfully synced.")


def uploadProject(cluster_address="Servers.default", verbose=False, setup=True):
    """
    Main function to create directories and sync folders to the cluster.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the local paths
    local_path_MTS2D = os.path.abspath(os.path.join(script_dir, "..", "..", "MTS2D"))
    local_path_SS = os.path.abspath(os.path.join(script_dir, ".."))
    cluster_base_path = "~/simulation/"

    # Common exclude list
    exclude_list = [
        ".git",
        "build",
        "build-release",
        "libs/**-build",
        "libs/**-subbuild",
        "Visuals/",
        "Plots/",
        "venv/",
        "bootstrapData",
        "_gael_2D",
        "profiling",
        "MTMath",
    ]

    # Step 1: Create directories on the cluster
    create_directories(cluster_address, cluster_base_path, verbose=verbose)

    # Step 2: Sync folders to the cluster
    sync_folders(
        cluster_address=f"{cluster_address}",
        cluster_base_path=cluster_base_path,
        exclude_list=exclude_list,
        local_paths=[local_path_MTS2D, local_path_SS],
        verbose=verbose,
    )

    # Step 3: Optionally run setup
    if setup:
        run_setup(cluster_address, verbose)


def run_setup(cluster_address, verbose):
    user = getServerUserName(cluster_address)
    ssh_command = [
        "ssh",
        f"{user}@{cluster_address}",
        "cd ~/simulation/SimulationScripts && pip3 install ./Management ./Plotting",
    ]

    output_options = None if verbose else subprocess.DEVNULL

    if verbose:
        print("Running setup")
    subprocess.run(
        ssh_command, check=True, stdout=output_options, stderr=output_options
    )


def run_full_setup(cluster_address, verbose=False):
    return  # This doesn't quite work
    ssh_command = [
        "ssh",
        f"elundheim@{cluster_address}",
        "cd ~/simulation/SimulationScripts && ./setup_env.sh",
    ]

    output_options = None if verbose else subprocess.DEVNULL

    if verbose:
        print("Running setup")
    subprocess.run(
        ssh_command, check=True, stdout=output_options, stderr=output_options
    )


def download_folders(cluster_address, configs, destination):
    # Connect to the server
    ssh = connectToCluster(cluster_address, False)

    # Check if /data2 exists, otherwise use /data
    stdin, stdout, stderr = ssh.exec_command(
        "if [ -d /data2 ]; then echo '/data2'; else echo '/data'; fi"
    )
    base_dir = stdout.read().strip().decode()

    user = getServerUserName(cluster_address)
    data_path = os.path.join(base_dir, user)

    remote_folder_name = "MTS2D_output"
    remote_folder_path = f"/{data_path}/{remote_folder_name}"
    # List all folders within the output folder
    command = f"cd {remote_folder_path}; ls -d */"
    stdin, stdout, stderr = ssh.exec_command(command)
    folders = stdout.read().strip().decode().split("\n")
    folders = [folder.rstrip("/") for folder in folders]  # Clean up folder names

    # Now we check if our folder is in this cluster
    outPaths = []
    for config in configs:
        folderToDownload = config.generate_name(withExtension=False)

        if folderToDownload not in folders:
            # No match: we do nothing
            continue
        else:
            remote_ssh_folder_path = (
                f"{user}@{cluster_address}:{remote_folder_path}/{folderToDownload}"
            )
            local_folder_path = os.path.join(destination, folderToDownload)

            # Ensure local directory exists
            if not os.path.exists(local_folder_path):
                os.makedirs(local_folder_path)

            # rsync command to download folder without overwriting existing files
            rsync_command = [
                "rsync",
                "-avz",
                "--progress",
                remote_ssh_folder_path,
                destination,
            ]

            # Run the rsync command
            subprocess.run(rsync_command)

            # If the rsync was successful, return the local folder path
            outPaths.append((local_folder_path, config))

    return outPaths


def get_ssh_config():
    from paramiko import SSHConfig

    config_file = os.path.expanduser("~/.ssh/config")
    ssh_config = SSHConfig()
    with open(config_file) as f:
        ssh_config.parse(f)
    return ssh_config


def connectToCluster(cluster_address=Servers.default, verbose=True):
    from paramiko import ProxyCommand, SSHClient, AutoAddPolicy, SSHException

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


def getServerUserName(cluster_address):
    ssh_config = get_ssh_config()
    config = ssh_config.lookup(cluster_address)
    return config["user"]


def get_server_short_name(full_address):
    return full_address.split(".")[0]


if __name__ == "__main__":
    ssh = connectToCluster(Servers.mesopsl)
    ssh = connectToCluster(Servers.galois)

import os
import subprocess
from connectToCluster import Servers
from getpass import getpass
import pexpect
from urllib.request import urlretrieve
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient


def generate_ssh_key(key_path):
    """Generate an SSH key pair if it doesn't already exist."""
    if not os.path.exists(key_path) and not os.path.exists(key_path + ".pub"):
        subprocess.run(
            ["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", key_path, "-N", ""],
            check=True,
        )
        print(f"SSH key generated at {key_path}")
    else:
        print("SSH key already exists.")


def copy_ssh_key_to_server(server, username, key_path, password):
    """Copy the public SSH key to the server's authorized keys using sshpass."""
    command = f"sshpass -p {password} ssh-copy-id -o StrictHostKeyChecking=no -i {key_path} {username}@{server}"
    try:
        result = subprocess.run(
            command, shell=True, check=True, stderr=subprocess.PIPE, text=True
        )
        print(f"SSH key copied to {server}")

    except subprocess.CalledProcessError as e:
        print(f"Failed to install SSH key on {server}: {e.stderr}")


def change_password(server, username, old_password, new_password):
    ssh_command = f"ssh {username}@{server}"
    child = pexpect.spawn(ssh_command, timeout=3)  # Increase the timeout to 60 seconds

    # Handle both the password prompt and any welcome messages or warnings
    patterns = ["password:", "System restart required", f"{username}@.*\$ "]
    index = child.expect(patterns)
    if index == 0:
        child.sendline(old_password)
        # Now expect the shell prompt
        child.expect(f"{username}@.*\$ ")
    elif index == 1 or index == 2:
        # Handle the system restart message or directly at the prompt
        print("Handling special case or at prompt")

    child.sendline("passwd")
    child.expect([r"\(current\) UNIX password:\s*"])
    child.sendline(old_password)
    child.expect([r"Enter new UNIX password:\s*"])
    child.sendline(new_password)
    child.expect([r"Retype new UNIX password: "])
    child.sendline(new_password)
    child.expect(rf"{username}@[^\s:]*:.*\$ ")
    print(f"Password changed on {server}")
    child.close()


def main():
    username = "elundheim"  # Change this to your actual username on the servers

    key_path = os.path.expanduser("~/.ssh/id_rsa")  # Automatically get the SSH key path
    password = getpass(
        "Enter your SSH password (will not be echoed): "
    )  # Securely enter password

    # Generate SSH key pair if it doesn't exist
    generate_ssh_key(key_path)

    # Loop through servers and copy the SSH key
    for server in [Servers.mesopsl]:
        copy_ssh_key_to_server(server, username, key_path, password)

    # Prompt for the new password
    # new_password = getpass("Enter the new password (will not be echoed): ")

    # Change password on each server
    # for server in Servers.servers:
    #     try:
    #         change_password(server, username, password, new_password)
    #     except Exception as e:
    #         print(f"Failed on {server}: {e}")
    #         continue


def get_vscode_commit_id():
    # Run the VS Code version command and parse output
    result = subprocess.run(["code", "--version"], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8").split()
    version, commit_id = output[0], output[1]
    print(f"Detected VS Code version: {version}, Commit ID: {commit_id}")
    return commit_id


def download_vscode_server(commit_id, local_path):
    url = f"https://update.code.visualstudio.com/commit:{commit_id}/server-linux-x64/stable"
    if not os.path.exists(local_path):
        print(f"Downloading VS Code Server from {url}")
        urlretrieve(url, local_path)
        print("Download completed.")
    else:
        print("File already downloaded.")


def scp_transfer(local_path, remote_path, hostname, username, password=None):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password)
    scp = SCPClient(ssh.get_transport())
    print(f"Transferring {local_path} to {hostname}:{remote_path}")
    scp.put(local_path, remote_path)
    scp.close()
    ssh.close()
    print("Transfer completed.")


if __name__ == "__main__":
    main()

    # commit_id = get_vscode_commit_id()
    # local_path = "/Users/eliaslundheim/Downloads/vscode-server-linux-x64.tar.gz"
    # hostname = "remote-server"  # Update this with your remote server's address
    # username = "elundheim"  # Update this with your SSH username
    # remote_path = "/work/elundheim/vscode-server-linux-x64.tar.gz"

    # # download_vscode_server(commit_id, local_path)
    # #scp_transfer(local_path, remote_path, hostname, username)

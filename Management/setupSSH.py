import os
import subprocess
from connectToCluster import Servers
from getpass import getpass
import subprocess
import pexpect


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
        subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        print(f"SSH key copied to {server}")

    except subprocess.CalledProcessError as e:
        print(f"Failed to install SSH key on {server}: {e.stderr.decode()}")


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
    key_path = "/home/elias/.ssh/id_rsa"  # SSH key path
    password = getpass(
        "Enter your SSH password (will not be echoed): "
    )  # Securely enter password

    # Generate SSH key pair if it doesn't exist
    generate_ssh_key(key_path)

    # Loop through servers and copy the SSH key
    for server in Servers.servers:
        copy_ssh_key_to_server(server, username, key_path, password)

    # Prompt for the new password
    new_password = getpass("Enter the new password (will not be echoed): ")

    # Change password on each server
    for server in Servers.servers:
        try:
            change_password(server, username, password, new_password)
        except:
            print(f"Failed on {server}")
            continue


if __name__ == "__main__":
    main()

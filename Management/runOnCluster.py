from connectToCluster import uploadProject, Servers
from fabric import Connection
import textwrap

SERVER_USER = 'elundheim'
SERVER_KEY_PATH = "/home/elias/.ssh/id_rsa"

def run_remote_script(server_hostname, script_path):
    # Establish the SSH connection
    connect_kwargs = {"key_filename": SERVER_KEY_PATH}  # Path to your SSH private key
    with Connection(host=server_hostname, user=SERVER_USER, connect_kwargs=connect_kwargs) as c:
        # Execute the remote command (your Python script)
        result = c.run(f'python3 -u {script_path}', hide=False, warn=True)

        # `hide=False` means output and errors are printed in real time
        # `warn=True` means execution won't stop on errors (similar to try/except)

        # Check the result
        if result.ok:
            print("Script executed successfully.")
        else:
            print(f"Script execution failed: {result.stderr}")

def find_outpath_on_server(server_hostname):
    # Establish the SSH connection
    connect_kwargs = {"key_filename": SERVER_KEY_PATH}  # Path to your SSH private key
    with Connection(host=server_hostname, user=SERVER_USER, connect_kwargs=connect_kwargs) as c:
        # Execute the remote command (your Python script)
        result = c.run(f'python3 -u /home/elundheim/simulation/Management/simulationManager.py', hide=True, warn=True)
    return result.stdout.strip()

def run_remote_command(server_hostname, command):
    # Establish the SSH connection
    connect_kwargs = {"key_filename": SERVER_KEY_PATH}  # Path to your SSH private key
    with Connection(host=server_hostname, user=SERVER_USER, connect_kwargs=connect_kwargs) as c:
        # Execute the remote command (your Python script)
        result = c.run(command, hide=False, warn=True)

        # `hide=False` means output and errors are printed in real time
        # `warn=True` means execution won't stop on errors (similar to try/except)

        # Check the result
        if result.ok:
            print("Script executed successfully.")
        else:
            print(f"Script execution failed: {result.stderr}")

def queue_remote_job(server_hostname, command, job_name, nrThreads):
    outPath="/home/elundheim/simulation/Job/"
    output_file = outPath + f"log-{job_name}.out"
    error_file = outPath + f"err-{job_name}.err"
    # Establish the SSH connection
    connect_kwargs = {"key_filename": SERVER_KEY_PATH}  # Path to your SSH private key
    with Connection(host=server_hostname, user=SERVER_USER, connect_kwargs=connect_kwargs) as c:
        # Create a batch script content
        batch_script = textwrap.dedent(f"""
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --time=3-06:00:00
            #SBATCH --ntasks={nrThreads}
            #SBATCH --output={output_file}
            #SBATCH --error={error_file}
            {command}
        """).strip()
        # Create the batch script on the server
        batch_script_path = "/home/elundheim/simulation/Job/" + job_name + ".sh"
        c.run(f'echo "{batch_script}" > {batch_script_path}')
        
        # Submit the batch script to Slurm
        result = c.run(f'sbatch {batch_script_path}', hide=True, warn=True)

        # Check the submission result
        if result.ok:
            print("Batch script submitted successfully.")
            print(result.stdout)  # This will include the Slurm job ID
            # Extract the Slurm job ID from the result
            job_id_line = result.stdout.strip().split()[-1]
            try:
                slurm_job_id = int(job_id_line)
            except ValueError as e:
                raise(Exception(f"Error parsing jobID: {e}"))
        else:
            print(f"Batch script submission failed: {result.stderr}")

    return slurm_job_id  # Return the Slurm job ID


if __name__ == "__main__":
    # Choose which server to run on
    server = Servers.servers[4]
    # Upload/sync the project
    uploadProject(server)
    # Choose script to run
    script_path = '/home/elundheim/simulation/Management/benchmarking.py'
    # Generate sbatch script

    # Queue the script on the server
    run_remote_script(server, script_path)
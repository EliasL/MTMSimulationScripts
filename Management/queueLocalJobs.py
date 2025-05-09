import os
import textwrap
import subprocess


def get_batch_script(command, job_name, nrThreads, outPath):
    output_file = os.path.join(outPath, f"log-{job_name}.out")
    error_file = os.path.join(outPath, f"err-{job_name}.err")

    # Create a batch script content
    batch_script = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time=0-19:00:00
        #SBATCH --ntasks={nrThreads}
        #SBATCH --output={output_file}
        #SBATCH --error={error_file}
        # Set a high nice value to decrease priority
        # SBATCH --nice=10000

        # Load Modules
        module load cmake 

        # Command to run
        {command}
    """).strip()
    return batch_script


def get_threads_from_command(command):
    key = "nrThreads"
    if key not in command:
        # Return default nrThreads value
        # It would be nice to import configGenerator, but its dificult to do
        # when this is run from a cluster
        return 1
    return int(command.split(f" {key}=")[1].split(" ")[0])


def queue_local_jobs(commands, job_name, useQueueSystem=True):
    base_path = "~/simulation/MTS2D/"
    # Expand the user's home directory and check if the path exists
    base_path = os.path.expanduser(base_path)
    outPath = os.path.join(base_path, "JobOutput")
    error_file = os.path.join(outPath, f"err-{job_name}.err")

    # Ensure the base path exists
    if not os.path.exists(base_path):
        raise Exception(f"The directory {base_path} does not exist.")

    # Ensure the JobOutput directory exists
    os.makedirs(outPath, exist_ok=True)

    # Truncate the error file to clear old errors
    open(error_file, "w").close()

    batch_script_path = os.path.join(outPath, f"{job_name}.sh")

    for command in commands:
        nrThreads = get_threads_from_command(command)
        batch_script = get_batch_script(command, job_name, nrThreads, outPath)

        # Write the batch script to a file
        with open(batch_script_path, "w") as f:
            f.write(batch_script)

        if useQueueSystem:
            # Run the batch script locally using subprocess
            result = subprocess.run(["sbatch", batch_script_path])
            if result.returncode != 0:
                print(f"Batch script execution failed: {result.stderr}")
                return None
        else:
            print("Warning! Running processes outside of the SLURM queue system!")
            print("Make sure you have the permission of Sylvain.")
            run_command_outside_of_queue(command)


def run_command_outside_of_queue(command):
    # Does not work
    # Expand the '~' to the absolute path
    command = command.strip().split(" ")
    path = command[1]
    assert ".py" in path, f"This is maybe not the path? {path}"
    script_path = os.path.expanduser(path)
    # Define the command as a list
    command[1] = script_path
    try:
        process = subprocess.Popen(
            command,
            stderr=subprocess.PIPE,
        )
        print(f"Process started with PID: {process.pid}")
        return process
    except FileNotFoundError as e:
        print(f"Executable not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def install_editable_package():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", script_dir]
        )
        print(f"Package installed successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")


if __name__ == "__main__":
    import sys
    import ast

    # install_editable_package()
    if len(sys.argv) >= 1:
        kwargs_string = "".join(sys.argv[1:])
        kwargs_string = kwargs_string.replace("\u203d", '"')
        kwargs = ast.literal_eval(kwargs_string)
        queue_local_jobs(useQueueSystem=True, **kwargs)
    else:
        raise ValueError(f"No args {sys.argv}.")

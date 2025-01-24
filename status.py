import subprocess
from pathlib import Path
import re
from Management.clusterStatus import get_all_server_info, display_server_info
from Management.dataManager import DataManager

# from Management.configGenerator import ConfigGenerator
from Management.jobManager import JobManager
from Management.multiServerJob import (
    bigJob,
    build_on_all_servers,
    JobManager,
    get_server_short_name,
)


def disp_data():
    dm = DataManager()
    # dm.clean_projects_on_servers()
    # dm.clean_projects_on_servers()
    # dm.delete_useless_dumps(False)
    nrThreads = 1
    nrSeeds = 40
    configs, labels = bigJob(nrThreads, nrSeeds)
    # dm.delete_data_from_configs(configs, dryRun=True)
    # dm.delete_folders_below_size(100, dryRun=True)
    dm.printData()
    dm.findData(silent=True)
    print("^   Old data above   ^")
    print("v Updated data below v")
    dm.printData()
    # dm.delete_all_found_data()


def disp_servers():
    info = get_all_server_info()
    display_server_info(info)


def disp_jobs():
    j = JobManager()
    j.findAndShowSlurmJobs()
    j.showProcesses()


def run_script():
    # Path to your .scpt file
    script_path = f"{Path(__file__).resolve().parent}/Management/startMonitoring.scpt"

    # Get the current directory
    current_dir = Path(__file__).resolve().parent

    # Path to the virtual environment Python
    venv_python_path = current_dir / "venv" / "bin" / "python"

    # Read the AppleScript file
    with open(script_path, "r") as file:
        applescript = file.read()

    # Replace placeholders in the AppleScript
    applescript = applescript.replace("PATH", str(current_dir))
    applescript = applescript.replace(
        "python", str(venv_python_path)
    )  # Replace python command with venv Python

    # Write the modified AppleScript to a temporary file
    temp_script_path = current_dir / "temp_startMonitoring.scpt"
    with open(temp_script_path, "w") as file:
        file.write(applescript)

    # Running the AppleScript
    process = subprocess.run(
        ["osascript", temp_script_path], capture_output=True, text=True
    )
    # Getting the output
    stderr = process.stderr

    # Check if there was an error
    if process.returncode != 0:
        print(f"Error executing script: {stderr}")

    # Clean up the temporary file
    temp_script_path.unlink()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        task = sys.argv[1]
        if task == "disp_data":
            disp_data()
        elif task == "disp_servers":
            disp_servers()
        elif task == "disp_jobs":
            disp_jobs()
        else:
            raise ValueError(
                f"No such task {task}. The options are disp_<data/servers/jobs>"
            )
    else:
        # disp_jobs()
        run_script()

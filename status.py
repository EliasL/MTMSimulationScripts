import subprocess
from pathlib import Path
from Management.clusterStatus import get_all_server_info, display_server_info
from Management.dataManager import DataManager
from Management.connectToCluster import Servers
from Management.simulationStatus import print_status_table


# from Management.configGenerator import ConfigGenerator
from Management.jobManager import JobManager


def disp_data():
    dm = DataManager()
    # dm.clean_projects_on_servers()
    # dm.clean_projects_on_servers()
    # dm.delete_useless_dumps(False)
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
    j.findAndShowProcesses()


def checkStatus(configs, labels=None, force_update=False, check_running=False, search_remote="conditional"):
    """
    Print simulation progress from CSV files.

    search_remote="conditional" checks cached/local CSVs first and only queries
    servers for missing or unfinished runs. Use False for local-only and True
    to query servers for every config. Set check_running=True to scan servers
    for live processes.
    """
    if isinstance(labels, bool):
        old_force_update = labels
        labels = None
        force_update, check_running, search_remote = (
            old_force_update,
            force_update,
            check_running,
        )

    print_status_table(
        configs,
        force_update=force_update,
        check_running=check_running,
        search_remote=search_remote,
        labels=labels,
    )


def run_script():
    # Path to your .scpt file
    script_path = f"{Path(__file__).resolve().parent}/Management/startMonitoring.scpt"

    # Get the current directory
    current_dir = Path(__file__).resolve().parent

    # Path to the virtual environment Python
    venv_python_path = current_dir / ".venv" / "bin" / "python"

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
                f"No such task {task}. The options are disp_<data/servers/jobs> or sylvain_status"
            )
    else:
        from Management.jobs import sylvainBatches, size_scaling_job
        # dm = DataManager()
        # dm.clean_projects_on_servers()
        # disp_jobs()
        # disp_servers()
        # configs = []
        # labels = []
        # for batch in [-2, -1]:
        #     batch_configs, batch_labels = sylvainBatches(batch)
        #     configs.extend(batch_configs)
        #     labels.extend(f"batch={batch}, {label}" for label in batch_labels)
        configs, labels = size_scaling_job()
        checkStatus(
            configs,
            labels=labels,
        )


        # This is where you create a terminal with all three displayed
        #run_script()

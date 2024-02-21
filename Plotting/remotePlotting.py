import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from makeAveragePlot import make_average_plot

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / 'Management'))
# Now we can import from Management
from connectToCluster import connectToCluster, Servers, get_server_short_name
from configGenerator import ConfigGenerator

def handleLocalPath(dataPath, configs):
    local_data_folder_name = "MTS2D_output"
    names = [config.generate_name(False) for config in configs]
    
    existing_paths = []  # This will store the paths to existing data files
    base_path = os.path.join(dataPath, local_data_folder_name)
    
    for name in names:
        # Construct the path to the specific data folder for this configuration
        folder_path = os.path.join(base_path, name)
        # Construct the path to the macroData.csv file within the data folder
        file_path = os.path.join(folder_path, "macroData.csv")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # If it exists, add its path to the list of existing paths
            existing_paths.append(file_path)
    
    return existing_paths


def get_csv_from_server(server, configs):
    if "espci.fr" not in server: 
        # server is actually not a ssh address, but a local path
        return handleLocalPath(server, configs)

    # Connect to the server
    ssh = connectToCluster(server, False)

    # Check if /data2 exists, otherwise use /data
    stdin, stdout, stderr = ssh.exec_command("if [ -d /data2 ]; then echo '/data2'; else echo '/data'; fi")
    base_dir = stdout.read().strip().decode()

    user="elundheim"
    data_path = os.path.join(base_dir, user)

    remote_folder_name = "MTS2D_output"

    # List all folders within the output folder
    command = f"cd /{data_path}/{remote_folder_name}; ls -d */"
    stdin, stdout, stderr = ssh.exec_command(command)
    folders = stdout.read().strip().decode().split('\n')
    folders = [folder.rstrip('/') for folder in folders]  # Clean up folder names
    
    names = [config.generate_name(False) for config in configs]

    newPaths = []
    sftp = ssh.open_sftp()
    folder_path = "/tmp/MTS2"
    os.makedirs(folder_path, exist_ok=True)  # This line ensures the MTS2 folder is created
    for name in names:
        #name = config.generate_name(withExtension=False)
        if name in folders:
            remote_file_path = f"{data_path}/{remote_folder_name}/{name}/macroData.csv"
            local_file_path = os.path.join(folder_path, f"{name}.csv")  # Temporary path on the local PC
            sftp.get(remote_file_path, local_file_path)  # Download the file
            newPaths.append(local_file_path)

    return newPaths

# This function searches all the servers for the given config file,
# downloads the csv file associated with the config file to a temp file,
# and returns the new local path to the csv
def get_csv_files(configs):
    newPaths = []
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    with ThreadPoolExecutor(max_workers=len(Servers.serversAndLocal)) as executor:
        future_to_server = {executor.submit(get_csv_from_server, server, configs): server for server in Servers.serversAndLocal}
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            try:
                paths = future.result()
                if paths:
                    newPaths.append(paths)
            except Exception as exc:
                print(f'{server} generated an exception: {exc}')
    # Flatten the paths
    newPaths = [path for sublist in newPaths for path in sublist]
    return newPaths

if __name__ == "__main__":

    seeds = range(0,10)
    configs = ConfigGenerator.generate_over_seeds(seeds, nx=100, ny=100, startLoad=0.15, 
                            loadIncrement=0.00001, maxLoad=1, nrThreads=1) 
    paths = get_csv_files(configs)
    if paths:
        make_average_plot("seeds", paths)    
    else:
        print("No files found")
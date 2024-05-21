import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading

from makePlots import makeEnergyPlot
# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / 'Management'))
# Now we can import from Management
from connectToCluster import connectToCluster, Servers
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

# Shared variables
completed_servers = 0
nr_files = 0
lock = threading.Lock()  # Create a lock for thread-safe operations

def update_progress(total_files):
    with lock:  # Acquire lock before modifying shared variables
        global completed_servers, nr_files
        sys.stdout.write(f"\r{completed_servers}/{len(Servers.servers)} servers, {nr_files}/{total_files} files")
        sys.stdout.flush()


def get_csv_from_server(server, configs):
    global nr_files
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
    folder_path = "/tmp/MTS2D"
    os.makedirs(folder_path, exist_ok=True)  # This line ensures the MTS2D folder is created
    nr_files=0
    # Using ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_name = {
            executor.submit(download_file, name, folders, data_path, remote_folder_name, folder_path, ssh): name
            for name in names
        }
        for future in as_completed(future_to_name):
            result = future.result()
            if result:
                newPaths.append(result)
                with lock:  # Safe update
                    nr_files += 1
                update_progress(len(names))  

    return newPaths

def download_file(name, folders, data_path, remote_folder_name, folder_path, ssh):
    if name in folders:
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                sftp = ssh.open_sftp()
                remote_file_path = f"{data_path}/{remote_folder_name}/{name}/macroData.csv"
                local_file_path = os.path.join(folder_path, f"{name}.csv")
                sftp.get(remote_file_path, local_file_path)
                sftp.close()
                return local_file_path
            except Exception as e:
                attempts += 1
                time.sleep(random.uniform(1, 3))  # Random delay to prevent synchronized reconnection attempts
                print(f"Attempt {attempts} failed for {name}: {e}")
                if attempts >= max_attempts:
                    print(f"Error downloading {name}: {e}")
    return None

def search_for_cvs_files(configs, useOldFiles=False):
    # We also only include files that are less than an hour old

    paths = []
    remaining_configs = []
    last_search_folder = False
    # Folders to search in 
    search_folders = ["/tmp/MTS2D"]
    for folder in search_folders:
        if folder == search_folders[-1]:
            last_search_folder = True

        # Get all files in folder (without extension)
        files = [os.path.splitext(f)[0] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for config in configs:
            file_path = os.path.join(folder, config.generate_name(False) + '.csv')
            if config.generate_name(False) in files:
                # Check if file is less than one hour old
                file_mod_time = os.path.getmtime(file_path)
                if time.time() - file_mod_time < 3600 or useOldFiles:  # 3600 seconds = 1 hour
                    paths.append(file_path)
                else:
                    # We don't want old files, so we redo everything
                    return [], configs
            elif last_search_folder:
                remaining_configs.append(config)
                
    return paths, remaining_configs


# This function searches all the servers for the given config file,
# downloads the csv file associated with the config file to a temp file,
# and returns the new local path to the csv
def get_csv_files(configs, useOldFiles=False):
    global completed_servers

    # First check if the files have already been downloaded
    paths, configs = search_for_cvs_files(configs, useOldFiles)
    if len(configs)==0:
        print("All files already downloaded.")
        return paths
    elif len(paths)!=0:
        print(f"{len(paths)} files found, searching for the remaining {len(configs)}.")

    # Second check local path to see if we can avoid checking the servers
    localPaths = get_csv_from_server(Servers.local_path_mac, configs)
    if len(localPaths) == len(configs):
        # We have found all the requested files, so we don't need to search more.
        print(f"{len(localPaths)} files found. Not searching servers.")
        return paths+localPaths

    print("Searching servers for files...")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
        future_to_server = {executor.submit(get_csv_from_server, server, configs): server for server in Servers.servers}
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            with lock:
               completed_servers += 1  # Increment completed count
            update_progress(len(configs))  
            try:
                server_paths = future.result()
                if server_paths:
                    # We extend, not append
                    paths += server_paths
            except Exception as exc:
                print(f'\n{server} generated an exception: {exc}')
                print("Trying to use old files... ")
                if useOldFiles==False:
                    return get_csv_files(configs, useOldFiles=True)
    print('') # New line from progress indicator

    return paths

if __name__ == "__main__":

    seeds = range(0,10)
    configs = ConfigGenerator.generate_over_seeds(seeds, rows=100, cols=100, startLoad=0.15, 
                            loadIncrement=0.00001, maxLoad=1, nrThreads=1) 
    paths = get_csv_files(configs)
    if paths:
        makeEnergyPlot(paths, "seeds")    
    else:
        print("No files found")
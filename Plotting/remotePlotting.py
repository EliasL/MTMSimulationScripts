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



def get_csv_from_server(server, configs):
    # Connect to the server
    ssh = connectToCluster(server, False)

    # Check if /data2 exists, otherwise use /data
    stdin, stdout, stderr = ssh.exec_command("if [ -d /data2 ]; then echo '/data2'; else echo '/data'; fi")
    base_dir = stdout.read().strip().decode()

    user="elundheim"
    data_path = os.path.join(base_dir, user)

    folder_name = "MTS2D_output"

    # List all folders within the output folder
    command = f"cd /{data_path}/{folder_name}; ls -d */"
    stdin, stdout, stderr = ssh.exec_command(command)
    folders = stdout.read().strip().decode().split('\n')
    folders = [folder.rstrip('/') for folder in folders]  # Clean up folder names
    

    names = ['s100x100l0.15,1e-05,0.7t1n0.05m10s0',
             's100x100l0.15,1e-05,0.7t1n0.05m1s0',
             's100x100l0.15,1e-05,0.7t1n0.05m3s0',
             's100x100l0.15,1e-05,0.7t1n0.05m5s0',
             's100x100l0.15,1e-05,0.7t1n0.05m7s0']
    names = [config.generate_name(False) for config in configs]

    newPaths = []
    sftp = ssh.open_sftp()
    for name in names:
        #name = config.generate_name(withExtension=False)
        if name in folders:
            remote_file_path = f"{data_path}/{folder_name}/{name}/macroData.csv"
            local_file_path = os.path.join("/tmp/MTS2", f"{name}.csv")  # Temporary path on the local PC
            sftp.get(remote_file_path, local_file_path)  # Download the file
            newPaths.append(local_file_path)

    return newPaths

# This function searches all the servers for the given config file,
# downloads the csv file associated with the config file to a temp file,
# and returns the new local path to the csv
def get_csv_files(configs):
    newPaths = []
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
        future_to_server = {executor.submit(get_csv_from_server, server, configs): server for server in Servers.servers}
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
    if newPaths:
        make_average_plot(newPaths)    
    else:
        print("No files found")

if __name__ == "__main__":

    seeds = range(0,10)
    configs = ConfigGenerator.generate_over_seeds(seeds, nx=100, ny=100, startLoad=0.15, 
                            loadIncrement=0.0001, maxLoad=1, nrThreads=4) 
    get_csv_files(configs)
import re
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from connectToCluster import connectToCluster, Servers, get_server_short_name
from tabulate import tabulate
import subprocess
import os
from tqdm import tqdm


"""
Search through all the servers and identify all the data in all the servers
"""


class DataManager:
    def __init__(self) -> None:
        self.data = {}
        self.user = "elundheim"

    def find_data_on_server(self, server):
        # Connect to the server
        ssh = connectToCluster(server, False)

        # Check if /data2 exists, otherwise use /data
        stdin, stdout, stderr = ssh.exec_command(
            "if [ -d /data2 ]; then echo '/data2'; else echo '/data'; fi"
        )
        base_dir = stdout.read().strip().decode()

        data_path = os.path.join(base_dir, self.user)

        # Navigate to the base_dir and get the first folder name
        command = f"cd /{data_path}; ls -d */ | head -n 1"
        stdin, stdout, stderr = ssh.exec_command(command)
        folder_name = stdout.read().strip().decode().rstrip("/")

        # If there is no data, we can return nothing now
        if folder_name == "":
            return []

        # Warning if the folder is not MTS2D_output
        if folder_name != "MTS2D_output":
            print(
                f"Warning: The folder in {data_path} on {server} is not called MTS2D_output. Found: {folder_name}"
            )

        # List all folders within the output folder
        fullPath = f"{data_path}/{folder_name}"
        command = f"cd {fullPath}; ls -d */"
        stdin, stdout, stderr = ssh.exec_command(command)
        folders = stdout.read().strip().decode().split("\n")
        folders = [
            os.path.join(fullPath, folder.rstrip("/")) for folder in folders
        ]  # Clean up folder names

        dataSize = [get_directory_size(ssh, folder) for folder in folders]

        # Save the list of folders and sizes in data dictionary
        return folders, dataSize

    def find_data_on_disk(self, path):
        # Construct the path to the MTS2D_output directory
        mts_output_path = os.path.join(path, "MTS2D_output")

        # Check if MTS2D_output exists
        if not os.path.isdir(mts_output_path):
            raise FileNotFoundError(f"The directory {mts_output_path} does not exist.")

        # List all folders within the MTS2D_output folder
        folders = next(os.walk(mts_output_path))[1]
        folders = [
            os.path.join(mts_output_path, folder) for folder in folders
        ]  # Clean up folder names

        dataSize = [get_directory_size(None, folder) for folder in folders]

        # Save the list of folders and sizes in data dictionary
        return folders, dataSize

    def findData(self):
        # Use ThreadPoolExecutor to execute find_data_on_server in
        # parallel across all servers plus one to find the data stored locally
        with ThreadPoolExecutor(max_workers=len(Servers.servers) + 1) as executor:
            future_to_server = {
                executor.submit(self.find_data_on_server, server): server
                for server in Servers.servers
            }
            future_to_server[
                executor.submit(self.find_data_on_disk, "/Volumes/data")
            ] = "Local ssd"

            # Wrap the as_completed method with tqdm for progress indication
            progress_bar = tqdm(
                as_completed(future_to_server),
                total=len(future_to_server),
                desc="Gathering data",
            )
            for future in progress_bar:
                server = future_to_server[future]
                try:
                    folders_and_sizes = future.result()
                    if folders_and_sizes:
                        self.data[server] = folders_and_sizes
                except Exception as exc:
                    print(f"{server} generated an exception: {exc}")

    def printData(self):
        table_data = []
        for server, folders_and_sizes in self.data.items():
            if folders_and_sizes:  # If there are folders and sizes
                grouped_folders = self.parse_and_group_seeds(folders_and_sizes)
                if grouped_folders:
                    folders, sizes = zip(*grouped_folders)
                    server = get_server_short_name(server)
                    table_data.append([server, "\n".join(folders), "\n".join(sizes)])

        # Displaying the table with a separator between servers
        table = tabulate(
            table_data, headers=["Server", "Folders", "Sizes"], tablefmt="grid"
        )
        print(table)

    def delete_all_found_data(self, dryRun=True):
        for server, (folders, size) in self.data.items():
            if folders:
                self.delete_data_on_server(server, folders, dryRun)

    def delete_data_from_configs(self, configs, dryRun=True):
        names = {x.name for x in configs}  # Use a set for faster lookup
        for server, foldersAndSizes in self.data.items():
            if foldersAndSizes:
                folders, sizes = foldersAndSizes
                filteredFolders = [f for f in folders if f.split("/")[-1] in names]
                if filteredFolders:
                    self.delete_data_on_server(server, filteredFolders, dryRun)

    def delete_data_on_server(self, server, folders, dryRun=True):
        if server == "Local ssd":  # Check if the server is local
            if not dryRun:
                for folder in folders:
                    try:
                        shutil.rmtree(folder)  # Use shutil to delete local directories
                        print(f"Successfully deleted {folder} on local SSD")
                    except Exception as e:
                        print(f"Error deleting {folder} on local SSD: {e}")
        else:
            try:
                ssh = connectToCluster(server, False)
            except Exception as e:
                print(f"Error connecting to {server}: {e}")
            print(f"Are you sure you want to delete these folders on {server}?")
            self.print_grouped_folders(folders)
            if input("yes/no: ") != "yes":
                return

            for folder in folders:
                delete_command = f"rm -r {folder}"
                if not dryRun:
                    stdin, stdout, stderr = ssh.exec_command(delete_command)
                    errors = stderr.read().decode().strip()
                    if errors:
                        print(f"Error deleting {folder} on {server}: {errors}")
                    else:
                        print(f"Successfully deleted {folder} on {server}")
                else:
                    print("Not deleting due to DryRun.")

    def delete_folders_below_size(self, min_size_in_bytes, dryRun=True):
        for server, (folders, sizes) in self.data.items():
            if folders:
                small_folders = []
                small_sizes = []
                for folder, size in zip(folders, sizes):
                    if parse_size(size) < min_size_in_bytes:
                        small_folders.append(folder)
                        small_sizes.append(
                            size
                        )  # Track sizes corresponding to small_folders

                if small_folders:  # Check to ensure there are indeed folders to delete
                    print(
                        f"Folders to delete on {server} because they are smaller than {min_size_in_bytes} bytes:"
                    )
                    for folder, size in zip(small_folders, small_sizes):
                        print(f"{folder} - {size} - {parse_size(size)}")
                        if not dryRun:
                            self.delete_data_on_server(server, [folder], dryRun=False)

    def print_grouped_folders(self, folders, sizes=None):
        if sizes is None:
            sizes = ["0BB/1BB (0.0%)"] * len(folders)
        groups = self.parse_and_group_seeds((folders, sizes))
        for group in groups:
            print(group[0])

    def parse_and_group_seeds(self, folders_and_sizes):
        folder_paths, sizes = folders_and_sizes
        # Regex to match the base part of the folder name and the seed
        pattern = re.compile(r"(.*)s(\d+)$")

        # Parse folder names into base names and seeds
        parsed_folders = []
        for folderPath, size in zip(folder_paths, sizes):
            folder = folderPath.split("/")[-1]
            match = pattern.match(folder)
            if match:
                base_name, seed = match.groups()
                parsed_folders.append((base_name, int(seed), folder, size))

        # Sort by base name and seed to ensure correct grouping
        parsed_folders.sort(key=lambda x: (x[0], x[1]))

        grouped_folders = {}
        # Group by base name
        for base_name, group in groupby(parsed_folders, key=lambda x: x[0]):
            # Extract and sort seeds within each base name group
            seeds = [item for item in group]
            # Group consecutive seeds and calculate total size
            grouped_seeds = []
            for k, g in groupby(enumerate(seeds), lambda i_x: i_x[0] - i_x[1][1]):
                seq = list(g)
                grouped_size = sum_folder_sizes(
                    [item[1][3] for item in seq]
                )  # Calculate total size for the group
                if len(seq) > 1:
                    start, end = seq[0][1], seq[-1][1]
                    grouped_seeds.append((f"{start[1]}-{end[1]}", grouped_size))
                else:
                    single = seq[0][1]
                    grouped_seeds.append((str(single[1]), grouped_size))
            # Reconstruct the folder names for each base name group and associate sizes
            for seed_group, size in grouped_seeds:
                original_folder = seeds[0][2].rsplit("s", 1)[
                    0
                ]  # Get one example folder and remove seed
                grouped_folder = f"{original_folder}s{seed_group}"
                if base_name in grouped_folders:
                    grouped_folders[base_name].append((grouped_folder, size))
                else:
                    grouped_folders[base_name] = [(grouped_folder, size)]

        # Flatten the grouped_folders dictionary to a list and include sizes
        final_grouped_folders = []
        for base_name, folders in grouped_folders.items():
            for folder, size in folders:
                final_grouped_folders.append((folder, size))

        # Sort the folders by size, largest first
        final_grouped_folders.sort(key=lambda x: parse_size(x[1]), reverse=True)

        return final_grouped_folders

    def clean_projects_on_servers(self):
        """
        Deletes the simulation folder in the home directory on all the servers.
        """

        def clean_folder_on_server(server):
            """
            Connects to the server and deletes the simulation folder.
            """
            try:
                ssh = connectToCluster(server, False)
                command = "rm -rf ~/simulation"  # Adjust the path as needed
                ssh.exec_command(command)
                print(f"Successfully cleaned simulation folder on {server}")
            except Exception as e:
                print(f"Error cleaning simulation folder on {server}: {e}")

        # Use ThreadPoolExecutor to execute clean_folder_on_server in parallel across all servers
        with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
            future_to_server = {
                executor.submit(clean_folder_on_server, server): server
                for server in Servers.servers
            }
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    future.result()  # We're just checking for exceptions here
                except Exception as exc:
                    print(f"{server} generated an exception: {exc}")


def get_directory_size(ssh, path):
    if ssh is None:  # Local directory path
        du_command = f"du -sh {path}"
        df_command = f"df -h {path} | awk 'NR==2{{print $4}}'"

        # Execute the du command locally for directory size
        du_process = subprocess.run(
            ["du", "-sh", path], stdout=subprocess.PIPE, text=True
        )
        du_output = du_process.stdout.strip()

        # Execute the df command locally for free disk space
        df_process = subprocess.run(
            ["df", "-H", path], stdout=subprocess.PIPE, text=True
        )
        df_output_lines = df_process.stdout.split("\n")
        for line in df_output_lines:
            if line:
                columns = line.split()
                # Assuming the 'Mounted on' is the last column
                mount_point = columns[-1]
                if path.startswith(mount_point):
                    # Found the matching mount point, return the whole line or specific data
                    free_space = columns[3]

        def convert_gb_to_tb(size_str):
            # Match the numeric part and the unit (G)
            match = re.match(r"(\d+)(G)", size_str)
            if match:
                size_gb = int(match.group(1))
                # Convert GB to TB, keeping one decimal place
                size_tb = round(size_gb / 1024, 1)
                return f"{size_tb}T"
            return size_str

        # Mac chooses 3990GB over switching to TB which was very annoying
        free_space = convert_gb_to_tb(free_space)

    else:  # SSH connection object
        du_command = f"du -sh {path}"
        df_command = f"df -h {path} | awk 'NR==2{{print $4}}'"

        # Execute the du command via SSH for directory size
        stdin, stdout, stderr = ssh.exec_command(du_command)
        du_output = stdout.read().decode("utf-8").strip()
        du_error = stderr.read().decode("utf-8").strip()
        if du_error:
            print(f"Error calculating directory size: {du_error}")
            return None

        # Execute the df command via SSH for free disk space
        stdin, stdout, stderr = ssh.exec_command(df_command)
        df_output = stdout.read().decode("utf-8").strip()
        df_error = stderr.read().decode("utf-8").strip()
        if df_error:
            print(f"Error getting free disk space: {df_error}")
            return None
        free_space = df_output

    # Extract the size part from du output
    size = du_output.split("\t")[0]
    frac = f"{size}B/{free_space}B"
    return f"{frac} ({round(calculate_fraction_percentage(frac),1)}%)"


def parse_unit(unit):
    """Convert unit to the corresponding number of bytes."""
    size_map = {
        "BB": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }
    return size_map.get(unit.upper(), 1)


def calculate_fraction_percentage(input_str):
    """Calculate the fraction as a percentage with unit conversions."""
    # Split the input string by the slash '/'
    first, second = input_str.split("/")

    # Extract numbers and units from both parts
    num1, unit1 = float(first[:-2]), first[-2:]
    num2, unit2 = float(second[:-2]), second[-2:]

    # Convert units to bytes
    bytes1 = num1 * parse_unit(unit1)
    bytes2 = num2 * parse_unit(unit2)

    # Calculate the fraction as a percentage
    fraction = (bytes1 / bytes2) * 100

    return fraction


def convert_to_bytes(value, unit):
    """Convert a value with a unit to bytes."""
    return value * parse_unit(unit)


def parse_size(size_str):
    """
    size_str (str): The size string, e.g., "18MB/3.9TB (0.0%)"

    Returns:
    int: The size in bytes.
    """
    match = re.match(r"(\d+(?:\.\d+)?)(B|KB|MB|GB|TB|PB)", size_str.split("/")[0])
    if match:
        used_size, unit = match.groups()
        used_size = float(used_size) * parse_unit(unit)
        return int(used_size)
    else:
        raise (Exception("Units not found"))


def sum_folder_sizes(str_list):
    # TODO does not work
    # Maybe it does work?
    # Initialize total bytes
    total_bytes = 0
    denominator = ""  # To store the common denominator part for later use

    for s in str_list:
        # Extract numerator and denominator
        numerator, denominator = s.split("/")[0], s.split("/")[1]
        numerator_value, numerator_unit = (
            float(numerator[:-2]),
            numerator[-2:],
        )  # Adjusted to handle 'B'

        # Convert numerator to bytes and add to total
        total_bytes += convert_to_bytes(numerator_value, numerator_unit)

    # Assuming denominator is always the same for all items, use the last one to format the result
    _, denominator_unit = (
        float(denominator.split(" ")[0][:-2]),
        denominator.split(" ")[0][-2:],
    )
    denominator = denominator.split(" ")[0]

    # Convert total bytes back to the largest possible unit while maintaining the original unit of the denominator
    # for consistency in the representation
    units = ["BB", "KB", "MB", "GB", "TB", "PB"]
    # Get the index of the unit to convert back to the same or smaller unit
    unit_index = units.index(denominator_unit)

    # Start from the denominator's unit and go down to find a suitable unit
    for i in range(unit_index, -1, -1):
        if total_bytes >= parse_unit(units[i]):
            total_value = total_bytes / parse_unit(units[i])
            total_unit = units[i]
            break
    else:
        total_value = total_bytes  # If no suitable unit found, use bytes
        total_unit = "BB"

    # Format the sum with the same denominator

    frac = f"{total_value:.0f}{total_unit}/{denominator}"
    return f"{frac} ({round(calculate_fraction_percentage(frac),1)}%)"


if __name__ == "__main__":
    from configGenerator import ConfigGenerator

    dm = DataManager()
    # dm.clean_projects_on_servers()
    configs, labels = ConfigGenerator.generate(
        seed=range(40),
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=[1e-5, 4e-5, 1e-4, 2e-4],
        maxLoad=1.0,
        LBFGSEpsg=[1e-4, 5e-5, 1e-5, 1e-6],
        scenario="simpleShear",
    )
    dm.findData()
    # dm.clean_projects_on_servers()
    # dm.delete_data_from_configs(configs)
    # dm.printData()
    # dm.delete_folders_below_size(1e8)
    # dm.delete_all_found_data()

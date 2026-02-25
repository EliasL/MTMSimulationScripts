import re
from itertools import groupby
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from .connectToCluster import connectToCluster, Servers, get_server_short_name
from .runOnCluster import run_remote_script
from tabulate import tabulate
import subprocess
import os
from tqdm import tqdm
import json
from datetime import datetime, timedelta

"""
Search through all the servers and identify all the data in all the servers
"""


class DataManager:
    # Construct the path to 'data.json' in the same directory as this script
    dataPath = os.path.join(os.path.dirname(__file__), "data.json")

    def __init__(self) -> None:
        self.data = {}
        self.user = "elundheim"

        # Check if the file exists and load the data
        if os.path.exists(DataManager.dataPath):
            with open(DataManager.dataPath, "r") as f:
                self.data = json.load(f)

    def save_data(self):
        # Save the data to 'data.json'
        with open(DataManager.dataPath, "w") as f:
            json.dump(self.data, f)

    def parse_approximate_data_lines(self, lines):
        # Prepare lists to hold paths and sizes
        folders = []
        sizes = []
        free_space = {}

        if isinstance(lines, list):
            lines_list = lines
        else:
            lines_list = lines.split("\n")

        # Process each line
        for line in lines_list:
            # Split by tab to separate the path and the size
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] == "FREE":
                try:
                    free_space[parts[1]] = float(parts[2])
                except ValueError:
                    pass
                continue
            if len(parts) == 2:  # Make sure the line is properly formatted
                folders.append(parts[0])  # The first part is the folder path
                sizes.append(
                    int(parts[1])
                )  # The second part is the size, converted to int
        # The last numeric line is the free space in GB (fallback)
        free_space_in_gb = -1
        if not free_space:
            for line in reversed(lines_list):
                if not line:
                    continue
                try:
                    free_space_in_gb = float(line.strip())
                    break
                except ValueError:
                    continue
        return folders, sizes, (free_space if free_space else free_space_in_gb)

    def find_data_on_server(self, server, silent):
        remote_script_path = (
            "~/simulation/SimulationScripts/Management/approximateData.py"
        )
        lines = run_remote_script(server, remote_script_path, silent)
        if len(lines) == 0:
            return None

        return self.parse_approximate_data_lines(lines)

    def find_data_on_disk(self, path):
        # Define the path to your approximateData.py script
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # __file__ refers to the current script file

        # Define the relative path to your approximateData.py script
        relative_script_path = "approximateData.py"

        # Construct the full path by combining the base_dir and the relative path
        local_script_path = os.path.join(base_dir, relative_script_path)

        # Run the script using subprocess and capture the output
        try:
            result = subprocess.run(
                ["python3", local_script_path, path],  # Pass the path as an argument
                text=True,  # Get the output as a string
                capture_output=True,  # Capture stdout and stderr
                check=True,  # Raise an error if the script fails
            )
            # Split the result into lines
            lines = result.stdout.strip().split("\n")

            if len(lines) == 0:
                return None
            return self.parse_approximate_data_lines(lines)

        except subprocess.CalledProcessError as e:
            # Print detailed error information including stdout and stderr
            print(f"Error running script: {e}")
            print(f"Script output (stdout): {e.stdout}")
            print(f"Script error (stderr): {e.stderr}")
            return None

    def findData(self, silent=True, autoUpdate=False):
        print("Collecting data from servers...")
        if autoUpdate:
            # If this is an autoupdate, we don't want to check for new data
            # if it is less than 24 hours since the last time the data was updated.
            if "date" in self.data:
                last_update_time = datetime.fromisoformat(self.data["date"])
                time_difference = datetime.now() - last_update_time
                if time_difference < timedelta(hours=12):
                    # If the data was updated less than 12 hours ago, return None
                    return None

        # Use ThreadPoolExecutor to execute find_data_on_server in
        # parallel across all servers plus one to find the data stored locally
        with ThreadPoolExecutor(max_workers=len(Servers.servers) + 1) as executor:
            futures_to_server = {
                executor.submit(self.find_data_on_server, server, silent): server
                for server in Servers.servers
            }
            if os.path.isdir("/Volumes/data/MTS2D_output"):
                futures_to_server[
                    executor.submit(self.find_data_on_disk, "/Volumes/data")
                ] = "Local ssd"

            # Wait for all futures to complete
            for future in as_completed(futures_to_server):
                server = futures_to_server[future]
                folders_and_sizes = future.result()
                if folders_and_sizes:
                    self.data[server] = folders_and_sizes
                try:
                    pass
                except Exception as exc:
                    print(f"{server} generated an exception: {exc}")
        self.data["date"] = datetime.now().isoformat()
        self.save_data()

    def printData(self):
        table_data = []

        data_copy = self.data.copy()
        date = data_copy.pop("date", None)  # Use .pop to remove 'date' safely
        print(date)
        # Iterate over the remaining data after removing 'date'
        for server, (folders, sizes, free_space_in_GB) in data_copy.items():
            if folders:  # If there are folders and sizes
                # Track data usage per mount before grouping alters names
                usage_by_mount = {"d1": 0, "d2": 0}
                for folder_path, size in zip(folders, sizes):
                    if folder_path.startswith("/data2"):
                        usage_by_mount["d2"] += size
                    elif folder_path.startswith("/data"):
                        usage_by_mount["d1"] += size

                grouped_folders = self.parse_and_group_seeds((folders, sizes))
                if grouped_folders:
                    folders, sizes = zip(*grouped_folders)
                    sizes = [f"{nr}{unit}" for nr, unit in sizes]
                    server = get_server_short_name(server)
                    server_lines = [f"{server}"]
                    def fmt_tb_from_gb(value_gb):
                        return f"{value_gb / 1024:.1f}TB"
                    def fmt_tb_from_bytes(value_bytes):
                        return f"{value_bytes / (1024**4):.1f}TB"
                    if isinstance(free_space_in_GB, dict):
                        d1 = free_space_in_GB.get("/data")
                        d2 = free_space_in_GB.get("/data2")
                        if d1 is not None:
                            server_lines.append(f"(d1) {fmt_tb_from_gb(d1)} free")
                            if usage_by_mount["d1"] > 0:
                                server_lines.append(
                                    f"(d1 me) {fmt_tb_from_bytes(usage_by_mount['d1'])}"
                                )
                        if d2 is not None:
                            server_lines.append(f"(d2) {fmt_tb_from_gb(d2)} free")
                            if usage_by_mount["d2"] > 0:
                                server_lines.append(
                                    f"(d2 me) {fmt_tb_from_bytes(usage_by_mount['d2'])}"
                                )
                        if d1 is None and d2 is None and free_space_in_GB:
                            # Fallback to first available entry
                            key = next(iter(free_space_in_GB.keys()))
                            server_lines.append(
                                f"{fmt_tb_from_gb(free_space_in_GB[key])} free"
                            )
                    else:
                        try:
                            server_lines.append(
                                f"{fmt_tb_from_gb(float(free_space_in_GB))} free"
                            )
                        except Exception:
                            server_lines.append("N/A")
                    table_data.append(
                        [
                            "\n".join(server_lines),
                            "\n".join(folders),
                            "\n".join(sizes),
                        ]
                    )

        # Displaying the table with a separator between servers
        table = tabulate(
            table_data, headers=["Server", "Folders", "Sizes"], tablefmt="grid"
        )
        print(table)

    def delete_all_found_data(self, dryRun=True):
        for server, (folders, sizes, free_space_in_GB) in self.data.items():
            if folders:
                self.delete_data_on_server(server, folders, dryRun)

    def delete_data_from_configs(self, configs, dryRun=True):
        names = {x.name for x in configs}  # Use a set for faster lookup
        somethingFound = False
        for server, foldersAndSizes in self.data.items():
            if foldersAndSizes and not isinstance(foldersAndSizes, str):
                folders, sizes, free_space_in_GB = foldersAndSizes
                filteredFolders = [f for f in folders if f.split("/")[-1] in names]
                if filteredFolders:
                    somethingFound = True
                    self.delete_data_on_server(server, filteredFolders, dryRun)
        if not somethingFound:
            print(
                "No data found to delete. (Did you remember to update the dataManager? Run findData)"
            )

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
            if not dryRun and input("yes/no: ") != "yes":
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

    def delete_folders_below_size(self, min_size_in_mega_bytes, dryRun=True):
        for server, (folders, sizes) in self.data.items():
            if folders:
                small_folders = []
                small_sizes = []
                for folder, size in zip(folders, sizes):
                    if size < min_size_in_mega_bytes * 1e6:
                        small_folders.append(folder)
                        small_sizes.append(
                            size
                        )  # Track sizes corresponding to small_folders

                if small_folders:  # Check to ensure there are indeed folders to delete
                    print(
                        f"Folders to delete on {server} because they are smaller than {min_size_in_mega_bytes}MB:"
                    )
                    for folder, size in zip(small_folders, small_sizes):
                        print(f"{folder} - {size} - {bytes_to_readable(size)}")
                    if not dryRun:
                        self.delete_data_on_server(server, small_folders, dryRun=False)

    def delete_useless_dumps(self, dryRun=True):
        """
        Deletes unnecessary dump files in parallel across all servers, keeping only the latest and
        one per each 10% load increment, with a progress bar for all folders.
        """
        with ThreadPoolExecutor() as executor:
            futures = []
            # Summing all folders to set up the total for tqdm
            total_folders = sum(
                len(folders)
                for server, (folders, sizes, free_space) in self.data.items()
                if folders
            )
            progress_bar = tqdm(total=total_folders, desc="Deleting dumps")

            for server, (folders, sizes, free_space) in self.data.items():
                if folders:
                    # Submit a task for each server to the executor
                    future = executor.submit(
                        self._delete_useless_dumps_on_server,
                        server,
                        folders,
                        dryRun,
                        progress_bar,  # Pass the progress bar to each server function
                    )
                    futures.append(future)

            # Wait for all futures to complete
            for future in as_completed(futures):
                pass  # Futures themselves update the progress bar

            progress_bar.close()  # Ensure the progress bar is closed properly

    def _delete_useless_dumps_on_server(self, server, folders, dryRun, progress_bar):
        """
        Process all folders on a given server sequentially, updating the passed tqdm progress bar.
        """
        ssh = connectToCluster(server, False)
        for folder in folders:
            self._delete_useless_dumps_in_folder(
                ssh, os.path.join(folder, "dumps"), dryRun
            )
            progress_bar.update(1)  # Update progress for each folder processed
        ssh.close()  # Ensure the SSH connection is closed after processing all folders

    def _delete_useless_dumps_in_folder(self, ssh, dump_dir, dryRun=True):
        """
        Retrieves and deletes unnecessary dump files while keeping one per each 10% load increment
        and the latest dump.
        """
        # Command to list all files with their modification time, sorted by modification time
        list_files_command = f"ls -lt  {dump_dir} | awk '{{print $9}}'"
        stdin, stdout, stderr = ssh.exec_command(list_files_command)
        dumps = stdout.read().strip().decode().split("\n")

        load_regex = re.compile(
            r"dump_l(\d+(?:\.\d+)?)(?:_[\w\.]+)?\.mstd", re.IGNORECASE
        )
        load_to_files = {}
        latest_file = dumps[0]

        for dump in dumps:
            match = load_regex.search(dump)
            if match:
                load = float(match.group(1))
                load_rounded = round(load, 1)
                if load_rounded not in load_to_files:
                    load_to_files[load_rounded] = []
                load_to_files[load_rounded].append(dump)

        # Determine the files to keep
        dumps_to_keep = {latest_file}  # Initialize with the latest file

        # Keep one file per load group
        for load_group, files in load_to_files.items():
            files.sort()  # Optional, depending on how the filenames are structured
            dumps_to_keep.add(files[0])  # Keep the first file of each group

        # Files to delete are those not in dumps_to_keep
        dumps_to_delete = set(dumps) - dumps_to_keep

        # Execute deletion
        if not dryRun:
            if dumps_to_delete:
                delete_command = f"rm -f {' '.join([os.path.join(dump_dir, dump) for dump in dumps_to_delete])}"
                stdin, stdout, stderr = ssh.exec_command(delete_command)
                errors = stderr.read().decode().strip()
                if errors:
                    print(f"Error deleting files in {dump_dir}: {errors}")
        else:
            print("Dry run enabled. Files to delete:")
            for dump in dumps_to_delete:
                print(dump)
            print("Files to keep:")
            for dump in dumps_to_keep:
                print(" ", dump)

    def print_grouped_folders(self, folders, sizes=None):
        if sizes is None:
            sizes = [0] * len(folders)
        groups = self.parse_and_group_seeds((folders, sizes))
        for group in groups:
            print(group[0])

    def parse_and_group_seeds(self, folders_and_sizes):
        folder_paths, sizes = folders_and_sizes
        # Regex to match the base part of the folder name and the seed
        pattern = re.compile(r"(.*)s(\d+)$")

        # Aggregate sizes per (base_name, seed) to avoid duplicate seeds
        by_base = {}
        has_d1 = any(p.startswith("/data/") or p.startswith("/data") for p in folder_paths)
        has_d2 = any(p.startswith("/data2/") or p.startswith("/data2") for p in folder_paths)
        use_prefix = has_d1 and has_d2
        for folderPath, size in zip(folder_paths, sizes):
            folder = folderPath.split("/")[-1]
            match = pattern.match(folder)
            if not match:
                continue
            base_name, seed_str = match.groups()
            if use_prefix:
                if folderPath.startswith("/data2"):
                    base_name = f"(d2){base_name}"
                elif folderPath.startswith("/data"):
                    base_name = f"(d1){base_name}"
            seed = int(seed_str)
            base_entry = by_base.setdefault(base_name, {})
            base_entry[seed] = base_entry.get(seed, 0) + size

        final_grouped_folders = []
        for base_name, seed_sizes in by_base.items():
            if not seed_sizes:
                continue
            # Sort seeds to group consecutive ranges
            sorted_seeds = sorted(seed_sizes.items(), key=lambda x: x[0])
            start_seed, end_seed = sorted_seeds[0][0], sorted_seeds[0][0]
            current_sizes = [sorted_seeds[0][1]]

            for seed, size in sorted_seeds[1:]:
                if seed == end_seed + 1:
                    end_seed = seed
                    current_sizes.append(size)
                else:
                    grouped_size = sum_folder_sizes(current_sizes)
                    seed_group = (
                        str(start_seed)
                        if start_seed == end_seed
                        else f"{start_seed}-{end_seed}"
                    )
                    grouped_folder = f"{base_name}s{seed_group}"
                    final_grouped_folders.append((grouped_folder, grouped_size))

                    # Reset for next group
                    start_seed = end_seed = seed
                    current_sizes = [size]

            # Finalize the last group
            grouped_size = sum_folder_sizes(current_sizes)
            seed_group = (
                str(start_seed)
                if start_seed == end_seed
                else f"{start_seed}-{end_seed}"
            )
            grouped_folder = f"{base_name}s{seed_group}"
            final_grouped_folders.append((grouped_folder, grouped_size))

        # Sort the folders by size, largest first
        final_grouped_folders.sort(key=lambda x: convert_to_bytes(*x[1]), reverse=True)

        return final_grouped_folders

    def clean_projects_on_servers(self, specificServers=None, onlyPrefered=False):
        """
        Deletes the simulation folder in the home directory on all the servers.
        """
        if onlyPrefered:
            specificServers = Servers.preferedServers

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

        serversToClean = Servers.servers
        if specificServers:
            serversToClean = specificServers
        # Use ThreadPoolExecutor to execute clean_folder_on_server in parallel across all servers
        with ThreadPoolExecutor(max_workers=len(serversToClean)) as executor:
            future_to_server = {
                executor.submit(clean_folder_on_server, server): server
                for server in serversToClean
            }
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    future.result()  # We're just checking for exceptions here
                except Exception as exc:
                    print(f"{server} generated an exception: {exc}")


def get_free_space(ssh, path):
    if ssh is None:  # Local directory path
        df_command = f"df -h {path} | awk 'NR==2{{print $4}}'"

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

    else:  # SSH connection object
        df_command = f"df -h {path} | awk 'NR==2{{print $4}}'"

        # Execute the df command via SSH for free disk space
        stdin, stdout, stderr = ssh.exec_command(df_command)
        df_output = stdout.read().decode("utf-8").strip()
        df_error = stderr.read().decode("utf-8").strip()
        if df_error:
            print(f"Error getting free disk space: {df_error}")
            return None
        free_space = df_output
    return free_space


# UNUSED
def get_directory_size(ssh, path, free_space=False):
    # Command to list all items in the directory with details
    list_details_command = f"ls -l {path}"
    stdin, stdout, stderr = ssh.exec_command(list_details_command)
    entries = stdout.read().strip().decode().split("\n")

    total_size = 0  # to accumulate the total size in kilobytes

    for entry in entries[1:]:  # Skipping the first line which is the total
        if not entry:
            continue
        parts = entry.split()
        if len(parts) < 9:
            print(parts)
            continue  # Skipping if the entry doesn't have enough parts to be valid

        # First character tells if it's a file (-) or directory (d)
        file_type = parts[0][0]
        file_name = parts[-1]  # Last part is the name of the file or directory
        full_path = os.path.join(path, file_name)

        if file_type == "d":  # Directory
            # Get a list of entries in the directory
            list_files_command = f"ls {full_path}"
            stdin, stdout, stderr = ssh.exec_command(list_files_command)
            child_entries = stdout.read().strip().decode().split()
            if child_entries:
                # Get the size of the first file using 'stat' for precision
                first_file_path = os.path.join(full_path, child_entries[0])
                stat_command = f"stat -c %s {first_file_path}"
                stdin, stdout, stderr = ssh.exec_command(stat_command)
                first_file_size = stdout.read().strip().decode()
                if first_file_size.isdigit():
                    first_file_size = int(first_file_size)  # Size in bytes
                    # Approximate total size by multiplying the first file size by the number of files
                    # Convert bytes to kilobytes
                    total_size += (first_file_size * len(child_entries)) / 1024
        else:  # File
            # Just get the size from the ls output which is in parts[4] in bytes
            total_size += int(parts[4]) / 1024  # Converting bytes to kilobytes
            continue

    # Format the size for output
    return format_size(str(total_size) + "K", free_space)


def format_size(size, free_space):
    frac = f"{size}B/{free_space}B"
    return f"{frac} ({round(calculate_fraction_percentage(frac), 1)}%)"


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


def bytes_to_readable(byte_size):
    """
    Convert a byte size into a more readable format with a number between 1 and 999 and the appropriate unit.
    """
    # Define the conversion factors and corresponding units
    units = [
        ("B", 1),
        ("KB", 1024),
        ("MB", 1024**2),
        ("GB", 1024**3),
        ("TB", 1024**4),
        ("PB", 1024**5),
    ]

    # Iterate over the units to find the most suitable one
    for unit_name, unit_value in units:
        readable_number = byte_size / unit_value
        # Check if the converted number is between 1 and 999
        if 0.9 <= readable_number < 1000:
            # Return the number rounded to 2 decimal places and the unit
            return round(readable_number, 2), unit_name

    # If no suitable unit is found (which is unlikely unless byte_size is extremely large), return in petabytes
    return round(byte_size / units[-1][1], 2), units[-1][0]


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


def old_parse_size(size_str):
    """
    size_str (str): The size string, e.g., "18MB/3.9TB (0.0%), or 3.4GB"

    Returns:
    int: The size in bytes.
    """
    if "/" in size_str:
        size_str = size_str.split("/")[0]
    match = re.match(r"(\d+(?:\.\d+)?)(B|KB|MB|GB|TB|PB)", size_str)
    if match:
        used_size, unit = match.groups()
        used_size = float(used_size) * parse_unit(unit)
        return int(used_size)
    else:
        raise (Exception("Units not found"))


def sum_folder_sizes(sizes):
    return bytes_to_readable(sum(sizes))


def sum_folder_sizes_with_fraction(str_list):
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
    return f"{frac} ({round(calculate_fraction_percentage(frac), 1)}%)"

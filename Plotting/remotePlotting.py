import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading
from collections import defaultdict
import pandas as pd
from pandas.errors import ParserError
from .makePlots import (
    makePlot,
    makeAverageComparisonPlot,
    add_power_law_line,
    duration_to_seconds,
)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from .fixLineNumbers import fix_csv_files_in_data_folder
from Management.connectToCluster import getServerUserName
from tqdm import tqdm
import numpy as np
from Plotting.plotPowerLaw import (
    plot_powerlaw,
    plot_plastic_counts_compare,
    plot_plastic_energy_scatter,
    get_group_structure,
)
from Management.updateCSV import fix_csv_files

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Management"))
# Now we can import from Management
from Management.connectToCluster import connectToCluster, Servers, download_folders
from Management.configGenerator import ConfigGenerator, SimulationConfig
from Management.updateCSV import fix_mixed_macrodata_csv

FOLDER_PATH = "/Users/elias/Work/PhD/Code/remoteData"
FOLDER_PATH = "/Users/eliaslundheim/work/PhD/remoteData"
MACRO_PATH = os.path.join(FOLDER_PATH, "macro")
PLOTS_PATH = os.path.join(FOLDER_PATH, "plots")
RAW_DATA_PATH = os.path.join(FOLDER_PATH, "data")

OLD_TO_NEW_KEYS = {
    "Line nr": None,  # Probably an index or redundant
    "Load": "load",
    "Avg energy": "avg_energy",
    "Max energy": "max_energy",
    "Avg RSS": "avg_RSS",
    "Nr plastic deformations": "nr_elements_with_m3_fix_change",
    "Nr FIRE iterations": "nr_iterations",  # Assumed generic for all solvers
    "Nr LBFGS iterations": "nr_iterations",
    "Nr CG iterations": "nr_iterations",
    "Nr FIRE func evals": "nr_func_evals",
    "Nr LBFGS func evals": "nr_func_evals",
    "Nr CG func evals": "nr_func_evals",
    "FIRE Term reason": "FIRE_Term_reason",
    "LBFGS Term reason": "LBFGS_Term_reason",
    "CG Term reason": "CG_Term_reason",
    "Run time": "run_time",
    "Est time remaining": "est_time_remaining",
    "maxX": "maxX",
    "minX": "minX",
    "maxY": "maxY",
    "minY": "minY",
    "dt_start": None,  # No direct equivalent provided
}


def unused_update_headers_in_file(csv_path):
    print("don't use. Use the one in updateCSV.py")
    df = pd.read_csv(csv_path)

    rename_dict = {
        old: new
        for old, new in OLD_TO_NEW_KEYS.items()
        if new is not None and old in df.columns
    }

    df.rename(columns=rename_dict, inplace=True)

    df.to_csv(csv_path, index=False)


def smart_read_csv(file_path):
    df = pd.read_csv(file_path)

    # Check if using old keys by checking the header
    old_keys = set(OLD_TO_NEW_KEYS.keys())
    new_keys = set(
        [
            "load_step",
            "load",
            "avg_energy",
            "avg_energy_change",
            "max_energy",
            "max_force",
            "avg_RSS",
            "nr_elements_with_m3_fix_change",
            "max_plastic_deformation",
            "max_positive_plastic_jump",
            "max_negative_plastic_jump",
            "nr_iterations",
            "nr_func_evals",
            "LBFGS_Term_reason",
            "CG_Term_reason",
            "FIRE_Term_reason",
            "run_time",
            "minimization_time",
            "write_time",
            "est_time_remaining",
            "cmX",
            "cmY",
            "maxX",
            "minX",
            "maxY",
            "minY",
        ]
    )

    if old_keys & set(df.columns):  # old keys present
        # Convert old column names to new ones
        rename_dict = {
            old: new
            for old, new in OLD_TO_NEW_KEYS.items()
            if new is not None and old in df.columns
        }
        df = df.rename(columns=rename_dict)

    # Add missing new keys as NaN
    for key in new_keys:
        if key not in df.columns:
            df[key] = pd.NA

    # Reorder columns (optional)
    df = df[[col for col in new_keys if col in df.columns]]

    return df


def handleLocalPath(dataPath, configs, returnCsv=True):
    local_data_folder_name = "MTS2D_output"
    names = [config.generate_name(False) for config in configs]

    existing_paths = []  # This will store the paths to existing data files
    base_path = os.path.join(dataPath, local_data_folder_name)

    for name in names:
        # Construct the path to the specific data folder for this configuration
        folder_path = os.path.join(base_path, name)
        if returnCsv:
            # Construct the path to the macroData.csv file within the data folder
            file_path = os.path.join(folder_path, "macroData.csv")
        else:
            file_path = folder_path

        # Check if the file exists
        if os.path.exists(file_path):
            # If it exists, add its path to the list of existing paths
            existing_paths.append(file_path)

    # fix_csv_files(existing_paths, use_tqdm=False)
    return existing_paths


# Shared variables
completed_servers = 0
nr_files = 0
lock = threading.Lock()  # Create a lock for thread-safe operations


def update_progress(total_files):
    with lock:  # Acquire lock before modifying shared variables
        global completed_servers, nr_files
        sys.stdout.write(
            f"\r{completed_servers}/{len(Servers.servers)} servers, {nr_files}/{total_files} files"
        )
        sys.stdout.flush()


def delete_double_folders(data_paths, ssh, server):
    """
    TEMP / SAFETY VERSION
    - Looks for remote_folder_name under each data_path
    - Computes folder sizes
    - Prints sizes for all existing folders
    - Shows subfolders of the smallest one
    - Asks for explicit confirmation before deleting
    """
    # Collect existing MTS2D_output dirs and their sizes
    existing_dirs = []

    def human_readable_size(num_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.1f} PB"

    for dp in data_paths:
        # Example: /data/user
        remote_dir = f"/{dp}"
        # Normalize // -> / a bit
        while "//" in remote_dir:
            remote_dir = remote_dir.replace("//", "/")

        # Check if directory exists and get its size in bytes
        check_and_size_cmd = (
            f'if [ -d "{remote_dir}" ]; then '
            f"  du -sb \"{remote_dir}\" 2>/dev/null | awk '{{print $1}}'; "
            f"else "
            f'  echo ""; '
            f"fi"
        )
        stdin, stdout, stderr = ssh.exec_command(check_and_size_cmd)
        out = stdout.read().decode().strip()

        if not out:
            # Directory does not exist on this data path
            print(f"[INFO] {server}:{remote_dir} does not exist, skipping.")
            continue

        try:
            size_bytes = int(out.splitlines()[0])
        except ValueError:
            print(
                f"[WARN] Could not parse size for {server}:{remote_dir} (output: {out!r}). Skipping."
            )
            continue

        existing_dirs.append((remote_dir, size_bytes))

    if len(existing_dirs) < 2:
        print(
            "[INFO] delete_double_folders: fewer than two existing "
            f'"{dp}" folders found on {server}. Nothing to compare.'
        )
        return

    # Print sizes of all candidate folders
    print(f"\n[INFO] Found {len(existing_dirs)} '{dp}' folders on {server}:")
    for path, size in existing_dirs:
        print(f"  {path}: {size} bytes ({human_readable_size(size)})")

    # Pick the smallest folder by size
    existing_dirs.sort(key=lambda x: x[1])
    smaller_dir, smaller_size = existing_dirs[0]

    print(f"\n[INFO] Smallest folder appears to be: {smaller_dir}")
    print(f"  Size: {smaller_size} bytes ({human_readable_size(smaller_size)})")

    # List subfolders of the smaller folder
    list_sub_cmd = (
        f'cd "{smaller_dir}" 2>/dev/null && '
        f'ls -d */ 2>/dev/null || echo "(no subfolders or directory not accessible)"'
    )
    stdin, stdout, stderr = ssh.exec_command(list_sub_cmd)
    subfolders = stdout.read().decode().strip()

    print(f"\n[INFO] Subfolders in {smaller_dir}:")
    print(subfolders if subfolders else "(no subfolders)")

    # Explicit confirmation
    confirmation = (
        input(
            f"\n[CONFIRM] Delete the smaller folder on {server}?\n"
            f"  Folder : {smaller_dir}\n"
            f"  Size   : {smaller_size} bytes ({human_readable_size(smaller_size)})\n"
            f"Type 'yes' to permanently delete this folder: "
        )
        .strip()
        .lower()
    )

    if confirmation != "yes":
        print("[INFO] Deletion aborted by user.")
        return

    # Perform the deletion
    delete_cmd = f'rm -rf "{smaller_dir}"'
    print(f"[ACTION] Executing: {delete_cmd}")
    stdin, stdout, stderr = ssh.exec_command(delete_cmd)
    err = stderr.read().decode().strip()

    if err:
        print(f"[WARN] rm -rf reported on {server}:{smaller_dir}:\n{err}")
    else:
        print(f"[OK] Deleted {smaller_dir} on {server}.")


def get_csv_from_server(server, configs):
    global nr_files

    if server[0] == "/":
        # server is actually not a ssh address, but a local path
        print(f"Searching local data path: {server}")
        return handleLocalPath(server, configs)

    # Connect to the server
    ssh = connectToCluster(server, False)

    user = getServerUserName(server)

    # Discover all base paths (/data and /data2) that contain this user directory
    discover_cmd = (
        f"for base in /data /data2; do "
        f'  if [ -d "$base" ] && [ -d "$base/{user}" ]; then '
        f"    printf '%s\\n' \"$base/{user}\"; "
        f"  fi; "
        f"done"
    )
    stdin, stdout, stderr = ssh.exec_command(discover_cmd)
    data_paths = [
        line.strip() for line in stdout.read().decode().splitlines() if line.strip()
    ]

    if not data_paths:
        raise RuntimeError(
            f"No data directory found for user '{user}' under /data or /data2 on {server}."
        )

    remote_folder_name = "MTS2D_output"
    names = [config.generate_name(False) for config in configs]
    newPaths = []

    # Ensure the local output folder exists
    os.makedirs(MACRO_PATH, exist_ok=True)

    print(f"Searching {server} in {', '.join(data_paths)}.")

        # TEMP PART
        # DANGEROUS DOUBLE CHECK BEFORE MAKING LIVE
        # IF there are two datapaths on the folder, delete the folder with less data
        # Print the size of both folders, then the sub folders of the smaller folder
        # Ask the user if the smaller folder should be deleted
        # Delete the folder
        # delete_double_folders(data_paths, ssh, server)

    # Go through each user directory we found (/data/<user>, /data2/<user>, ...)
    for data_path in data_paths:
        # List all folders within the remote MTS2D output folder for this user
        command = (
            f"cd /{data_path}/{remote_folder_name} 2>/dev/null && "
            f"ls -d */ 2>/dev/null || true"
        )
        stdin, stdout, stderr = ssh.exec_command(command)
        raw = stdout.read().strip().decode()

        if not raw:
            # No output folder or no subfolders for this base path
            # print(f"No folders found in /{data_path}/{remote_folder_name} on {server}")
            continue

        folders = [folder.rstrip("/") for folder in raw.split("\n") if folder]

        # Using ThreadPoolExecutor to download files in parallel for this user path
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_to_name = {
                executor.submit(
                    download_file,
                    name,
                    folders,
                    data_path,
                    remote_folder_name,
                    MACRO_PATH,
                    ssh,
                    # server + data_path.split("/")[1],
                ): name
                for name in names
            }

            for future in as_completed(future_to_name):
                result = future.result()
                if result:
                    print(f"Found {result.split('/')[-1]} on {server}")
                    newPaths.append(result)
                    with lock:  # Safe update
                        nr_files += 1
                    # update_progress(len(names))

    # Apply header updates on all new files
    # for path in newPaths:
    #    unused_update_headers_in_file(path)

    return newPaths


def _merge_candidate_sources(target, source_dict):
    for name, entries in source_dict.items():
        target[name].extend(entries)


def _is_macro_path(path):
    try:
        return os.path.commonpath([MACRO_PATH, path]) == MACRO_PATH
    except ValueError:
        return False


def _scan_local_csv_candidates(configs):
    candidates = defaultdict(list)
    search_folders = ["/tmp/MTS2D", MACRO_PATH]
    for folder in search_folders:
        if not os.path.isdir(folder):
            continue
        for config in configs:
            path = os.path.join(folder, f"{config.name}.csv")
            if os.path.isfile(path):
                candidates[config.name].append((f"local:{folder}", path))

    local_base = getattr(Servers, "local_path_mac", None)
    if local_base:
        base_path = os.path.join(local_base, "MTS2D_output")
        for config in configs:
            path = os.path.join(base_path, config.name, "macroData.csv")
            if os.path.isfile(path):
                candidates[config.name].append((f"local_data:{local_base}", path))
    return candidates


def _scan_remote_csv_candidates(server, configs):
    candidates = defaultdict(list)
    if server[0] == "/":
        return candidates

    ssh = connectToCluster(server, False)
    user = getServerUserName(server)

    discover_cmd = (
        f"for base in /data /data2; do "
        f'  if [ -d "$base" ] && [ -d "$base/{user}" ]; then '
        f"    printf '%s\\n' \"$base/{user}\"; "
        f"  fi; "
        f"done"
    )
    stdin, stdout, stderr = ssh.exec_command(discover_cmd)
    data_paths = [
        line.strip() for line in stdout.read().decode().splitlines() if line.strip()
    ]

    if not data_paths:
        return candidates

    remote_folder_name = "MTS2D_output"
    names = {config.name for config in configs}

    for data_path in data_paths:
        command = (
            f"cd /{data_path}/{remote_folder_name} 2>/dev/null && "
            f"ls -d */ 2>/dev/null || true"
        )
        stdin, stdout, stderr = ssh.exec_command(command)
        raw = stdout.read().strip().decode()
        if not raw:
            continue
        folders = {folder.rstrip("/") for folder in raw.split("\n") if folder}
        matches = names & folders
        if not matches:
            continue
        for name in matches:
            remote_path = (
                f"{server}:{data_path}/{remote_folder_name}/{name}/macroData.csv"
            )
            candidates[name].append((f"remote:{server}", remote_path))

    return candidates


def _format_duplicate_sources(dupes):
    lines = ["Multiple CSV sources found for the same config:"]
    for name, entries in sorted(dupes.items()):
        lines.append(f"{name}.csv is found on:")
        for _, path in entries:
            lines.append(f"  {path}")
    return "\n".join(lines)


def download_file(
    name, folders, data_path, remote_folder_name, folder_path, ssh, newName=""
):
    if name in folders:
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            try:
                sftp = ssh.open_sftp()
                remote_file_path = (
                    f"{data_path}/{remote_folder_name}/{name}/macroData.csv"
                )
                local_file_path = os.path.join(folder_path, f"{newName}{name}.csv")
                sftp.get(remote_file_path, local_file_path)
                sftp.close()
                return local_file_path
            except Exception as e:
                attempts += 1
                time.sleep(
                    random.uniform(1, 3)
                )  # Random delay to prevent synchronized reconnection attempts
                # print(f"Attempt {attempts} failed for {name}: {e}")
                if attempts >= max_attempts:
                    print(f"Error downloading {name}: {e}")
    return None


def search_for_cvs_files(
    configs,
    useOldFiles=False,
    forceUpdate=False,
    debug_download=False,
    fix_files=False,
):
    """
    Searches for CSV files corresponding to given configurations in predefined folders.

    - If `forceUpdate` is True, returns immediately with no files.
    - Only includes files that are less than 12 hours old unless `useOldFiles` is True.
    - Ensures all search directories exist.
    - Optionally fixes mixed-header CSVs when `fix_files` is True.
    - Files are considered valid if their "Est_time_remaining" column is 0 or missing.

    Returns:
        paths (list): List of valid file paths.
        remaining_configs (list): List of configurations still needing files.
    """

    # If forced update, return no files.
    if forceUpdate:
        if debug_download:
            print("forceUpdate=True: skipping local cache scan.")
        return [], configs

    found_paths: dict[str, str] = {}
    remaining_configs = []
    # If an incomplete file is older than x hours, we update it
    updateAfterHours = 12
    search_folders = ["/tmp/MTS2D", MACRO_PATH]  # Directories to search in

    for i, folder in enumerate(search_folders):
        os.makedirs(folder, exist_ok=True)  # Ensure folder exists
        # Get existing CSV file names (without extensions) for quick lookup
        existing_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        }

        for config in configs:
            file_path = os.path.join(folder, f"{config.name}.csv")

            if config.name in existing_files:
                # Read estimated time remaining from CSV file
                file_mod_time = os.path.getmtime(file_path)
                age_ok = time.time() - file_mod_time < updateAfterHours * 3600

                try:
                    if fix_files:
                        fix_mixed_macrodata_csv(file_path, inplace=True)
                    df = pd.read_csv(file_path)
                except Exception as fix_exc:
                    if fix_files:
                        print(f"Failed to fix {file_path}: {fix_exc}")
                    else:
                        print(f"Failed to read {file_path}: {fix_exc}")
                    if age_ok or useOldFiles:
                        if debug_download:
                            age_min = (time.time() - file_mod_time) / 60.0
                            reason = "parse/fix" if fix_files else "parse"
                            print(
                                f"Using local file despite {reason} error (age={age_min:.1f} min): {file_path}"
                            )
                        found_paths[config.name] = file_path
                    continue

                if df.empty:
                    if age_ok or useOldFiles:
                        if debug_download:
                            age_min = (time.time() - file_mod_time) / 60.0
                            print(
                                f"Using local file despite empty CSV (age={age_min:.1f} min): {file_path}"
                            )
                        found_paths[config.name] = file_path
                    continue

                keys = df.keys()
                if "est_time_remaining" in keys:
                    est_time_remaining = df["est_time_remaining"]
                    time_remaining = (
                        duration_to_seconds(est_time_remaining.iloc[-1])
                        if not est_time_remaining.empty
                        else None
                    )
                else:
                    time_remaining = -1

                if time_remaining is None or time_remaining > 0:
                    # File might still be processing; check age
                    if age_ok or useOldFiles:
                        # Include if recent enough
                        if debug_download:
                            age_min = (time.time() - file_mod_time) / 60.0
                            print(
                                f"Using local file (age={age_min:.1f} min, "
                                f"time_remaining={time_remaining}): {file_path}"
                            )
                        found_paths[config.name] = file_path
                    else:
                        # Still needs processing
                        if debug_download:
                            age_hr = (time.time() - file_mod_time) / 3600.0
                            print(
                                f"Skipping local file (age={age_hr:.2f} h, "
                                f"time_remaining={time_remaining}): {file_path}"
                            )
                        continue
                else:
                    # File is done processing
                    if debug_download:
                        print(f"Using local file (complete): {file_path}")
                    found_paths[config.name] = file_path

    remaining_configs = [c for c in configs if c.name not in found_paths]

    if debug_download and remaining_configs:
        preview = ", ".join(c.name for c in remaining_configs[:5])
        suffix = "..." if len(remaining_configs) > 5 else ""
        print(
            f"Local cache miss for {len(remaining_configs)} configs: {preview}{suffix}"
        )

    return list(found_paths.values()), remaining_configs


# Converts config to a path, but if given paths, it matches the given
# config with the path is is most likely to corespond to.
# If a config could match with two paths, the first path found is chosen
def configToPath(config, paths=None):
    if paths is not None:
        # Search for the coresponding path and config
        matches = [path for path in paths if config.name in path]
        if matches:
            return matches[0]
        else:
            return None
    else:
        # Assume it is found in the MACRO_PATH
        return f"{MACRO_PATH}/{config.name}.csv"


def flatToStructure(config_groups, label_groups, found_paths=None):
    # This function searches for where the file WOULD be if it was
    # successfully downloaded, therefore preserving the structure of the groups
    paths = []
    labels = []
    for config_group, label_group in zip(config_groups, label_groups):
        matchingPaths, matchingLabels = rematchPathsAndLabels(
            config_group, label_group, found_paths
        )
        if matchingPaths:
            paths.append(matchingPaths)
            labels.append(matchingLabels)
    return paths, labels


# Given two lists of matched configs and labels and and unstructured list of paths,
# this function returns the same lists of labels and paths, but such that the
# order they have correspond to eachother and match the order of the configs.
def rematchPathsAndLabels(configs, labels, paths):
    matched_paths = []
    matched_labels = []
    assert len(configs) == len(labels)
    for config, label in zip(configs, labels):
        path = configToPath(config, paths)
        if path and os.path.isfile(path):
            matched_paths.append(path)
            matched_labels.append(label)
        else:
            print(f"Warning: missing file:\n{config.name}")
    return matched_paths, matched_labels


def _path_to_config_name(path: str) -> str:
    base = os.path.basename(path)
    if base == "macroData.csv":
        return os.path.basename(os.path.dirname(path))
    return os.path.splitext(base)[0]


def flattenConfigList(listOfListsOfConfigs):
    # Check if the first element is a list and contains instances of SimulationConfig
    if isinstance(listOfListsOfConfigs[0], SimulationConfig):
        # Here we don't need to flaten at all
        return listOfListsOfConfigs
    elif isinstance(listOfListsOfConfigs[0][0], SimulationConfig):
        # Use list comprehension to flatten the list of lists
        return [config for sublist in listOfListsOfConfigs for config in sublist]
    else:
        raise ValueError(
            "The input must be a list or a list of lists of SimulationConfig instances."
        )


# This function searches all the servers for the given config file,
# downloads the csv file associated with the config file to a temp file,
# and returns the new local path to the csv
def get_csv_files(
    all_configs,
    labels=[],
    useOldFiles=False,
    forceUpdate=False,
    fullScan=False,
    debug_download=False,
    fix_files=False,
):
    nested = False
    config_groups = all_configs
    if not isinstance(all_configs[0], SimulationConfig):
        nested = True
        all_configs = [config for sublist in config_groups for config in sublist]

    global completed_servers, nr_files

    completed_servers, nr_files = 0, 0

    if fullScan:
        candidates = defaultdict(list)
        _merge_candidate_sources(candidates, _scan_local_csv_candidates(all_configs))
        if Servers.servers:
            with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
                future_to_server = {
                    executor.submit(
                        _scan_remote_csv_candidates, server, all_configs
                    ): server
                    for server in Servers.servers
                }
                for future in as_completed(future_to_server):
                    try:
                        remote_candidates = future.result()
                    except Exception as exc:
                        server = future_to_server[future]
                        print(f"{server} duplicate scan failed: {exc}")
                        continue
                    _merge_candidate_sources(candidates, remote_candidates)

        dupes = {}
        for name, entries in candidates.items():
            unique = []
            seen = set()
            for source, path in entries:
                if path in seen:
                    continue
                seen.add(path)
                unique.append((source, path))
            non_macro = [entry for entry in unique if not _is_macro_path(entry[1])]
            if len(non_macro) > 1:
                dupes[name] = unique
        if dupes:
            raise RuntimeError(_format_duplicate_sources(dupes))

    def _merge_paths(found: dict[str, str], new_paths: list[str]):
        for p in new_paths:
            name = _path_to_config_name(p)
            found[name] = p

    # First check if the files have already been downloaded
    paths, remaining_configs = search_for_cvs_files(
        all_configs,
        useOldFiles,
        forceUpdate,
        debug_download=debug_download,
        fix_files=fix_files,
    )
    found_paths: dict[str, str] = {}
    _merge_paths(found_paths, paths)
    if len(remaining_configs) == 0:
        print("All files already downloaded.")
        if nested:
            paths, labels = flatToStructure(config_groups, labels, paths)
        return paths, labels
    elif len(paths) != 0:
        print(
            f"{len(paths)} files found, searching for the remaining {len(remaining_configs)}."
        )
    if len(paths) == 0 and useOldFiles:
        print("No files found!")
        # raise Exception("No files found!")

    # Second check local path to see if we can avoid checking the servers
    localPaths = get_csv_from_server(Servers.local_path_mac, remaining_configs)
    _merge_paths(found_paths, localPaths)
    remaining_configs = [c for c in remaining_configs if c.name not in found_paths]
    if len(remaining_configs) == 0:
        print(f"{len(localPaths)} files found locally. Not searching servers.")
        paths = list(found_paths.values())
        if nested:
            paths, labels = flatToStructure(config_groups, labels, paths)
        return paths, labels

    if remaining_configs:
        print(
            f"{len(localPaths)} files found locally. Searching servers for {len(remaining_configs)} remaining files."
        )

    if Servers.servers:
        server_list = ", ".join(Servers.servers)
        print(f"Searching {len(Servers.servers)} servers for files: {server_list}")
    else:
        print("No servers configured; skipping server search.")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    # get_csv_from_server(Servers.poincare, configs)
    nr_threads = len(Servers.servers) if Servers.servers else 1
    with ThreadPoolExecutor(max_workers=nr_threads) as executor:
        future_to_server = {
            executor.submit(get_csv_from_server, server, remaining_configs): server
            for server in Servers.servers
        }
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            with lock:
                completed_servers += 1  # Increment completed count
            # update_progress(len(remaining_configs))
            try:
                server_paths = future.result()
                if server_paths:
                    _merge_paths(found_paths, server_paths)
            except Exception as exc:
                print(f"\n{server} generated an exception: {exc}")
                print("Continuing with remaining servers.")

    remaining_configs = [c for c in all_configs if c.name not in found_paths]
    if remaining_configs and not useOldFiles:
        old_paths, _ = search_for_cvs_files(
            remaining_configs,
            useOldFiles=True,
            forceUpdate=False,
            debug_download=debug_download,
            fix_files=fix_files,
        )
        _merge_paths(found_paths, old_paths)
        remaining_configs = [c for c in all_configs if c.name not in found_paths]
        if remaining_configs:
            print(f"Missing {len(remaining_configs)} files after fallback.")
    print("")  # New line from progress indicator
    print(f"Found {len(found_paths)} files.")
    paths = list(found_paths.values())
    if nested:
        paths, labels = flatToStructure(config_groups, labels, paths)
    else:
        # The paths are returned in psedu random order, so we need to
        # match them with their correct label again
        paths, labels = rematchPathsAndLabels(all_configs, labels, paths)
    return paths, labels


def get_csv_from_folder(folderPath):
    return [
        os.path.join(folderPath, f)
        for f in os.listdir(folderPath)
        if f.endswith(".csv")
    ]


def get_folders_from_servers(configs, fix=True):
    configs = flattenConfigList(configs)
    print("Searching servers for folders...")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    pathsAndConfig = []
    with ThreadPoolExecutor(max_workers=len(Servers.servers)) as executor:
        future_to_server = {
            executor.submit(download_folders, server, configs, RAW_DATA_PATH): server
            for server in Servers.servers
        }
        for future in as_completed(future_to_server):
            pAndC = future.result()
            pathsAndConfig.extend(pAndC)

    new_paths = [None] * len(configs)  # old order
    for i in range(len(configs)):
        for p, c in pathsAndConfig:
            if c.name == configs[i].name:
                new_paths[i] = p
                continue
    # Remove none objects not found
    new_paths = [c for c in new_paths if c is not None]

    if fix:
        fix_csv_files_in_data_folder(Path(new_paths[0]).parent)

    # We also check local files
    localPaths = handleLocalPath(Servers.local_path_mac, configs, returnCsv=False)
    return localPaths + new_paths


def set_font_size(ax, axis_size=17, legend_size=17, tick_size=17, extra_size=0):
    # Add extra_size to the main font sizes
    axis_size += extra_size
    legend_size += extra_size
    tick_size += extra_size

    # Set axis labels font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=axis_size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_size)

    # Adjust the font size for the legend, if it exists
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)

    # Set tick labels font size for both x and y axes
    ax.tick_params(axis="both", which="major", labelsize=tick_size)


def synchronize_y_limits(ax_list):
    """
    Synchronize the y-limits of a list of Axes objects based on the overall min and max y-values.

    Parameters:
    ax_list (list): List of Matplotlib Axes objects.
    """
    min_y = float("inf")
    max_y = float("-inf")
    ax_list = np.array(ax_list).flatten()

    # Iterate over each Axes object to find the overall min and max y-values
    for ax in ax_list:
        # Get data from lines (e.g., plot, plot_date)
        for line in ax.get_lines():
            y_data = line.get_ydata()
            if len(y_data) > 0:
                min_y = min(min_y, np.nanmin(y_data))
                max_y = max(max_y, np.nanmax(y_data))

        # Get data from scatter plots
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if offsets.size > 0:
                y_data = offsets[:, 1]  # Extract y-values
                min_y = min(min_y, np.nanmin(y_data))
                max_y = max(max_y, np.nanmax(y_data))

    # Set the y-limits for all Axes objects
    for ax in ax_list:
        if ax.get_yscale() == "log":
            ax.set_ylim(min_y * 0.5, max_y * 2)
        else:
            ax.set_ylim(min_y, max_y)


def createVideoes(configs, paths=None, **kwargs):
    from .makeAnimations import makeAnimations

    if not paths:
        # Download the folders associated with the configs from the server
        paths = get_folders_from_servers(configs, fix=False)
    for path in paths:
        makeAnimations(path, **kwargs)


def createPlotsWithImages(configs, paths, metric, **kwargs):
    if not paths:
        # Download the folders associated with the configs from the server
        paths = get_folders_from_servers(configs)

    base = 5 if len(configs) == 3 else 7
    # Create a figure with subplots, one for each configuration
    fig, axes = plt.subplots(1, len(configs), figsize=(base * len(configs), base))

    # If there's only one configuration, axes won't be a list, so convert it into one
    if len(configs) == 1:
        axes = [axes]

    colors = {"FIRE": "#d24646", "LBFGS": "#008743", "CG": "#ffa701"}
    colors = {"LBFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}
    sp = len(configs) == 1  # Single plot
    # Loop over the configurations, paths, and axes
    for ax, path, config, mark in zip(axes, paths, configs, "abc"):
        # Call the provided plot function (either makeStressPlot or makeEnergyPlot)
        fig, ax = makePlot(
            path + "/macroData.csv",
            name=config.name + f"_{metric}+.pdf",
            add_images=True,
            metric=metric,
            ax=ax,
            fig=fig,
            save=False,
            xlim=(0.15, 1),
            colors=[colors[config.minimizer]],
            use_y_axis_name=config.minimizer == "LBFGS" if not sp else True,
            add_cbar=config.minimizer == "FIRE" if not sp else True,
            mark=mark if not sp else None,
            legend=config.minimizer,
            legend_loc="upper left",
            mark_fontsize=20 + 2 * len(configs),
            **kwargs,
        )
        set_font_size(ax, extra_size=2 * len(configs))
        fig.tight_layout()

    if sp:
        method = configs[0].minimizer
    else:
        method = "combined"

    # Save the combined figure
    plt.savefig(f"Plots/{method}_{metric}_plots.pdf")


def stressPlotWithImages(configs, paths=None):
    createPlotsWithImages(
        configs=configs,
        paths=paths,
        ylim=(0, 0.27),
        mark_pos=(0.85, 0.15),
        image_pos=[
            [0.3, 0.01],  # first image, bottom middle
            [0.03, 0.5],  # second image, upper left
            [0.6, 0.55],  # upper right
        ],
        image_size=[0.37, 0.4, 0.4],
        Y="avg_RSS",
        metric="stress",
    )


def energyPlotWithImages(configs, paths=None):
    createPlotsWithImages(
        configs=configs,
        paths=paths,
        ylim=(0, 0.047),
        mark_pos=(0.7, 0.95),
        image_pos=[
            [0.02, 0.5],  # first image, upper left
            [0.29, 0.02],  # second image, lower center
            [0.6, 0.1],  # upper right
        ],
        image_size=[0.4, 0.4, 0.4],
        Y="avg_energy",
        metric="energy",
    )


def plotWholeRangePowerLaw(paths, Y, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Define limits
    if Y == "avg_energy":
        ylim = [8e-3, 2e7]
    elif Y == "avg_RSS":
        ylim = [1e-5, 2e5]
    for ax, group, method, mark in zip(axes, paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]
        plot_powerlaw(
            group_paths=group,
            group_labels=None,
            postRegime=False,
            save=False,
            show=False,
        )
        if Y == "avg_energy":
            add_power_law_line(ax, -0.85, [5e-7, 3e-4], 7e-1)
            add_power_law_line(ax, -2.5, [3e-4, 9e-3], 1e-6, linestyle="-.")
        if Y == "avg_RSS":
            add_power_law_line(ax, -2.8, [3e-5, 5e-4], 5e-9, linestyle="-.")
        set_font_size(ax)

    synchronize_y_limits(axes)

    fig.tight_layout()
    name = "energy" if Y == "avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_full_range.pdf")


def plotPreYieldPowerLaw(paths, Y, **kwargs):
    # Define preyield range
    preYield = (0.15, 0.45)
    # Define limits

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, group, method, mark in zip(range(3), paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        makeAverageComparisonPlot(
            [group],
            Y=Y,
            xlim=preYield,
            ax=axes[0, i],
            use_y_axis_name=method == "L-BFGS",
            fig=fig,
            save=False,
            mark=mark.upper(),
            mark_pos=(0.85, 0.1),
            **kwargs,
        )

        plot_powerlaw(
            group_paths=group,
            group_labels=None,
            postRegime=False,
            save=False,
            show=False,
        )
        set_font_size(axes[0, i])
        set_font_size(axes[1, i])

    synchronize_y_limits(axes[0])
    synchronize_y_limits(axes[1])

    fig.tight_layout()
    name = "energy" if Y == "avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_preYield.pdf")


def plotPostYieldPowerLaw(paths, Y, **kwargs):
    # Define preyield range
    postYield = (0.7, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, group, method, mark in zip(range(3), paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        makeAverageComparisonPlot(
            [group],
            Y=Y,
            xlim=postYield,
            ax=axes[0, i],
            use_y_axis_name=method == "L-BFGS",
            fig=fig,
            save=False,
            mark=mark,
            mark_pos=(0.85, 0.1),
            **kwargs,
        )

        plot_powerlaw(
            group_paths=group,
            group_labels=None,
            postRegime=True,
            save=False,
            show=False,
        )
        set_font_size(axes[0, i])
        set_font_size(axes[1, i])

    synchronize_y_limits(axes[0])
    synchronize_y_limits(axes[1])
    fig.tight_layout()
    name = "energy" if Y == "avg_energy" else "stress"
    # Display all plots in a row
    plt.savefig(f"Plots/combined_{name}_powerlaw_postYield.pdf")


def plotWindowPowerLaw(paths, Y, show_lambda=False, **kwargs):
    # Define limits
    if Y == "avg_energy":
        ylim = [0.62, 0.83]
    elif Y == "avg_RSS":
        ylim = [0.95, 1.34]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, group, method, mark in zip(axes, paths, ["L-BFGS", "CG", "FIRE"], "abc"):
        kwargs["labels"] = [[method]]

        plot_powerlaw(
            group_paths=group,
            group_labels=None,
            postRegime=False,
            save=False,
            show=False,
        )
        set_font_size(ax)

    fig.tight_layout()
    name = "energy" if Y == "avg_energy" else "stress"
    name = name + "_withLambda" if show_lambda else name
    # Display all plots in a row
    plt.savefig(f"Plots/combined_window_{name}_powerlaw.pdf")


def plotAverage(config_groups, labels, useStress=False, group_labels=None, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    kwargs["labels"] = labels
    yColumns = ["avg_energy"]
    if useStress:
        yColumns.append("avg_RSS")
    print("Plotting...")
    for Y in yColumns:
        makeAverageComparisonPlot(paths, Y=Y, group_labels=group_labels, **kwargs)


def plotTime(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    print("Plotting...")
    for Y in [
        "Minimization_time",
        "nr_LBFGS_iterations",
        "nr_LBFGS_func_evals",
    ]:  # "Write_time", "Run_time", "Est_time_remaining"]:
        fig, ax = makePlot(
            paths,
            Y=Y,
            name=f"{Y.replace(' ', '_')}.pdf",
            labels=labels,
            legend=True,
            use_title=True,
            **kwargs,
        )


def plotEnergy(configs, labels, name="Energy", **kwargs):
    paths, labels = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )

    paths = fix_csv_files(paths)

    if len(paths) == 0:
        print("No files found for plotting energy.")
        return
    if kwargs.get("colors") == "minimizer":
        base_colors = {"LBFGS": "#56BD94", "CG": "#9456BD", "FIRE": "#BD9456"}
        kwargs["colors"] = to_rgba(base_colors[configs[0].minimizer], alpha=0.2)
    if len(labels) == len(paths):
        kwargs["legend"] = labels
    elif kwargs.get("legend") is None:
        kwargs["legend"] = name

    if isinstance(paths, list) and isinstance(paths[0], list):
        paths = [p for p_list in paths for p in p_list]
        labels = [l for l_list in labels for l in l_list]

    fig, ax = makePlot(
        paths,
        name=f"{name}.pdf",
        labels=labels,
        **kwargs,
    )
    plt.close(fig)


def plotLog(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )
    kwargs["labels"] = labels

    print("Plotting...")
    # Iterate over the groups and methods, and plot each one in a separate subplot
    for Y, dropLim in zip(
        ["avg_energy", "avg_RSS"],
        [[5e-7, None], [5e-4, None]],
    ):
        kwargs["dropLim"] = dropLim
        # makeAverageComparisonPlot(paths, Y=Y, **kwargs)
        ## makeLogPlotComparison(paths, Y=Y, **kwargs)
        plotWholeRangePowerLaw(paths, Y, **kwargs)
        plotPreYieldPowerLaw(paths, Y, **kwargs)
        # plotPostYieldPowerLaw(paths, Y, **kwargs)
        # plotWindowPowerLaw(paths, Y, **kwargs)

    # makeLogPlotComparison(paths, f"{name} - EnergyPowerLawWindow", window=True, **kwargs)
    # makeEnergyAvalancheComparison(paths, f"{name} - Histogram", **kwargs)
    # makeItterationsPlot(paths, f"{name}Itterations.pdf", **kwargs)


def plotLog2(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )

    paths = fix_csv_files(paths)
    paths, labels = get_group_structure(paths, labels)

    # print(np.array(paths).size)
    for ps, ls in zip(paths, labels):
        plot_plastic_energy_scatter(ps, ls, postRegime=kwargs.get("postRegime", True))
    plot_powerlaw(paths, labels, **kwargs)

def plotLogCompare(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )

    paths = fix_csv_files(paths)
    paths, labels = get_group_structure(paths, labels)

    plot_powerlaw(paths, labels, **kwargs)

def plotPlasticCounts(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )

    plot_plastic_counts_compare(paths, labels, **kwargs)


if __name__ == "__main__":
    seeds = range(0, 60)
    paths = ConfigGenerator.generate_over_seeds(
        seeds,
        rows=60,
        cols=60,
        startLoad=0.15,
        nrThreads=1,
        loadIncrement=1e-5,
        maxLoad=1.0,
        LBFGSEpsx=1e-6,
        minimizer="LBFGS",
        scenario="simpleShear",
    )
    # paths = get_csv_files(configs)
    paths = get_csv_from_folder(
        "/Volumes/data/MTS2D_output/FailedStrangeFireSimulatinos"
    )
    if paths:
        makePlot(paths, "ParamExploration.pdf", show=True, legend=False, ylim=(-100, 2))

        # makeTimePlot(paths, "Run time.pdf", show=True, legend=True)
        # makeItterationsPlot(paths, "ParamExploration.pdf", show=True)
        # makePowerLawPlot(paths, "ParamExplorationPowerLaw.pdf", show=True)
    else:
        print("No files found")

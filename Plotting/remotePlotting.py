import os
import sys
import hashlib
import re
import math
import posixpath
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from pathlib import PurePosixPath
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
    safePath,
    energy_drop_symbol,
)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from .fixLineNumbers import fix_csv_files_in_data_folder
from Management.connectToCluster import getServerUserName
from tqdm import tqdm
import numpy as np
from Plotting.plotPowerLaw import (
    plot_powerlaw,
    plot_powerlaw_compare,
    plot_plastic_counts_compare,
    plot_plastic_energy_scatter,
    get_group_structure,
    get_energy_drops,
    get_stress_drops,
    _drop_quantity_label,
    pretty_variant_label,
    strip_seed_from_label,
)
from Management.updateCSV import fix_csv_files, read_macrodata_csv
from Plotting.dataFunctions import infer_strain_from_vtu

# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Management"))
# Now we can import from Management
from Management.connectToCluster import connectToCluster, Servers, download_folders
from Management.configGenerator import ConfigGenerator, SimulationConfig
from Plotting.remoteDataPaths import FOLDER_PATH, MACRO_PATH, PLOTS_PATH, RAW_DATA_PATH

REMOTE_FOLDER_NAME = "MTS2D_output"
CACHE_UPDATE_AFTER_HOURS = 12

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
    names = [config.generate_name(False) for config in configs]

    existing_paths = []  # This will store the paths to existing data files
    base_path = os.path.join(dataPath, REMOTE_FOLDER_NAME)

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

    ssh = connectToCluster(server, False)
    if ssh is None:
        return []

    names = [config.generate_name(False) for config in configs]
    newPaths = []
    os.makedirs(MACRO_PATH, exist_ok=True)

    try:
        data_paths = _discover_remote_data_paths(ssh, server)
        if not data_paths:
            print(f"No data directory found on {server}.")
            return []

        print(f"Searching {server} in {', '.join(data_paths)}.")

        # TEMP PART
        # DANGEROUS DOUBLE CHECK BEFORE MAKING LIVE
        # IF there are two datapaths on the folder, delete the folder with less data
        # Print the size of both folders, then the sub folders of the smaller folder
        # Ask the user if the smaller folder should be deleted
        # Delete the folder
        # delete_double_folders(data_paths, ssh, server)

        for data_path in data_paths:
            folders = _list_remote_output_folders(ssh, data_path)
            if not folders:
                continue

            with ThreadPoolExecutor(max_workers=7) as executor:
                future_to_name = {
                    executor.submit(
                        download_file,
                        name,
                        folders,
                        data_path,
                        REMOTE_FOLDER_NAME,
                        MACRO_PATH,
                        ssh,
                    ): name
                    for name in names
                }

                for future in as_completed(future_to_name):
                    result = future.result()
                    if result:
                        print(f"Found {result.split('/')[-1]} on {server}")
                        newPaths.append(result)
                        with lock:
                            nr_files += 1

        return newPaths
    finally:
        ssh.close()


def _merge_candidate_sources(target, source_dict):
    for name, entries in source_dict.items():
        target[name].extend(entries)


def _is_macro_path(path):
    try:
        return os.path.commonpath([MACRO_PATH, path]) == MACRO_PATH
    except ValueError:
        return False


def _normalize_remote_path(*parts):
    cleaned_parts = [str(part).strip("/") for part in parts if part]
    return "/" + "/".join(cleaned_parts)


def _discover_remote_data_paths(ssh, server):
    user = getServerUserName(server)
    discover_cmd = (
        f"for base in /data /data2; do "
        f'  if [ -d "$base" ] && [ -d "$base/{user}" ]; then '
        f"    printf '%s\\n' \"$base/{user}\"; "
        f"  fi; "
        f"done"
    )
    stdin, stdout, stderr = ssh.exec_command(discover_cmd)
    return [line.strip() for line in stdout.read().decode().splitlines() if line.strip()]


def _list_remote_output_folders(ssh, data_path):
    remote_root = _normalize_remote_path(data_path, REMOTE_FOLDER_NAME)
    command = f'cd "{remote_root}" 2>/dev/null && ls -d */ 2>/dev/null || true'
    stdin, stdout, stderr = ssh.exec_command(command)
    raw = stdout.read().strip().decode()
    if not raw:
        return []
    return [folder.rstrip("/") for folder in raw.split("\n") if folder]


def _merge_found_paths(found_paths, new_paths):
    for path in new_paths:
        name = _path_to_config_name(path)
        found_paths[name] = path


def _get_remaining_configs(configs, found_paths):
    return [config for config in configs if config.name not in found_paths]


def _count_matched_paths(paths):
    if not paths:
        return 0
    if isinstance(paths[0], list):
        return sum(len(group) for group in paths)
    return len(paths)


def _can_rematch_labels(configs, labels):
    return len(labels) == len(configs)


def _finalize_csv_matches(all_configs, labels, nested, config_groups, found_paths):
    raw_paths = list(found_paths.values())
    if nested:
        matched_paths, matched_labels = flatToStructure(config_groups, labels, raw_paths)
    elif _can_rematch_labels(all_configs, labels):
        matched_paths, matched_labels = rematchPathsAndLabels(
            all_configs, labels, raw_paths
        )
    else:
        matched_paths, matched_labels = raw_paths, labels

    matched_count = _count_matched_paths(matched_paths)
    missing_count = len(all_configs) - matched_count
    if missing_count:
        print(f"Still missing {missing_count} requested files.")

    return matched_paths, matched_labels


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
    if ssh is None:
        return candidates

    try:
        data_paths = _discover_remote_data_paths(ssh, server)
        if not data_paths:
            return candidates

        names = {config.name for config in configs}

        for data_path in data_paths:
            folders = set(_list_remote_output_folders(ssh, data_path))
            matches = names & folders
            if not matches:
                continue
            for name in matches:
                remote_path = (
                    f"{server}:{_normalize_remote_path(data_path, REMOTE_FOLDER_NAME, name, 'macroData.csv')}"
                )
                candidates[name].append((f"remote:{server}", remote_path))

        return candidates
    finally:
        ssh.close()


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


def _config_name(config):
    return getattr(config, "name", None) or config.generate_name(False)


def _flatten_mesh_configs(configs):
    if isinstance(configs, SimulationConfig):
        return [configs]
    configs = list(configs)
    if not configs:
        return []
    return flattenConfigList(configs)


def _safe_relative_parts(relative_path):
    pure_path = PurePosixPath(relative_path)
    if pure_path.is_absolute() or ".." in pure_path.parts:
        raise ValueError(f"Unsafe VTU path in collection.pvd: {relative_path}")
    return pure_path.parts


def _local_mesh_path(folder, relative_path):
    return Path(folder).joinpath(*_safe_relative_parts(relative_path))


def _mesh_load(relative_path, fallback=None):
    load = infer_strain_from_vtu(relative_path)
    if load is not None and np.isfinite(load):
        return float(load)
    if fallback is None:
        return None
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return None


def _mesh_entries_from_pvd_text(pvd_text):
    root = ET.fromstring(pvd_text)
    entries = []
    for dataset in root.iter("DataSet"):
        relative_path = dataset.attrib.get("file")
        if not relative_path or not relative_path.endswith(".vtu"):
            continue
        entries.append(
            {
                "file": relative_path,
                "load": _mesh_load(relative_path, dataset.attrib.get("timestep")),
            }
        )
    if not entries:
        raise ValueError("No VTU DataSet entries found in collection.pvd.")
    return entries


def _mesh_entries_from_local_vtus(folder):
    folder = Path(folder)
    vtu_files = sorted((folder / "data").glob("*.vtu"))
    if not vtu_files:
        vtu_files = sorted(folder.glob("*.vtu"))
    entries = []
    for vtu_file in vtu_files:
        relative_path = vtu_file.relative_to(folder).as_posix()
        entries.append({"file": relative_path, "load": _mesh_load(str(vtu_file))})
    return entries


def _read_local_mesh_entries(folder):
    folder = Path(folder)
    pvd_path = folder / "collection.pvd"
    if pvd_path.exists():
        pvd_text = pvd_path.read_text()
        return _mesh_entries_from_pvd_text(pvd_text), pvd_text
    return _mesh_entries_from_local_vtus(folder), None


def _remote_exists(sftp, remote_path):
    try:
        sftp.stat(remote_path)
        return True
    except OSError:
        return False


def _read_remote_mesh_entries(sftp, remote_folder):
    pvd_path = posixpath.join(remote_folder, "collection.pvd")
    if _remote_exists(sftp, pvd_path):
        with sftp.open(pvd_path, "r") as pvd_file:
            raw_pvd = pvd_file.read()
            pvd_text = raw_pvd.decode() if isinstance(raw_pvd, bytes) else raw_pvd
        return _mesh_entries_from_pvd_text(pvd_text), pvd_text

    entries = []
    for relative_dir in ("data", ""):
        remote_dir = posixpath.join(remote_folder, relative_dir) if relative_dir else remote_folder
        if not _remote_exists(sftp, remote_dir):
            continue
        for filename in sftp.listdir(remote_dir):
            if not filename.endswith(".vtu"):
                continue
            relative_path = posixpath.join(relative_dir, filename) if relative_dir else filename
            entries.append({"file": relative_path, "load": _mesh_load(relative_path)})
        if entries:
            break
    return entries, None


def _select_mesh_entry(entries, load):
    if not entries:
        raise ValueError("No VTU files found for requested mesh.")

    entries_with_load = [entry for entry in entries if entry["load"] is not None]
    if not entries_with_load:
        raise ValueError("Could not infer load values for any VTU files.")

    if math.isinf(float(load)):
        return max(entries_with_load, key=lambda entry: entry["load"])
    target_load = float(load)
    return min(entries_with_load, key=lambda entry: abs(entry["load"] - target_load))


def _format_mesh_load(load):
    load = float(load)
    if math.isinf(load):
        return "final load"
    return f"load {load:g}"


def _copy_root_small_files(source_folder, dest_folder):
    source_folder = Path(source_folder)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    for source_path in source_folder.iterdir():
        if source_path.is_file() and source_path.suffix.lower() == ".csv":
            shutil.copy2(source_path, dest_folder / source_path.name)


def _download_root_small_files(sftp, remote_folder, dest_folder):
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    for filename in sftp.listdir(remote_folder):
        if Path(filename).suffix.lower() != ".csv":
            continue
        remote_file = posixpath.join(remote_folder, filename)
        local_file = dest_folder / filename
        sftp.get(remote_file, str(local_file))


def _copy_mesh_from_local_folder(source_folder, config_name, load, *, missing_ok=False):
    source_folder = Path(source_folder)
    if not source_folder.is_dir():
        return None

    entries, _ = _read_local_mesh_entries(source_folder)
    entry = _select_mesh_entry(entries, load)
    source_vtu = _local_mesh_path(source_folder, entry["file"])
    if not source_vtu.is_file():
        if missing_ok:
            return None
        raise FileNotFoundError(f"Selected VTU is missing: {source_vtu}")

    dest_folder = Path(RAW_DATA_PATH) / config_name
    dest_vtu = _local_mesh_path(dest_folder, entry["file"])
    if source_folder.resolve() != dest_folder.resolve():
        _copy_root_small_files(source_folder, dest_folder)
        dest_vtu.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_vtu, dest_vtu)

    return str(dest_vtu)


def _download_mesh_from_remote_folder(sftp, remote_folder, config_name, load):
    entries, _ = _read_remote_mesh_entries(sftp, remote_folder)
    entry = _select_mesh_entry(entries, load)
    dest_folder = Path(RAW_DATA_PATH) / config_name
    dest_vtu = _local_mesh_path(dest_folder, entry["file"])

    _download_root_small_files(sftp, remote_folder, dest_folder)

    dest_vtu.parent.mkdir(parents=True, exist_ok=True)
    remote_vtu = posixpath.join(remote_folder, entry["file"])
    try:
        total_bytes = sftp.stat(remote_vtu).st_size
    except OSError:
        total_bytes = None

    description = f"Downloading {PurePosixPath(entry['file']).name[:32]}"
    if total_bytes:
        with tqdm(
            total=total_bytes,
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            transferred = [0]

            def update_progress(done, total):
                increment = done - transferred[0]
                if increment > 0:
                    progress.update(increment)
                transferred[0] = done

            sftp.get(remote_vtu, str(dest_vtu), callback=update_progress)
    else:
        tqdm.write(f"{description}...")
        sftp.get(remote_vtu, str(dest_vtu))
    return str(dest_vtu)


def getMeshAt(configs, load, forceUpdate=False):
    configs = _flatten_mesh_configs(configs)
    found_paths = {}
    load_label = _format_mesh_load(load)
    print(f"Looking for {len(configs)} mesh file(s) at {load_label}.")

    if not forceUpdate:
        for config in tqdm(configs, desc="Checking mesh cache", unit="mesh"):
            name = _config_name(config)
            path = _copy_mesh_from_local_folder(
                Path(RAW_DATA_PATH) / name,
                name,
                load,
                missing_ok=True,
            )
            if path is not None:
                found_paths[name] = path
        if found_paths:
            print(f"Using {len(found_paths)} cached mesh file(s).")
    else:
        print("Skipping mesh cache because forceUpdate=True.")

    remaining_configs = [config for config in configs if _config_name(config) not in found_paths]
    if remaining_configs:
        print(f"Checking local data volume for {len(remaining_configs)} mesh file(s).")
    for config in tqdm(remaining_configs, desc="Checking local data", unit="mesh"):
        name = _config_name(config)
        source_folder = Path(Servers.local_path_mac) / REMOTE_FOLDER_NAME / name
        path = _copy_mesh_from_local_folder(source_folder, name, load, missing_ok=False)
        if path is not None:
            found_paths[name] = path

    remaining_configs = [config for config in configs if _config_name(config) not in found_paths]
    if remaining_configs:
        print(f"Searching {len(Servers.search_servers)} server(s) for {len(remaining_configs)} mesh file(s).")
    for server in Servers.search_servers:
        if not remaining_configs:
            break
        print(f"Searching {server} ({len(remaining_configs)} remaining).")
        ssh = connectToCluster(server, False)
        if ssh is None:
            continue
        sftp = None
        try:
            sftp = ssh.open_sftp()
            data_paths = _discover_remote_data_paths(ssh, server)
            for data_path in data_paths:
                remote_folders = set(_list_remote_output_folders(ssh, data_path))
                for config in list(remaining_configs):
                    name = _config_name(config)
                    if name not in remote_folders:
                        continue
                    remote_folder = _normalize_remote_path(data_path, REMOTE_FOLDER_NAME, name)
                    tqdm.write(f"Found {name} on {server}; downloading selected VTU.")
                    found_paths[name] = _download_mesh_from_remote_folder(
                        sftp,
                        remote_folder,
                        name,
                        load,
                    )
                    remaining_configs = [
                        item
                        for item in remaining_configs
                        if _config_name(item) not in found_paths
                    ]
                if not remaining_configs:
                    break
        finally:
            if sftp is not None:
                try:
                    sftp.close()
                except Exception:
                    pass
            ssh.close()

    missing = [config for config in configs if _config_name(config) not in found_paths]
    if missing:
        preview = "\n".join(f"  {_config_name(config)}" for config in missing[:5])
        raise FileNotFoundError(
            f"Could not find mesh data for {len(missing)} configs. First missing configs:\n{preview}"
        )

    return [found_paths[_config_name(config)] for config in configs]


def getFinalMesh(configs, forceUpdate=False):
    return getMeshAt(configs, load=float("inf"), forceUpdate=forceUpdate)


def search_for_cvs_files(
    configs,
    useOldFiles=False,
    forceUpdate=False,
    debug_download=False,
    fix_files=True,
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
    search_folders = ["/tmp/MTS2D", MACRO_PATH]  # Directories to search in

    for folder in search_folders:
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
                age_ok = time.time() - file_mod_time < CACHE_UPDATE_AFTER_HOURS * 3600

                try:
                    df = read_macrodata_csv(
                        file_path,
                        fix_mixed=True,
                        update_header=False,
                        warn_on_dtype=True,
                    )
                except Exception as fix_exc:
                    print(f"Failed to read {file_path}: {fix_exc}")
                    if age_ok or useOldFiles:
                        if debug_download:
                            age_min = (time.time() - file_mod_time) / 60.0
                            print(
                                f"Using local file despite parse error (age={age_min:.1f} min): {file_path}"
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
                    if est_time_remaining.empty:
                        time_remaining = None
                    else:
                        try:
                            time_remaining = duration_to_seconds(est_time_remaining.iloc[-1])
                        except (TypeError, ValueError):
                            time_remaining = None
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

    remaining_configs = _get_remaining_configs(configs, found_paths)

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
    if isinstance(config, (str, os.PathLike)):
        return os.fspath(config)
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
        if isinstance(config, (str, os.PathLike)):
            path = os.fspath(config)
            if os.path.isfile(path):
                matched_paths.append(path)
                matched_labels.append(label)
            else:
                raise FileNotFoundError(f"Explicit path does not exist: {path}")
            continue
        path = configToPath(config, paths)
        if path and os.path.isfile(path):
            matched_paths.append(path)
            matched_labels.append(label)
        else:
            print(f"Warning: missing file:\n{config.name}")
    return matched_paths, matched_labels


def _split_requested_inputs(all_configs):
    if isinstance(all_configs, (SimulationConfig, str, os.PathLike)):
        nested = False
        item = os.fspath(all_configs) if isinstance(all_configs, os.PathLike) else all_configs
        config_groups = [[item]]
    else:
        all_configs = list(all_configs)
        if len(all_configs) == 0:
            return False, [], [], []
        nested = any(isinstance(item, (list, tuple, np.ndarray)) for item in all_configs)
        if nested:
            groups = [
                group if isinstance(group, (list, tuple, np.ndarray)) else [group]
                for group in all_configs
            ]
            config_groups = [
                [os.fspath(item) if isinstance(item, os.PathLike) else item for item in group]
                for group in groups
            ]
        else:
            config_groups = [[
                os.fspath(item) if isinstance(item, os.PathLike) else item
                for item in all_configs
            ]]

    all_items = [item for group in config_groups for item in group]
    configs = []
    for item in all_items:
        if isinstance(item, SimulationConfig):
            configs.append(item)
        elif isinstance(item, str):
            if not os.path.isfile(item):
                raise FileNotFoundError(f"Explicit path does not exist: {item}")
        else:
            raise TypeError(
                "get_csv_files expects SimulationConfig objects or existing file paths."
            )

    return nested, config_groups, all_items, configs


def _normalize_requested_labels(labels, config_groups, nested):
    if labels is None:
        labels = []
    elif isinstance(labels, (str, os.PathLike)):
        labels = [str(labels)]
    else:
        labels = list(labels)

    if not nested:
        target_len = len(config_groups[0]) if config_groups else 0
        return labels[:target_len] + [""] * max(0, target_len - len(labels))

    if not labels:
        return [[""] * len(group) for group in config_groups]
    if isinstance(labels[0], (list, tuple, np.ndarray)):
        normalized = []
        for group, group_labels in zip(config_groups, labels):
            group_labels = list(group_labels)
            normalized.append(
                group_labels[: len(group)] + [""] * max(0, len(group) - len(group_labels))
            )
        while len(normalized) < len(config_groups):
            normalized.append([""] * len(config_groups[len(normalized)]))
        return normalized
    if len(labels) == len(config_groups):
        return [[label] * len(group) for label, group in zip(labels, config_groups)]
    return [[""] * len(group) for group in config_groups]


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
    fix_files=True,
):
    nested, config_groups, all_items, configs = _split_requested_inputs(all_configs)
    normalized_labels = _normalize_requested_labels(labels, config_groups, nested)
    if not all_items:
        return [], []

    global completed_servers, nr_files

    completed_servers, nr_files = 0, 0

    if fullScan and configs:
        candidates = defaultdict(list)
        _merge_candidate_sources(candidates, _scan_local_csv_candidates(configs))
        if Servers.search_servers:
            with ThreadPoolExecutor(max_workers=len(Servers.search_servers)) as executor:
                future_to_server = {
                    executor.submit(
                        _scan_remote_csv_candidates, server, configs
                    ): server
                    for server in Servers.search_servers
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

    if not configs:
        print(f"Using {len(all_items)} explicit file paths.")
        return _finalize_csv_matches(
            all_items,
            normalized_labels,
            nested,
            config_groups,
            {},
        )

    # First check if the files have already been downloaded
    paths, remaining_configs = search_for_cvs_files(
        configs,
        useOldFiles,
        forceUpdate,
        debug_download=debug_download,
        fix_files=fix_files,
    )
    found_paths: dict[str, str] = {}
    _merge_found_paths(found_paths, paths)
    if len(remaining_configs) == 0:
        print("All files already downloaded.")
        return _finalize_csv_matches(
            all_items,
            normalized_labels,
            nested,
            config_groups,
            found_paths,
        )
    elif len(paths) != 0:
        print(
            f"Using {len(paths)} cached files. Searching for {len(remaining_configs)} remaining files."
        )
    if len(paths) == 0 and useOldFiles:
        print("No files found!")
        # raise Exception("No files found!")

    # Second check local path to see if we can avoid checking the servers
    localPaths = get_csv_from_server(Servers.local_path_mac, remaining_configs)
    _merge_found_paths(found_paths, localPaths)
    remaining_configs = _get_remaining_configs(remaining_configs, found_paths)
    if len(remaining_configs) == 0:
        print(f"{len(localPaths)} files found locally. Not searching servers.")
        return _finalize_csv_matches(
            all_items,
            normalized_labels,
            nested,
            config_groups,
            found_paths,
        )

    if remaining_configs:
        print(
            f"Found {len(localPaths)} files in {Servers.local_path_mac}. Searching servers for {len(remaining_configs)} remaining files."
        )

    if Servers.search_servers:
        server_list = ", ".join(Servers.search_servers)
        print(f"Searching {len(Servers.search_servers)} servers for files: {server_list}")
    else:
        print("No servers configured; skipping server search.")
    # Use ThreadPoolExecutor to execute find_data_on_server in parallel across all servers
    # get_csv_from_server(Servers.poincare, configs)
    nr_threads = len(Servers.search_servers) if Servers.search_servers else 1
    with ThreadPoolExecutor(max_workers=nr_threads) as executor:
        future_to_server = {
            executor.submit(get_csv_from_server, server, remaining_configs): server
            for server in Servers.search_servers
        }
        for future in as_completed(future_to_server):
            server = future_to_server[future]
            with lock:
                completed_servers += 1  # Increment completed count
            # update_progress(len(remaining_configs))
            try:
                server_paths = future.result()
                if server_paths:
                    _merge_found_paths(found_paths, server_paths)
            except Exception as exc:
                print(f"{server} search failed: {exc}")
                print("Continuing with remaining servers.")

    remaining_configs = _get_remaining_configs(configs, found_paths)
    if remaining_configs and not useOldFiles:
        old_paths, _ = search_for_cvs_files(
            remaining_configs,
            useOldFiles=True,
            forceUpdate=False,
            debug_download=debug_download,
            fix_files=fix_files,
        )
        _merge_found_paths(found_paths, old_paths)
        remaining_configs = _get_remaining_configs(configs, found_paths)
        if remaining_configs:
            print(f"Missing {len(remaining_configs)} files after fallback.")
    return _finalize_csv_matches(
        all_items,
        normalized_labels,
        nested,
        config_groups,
        found_paths,
    )


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
    with ThreadPoolExecutor(max_workers=len(Servers.search_servers)) as executor:
        future_to_server = {
            executor.submit(download_folders, server, configs, RAW_DATA_PATH): server
            for server in Servers.search_servers
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

def plotStress(configs, labels, name="Stress", **kwargs):
    plotEnergy(configs, labels, name=name, Y="avg_sigma12", **kwargs)

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

    plot_powerlaw_compare(paths, labels, **kwargs)

def plotPlasticCounts(config_groups, labels, **kwargs):
    paths, labels = get_csv_files(
        config_groups, labels=labels, useOldFiles=False, forceUpdate=False
    )

    plot_plastic_counts_compare(paths, labels, **kwargs)


def _pretty_reversibility_axis_label(x_axis_col):
    if not isinstance(x_axis_col, str):
        return str(x_axis_col)

    match = re.fullmatch(r"rev_(.+)_diff", x_axis_col)
    if not match:
        return x_axis_col

    quantity = match.group(1)
    quantity_lower = quantity.lower()
    symbol_map = {
        "u": r"\mathbf{u}",
        "sigma": r"\sigma",
        "stress": r"\sigma",
        "sigma12": r"\sigma_{12}",
        "p12": r"P_{12}",
    }
    symbol = symbol_map.get(quantity_lower)
    if symbol is None:
        if quantity_lower in {"energy", "avg_energy", "total_energy"}:
            symbol = energy_drop_symbol("energy")
        else:
            escaped = quantity.replace("_", r"\_")
            symbol = rf"\mathrm{{{escaped}}}"

    return rf"$\Delta_{{\mathrm{{rev}}}} {symbol}$"


def plotReversibilityEnergyDropCorrelation(
    configs,
    labels=None,
    show=False,
    save=True,
    includeStressDrops=True,
    xAxisCol="rev_u_diff",
    strainLim="auto",
    postRegime=True,
    averageEnergy=False,
    name="reversibility_energyDrop_correlation",
):
    if postRegime not in (True, False, None):
        raise ValueError(
            f"postRegime must be True (post-yield), False (pre-yield), or None (all); got {postRegime!r}"
        )

    paths, _ = get_csv_files(
        configs, labels=labels, useOldFiles=False, forceUpdate=False
    )
    paths, labels = get_group_structure(paths, labels)
    if not paths:
        raise RuntimeError("No CSV paths found for reversibility job.")

    energy_drop_specs = [
        {
            "key": "stress_corrected_energy",
            "fn": get_energy_drops,
            "marker": "o",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                averageEnergy=averageEnergy,
                stress_corrected=True,
            ),
            "fallback_label": "Stress-corrected energy drop",
        },
        {
            "key": "inter_strain_energy",
            "fn": get_energy_drops,
            "marker": "s",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                averageEnergy=averageEnergy,
                stress_corrected=False,
                energy_type="energy_change",
            ),
            "fallback_label": "Inter-strain energy drop",
        },
        {
            "key": "relaxation_energy",
            "fn": get_energy_drops,
            "marker": "^",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                averageEnergy=averageEnergy,
                stress_corrected=False,
                energy_type="e_change_from_init",
            ),
            "fallback_label": "Relaxation energy drop",
        },
    ]
    stress_drop_specs = [
        {
            "key": "stress_corrected",
            "fn": get_stress_drops,
            "marker": "o",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                stress_type="stress_corrected",
            ),
            "fallback_label": "Elasticity-corrected stress drop",
        },
        {
            "key": "inter_strain_stress",
            "fn": get_stress_drops,
            "marker": "s",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                stress_type="inter_strain",
            ),
            "fallback_label": "Inter-strain stress drop",
        },
        {
            "key": "relaxation_stress",
            "fn": get_stress_drops,
            "marker": "^",
            "kwargs": dict(
                strainLim=strainLim,
                postRegime=postRegime,
                stress_type="relaxation",
            ),
            "fallback_label": "Relaxation stress drop",
        },
    ]

    n_groups = len(paths)
    blue_cmap = plt.get_cmap("Blues")
    if n_groups <= 1:
        shade_vals = np.array([0.75])
    else:
        shade_vals = np.linspace(0.45, 0.9, n_groups)
    group_colors = [blue_cmap(v) for v in shade_vals]

    group_display_labels = []
    for idx, group_labels in enumerate(labels):
        cleaned = [strip_seed_from_label(lbl) for lbl in group_labels if lbl]
        cleaned = [lbl for lbl in cleaned if lbl]
        if not cleaned:
            group_display_labels.append(f"group {idx + 1}")
            continue
        unique_cleaned = list(dict.fromkeys(cleaned))
        base_label = unique_cleaned[0]
        group_display_labels.append(pretty_variant_label(base_label) or base_label)

    regime_tag = (
        "postYield" if postRegime is True else "preYield" if postRegime is False else "allYield"
    )
    base = os.path.splitext(name)[0] if name else "reversibility_energyDrop_correlation"
    flat_paths = [str(path) for group in paths for path in group]
    signature = hashlib.sha1("|".join(flat_paths).encode("utf-8")).hexdigest()[:10]
    x_axis_label = _pretty_reversibility_axis_label(xAxisCol)

    def _plot_drop_specs(drop_specs, *, title, y_label, file_suffix):
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = 0
        shape_labels = {}

        for group_idx, group_paths in enumerate(paths):
            if not group_paths:
                continue
            group_color = group_colors[group_idx]
            for spec in drop_specs:
                drops, info = spec["fn"](group_paths, **spec["kwargs"])
                df_info = info.get("df")
                if df_info is None:
                    raise ValueError(
                        "data_info['df'] is missing; cannot extract x-axis values."
                    )
                if xAxisCol not in df_info:
                    raise KeyError(
                        f"Missing xAxisCol '{xAxisCol}' in data_info['df']."
                    )
                if "mask" in info:
                    combined_mask = np.asarray(info["mask"], dtype=bool)
                else:
                    masks = info.get("masks")
                    if not masks:
                        raise ValueError(
                            "data_info does not contain 'mask' or 'masks'."
                        )
                    if isinstance(masks, (list, tuple)):
                        combined_mask = np.concatenate(
                            [np.asarray(mask, dtype=bool) for mask in masks]
                        )
                    else:
                        combined_mask = np.asarray(masks, dtype=bool)
                x_all = np.asarray(df_info[xAxisCol], dtype=float)
                if x_all.shape[0] != combined_mask.shape[0]:
                    raise ValueError(
                        f"x-axis length mismatch: len(df['{xAxisCol}'])={x_all.shape[0]} "
                        f"but mask length={combined_mask.shape[0]}"
                    )
                x_vals = x_all[combined_mask]
                y_vals = np.asarray(drops, dtype=float)
                if x_vals.shape[0] != y_vals.shape[0]:
                    raise ValueError(
                        f"Drop/reversibility length mismatch: "
                        f"drops={y_vals.shape[0]}, {xAxisCol}={x_vals.shape[0]}"
                    )

                valid = (
                    np.isfinite(x_vals)
                    & np.isfinite(y_vals)
                    & (x_vals > 0)
                    & (y_vals > 0)
                )
                x_vals = x_vals[valid]
                y_vals = y_vals[valid]
                if x_vals.size == 0:
                    continue

                drop_label = info.get("drop_label")
                if drop_label is not None and spec["key"] not in shape_labels:
                    shape_labels[spec["key"]] = rf"${_drop_quantity_label(drop_label)}$"

                ax.scatter(
                    x_vals,
                    y_vals,
                    marker=spec["marker"],
                    s=18,
                    facecolors="none",
                    edgecolors=group_color,
                    linewidths=1.0,
                )
                plotted += 1

        if plotted == 0:
            raise RuntimeError("No valid reversibility/drop data points were found.")

        color_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=group_colors[i],
                markersize=6,
                label=group_display_labels[i],
            )
            for i in range(len(group_display_labels))
        ]
        shape_handles = []
        for spec in drop_specs:
            label_text = shape_labels.get(spec["key"], spec["fallback_label"])
            shape_handles.append(
                Line2D(
                    [],
                    [],
                    marker=spec["marker"],
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markersize=6,
                    label=label_text,
                )
            )
        legend_handles = [
            Line2D([], [], linestyle="None", label="Settings (color)"),
            *color_handles,
            Line2D([], [], linestyle="None", label="Drop Type (shape)"),
            *shape_handles,
        ]

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(handles=legend_handles, loc="upper left", ncol=2, frameon=True)
        fig.tight_layout()

        if save:
            suffix = f"_{safePath(file_suffix)}" if file_suffix else ""
            xaxis_tag = safePath(f"x_{xAxisCol}")
            save_name = (
                f"{safePath(base)}{suffix}_{xaxis_tag}_{regime_tag}_{signature}.png"
            )
            save_path = os.path.join("Plots", save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Saved figure to {save_path}")
        return fig, ax

    fig, ax = _plot_drop_specs(
        energy_drop_specs,
        title="Reversibility vs energy-drop correlation",
        y_label="Energy drop magnitude",
        file_suffix=None,
    )

    if includeStressDrops:
        _plot_drop_specs(
            stress_drop_specs,
            title="Reversibility vs stress-drop correlation",
            y_label="Stress drop magnitude",
            file_suffix="stressOnly",
        )

    if show:
        plt.show()
    return fig, ax

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
        experiment="simpleShear",
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
